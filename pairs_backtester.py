"""
Pairs Trading Backtester v1.0
Тестирование стратегии mean-reversion на исторических данных.

Использует ту же логику что и scanner v10.2:
  - Kalman Filter для адаптивного hedge ratio
  - MAD-based robust Z-score с адаптивным окном
  - Adaptive thresholds по confidence / quality / TF

Автор: Claude + User
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ═══════════════════════════════════════════════════════
# CORE FUNCTIONS (из mean_reversion_analysis.py)
# ═══════════════════════════════════════════════════════

def kalman_hedge_ratio(series1, series2, delta=1e-4, ve=1e-3):
    """Kalman Filter для динамического hedge ratio."""
    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)
    n = min(len(s1), len(s2))
    if n < 10:
        return None
    s1, s2 = s1[:n], s2[:n]

    init_n = min(30, n // 3)
    try:
        X_init = np.column_stack([np.ones(init_n), s2[:init_n]])
        beta_init = np.linalg.lstsq(X_init, s1[:init_n], rcond=None)[0]
    except Exception:
        beta_init = np.array([0.0, 1.0])

    beta = beta_init.copy()
    P = np.eye(2) * 1.0
    Q = np.eye(2) * delta
    R = ve

    hedge_ratios = np.zeros(n)
    intercepts = np.zeros(n)
    trading_spread = np.zeros(n)

    for t in range(n):
        x_t = np.array([1.0, s2[t]])
        P = P + Q
        y_hat = x_t @ beta
        e_t = s1[t] - y_hat
        S_t = x_t @ P @ x_t + R
        K_t = P @ x_t / S_t
        beta = beta + K_t * e_t
        P = P - np.outer(K_t, x_t) @ P
        P = (P + P.T) / 2
        np.fill_diagonal(P, np.maximum(np.diag(P), 1e-10))

        intercepts[t] = beta[0]
        hedge_ratios[t] = beta[1]
        trading_spread[t] = s1[t] - beta[1] * s2[t] - beta[0]

    return {
        'hedge_ratios': hedge_ratios,
        'intercepts': intercepts,
        'spread': trading_spread,
        'hr_final': float(hedge_ratios[-1]),
        'hr_std': float(np.sqrt(P[1, 1])),
    }


def calculate_adaptive_robust_zscore(spread, halflife_bars=None, min_w=10, max_w=60):
    """MAD-based Z-score с адаптивным окном."""
    spread = np.array(spread, dtype=float)
    n = len(spread)

    if halflife_bars is not None and not np.isinf(halflife_bars) and halflife_bars > 0:
        window = int(np.clip(2.5 * halflife_bars, min_w, max_w))
    else:
        window = 30

    if n < window + 1:
        window = max(10, n // 2)
        if n < window + 1:
            s = np.std(spread)
            zs = (spread - np.mean(spread)) / s if s > 1e-10 else np.zeros_like(spread)
            return zs, window
        
    zscore_series = np.full(n, np.nan)
    for i in range(window, n):
        lb = spread[i - window:i]
        med = np.median(lb)
        mad = np.median(np.abs(lb - med)) * 1.4826
        if mad < 1e-10:
            s = np.std(lb)
            zscore_series[i] = (spread[i] - np.mean(lb)) / s if s > 1e-10 else 0.0
        else:
            zscore_series[i] = (spread[i] - med) / mad

    return zscore_series, window


def calculate_halflife(spread):
    """Half-life из OLS на spread."""
    spread = np.array(spread, dtype=float)
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    
    if len(spread_lag) < 5:
        return np.inf
    
    # OLS: spread_diff = b * spread_lag
    sx = np.sum(spread_lag)
    sy = np.sum(spread_diff)
    sxy = np.sum(spread_lag * spread_diff)
    sx2 = np.sum(spread_lag ** 2)
    n = len(spread_lag)
    
    denom = n * sx2 - sx ** 2
    if abs(denom) < 1e-10:
        return np.inf
    b = (n * sxy - sx * sy) / denom
    
    if b >= 0:
        return np.inf
    return float(-np.log(2) / b)


def calculate_ou_parameters(spread, dt=1.0):
    """OU: dX = θ(μ - X)dt + σdW."""
    try:
        if len(spread) < 20:
            return None
        spread = np.array(spread, dtype=float)
        y, x = np.diff(spread), spread[:-1]
        n = len(x)
        sx, sy = np.sum(x), np.sum(y)
        sxy, sx2 = np.sum(x * y), np.sum(x ** 2)
        denom = n * sx2 - sx ** 2
        if abs(denom) < 1e-10:
            return None
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        theta = max(0.001, min(10.0, -b / dt))
        mu = a / theta if theta > 0 else 0.0
        sigma = np.std(y - (a + b * x))
        halflife = np.log(2) / theta if theta > 0 else 999.0
        return {'theta': float(theta), 'mu': float(mu), 'sigma': float(sigma),
                'halflife_ou': float(halflife)}
    except Exception:
        return None


def cointegration_test(s1, s2):
    """Тест Engle-Granger."""
    from statsmodels.tsa.stattools import coint
    try:
        score, pvalue, _ = coint(s1, s2)
        return pvalue
    except:
        return 1.0


def adf_test(spread):
    """ADF тест стационарности."""
    from statsmodels.tsa.stattools import adfuller
    try:
        result = adfuller(np.array(spread, dtype=float), autolag='AIC')
        return result[1] < 0.05
    except:
        return False


def calculate_hurst(spread, max_k=None):
    """Hurst exponent через DFA."""
    try:
        ts = np.array(spread, dtype=float)
        ts = ts[~np.isnan(ts)]
        n = len(ts)
        if n < 50:
            return 0.5
        
        if max_k is None:
            max_k = n // 4
        
        min_k = 4
        scales = np.unique(np.logspace(
            np.log10(min_k), np.log10(max_k), num=20
        ).astype(int))
        scales = scales[(scales >= min_k) & (scales <= max_k)]
        
        if len(scales) < 4:
            return 0.5
        
        y = np.cumsum(ts - np.mean(ts))
        fluctuations = []
        valid_scales = []
        
        for s in scales:
            n_segments = n // s
            if n_segments < 1:
                continue
            F2 = 0
            for v in range(n_segments):
                segment = y[v * s:(v + 1) * s]
                x = np.arange(s)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                F2 += np.mean((segment - trend) ** 2)
            F2 /= n_segments
            if F2 > 0:
                fluctuations.append(np.sqrt(F2))
                valid_scales.append(s)
        
        if len(valid_scales) < 4:
            return 0.5
        
        log_s = np.log(valid_scales)
        log_f = np.log(fluctuations)
        coeffs = np.polyfit(log_s, log_f, 1)
        H = float(coeffs[0])
        return max(0.01, min(0.99, H))
    except:
        return 0.5


# ═══════════════════════════════════════════════════════
# TRADE & BACKTEST DATA STRUCTURES
# ═══════════════════════════════════════════════════════

@dataclass
class Trade:
    """Одна сделка."""
    entry_bar: int
    entry_time: datetime
    entry_z: float
    entry_spread: float
    entry_price1: float
    entry_price2: float
    entry_hr: float
    direction: str          # LONG / SHORT (spread)
    
    exit_bar: int = 0
    exit_time: datetime = None
    exit_z: float = 0.0
    exit_spread: float = 0.0
    exit_price1: float = 0.0
    exit_price2: float = 0.0
    exit_reason: str = ''
    pnl_pct: float = 0.0   # P&L в %
    bars_held: int = 0


@dataclass
class BacktestResult:
    """Результаты бэктеста."""
    trades: List[Trade]
    equity_curve: np.ndarray
    spread_series: np.ndarray
    zscore_series: np.ndarray
    hr_series: np.ndarray
    price1: np.ndarray
    price2: np.ndarray
    timestamps: list
    
    # Summary
    total_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    avg_bars_held: float = 0.0
    max_bars_held: int = 0


# ═══════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════

def run_backtest(
    price1: np.ndarray,
    price2: np.ndarray,
    timestamps: list,
    timeframe: str = '4h',
    train_window: int = 200,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.5,
    max_hold_bars: int = 100,
    commission_pct: float = 0.1,
    min_quality_z: float = 1.5,
) -> BacktestResult:
    """
    Walk-forward бэктест пары.
    
    Логика:
      1. На каждом баре t, берём окно [t-train_window : t] 
      2. Вычисляем Kalman HR + spread + Z-score
      3. Если нет позиции и |Z| > entry_z → вход
      4. Если в позиции:
         - |Z| < exit_z → выход (mean reversion)
         - Z переключил знак и |Z| > min → выход с прибылью
         - |Z| > stop_z → стоп
         - bars_held > max_hold → тайм-аут
    
    P&L (dollar-neutral):
      LONG spread = buy $1 coin1, sell $HR coin2
      Return = r1 - HR * r2  (где r = price_change / price_entry)
      Нормируем на (1 + |HR|) для total capital
    """
    n = len(price1)
    assert len(price2) == n, "Price arrays must have same length"
    
    hours_per_bar = {'1h': 1, '2h': 2, '4h': 4, '1d': 24, '15m': 0.25}.get(timeframe, 4)
    
    # Storage для полных серий
    full_spread = np.full(n, np.nan)
    full_zscore = np.full(n, np.nan)
    full_hr = np.full(n, np.nan)
    equity = np.ones(n)  # нормализованная equity curve
    
    trades: List[Trade] = []
    current_trade: Optional[Trade] = None
    
    # Walk-forward
    for t in range(train_window, n):
        # 1. Окно для Kalman + Z
        w_start = max(0, t - train_window)
        p1_window = price1[w_start:t + 1]
        p2_window = price2[w_start:t + 1]
        
        # 2. Kalman
        kf = kalman_hedge_ratio(p1_window, p2_window, delta=1e-4)
        if kf is None:
            equity[t] = equity[t - 1]
            continue
        
        spread_window = kf['spread']
        hr_current = kf['hr_final']
        
        full_spread[t] = spread_window[-1]
        full_hr[t] = hr_current
        
        # 3. Half-life → adaptive Z window
        hl_days = calculate_halflife(spread_window)
        hl_hours = hl_days * 24
        hl_bars = hl_hours / hours_per_bar if hl_hours < 9999 else None
        
        # 4. Z-score
        zscores, z_win = calculate_adaptive_robust_zscore(
            spread_window, halflife_bars=hl_bars
        )
        
        z_current = zscores[-1] if not np.isnan(zscores[-1]) else 0.0
        full_zscore[t] = z_current
        
        # ═══ TRADE LOGIC ═══
        
        if current_trade is not None:
            # В позиции — проверяем выход
            bars_held = t - current_trade.entry_bar
            
            exit_signal = False
            exit_reason = ''
            
            if current_trade.direction == 'LONG':
                # LONG spread: ждём Z вернётся к 0 (или выше)
                if z_current >= -exit_z and z_current <= exit_z:
                    exit_signal = True
                    exit_reason = 'MEAN_REVERT'
                elif z_current > entry_z * 0.5:
                    # Перехлест — Z перешёл на другую сторону
                    exit_signal = True
                    exit_reason = 'OVERSHOOT'
                elif z_current < -stop_z:
                    exit_signal = True
                    exit_reason = 'STOP_LOSS'
                elif bars_held >= max_hold_bars:
                    exit_signal = True
                    exit_reason = 'TIMEOUT'
            else:  # SHORT
                if z_current <= exit_z and z_current >= -exit_z:
                    exit_signal = True
                    exit_reason = 'MEAN_REVERT'
                elif z_current < -entry_z * 0.5:
                    exit_signal = True
                    exit_reason = 'OVERSHOOT'
                elif z_current > stop_z:
                    exit_signal = True
                    exit_reason = 'STOP_LOSS'
                elif bars_held >= max_hold_bars:
                    exit_signal = True
                    exit_reason = 'TIMEOUT'
            
            if exit_signal:
                # Закрываем сделку
                current_trade.exit_bar = t
                current_trade.exit_time = timestamps[t]
                current_trade.exit_z = z_current
                current_trade.exit_spread = spread_window[-1]
                current_trade.exit_price1 = price1[t]
                current_trade.exit_price2 = price2[t]
                current_trade.exit_reason = exit_reason
                current_trade.bars_held = bars_held
                
                # P&L: dollar-neutral
                r1 = (price1[t] - current_trade.entry_price1) / current_trade.entry_price1
                r2 = (price2[t] - current_trade.entry_price2) / current_trade.entry_price2
                hr = current_trade.entry_hr
                
                if current_trade.direction == 'LONG':
                    # buy coin1, sell HR * coin2
                    raw_pnl = r1 - hr * r2
                else:
                    # sell coin1, buy HR * coin2
                    raw_pnl = -r1 + hr * r2
                
                # Нормируем на вложенный капитал (1 + |HR|)
                pnl_pct = raw_pnl / (1 + abs(hr)) * 100
                # Минус комиссии (вход + выход, обе ноги)
                pnl_pct -= commission_pct * 4  # 2 ноги × 2 стороны (open+close)
                
                current_trade.pnl_pct = pnl_pct
                trades.append(current_trade)
                current_trade = None
        
        else:
            # Нет позиции — проверяем вход
            if abs(z_current) >= entry_z and abs(z_current) < stop_z:
                direction = 'LONG' if z_current < 0 else 'SHORT'
                current_trade = Trade(
                    entry_bar=t,
                    entry_time=timestamps[t],
                    entry_z=z_current,
                    entry_spread=spread_window[-1],
                    entry_price1=price1[t],
                    entry_price2=price2[t],
                    entry_hr=hr_current,
                    direction=direction,
                )
        
        # Equity update
        if current_trade is not None:
            # MTM P&L текущей открытой позиции
            r1 = (price1[t] - current_trade.entry_price1) / current_trade.entry_price1
            r2 = (price2[t] - current_trade.entry_price2) / current_trade.entry_price2
            hr = current_trade.entry_hr
            if current_trade.direction == 'LONG':
                mtm = (r1 - hr * r2) / (1 + abs(hr))
            else:
                mtm = (-r1 + hr * r2) / (1 + abs(hr))
            equity[t] = equity[current_trade.entry_bar - 1] * (1 + mtm)
        else:
            equity[t] = equity[t - 1]
    
    # ═══ SUMMARY ═══
    result = BacktestResult(
        trades=trades,
        equity_curve=equity,
        spread_series=full_spread,
        zscore_series=full_zscore,
        hr_series=full_hr,
        price1=price1,
        price2=price2,
        timestamps=timestamps,
    )
    
    if len(trades) > 0:
        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        result.total_trades = len(trades)
        result.win_rate = len(wins) / len(trades) * 100
        result.avg_pnl = np.mean(pnls)
        result.total_pnl = np.sum(pnls)
        result.avg_bars_held = np.mean([t.bars_held for t in trades])
        result.max_bars_held = max(t.bars_held for t in trades)
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        result.profit_factor = gross_profit / gross_loss
        
        # Sharpe (annualized)
        if len(pnls) > 1:
            avg_hold = result.avg_bars_held * hours_per_bar  # hours
            trades_per_year = 8760 / max(avg_hold, 1)  # hours in year
            result.sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(min(trades_per_year, 365))
        
        # Max Drawdown
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd * 100
    
    return result


# ═══════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_ohlcv_cached(exchange_name, symbol, timeframe, lookback_days):
    """Загрузка данных с кэшированием."""
    exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    exchange.load_markets()
    
    bars_per_day = {'1h': 24, '4h': 6, '1d': 1, '2h': 12, '15m': 96}.get(timeframe, 6)
    limit = lookback_days * bars_per_day
    
    # OKX ограничивает 300 за запрос — делаем пагинацию
    max_per_request = 300
    all_data = []
    
    if limit <= max_per_request:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        all_data = ohlcv
    else:
        # Пагинация: загружаем частями
        tf_ms = {'1h': 3600000, '4h': 14400000, '1d': 86400000,
                 '2h': 7200000, '15m': 900000}.get(timeframe, 14400000)
        end_ts = exchange.milliseconds()
        start_ts = end_ts - limit * tf_ms
        
        current = start_ts
        while current < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe,
                                             since=int(current), limit=max_per_request)
                if not ohlcv:
                    break
                all_data.extend(ohlcv)
                current = ohlcv[-1][0] + tf_ms
                time.sleep(0.15)
            except Exception:
                break
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    return df


def get_top_coins_cached(exchange_name, limit=100):
    """Получить топ монет по объему."""
    try:
        exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
        exchange.load_markets()
        tickers = exchange.fetch_tickers()
        
        usdt_pairs = {k: v for k, v in tickers.items()
                      if '/USDT' in k and ':' not in k}
        
        valid = []
        for sym, t in usdt_pairs.items():
            try:
                vol = float(t.get('quoteVolume', 0)) or float(t.get('volume', 0))
                if vol > 0:
                    valid.append((sym.replace('/USDT', ''), vol))
            except:
                continue
        
        valid.sort(key=lambda x: -x[1])
        return [c[0] for c in valid[:limit]]
    except Exception as e:
        st.error(f"Ошибка загрузки монет: {e}")
        return ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'AVAX', 'DOT',
                'LINK', 'UNI', 'ATOM', 'LTC', 'NEAR', 'FIL', 'AAVE']


# ═══════════════════════════════════════════════════════
# PRE-ANALYSIS (качество пары)
# ═══════════════════════════════════════════════════════

def analyze_pair_quality(p1, p2, timeframe='4h'):
    """Быстрый анализ качества пары перед бэктестом."""
    s1, s2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    n = min(len(s1), len(s2))
    s1, s2 = s1[:n], s2[:n]
    
    # Cointegration
    pvalue = cointegration_test(s1, s2)
    
    # Kalman
    kf = kalman_hedge_ratio(s1, s2)
    if kf is None:
        return None
    
    spread = kf['spread']
    hr = kf['hr_final']
    hr_std = kf['hr_std']
    
    # Hurst
    hurst = calculate_hurst(spread)
    
    # ADF
    adf_ok = adf_test(spread)
    
    # Half-life
    hl = calculate_halflife(spread)
    
    # OU
    dt = {'1h': 1/24, '4h': 1/6, '1d': 1}.get(timeframe, 1/6)
    ou = calculate_ou_parameters(spread, dt=dt)
    
    # Z-score
    hours_per_bar = {'1h': 1, '4h': 4, '1d': 24}.get(timeframe, 4)
    hl_bars = (hl * 24 / hours_per_bar) if hl < 999 else None
    zscores, zw = calculate_adaptive_robust_zscore(spread, halflife_bars=hl_bars)
    z_current = zscores[~np.isnan(zscores)][-1] if any(~np.isnan(zscores)) else 0
    
    return {
        'pvalue': pvalue,
        'cointegrated': pvalue < 0.05,
        'hedge_ratio': hr,
        'hr_std': hr_std,
        'hurst': hurst,
        'adf_stationary': adf_ok,
        'halflife_days': hl,
        'halflife_hours': hl * 24,
        'z_current': z_current,
        'z_window': zw,
        'ou_theta': ou['theta'] if ou else 0,
        'n_bars': n,
        'spread': spread,
        'hr_series': kf['hedge_ratios'],
    }


# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_backtest_results(result: BacktestResult, coin1: str, coin2: str):
    """Plotly dashboard с результатами бэктеста."""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            f'Equity Curve (капитал)',
            f'Z-Score спреда',
            f'Спред (Kalman)',
            f'Hedge Ratio',
        ],
        row_heights=[0.30, 0.30, 0.25, 0.15],
    )
    
    ts = result.timestamps
    
    # 1. Equity
    fig.add_trace(go.Scatter(
        x=ts, y=result.equity_curve,
        name='Equity', line=dict(color='#00d4aa', width=2),
        fill='tozeroy', fillcolor='rgba(0,212,170,0.1)',
    ), row=1, col=1)
    
    # Начальная линия
    fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
                  opacity=0.5, row=1, col=1)
    
    # 2. Z-score
    z = result.zscore_series
    fig.add_trace(go.Scatter(
        x=ts, y=z, name='Z-Score',
        line=dict(color='#4fc3f7', width=1.5),
    ), row=2, col=1)
    
    # Зоны
    fig.add_hline(y=0, line_dash='solid', line_color='gray',
                  opacity=0.3, row=2, col=1)
    
    # 3. Spread
    fig.add_trace(go.Scatter(
        x=ts, y=result.spread_series, name='Spread',
        line=dict(color='#ffa726', width=1.5),
    ), row=3, col=1)
    
    # 4. HR
    fig.add_trace(go.Scatter(
        x=ts, y=result.hr_series, name='Hedge Ratio',
        line=dict(color='#ab47bc', width=1.5),
    ), row=4, col=1)
    
    # Сделки — markers
    for trade in result.trades:
        color = '#4caf50' if trade.pnl_pct > 0 else '#f44336'
        
        # Entry marker на Z-score
        fig.add_trace(go.Scatter(
            x=[trade.entry_time], y=[trade.entry_z],
            mode='markers',
            marker=dict(symbol='triangle-up' if trade.direction == 'LONG' else 'triangle-down',
                       size=12, color=color, line=dict(width=1, color='white')),
            name=f'{"▲" if trade.direction == "LONG" else "▼"} {trade.pnl_pct:+.2f}%',
            showlegend=False,
            hovertext=f"{trade.direction} | Entry Z={trade.entry_z:.2f}<br>"
                      f"P&L: {trade.pnl_pct:+.2f}% | {trade.exit_reason}<br>"
                      f"Bars: {trade.bars_held}",
        ), row=2, col=1)
        
        # Exit marker
        fig.add_trace(go.Scatter(
            x=[trade.exit_time], y=[trade.exit_z],
            mode='markers',
            marker=dict(symbol='x', size=10, color=color,
                       line=dict(width=2, color=color)),
            showlegend=False,
        ), row=2, col=1)
        
        # Закрашенная зона сделки
        fig.add_vrect(
            x0=trade.entry_time, x1=trade.exit_time,
            fillcolor=color, opacity=0.06, line_width=0,
            row=2, col=1,
        )
    
    fig.update_layout(
        height=900,
        template='plotly_dark',
        title=f'Backtest: {coin1}/{coin2}',
        showlegend=False,
        margin=dict(l=60, r=30, t=50, b=30),
    )
    
    fig.update_yaxes(title_text='Equity', row=1, col=1)
    fig.update_yaxes(title_text='Z-Score', row=2, col=1)
    fig.update_yaxes(title_text='Spread', row=3, col=1)
    fig.update_yaxes(title_text='HR', row=4, col=1)
    
    return fig


def plot_trade_distribution(trades: List[Trade]):
    """Распределение P&L по сделкам."""
    if not trades:
        return None
    
    pnls = [t.pnl_pct for t in trades]
    colors = ['#4caf50' if p > 0 else '#f44336' for p in pnls]
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['P&L по сделкам', 'Распределение P&L'])
    
    # Bar chart
    fig.add_trace(go.Bar(
        x=list(range(1, len(pnls) + 1)),
        y=pnls,
        marker_color=colors,
        name='P&L %',
    ), row=1, col=1)
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=pnls, nbinsx=20,
        marker_color='#4fc3f7',
        name='Distribution',
    ), row=1, col=2)
    
    fig.add_vline(x=0, line_dash='dash', line_color='white',
                  opacity=0.5, row=1, col=2)
    
    fig.update_layout(
        height=350, template='plotly_dark',
        showlegend=False,
        margin=dict(l=50, r=30, t=40, b=30),
    )
    
    return fig


# ═══════════════════════════════════════════════════════
# MULTI-PAIR SCANNER BACKTEST
# ═══════════════════════════════════════════════════════

def scan_and_backtest(exchange_name, coins, timeframe, lookback_days, 
                      entry_z, exit_z, stop_z, max_hold, commission,
                      progress_bar):
    """
    Автоматический скан + бэктест всех пар.
    Берёт первые 2/3 данных для обучения, последнюю 1/3 для теста.
    """
    from statsmodels.tsa.stattools import coint
    
    # 1. Загрузка данных
    progress_bar.progress(0.05, "Загрузка данных...")
    price_data = {}
    for i, coin in enumerate(coins):
        symbol = f"{coin}/USDT"
        try:
            df = fetch_ohlcv_cached(exchange_name, symbol, timeframe, lookback_days)
            if df is not None and len(df) > 50:
                price_data[coin] = df['close']
        except:
            pass
        progress_bar.progress(0.05 + 0.25 * (i + 1) / len(coins),
                            f"Загружено {len(price_data)}/{i+1} монет...")
        time.sleep(0.05)
    
    if len(price_data) < 2:
        st.error("Недостаточно данных")
        return []
    
    # 2. Коинтеграция — быстрый скан
    progress_bar.progress(0.35, "Тест коинтеграции...")
    coin_list = list(price_data.keys())
    pairs_with_pvalue = []
    
    total_pairs = len(coin_list) * (len(coin_list) - 1) // 2
    idx = 0
    for i in range(len(coin_list)):
        for j in range(i + 1, len(coin_list)):
            idx += 1
            c1, c2 = coin_list[i], coin_list[j]
            s1 = price_data[c1].dropna()
            s2 = price_data[c2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < 50:
                continue
            
            pval = cointegration_test(s1[common].values, s2[common].values)
            if pval < 0.10:
                pairs_with_pvalue.append((c1, c2, pval))
            
            if idx % 100 == 0:
                progress_bar.progress(
                    0.35 + 0.25 * idx / total_pairs,
                    f"Коинтеграция: {idx}/{total_pairs}"
                )
    
    # Сортируем по p-value
    pairs_with_pvalue.sort(key=lambda x: x[2])
    top_pairs = pairs_with_pvalue[:30]  # Топ-30
    
    if not top_pairs:
        st.warning("Коинтегрированных пар не найдено")
        return []
    
    st.info(f"📊 Найдено {len(pairs_with_pvalue)} коинтегрированных пар, тестируем топ-{len(top_pairs)}")
    
    # 3. Бэктест каждой пары
    all_results = []
    
    for k, (c1, c2, pval) in enumerate(top_pairs):
        progress_bar.progress(
            0.65 + 0.35 * (k + 1) / len(top_pairs),
            f"Бэктест {c1}/{c2} ({k+1}/{len(top_pairs)})..."
        )
        
        s1 = price_data[c1].dropna()
        s2 = price_data[c2].dropna()
        common = s1.index.intersection(s2.index)
        p1 = s1[common].values
        p2 = s2[common].values
        ts_list = list(common)
        
        if len(p1) < 100:
            continue
        
        # Определяем train window (60% данных)
        train_w = max(50, int(len(p1) * 0.6))
        
        # Быстрая проверка качества
        qual = analyze_pair_quality(p1, p2, timeframe)
        if qual is None:
            continue
        if qual['hurst'] > 0.55:
            continue  # не mean-reverting
        if abs(qual['hedge_ratio']) > 50:
            continue  # unreasonable HR
        
        # Бэктест
        bt = run_backtest(
            p1, p2, ts_list,
            timeframe=timeframe,
            train_window=train_w,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_hold_bars=max_hold,
            commission_pct=commission,
        )
        
        if bt.total_trades >= 1:
            all_results.append({
                'coin1': c1, 'coin2': c2,
                'pvalue': pval,
                'hurst': qual['hurst'],
                'halflife_h': qual['halflife_hours'],
                'hr': qual['hedge_ratio'],
                'result': bt,
            })
    
    return all_results


# ═══════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="Pairs Backtester",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    /* Dark theme fixes */
    .trade-win { color: #4caf50; font-weight: bold; }
    .trade-loss { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Pairs Trading Backtester")
st.caption("v1.0 | Тестирование mean-reversion стратегии на исторических данных")

# ═══ SIDEBAR ═══
with st.sidebar:
    st.header("⚙️ Настройки")
    
    mode = st.radio("Режим", ["🎯 Одна пара", "🔍 Автоскан"], index=0)
    
    st.subheader("Данные")
    exchange = st.selectbox("Биржа", ['okx', 'bybit', 'binance'], index=0)
    timeframe = st.selectbox("Таймфрейм", ['1h', '4h', '1d'], index=1)
    lookback = st.slider("Период (дней)", 30, 365, 180)
    
    st.subheader("Параметры стратегии")
    entry_z = st.slider("Z для входа", 1.0, 4.0, 2.0, 0.1,
                        help="Порог |Z| для открытия позиции")
    exit_z = st.slider("Z для выхода", 0.0, 1.5, 0.5, 0.1,
                       help="Спред вернулся к mean когда |Z| < этого значения")
    stop_z = st.slider("Z для стопа", 3.0, 7.0, 4.5, 0.5,
                       help="Стоп-лосс если |Z| превышает")
    max_hold = st.slider("Макс. баров в сделке", 20, 300, 100, 10)
    commission = st.slider("Комиссия (%)", 0.0, 0.5, 0.1, 0.01,
                          help="Комиссия за сделку (одна нога)")

# ═══ MAIN ═══

if mode == "🎯 Одна пара":
    st.subheader("Бэктест одной пары")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        coin1 = st.text_input("Монета 1", value="XRP", 
                              help="Тикер без /USDT").upper().strip()
    with col2:
        coin2 = st.text_input("Монета 2", value="AVAX",
                              help="Тикер без /USDT").upper().strip()
    with col3:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Запустить", type="primary", use_container_width=True)
    
    if run_btn and coin1 and coin2:
        progress = st.progress(0, "Загрузка данных...")
        
        try:
            # Загрузка
            progress.progress(0.1, f"Загружаю {coin1}/USDT...")
            df1 = fetch_ohlcv_cached(exchange, f"{coin1}/USDT", timeframe, lookback)
            progress.progress(0.3, f"Загружаю {coin2}/USDT...")
            df2 = fetch_ohlcv_cached(exchange, f"{coin2}/USDT", timeframe, lookback)
            
            if df1 is None or df2 is None:
                st.error("❌ Не удалось загрузить данные. Проверьте тикеры.")
                st.stop()
            
            # Align
            common = df1.index.intersection(df2.index)
            if len(common) < 50:
                st.error(f"❌ Слишком мало общих баров: {len(common)}")
                st.stop()
            
            p1 = df1.loc[common, 'close'].values
            p2 = df2.loc[common, 'close'].values
            timestamps = list(common)
            
            st.info(f"📊 Загружено {len(common)} баров ({timeframe}) с {common[0].strftime('%Y-%m-%d')} по {common[-1].strftime('%Y-%m-%d')}")
            
            # Анализ качества
            progress.progress(0.5, "Анализ качества пары...")
            qual = analyze_pair_quality(p1, p2, timeframe)
            
            if qual:
                qcol1, qcol2, qcol3, qcol4, qcol5, qcol6 = st.columns(6)
                qcol1.metric("P-value", f"{qual['pvalue']:.4f}",
                            delta="✅ Coint" if qual['cointegrated'] else "❌ No coint")
                qcol2.metric("Hurst", f"{qual['hurst']:.3f}",
                            delta="✅ MR" if qual['hurst'] < 0.45 else "⚠️ Weak")
                qcol3.metric("Half-life", f"{qual['halflife_hours']:.1f}ч")
                qcol4.metric("HR", f"{qual['hedge_ratio']:.4f}")
                qcol5.metric("ADF", "✅" if qual['adf_stationary'] else "❌")
                qcol6.metric("Z сейчас", f"{qual['z_current']:.2f}")
            
            # Бэктест
            progress.progress(0.6, "Walk-forward бэктест...")
            train_w = max(50, int(len(p1) * 0.4))
            
            result = run_backtest(
                p1, p2, timestamps,
                timeframe=timeframe,
                train_window=train_w,
                entry_z=entry_z,
                exit_z=exit_z,
                stop_z=stop_z,
                max_hold_bars=max_hold,
                commission_pct=commission,
            )
            
            progress.progress(1.0, "Готово!")
            time.sleep(0.3)
            progress.empty()
            
            # ═══ РЕЗУЛЬТАТЫ ═══
            st.divider()
            st.subheader("📈 Результаты бэктеста")
            
            if result.total_trades == 0:
                st.warning("⚠️ Ни одной сделки за период. Попробуйте снизить Z для входа или увеличить период.")
            else:
                # KPI
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Сделок", result.total_trades)
                m2.metric("Win Rate", f"{result.win_rate:.1f}%",
                         delta="good" if result.win_rate > 50 else "low")
                m3.metric("Total P&L", f"{result.total_pnl:+.2f}%",
                         delta="profit" if result.total_pnl > 0 else "loss")
                m4.metric("Avg P&L", f"{result.avg_pnl:+.2f}%")
                m5.metric("Max DD", f"{result.max_drawdown:.1f}%")
                m6.metric("Profit Factor", f"{result.profit_factor:.2f}")
                
                m7, m8, m9 = st.columns(3)
                m7.metric("Sharpe", f"{result.sharpe:.2f}")
                m8.metric("Avg Hold", f"{result.avg_bars_held:.0f} баров")
                m9.metric("Max Hold", f"{result.max_bars_held} баров")
                
                # Графики
                fig_main = plot_backtest_results(result, coin1, coin2)
                st.plotly_chart(fig_main, use_container_width=True)
                
                fig_dist = plot_trade_distribution(result.trades)
                if fig_dist:
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Таблица сделок
                st.subheader("📋 Журнал сделок")
                trades_data = []
                for i, t in enumerate(result.trades, 1):
                    trades_data.append({
                        '#': i,
                        'Вход': t.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(t.entry_time, 'strftime') else str(t.entry_time),
                        'Выход': t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time and hasattr(t.exit_time, 'strftime') else str(t.exit_time),
                        'Dir': t.direction,
                        'Entry Z': f"{t.entry_z:.2f}",
                        'Exit Z': f"{t.exit_z:.2f}",
                        'HR': f"{t.entry_hr:.4f}",
                        'Bars': t.bars_held,
                        'P&L %': f"{t.pnl_pct:+.2f}",
                        'Причина': t.exit_reason,
                    })
                
                df_trades = pd.DataFrame(trades_data)
                st.dataframe(df_trades, use_container_width=True, hide_index=True)
                
                # Exit reasons breakdown
                if result.trades:
                    reasons = {}
                    for t in result.trades:
                        r = t.exit_reason
                        if r not in reasons:
                            reasons[r] = {'count': 0, 'pnl': 0}
                        reasons[r]['count'] += 1
                        reasons[r]['pnl'] += t.pnl_pct
                    
                    st.subheader("📊 Выходы по причинам")
                    rcols = st.columns(len(reasons))
                    for col, (reason, stats) in zip(rcols, reasons.items()):
                        avg = stats['pnl'] / stats['count']
                        col.metric(
                            reason.replace('_', ' '),
                            f"{stats['count']} сделок",
                            f"avg {avg:+.2f}%"
                        )
        
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            import traceback
            st.code(traceback.format_exc())

else:
    # ═══ АВТОСКАН ═══
    st.subheader("🔍 Автоматический скан + бэктест")
    st.caption("Сканирует пары на коинтеграцию и тестирует каждую")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        n_coins = st.slider("Количество монет", 10, 50, 20,
                            help="Больше монет = больше пар = дольше")
    with col2:
        run_scan = st.button("🔍 Сканировать и тестировать", type="primary",
                            use_container_width=True)
    
    if run_scan:
        progress = st.progress(0, "Инициализация...")
        
        try:
            # Получаем монеты
            coins = get_top_coins_cached(exchange, n_coins)
            st.info(f"📊 Монеты: {', '.join(coins[:20])}{'...' if len(coins) > 20 else ''}")
            
            # Скан + бэктест
            all_results = scan_and_backtest(
                exchange, coins, timeframe, lookback,
                entry_z, exit_z, stop_z, max_hold, commission,
                progress
            )
            
            progress.progress(1.0, "Готово!")
            time.sleep(0.3)
            progress.empty()
            
            if not all_results:
                st.warning("⚠️ Ни одной пары с торговыми сигналами")
            else:
                # Сортировка по total P&L
                all_results.sort(key=lambda x: -x['result'].total_pnl)
                
                st.success(f"✅ Протестировано {len(all_results)} пар с торговыми сигналами")
                
                # Summary table
                summary = []
                for r in all_results:
                    bt = r['result']
                    summary.append({
                        'Пара': f"{r['coin1']}/{r['coin2']}",
                        'P-val': f"{r['pvalue']:.4f}",
                        'Hurst': f"{r['hurst']:.3f}",
                        'HL(ч)': f"{r['halflife_h']:.0f}",
                        'HR': f"{r['hr']:.4f}",
                        'Сделок': bt.total_trades,
                        'Win%': f"{bt.win_rate:.0f}",
                        'Total P&L': f"{bt.total_pnl:+.1f}%",
                        'Avg P&L': f"{bt.avg_pnl:+.2f}%",
                        'MaxDD': f"{bt.max_drawdown:.1f}%",
                        'Sharpe': f"{bt.sharpe:.1f}",
                        'PF': f"{bt.profit_factor:.2f}",
                    })
                
                df_summary = pd.DataFrame(summary)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
                
                # Detailed view per pair
                st.divider()
                st.subheader("📈 Детали по парам")
                
                for r in all_results[:10]:  # Топ-10
                    bt = r['result']
                    with st.expander(
                        f"{'🟢' if bt.total_pnl > 0 else '🔴'} "
                        f"{r['coin1']}/{r['coin2']} — "
                        f"P&L: {bt.total_pnl:+.1f}% | "
                        f"{bt.total_trades} сделок | "
                        f"WR: {bt.win_rate:.0f}%"
                    ):
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Total P&L", f"{bt.total_pnl:+.1f}%")
                        mc2.metric("Win Rate", f"{bt.win_rate:.0f}%")
                        mc3.metric("Sharpe", f"{bt.sharpe:.1f}")
                        mc4.metric("Max DD", f"{bt.max_drawdown:.1f}%")
                        
                        fig = plot_backtest_results(bt, r['coin1'], r['coin2'])
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.divider()
st.caption("""
**Pairs Trading Backtester v1.0** | Kalman Filter HR + MAD Z-Score + Walk-Forward

⚠️ Это бэктест — реальная торговля может отличаться из-за проскальзывания, ликвидности, задержек исполнения.

Параметры стратегии:
- **Entry**: открываем позицию когда |Z-score| > порог входа
- **Exit**: закрываем когда Z-score вернулся к ~0 (mean reversion), достиг стопа, или таймаут
- **Dollar-neutral**: на каждую сделку — buy $1 одной ноги, sell $HR другой ноги
- **P&L**: нормирован на вложенный капитал (1+|HR|), минус 4× комиссия (open/close × 2 ноги)
""")
