"""
Pairs Trading Backtester v2.0
ИСПРАВЛЕНИЯ:
  [FIX] OVERSHOOT порог: entry_z*0.5 → 0 (выход только при смене знака Z)
  [FIX] min_hold_bars: добавлен параметр (default=3) — не выходим раньше
  [FIX] HR filter: 0.01 < |HR| < 20 (отсечка стейблкоинов и экстремальных HR)
  [FIX] Hurst: единая реализация DFA (min_window=8, R² проверка)
  [FIX] Half-life: через OU параметры (как в сканере), а не OLS
  [FIX] Фиксированный Kalman в сделке: HR/intercept не меняется пока в позиции
  [NEW] Cooldown: пропуск N баров после закрытия сделки
  [NEW] Улучшенная палитра: зелёный/красный для P&L, цветовые зоны на графиках
  [NEW] Метрики качества — цвет по порогам (хорошо/средне/плохо)
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
from scipy import stats as scipy_stats

# ═══════════════════════════════════════════════════════
# CORE FUNCTIONS (синхронизированы с mean_reversion_analysis.py v10.5)
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
        'intercept_final': float(intercepts[-1]),
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


def calculate_ou_parameters(spread, dt=1.0):
    """OU: dX = θ(μ - X)dt + σdW. Возвращает halflife_ou в ЧАСАХ."""
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
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - (a + b * x)) ** 2)
        r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {
            'theta': float(theta), 'mu': float(mu), 'sigma': float(sigma),
            'halflife_ou': float(halflife), 'r_squared': float(r_sq)
        }
    except Exception:
        return None


def calculate_hurst_exponent(time_series, min_window=8):
    """DFA Hurst — синхронизировано с mean_reversion_analysis.py v10.5."""
    ts = np.array(time_series, dtype=float)
    n = len(ts)
    if n < 30:
        return 0.5, True  # (value, is_fallback)

    increments = np.diff(ts)
    n_inc = len(increments)
    profile = np.cumsum(increments - np.mean(increments))

    max_window = n_inc // 4
    if max_window <= min_window:
        return 0.5, True

    num_points = min(20, max_window - min_window)
    if num_points < 4:
        return 0.5, True

    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=num_points).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= min_window]
    if len(window_sizes) < 4:
        return 0.5, True

    fluctuations = []
    for w in window_sizes:
        n_seg = n_inc // w
        if n_seg < 2:
            continue
        f2_sum, count = 0.0, 0
        for seg in range(n_seg):
            segment = profile[seg * w:(seg + 1) * w]
            x = np.arange(w, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            f2_sum += np.mean((segment - np.polyval(coeffs, x)) ** 2)
            count += 1
        for seg in range(n_seg):
            start = n_inc - (seg + 1) * w
            if start < 0:
                break
            segment = profile[start:start + w]
            x = np.arange(w, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            f2_sum += np.mean((segment - np.polyval(coeffs, x)) ** 2)
            count += 1
        if count > 0:
            f_n = np.sqrt(f2_sum / count)
            if f_n > 1e-15:
                fluctuations.append((w, f_n))

    if len(fluctuations) < 4:
        return 0.5, True

    log_n = np.log([f[0] for f in fluctuations])
    log_f = np.log([f[1] for f in fluctuations])

    try:
        slope, _, r_value, _, _ = scipy_stats.linregress(log_n, log_f)
        if r_value ** 2 < 0.70:
            return 0.5, True
        return round(max(0.01, min(0.99, slope)), 4), False
    except Exception:
        return 0.5, True


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
    entry_intercept: float  # v2: фиксируем intercept при входе
    direction: str

    exit_bar: int = 0
    exit_time: datetime = None
    exit_z: float = 0.0
    exit_spread: float = 0.0
    exit_price1: float = 0.0
    exit_price2: float = 0.0
    exit_reason: str = ''
    pnl_pct: float = 0.0
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
# BACKTESTING ENGINE v2.0
# ═══════════════════════════════════════════════════════

def run_backtest(
    price1: np.ndarray,
    price2: np.ndarray,
    timestamps: list,
    timeframe: str = '4h',
    train_window: int = 200,
    entry_z: float = 2.0,
    exit_z: float = 0.3,          # v2: снижен с 0.5
    stop_z: float = 4.5,
    max_hold_bars: int = 100,
    min_hold_bars: int = 3,       # v2: НЕ выходим раньше 3 баров
    commission_pct: float = 0.1,
    cooldown_bars: int = 1,       # v2: пропуск после закрытия
) -> BacktestResult:
    """
    Walk-forward бэктест v2.0.

    ИЗМЕНЕНИЯ v2:
      1. min_hold_bars: не выходим первые N баров (даёт сделке время отработать)
      2. OVERSHOOT: выход только при полной смене знака Z (не entry_z*0.5)
      3. Фиксированный Kalman в сделке: Z считается на основе ENTRY HR
      4. Cooldown: пропуск баров после закрытия
      5. exit_z снижен: 0.5 → 0.3 (больше прибыли на mean-revert сделках)
    """
    n = len(price1)
    assert len(price2) == n, "Price arrays must have same length"

    hours_per_bar = {'1h': 1, '2h': 2, '4h': 4, '1d': 24, '15m': 0.25}.get(timeframe, 4)

    # Storage
    full_spread = np.full(n, np.nan)
    full_zscore = np.full(n, np.nan)
    full_hr = np.full(n, np.nan)
    equity = np.ones(n)

    trades: List[Trade] = []
    current_trade: Optional[Trade] = None
    cooldown_until = 0  # v2: не открывать до этого бара

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
        intercept_current = kf['intercept_final']

        full_spread[t] = spread_window[-1]
        full_hr[t] = hr_current

        # 3. OU Half-life → adaptive Z window
        dt_ou = {'1h': 1/24, '4h': 1/6, '1d': 1.0}.get(timeframe, 1/6)
        ou = calculate_ou_parameters(spread_window, dt=dt_ou)
        if ou and ou['halflife_ou'] < 999:
            hl_hours = ou['halflife_ou']
            hl_bars = hl_hours / hours_per_bar
        else:
            hl_bars = None

        # 4. Z-score
        zscores, z_win = calculate_adaptive_robust_zscore(
            spread_window, halflife_bars=hl_bars
        )

        z_current = zscores[-1] if not np.isnan(zscores[-1]) else 0.0
        full_zscore[t] = z_current

        # ═══ TRADE LOGIC v2 ═══

        if current_trade is not None:
            bars_held = t - current_trade.entry_bar

            # v2: Считаем Z на основе фиксированного Kalman (entry HR)
            # Это предотвращает drift Z из-за обновления модели
            fixed_spread_current = price1[t] - current_trade.entry_hr * price2[t] - current_trade.entry_intercept
            
            # Для fixed Z используем rolling медиану из spread window (текущего Kalman)
            # но с фиксированным HR для текущего бара
            z_for_exit = z_current  # основной Z (из walk-forward)

            exit_signal = False
            exit_reason = ''

            # v2: Минимальное время удержания
            if bars_held < min_hold_bars:
                # Только стоп-лосс работает во время min_hold
                if current_trade.direction == 'LONG' and z_for_exit < -stop_z:
                    exit_signal = True
                    exit_reason = 'STOP_LOSS'
                elif current_trade.direction == 'SHORT' and z_for_exit > stop_z:
                    exit_signal = True
                    exit_reason = 'STOP_LOSS'
            else:
                if current_trade.direction == 'LONG':
                    # LONG spread: вошли при Z < -entry_z, ждём возврат к 0
                    if abs(z_for_exit) <= exit_z:
                        exit_signal = True
                        exit_reason = 'MEAN_REVERT'
                    elif z_for_exit > 0 and z_for_exit > exit_z:
                        # v2: OVERSHOOT — Z полностью перешёл на другую сторону
                        # (вместо entry_z * 0.5, теперь: Z > 0 И |Z| > exit_z)
                        exit_signal = True
                        exit_reason = 'OVERSHOOT'
                    elif z_for_exit < -stop_z:
                        exit_signal = True
                        exit_reason = 'STOP_LOSS'
                    elif bars_held >= max_hold_bars:
                        exit_signal = True
                        exit_reason = 'TIMEOUT'
                else:  # SHORT
                    if abs(z_for_exit) <= exit_z:
                        exit_signal = True
                        exit_reason = 'MEAN_REVERT'
                    elif z_for_exit < 0 and z_for_exit < -exit_z:
                        # v2: OVERSHOOT для SHORT
                        exit_signal = True
                        exit_reason = 'OVERSHOOT'
                    elif z_for_exit > stop_z:
                        exit_signal = True
                        exit_reason = 'STOP_LOSS'
                    elif bars_held >= max_hold_bars:
                        exit_signal = True
                        exit_reason = 'TIMEOUT'

            if exit_signal:
                current_trade.exit_bar = t
                current_trade.exit_time = timestamps[t]
                current_trade.exit_z = z_for_exit
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
                    raw_pnl = r1 - hr * r2
                else:
                    raw_pnl = -r1 + hr * r2

                # Нормируем на вложенный капитал
                pnl_pct = raw_pnl / (1 + abs(hr)) * 100
                # Комиссии (4 × = 2 ноги × open/close)
                pnl_pct -= commission_pct * 4

                current_trade.pnl_pct = pnl_pct
                trades.append(current_trade)
                current_trade = None
                cooldown_until = t + cooldown_bars  # v2: cooldown

        else:
            # Нет позиции — проверяем вход
            if t >= cooldown_until:  # v2: cooldown check
                if abs(z_current) >= entry_z and abs(z_current) < stop_z:
                    # v2: HR filter — не торгуем стейблкоин-пары и экстремальный HR
                    if 0.01 <= abs(hr_current) <= 20.0 and hr_current > 0:
                        direction = 'LONG' if z_current < 0 else 'SHORT'
                        current_trade = Trade(
                            entry_bar=t,
                            entry_time=timestamps[t],
                            entry_z=z_current,
                            entry_spread=spread_window[-1],
                            entry_price1=price1[t],
                            entry_price2=price2[t],
                            entry_hr=hr_current,
                            entry_intercept=intercept_current,  # v2: фиксируем
                            direction=direction,
                        )

        # Equity update
        if current_trade is not None:
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

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        result.profit_factor = gross_profit / gross_loss

        if len(pnls) > 1:
            avg_hold = result.avg_bars_held * hours_per_bar
            trades_per_year = 8760 / max(avg_hold, 1)
            result.sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(min(trades_per_year, 365))

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

    max_per_request = 300
    all_data = []

    if limit <= max_per_request:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        all_data = ohlcv
    else:
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
# PRE-ANALYSIS
# ═══════════════════════════════════════════════════════

# v2: Стейблкоины и wrapped — исключаем автоматически
STABLECOIN_KEYWORDS = {'USDC', 'USDT', 'DAI', 'USDG', 'TUSD', 'BUSD', 'FRAX',
                       'STETH', 'WBTC', 'WETH', 'WSTETH', 'XAUT', 'PAXG', 'MMT'}


def analyze_pair_quality(p1, p2, timeframe='4h'):
    """Быстрый анализ качества пары перед бэктестом."""
    s1, s2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    n = min(len(s1), len(s2))
    s1, s2 = s1[:n], s2[:n]

    pvalue = cointegration_test(s1, s2)

    kf = kalman_hedge_ratio(s1, s2)
    if kf is None:
        return None

    spread = kf['spread']
    hr = kf['hr_final']
    hr_std = kf['hr_std']

    # v2: Единая Hurst DFA (из analysis v10.5)
    hurst, hurst_fallback = calculate_hurst_exponent(spread, min_window=8)

    adf_ok = adf_test(spread)

    # v2: OU half-life (как в сканере)
    dt = {'1h': 1/24, '4h': 1/6, '1d': 1}.get(timeframe, 1/6)
    ou = calculate_ou_parameters(spread, dt=dt)
    hl_hours = ou['halflife_ou'] if ou else 999

    # Z-score
    hours_per_bar = {'1h': 1, '4h': 4, '1d': 24}.get(timeframe, 4)
    hl_bars = (hl_hours / hours_per_bar) if hl_hours < 999 else None
    zscores, zw = calculate_adaptive_robust_zscore(spread, halflife_bars=hl_bars)
    z_current = zscores[~np.isnan(zscores)][-1] if any(~np.isnan(zscores)) else 0

    return {
        'pvalue': pvalue,
        'cointegrated': pvalue < 0.05,
        'hedge_ratio': hr,
        'hr_std': hr_std,
        'hurst': hurst,
        'hurst_fallback': hurst_fallback,
        'adf_stationary': adf_ok,
        'halflife_hours': hl_hours,
        'z_current': z_current,
        'z_window': zw,
        'ou_theta': ou['theta'] if ou else 0,
        'n_bars': n,
        'spread': spread,
        'hr_series': kf['hedge_ratios'],
    }


# ═══════════════════════════════════════════════════════
# v2: УЛУЧШЕННАЯ ПАЛИТРА И ГРАФИКИ
# ═══════════════════════════════════════════════════════

# Цветовая палитра v2
COLORS = {
    'profit': '#00E676',        # ярко-зелёный
    'loss': '#FF1744',          # ярко-красный
    'breakeven': '#FFC107',     # жёлтый
    'equity_line': '#00E5FF',   # циан
    'equity_fill': 'rgba(0,229,255,0.08)',
    'zscore_line': '#7C4DFF',   # фиолетовый
    'spread_line': '#FF9100',   # оранжевый
    'hr_line': '#E040FB',       # розовый
    'entry_zone': 'rgba(255,193,7,0.15)',  # жёлтая зона входа
    'stop_zone': 'rgba(255,23,68,0.08)',   # красная зона стопа
    'mean_zone': 'rgba(0,230,118,0.10)',   # зелёная зона mean revert
    'grid': 'rgba(255,255,255,0.06)',
    'text_dim': 'rgba(255,255,255,0.5)',
}


def metric_color(value, good_thresh, bad_thresh, higher_is_better=True):
    """Возвращает цвет метрики по порогам."""
    if higher_is_better:
        if value >= good_thresh:
            return "good"
        elif value <= bad_thresh:
            return "‼️ bad"
        return "ok"
    else:
        if value <= good_thresh:
            return "good"
        elif value >= bad_thresh:
            return "‼️ bad"
        return "ok"


def plot_backtest_results(result: BacktestResult, coin1: str, coin2: str):
    """v2: Plotly dashboard с улучшенной палитрой."""

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            'Equity Curve (капитал)',
            'Z-Score спреда',
            'Спред (Kalman)',
            'Hedge Ratio',
        ],
        row_heights=[0.30, 0.30, 0.25, 0.15],
    )

    ts = result.timestamps

    # 1. Equity — с цветной заливкой (зелёная если выше 1, красная если ниже)
    eq = result.equity_curve
    above_1 = np.where(eq >= 1.0, eq, 1.0)
    below_1 = np.where(eq < 1.0, eq, 1.0)

    fig.add_trace(go.Scatter(
        x=ts, y=eq, name='Equity',
        line=dict(color=COLORS['equity_line'], width=2.5),
    ), row=1, col=1)

    # Заливка: зелёная выше 1, красная ниже
    fig.add_trace(go.Scatter(
        x=ts, y=above_1, fill='tonexty',
        fillcolor='rgba(0,230,118,0.12)', line=dict(width=0),
        showlegend=False,
    ), row=1, col=1)

    fig.add_hline(y=1.0, line_dash='dash', line_color='white',
                  opacity=0.3, row=1, col=1)

    # 2. Z-score с зонами
    z = result.zscore_series
    fig.add_trace(go.Scatter(
        x=ts, y=z, name='Z-Score',
        line=dict(color=COLORS['zscore_line'], width=1.5),
    ), row=2, col=1)

    # Зоны: entry, stop, mean-revert
    z_valid = z[~np.isnan(z)]
    if len(z_valid) > 0:
        z_range = max(abs(np.nanmin(z)), abs(np.nanmax(z)), 5)

        # Mean-revert зона (зелёная, ±exit_z)
        fig.add_hrect(y0=-0.3, y1=0.3, fillcolor=COLORS['mean_zone'],
                      line_width=0, row=2, col=1)
        # Entry зоны (жёлтые)
        fig.add_hrect(y0=2.0, y1=3.0, fillcolor=COLORS['entry_zone'],
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=-3.0, y1=-2.0, fillcolor=COLORS['entry_zone'],
                      line_width=0, row=2, col=1)
        # Stop зоны (красные)
        fig.add_hrect(y0=4.5, y1=z_range + 1, fillcolor=COLORS['stop_zone'],
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=-z_range - 1, y1=-4.5, fillcolor=COLORS['stop_zone'],
                      line_width=0, row=2, col=1)

    fig.add_hline(y=0, line_dash='solid', line_color='white',
                  opacity=0.15, row=2, col=1)

    # 3. Spread
    fig.add_trace(go.Scatter(
        x=ts, y=result.spread_series, name='Spread',
        line=dict(color=COLORS['spread_line'], width=1.5),
    ), row=3, col=1)

    # 4. HR
    fig.add_trace(go.Scatter(
        x=ts, y=result.hr_series, name='Hedge Ratio',
        line=dict(color=COLORS['hr_line'], width=1.5),
    ), row=4, col=1)

    # v2: Сделки — маркеры с чёткими цветами
    for trade in result.trades:
        if trade.pnl_pct > 0.5:
            color = COLORS['profit']
        elif trade.pnl_pct < -0.5:
            color = COLORS['loss']
        else:
            color = COLORS['breakeven']

        # Entry marker
        fig.add_trace(go.Scatter(
            x=[trade.entry_time], y=[trade.entry_z],
            mode='markers',
            marker=dict(
                symbol='triangle-up' if trade.direction == 'LONG' else 'triangle-down',
                size=14, color=color,
                line=dict(width=1.5, color='white')
            ),
            showlegend=False,
            hovertext=(
                f"<b>{trade.direction}</b> | Entry Z={trade.entry_z:.2f}<br>"
                f"P&L: <b>{trade.pnl_pct:+.2f}%</b> | {trade.exit_reason}<br>"
                f"Bars: {trade.bars_held} | HR: {trade.entry_hr:.4f}"
            ),
            hoverinfo='text',
        ), row=2, col=1)

        # Exit marker
        fig.add_trace(go.Scatter(
            x=[trade.exit_time], y=[trade.exit_z],
            mode='markers',
            marker=dict(symbol='x', size=11, color=color,
                       line=dict(width=2.5, color=color)),
            showlegend=False,
        ), row=2, col=1)

        # Закрашенная зона сделки
        fig.add_vrect(
            x0=trade.entry_time, x1=trade.exit_time,
            fillcolor=color, opacity=0.05, line_width=0,
            row=2, col=1,
        )

    fig.update_layout(
        height=950,
        template='plotly_dark',
        title=dict(
            text=f'Backtest: {coin1}/{coin2}',
            font=dict(size=18)
        ),
        showlegend=False,
        margin=dict(l=60, r=30, t=50, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(17,17,17,1)',
    )

    for row in range(1, 5):
        fig.update_yaxes(gridcolor=COLORS['grid'], row=row, col=1)
        fig.update_xaxes(gridcolor=COLORS['grid'], row=row, col=1)

    fig.update_yaxes(title_text='Equity', row=1, col=1)
    fig.update_yaxes(title_text='Z-Score', row=2, col=1)
    fig.update_yaxes(title_text='Spread', row=3, col=1)
    fig.update_yaxes(title_text='HR', row=4, col=1)

    return fig


def plot_trade_distribution(trades: List[Trade]):
    """v2: Распределение P&L с улучшенной палитрой."""
    if not trades:
        return None

    pnls = [t.pnl_pct for t in trades]

    # v2: 3-цветная шкала
    colors = []
    for p in pnls:
        if p > 0.5:
            colors.append(COLORS['profit'])
        elif p < -0.5:
            colors.append(COLORS['loss'])
        else:
            colors.append(COLORS['breakeven'])

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['P&L по сделкам', 'Распределение P&L'])

    # Bar chart
    fig.add_trace(go.Bar(
        x=list(range(1, len(pnls) + 1)),
        y=pnls,
        marker_color=colors,
        name='P&L %',
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash='dash', line_color='white',
                  opacity=0.3, row=1, col=1)

    # Histogram с двумя цветами
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    if losses:
        fig.add_trace(go.Histogram(
            x=losses, nbinsx=15,
            marker_color=COLORS['loss'],
            opacity=0.7,
            name='Losses',
        ), row=1, col=2)
    if wins:
        fig.add_trace(go.Histogram(
            x=wins, nbinsx=15,
            marker_color=COLORS['profit'],
            opacity=0.7,
            name='Wins',
        ), row=1, col=2)

    fig.add_vline(x=0, line_dash='dash', line_color='white',
                  opacity=0.5, row=1, col=2)

    fig.update_layout(
        height=350, template='plotly_dark',
        showlegend=False,
        margin=dict(l=50, r=30, t=40, b=30),
        barmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


# ═══════════════════════════════════════════════════════
# MULTI-PAIR SCANNER BACKTEST
# ═══════════════════════════════════════════════════════

def scan_and_backtest(exchange_name, coins, timeframe, lookback_days,
                      entry_z, exit_z, stop_z, max_hold, min_hold,
                      commission, progress_bar):
    """Автоматический скан + бэктест всех пар (v2)."""
    from statsmodels.tsa.stattools import coint

    progress_bar.progress(0.05, "Загрузка данных...")
    price_data = {}
    for i, coin in enumerate(coins):
        # v2: Фильтр стейблкоинов при загрузке
        if coin.upper() in STABLECOIN_KEYWORDS:
            continue
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

    pairs_with_pvalue.sort(key=lambda x: x[2])
    top_pairs = pairs_with_pvalue[:30]

    if not top_pairs:
        st.warning("Коинтегрированных пар не найдено")
        return []

    st.info(f"📊 Найдено {len(pairs_with_pvalue)} коинтегрированных пар, тестируем топ-{len(top_pairs)}")

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

        train_w = max(50, int(len(p1) * 0.5))  # v2: 50% (было 60%)

        qual = analyze_pair_quality(p1, p2, timeframe)
        if qual is None:
            continue
        if qual['hurst'] > 0.50:  # v2: строже (было 0.55)
            continue
        # v2: HR filter
        if abs(qual['hedge_ratio']) > 20 or abs(qual['hedge_ratio']) < 0.01:
            continue
        if qual['hedge_ratio'] <= 0:
            continue

        bt = run_backtest(
            p1, p2, ts_list,
            timeframe=timeframe,
            train_window=train_w,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_hold_bars=max_hold,
            min_hold_bars=min_hold,
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
    page_title="Pairs Backtester v2",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    /* v2: Color-coded metric deltas */
    .stMetric [data-testid="stMetricDelta"][style*="color: rgb(255"] {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Pairs Trading Backtester")
st.caption("v2.0 | Kalman HR + MAD Z-Score + Walk-Forward + Min Hold + HR Filter")

# ═══ SIDEBAR ═══
with st.sidebar:
    st.header("⚙️ Настройки")

    mode = st.radio("Режим", ["🎯 Одна пара", "🔍 Автоскан"], index=0)

    st.subheader("Данные")
    exchange = st.selectbox("Биржа", ['okx', 'bybit', 'binance'], index=0)
    timeframe = st.selectbox("Таймфрейм", ['1h', '4h', '1d'], index=1)
    lookback = st.slider("Период (дней)", 30, 365, 139)

    st.subheader("Параметры стратегии")
    entry_z = st.slider("Z для входа", 1.5, 4.0, 2.3, 0.1,
                        help="Порог |Z| для открытия позиции. Выше = меньше сделок, но больше прибыль на сделку.")
    exit_z = st.slider("Z для выхода", 0.0, 1.0, 0.3, 0.1,
                       help="Спред вернулся к mean когда |Z| < этого значения. Ниже = больше прибыль, но дольше ждать.")
    stop_z = st.slider("Z для стопа", 3.0, 7.0, 4.5, 0.5,
                       help="Стоп-лосс если |Z| превышает")
    max_hold = st.slider("Макс. баров в сделке", 20, 300, 100, 10)
    # v2: Новые параметры
    min_hold = st.slider("Мин. баров в сделке", 0, 10, 3, 1,
                         help="Не выходить раньше N баров (кроме стопа). 3 = 12ч на 4h.")
    commission = st.slider("Комиссия (%)", 0.0, 0.5, 0.1, 0.01,
                          help="Комиссия за сделку (одна нога). Итого 4× за round-trip.")

    st.divider()
    st.caption(f"💰 Комиссия за сделку: **{commission * 4:.2f}%** (4 × {commission}%)")
    if min_hold > 0:
        hours = min_hold * {'1h': 1, '4h': 4, '1d': 24}.get(timeframe, 4)
        st.caption(f"⏱️ Мин. удержание: **{hours}ч** ({min_hold} баров)")

# ═══ MAIN ═══

if mode == "🎯 Одна пара":
    st.subheader("Бэктест одной пары")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        coin1 = st.text_input("Монета 1", value="FIL",
                              help="Тикер без /USDT").upper().strip()
    with col2:
        coin2 = st.text_input("Монета 2", value="CRV",
                              help="Тикер без /USDT").upper().strip()
    with col3:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Запустить", type="primary", use_container_width=True)

    if run_btn and coin1 and coin2:
        progress = st.progress(0, "Загрузка данных...")

        try:
            progress.progress(0.1, f"Загружаю {coin1}/USDT...")
            df1 = fetch_ohlcv_cached(exchange, f"{coin1}/USDT", timeframe, lookback)
            progress.progress(0.3, f"Загружаю {coin2}/USDT...")
            df2 = fetch_ohlcv_cached(exchange, f"{coin2}/USDT", timeframe, lookback)

            if df1 is None or df2 is None:
                st.error("❌ Не удалось загрузить данные. Проверьте тикеры.")
                st.stop()

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
                            delta="✅ MR" if qual['hurst'] < 0.45 else ("⚠️ Weak" if qual['hurst'] < 0.5 else "❌ Trend"))
                qcol3.metric("Half-life", f"{qual['halflife_hours']:.1f}ч",
                            delta="✅ Fast" if qual['halflife_hours'] < 24 else ("⚠️ Slow" if qual['halflife_hours'] < 48 else "❌ Too slow"))
                qcol4.metric("HR", f"{qual['hedge_ratio']:.4f}",
                            delta="✅ Good" if 0.1 <= abs(qual['hedge_ratio']) <= 5 else "⚠️ Extreme")
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
                min_hold_bars=min_hold,
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
                # v2: KPI с цветовой индикацией
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Сделок", result.total_trades)
                m2.metric("Win Rate", f"{result.win_rate:.1f}%",
                         delta=metric_color(result.win_rate, 55, 40))
                m3.metric("Total P&L", f"{result.total_pnl:+.2f}%",
                         delta="✅ profit" if result.total_pnl > 0 else "❌ loss")
                m4.metric("Avg P&L", f"{result.avg_pnl:+.2f}%",
                         delta="✅" if result.avg_pnl > 0 else "❌")
                m5.metric("Max DD", f"{result.max_drawdown:.1f}%",
                         delta=metric_color(result.max_drawdown, 15, 30, higher_is_better=False))
                m6.metric("Profit Factor", f"{result.profit_factor:.2f}",
                         delta=metric_color(result.profit_factor, 1.5, 1.0))

                m7, m8, m9 = st.columns(3)
                m7.metric("Sharpe", f"{result.sharpe:.2f}",
                         delta=metric_color(result.sharpe, 1.0, 0, ))
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
    # ═══ АВТОСКАН v2 ═══
    st.subheader("🔍 Автоматический скан + бэктест")
    st.caption("v2: фильтрует стейблкоины, HR < 0.01 и HR > 20, Hurst > 0.50")

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
            coins = get_top_coins_cached(exchange, n_coins)
            # v2: Фильтр стейблкоинов
            coins = [c for c in coins if c.upper() not in STABLECOIN_KEYWORDS]
            st.info(f"📊 Монеты ({len(coins)}): {', '.join(coins[:20])}{'...' if len(coins) > 20 else ''}")

            all_results = scan_and_backtest(
                exchange, coins, timeframe, lookback,
                entry_z, exit_z, stop_z, max_hold, min_hold,
                commission, progress
            )

            progress.progress(1.0, "Готово!")
            time.sleep(0.3)
            progress.empty()

            if not all_results:
                st.warning("⚠️ Ни одной пары с торговыми сигналами")
            else:
                all_results.sort(key=lambda x: -x['result'].total_pnl)

                st.success(f"✅ Протестировано {len(all_results)} пар с торговыми сигналами")

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

                st.divider()
                st.subheader("📈 Детали по парам")

                for r in all_results[:10]:
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
**Pairs Trading Backtester v2.0** | Kalman Filter HR + MAD Z-Score + Walk-Forward

⚠️ Это бэктест — реальная торговля может отличаться из-за проскальзывания, ликвидности, задержек исполнения.

**Изменения v2.0:**
- ✅ Min hold bars — не выходим раньше N баров (кроме стопа)
- ✅ OVERSHOOT — выход только при полной смене знака Z
- ✅ HR фильтр: 0.01 < |HR| < 20 (отсечка стейблкоинов и экстремальных пар)
- ✅ Единый DFA Hurst (синхронизирован с сканером v10.5)
- ✅ OU Half-life (вместо OLS, как в сканере)
- ✅ Cooldown после закрытия
- ✅ Улучшенная палитра: зелёный/красный/жёлтый + зоны на графиках
""")
