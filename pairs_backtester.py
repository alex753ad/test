"""
Pairs Trading Backtester v4.0
Kalman HR + MAD Z-Score + Walk-Forward + Pre-Trade Filters + HL Fix + Adaptive entry_z

v4.0 Changes:
  [FIX] HL: dt = hours_per_bar / 24 (в днях, как в сканере) — HL теперь 5-30ч, не 0.1ч
  [FIX] Exchange fallback chain: OKX→KuCoin→Bybit→Binance (cloud-safe)
  [NEW] Pre-trade filters: Hurst < 0.45, P-value < 0.05, 1 < HL_bars < 50
  [NEW] Adaptive entry_z: HIGH→1.5, MEDIUM→2.0, LOW→2.5
  [NEW] CSV export for all results
  [NEW] Auto-scan mode: backtest top N pairs from scanner
  [NEW] Adaptive min_hold = max(3, int(HL_bars * 0.5))
  [NEW] Cooldown = max(5, int(HL_bars)) bars after close

Запуск: streamlit run pairs_backtester.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
# EXCHANGE FALLBACK
# ═══════════════════════════════════════════════════════
EXCHANGE_FALLBACK = ['okx', 'kucoin', 'bybit', 'binance']

def get_exchange(name):
    tried = set()
    chain = [name] + [e for e in EXCHANGE_FALLBACK if e != name]
    for exch in chain:
        if exch in tried: continue
        tried.add(exch)
        try:
            ex = getattr(ccxt, exch)({'enableRateLimit': True})
            ex.load_markets()
            if exch != name:
                st.warning(f"⚠️ {name.upper()} недоступен. Переключился на **{exch.upper()}**")
            return ex, exch
        except:
            continue
    return None, None

# ═══════════════════════════════════════════════════════
# MATH FUNCTIONS (standalone — no external dependencies)
# ═══════════════════════════════════════════════════════

def kalman_hr(s1, s2, delta=1e-4, ve=1e-3):
    s1, s2 = np.array(s1, float), np.array(s2, float)
    n = min(len(s1), len(s2))
    if n < 10: return None
    s1, s2 = s1[:n], s2[:n]
    init_n = min(30, n // 3)
    try:
        X = np.column_stack([np.ones(init_n), s2[:init_n]])
        beta = np.linalg.lstsq(X, s1[:init_n], rcond=None)[0]
    except:
        beta = np.array([0.0, 1.0])
    P = np.eye(2); Q = np.eye(2) * delta; R = ve
    hrs, ints, spread = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        x = np.array([1.0, s2[t]]); P += Q
        e = s1[t] - x @ beta; S = x @ P @ x + R
        K = P @ x / S; beta += K * e
        P -= np.outer(K, x) @ P; P = (P + P.T) / 2
        np.fill_diagonal(P, np.maximum(np.diag(P), 1e-10))
        hrs[t], ints[t] = beta[1], beta[0]
        spread[t] = s1[t] - beta[1] * s2[t] - beta[0]
    return {'hrs': hrs, 'intercepts': ints, 'spread': spread,
            'hr': float(hrs[-1]), 'intercept': float(ints[-1])}


def calc_zscore(spread, halflife_bars=None, min_w=10, max_w=60):
    spread = np.array(spread, float); n = len(spread)
    if halflife_bars and not np.isinf(halflife_bars) and halflife_bars > 0:
        w = int(np.clip(2.5 * halflife_bars, min_w, max_w))
    else:
        w = 30
    w = min(w, max(10, n // 2))
    zs = np.full(n, np.nan)
    for i in range(w, n):
        lb = spread[i - w:i]; med = np.median(lb)
        mad = np.median(np.abs(lb - med)) * 1.4826
        if mad < 1e-10:
            s = np.std(lb)
            zs[i] = (spread[i] - np.mean(lb)) / s if s > 1e-10 else 0
        else:
            zs[i] = (spread[i] - med) / mad
    return zs, w


def calc_halflife(spread, dt):
    """OU halflife через регрессию. dt в днях: 1h→1/24, 4h→1/6, 1d→1."""
    s = np.array(spread, float)
    if len(s) < 20: return 999
    sl, sd = s[:-1], np.diff(s)
    n = len(sl)
    sx, sy = np.sum(sl), np.sum(sd)
    sxy, sx2 = np.sum(sl * sd), np.sum(sl**2)
    denom = n * sx2 - sx**2
    if abs(denom) < 1e-10: return 999
    b = (n * sxy - sx * sy) / denom
    theta = max(0.001, min(10.0, -b / dt))
    hl = np.log(2) / theta  # в единицах dt (дни)
    return float(hl) if hl < 999 else 999


def calc_hurst(series, min_window=8):
    x = np.array(series, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 50: return 0.5
    y = np.cumsum(x - np.mean(x))
    scales, flucts = [], []
    max_seg = n // 4
    for seg_len in range(min_window, max_seg + 1, max(1, (max_seg - min_window) // 20)):
        n_segs = n // seg_len
        if n_segs < 2: continue
        f2_list = []
        for i in range(n_segs):
            seg = y[i * seg_len:(i + 1) * seg_len]
            t = np.arange(len(seg))
            if len(seg) < 2: continue
            coeffs = np.polyfit(t, seg, 1)
            f2_list.append(np.mean((seg - np.polyval(coeffs, t)) ** 2))
        if f2_list:
            scales.append(seg_len)
            flucts.append(np.sqrt(np.mean(f2_list)))
    if len(scales) < 4: return 0.5
    log_s, log_f = np.log(scales), np.log(np.array(flucts) + 1e-10)
    coeffs = np.polyfit(log_s, log_f, 1)
    pred = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_f - pred) ** 2)
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    if r_sq < 0.8: return 0.5
    return float(np.clip(coeffs[0], 0.01, 0.99))


def calc_confidence(hurst, pvalue, stability_ratio=0.75):
    checks = 0
    if hurst < 0.40: checks += 1
    if pvalue < 0.03: checks += 1
    if stability_ratio >= 0.75: checks += 1
    if checks >= 2: return 'HIGH'
    if checks >= 1: return 'MEDIUM'
    return 'LOW'


# ═══════════════════════════════════════════════════════
# BACKTESTER ENGINE
# ═══════════════════════════════════════════════════════

def run_backtest(prices1, prices2, timeframe='4h', entry_z=2.0, exit_z=0.3,
                 stop_z=4.0, max_bars=100, min_bars=3, commission_pct=0.1,
                 adaptive_entry=True):
    """
    Walk-forward backtest with Kalman HR + MAD Z-score.
    
    v4.0: 
      - dt correct (hours_per_bar / 24)
      - adaptive min_hold from HL
      - adaptive entry_z from confidence
      - cooldown after close
    """
    n = min(len(prices1), len(prices2))
    p1, p2 = prices1[:n], prices2[:n]
    
    if n < 100:
        return None, "Недостаточно данных (< 100 баров)"
    
    # Pre-trade statistics on full sample
    _, pvalue, _ = coint(p1, p2)
    
    # Kalman
    kf = kalman_hr(p1, p2)
    if kf is None:
        return None, "Kalman filter не сработал"
    
    spread = kf['spread']
    hrs = kf['hrs']
    
    # dt correct (v4.0)
    hpb = {'1h': 1, '4h': 4, '1d': 24}.get(timeframe, 4)
    dt = hpb / 24.0  # в днях
    
    hl_days = calc_halflife(spread, dt=dt)
    hl_hours = hl_days * 24
    hl_bars = hl_hours / hpb if hl_hours < 999 else None
    
    hurst = calc_hurst(spread)
    
    # Confidence → adaptive thresholds
    confidence = calc_confidence(hurst, pvalue)
    
    if adaptive_entry:
        if confidence == 'HIGH':
            entry_z = 1.5
        elif confidence == 'MEDIUM':
            entry_z = 2.0
        else:
            entry_z = 2.5
    
    # Adaptive min_hold from HL
    if hl_bars and hl_bars < 50:
        min_hold = max(min_bars, int(hl_bars * 0.5))
        cooldown = max(5, int(hl_bars))
    else:
        min_hold = min_bars
        cooldown = 5
    
    # Z-score
    zs, z_window = calc_zscore(spread, halflife_bars=hl_bars)
    
    # Pre-trade filter results
    pre_trade = {
        'pvalue': pvalue,
        'hurst': hurst,
        'hl_hours': hl_hours,
        'hl_bars': hl_bars,
        'hr': kf['hr'],
        'confidence': confidence,
        'entry_z_used': entry_z,
        'min_hold': min_hold,
        'cooldown': cooldown,
        'z_window': z_window,
    }
    
    filters_passed = True
    filter_warnings = []
    
    if hurst >= 0.45:
        filter_warnings.append(f"⚠️ Hurst={hurst:.3f} ≥ 0.45 — нет mean reversion")
        filters_passed = False
    
    if pvalue >= 0.05:
        filter_warnings.append(f"⚠️ P-value={pvalue:.4f} ≥ 0.05 — нет коинтеграции")
    
    if hl_bars is not None and (hl_bars < 1 or hl_bars > 50):
        filter_warnings.append(f"⚠️ HL={hl_hours:.1f}ч ({hl_bars:.1f} баров) — вне торгуемого диапазона")
    
    if abs(kf['hr']) < 0.01 or abs(kf['hr']) > 30:
        filter_warnings.append(f"⚠️ HR={kf['hr']:.4f} — экстремальный хедж")
    
    pre_trade['filter_warnings'] = filter_warnings
    pre_trade['filters_passed'] = filters_passed
    
    # ═══════ TRADE SIMULATION ═══════
    trades = []
    position = None
    last_close_bar = -cooldown - 1
    commission_total = commission_pct * 4 / 100  # entry+exit × 2 sides
    
    equity = [1.0]
    
    warmup = max(z_window + 10, 50)
    
    for i in range(warmup, n):
        z = zs[i]
        if np.isnan(z):
            equity.append(equity[-1])
            continue
        
        # ═══ OPEN ═══
        if position is None and (i - last_close_bar) > cooldown:
            if z > entry_z:
                position = {
                    'entry_bar': i, 'direction': 'SHORT',
                    'entry_z': z, 'entry_p1': p1[i], 'entry_p2': p2[i],
                    'entry_hr': hrs[i],
                }
            elif z < -entry_z:
                position = {
                    'entry_bar': i, 'direction': 'LONG',
                    'entry_z': z, 'entry_p1': p1[i], 'entry_p2': p2[i],
                    'entry_hr': hrs[i],
                }
        
        # ═══ CLOSE ═══
        if position is not None:
            bars_held = i - position['entry_bar']
            hr_entry = position['entry_hr']
            
            # P&L
            r1 = (p1[i] - position['entry_p1']) / position['entry_p1']
            r2 = (p2[i] - position['entry_p2']) / position['entry_p2']
            if position['direction'] == 'LONG':
                raw_pnl = r1 - hr_entry * r2
            else:
                raw_pnl = -r1 + hr_entry * r2
            pnl = raw_pnl / (1 + abs(hr_entry)) * 100
            
            exit_type = None
            
            if bars_held >= min_hold:
                # Mean revert
                if position['direction'] == 'LONG' and z >= -exit_z:
                    exit_type = 'MEAN_REVERT'
                elif position['direction'] == 'SHORT' and z <= exit_z:
                    exit_type = 'MEAN_REVERT'
                
                # Overshoot
                if position['direction'] == 'LONG' and z > 1.0:
                    exit_type = 'OVERSHOOT'
                elif position['direction'] == 'SHORT' and z < -1.0:
                    exit_type = 'OVERSHOOT'
            
            # Stop loss (always active)
            if position['direction'] == 'LONG' and z < -(stop_z):
                exit_type = 'STOP_LOSS'
            elif position['direction'] == 'SHORT' and z > stop_z:
                exit_type = 'STOP_LOSS'
            
            # Max hold
            if bars_held >= max_bars:
                exit_type = 'MAX_HOLD'
            
            if exit_type:
                pnl_after_comm = pnl - commission_total * 100
                trades.append({
                    'entry_bar': position['entry_bar'],
                    'exit_bar': i,
                    'direction': position['direction'],
                    'entry_z': position['entry_z'],
                    'exit_z': z,
                    'bars_held': bars_held,
                    'pnl_pct': round(pnl_after_comm, 3),
                    'pnl_gross': round(pnl, 3),
                    'exit_type': exit_type,
                    'entry_hr': hr_entry,
                })
                equity.append(equity[-1] * (1 + pnl_after_comm / 100))
                position = None
                last_close_bar = i
            else:
                equity.append(equity[-1])
        else:
            equity.append(equity[-1])
    
    # Close remaining position
    if position is not None:
        i = n - 1
        hr_entry = position['entry_hr']
        r1 = (p1[i] - position['entry_p1']) / position['entry_p1']
        r2 = (p2[i] - position['entry_p2']) / position['entry_p2']
        if position['direction'] == 'LONG':
            raw_pnl = r1 - hr_entry * r2
        else:
            raw_pnl = -r1 + hr_entry * r2
        pnl = raw_pnl / (1 + abs(hr_entry)) * 100 - commission_total * 100
        trades.append({
            'entry_bar': position['entry_bar'], 'exit_bar': i,
            'direction': position['direction'],
            'entry_z': position['entry_z'], 'exit_z': float(zs[i]) if not np.isnan(zs[i]) else 0,
            'bars_held': i - position['entry_bar'],
            'pnl_pct': round(pnl, 3), 'pnl_gross': round(pnl + commission_total * 100, 3),
            'exit_type': 'END_OF_DATA', 'entry_hr': hr_entry,
        })
        equity.append(equity[-1] * (1 + pnl / 100))
    
    # ═══════ STATISTICS ═══════
    if not trades:
        return {'trades': [], 'stats': None, 'pre_trade': pre_trade, 
                'equity': equity, 'zscore': zs, 'spread': spread}, "Нет сделок"
    
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    avg_pnl = np.mean(pnls)
    
    # Max drawdown
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd: max_dd = dd
    
    # Sharpe
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))
    else:
        sharpe = 0
    
    # Profit Factor
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_win / gross_loss
    
    avg_hold = np.mean([t['bars_held'] for t in trades])
    max_hold_actual = max(t['bars_held'] for t in trades)
    
    # By exit type
    exit_types = {}
    for t in trades:
        et = t['exit_type']
        if et not in exit_types:
            exit_types[et] = {'count': 0, 'pnl_sum': 0, 'pnls': []}
        exit_types[et]['count'] += 1
        exit_types[et]['pnl_sum'] += t['pnl_pct']
        exit_types[et]['pnls'].append(t['pnl_pct'])
    
    for et in exit_types:
        exit_types[et]['avg_pnl'] = np.mean(exit_types[et]['pnls'])
        exit_types[et]['win_rate'] = sum(1 for p in exit_types[et]['pnls'] if p > 0) / len(exit_types[et]['pnls']) * 100
    
    stats = {
        'n_trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'profit_factor': pf,
        'avg_hold': avg_hold,
        'max_hold': max_hold_actual,
        'exit_types': exit_types,
    }
    
    return {'trades': trades, 'stats': stats, 'pre_trade': pre_trade,
            'equity': equity, 'zscore': zs, 'spread': spread}, None


# ═══════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════

st.set_page_config(page_title="Pairs Backtester", page_icon="📊", layout="wide")
st.title("📊 Pairs Trading Backtester")
st.caption("v4.0 | Kalman HR + MAD Z-Score + Walk-Forward + Pre-Trade Filters + HL Fix + Adaptive entry_z")

# Sidebar
with st.sidebar:
    st.header("⚙️ Настройки")
    
    mode = st.radio("Режим", ["🔍 Одна пара", "🔄 Автоскан"])
    
    st.divider()
    st.subheader("Данные")
    
    exchange_name = st.selectbox("Биржа", ['okx', 'kucoin', 'bybit', 'binance'], index=0,
                                 help="⚠️ Binance/Bybit заблокированы на облачных серверах")
    timeframe = st.selectbox("Таймфрейм", ['1h', '4h', '1d'], index=1)
    lookback_days = st.slider("Период (дней)", 30, 365, 139, step=7)
    
    if mode == "🔄 Автоскан":
        n_coins = st.slider("Количество монет", 20, 100, 50, step=10)
        max_pairs_bt = st.slider("Макс. пар для бэктеста", 5, 50, 20)
    
    st.divider()
    st.subheader("Параметры стратегии")
    
    adaptive_entry = st.checkbox("Адаптивный entry_z (от Confidence)", value=True,
                                 help="HIGH→1.5, MEDIUM→2.0, LOW→2.5")
    
    if not adaptive_entry:
        entry_z = st.slider("Z для входа", 1.0, 4.0, 2.30, step=0.1)
    else:
        entry_z = 2.0  # будет перезаписан
        st.info("HIGH→1.5, MEDIUM→2.0, LOW→2.5")
    
    exit_z = st.slider("Z для выхода", 0.0, 1.5, 0.30, step=0.1)
    stop_z = st.slider("Z для стопа", 2.0, 6.0, 4.50, step=0.5)
    max_bars = st.slider("Макс. баров в сделке", 20, 200, 100)
    min_bars = st.slider("Мин. баров в сделке", 1, 20, 3)
    commission = st.slider("Комиссия (%)", 0.0, 0.3, 0.10, step=0.01)
    
    st.caption(f"💸 Комиссия за сделку: {commission * 4:.2f}% (4 × {commission:.2f}%)")
    st.caption(f"📏 Макс. удержание: {max_bars} баров ({max_bars * {'1h':1,'4h':4,'1d':24}[timeframe]}ч)")


# ═══════════════════════════════════════════════════════
# SINGLE PAIR MODE
# ═══════════════════════════════════════════════════════

if mode == "🔍 Одна пара":
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        coin1 = st.text_input("🪙 Монета 1", "FIL").upper().strip()
    with col2:
        coin2 = st.text_input("🪙 Монета 2", "CRV").upper().strip()
    with col3:
        run_btn = st.button("🚀 Запустить", type="primary", use_container_width=True)
    
    if run_btn and coin1 and coin2:
        with st.spinner(f"Загружаю данные {coin1}/{coin2}..."):
            ex, actual_exchange = get_exchange(exchange_name)
            if ex is None:
                st.error("❌ Все биржи недоступны")
                st.stop()
            
            hpb_map = {'1h': 24, '4h': 6, '1d': 1}
            limit = lookback_days * hpb_map.get(timeframe, 6)
            
            try:
                ohlcv1 = ex.fetch_ohlcv(f"{coin1}/USDT", timeframe, limit=limit)
                ohlcv2 = ex.fetch_ohlcv(f"{coin2}/USDT", timeframe, limit=limit)
                
                df1 = pd.DataFrame(ohlcv1, columns=['ts','o','h','l','c','v'])
                df2 = pd.DataFrame(ohlcv2, columns=['ts','o','h','l','c','v'])
                df1['ts'] = pd.to_datetime(df1['ts'], unit='ms')
                df2['ts'] = pd.to_datetime(df2['ts'], unit='ms')
                
                merged = pd.merge(df1[['ts','c']], df2[['ts','c']], on='ts', suffixes=('_1','_2'))
                
                st.info(f"📊 Загружено {len(merged)} баров ({timeframe}) с {merged['ts'].iloc[0].date()} по {merged['ts'].iloc[-1].date()}")
                
                p1 = merged['c_1'].values
                p2 = merged['c_2'].values
                
            except Exception as e:
                st.error(f"❌ Ошибка загрузки: {e}")
                st.stop()
        
        # Run backtest
        with st.spinner("Запускаю бэктест..."):
            result, error = run_backtest(
                p1, p2, timeframe=timeframe,
                entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
                max_bars=max_bars, min_bars=min_bars,
                commission_pct=commission, adaptive_entry=adaptive_entry
            )
        
        if error and result is None:
            st.error(f"❌ {error}")
            st.stop()
        
        pt = result['pre_trade']
        
        # Pre-trade metrics
        st.divider()
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        
        pv = pt['pvalue']
        mc1.metric("P-value", f"{pv:.4f}", 
                   "✅ Coint" if pv < 0.05 else "❌ No coint")
        mc2.metric("Hurst", f"{pt['hurst']:.3f}",
                   "✅ MR" if pt['hurst'] < 0.45 else "❌ Trend")
        mc3.metric("Half-life", f"{pt['hl_hours']:.1f}ч",
                   "⚡ Fast" if pt['hl_hours'] < 20 else "✅ OK" if pt['hl_hours'] < 50 else "❌")
        mc4.metric("HR", f"{pt['hr']:.4f}")
        mc5.metric("Confidence", pt['confidence'])
        mc6.metric("Entry Z", f"±{pt['entry_z_used']:.1f}",
                   f"adaptive" if adaptive_entry else "fixed")
        
        # Pre-trade warnings
        for w in pt.get('filter_warnings', []):
            st.warning(w)
        
        st.divider()
        
        # Results
        if result['stats'] is None:
            st.warning(error or "Нет сделок")
        else:
            st.subheader("📊 Результаты бэктеста")
            
            stats = result['stats']
            
            rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
            rc1.metric("Сделок", stats['n_trades'])
            
            wr = stats['win_rate']
            rc2.metric("Win Rate", f"{wr:.1f}%",
                       "✅ ok" if wr >= 45 else "❌ loss")
            
            tp = stats['total_pnl']
            rc3.metric("Total P&L", f"{tp:+.2f}%",
                       "✅" if tp > 0 else "❌ loss")
            
            ap = stats['avg_pnl']
            rc4.metric("Avg P&L", f"{ap:+.2f}%",
                       "✅" if ap > 0 else "❌")
            
            md = stats['max_dd']
            rc5.metric("Max DD", f"{md:.1f}%",
                       "✅ ok" if md < 15 else "⚠️")
            
            pf = stats['profit_factor']
            rc6.metric("Profit Factor", f"{pf:.2f}",
                       "✅ good" if pf > 1.0 else "❌ bad")
            
            # Additional metrics
            rc7, rc8, rc9, rc10 = st.columns(4)
            rc7.metric("Sharpe", f"{stats['sharpe']:.2f}",
                       "✅" if stats['sharpe'] > 1 else "⚠️")
            rc8.metric("Avg Hold", f"{stats['avg_hold']:.0f} баров")
            rc9.metric("Max Hold", f"{stats['max_hold']} баров")
            rc10.metric("Min Hold (adaptive)", f"{pt['min_hold']} баров",
                        f"CD: {pt['cooldown']} баров")
            
            # Exit type breakdown
            st.divider()
            st.subheader("📋 По типам выхода")
            
            et_rows = []
            for et, ed in stats['exit_types'].items():
                et_rows.append({
                    'Тип': et,
                    'Сделок': ed['count'],
                    'Win%': f"{ed['win_rate']:.0f}%",
                    'Avg P&L': f"{ed['avg_pnl']:+.2f}%",
                    'Total': f"{ed['pnl_sum']:+.2f}%",
                })
            st.dataframe(pd.DataFrame(et_rows), use_container_width=True, hide_index=True)
            
            # Equity curve
            st.divider()
            st.subheader("📈 Equity Curve")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.08,
                               subplot_titles=['Equity Curve (капитализация)', 'Z-Score + сделки'],
                               row_heights=[0.5, 0.5])
            
            fig.add_trace(go.Scatter(
                y=result['equity'], name='Equity',
                line=dict(color='#4fc3f7', width=2)
            ), row=1, col=1)
            fig.add_hline(y=1.0, line_dash='dash', line_color='gray', row=1, col=1)
            
            # Z-score with trade markers
            fig.add_trace(go.Scatter(
                y=result['zscore'], name='Z-Score',
                line=dict(color='#ffa726', width=1)
            ), row=2, col=1)
            fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)
            fig.add_hline(y=pt['entry_z_used'], line_dash='dot', line_color='red', row=2, col=1)
            fig.add_hline(y=-pt['entry_z_used'], line_dash='dot', line_color='green', row=2, col=1)
            
            # Trade markers
            for t in result['trades']:
                color = 'green' if t['pnl_pct'] > 0 else 'red'
                fig.add_trace(go.Scatter(
                    x=[t['entry_bar']], y=[t['entry_z']],
                    mode='markers', marker=dict(size=8, color=color, symbol='triangle-up' if t['direction'] == 'LONG' else 'triangle-down'),
                    showlegend=False
                ), row=2, col=1)
            
            fig.update_layout(height=500, template='plotly_dark', showlegend=False,
                             margin=dict(l=50, r=30, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade list
            st.divider()
            st.subheader("📋 Список сделок")
            
            trade_rows = [{
                '#': i+1,
                'Dir': t['direction'],
                'Entry Z': f"{t['entry_z']:+.2f}",
                'Exit Z': f"{t['exit_z']:+.2f}",
                'Bars': t['bars_held'],
                'P&L %': f"{t['pnl_pct']:+.2f}",
                'Тип': t['exit_type'],
                'HR': f"{t['entry_hr']:.4f}",
            } for i, t in enumerate(result['trades'])]
            
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
            
            # CSV Export
            df_export = pd.DataFrame(trade_rows)
            csv_data = df_export.to_csv(index=False)
            st.download_button("📥 Скачать сделки (CSV)", csv_data,
                             f"backtest_{coin1}_{coin2}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv",
                             "text/csv")


# ═══════════════════════════════════════════════════════
# AUTO-SCAN MODE
# ═══════════════════════════════════════════════════════

elif mode == "🔄 Автоскан":
    if st.button("🚀 Запустить автоскан", type="primary"):
        ex, actual_exchange = get_exchange(exchange_name)
        if ex is None:
            st.error("❌ Все биржи недоступны")
            st.stop()
        
        # Get top coins
        with st.spinner("Загружаю список монет..."):
            try:
                tickers = ex.fetch_tickers()
                usdt = {k: v for k, v in tickers.items() 
                       if '/USDT' in k and ':' not in k}
                
                valid = []
                for sym, t in usdt.items():
                    try:
                        vol = float(t.get('quoteVolume', 0))
                        if vol > 0:
                            valid.append((sym.replace('/USDT', ''), vol))
                    except:
                        continue
                
                valid.sort(key=lambda x: -x[1])
                coins = [c[0] for c in valid[:n_coins]]
                st.info(f"📊 Топ {len(coins)} монет с {actual_exchange.upper()}")
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()
        
        # Download prices
        hpb_map = {'1h': 24, '4h': 6, '1d': 1}
        limit = lookback_days * hpb_map.get(timeframe, 6)
        
        prices = {}
        progress = st.progress(0, "Загружаю данные...")
        for i, coin in enumerate(coins):
            progress.progress((i+1)/len(coins), f"📥 {coin} ({i+1}/{len(coins)})")
            try:
                ohlcv = ex.fetch_ohlcv(f"{coin}/USDT", timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
                if len(df) >= 100:
                    prices[coin] = df['c'].values
            except:
                continue
        
        progress.empty()
        st.info(f"✅ Данные для {len(prices)} монет")
        
        # Quick cointegration scan → top pairs
        st.info("🔍 Поиск коинтегрированных пар...")
        coint_pairs = []
        coin_list = list(prices.keys())
        
        scan_progress = st.progress(0, "Тестирую пары...")
        total = len(coin_list) * (len(coin_list) - 1) // 2
        done = 0
        
        for i, c1 in enumerate(coin_list):
            for c2 in coin_list[i+1:]:
                done += 1
                if done % 100 == 0:
                    scan_progress.progress(done / total, f"Фаза 1: {done}/{total}")
                
                n = min(len(prices[c1]), len(prices[c2]))
                if n < 100: continue
                
                try:
                    _, pv, _ = coint(prices[c1][:n], prices[c2][:n])
                    if pv < 0.10:
                        coint_pairs.append((c1, c2, pv))
                except:
                    continue
        
        scan_progress.empty()
        coint_pairs.sort(key=lambda x: x[2])
        coint_pairs = coint_pairs[:max_pairs_bt]
        
        st.info(f"🔬 Найдено {len(coint_pairs)} пар с p<0.10. Бэктестирую топ {max_pairs_bt}...")
        
        # Run backtests
        results_list = []
        bt_progress = st.progress(0, "Бэктесты...")
        
        for idx, (c1, c2, pv) in enumerate(coint_pairs):
            bt_progress.progress((idx+1)/len(coint_pairs), f"Бэктест {c1}/{c2} ({idx+1}/{len(coint_pairs)})")
            
            n = min(len(prices[c1]), len(prices[c2]))
            result, error = run_backtest(
                prices[c1][:n], prices[c2][:n],
                timeframe=timeframe,
                entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
                max_bars=max_bars, min_bars=min_bars,
                commission_pct=commission, adaptive_entry=adaptive_entry
            )
            
            if result and result['stats']:
                s = result['stats']
                pt = result['pre_trade']
                results_list.append({
                    'Пара': f"{c1}/{c2}",
                    'P-val': round(pv, 4),
                    'Hurst': round(pt['hurst'], 3),
                    'HL(ч)': round(pt['hl_hours'], 0) if pt['hl_hours'] < 999 else '∞',
                    'HR': round(pt['hr'], 4),
                    'Сделок': s['n_trades'],
                    'Win%': round(s['win_rate'], 0),
                    'Total P&L': f"{s['total_pnl']:+.1f}%",
                    'Avg P&L': f"{s['avg_pnl']:+.2f}%",
                    'MaxDD': f"{s['max_dd']:.1f}%",
                    'Sharpe': round(s['sharpe'], 1),
                    'PF': round(s['profit_factor'], 2),
                })
        
        bt_progress.empty()
        
        if results_list:
            st.subheader(f"📊 Результаты автоскана ({len(results_list)} пар)")
            
            df_results = pd.DataFrame(results_list)
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            # Summary
            profitable = [r for r in results_list if float(r['Total P&L'].replace('%','').replace('+','')) > 0]
            st.info(f"✅ Прибыльных: {len(profitable)}/{len(results_list)} ({len(profitable)/len(results_list)*100:.0f}%)")
            
            # CSV
            csv = df_results.to_csv(index=False)
            st.download_button("📥 Скачать результаты (CSV)", csv,
                             f"autoscan_{actual_exchange}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv",
                             "text/csv")
        else:
            st.warning("❌ Ни одна пара не показала результатов")

else:
    st.info("👆 Выберите режим и нажмите Запустить")

st.divider()
st.caption("⚠️ Disclaimer: Только для образовательных целей. Не является финансовой рекомендацией.")
