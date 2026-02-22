"""
Pairs Trading Backtester v7.0
v7.0 Changes:
  [DRY] Import math from mean_reversion_analysis.py (fallback: local copies)
  [NEW] P-value gate: p_adj < 0.10 обязательный фильтр (FDR correction)
  [NEW] HR minimum gate: |HR| < 0.01 → фильтр (PEPE/ONDO HR≈0 bug)
  [NEW] Slippage model: 0.05% per leg beyond commission
  [FIX] Kalman init: Ridge regularization (handles price spikes)
  [KEEP] Hurst DFA on increments, ADX regime, trailing stop, walk-forward

Запуск: streamlit run pairs_backtester.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta, timezone

MSK = timezone(timedelta(hours=3))
def now_msk():
    return datetime.now(MSK)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
# DRY: Import from analysis module (fallback: local copies)
# ═══════════════════════════════════════════════════════
try:
    from mean_reversion_analysis import (
        calculate_hurst_exponent as _mra_hurst,
        kalman_hedge_ratio as _mra_kalman,
        calculate_adaptive_robust_zscore as _mra_zscore,
        calculate_confidence as _mra_confidence,
        detect_spread_regime as _mra_regime,
        calc_halflife_from_spread as _mra_halflife,
        cusum_structural_break as _mra_cusum,
        johansen_test as _mra_johansen,
    )
    _USE_MRA = True
except ImportError:
    _USE_MRA = False

# ═══════════════════════════════════════════════════════
# EXCHANGE FALLBACK
# ═══════════════════════════════════════════════════════
EXCHANGE_FALLBACK = ['okx', 'kucoin', 'bybit', 'binance']
EXCLUDE_COINS = {'USDC', 'USDT', 'USDG', 'DAI', 'TUSD', 'BUSD', 'FDUSD',
                 'STETH', 'WSTETH', 'WETH', 'WBTC', 'CBETH', 'RETH',
                 'OKSOL', 'JITOSOL', 'MSOL', 'BNSOL', 'BETH'}

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
                st.warning(f"⚠️ {name.upper()} недоступен → **{exch.upper()}**")
            return ex, exch
        except Exception:
            continue
    return None, None

# ═══════════════════════════════════════════════════════
# MATH — DRY: use analysis module if available, else local
# ═══════════════════════════════════════════════════════

def kalman_hr(s1, s2, delta=1e-4, ve=1e-3):
    """Kalman filter HR. v7.0: Ridge regularization for init window."""
    if _USE_MRA:
        result = _mra_kalman(s1, s2, delta=delta, ve=ve)
        if result is not None:
            return result
    # Local fallback
    s1, s2 = np.array(s1, float), np.array(s2, float)
    n = min(len(s1), len(s2))
    if n < 10: return None
    s1, s2 = s1[:n], s2[:n]
    init_n = min(30, n // 3)
    try:
        # v7.0: Ridge regularization (λ=0.01) prevents spike sensitivity
        X = np.column_stack([np.ones(init_n), s2[:init_n]])
        XtX = X.T @ X + 0.01 * np.eye(2)  # Ridge
        beta = np.linalg.solve(XtX, X.T @ s1[:init_n])
    except Exception:
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
    hl = np.log(2) / theta
    return float(hl) if hl < 999 else 999


def calc_hurst(spread, min_window=8):
    """DFA Hurst — delegates to mean_reversion_analysis when available."""
    if _USE_MRA:
        return _mra_hurst(spread, min_window=min_window)
    # Fallback: local DFA (identical logic)
    ts = np.array(spread, dtype=float)
    n = len(ts)
    if n < 30:
        return 0.5

    # v6.0: DFA на ИНКРЕМЕНТАХ (как в сканере!)
    increments = np.diff(ts)
    n_inc = len(increments)
    profile = np.cumsum(increments - np.mean(increments))

    max_window = n_inc // 4
    if max_window <= min_window:
        return 0.5

    num_points = min(20, max_window - min_window)
    if num_points < 4:
        return 0.5

    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=num_points).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= min_window]
    if len(window_sizes) < 4:
        return 0.5

    fluctuations = []
    for w in window_sizes:
        n_seg = n_inc // w
        if n_seg < 2:
            continue
        f2_sum, count = 0.0, 0
        # Forward segments
        for seg in range(n_seg):
            segment = profile[seg * w:(seg + 1) * w]
            x = np.arange(w, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            f2_sum += np.mean((segment - np.polyval(coeffs, x)) ** 2)
            count += 1
        # Backward segments
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
        return 0.5

    log_n = np.log([f[0] for f in fluctuations])
    log_f = np.log([f[1] for f in fluctuations])

    try:
        slope, _, r_value, _, _ = stats.linregress(log_n, log_f)
        if r_value ** 2 < 0.70:
            return 0.5
        return round(max(0.01, min(0.99, slope)), 4)
    except:
        return 0.5


def calc_confidence(hurst, pvalue, stability_ratio=0.75):
    """v11.0: Hurst hard gate — H>=0.45 → max MEDIUM."""
    checks = 0
    if hurst < 0.40: checks += 1
    if pvalue < 0.03: checks += 1
    if stability_ratio >= 0.75: checks += 1
    if checks >= 2 and hurst < 0.45: return 'HIGH'
    if checks >= 1: return 'MEDIUM'
    return 'LOW'


def calc_continuous_threshold(confidence, quality_score, hurst, timeframe='4h'):
    """v11.0: Min threshold 1.5 (не 1.2)."""
    base_map = {'1h': {'HIGH': 1.8, 'MEDIUM': 2.3, 'LOW': 2.8},
                '4h': {'HIGH': 1.5, 'MEDIUM': 2.0, 'LOW': 2.5},
                '1d': {'HIGH': 1.3, 'MEDIUM': 1.8, 'LOW': 2.3}}
    base = base_map.get(timeframe, base_map['4h']).get(confidence, 2.5)
    q_adj = max(0, (quality_score - 50)) / 250.0
    h_adj = 0.50 if hurst >= 0.45 else 0.0 if hurst >= 0.35 else -0.05 if hurst >= 0.20 else -0.10
    return round(max(1.5, min(3.5, base - q_adj + h_adj)), 2)


# ═══════════════════════════════════════════════════════
# BACKTESTER ENGINE v6.0
# ═══════════════════════════════════════════════════════

def run_backtest(prices1, prices2, timeframe='4h', entry_z=2.0, exit_z=0.3,
                 stop_z=4.0, max_bars=100, min_bars=3, commission_pct=0.1,
                 slippage_pct=0.05,
                 adaptive_entry=True, trailing_stop=True,
                 walk_forward=False, wf_train_pct=0.70):
    """Walk-forward backtest with trailing stop and hard pre-filters."""
    n = min(len(prices1), len(prices2))
    p1, p2 = prices1[:n], prices2[:n]
    
    if n < 100:
        return None, "Недостаточно данных (< 100 баров)"
    
    # Walk-forward split
    if walk_forward:
        train_n = int(n * wf_train_pct)
        test_start = train_n
    else:
        train_n = n
        test_start = 0
    
    # Full-sample Kalman
    kf = kalman_hr(p1, p2)
    if kf is None:
        return None, "Kalman filter не сработал"
    
    spread = kf['spread']
    hrs = kf['hrs']
    
    # Pre-trade stats (on train portion)
    train_spread = spread[:train_n]
    try:
        _, pvalue, _ = coint(p1[:train_n], p2[:train_n])
    except:
        pvalue = 1.0
    
    hpb = {'1h': 1, '4h': 4, '1d': 24}.get(timeframe, 4)
    dt = hpb / 24.0
    
    hl_days = calc_halflife(train_spread, dt=dt)
    hl_hours = hl_days * 24
    hl_bars = hl_hours / hpb if hl_hours < 999 else None
    
    # v6.0: FIXED Hurst (DFA on increments — identical to scanner)
    hurst = calc_hurst(train_spread)
    
    # ═══ HARD PRE-FILTERS v7.0 ═══
    hr_final = kf['hr']
    filter_fail = False
    filter_reasons = []
    
    if hurst >= 0.45:
        filter_fail = True
        filter_reasons.append(f"Hurst={hurst:.3f} ≥ 0.45 (нет mean reversion)")
    
    if hr_final <= 0:
        filter_fail = True
        filter_reasons.append(f"HR={hr_final:.4f} ≤ 0 (отрицательный хедж)")
    
    # v7.0: HR minimum gate — HR≈0 means no hedge (PEPE/ONDO bug)
    if 0 < abs(hr_final) < 0.01:
        filter_fail = True
        filter_reasons.append(f"|HR|={abs(hr_final):.6f} < 0.01 (нет хеджа)")
    
    if abs(hr_final) > 30:
        filter_fail = True
        filter_reasons.append(f"|HR|={abs(hr_final):.1f} > 30 (экстремальный)")
    
    # v7.0: P-value gate (FDR-like filter)
    if pvalue > 0.10:
        filter_fail = True
        filter_reasons.append(f"P-value={pvalue:.4f} > 0.10 (нет коинтеграции)")
    elif pvalue > 0.05:
        filter_reasons.append(f"⚠️ P-value={pvalue:.4f} > 0.05 (слабая коинтеграция)")
    
    if hl_bars is not None and (hl_bars < 1 or hl_bars > 80):
        filter_reasons.append(f"HL={hl_hours:.1f}ч ({hl_bars:.0f} bars) — вне диапазона")
    
    # v7.0: Regime detection — DRY (use analysis module if available)
    spread_adx = 0
    try:
        if _USE_MRA and len(train_spread) >= 60:
            regime_info = _mra_regime(train_spread)
            spread_adx = regime_info.get('adx', 0)
        elif len(train_spread) >= 60:
            diff = np.diff(train_spread)
            w = 20
            pos_s = pd.Series(np.maximum(diff, 0)).rolling(w, min_periods=1).mean().values
            neg_s = pd.Series(np.maximum(-diff, 0)).rolling(w, min_periods=1).mean().values
            atr_s = pd.Series(np.abs(diff)).rolling(w, min_periods=1).mean().values
            di_p = pos_s / (atr_s + 1e-10) * 100
            di_m = neg_s / (atr_s + 1e-10) * 100
            dx = np.abs(di_p - di_m) / (di_p + di_m + 1e-10) * 100
            spread_adx = float(pd.Series(dx).rolling(w, min_periods=1).mean().iloc[-1])
        
        if spread_adx > 35:
            filter_fail = True
            filter_reasons.append(f"ADX={spread_adx:.0f} > 35 (сильный тренд спреда)")
        elif spread_adx > 25:
            filter_reasons.append(f"⚠️ ADX={spread_adx:.0f} > 25 (возможный тренд)")
    except Exception:
        spread_adx = 0
    
    # HR magnitude warning
    if abs(hr_final) > 5:
        filter_reasons.append(f"⚠️ |HR|={abs(hr_final):.1f} > 5 (капитальный дисбаланс)")
    
    # v12.0: CUSUM structural break test
    cusum_break = False
    if _USE_MRA and len(train_spread) >= 60:
        try:
            # Approximate current Z for CUSUM amplifier
            approx_z = 0
            if len(train_spread) > 20:
                mu = np.mean(train_spread[-30:])
                sd = np.std(train_spread[-30:])
                if sd > 1e-10:
                    approx_z = (train_spread[-1] - mu) / sd
            cusum_info = _mra_cusum(train_spread, 
                                    min_tail=min(30, len(train_spread) // 5),
                                    zscore=approx_z)
            if cusum_info.get('has_break'):
                filter_fail = True
                cusum_break = True
                filter_reasons.append(
                    f"CUSUM BREAK: {cusum_info.get('risk_level','?')} "
                    f"score={cusum_info['cusum_score']:.1f}σ")
            elif cusum_info.get('risk_level') == 'HIGH':
                filter_reasons.append(
                    f"⚠️ CUSUM HIGH: score={cusum_info['cusum_score']:.1f}σ")
        except Exception:
            pass
    
    # Quality score (simplified)
    q_score = max(0, min(100,
        (25 if pvalue < 0.01 else 12 if pvalue < 0.05 else 0) +
        (20 if hurst < 0.30 else 15 if hurst < 0.40 else 10 if hurst < 0.48 else 0) +
        (15 if pvalue < 0.05 else 0) +  # ADF approx
        (15 if 0.1 <= abs(hr_final) <= 10 else 5 if abs(hr_final) <= 30 else 0) +
        15  # stability placeholder
    ))
    
    confidence = calc_confidence(hurst, pvalue)
    
    if adaptive_entry:
        entry_z = calc_continuous_threshold(confidence, q_score, hurst, timeframe)
    
    adaptive_stop = max(entry_z + 2.0, stop_z)
    
    # Adaptive min_hold
    if hl_bars and hl_bars < 50:
        min_hold = max(min_bars, int(hl_bars * 0.5))
        cooldown = max(5, int(hl_bars))
    else:
        min_hold = min_bars
        cooldown = 5
    
    # Z-score
    zs, z_window = calc_zscore(spread, halflife_bars=hl_bars)
    
    pre_trade = {
        'pvalue': pvalue, 'hurst': hurst,
        'hl_hours': hl_hours, 'hl_bars': hl_bars,
        'hr': hr_final, 'confidence': confidence, 'q_score': q_score,
        'entry_z_used': entry_z, 'stop_z_used': adaptive_stop,
        'min_hold': min_hold, 'cooldown': cooldown, 'z_window': z_window,
        'filter_fail': filter_fail, 'filter_reasons': filter_reasons,
        'walk_forward': walk_forward, 'train_bars': train_n if walk_forward else n,
        'test_bars': n - train_n if walk_forward else n,
    }
    
    if filter_fail:
        return {'trades': [], 'stats': None, 'pre_trade': pre_trade,
                'equity': [1.0], 'zscore': zs, 'spread': spread}, \
               f"Pre-filter fail: {'; '.join(filter_reasons)}"
    
    # ═══════ TRADE SIMULATION ═══════
    trades = []
    position = None
    last_close_bar = -cooldown - 1
    # v7.0: Total cost = commission (4 legs) + slippage (2 entries + 2 exits)
    slippage_total = slippage_pct * 4 / 100 if slippage_pct else 0
    commission_total = commission_pct * 4 / 100 + slippage_total
    
    equity = [1.0]
    warmup = max(z_window + 10, 50)
    sim_start = max(warmup, test_start) if walk_forward else warmup
    
    for i in range(sim_start, n):
        z = zs[i]
        if np.isnan(z):
            equity.append(equity[-1])
            continue
        
        # ═══ OPEN ═══
        if position is None and (i - last_close_bar) > cooldown:
            # v17: PRE-ENTRY GUARD — don't open if Z already past stop level
            # This eliminates Bars=0 STOP_LOSS phantom trades (PnL_gross=0.0)
            if z > entry_z and z < adaptive_stop:
                position = {
                    'entry_bar': i, 'direction': 'SHORT',
                    'entry_z': z, 'entry_p1': p1[i], 'entry_p2': p2[i],
                    'entry_hr': hrs[i], 'best_pnl': 0, 'trailing_active': False,
                }
            elif z < -entry_z and z > -adaptive_stop:
                position = {
                    'entry_bar': i, 'direction': 'LONG',
                    'entry_z': z, 'entry_p1': p1[i], 'entry_p2': p2[i],
                    'entry_hr': hrs[i], 'best_pnl': 0, 'trailing_active': False,
                }
        
        # ═══ CLOSE ═══
        if position is not None:
            bars_held = i - position['entry_bar']
            hr_entry = position['entry_hr']
            
            r1 = (p1[i] - position['entry_p1']) / position['entry_p1']
            r2 = (p2[i] - position['entry_p2']) / position['entry_p2']
            if position['direction'] == 'LONG':
                raw_pnl = r1 - hr_entry * r2
            else:
                raw_pnl = -r1 + hr_entry * r2
            pnl = raw_pnl / (1 + abs(hr_entry)) * 100
            
            # v6.0: Trailing stop
            if trailing_stop and pnl > position['best_pnl']:
                position['best_pnl'] = pnl
            if trailing_stop and position['best_pnl'] >= 1.0:
                position['trailing_active'] = True
            
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
            
            # Trailing stop — if PnL was ≥1% but now drops to 0%
            if trailing_stop and position['trailing_active'] and pnl <= 0 and bars_held >= min_hold:
                exit_type = 'TRAILING_STOP'
            
            # Hard stop loss
            if position['direction'] == 'LONG' and z < -adaptive_stop:
                exit_type = 'STOP_LOSS'
            elif position['direction'] == 'SHORT' and z > adaptive_stop:
                exit_type = 'STOP_LOSS'
            
            # Max hold
            if bars_held >= max_bars:
                exit_type = 'MAX_HOLD'
            
            if exit_type:
                pnl_after_comm = pnl - commission_total * 100
                trades.append({
                    'entry_bar': position['entry_bar'], 'exit_bar': i,
                    'direction': position['direction'],
                    'entry_z': position['entry_z'], 'exit_z': z,
                    'bars_held': bars_held,
                    'pnl_pct': round(pnl_after_comm, 3),
                    'pnl_gross': round(pnl, 3),
                    'exit_type': exit_type, 'entry_hr': hr_entry,
                    'best_pnl': round(position['best_pnl'], 3),
                })
                equity.append(equity[-1] * (1 + pnl_after_comm / 100))
                position = None
                last_close_bar = i
            else:
                equity.append(equity[-1])
        else:
            equity.append(equity[-1])
    
    # Close remaining
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
            'best_pnl': round(position.get('best_pnl', 0), 3),
        })
        equity.append(equity[-1] * (1 + pnl / 100))
    
    if not trades:
        return {'trades': [], 'stats': None, 'pre_trade': pre_trade,
                'equity': equity, 'zscore': zs, 'spread': spread}, "Нет сделок"
    
    # Statistics
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    peak = equity[0]; max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd: max_dd = dd
    
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    
    exit_types = {}
    for t in trades:
        et = t['exit_type']
        exit_types.setdefault(et, {'count': 0, 'pnl_sum': 0, 'pnls': []})
        exit_types[et]['count'] += 1
        exit_types[et]['pnl_sum'] += t['pnl_pct']
        exit_types[et]['pnls'].append(t['pnl_pct'])
    for et in exit_types:
        exit_types[et]['avg_pnl'] = np.mean(exit_types[et]['pnls'])
        exit_types[et]['win_rate'] = sum(1 for p in exit_types[et]['pnls'] if p > 0) / len(exit_types[et]['pnls']) * 100
    
    stats_dict = {
        'n_trades': len(trades),
        'win_rate': len(wins) / len(pnls) * 100,
        'total_pnl': sum(pnls),
        'avg_pnl': np.mean(pnls),
        'max_dd': max_dd,
        'sharpe': sharpe,
        'profit_factor': (sum(wins) if wins else 0) / gross_loss,
        'avg_hold': np.mean([t['bars_held'] for t in trades]),
        'max_hold': max(t['bars_held'] for t in trades),
        'exit_types': exit_types,
    }
    
    return {'trades': trades, 'stats': stats_dict, 'pre_trade': pre_trade,
            'equity': equity, 'zscore': zs, 'spread': spread}, None


# ═══════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════

st.set_page_config(page_title="Pairs Backtester", page_icon="📊", layout="wide")
st.title("📊 Pairs Trading Backtester")
st.caption("v11.0 | 21.02.2026 | Cost-aware threshold + Johansen + Moscow time")

with st.sidebar:
    st.header("⚙️ Настройки")
    mode = st.radio("Режим", ["🔍 Одна пара", "🔄 Автоскан"])
    
    st.divider()
    st.subheader("Данные")
    exchange_name = st.selectbox("Биржа", ['okx', 'kucoin', 'bybit', 'binance'], index=0,
                                 help="⚠️ Binance/Bybit недоступны на облаке")
    timeframe = st.selectbox("Таймфрейм", ['1h', '4h', '1d'], index=1)
    lookback_days = st.slider("Период (дней)", 30, 365, 139, step=7)
    
    if mode == "🔄 Автоскан":
        n_coins = st.slider("Монет", 20, 100, 50, step=10)
        max_pairs_bt = st.slider("Макс. пар", 5, 50, 20)
    
    st.divider()
    st.subheader("Стратегия")
    adaptive_entry = st.checkbox("Адаптивный threshold", value=True)
    if not adaptive_entry:
        entry_z = st.slider("Z входа", 1.5, 4.0, 2.0, step=0.1)
    else:
        entry_z = 2.0
    
    exit_z = st.slider("Z выхода", 0.0, 1.5, 0.30, step=0.1)
    stop_z = st.slider("Z стопа (base)", 3.0, 6.0, 4.0, step=0.5)
    max_bars = st.slider("Макс. баров", 20, 200, 100)
    commission = st.slider("Комиссия (%)", 0.0, 0.3, 0.10, step=0.01,
                           help="Обычно 0.08-0.10% для мейкеров")
    slippage = st.slider("Проскальзывание (%)", 0.0, 0.2, 0.05, step=0.01,
                          help="v7.0: Bid/Ask спред + market impact. 0.05% для ликвидных пар")
    trailing_stop = st.checkbox("🔄 Trailing Stop", value=True,
                                help="При PnL≥1% — стоп сдвигается. Если PnL упал до 0% → выход.")
    walk_forward = st.checkbox("📊 Walk-Forward", value=False,
                               help="70% train / 30% test — показывает реальную производительность")

# ═══════════════════════════════════════════════════════
# SINGLE PAIR
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
        with st.spinner(f"Загружаю {coin1}/{coin2}..."):
            ex, actual = get_exchange(exchange_name)
            if ex is None:
                st.error("❌ Биржи недоступны"); st.stop()
            
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
                p1, p2 = merged['c_1'].values, merged['c_2'].values
                st.info(f"📊 {len(merged)} баров ({timeframe}), {merged['ts'].iloc[0].date()} — {merged['ts'].iloc[-1].date()}")
            except Exception as e:
                st.error(f"❌ {e}"); st.stop()
        
        with st.spinner("Бэктест..."):
            result, error = run_backtest(p1, p2, timeframe=timeframe,
                entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
                max_bars=max_bars, commission_pct=commission, slippage_pct=slippage,
                adaptive_entry=adaptive_entry, trailing_stop=trailing_stop,
                walk_forward=walk_forward)
        
        if error and result is None:
            st.error(f"❌ {error}"); st.stop()
        
        pt = result['pre_trade']
        
        # Pre-trade panel
        pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
        pm1.metric("P-value", f"{pt['pvalue']:.4f}", "✅" if pt['pvalue'] < 0.05 else "❌")
        pm2.metric("Hurst", f"{pt['hurst']:.3f}", "✅ MR" if pt['hurst'] < 0.35 else "⚠️" if pt['hurst'] < 0.45 else "❌")
        pm3.metric("HL", f"{pt['hl_hours']:.0f}ч")
        pm4.metric("HR", f"{pt['hr']:.4f}")
        pm5.metric("Conf", pt['confidence'])
        pm6.metric("Entry Z", f"±{pt['entry_z_used']}")
        
        if pt.get('filter_fail'):
            for r in pt['filter_reasons']:
                st.error(f"🚫 {r}")
            st.warning("Пара не прошла pre-trade фильтры. Торговля не рекомендуется.")
        
        if walk_forward:
            st.info(f"📊 Walk-Forward: {pt['train_bars']} баров train / {pt['test_bars']} баров test")
        
        # Results
        if result['stats'] is not None:
            s = result['stats']
            
            st.subheader("📊 Результаты")
            rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
            rc1.metric("Сделок", s['n_trades'])
            rc2.metric("Win%", f"{s['win_rate']:.0f}%", "✅" if s['win_rate'] >= 45 else "❌")
            rc3.metric("Total P&L", f"{s['total_pnl']:+.1f}%", "✅" if s['total_pnl'] > 0 else "❌")
            rc4.metric("Avg P&L", f"{s['avg_pnl']:+.2f}%")
            rc5.metric("MaxDD", f"{s['max_dd']:.1f}%")
            rc6.metric("PF", f"{s['profit_factor']:.2f}", "✅" if s['profit_factor'] > 1 else "❌")
            
            # Exit types
            with st.expander("📋 По типам выхода"):
                for et, ed in s['exit_types'].items():
                    st.markdown(f"**{et}**: {ed['count']} сделок, WR={ed['win_rate']:.0f}%, Avg={ed['avg_pnl']:+.2f}%")
            
            # Equity chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                               subplot_titles=['Equity Curve', 'Z-Score + trades'], row_heights=[0.5, 0.5])
            fig.add_trace(go.Scatter(y=result['equity'], name='Equity', line=dict(color='#4fc3f7', width=2)), row=1, col=1)
            fig.add_hline(y=1.0, line_dash='dash', line_color='gray', row=1, col=1)
            fig.add_trace(go.Scatter(y=result['zscore'], name='Z', line=dict(color='#ffa726', width=1)), row=2, col=1)
            fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)
            fig.add_hline(y=pt['entry_z_used'], line_dash='dot', line_color='red', row=2, col=1)
            fig.add_hline(y=-pt['entry_z_used'], line_dash='dot', line_color='green', row=2, col=1)
            for t in result['trades']:
                color = 'green' if t['pnl_pct'] > 0 else 'red'
                sym = 'triangle-up' if t['direction'] == 'LONG' else 'triangle-down'
                fig.add_trace(go.Scatter(x=[t['entry_bar']], y=[t['entry_z']], mode='markers',
                    marker=dict(size=8, color=color, symbol=sym), showlegend=False), row=2, col=1)
            if walk_forward:
                fig.add_vline(x=pt['train_bars'], line_dash='dash', line_color='yellow',
                             annotation_text='Train|Test', row=2, col=1)
            fig.update_layout(height=500, template='plotly_dark', showlegend=False, margin=dict(l=50,r=30,t=40,b=30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Trades table
            with st.expander("📋 Список сделок"):
                rows = [{'#': i+1, 'Dir': t['direction'], 'Entry Z': f"{t['entry_z']:+.2f}",
                         'Exit Z': f"{t['exit_z']:+.2f}", 'Bars': t['bars_held'],
                         'P&L': f"{t['pnl_pct']:+.2f}%", 'Best': f"{t['best_pnl']:+.1f}%",
                         'Тип': t['exit_type']}
                        for i, t in enumerate(result['trades'])]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            
            # CSV — trades + pre-trade summary
            export_trades = []
            for i, t in enumerate(result['trades']):
                export_trades.append({
                    'Pair': f"{coin1}/{coin2}", 'TF': timeframe,
                    '#': i+1, 'Direction': t['direction'],
                    'Entry_Z': t['entry_z'], 'Exit_Z': t['exit_z'],
                    'Bars': t['bars_held'], 'PnL%': t['pnl_pct'],
                    'PnL_gross%': t['pnl_gross'], 'Best_PnL%': t['best_pnl'],
                    'Exit_Type': t['exit_type'],
                })
            df_trades = pd.DataFrame(export_trades)
            
            # Pre-trade summary row
            summary = pd.DataFrame([{
                'Pair': f"{coin1}/{coin2}", 'TF': timeframe,
                '#': 'SUMMARY', 'Direction': f"Trades={s['n_trades']}",
                'Entry_Z': f"Hurst={pt['hurst']:.3f}", 
                'Exit_Z': f"HL={pt['hl_hours']:.0f}h",
                'Bars': f"HR={pt['hr']:.4f}",
                'PnL%': s['total_pnl'],
                'PnL_gross%': f"WR={s['win_rate']:.0f}%",
                'Best_PnL%': f"PF={s['profit_factor']:.2f}",
                'Exit_Type': f"Sharpe={s['sharpe']:.1f}",
            }])
            df_full = pd.concat([summary, df_trades], ignore_index=True)
            
            st.download_button("📥 Скачать результат (CSV)", df_full.to_csv(index=False),
                f"backtest_{coin1}_{coin2}_{timeframe}_{now_msk().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
        elif error:
            st.warning(f"⚠️ {error}")


# ═══════════════════════════════════════════════════════
# AUTO-SCAN
# ═══════════════════════════════════════════════════════

elif mode == "🔄 Автоскан":
    if st.button("🚀 Автоскан", type="primary"):
        ex, actual = get_exchange(exchange_name)
        if ex is None: st.error("❌ Биржи недоступны"); st.stop()
        
        with st.spinner("Монеты..."):
            try:
                tickers = ex.fetch_tickers()
                usdt = {k: v for k, v in tickers.items() if '/USDT' in k and ':' not in k}
                valid = [(sym.replace('/USDT',''), float(t.get('quoteVolume',0)))
                         for sym, t in usdt.items()
                         if float(t.get('quoteVolume',0)) > 0 and sym.replace('/USDT','') not in EXCLUDE_COINS]
                valid.sort(key=lambda x: -x[1])
                coins = [c[0] for c in valid[:n_coins]]
                st.info(f"📊 {len(coins)} монет с {actual.upper()}")
            except Exception as e: st.error(f"❌ {e}"); st.stop()
        
        hpb_map = {'1h': 24, '4h': 6, '1d': 1}
        limit = lookback_days * hpb_map.get(timeframe, 6)
        
        prices = {}
        prog = st.progress(0, "Данные...")
        for i, coin in enumerate(coins):
            prog.progress((i+1)/len(coins), f"📥 {coin}")
            try:
                ohlcv = ex.fetch_ohlcv(f"{coin}/USDT", timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
                if len(df) >= 100: prices[coin] = df['c'].values
            except: continue
        prog.empty()
        
        # Cointegration scan with FDR correction (v7.0)
        raw_pairs = []
        all_pvalues = []
        coin_list = list(prices.keys())
        total = len(coin_list) * (len(coin_list) - 1) // 2
        sprog = st.progress(0, "Коинтеграция...")
        done = 0
        for i, c1 in enumerate(coin_list):
            for c2 in coin_list[i+1:]:
                done += 1
                if done % 100 == 0: sprog.progress(done/total)
                n = min(len(prices[c1]), len(prices[c2]))
                if n < 100: continue
                try:
                    _, pv, _ = coint(prices[c1][:n], prices[c2][:n])
                    all_pvalues.append(pv)
                    if pv < 0.10:
                        raw_pairs.append((c1, c2, pv, len(all_pvalues) - 1))
                except:
                    all_pvalues.append(1.0)
        sprog.empty()
        
        # v7.0: FDR correction (Benjamini-Hochberg)
        coint_pairs = []
        fdr_count = 0
        if all_pvalues:
            pvals = np.array(all_pvalues)
            m = len(pvals)
            sorted_idx = np.argsort(pvals)
            adj = np.ones(m)
            for rank, idx in enumerate(sorted_idx):
                adj[idx] = pvals[idx] * m / (rank + 1)
            for k in range(m-2, -1, -1):
                adj[sorted_idx[k]] = min(adj[sorted_idx[k]], 
                    adj[sorted_idx[k+1]] if k+1 < m else 1.0)
            adj = np.clip(adj, 0, 1)
            
            for c1, c2, pv, pv_idx in raw_pairs:
                p_adj = float(adj[pv_idx])
                if pv < 0.05 or p_adj < 0.10:
                    coint_pairs.append((c1, c2, pv))
                    if p_adj < 0.05:
                        fdr_count += 1
        
        coint_pairs.sort(key=lambda x: x[2])
        coint_pairs = coint_pairs[:max_pairs_bt]
        
        st.info(f"🔬 {len(coint_pairs)} пар (p<0.05 | FDR<0.10) из {len(all_pvalues)}. FDR✅: {fdr_count}")
        
        results_list = []
        bprog = st.progress(0, "Бэктесты...")
        for idx, (c1, c2, pv) in enumerate(coint_pairs):
            bprog.progress((idx+1)/len(coint_pairs), f"{c1}/{c2}")
            n = min(len(prices[c1]), len(prices[c2]))
            result, error = run_backtest(prices[c1][:n], prices[c2][:n],
                timeframe=timeframe, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z,
                max_bars=max_bars, commission_pct=commission, slippage_pct=slippage,
                adaptive_entry=adaptive_entry,
                trailing_stop=trailing_stop, walk_forward=walk_forward)
            
            pt = result['pre_trade'] if result else {}
            if result and result['stats']:
                s = result['stats']
                results_list.append({
                    'Пара': f"{c1}/{c2}", 'P-val': round(pv, 4),
                    'Hurst': round(pt['hurst'], 3), 'HL(ч)': round(pt['hl_hours'], 0),
                    'HR': round(pt['hr'], 4), 'Сделок': s['n_trades'],
                    'Win%': round(s['win_rate'], 0),
                    'Total P&L': f"{s['total_pnl']:+.1f}%",
                    'Avg P&L': f"{s['avg_pnl']:+.2f}%",
                    'MaxDD': f"{s['max_dd']:.1f}%",
                    'Sharpe': round(s['sharpe'], 1),
                    'PF': round(s['profit_factor'], 2),
                })
            elif pt.get('filter_fail'):
                results_list.append({
                    'Пара': f"{c1}/{c2}", 'P-val': round(pv, 4),
                    'Hurst': round(pt.get('hurst', 0.5), 3), 'HL(ч)': round(pt.get('hl_hours', 0), 0),
                    'HR': round(pt.get('hr', 0), 4), 'Сделок': '🚫',
                    'Win%': '-', 'Total P&L': '🚫 FILTERED',
                    'Avg P&L': '-', 'MaxDD': '-', 'Sharpe': '-', 'PF': '-',
                })
        
        bprog.empty()
        
        if results_list:
            st.subheader(f"📊 Результаты ({len(results_list)} пар)")
            df_r = pd.DataFrame(results_list)
            st.dataframe(df_r, use_container_width=True, hide_index=True)
            
            traded = [r for r in results_list if r['Сделок'] != '🚫']
            filtered = [r for r in results_list if r['Сделок'] == '🚫']
            profitable = [r for r in traded if '+' in str(r.get('Total P&L', ''))]
            st.info(f"✅ Торговали: {len(traded)}, Отфильтровано: {len(filtered)}, Прибыльных: {len(profitable)}")
            
            st.download_button("📥 CSV", df_r.to_csv(index=False),
                f"autoscan_{actual}_{timeframe}_{now_msk().strftime('%Y%m%d')}.csv", "text/csv")

st.divider()
st.caption("⚠️ Только для образовательных целей. Не является финансовой рекомендацией.")
