#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
support_v2 — Recompute the common-support (CS) diagnosis with a strictly
comparable 5-NN-to-5-NN method on DISTINCT private firms.

Key changes vs the old 5-NN-to-1-NN method (which compared an SOE 5-NN-model's
*nearest* observation against a private 1-NN *observation* threshold):
  1. Distance is now the SAME statistic for SOE and private calibration:
     the distance to the 5th *distinct private firm* (d^(5)), i.e. for each
     private firm j we keep min_{s in j} ||x - z_{j,s}||, sort those firm
     distances, and take the 5th smallest.
  2. For a private calibration observation, its OWN firm_id is removed ENTIRELY
     (all years) so adjacent years of the same firm cannot deflate the baseline.
  3. Preprocessing (P5/P95 winsorize + median impute + StandardScaler) is fitted
     ONLY on the forecast-year's historical private training pool
     (target_year < ForecastYear), identical to the main GBM pipeline.
  4. P90/P95 thresholds are computed PER ForecastYear from the private d^(5)
     distribution (no fixed full-period threshold).

This module does NOT retrain any model and does NOT modify any firm-year gap.
It reads the cached master SOE panel (outputs/revision3/_master_long_panel.csv,
gaps produced by the rolling ex-ante GBM) and only recomputes the support label.

Writes everything to outputs/support_v2/ (new directory; nothing is overwritten).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ---- imports from the existing audit modules (same pipeline) ----
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'revision3'))
sys.path.insert(0, str(BASE.parent / 'revision2'))

from _audit_common import load_screened_panel, feats_in, prepare_train  # noqa: E402
from _audit3 import compute_long_panel  # noqa: E402

OUT = BASE
OUT.mkdir(exist_ok=True)

YEARS = list(range(2019, 2026))
SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']
SAMPLE_COL = {'CSI300': 'in_csi300', 'CSI500': 'in_csi500', 'CSI1000': 'in_csi1000'}
RNG_SEED = 42
N_BOOT = 2000
CHUNK_PRIV = 800      # rows per chunk for private cdist
CHUNK_SOE = 512       # rows per chunk for SOE cdist


# ============================================================================
# distinct-firm 5-NN helpers
# ============================================================================

def build_reference(tr, feats):
    """Fit the main-model preprocessing on the historical private pool `tr`,
    and return (prep, Z_sorted, firm_starts, firm_counts, n_obs, n_firms).
    Z_sorted is the standardized 9-feature matrix of the EFFECTIVE training
    observations (dropna on feats+EBI_A), sorted by firm_id."""
    prep = prepare_train(tr, feats)
    tv = prep['tv'].copy()
    Z = prep['scl'].transform(prep['imp'].transform(tv[prep['ncw']].values))
    fid = tv['firm_id'].astype(str).values
    order = np.argsort(fid, kind='stable')
    Z_sorted = Z[order]
    fid_sorted = fid[order]
    firm_codes, firm_starts, firm_counts = np.unique(
        fid_sorted, return_index=True, return_counts=True)
    return prep, Z_sorted, firm_starts, firm_counts, int(len(tv)), int(len(firm_codes))


def _firm_min(D, firm_starts):
    """D: (n_q, n_priv); -> (n_q, n_firms) min distance per private firm."""
    return np.minimum.reduceat(D, firm_starts, axis=1)


def _d5(fm, own_firm_idx=None):
    """5th smallest distinct-firm distance (index 4 of ascending sort)."""
    if own_firm_idx is not None:
        fm[np.arange(fm.shape[0]), own_firm_idx] = np.inf
    return np.partition(fm, 4, axis=1)[:, 4]


def soe_d5(Z_soe, Z_priv, firm_starts):
    """d^(5) for SOE queries (no own-firm exclusion; SOE not in private pool)."""
    n = Z_soe.shape[0]
    out = np.empty(n)
    for s in range(0, n, CHUNK_SOE):
        D = cdist(Z_soe[s:s + CHUNK_SOE], Z_priv)
        out[s:s + CHUNK_SOE] = _d5(_firm_min(D, firm_starts))
    return out


def private_d5(Z_priv, firm_starts, firm_counts):
    """d^(5) for private calibration obs (own firm removed entirely)."""
    n = Z_priv.shape[0]
    n_f = len(firm_starts)
    own = np.repeat(np.arange(n_f), firm_counts)   # firm index of each sorted row
    out = np.empty(n)
    for s in range(0, n, CHUNK_PRIV):
        e = min(s + CHUNK_PRIV, n)
        D = cdist(Z_priv[s:e], Z_priv)
        fm = _firm_min(D, firm_starts)
        fm[np.arange(e - s), own[s:e]] = np.inf
        out[s:e] = _d5(fm)
    return out


# ============================================================================
# load data
# ============================================================================

all_a = load_screened_panel()
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()
long = compute_long_panel()                      # cached master panel (gaps fixed)
long['firm_id'] = long['firm_id'].astype(str)

assert list(feats) == ['FirmSize', 'AssetTurnover', 'FinancialLeverage',
                       'FixedAssetsRatio', 'CurrentRatio', 'CapexRatio',
                       'WorkingCapitalRatio', 'MarketShare', 'HHI'], feats

# ============================================================================
# per-year support computation
# ============================================================================

threshold_rows = []
firmyear_parts = []

for fc in YEARS:
    tr = train_pool[train_pool['target_year'] < fc]
    # QA #5/#6: reference pool must be strictly ex-ante (no year-T outcome)
    assert tr['target_year'].max() < fc

    prep, Z_priv, firm_starts, firm_counts, n_obs, n_firms = build_reference(tr, feats)

    # private calibration distance distribution + per-year thresholds
    d5_priv = private_d5(Z_priv, firm_starts, firm_counts)
    p50, p75, p90, p95, p99 = np.quantile(d5_priv, [0.50, 0.75, 0.90, 0.95, 0.99])

    threshold_rows.append({
        'ForecastYear': fc,
        'N_private_observations': n_obs,
        'N_private_firms': n_firms,
        'd5_private_P50': p50, 'd5_private_P75': p75,
        'd5_private_P90': p90, 'd5_private_P95': p95, 'd5_private_P99': p99,
    })

    # SOE queries for this year (from the cached master panel)
    soe = long[long['year'] == fc].copy()
    soe_w = soe[feats].copy()
    for c in feats:
        soe_w[c] = soe_w[c].clip(prep['qlo'][c], prep['qhi'][c])
    Z_soe = prep['scl'].transform(prep['imp'].transform(soe_w.values))
    d5_soe = soe_d5(Z_soe, Z_priv, firm_starts)

    status = np.where(d5_soe <= p90, 'Good',
                      np.where(d5_soe <= p95, 'Boundary', 'Extrapolation'))

    part = pd.DataFrame({
        'firm_id': soe['firm_id'].values,
        'ForecastYear': fc,
        'in_csi300': soe['in_csi300'].values,
        'in_csi500': soe['in_csi500'].values,
        'in_csi1000': soe['in_csi1000'].values,
        'GBM_predicted_EBIA': soe['EBI_A'].values + soe['gap'].values,
        'actual_EBIA': soe['EBI_A'].values,
        'gap_pp': soe['gap'].values * 100.0,
        'total_assets': soe['资产总计_target'].values,
        'amount_gap': soe['amount_gap_yi'].values,
        'd5_distance': d5_soe,
        'P90_year': p90, 'P95_year': p95,
        'support_status': status,
    })
    firmyear_parts.append(part)

threshold_df = pd.DataFrame(threshold_rows)
firmyear_df = pd.concat(firmyear_parts, ignore_index=True)

# ---- CSV 1 ----
threshold_df.to_csv(OUT / 'support_thresholds_by_year_v2.csv', index=False,
                    encoding='utf-8-sig')
# ---- CSV 2 ----
firmyear_df.to_csv(OUT / 'soe_support_firmyear_v2.csv', index=False,
                   encoding='utf-8-sig')

# merge the NEW support back onto the cached master panel (gap untouched)
long_v2 = long.merge(
    firmyear_df[['firm_id', 'ForecastYear', 'd5_distance', 'P90_year', 'P95_year',
                 'support_status']],
    left_on=['firm_id', 'year'], right_on=['firm_id', 'ForecastYear'],
    how='left', validate='one_to_one', suffixes=('', '_v2'))
assert long_v2['support_status'].notna().all()

# ============================================================================
# CSV 3: year x sample summary
# ============================================================================

summary_rows = []
for fc in YEARS:
    for sample in SAMPLES:
        if sample == 'All A-share SOE':
            m = firmyear_df['ForecastYear'] == fc
        else:
            m = (firmyear_df['ForecastYear'] == fc) & firmyear_df[SAMPLE_COL[sample]].astype(bool)
        s = firmyear_df[m]
        n = len(s)
        g = s['support_status'] == 'Good'
        b = s['support_status'] == 'Boundary'
        cs = g | b
        summary_rows.append({
            'ForecastYear': fc, 'sample': sample, 'N_SOE': n,
            'Good_N': int(g.sum()), 'Boundary_N': int(b.sum()),
            'Extrapolation_N': int((~cs).sum()),
            'Good_share': g.mean(), 'Boundary_share': b.mean(),
            'Extrapolation_share': (~cs).mean(), 'CS_share': cs.mean(),
            'Full_gap_pp': s['gap_pp'].mean(),
            'Good_gap_pp': s.loc[g, 'gap_pp'].mean() if g.sum() else np.nan,
            'CS_gap_pp': s.loc[cs, 'gap_pp'].mean() if cs.sum() else np.nan,
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / 'support_year_sample_summary_v2.csv', index=False,
                  encoding='utf-8-sig')

# ============================================================================
# CSV 4: multi-year summary with firm-cluster bootstrap
# ============================================================================

def annual_mean(gap, year, mask=None):
    vals = []
    for y in YEARS:
        mm = (year == y)
        if mask is not None:
            mm = mm & mask
        if mm.any():
            vals.append(gap[mm].mean())
    return float(np.mean(vals)) if vals else np.nan


def cluster_bootstrap(sub, seed=RNG_SEED, n_boot=N_BOOT):
    """sub has columns firm_id, year, gap, support_status.
    Returns dict of point estimates + SE + 95% CI + p (two-sided percentile)."""
    sub = sub.sort_values('firm_id').reset_index(drop=True)
    fid = sub['firm_id'].values
    gap = sub['gap'].values
    year = sub['year'].values
    cs_flag = sub['support_status'].isin(['Good', 'Boundary']).values

    uniq, starts = np.unique(fid, return_index=True)
    counts = np.diff(np.append(starts, len(fid))).astype(int)
    n_firms = len(uniq)

    goodshare = np.array([
        (sub['support_status'].values[starts[i]:starts[i] + counts[i]] == 'Good').mean()
        for i in range(n_firms)])
    csshare = np.array([
        np.isin(sub['support_status'].values[starts[i]:starts[i] + counts[i]],
                ['Good', 'Boundary']).mean()
        for i in range(n_firms)])
    firm_mean_gap = np.array([gap[starts[i]:starts[i] + counts[i]].mean()
                              for i in range(n_firms)])

    pt_full = annual_mean(gap, year)
    pt_cs = annual_mean(gap, year, cs_flag)
    sel_pers = goodshare >= 0.8
    pt_pers = float(firm_mean_gap[sel_pers].mean()) if sel_pers.any() else np.nan

    rng = np.random.default_rng(seed)
    boot_full = np.full(n_boot, np.nan)
    boot_cs = np.full(n_boot, np.nan)
    boot_pers = np.full(n_boot, np.nan)

    for b in range(n_boot):
        codes = rng.integers(0, n_firms, size=n_firms)
        lens = counts[codes]
        block_start = np.repeat(np.cumsum(lens) - lens, lens)
        within = np.arange(lens.sum()) - block_start
        gidx = np.repeat(starts[codes], lens) + within

        g_b = gap[gidx]; y_b = year[gidx]; c_b = cs_flag[gidx]
        boot_full[b] = annual_mean(g_b, y_b)
        boot_cs[b] = annual_mean(g_b, y_b, c_b)

        sel = codes[goodshare[codes] >= 0.8]
        boot_pers[b] = firm_mean_gap[sel].mean() if len(sel) else np.nan

    def summ(b):
        b = b[~np.isnan(b)]
        se = float(np.std(b, ddof=1)) if len(b) > 1 else np.nan
        lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan, np.nan)
        p = min(1.0, 2.0 * min(float(np.mean(b <= 0)), float(np.mean(b >= 0))))
        return se, lo, hi, p

    s_full = summ(boot_full)
    s_cs = summ(boot_cs)
    s_pers = summ(boot_pers)

    return {
        'Full_gap_pp': pt_full * 100, 'Full_SE': s_full[0] * 100,
        'Full_CI_low': s_full[1] * 100, 'Full_CI_high': s_full[2] * 100, 'Full_p': s_full[3],
        'CS_gap_pp': pt_cs * 100, 'CS_SE': s_cs[0] * 100,
        'CS_CI_low': s_cs[1] * 100, 'CS_CI_high': s_cs[2] * 100, 'CS_p': s_cs[3],
        'PersistentGood_gap_pp': pt_pers * 100 if not np.isnan(pt_pers) else np.nan,
        'PersistentGood_SE': s_pers[0] * 100,
        'PersistentGood_CI_low': s_pers[1] * 100, 'PersistentGood_CI_high': s_pers[2] * 100,
        'PersistentGood_p': s_pers[3],
        'Mean_GoodShare': float(np.mean(goodshare)),
        'Mean_CSShare': float(np.mean(csshare)),
        'N_firms': n_firms,
        'N_PersistentGood_firms': int(sel_pers.sum()),
    }


multiyear_rows = []
for sample in SAMPLES:
    if sample == 'All A-share SOE':
        sub = long_v2.copy()
    else:
        sub = long_v2[long_v2[SAMPLE_COL[sample]].astype(bool)].copy()
    r = cluster_bootstrap(sub)
    r['sample'] = sample
    multiyear_rows.append(r)

multiyear_df = pd.DataFrame(multiyear_rows)
col_order = ['sample', 'Full_gap_pp', 'Full_CI_low', 'Full_CI_high',
             'CS_gap_pp', 'CS_CI_low', 'CS_CI_high',
             'PersistentGood_gap_pp', 'PersistentGood_CI_low', 'PersistentGood_CI_high',
             'Mean_GoodShare', 'Mean_CSShare', 'N_firms', 'N_PersistentGood_firms']
multiyear_df = multiyear_df[col_order]
multiyear_df.to_csv(OUT / 'support_multiyear_summary_v2.csv', index=False,
                    encoding='utf-8-sig')

# ============================================================================
# CSV 5: 2025 amount-gap decomposition
# ============================================================================

amount_rows = []
for sample in SAMPLES:
    if sample == 'All A-share SOE':
        s = firmyear_df[firmyear_df['ForecastYear'] == 2025]
    else:
        s = firmyear_df[(firmyear_df['ForecastYear'] == 2025)
                        & firmyear_df[SAMPLE_COL[sample]].astype(bool)]
    g = s[s['support_status'] == 'Good']
    b = s[s['support_status'] == 'Boundary']
    e = s[s['support_status'] == 'Extrapolation']
    cs = s[s['support_status'].isin(['Good', 'Boundary'])]
    total_amt = s['amount_gap'].sum()
    amount_rows.append({
        'sample': sample,
        'N_good': len(g), 'N_boundary': len(b), 'N_extrapolation': len(e),
        'Good_gap_pp': g['gap_pp'].mean() if len(g) else np.nan,
        'Boundary_gap_pp': b['gap_pp'].mean() if len(b) else np.nan,
        'Extrapolation_gap_pp': e['gap_pp'].mean() if len(e) else np.nan,
        'Good_amount_yi': g['amount_gap'].sum(),
        'Boundary_amount_yi': b['amount_gap'].sum(),
        'Extrapolation_amount_yi': e['amount_gap'].sum(),
        'Good_amount_share': g['amount_gap'].sum() / total_amt if total_amt else np.nan,
        'Boundary_amount_share': b['amount_gap'].sum() / total_amt if total_amt else np.nan,
        'Extrapolation_amount_share': e['amount_gap'].sum() / total_amt if total_amt else np.nan,
        'Total_amount_yi': total_amt,
        'CS_amount_yi': cs['amount_gap'].sum(),
        'Extrapolation_amount_yi_2': e['amount_gap'].sum(),
    })
amount_df = pd.DataFrame(amount_rows)
amount_df.to_csv(OUT / 'support_amount_2025_v2.csv', index=False,
                 encoding='utf-8-sig')

# ============================================================================
# CSV 6: transition matrix (same SOE, consecutive forecast years)
# ============================================================================

trans = []
for fid, g in long_v2.groupby('firm_id'):
    g = g.sort_values('year')
    for (y0, s0), (y1, s1) in zip(g[['year', 'support_status']].values,
                                   g[['year', 'support_status']].values[1:]):
        if int(y1) == int(y0) + 1:
            trans.append((s0, s1))
trans_df = pd.DataFrame(trans, columns=['from', 'to'])
statuses = ['Good', 'Boundary', 'Extrapolation']
trans_rows = []
for f in statuses:
    rowsum = (trans_df['from'] == f).sum()
    for t in statuses:
        n = ((trans_df['from'] == f) & (trans_df['to'] == t)).sum()
        trans_rows.append({
            'from': f, 'to': t, 'n': n,
            'row_normalized_prob': n / rowsum if rowsum else np.nan,
        })
transition_df = pd.DataFrame(trans_rows)
transition_df.to_csv(OUT / 'support_transition_matrix_v2.csv', index=False,
                     encoding='utf-8-sig')

print('✓ CSVs written:')
for f in ['support_thresholds_by_year_v2.csv', 'soe_support_firmyear_v2.csv',
          'support_year_sample_summary_v2.csv', 'support_multiyear_summary_v2.csv',
          'support_amount_2025_v2.csv', 'support_transition_matrix_v2.csv']:
    print('   ', f)
