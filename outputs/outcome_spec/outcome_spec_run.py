#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
outcome_spec_run — outcome-definition audit: 3-specification comparison.

Specifications (identical sample / features / preprocessing / model / time split):
  CURRENT_REPLICATION : target = winsorized EBI/A (private P5/P95, per-year ex-ante);
                        actual = RAW SOE EBI/A;  gap = pred_w - raw.
  RAW_RAW            : target = RAW private EBI/A;  actual = RAW SOE EBI/A;
                        gap = pred_raw - raw.
  WIN_WINSYM         : target = winsorized EBI/A (SAME as CURRENT); actual = SOE EBI/A
                        clipped by the SAME private (L_T, U_T);  gap = pred_w - clip(raw).

Only the outcome/target side changes; the 9 X features keep their existing P5/P95
feature-winsorization, median imputation, standardization, one-hot industry, and the
GBM hyperparameters are untouched. Nothing here modifies the paper or any existing file.

Reuses the production-identical pipeline from outputs/revision2/_audit_common.py
(prepare_train / build / feats_in / load_screened_panel) and _audit3.load_yearly_codes.

Writes all outputs to outputs/outcome_spec/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'revision2'))
sys.path.insert(0, str(BASE.parent / 'revision3'))

from _audit_common import (  # noqa: E402
    load_screened_panel, load_yearly_codes, feats_in, build, prepare_train, cluster_se,
)
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline                              # noqa: E402
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # noqa: E402
from sklearn.linear_model import LinearRegression                  # noqa: E402

OUT = BASE
OUT.mkdir(exist_ok=True)

YEARS = list(range(2019, 2026))
SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']
SAMPLE_COL = {'CSI300': 'in_csi300', 'CSI500': 'in_csi500', 'CSI1000': 'in_csi1000'}
INDEXES = ['CSI300', 'CSI500', 'CSI1000']
SPECS = ['CURRENT_REPLICATION', 'RAW_RAW', 'WIN_WINSYM']
RNG_SEED = 42
N_BOOT = 2000


# ============================================================================
# load
# ============================================================================

all_a = load_screened_panel()
feats = feats_in(all_a)
assert list(feats) == ['FirmSize', 'AssetTurnover', 'FinancialLeverage',
                       'FixedAssetsRatio', 'CurrentRatio', 'CapexRatio',
                       'WorkingCapitalRatio', 'MarketShare', 'HHI'], feats
train_pool = all_a[all_a['is_private']].copy()
yearly_codes = load_yearly_codes()


# ============================================================================
# OOF helper (reuses the production GroupKFold(5)-by-firm framework)
# ============================================================================

def oof_metrics(prep, y, model_type='gbm'):
    """GroupKFold(5) cross_val_predict on the preprocessed training matrix.
    Returns dict with R2/RMSE/MAE/mean_residual/calibration or None if <2 folds."""
    X = prep['Xt']
    groups = prep['tv']['firm_id'].values
    n_firms = len(np.unique(groups))
    n_splits = min(5, n_firms)
    if n_splits < 2:
        return None
    model_cv = build(model_type)
    cv_preds = cross_val_predict(Pipeline([('model', model_cv)]), X, y,
                                 cv=GroupKFold(n_splits).split(X, y, groups=groups))
    resid = cv_preds - y
    lr = LinearRegression().fit(cv_preds.reshape(-1, 1), y)
    return {
        'OOF_N': int(len(y)),
        'R2': float(r2_score(y, cv_preds)),
        'RMSE': float(np.sqrt(mean_squared_error(y, cv_preds))),
        'MAE': float(mean_absolute_error(y, cv_preds)),
        'Mean_residual': float(resid.mean()),
        'Calibration_intercept': float(lr.intercept_),
        'Calibration_slope': float(lr.coef_[0]),
        'cv_preds': cv_preds,
        'y': y,
    }


# ============================================================================
# main loop
# ============================================================================

firmyear_parts = []
threshold_rows = []
oof_by_year = []          # per (spec, ForecastYear)
oof_pool = {s: {'preds': [], 'y': []} for s in ['CURRENT_REPLICATION', 'RAW_RAW']}

for fc in YEARS:
    tr = train_pool[train_pool['target_year'] < fc]
    assert tr['target_year'].max() < fc, f'ex-ante violation at {fc}'

    prep = prepare_train(tr, feats)
    tv = prep['tv']
    yl = float(tv['EBI_A'].quantile(0.05))
    yh = float(tv['EBI_A'].quantile(0.95))
    # QA: target == clip(raw, yl, yh)
    assert np.allclose(prep['yt'], np.clip(tv['EBI_A'].values, yl, yh)), 'target winsor mismatch'

    soe = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc)].copy()

    # ---- preprocess SOE features EXACTLY as predict_gap does ----
    pv = soe.copy()
    for c in feats:
        pv[c + '_w'] = pv[c].clip(prep['qlo'][c], prep['qhi'][c])
    pd2 = pv.dropna(subset=prep['ncw']).copy()
    X_soe = np.hstack([
        prep['scl'].transform(prep['imp'].transform(pd2[prep['ncw']].values)),
        prep['ohe'].transform(pd2[['industry_l2']].fillna('Unknown')),
    ])
    raw_actual = pd2['EBI_A'].values

    # ---- CURRENT + WIN model (winsorized target) ----
    m_w = build('gbm')
    m_w.fit(prep['Xt'], prep['yt'])
    pred_w = m_w.predict(X_soe)

    # ---- RAW model (raw target) ----
    yt_raw = tv['EBI_A'].values
    m_r = build('gbm')
    m_r.fit(prep['Xt'], yt_raw)
    pred_r = m_r.predict(X_soe)

    actual_w = np.clip(raw_actual, yl, yh)

    gap_c = pred_w - raw_actual      # CURRENT_REPLICATION
    gap_r = pred_r - raw_actual      # RAW_RAW
    gap_w = pred_w - actual_w        # WIN_WINSYM

    # ---- index membership ----
    for idx in INDEXES:
        codes = yearly_codes[idx][fc]
        pd2[f'in_{idx.lower()}'] = pd2['firm_id'].apply(lambda f: f in codes)

    part = pd.DataFrame({
        'firm_id': pd2['firm_id'].values,
        'ForecastYear': fc,
        'in_csi300': pd2['in_csi300'].values,
        'in_csi500': pd2['in_csi500'].values,
        'in_csi1000': pd2['in_csi1000'].values,
        'raw_actual_EBIA': raw_actual,
        'winsorized_actual_EBIA': actual_w,
        'CURRENT_prediction': pred_w,
        'CURRENT_gap_pp': gap_c * 100.0,
        'RAW_prediction': pred_r,
        'RAW_gap_pp': gap_r * 100.0,
        'WIN_prediction': pred_w,
        'WIN_gap_pp': gap_w * 100.0,
        'winsor_lower_bound': yl,
        'winsor_upper_bound': yh,
        'was_lower_clipped': raw_actual < yl,
        'was_upper_clipped': raw_actual > yh,
        'total_assets': pd2['资产总计_target'].values,
    })
    firmyear_parts.append(part)

    # ---- target winsorization thresholds ----
    threshold_rows.append({
        'ForecastYear': fc,
        'N_private_train_obs': int(len(tv)),
        'N_private_train_firms': int(tv['firm_id'].nunique()),
        'Raw_target_mean': float(tv['EBI_A'].mean()),
        'Raw_target_sd': float(tv['EBI_A'].std(ddof=0)),
        'P05': yl,
        'P95': yh,
        'Min': float(tv['EBI_A'].min()),
        'Max': float(tv['EBI_A'].max()),
    })

    # ---- OOF per spec (CURRENT & WIN share the winsorized-target model) ----
    oof_w = oof_metrics(prep, prep['yt'])       # winsorized target
    oof_r = oof_metrics(prep, yt_raw)           # raw target
    for spec, om in [('CURRENT_REPLICATION', oof_w),
                     ('RAW_RAW', oof_r),
                     ('WIN_WINSYM', oof_w)]:
        if om is not None:
            oof_by_year.append({'Specification': spec, 'ForecastYear': fc,
                                'OOF_N': om['OOF_N'], 'R2': om['R2'], 'RMSE': om['RMSE'],
                                'MAE': om['MAE'], 'Mean_residual': om['Mean_residual'],
                                'Calibration_intercept': om['Calibration_intercept'],
                                'Calibration_slope': om['Calibration_slope']})
    if oof_w is not None:
        oof_pool['CURRENT_REPLICATION']['preds'].append(oof_w['cv_preds'])
        oof_pool['CURRENT_REPLICATION']['y'].append(oof_w['y'])
    if oof_r is not None:
        oof_pool['RAW_RAW']['preds'].append(oof_r['cv_preds'])
        oof_pool['RAW_RAW']['y'].append(oof_r['y'])

firmyear = pd.concat(firmyear_parts, ignore_index=True)
threshold_df = pd.DataFrame(threshold_rows)

# ============================================================================
# output 2: target_winsor_thresholds_by_year.csv
# ============================================================================
threshold_df.to_csv(OUT / 'target_winsor_thresholds_by_year.csv', index=False,
                    encoding='utf-8-sig')

# ============================================================================
# output 7: firmyear_outcome_spec_comparison.csv
# ============================================================================
fy_cols = ['firm_id', 'ForecastYear', 'in_csi300', 'in_csi500', 'in_csi1000',
           'raw_actual_EBIA', 'winsorized_actual_EBIA',
           'CURRENT_prediction', 'CURRENT_gap_pp', 'RAW_prediction', 'RAW_gap_pp',
           'WIN_prediction', 'WIN_gap_pp', 'winsor_lower_bound', 'winsor_upper_bound',
           'was_lower_clipped', 'was_upper_clipped']
firmyear[fy_cols].to_csv(OUT / 'firmyear_outcome_spec_comparison.csv', index=False,
                         encoding='utf-8-sig')

# ============================================================================
# output 3: outcome_spec_gap_comparison.csv  (Spec x Year x Sample)
# ============================================================================

gap_cmp_rows = []
for spec in SPECS:
    gapcol = {'CURRENT_REPLICATION': 'CURRENT_gap_pp',
              'RAW_RAW': 'RAW_gap_pp',
              'WIN_WINSYM': 'WIN_gap_pp'}[spec]
    predcol = {'CURRENT_REPLICATION': 'CURRENT_prediction',
               'RAW_RAW': 'RAW_prediction',
               'WIN_WINSYM': 'WIN_prediction'}[spec]
    actcol = 'raw_actual_EBIA' if spec in ('CURRENT_REPLICATION', 'RAW_RAW') else 'winsorized_actual_EBIA'
    for fc in YEARS:
        for sample in SAMPLES:
            sub = firmyear[firmyear['ForecastYear'] == fc]
            if sample != 'All A-share SOE':
                sub = sub[sub[SAMPLE_COL[sample]].astype(bool)]
            if len(sub) == 0:
                continue
            g = sub[gapcol].values
            fid = sub['firm_id'].values
            se, n_f = cluster_se(g, fid)
            m = float(g.mean())
            gap_cmp_rows.append({
                'Specification': spec, 'ForecastYear': fc, 'Sample': sample,
                'N': int(len(sub)),
                'Mean_actual': float(sub[actcol].mean()),
                'Mean_predicted': float(sub[predcol].mean()),
                'Mean_gap_pp': m,
                'Firm_clustered_SE_pp': se,
                'CI95_low_pp': m - 1.96 * se if not np.isnan(se) else np.nan,
                'CI95_high_pp': m + 1.96 * se if not np.isnan(se) else np.nan,
            })
gap_cmp = pd.DataFrame(gap_cmp_rows)
gap_cmp.to_csv(OUT / 'outcome_spec_gap_comparison.csv', index=False, encoding='utf-8-sig')


# ============================================================================
# output 4: outcome_spec_multiyear_summary.csv  (Spec x Sample, cluster bootstrap)
# ============================================================================

def annual_mean(gap, year):
    vals = [gap[year == y].mean() for y in YEARS if (year == y).any()]
    return float(np.mean(vals)) if vals else np.nan


def cluster_bootstrap_full(gap, year, fid, seed=RNG_SEED, n_boot=N_BOOT):
    """Firm-cluster bootstrap (resample firm_id) of the annual-mean gap."""
    order = np.argsort(fid, kind='stable')
    fid_s = fid[order]; gap_s = gap[order]; year_s = year[order]
    uniq, starts = np.unique(fid_s, return_index=True)
    counts = np.diff(np.append(starts, len(fid_s))).astype(int)
    n_firms = len(uniq)
    pt = annual_mean(gap_s, year_s)
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        codes = rng.integers(0, n_firms, size=n_firms)
        lens = counts[codes]
        block_start = np.repeat(np.cumsum(lens) - lens, lens)
        within = np.arange(lens.sum()) - block_start
        gidx = np.repeat(starts[codes], lens) + within
        boot[b] = annual_mean(gap_s[gidx], year_s[gidx])
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    return pt, lo, hi, n_firms, int(len(gap_s))


multiyear_rows = []
for spec in SPECS:
    gapcol = {'CURRENT_REPLICATION': 'CURRENT_gap_pp',
              'RAW_RAW': 'RAW_gap_pp',
              'WIN_WINSYM': 'WIN_gap_pp'}[spec]
    for sample in SAMPLES:
        sub = firmyear if sample == 'All A-share SOE' else firmyear[firmyear[SAMPLE_COL[sample]].astype(bool)]
        pt, lo, hi, n_firms, n_fy = cluster_bootstrap_full(
            sub[gapcol].values, sub['ForecastYear'].values, sub['firm_id'].values)
        multiyear_rows.append({
            'Specification': spec, 'Sample': sample,
            'N_firmyears': n_fy, 'N_firms': n_firms,
            'Mean_gap_pp': pt, 'CI95_low_pp': lo, 'CI95_high_pp': hi,
        })
multiyear = pd.DataFrame(multiyear_rows)
multiyear.to_csv(OUT / 'outcome_spec_multiyear_summary.csv', index=False, encoding='utf-8-sig')


# ============================================================================
# output 5: outcome_spec_model_performance.csv  (Spec-level + per-year)
# ============================================================================

def pooled_metrics(preds_list, y_list):
    preds = np.concatenate(preds_list)
    y = np.concatenate(y_list)
    resid = preds - y
    lr = LinearRegression().fit(preds.reshape(-1, 1), y)
    return {
        'OOF_N': int(len(y)),
        'R2': float(r2_score(y, preds)),
        'RMSE': float(np.sqrt(mean_squared_error(y, preds))),
        'MAE': float(mean_absolute_error(y, preds)),
        'Mean_residual': float(resid.mean()),
        'Calibration_intercept': float(lr.intercept_),
        'Calibration_slope': float(lr.coef_[0]),
    }

perf_rows = []
# pooled (Specification-level; WIN == CURRENT by construction)
for spec in SPECS:
    if spec == 'WIN_WINSYM':
        pm = pooled_metrics(oof_pool['CURRENT_REPLICATION']['preds'],
                            oof_pool['CURRENT_REPLICATION']['y'])
    else:
        pm = pooled_metrics(oof_pool[spec]['preds'], oof_pool[spec]['y'])
    perf_rows.append({'Specification': spec, 'ForecastYear': 'Pooled', **pm})
# per-year
perf_rows += oof_by_year
perf = pd.DataFrame(perf_rows)
perf = perf[['Specification', 'ForecastYear', 'OOF_N', 'R2', 'RMSE', 'MAE',
             'Mean_residual', 'Calibration_intercept', 'Calibration_slope']]
perf.to_csv(OUT / 'outcome_spec_model_performance.csv', index=False, encoding='utf-8-sig')


# ============================================================================
# output 6: winsor_clipping_diagnostics.csv  (Year x Sample + pooled All SOE)
# ============================================================================

clip_rows = []
for fc in YEARS:
    for sample in SAMPLES:
        sub = firmyear[firmyear['ForecastYear'] == fc]
        if sample != 'All A-share SOE':
            sub = sub[sub[SAMPLE_COL[sample]].astype(bool)]
        n = len(sub)
        lo_c = sub['was_lower_clipped'].sum()
        up_c = sub['was_upper_clipped'].sum()
        tot = lo_c + up_c
        clipped = sub[sub['was_lower_clipped'] | sub['was_upper_clipped']]
        tot_assets = sub['total_assets'].sum()
        clip_rows.append({
            'ForecastYear': fc, 'Sample': sample, 'N': n,
            'Lower_clipped_N': int(lo_c),
            'Lower_clipped_pct': 100 * lo_c / n if n else np.nan,
            'Upper_clipped_N': int(up_c),
            'Upper_clipped_pct': 100 * up_c / n if n else np.nan,
            'Total_clipped_N': int(tot),
            'Total_clipped_pct': 100 * tot / n if n else np.nan,
            'Mean_raw_actual_clipped': float(clipped['raw_actual_EBIA'].mean()) if tot else np.nan,
            'Median_raw_actual_clipped': float(clipped['raw_actual_EBIA'].median()) if tot else np.nan,
            'Mean_raw_gap_clipped': float(clipped['CURRENT_gap_pp'].mean()) if tot else np.nan,
            'Assets_share_clipped': float(clipped['total_assets'].sum() / tot_assets) if tot_assets else np.nan,
        })
# pooled All SOE 2019-2025
sub = firmyear
n = len(sub); lo_c = sub['was_lower_clipped'].sum(); up_c = sub['was_upper_clipped'].sum(); tot = lo_c + up_c
clipped = sub[sub['was_lower_clipped'] | sub['was_upper_clipped']]
tot_assets = sub['total_assets'].sum()
clip_rows.append({
    'ForecastYear': 'Pooled', 'Sample': 'All A-share SOE', 'N': n,
    'Lower_clipped_N': int(lo_c), 'Lower_clipped_pct': 100 * lo_c / n,
    'Upper_clipped_N': int(up_c), 'Upper_clipped_pct': 100 * up_c / n,
    'Total_clipped_N': int(tot), 'Total_clipped_pct': 100 * tot / n,
    'Mean_raw_actual_clipped': float(clipped['raw_actual_EBIA'].mean()) if tot else np.nan,
    'Median_raw_actual_clipped': float(clipped['raw_actual_EBIA'].median()) if tot else np.nan,
    'Mean_raw_gap_clipped': float(clipped['CURRENT_gap_pp'].mean()) if tot else np.nan,
    'Assets_share_clipped': float(clipped['total_assets'].sum() / tot_assets) if tot_assets else np.nan,
})
clip_df = pd.DataFrame(clip_rows)
clip_df.to_csv(OUT / 'winsor_clipping_diagnostics.csv', index=False, encoding='utf-8-sig')

print('✓ outcome_spec CSVs written:')
for f in ['target_winsor_thresholds_by_year.csv', 'outcome_spec_gap_comparison.csv',
          'outcome_spec_multiyear_summary.csv', 'outcome_spec_model_performance.csv',
          'winsor_clipping_diagnostics.csv', 'firmyear_outcome_spec_comparison.csv']:
    print('   ', f)
print('   firmyear rows:', len(firmyear))
