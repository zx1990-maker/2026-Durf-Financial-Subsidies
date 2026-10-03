#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 1 (HIGHEST PRIORITY) — Expanding-window time-leakage audit.

Strict ex-ante requirement:  OutcomeYear_train < ForecastYear.
  i.e. when forecasting EBI/A realized in year `fc_year` (using features from
  fc_year-1), the training set must contain ONLY private firms whose EBI/A
  outcome was realized in a year strictly BEFORE fc_year.

The production scripts filter training with `target_year <= fc_year`, which
INCLUDES private firms whose outcome year equals the forecast year. We flag
this, quantify it, and re-estimate the rolling GBM gap under the corrected
window `target_year < fc_year`.

Outputs (all to outputs/revision2/):
  time_window_audit.csv
  time_window_original_vs_corrected.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit_common import (
    load_screened_panel, load_yearly_codes, feats_in, build, cluster_se,
    prepare_train, predict_gap, OUTPUT_DIR,
)

OUTPUT_DIR.mkdir(exist_ok=True)

print('Loading screened panel...')
all_a = load_screened_panel()
yearly_codes = load_yearly_codes()
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()

print(f'  Panel: {len(all_a)} obs, {all_a["firm_id"].nunique()} firms')
print(f'  Train pool (private): {len(train_pool)} obs, {train_pool["firm_id"].nunique()} firms')
print(f'  target_year range: {sorted(all_a["target_year"].unique())}')

# ============================================================================
# 1. Audit table: what each forecast-year training window actually contains
# ============================================================================
print('\n=== 1. Time-window audit table ===')

audit_rows = []
for fc_year in range(2019, 2026):
    soe = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc_year)]
    soe_feature_year = int(soe['feature_year'].min()) if len(soe) else np.nan

    # ORIGINAL (production) window: target_year <= fc_year
    tr_orig = train_pool[train_pool['target_year'] <= fc_year]
    tv_orig = tr_orig.dropna(subset=feats + ['EBI_A'])
    max_out_orig = int(tv_orig['target_year'].max()) if len(tv_orig) else np.nan
    n_same_year = int((tv_orig['target_year'] == fc_year).sum())

    # CORRECTED (ex-ante) window: target_year < fc_year
    tr_corr = train_pool[train_pool['target_year'] < fc_year]
    tv_corr = tr_corr.dropna(subset=feats + ['EBI_A'])
    max_out_corr = int(tv_corr['target_year'].max()) if len(tv_corr) else np.nan

    audit_rows.append({
        'forecast_year': fc_year,
        'soe_feature_year': soe_feature_year,
        'train_feature_year_min': int(tv_orig['feature_year'].min()) if len(tv_orig) else np.nan,
        'train_feature_year_max': int(tv_orig['feature_year'].max()) if len(tv_orig) else np.nan,
        'train_outcome_year_min': int(tv_orig['target_year'].min()) if len(tv_orig) else np.nan,
        'train_outcome_year_max': max_out_orig,
        'n_train_firm_year': int(len(tv_orig)),
        'n_train_firms': int(tv_orig['firm_id'].nunique()),
        'n_same_year_private_obs': n_same_year,
        'leakage_flag': bool(max_out_orig >= fc_year),
        'corrected_outcome_year_max': max_out_corr,
        'corrected_n_train_firm_year': int(len(tv_corr)),
    })

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUTPUT_DIR / 'time_window_audit.csv', index=False, encoding='utf-8-sig')
print(audit_df.to_string(index=False))

# ============================================================================
# 2. Original vs corrected rolling GBM gap (mutually-exclusive samples)
# ============================================================================
print('\n=== 2. Original vs corrected rolling GBM gap ===')

def index_label(fid, fc_year):
    labels = []
    for i in ['CSI300', 'CSI500', 'CSI1000']:
        if fid in yearly_codes[i][fc_year]:
            labels.append(i)
    return '+'.join(labels) if labels else 'Non-index'


def run_window(window):
    """window == 'original' (<=fc) or 'corrected' (<fc). Returns list of dicts."""
    rows = []
    for fc_year in range(2019, 2026):
        if window == 'original':
            tr = train_pool[train_pool['target_year'] <= fc_year]
        else:
            tr = train_pool[train_pool['target_year'] < fc_year]
        prep = prepare_train(tr, feats)
        if len(prep['tv']) < 50:
            continue
        m = build('gbm')
        m.fit(prep['Xt'], prep['yt'])

        soe_all = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc_year)].copy()
        if len(soe_all) == 0:
            continue
        soe_all['_sample'] = soe_all['firm_id'].apply(lambda f: index_label(f, fc_year))

        preds, gaps, pd2, actuals = predict_gap(prep, m, soe_all)
        if pd2 is None or len(pd2) == 0:
            continue
        pd2 = pd2.copy()
        pd2['gap'] = gaps
        pd2['_sample'] = pd2['firm_id'].apply(lambda f: index_label(f, fc_year))

        # All SOE + mutually-exclusive subsamples
        groups = {'All A-share SOE': pd2}
        for s in ['Non-index', 'CSI300', 'CSI500', 'CSI1000']:
            sub = pd2[pd2['_sample'].str.contains(s, na=False)]
            groups[f'{s} SOE'] = sub

        for gname, g in groups.items():
            if len(g) < 3:
                continue
            se, nf = cluster_se(g['gap'].values, g['firm_id'].values)
            mg = float(np.mean(g['gap'].values))
            rows.append({
                'forecast_year': fc_year, 'sample': gname, 'model': 'gbm',
                'window': window, 'mean_gap': mg,
                'clustered_se': se,
                'ci_95_lower': mg - 1.96 * se if not np.isnan(se) else np.nan,
                'ci_95_upper': mg + 1.96 * se if not np.isnan(se) else np.nan,
                'n_train_obs': int(len(prep['tv'])), 'n_train_firms': int(prep['tv']['firm_id'].nunique()),
                'n_pred': int(len(g)), 'n_pred_firms': int(nf),
            })
    return rows

orig_rows = run_window('original')
corr_rows = run_window('corrected')
comp = pd.DataFrame(orig_rows + corr_rows)
comp.to_csv(OUTPUT_DIR / 'time_window_original_vs_corrected.csv', index=False, encoding='utf-8-sig')

# Print a compact All-SOE comparison
print('\nAll A-share SOE: original vs corrected (GBM)')
piv = comp[comp['sample'] == 'All A-share SOE'].pivot(
    index='forecast_year', columns='window', values='mean_gap')
piv['diff'] = piv.get('corrected', np.nan) - piv.get('original', np.nan)
print(piv.round(4).to_string())

print('\n✓ audit_01_time_window.py complete')
print(f'  → {OUTPUT_DIR / "time_window_audit.csv"}')
print(f'  → {OUTPUT_DIR / "time_window_original_vs_corrected.csv"}')
