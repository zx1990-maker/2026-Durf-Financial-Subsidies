#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 8 — Central vs local SOE difference test (rolling GBM, 2019–2025).

For each forecast year, split SOE targets into Central (中央国有企业) and Local
(地方国有企业), compute the firm-clustered mean gap for each, and test the
difference (central − local) with firm-level clustering.

Outputs (outputs/revision2/):
  central_local_difference_test.csv
  fig_central_local_difference_by_year.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit_common import (
    load_screened_panel, feats_in, build, cluster_se, cluster_t_test,
    prepare_train, predict_gap, OUTPUT_DIR,
)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print('Loading screened panel...')
all_a = load_screened_panel()
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()

all_a['central'] = all_a['ownership'] == '中央国有企业'
all_a['local'] = all_a['ownership'] == '地方国有企业'

rows = []
for fc_year in range(2019, 2026):
    tr = train_pool[train_pool['target_year'] < fc_year]
    prep = prepare_train(tr, feats)
    if len(prep['tv']) < 50:
        continue
    m = build('gbm'); m.fit(prep['Xt'], prep['yt'])

    soe = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc_year)].copy()
    preds, gaps, pd2, actuals = predict_gap(prep, m, soe)
    if pd2 is None or len(pd2) == 0:
        continue
    pd2 = pd2.copy()
    pd2['gap'] = gaps
    cen = pd2[pd2['central']]
    loc = pd2[pd2['local']]

    c_m, c_se, c_lo, c_hi, c_nf, c_p = None, None, None, None, None, None
    l_m, l_se, l_lo, l_hi, l_nf, l_p = None, None, None, None, None, None
    if len(cen) >= 3:
        c_se, c_nf = cluster_se(cen['gap'].values, cen['firm_id'].values)
        c_m = float(cen['gap'].mean())
        c_lo = c_m - 1.96 * c_se if not np.isnan(c_se) else np.nan
        c_hi = c_m + 1.96 * c_se if not np.isnan(c_se) else np.nan
    if len(loc) >= 3:
        l_se, l_nf = cluster_se(loc['gap'].values, loc['firm_id'].values)
        l_m = float(loc['gap'].mean())
        l_lo = l_m - 1.96 * l_se if not np.isnan(l_se) else np.nan
        l_hi = l_m + 1.96 * l_se if not np.isnan(l_se) else np.nan

    diff, d_se, d_lo, d_hi, t, p = (np.nan,) * 6
    if len(cen) >= 3 and len(loc) >= 3:
        diff, d_se, d_lo, d_hi, t, p = cluster_t_test(
            cen['gap'].values, cen['firm_id'].values, loc['gap'].values, loc['firm_id'].values)

    rows.append({
        'year': fc_year,
        'central_n': len(cen), 'central_gap': c_m, 'central_se': c_se,
        'central_ci_lower': c_lo, 'central_ci_upper': c_hi,
        'local_n': len(loc), 'local_gap': l_m, 'local_se': l_se,
        'local_ci_lower': l_lo, 'local_ci_upper': l_hi,
        'diff_central_minus_local': diff, 'diff_se': d_se,
        'diff_ci_lower': d_lo, 'diff_ci_upper': d_hi, 't_stat': t, 'p_value': p,
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / 'central_local_difference_test.csv', index=False, encoding='utf-8-sig')
print('\n=== Central vs Local difference (GBM, rolling) ===')
print(df[['year', 'central_n', 'central_gap', 'local_n', 'local_gap',
          'diff_central_minus_local', 'diff_ci_lower', 'diff_ci_upper', 'p_value']].round(4).to_string(index=False))

# ============================================================================
# Figure
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

ax = axes[0]; ax.set_facecolor('white')
x = df['year'].values
ax.errorbar(x, df['central_gap'] * 100, yerr=(df['central_gap'] - df['central_ci_lower']) * 100,
            fmt='o-', color='#C0392B', capsize=3, markersize=5, linewidth=1.8, label='Central SOE')
ax.errorbar(x, df['local_gap'] * 100, yerr=(df['local_gap'] - df['local_ci_lower']) * 100,
            fmt='s--', color='#2980B9', capsize=3, markersize=5, linewidth=1.8, label='Local SOE')
ax.axhline(0, color='#999', linewidth=0.8)
ax.set_xlabel('Forecast year'); ax.set_ylabel('GBM gap (pp)')
ax.set_title('Central vs Local SOE gap (GBM)', fontweight='bold')
ax.set_xticks(x); ax.legend(frameon=False, fontsize=8.5); ax.grid(axis='y', alpha=0.3)

ax = axes[1]; ax.set_facecolor('white')
ax.errorbar(x, df['diff_central_minus_local'] * 100,
            yerr=(df['diff_central_minus_local'] - df['diff_ci_lower']) * 100,
            fmt='o-', color='#2C3E50', capsize=3, markersize=6, linewidth=1.8)
ax.axhline(0, color='#999', linewidth=0.8)
ax.set_xlabel('Forecast year'); ax.set_ylabel('Difference (central − local, pp)')
ax.set_title('Central − Local difference ± 95% CI', fontweight='bold')
ax.set_xticks(x); ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_central_local_difference_by_year.png', dpi=300, facecolor='white')
plt.close(fig)
print('\n✓ fig_central_local_difference_by_year.png saved')

print('\n✓ audit_08_central_local.py complete')
