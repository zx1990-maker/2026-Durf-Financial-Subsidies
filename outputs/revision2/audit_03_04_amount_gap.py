#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 3 + TASK 4 — Common-support amount-gap decomposition + full vs CS gap.

For each forecast year (2019–2025) classify SOE firms into good / boundary /
extrapolation support (5-NN distance to nearest private firm vs the P90/P95 of
the private firms' own internal nearest-neighbour distance), then decompose the
amount gap (gap_ratio × total assets) by support category, and compare the full
sample gap vs the common-support (good) subsample gap.

Outputs (outputs/revision2/):
  amount_gap_by_support_year.csv
  fig_amount_gap_support_decomposition.png
  gap_full_vs_common_support.csv
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
    load_screened_panel, load_yearly_codes, feats_in, build, cluster_se,
    prepare_train, predict_gap, OUTPUT_DIR,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print('Loading screened panel...')
all_a = load_screened_panel()
yearly_codes = load_yearly_codes()
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()


def support_category(private_df, soe_df, feats):
    """Return 'good'/'boundary'/'extrapolation' label per SOE row (aligned to soe_df)."""
    labels = np.array(['na'] * len(soe_df), dtype=object)
    priv = private_df.dropna(subset=feats)
    soe_valid = soe_df[feats].notna().all(axis=1)
    soe = soe_df[soe_valid]
    if len(priv) < 10 or len(soe) == 0:
        return labels
    scl = StandardScaler()
    X_priv = scl.fit_transform(priv[feats].values)
    nn_priv = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_priv)
    priv_dist = nn_priv.kneighbors(X_priv)[0][:, 1]
    p90 = np.quantile(priv_dist, 0.90)
    p95 = np.quantile(priv_dist, 0.95)
    X_soe = scl.transform(soe[feats].values)
    nn_soe = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(X_priv)
    soe_dist = nn_soe.kneighbors(X_soe)[0][:, 0]
    cat = np.where(soe_dist <= p90, 'good',
                   np.where(soe_dist <= p95, 'boundary', 'extrapolation'))
    valid_pos = np.where(soe_valid.values)[0]
    labels[valid_pos] = cat
    return labels


def index_label(fid, fc_year):
    labs = [i for i in ['CSI300', 'CSI500', 'CSI1000'] if fid in yearly_codes[i][fc_year]]
    return '+'.join(labs) if labs else 'Non-index'


amount_rows = []
full_vs_cs_rows = []

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
    pd2['amount_gap_yi'] = gaps * pd2['资产总计_target'].values / 1e8
    pd2['support'] = support_category(tr, pd2, feats)

    # ---- TASK 3: amount gap by support category (All SOE) ----
    for cat in ['good', 'boundary', 'extrapolation']:
        sub = pd2[pd2['support'] == cat]
        amount_rows.append({
            'year': fc_year, 'support_category': cat,
            'n_firms': len(sub),
            'sum_gap_amount_yi': float(sub['amount_gap_yi'].sum()),
            'mean_gap_ratio': float(sub['gap'].mean()) if len(sub) else np.nan,
            'share_of_firms': len(sub) / len(pd2) if len(pd2) else np.nan,
        })

    # ---- TASK 4: full vs common support (All SOE) ----
    full_vs_cs_rows.append({
        'year': fc_year, 'sample': 'All A-share SOE', 'restriction': 'Full',
        'n': len(pd2), 'mean_gap': float(pd2['gap'].mean()), 'median_gap': float(pd2['gap'].median()),
    })
    good = pd2[pd2['support'] == 'good']
    full_vs_cs_rows.append({
        'year': fc_year, 'sample': 'All A-share SOE', 'restriction': 'Common support',
        'n': len(good), 'mean_gap': float(good['gap'].mean()) if len(good) else np.nan,
        'median_gap': float(good['gap'].median()) if len(good) else np.nan,
    })

amount_df = pd.DataFrame(amount_rows)
amount_df.to_csv(OUTPUT_DIR / 'amount_gap_by_support_year.csv', index=False, encoding='utf-8-sig')
print('\n=== Amount gap by support category (All SOE, 亿元) ===')
print(amount_df.pivot_table(index='year', columns='support_category', values='sum_gap_amount_yi').round(1).to_string())

full_vs_cs = pd.DataFrame(full_vs_cs_rows)
full_vs_cs.to_csv(OUTPUT_DIR / 'gap_full_vs_common_support.csv', index=False, encoding='utf-8-sig')
print('\n=== Full vs common-support gap (All SOE) ===')
piv = full_vs_cs.pivot_table(index='year', columns='restriction', values='mean_gap')
piv['diff'] = piv['Common support'] - piv['Full']
print(piv.round(4).to_string())

# ---- Figure: stacked amount gap by support category ----
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
years = sorted(amount_df['year'].unique())
x = np.arange(len(years))
colors = {'good': '#2E86AB', 'boundary': '#E9C46A', 'extrapolation': '#E76F51'}
bottom = np.zeros(len(years))
for cat in ['good', 'boundary', 'extrapolation']:
    vals = amount_df[amount_df['support_category'] == cat].set_index('year').reindex(years)['sum_gap_amount_yi'].values
    ax.bar(x, vals, bottom=bottom, width=0.6, color=colors[cat], edgecolor='white',
           linewidth=0.5, label=cat.capitalize())
    bottom += vals
ax.axhline(0, color='#999', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
ax.set_xlabel('Forecast year'); ax.set_ylabel('Amount gap (亿元)')
ax.set_title('Amount gap decomposed by common-support status (All SOE, GBM)', fontweight='bold')
ax.legend(frameon=False, ncol=3, fontsize=8.5)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_amount_gap_support_decomposition.png', dpi=300, facecolor='white')
plt.close(fig)
print('\n✓ fig_amount_gap_support_decomposition.png saved')

print('\n✓ audit_03_04_amount_gap.py complete')
