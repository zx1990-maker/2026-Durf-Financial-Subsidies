#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 5 — KNN comparable-firm benchmark (5-NN and 10-NN).

Robustness: replace the GBM counterfactual with a transparent k-nearest-neighbour
comparable-firm estimator on the SAME feature space (winsorized numeric features +
industry_l2 one-hot, as used by the GBM). For each SOE firm-year, the benchmark is
the mean EBI/A of its k nearest private firms (rolling, target_year < forecast year).
Gap = benchmark - SOE actual EBI/A. Compare against the GBM gap.

Outputs (outputs/revision2/):
  knn_comparable_firm_level.csv
  knn_comparable_summary_by_year.csv
  fig_gbm_vs_knn5_gap.png
  fig_gbm_vs_knn10_gap.png
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
    load_screened_panel, feats_in, build, prepare_train, predict_gap, OUTPUT_DIR,
)
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
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()


def soe_feature_matrix(prep, soe):
    """Transform SOE rows onto the prep feature space; returns (pd2, Xp).

    Mirrors predict_gap's preprocessing so pd2 has the SAME rows (and order) as
    predict_gap's output. Returns None if no valid rows."""
    ncw = prep['ncw']
    pv = soe.copy()
    for c in feats:
        pv[c + '_w'] = pv[c].clip(prep['qlo'][c], prep['qhi'][c])
    pd2 = pv.dropna(subset=ncw).copy()
    if len(pd2) == 0:
        return None, None
    Xp = np.hstack([prep['scl'].transform(prep['imp'].transform(pd2[ncw].values)),
                    prep['ohe'].transform(pd2[['industry_l2']].fillna('Unknown'))])
    return pd2, Xp


firm_rows = []
summary_rows = []

for fc_year in range(2019, 2026):
    tr = train_pool[train_pool['target_year'] < fc_year]
    prep = prepare_train(tr, feats)
    if len(prep['tv']) < 50:
        continue
    soe = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc_year)].copy()

    # GBM gap (baseline) — aligned to the SAME valid rows
    m = build('gbm'); m.fit(prep['Xt'], prep['yt'])
    _, gaps_gbm, _, _ = predict_gap(prep, m, soe)
    if gaps_gbm is None or len(gaps_gbm) == 0:
        continue

    pd2, Xp = soe_feature_matrix(prep, soe)
    if pd2 is None:
        continue
    pd2 = pd2.copy()
    pd2['gap_gbm'] = gaps_gbm            # same row order as predict_gap's output
    actuals = pd2['EBI_A'].values

    # private benchmark outcomes (actual, un-winsorized EBI_A), aligned to prep['Xt']
    priv_actual = prep['tv']['EBI_A'].values
    Xt = prep['Xt']

    for k in [5, 10]:
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(Xt)
        _, nn_idx = nn.kneighbors(Xp)
        benchmark = priv_actual[nn_idx].mean(axis=1)
        pd2[f'gap_knn{k}'] = benchmark - actuals

    for _, r in pd2.iterrows():
        firm_rows.append({
            'firm_id': r['firm_id'], 'year': fc_year,
            'gap_gbm': r['gap_gbm'],
            'gap_knn5': r['gap_knn5'],
            'gap_knn10': r['gap_knn10'],
            'ebia_actual': r['EBI_A'],
        })

    for col, lab in [('gap_gbm', 'GBM'), ('gap_knn5', 'KNN5'), ('gap_knn10', 'KNN10')]:
        summary_rows.append({
            'year': fc_year, 'estimator': lab,
            'n_firms': len(pd2), 'mean_gap': float(pd2[col].mean()),
            'median_gap': float(pd2[col].median()),
        })

firm_df = pd.DataFrame(firm_rows)
firm_df.to_csv(OUTPUT_DIR / 'knn_comparable_firm_level.csv', index=False, encoding='utf-8-sig')
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / 'knn_comparable_summary_by_year.csv', index=False, encoding='utf-8-sig')

print('\n=== KNN vs GBM gap by year (mean) ===')
piv = summary_df.pivot_table(index='year', columns='estimator', values='mean_gap')[['GBM', 'KNN5', 'KNN10']]
print(piv.round(4).to_string())

print('\n=== Firm-level correlation of gaps (pooled) ===')
print(firm_df[['gap_gbm', 'gap_knn5', 'gap_knn10']].corr().round(3).to_string())

# ============================================================================
# Figures (one single-panel figure per k)
# ============================================================================
years = sorted(firm_df['year'].unique())

def gap_figure(k, col, fname):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    gbm = firm_df.groupby('year')['gap_gbm'].mean().reindex(years) * 100
    knn = firm_df.groupby('year')[col].mean().reindex(years) * 100
    ax.plot(years, gbm.values, marker='o', color='#2980B9', linewidth=1.8, label='GBM gap')
    ax.plot(years, knn.values, marker='s', color='#E76F51', linewidth=1.8, label=f'{k}-NN comparable-firm gap')
    ax.axhline(0, color='#999', linewidth=0.8)
    ax.set_xlabel('Forecast year'); ax.set_ylabel('Mean gap (pp)')
    ax.set_title(f'GBM vs {k}-NN comparable-firm gap (rolling)', fontweight='bold')
    ax.set_xticks(years); ax.legend(frameon=False, fontsize=8.5); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'✓ {fname} saved')

gap_figure(5, 'gap_knn5', 'fig_gbm_vs_knn5_gap.png')
gap_figure(10, 'gap_knn10', 'fig_gbm_vs_knn10_gap.png')

print('\n✓ audit_05_knn.py complete')
