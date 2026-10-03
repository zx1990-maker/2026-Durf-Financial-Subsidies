#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 4A — Multi-year common support (2019–2025).

Extends the 2025 single-year common-support analysis to the full panel:
  (a) year-by-year support classification,
  (b) firm-level support persistence (share of years in 'good' support),
  (c) multi-year mean gap by persistence bin,
  (d) good/boundary/extrapolation transition matrix across consecutive years,
  (e) multi-year amount gap by support status.

Outputs (outputs/revision2/):
  common_support_by_year_2019_2025.csv
  firm_persistent_support.csv
  persistent_support_summary.csv
  multiyear_common_support_gap.csv
  common_support_transition_matrix.csv
  multiyear_amount_gap_summary.csv
  fig_dynamic_common_support_gap.png
  fig_firm_support_share_distribution.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit_common import load_screened_panel, feats_in, build, prepare_train, predict_gap, OUTPUT_DIR
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
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()


def support_category(private_df, soe_df, feats):
    """good/boundary/extrapolation per SOE row (positional alignment)."""
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
    cat = np.where(soe_dist <= p90, 'good', np.where(soe_dist <= p95, 'boundary', 'extrapolation'))
    valid_pos = np.where(soe_valid.values)[0]
    labels[valid_pos] = cat
    return labels


# ---- compute firm-year gap + support for all years ----
long_rows = []
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
    for _, r in pd2.iterrows():
        long_rows.append({'firm_id': r['firm_id'], 'year': fc_year,
                          'gap': r['gap'], 'amount_gap_yi': r['amount_gap_yi'],
                          'support': r['support']})

long_df = pd.DataFrame(long_rows)
print(f'Firm-year rows: {len(long_df)}, firms: {long_df["firm_id"].nunique()}')

# ============================================================================
# (a) year-by-year support
# ============================================================================
cat_order = ['good', 'boundary', 'extrapolation']
by_year = []
for y in range(2019, 2026):
    sub = long_df[long_df['year'] == y]
    counts = {'year': y, 'n_soe': len(sub)}
    for c in cat_order:
        counts[f'n_{c}'] = int((sub['support'] == c).sum())
        counts[f'pct_{c}'] = float((sub['support'] == c).mean()) if len(sub) else np.nan
    by_year.append(counts)
by_year = pd.DataFrame(by_year)
by_year.to_csv(OUTPUT_DIR / 'common_support_by_year_2019_2025.csv', index=False, encoding='utf-8-sig')
print('\n=== Year-by-year support ===')
print(by_year.round(4).to_string(index=False))

# ============================================================================
# (b) firm-level persistence
# ============================================================================
firm_persist = long_df.groupby('firm_id').agg(
    n_years=('year', 'count'),
    n_good_years=('support', lambda s: int((s == 'good').sum())),
    n_extrap_years=('support', lambda s: int((s == 'extrapolation').sum())),
    mean_gap=('gap', 'mean'),
    mean_amount_gap_yi=('amount_gap_yi', 'mean'),
).reset_index()
# mean gap computed only over 'good'-support years (recomputed cleanly below)
mean_gap_good = []
for fid, g in long_df.groupby('firm_id'):
    good_g = g.loc[g['support'] == 'good', 'gap']
    mean_gap_good.append(float(good_g.mean()) if len(good_g) else np.nan)
firm_persist['mean_gap_good'] = mean_gap_good
firm_persist['support_share'] = firm_persist['n_good_years'] / firm_persist['n_years']
firm_persist.to_csv(OUTPUT_DIR / 'firm_persistent_support.csv', index=False, encoding='utf-8-sig')
print(f'\n=== Firm persistence (n={len(firm_persist)} firms) ===')
print(firm_persist[['n_years', 'n_good_years', 'support_share']].describe().round(3).to_string())

# ============================================================================
# (c) persistence summary bins
# ============================================================================
def share_bin(s):
    if s >= 1.0:
        return 'always (share=1)'
    elif s >= 0.5:
        return 'mostly (0.5–1)'
    elif s > 0.0:
        return 'rarely (0–0.5)'
    return 'never (share=0)'

firm_persist['bin'] = firm_persist['support_share'].apply(share_bin)
bin_order = ['always (share=1)', 'mostly (0.5–1)', 'rarely (0–0.5)', 'never (share=0)']
persist_summary = firm_persist.groupby('bin').agg(
    n_firms=('firm_id', 'count'),
    mean_support_share=('support_share', 'mean'),
).reindex(bin_order).reset_index()
persist_summary.to_csv(OUTPUT_DIR / 'persistent_support_summary.csv', index=False, encoding='utf-8-sig')
print('\n=== Persistence summary ===')
print(persist_summary.to_string(index=False))

# ============================================================================
# (d) multi-year mean gap by persistence bin
# ============================================================================
multi_gap = firm_persist.groupby('bin').agg(
    n_firms=('firm_id', 'count'),
    mean_firm_mean_gap=('mean_gap', 'mean'),
    median_firm_mean_gap=('mean_gap', 'median'),
    mean_gap_in_good_years=('mean_gap_good', 'mean'),
    mean_annual_amount_gap_yi=('mean_amount_gap_yi', 'mean'),
    sum_annual_amount_gap_yi=('mean_amount_gap_yi', 'sum'),
).reindex(bin_order).reset_index()
multi_gap.to_csv(OUTPUT_DIR / 'multiyear_common_support_gap.csv', index=False, encoding='utf-8-sig')
print('\n=== Multi-year gap by persistence bin (equal-weighted across firms) ===')
print(multi_gap.round(4).to_string(index=False))

# ============================================================================
# (e) transition matrix (good/boundary/extrapolation), consecutive years
# ============================================================================
trans_counts = pd.DataFrame(0, index=cat_order, columns=cat_order)
trans = 0
for fid, g in long_df.groupby('firm_id'):
    g = g.sort_values('year')
    for i in range(len(g) - 1):
        if g.iloc[i]['year'] + 1 == g.iloc[i + 1]['year']:
            a = g.iloc[i]['support']; b = g.iloc[i + 1]['support']
            if a in cat_order and b in cat_order:
                trans_counts.loc[a, b] += 1
                trans += 1
trans_prob = trans_counts.div(trans_counts.sum(axis=1), axis=0).round(3)
trans_prob['from_total'] = trans_counts.sum(axis=1)
trans_out = trans_counts.copy()
for c in cat_order:
    for c2 in cat_order:
        trans_out.loc[c, f'to_{c2}_prob'] = trans_prob.loc[c, c2]
trans_out.index.name = 'from_support'
trans_out.to_csv(OUTPUT_DIR / 'common_support_transition_matrix.csv', encoding='utf-8-sig')
print(f'\n=== Transition matrix (counts; {trans} transitions) ===')
print(trans_counts.to_string())
print('\n=== Row-normalized transition probabilities ===')
print(trans_prob.to_string())

# ============================================================================
# (f) multi-year amount gap by support status
# ============================================================================
multi_amt = long_df.groupby(['year', 'support']).agg(
    n_firms=('firm_id', 'count'),
    sum_gap_amount_yi=('amount_gap_yi', 'sum'),
    mean_gap=('gap', 'mean'),
).reset_index()
multi_amt.to_csv(OUTPUT_DIR / 'multiyear_amount_gap_summary.csv', index=False, encoding='utf-8-sig')
print('\n=== Multi-year amount gap by support status ===')
print(multi_amt.pivot_table(index='year', columns='support', values='sum_gap_amount_yi').round(1).to_string())

# ============================================================================
# Figures
# ============================================================================
# (g) dynamic gap by support status
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')
ax = axes[0]; ax.set_facecolor('white')
years = sorted(long_df['year'].unique())
colors = {'good': '#2E86AB', 'boundary': '#E9C46A', 'extrapolation': '#E76F51'}
for c in cat_order:
    sub = long_df[long_df['support'] == c].groupby('year')['gap'].mean().reindex(years)
    ax.plot(years, sub.values * 100, marker='o', markersize=5, color=colors[c],
            linewidth=1.8, label=c.capitalize())
ax.axhline(0, color='#999', linewidth=0.8)
ax.set_xlabel('Forecast year'); ax.set_ylabel('Mean gap (pp)')
ax.set_title('Mean gap by support status', fontweight='bold')
ax.set_xticks(years); ax.legend(frameon=False, fontsize=8.5); ax.grid(axis='y', alpha=0.3)

ax = axes[1]; ax.set_facecolor('white')
pct_good = by_year.set_index('year')['pct_good']
pct_extrap = by_year.set_index('year')['pct_extrapolation']
ax.plot(years, pct_good.values * 100, marker='o', color='#2E86AB', linewidth=1.8, label='Good support')
ax.plot(years, pct_extrap.values * 100, marker='s', color='#E76F51', linewidth=1.8, label='Extrapolation')
ax.set_xlabel('Forecast year'); ax.set_ylabel('Share of SOE firms (%)')
ax.set_title('Share of firms in / out of support', fontweight='bold')
ax.set_xticks(years); ax.legend(frameon=False, fontsize=8.5); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_dynamic_common_support_gap.png', dpi=300, facecolor='white')
plt.close(fig)
print('\n✓ fig_dynamic_common_support_gap.png saved')

# (h) firm support-share distribution
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.hist(firm_persist['support_share'].values, bins=20, color='#2E86AB', edgecolor='white', alpha=0.9)
ax.set_xlabel('Share of years in good support (per firm)')
ax.set_ylabel('Number of SOE firms')
ax.set_title('Distribution of firm-level common-support persistence (2019–2025)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_firm_support_share_distribution.png', dpi=300, facecolor='white')
plt.close(fig)
print('✓ fig_firm_support_share_distribution.png saved')

print('\n✓ audit_04a_multiyear_cs.py complete')
