#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 6 — Industry-restricted common support.

Addresses the concern that SOE/private firms differ systematically by industry.
Recomputes the common-support classification WITHIN each industry_l2: an SOE firm
is in 'good' support only if it is close (5-NN) to PRIVATE firms in the SAME
industry (relative to that industry's private internal NN-distance P90). Compares
the resulting share-of-good-support and the mean gap against the pooled
(cross-industry) classification.

Outputs (outputs/revision2/):
  industry_restricted_common_support.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit_common import load_screened_panel, feats_in, build, prepare_train, predict_gap, OUTPUT_DIR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

print('Loading screened panel...')
all_a = load_screened_panel()
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()


def pooled_support(private_df, soe_df, feats):
    """Cross-industry (pooled) good/boundary/extrapolation, as in TASK 3/4A."""
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
    p90 = np.quantile(priv_dist, 0.90); p95 = np.quantile(priv_dist, 0.95)
    X_soe = scl.transform(soe[feats].values)
    nn_soe = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(X_priv)
    soe_dist = nn_soe.kneighbors(X_soe)[0][:, 0]
    cat = np.where(soe_dist <= p90, 'good', np.where(soe_dist <= p95, 'boundary', 'extrapolation'))
    valid_pos = np.where(soe_valid.values)[0]
    labels[valid_pos] = cat
    return labels


def industry_restricted_support(private_df, soe_df, feats, min_priv=10):
    """Within-industry good/boundary/extrapolation classification."""
    labels = np.array(['na'] * len(soe_df), dtype=object)
    soe_valid = soe_df[feats].notna().all(axis=1)
    for ind in sorted(soe_df['industry_l2'].unique()):
        priv_ind = private_df[private_df['industry_l2'] == ind].dropna(subset=feats)
        in_mask = (soe_df['industry_l2'].values == ind) & soe_valid.values
        if in_mask.sum() == 0:
            continue
        if len(priv_ind) < min_priv:
            labels[in_mask] = 'insufficient_private'
            continue
        scl = StandardScaler()
        X_priv = scl.fit_transform(priv_ind[feats].values)
        nn_priv = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_priv)
        priv_dist = nn_priv.kneighbors(X_priv)[0][:, 1]
        p90 = np.quantile(priv_dist, 0.90); p95 = np.quantile(priv_dist, 0.95)
        soe_ind = soe_df[in_mask]
        X_soe = scl.transform(soe_ind[feats].values)
        nn_soe = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(X_priv)
        soe_dist = nn_soe.kneighbors(X_soe)[0][:, 0]
        cat = np.where(soe_dist <= p90, 'good', np.where(soe_dist <= p95, 'boundary', 'extrapolation'))
        labels[in_mask] = cat
    return labels


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
    pd2['support_pooled'] = pooled_support(tr, pd2, feats)
    pd2['support_industry'] = industry_restricted_support(tr, pd2, feats)

    pooled_good = pd2[pd2['support_pooled'] == 'good']
    ind_good = pd2[pd2['support_industry'] == 'good']
    ind_insuf = pd2[pd2['support_industry'] == 'insufficient_private']

    rows.append({
        'year': fc_year, 'n_soe': len(pd2),
        'n_good_pooled': len(pooled_good),
        'pct_good_pooled': len(pooled_good) / len(pd2),
        'n_good_industry': len(ind_good),
        'pct_good_industry': len(ind_good) / len(pd2),
        'n_insufficient_private': len(ind_insuf),
        'mean_gap_full': float(pd2['gap'].mean()),
        'mean_gap_pooled_good': float(pooled_good['gap'].mean()) if len(pooled_good) else np.nan,
        'mean_gap_industry_good': float(ind_good['gap'].mean()) if len(ind_good) else np.nan,
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / 'industry_restricted_common_support.csv', index=False, encoding='utf-8-sig')

print('\n=== Industry-restricted vs pooled common support ===')
print(df.round(4).to_string(index=False))

print('\n=== Mean gap: full vs pooled-good vs industry-good ===')
print(df[['year', 'mean_gap_full', 'mean_gap_pooled_good', 'mean_gap_industry_good']].round(4).to_string(index=False))

print('\n✓ audit_06_industry_cs.py complete')
