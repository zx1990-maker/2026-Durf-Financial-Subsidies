#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared module for outputs/revision3/ (supplementary tables + code audit).

Reuses the PRODUCTION-identical pipeline from outputs/revision2/_audit_common.py
(same screening, same index reconstruction, same GBM hyperparameters, same
winsorize/impute/scale/one-hot, same ex-ante expanding window `target_year < fc_year`).

This module does NOT modify raw data and does NOT overwrite revision2/final_results.
It writes only to outputs/revision3/.

Key entry point: compute_long_panel() builds the master firm-year panel (2019-2025)
with, for every SOE firm-year:
  gap           = predicted EBI/A - actual EBI/A   (GBM, rolling ex-ante)
  amount_gap_yi = gap * 资产总计_target / 1e8       (亿元)
  support       = good / boundary / extrapolation   (5-NN vs P90/P95 rule)
  in_csi300/500/1000 = index membership flags
The panel is cached to _master_long_panel.csv so every task reads identical numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV2 = Path(__file__).resolve().parent.parent / "revision2"
sys.path.insert(0, str(REV2))

from _audit_common import (                       # noqa: E402
    load_screened_panel, load_yearly_codes, feats_in,
    build, prepare_train, predict_gap,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.neighbors import NearestNeighbors    # noqa: E402

OUT3 = Path(__file__).resolve().parent
OUT3.mkdir(exist_ok=True)

RANDOM_STATE = 42
SAMPLE_INDEX = ['CSI300', 'CSI500', 'CSI1000']


def support_category(private_df, soe_df, feats):
    """good / boundary / extrapolation per SOE row (positional alignment).

    Identical to the production 5-NN rule used in outputs/revision2:
      - standardize the 9 numeric features on the PRIVATE sample only;
      - private internal nearest-neighbour distance P90 / P95;
      - SOE 5-NN distance to nearest private firm:
          <= P90 -> good;  <= P95 -> boundary;  > P95 -> extrapolation.
    """
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


def compute_long_panel(force=False):
    """Build (or reload) the master SOE firm-year panel for 2019-2025."""
    cache = OUT3 / "_master_long_panel.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache)
        df['firm_id'] = df['firm_id'].astype(str)
        return df

    all_a = load_screened_panel()
    yearly_codes = load_yearly_codes()
    feats = feats_in(all_a)
    train_pool = all_a[all_a['is_private']].copy()

    rows = []
    for fc_year in range(2019, 2026):
        tr = train_pool[train_pool['target_year'] < fc_year]   # strict ex-ante
        prep = prepare_train(tr, feats)
        if len(prep['tv']) < 50:
            continue
        m = build('gbm')
        m.fit(prep['Xt'], prep['yt'])

        soe = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc_year)].copy()
        preds, gaps, pd2, actuals = predict_gap(prep, m, soe)
        if pd2 is None or len(pd2) == 0:
            continue

        pd2 = pd2.copy()
        pd2['gap'] = gaps
        pd2['amount_gap_yi'] = gaps * pd2['资产总计_target'].values / 1e8
        pd2['support'] = support_category(tr, pd2, feats)
        pd2['year'] = fc_year
        for idx in SAMPLE_INDEX:
            codes = yearly_codes[idx][fc_year]
            pd2[f'in_{idx.lower()}'] = pd2['firm_id'].apply(lambda f: f in codes)

        keep = ['year', 'firm_id', 'firm_name', 'ownership', 'industry_l2',
                'EBI_A', 'gap', 'amount_gap_yi', 'support', '资产总计_target',
                'in_csi300', 'in_csi500', 'in_csi1000'] + feats
        rows.append(pd2[keep])

    long = pd.concat(rows, ignore_index=True)
    long['firm_id'] = long['firm_id'].astype(str)
    long.to_csv(cache, index=False, encoding='utf-8-sig')
    return long


def sample_mask(df, sample):
    """Return a boolean mask selecting `sample` rows of the long panel."""
    if sample == 'All A-share SOE':
        return pd.Series(True, index=df.index)
    col = {'CSI300': 'in_csi300', 'CSI500': 'in_csi500', 'CSI1000': 'in_csi1000'}[sample]
    return df[col].astype(bool)
