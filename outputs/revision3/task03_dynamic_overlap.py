#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 3 (revision3) — Year-by-year dynamic overlap table (2019-2025).

For each year × sample (All A-share SOE / CSI300 / CSI500 / CSI1000):
  good_support_pct, boundary_pct, extrapolation_pct   (5-NN support shares)
  propensity_auc   (logistic SOE-vs-private separation on the 9 features)
  FirmSize_SMD     (standardized mean difference, SOE sample vs private training)
  full_gap, common_support_gap (= good+boundary), delta_gap (= CS - Full)

Outputs (outputs/revision3/):
  table_dynamic_overlap_2019_2025.csv
  table_dynamic_overlap_summary.csv
  table_dynamic_overlap.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3, compute_long_panel, sample_mask
from _audit_common import load_screened_panel, feats_in

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT3.mkdir(exist_ok=True)

long = compute_long_panel()
all_a = load_screened_panel()
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()

SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']


def propensity_auc(priv, soe, feats):
    Xp = priv[feats].dropna().values
    Xs = soe[feats].dropna().values
    if len(Xp) < 10 or len(Xs) < 10:
        return np.nan
    X = np.vstack([Xp, Xs])
    y = np.array([0] * len(Xp) + [1] * len(Xs))
    scl = StandardScaler().fit(X)
    Xz = scl.transform(X)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xz, y)
    return roc_auc_score(y, clf.predict_proba(Xz)[:, 1])


def firm_size_smd(priv, soe):
    a = priv['FirmSize'].dropna()
    b = soe['FirmSize'].dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan


rows = []
for fc_year in range(2019, 2026):
    priv = train_pool[train_pool['target_year'] < fc_year]
    for sample in SAMPLES:
        sub = long[(long['year'] == fc_year) & sample_mask(long, sample)]
        if len(sub) == 0:
            continue
        n = len(sub)
        good = (sub['support'] == 'good')
        boundary = (sub['support'] == 'boundary')
        extrap = (sub['support'] == 'extrapolation')
        cs = good | boundary
        full_gap = sub['gap'].mean()
        cs_gap = sub.loc[cs, 'gap'].mean() if cs.any() else np.nan

        rows.append({
            'year': fc_year,
            'sample': sample,
            'good_support_pct': float(good.mean() * 100),
            'boundary_pct': float(boundary.mean() * 100),
            'extrapolation_pct': float(extrap.mean() * 100),
            'propensity_auc': propensity_auc(priv, sub, feats),
            'FirmSize_SMD': firm_size_smd(priv, sub),
            'full_gap': float(full_gap),
            'common_support_gap': float(cs_gap),
            'delta_gap': float(cs_gap - full_gap) if not np.isnan(cs_gap) else np.nan,
            'n': n,
        })

tab = pd.DataFrame(rows)
tab.to_csv(OUT3 / 'table_dynamic_overlap_2019_2025.csv', index=False, encoding='utf-8-sig')
print('\n=== table_dynamic_overlap_2019_2025 ===')
print(tab.round(4).to_string(index=False))

# ---- compressed summary (per sample) ----
summary_rows = []
for sample in SAMPLES:
    s = tab[tab['sample'] == sample]
    if len(s) == 0:
        continue
    summary_rows.append({
        'sample': sample,
        'mean_good_support_pct': s['good_support_pct'].mean(),
        'min_good_support_pct': s['good_support_pct'].min(),
        'max_good_support_pct': s['good_support_pct'].max(),
        'mean_extrapolation_pct': s['extrapolation_pct'].mean(),
        'mean_propensity_auc': s['propensity_auc'].mean(),
        'mean_FirmSize_SMD': s['FirmSize_SMD'].mean(),
        'mean_full_gap': s['full_gap'].mean(),
        'mean_cs_gap': s['common_support_gap'].mean(),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT3 / 'table_dynamic_overlap_summary.csv', index=False, encoding='utf-8-sig')
print('\n=== table_dynamic_overlap_summary ===')
print(summary.round(4).to_string(index=False))

# ---- Markdown ----
lines = ['# TASK 3 — 逐年 Dynamic Overlap 表（2019–2025）\n']
lines.append(
    '逐年 rolling ex-ante GBM + 5-NN common support 分类；`propensity_auc` 为 logistic '
    'SOE-vs-民营 在 9 个数值特征上的 ROC-AUC（越接近 1 表示重叠越差）；'
    '`FirmSize_SMD` 为 SOE 样本 vs 民营训练样本的 FirmSize 标准化均值差。'
    '`common_support_gap` = good+boundary 的 mean gap；`delta_gap` = CS − Full。\n'
)
lines.append('| Year | Sample | Good % | Boundary % | Extrapolation % | AUC | FirmSize SMD | Full gap | CS gap | Δ gap |')
lines.append('|---|---|---|---|---|---|---|---|---|---|')
for _, r in tab.iterrows():
    lines.append(
        f"| {int(r['year'])} | {r['sample']} | {r['good_support_pct']:.1f} | {r['boundary_pct']:.1f} | "
        f"{r['extrapolation_pct']:.1f} | {r['propensity_auc']:.3f} | {r['FirmSize_SMD']:.3f} | "
        f"{r['full_gap']:.4f} | {r['common_support_gap']:.4f} | {r['delta_gap']:.4f} |"
    )
lines.append('\n## 压缩版（per sample 均值）\n')
lines.append('| Sample | Mean good % | Min good % | Max good % | Mean extrap % | Mean AUC | Mean SMD | Mean full gap | Mean CS gap |')
lines.append('|---|---|---|---|---|---|---|---|---|')
for _, r in summary.iterrows():
    lines.append(
        f"| {r['sample']} | {r['mean_good_support_pct']:.1f} | {r['min_good_support_pct']:.1f} | "
        f"{r['max_good_support_pct']:.1f} | {r['mean_extrapolation_pct']:.1f} | "
        f"{r['mean_propensity_auc']:.3f} | {r['mean_FirmSize_SMD']:.3f} | "
        f"{r['mean_full_gap']:.4f} | {r['mean_cs_gap']:.4f} |"
    )

(OUT3 / 'table_dynamic_overlap.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'\n✓ {OUT3 / "table_dynamic_overlap_2019_2025.csv"}')
print(f'✓ {OUT3 / "table_dynamic_overlap_summary.csv"}')
print(f'✓ {OUT3 / "table_dynamic_overlap.md"}')
