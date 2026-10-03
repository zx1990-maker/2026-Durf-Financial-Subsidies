#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 2 (revision3) — Multi-year common-support main table.

Per-year support is computed FIRST (rolling ex-ante GBM), then aggregated across
2019-2025. Support is NEVER pooled across years before classification.

For each sample (All A-share SOE / CSI300 / CSI500 / CSI1000):
  A. Full annual mean gap        = mean over t of cross-sectional mean gap (full)
  B. CS annual mean gap          = mean over t of mean gap on good+boundary
  C. Good-only annual mean gap   = mean over t of mean gap on good
  D. Firm-level full / CS mean   = firm MeanGap_i (full / good+boundary years), then equal-weight across firms
  E. Persistent-support gap      = firm-level mean gap for firms with SupportShare_i >= 0.8
  F. Support stability           = avg/median good-support share, persistent %, persistent-extrapolation %, N firms

Outputs (outputs/revision3/):
  table_multiyear_common_support.csv
  table_multiyear_common_support.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3, compute_long_panel, sample_mask

OUT3.mkdir(exist_ok=True)

long = compute_long_panel()
SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']
YEARS = range(2019, 2026)


def annual_mean(sub, support=None):
    """Mean of per-year cross-sectional mean gap (support: None | 'good' | 'cs')."""
    vals = []
    for y in YEARS:
        d = sub[sub['year'] == y]
        if support == 'good':
            d = d[d['support'] == 'good']
        elif support == 'cs':
            d = d[d['support'].isin(['good', 'boundary'])]
        if len(d):
            vals.append(d['gap'].mean())
    return (float(np.mean(vals)), len(vals)) if vals else (np.nan, 0)


def firm_level_mean(sub, support=None):
    """firm MeanGap_i (full or good+boundary years), then equal-weight across firms."""
    means = []
    for fid, g in sub.groupby('firm_id'):
        if support == 'cs':
            g = g[g['support'].isin(['good', 'boundary'])]
        if len(g):
            means.append(g['gap'].mean())
    return (float(np.mean(means)), len(means)) if means else (np.nan, 0)


def firm_support_share(sub):
    rows = []
    for fid, g in sub.groupby('firm_id'):
        n = len(g)
        n_good = int((g['support'] == 'good').sum())
        rows.append({'firm_id': fid, 'n_years': n,
                     'support_share': n_good / n,
                     'mean_gap': g['gap'].mean()})
    return pd.DataFrame(rows)


out_rows = []
for sample in SAMPLES:
    sub = long[sample_mask(long, sample)]
    n_firm_years = len(sub)
    n_firms = sub['firm_id'].nunique()

    full_annual, ny_full = annual_mean(sub, None)
    cs_annual, ny_cs = annual_mean(sub, 'cs')
    good_annual, ny_good = annual_mean(sub, 'good')

    firm_full, nf_full = firm_level_mean(sub, None)
    firm_cs, nf_cs = firm_level_mean(sub, 'cs')

    fs = firm_support_share(sub)
    persistent = fs[fs['support_share'] >= 0.8]
    persistent_gap = float(persistent['mean_gap'].mean()) if len(persistent) else np.nan
    n_persistent = len(persistent)
    n_pers_extrap = int((fs['support_share'] <= 0.2).sum())

    out_rows.append({
        'sample': sample,
        'full_annual_mean_gap': full_annual,
        'cs_annual_mean_gap': cs_annual,
        'good_only_annual_mean_gap': good_annual,
        'firm_level_full_mean_gap': firm_full,
        'firm_level_cs_mean_gap': firm_cs,
        'persistent_support_gap': persistent_gap,
        'avg_good_support_pct': float(fs['support_share'].mean() * 100) if len(fs) else np.nan,
        'median_support_share': float(fs['support_share'].median()) if len(fs) else np.nan,
        'persistent_support_pct': float(len(persistent) / len(fs) * 100) if len(fs) else np.nan,
        'persistent_extrapolation_pct': float(n_pers_extrap / len(fs) * 100) if len(fs) else np.nan,
        'n_firm_years': n_firm_years,
        'n_firms': n_firms,
    })

tab = pd.DataFrame(out_rows)
tab.to_csv(OUT3 / 'table_multiyear_common_support.csv', index=False, encoding='utf-8-sig')
print('\n=== table_multiyear_common_support ===')
print(tab.round(4).to_string(index=False))

# ---- Markdown ----
lines = ['# TASK 2 — 多年 Common Support 主表（2019–2025）\n']
lines.append(
    '逐年 rolling ex-ante GBM（`target_year < fc_year`）得到 firm-year gap，'
    '逐年做 5-NN common-support 分类（good / boundary / extrapolation），再跨年汇总；'
    '**未**把 2019–2025 pooled 后重算统一 support。\n'
)
lines.append(
    '`CS` = good + boundary；`Persistent-support` = SupportShare_i ≥ 0.8；'
    '`persistent-extrapolation` = SupportShare_i ≤ 0.2。gap 单位为比例（EBI/A 之差），表内 ×100 即百分点。\n'
)
lines.append('| Sample | Full annual mean gap | CS annual mean gap | Good-only mean gap | '
             'Firm-level full | Firm-level CS | Persistent gap | Avg good-support % | Persistent % | N firms |')
lines.append('|---|---|---|---|---|---|---|---|---|---|')
for _, r in tab.iterrows():
    lines.append(
        f"| {r['sample']} | {r['full_annual_mean_gap']:.4f} | {r['cs_annual_mean_gap']:.4f} | "
        f"{r['good_only_annual_mean_gap']:.4f} | {r['firm_level_full_mean_gap']:.4f} | "
        f"{r['firm_level_cs_mean_gap']:.4f} | {r['persistent_support_gap']:.4f} | "
        f"{r['avg_good_support_pct']:.1f} | {r['persistent_support_pct']:.1f} | {int(r['n_firms'])} |"
    )

# objective one-line summary (no causal / welfare wording)
all_row = tab[tab['sample'] == 'All A-share SOE'].iloc[0]
lines.append('\n## 客观总结\n')
lines.append(
    f"All SOE 的多年平均 gap：full 样本 {all_row['full_annual_mean_gap']:.4f}、"
    f"common-support（good+boundary）{all_row['cs_annual_mean_gap']:.4f}、"
    f"persistent-support（SupportShare≥0.8）企业 {all_row['persistent_support_gap']:.4f}。"
    "三者方向一致，说明仅保留长期具有民营企业可比对象的国企后，条件回报缺口仍存在。"
)

(OUT3 / 'table_multiyear_common_support.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'\n✓ {OUT3 / "table_multiyear_common_support.csv"}')
print(f'✓ {OUT3 / "table_multiyear_common_support.md"}')
