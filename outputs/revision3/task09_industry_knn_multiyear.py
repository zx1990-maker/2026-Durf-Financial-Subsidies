#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision3 — Industry-restricted KNN multi-year average.

Aggregates the year-by-year industry-restricted KNN results
(industry_restricted_knn_summary.csv) into 2019-2025 simple averages plus the 2025
values, per sample (All A-share SOE / CSI300 / CSI500 / CSI1000).

Reports (all in percentage points):
  GBM multi-year mean, industry-KNN5 multi-year mean, industry-KNN10 multi-year mean,
  2025 GBM, 2025 KNN5, 2025 KNN10.

Output (outputs/revision3/):
  table_industry_knn_multiyear.csv
  table_industry_knn_multiyear.md
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3

OUT3.mkdir(exist_ok=True)

SRC = OUT3 / 'industry_restricted_knn_summary.csv'
d = pd.read_csv(SRC)
SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']

rows = []
for sample in SAMPLES:
    s = d[d['sample'] == sample]
    y25 = s[s['year'] == 2025].iloc[0]
    rows.append({
        'sample': sample,
        'GBM_multiyear_pp': round(s['GBM_gap'].mean() * 100, 3),
        'KNN5_multiyear_pp': round(s['industry_KNN5_gap'].mean() * 100, 3),
        'KNN10_multiyear_pp': round(s['industry_KNN10_gap'].mean() * 100, 3),
        'GBM_2025_pp': round(y25['GBM_gap'] * 100, 3),
        'KNN5_2025_pp': round(y25['industry_KNN5_gap'] * 100, 3),
        'KNN10_2025_pp': round(y25['industry_KNN10_gap'] * 100, 3),
    })

tab = pd.DataFrame(rows)
tab.to_csv(OUT3 / 'table_industry_knn_multiyear.csv', index=False, encoding='utf-8-sig')

print('\n=== table_industry_knn_multiyear (pp) ===')
print(tab.to_string(index=False))

lines = ['# 同行业 KNN 多年平均（industry-restricted KNN，2019–2025）\n']
lines.append('多年平均 = 2019–2025 逐年 gap 的简单平均；2025 列 = 2025 年单年值。'
             '单位为百分点（pp）。完整 CSI300/500/1000 明细见 industry_restricted_knn_summary.csv。\n')
lines.append('| Sample | GBM 多年 | KNN5 多年 | KNN10 多年 | GBM 2025 | KNN5 2025 | KNN10 2025 |')
lines.append('|---|---|---|---|---|---|---|')
for _, r in tab.iterrows():
    lines.append(
        f"| {r['sample']} | {r['GBM_multiyear_pp']:.2f} | {r['KNN5_multiyear_pp']:.2f} | {r['KNN10_multiyear_pp']:.2f} | "
        f"{r['GBM_2025_pp']:.2f} | {r['KNN5_2025_pp']:.2f} | {r['KNN10_2025_pp']:.2f} |"
    )
(OUT3 / 'table_industry_knn_multiyear.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'\n✓ {OUT3 / "table_industry_knn_multiyear.csv"}')
print(f'✓ {OUT3 / "table_industry_knn_multiyear.md"}')
