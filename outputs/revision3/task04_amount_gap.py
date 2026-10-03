#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 4 (revision3) — Amount-gap decomposition by common-support status.

AmountGap_it = Gap_it × Assets_it (亿元). Decompose 2019-2025 total amount gap into
good / boundary / extrapolation contributions, per sample (All A-share SOE / CSI300 /
CSI500 / CSI1000), plus asset shares.

Wording constraint: extrapolation is NOT labelled an "upper bound". It is described as
"amount-gap estimate is highly sensitive to observations outside the main
private-sector support region."

Outputs (outputs/revision3/):
  table_amount_gap_support_decomposition.csv
  table_amount_gap_support_2025.csv
  table_amount_gap_support.md
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
CATS = ['good', 'boundary', 'extrapolation']


def decompose(sub):
    total_firms = len(sub)
    total_assets = sub['资产总计_target'].sum()
    total_gap = sub['amount_gap_yi'].sum()
    out = {'total_firms': total_firms, 'total_assets': total_assets,
           'total_amount_gap': total_gap}
    for c in CATS:
        d = sub[sub['support'] == c]
        out[f'{c}_firms'] = len(d)
        out[f'{c}_assets'] = d['资产总计_target'].sum()
        out[f'{c}_amount_gap'] = d['amount_gap_yi'].sum()
    # shares
    for c in CATS:
        out[f'{c}_asset_share'] = out[f'{c}_assets'] / total_assets if total_assets else np.nan
        out[f'{c}_amount_share'] = out[f'{c}_amount_gap'] / total_gap if total_gap else np.nan
    out['good_plus_boundary_amount_gap'] = out['good_amount_gap'] + out['boundary_amount_gap']
    out['good_plus_boundary_asset_share'] = (out['good_assets'] + out['boundary_assets']) / total_assets if total_assets else np.nan
    out['good_plus_boundary_amount_share'] = out['good_plus_boundary_amount_gap'] / total_gap if total_gap else np.nan
    return out


rows = []
for year in range(2019, 2026):
    for sample in SAMPLES:
        sub = long[(long['year'] == year) & sample_mask(long, sample)]
        if len(sub) == 0:
            continue
        d = decompose(sub)
        d['year'] = year
        d['sample'] = sample
        rows.append(d)

tab = pd.DataFrame(rows)
col_order = ['year', 'sample', 'total_firms', 'good_firms', 'boundary_firms', 'extrapolation_firms',
             'total_assets', 'good_assets', 'boundary_assets', 'extrapolation_assets',
             'good_asset_share', 'boundary_asset_share', 'extrapolation_asset_share',
             'total_amount_gap', 'good_amount_gap', 'boundary_amount_gap', 'extrapolation_amount_gap',
             'good_amount_share', 'boundary_amount_share', 'extrapolation_amount_share',
             'good_plus_boundary_amount_gap', 'good_plus_boundary_asset_share', 'good_plus_boundary_amount_share']
tab = tab[col_order]
tab.to_csv(OUT3 / 'table_amount_gap_support_decomposition.csv', index=False, encoding='utf-8-sig')
print('\n=== table_amount_gap_support_decomposition (亿元) ===')
print(tab.round(2).to_string(index=False))

# ---- 2025 compressed table ----
d25 = tab[tab['year'] == 2025][['sample', 'total_amount_gap', 'good_amount_gap', 'boundary_amount_gap',
                                'extrapolation_amount_gap', 'extrapolation_amount_share',
                                'extrapolation_asset_share', 'total_firms']].copy()
d25 = d25.rename(columns={
    'total_amount_gap': 'Total amount gap', 'good_amount_gap': 'Good',
    'boundary_amount_gap': 'Boundary', 'extrapolation_amount_gap': 'Extrapolation',
    'extrapolation_amount_share': 'Extrapolation share of amount',
    'extrapolation_asset_share': 'Extrapolation share of assets',
    'total_firms': 'n_firms',
})
d25.to_csv(OUT3 / 'table_amount_gap_support_2025.csv', index=False, encoding='utf-8-sig')
print('\n=== table_amount_gap_support_2025 ===')
print(d25.round(3).to_string(index=False))

# ---- Markdown ----
lines = ['# TASK 4 — Amount Gap 的支持域分解（亿元）\n']
lines.append(
    '`AmountGap_it = Gap_it × Assets_it`，单位为亿元（`资产总计_target / 1e8`）。'
    '逐年 rolling ex-ante GBM + 5-NN common support 分类后，按 good / boundary / extrapolation 分解。\n'
)
lines.append('## 2025 年重点（per sample）\n')
lines.append('| Sample | Total amount gap | Good | Boundary | Extrapolation | Extrapolation share of amount | Extrapolation share of assets |')
lines.append('|---|---|---|---|---|---|---|')
for _, r in d25.iterrows():
    lines.append(
        f"| {r['sample']} | {r['Total amount gap']:.2f} | {r['Good']:.2f} | {r['Boundary']:.2f} | "
        f"{r['Extrapolation']:.2f} | {r['Extrapolation share of amount']*100:.1f}% | "
        f"{r['Extrapolation share of assets']*100:.1f}% |"
    )
lines.append('\n## 说明\n')
lines.append(
    '注意：extrapolation 部分**不**称为“上界”。它仅表示：'
    'amount-gap estimate is highly sensitive to observations outside the main '
    'private-sector support region。\n'
)
lines.append('\n## 2019–2025 逐年（All SOE）\n')
lines.append('| Year | Total | Good | Boundary | Extrapolation | Good+boundary | Extrapolation share of amount |')
lines.append('|---|---|---|---|---|---|---|')
for year in range(2019, 2026):
    r = tab[(tab['year'] == year) & (tab['sample'] == 'All A-share SOE')]
    if len(r) == 0:
        continue
    r = r.iloc[0]
    lines.append(
        f"| {year} | {r['total_amount_gap']:.1f} | {r['good_amount_gap']:.1f} | "
        f"{r['boundary_amount_gap']:.1f} | {r['extrapolation_amount_gap']:.1f} | "
        f"{r['good_plus_boundary_amount_gap']:.1f} | {r['extrapolation_amount_share']*100:.1f}% |"
    )

(OUT3 / 'table_amount_gap_support.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'\n✓ {OUT3 / "table_amount_gap_support_decomposition.csv"}')
print(f'✓ {OUT3 / "table_amount_gap_support_2025.csv"}')
print(f'✓ {OUT3 / "table_amount_gap_support.md"}')
