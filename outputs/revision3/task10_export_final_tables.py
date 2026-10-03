#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision3 — Export the two appendix tables as clean final CSVs.

  table_dynamic_common_support.csv : 2019-2025 year x sample dynamic common support
                                     (good/boundary/extrapolation %, ROC-AUC,
                                      FirmSize SMD, Full gap, CS gap)
  table_amount_gap_support.csv     : 2019-2025 year x sample amount-gap decomposition
                                     (total/good/boundary/extrapolation amount gap,
                                      extrapolation amount share, extrapolation asset share)

These are derived from the already-computed
  table_dynamic_overlap_2019_2025.csv
  table_amount_gap_support_decomposition.csv
(no recomputation). Gaps are reported in percentage points (pp); amount gaps in 亿元.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3

OUT3.mkdir(exist_ok=True)

# ---- Appendix Table 1: dynamic common support ----
dyn = pd.read_csv(OUT3 / 'table_dynamic_overlap_2019_2025.csv')
t1 = pd.DataFrame({
    'year': dyn['year'],
    'sample': dyn['sample'],
    'good_support_pct': dyn['good_support_pct'].round(2),
    'boundary_pct': dyn['boundary_pct'].round(2),
    'extrapolation_pct': dyn['extrapolation_pct'].round(2),
    'roc_auc': dyn['propensity_auc'].round(3),
    'firmsize_smd': dyn['FirmSize_SMD'].round(3),
    'full_gap_pp': (dyn['full_gap'] * 100).round(3),
    'cs_gap_pp': (dyn['common_support_gap'] * 100).round(3),
    'n': dyn['n'].astype(int),
})
t1.to_csv(OUT3 / 'table_dynamic_common_support.csv', index=False, encoding='utf-8-sig')

# ---- Appendix Table 2: amount-gap support decomposition ----
amt = pd.read_csv(OUT3 / 'table_amount_gap_support_decomposition.csv')
t2 = pd.DataFrame({
    'year': amt['year'],
    'sample': amt['sample'],
    'total_amount_gap_yi': amt['total_amount_gap'].round(2),
    'good_amount_gap_yi': amt['good_amount_gap'].round(2),
    'boundary_amount_gap_yi': amt['boundary_amount_gap'].round(2),
    'extrapolation_amount_gap_yi': amt['extrapolation_amount_gap'].round(2),
    'extrapolation_amount_share': amt['extrapolation_amount_share'].round(3),
    'extrapolation_asset_share': amt['extrapolation_asset_share'].round(3),
    'n_firms': amt['total_firms'].astype(int),
})
t2.to_csv(OUT3 / 'table_amount_gap_support.csv', index=False, encoding='utf-8-sig')

print('=== table_dynamic_common_support.csv ===')
print(t1.to_string(index=False))
print('\n=== table_amount_gap_support.csv (All SOE only) ===')
print(t2[t2['sample'] == 'All A-share SOE'].to_string(index=False))

print(f'\n✓ {OUT3 / "table_dynamic_common_support.csv"}')
print(f'✓ {OUT3 / "table_amount_gap_support.csv"}')
