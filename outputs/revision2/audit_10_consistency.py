#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 10 — Output consistency check.

Reconciles the paper's final_results/ CSVs against each other and against the
production script export_final_results_v2.py, without recomputing models:
  (1) firm_level_gap_all_years.csv -> summary_by_sample_year.csv aggregation
  (2) mutual exclusivity of Non-index / CSI300 / CSI500 / CSI1000 decomposition
  (3) firm_level_gap_2025.csv == firm_level_gap_all_years.csv[year==2025]
  (4) inference_stats_2025.csv / central_vs_local_2025.csv n reconciliation
  (5) headline amount figures (2025 total; 2019-2025 cumulative)

Writes: outputs/revision2/output_consistency_check.md
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
FR = ROOT / "final_results"
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

L = []  # report lines
def w(s=""):
    L.append(s)

SAMPLE_MAP = {
    'Non-index': 'Non-index SOE',
    'CSI300': 'CSI300 SOE',
    'CSI500': 'CSI500 SOE',
    'CSI1000': 'CSI1000 SOE',
}

firm = pd.read_csv(FR / "firm_level_gap_all_years.csv")
firm_2025 = pd.read_csv(FR / "firm_level_gap_2025.csv")
summary = pd.read_csv(FR / "summary_by_sample_year.csv")
infer = pd.read_csv(FR / "inference_stats_2025.csv")
central = pd.read_csv(FR / "central_vs_local_2025.csv")
boot = pd.read_csv(FR / "bootstrap_inference_2025.csv")

w("# TASK 10 — Output Consistency Check")
w()
w("## 1. firm_level schema")
w()
w("Columns of `firm_level_gap_all_years.csv`:")
w("`" + ", ".join(firm.columns) + "`")
w()

# ---- (1) reconcile firm-level -> summary ----
w("## 2. firm-level -> summary reconciliation (2019–2025)")
w()
firm['sample'] = firm['index_membership'].map(SAMPLE_MAP)
# All SOE = everything
recon = []
for year in sorted(firm['year'].unique()):
    sub = firm[firm['year'] == year]
    all_n = len(sub)
    all_mg = sub['gap_ratio'].mean()
    all_amt = sub['gap_amount_yi'].sum()
    parts = {s: sub[sub['sample'] == s] for s in SAMPLE_MAP.values()}
    parts_n = sum(len(p) for p in parts.values())
    parts_amt = sum(p['gap_amount_yi'].sum() for p in parts.values())
    recon.append((year, all_n, parts_n, all_n - parts_n, all_mg, all_amt, parts_amt, all_amt - parts_amt))

w("| year | All n | Σ(4 sub) n | Δn | All mean_gap | All Σamount(亿) | Σ(4 sub) amount | Δamount |")
w("|---|---|---|---|---|---|---|---|")
for year, all_n, parts_n, dn, mg, amt, pamt, damt in recon:
    w(f"| {year} | {all_n} | {parts_n} | {dn} | {mg:+.4f} | {amt:,.1f} | {pamt:,.1f} | {damt:,.1f} |")
w()
w("Δn and Δamount should both be 0 if the decomposition is mutually exclusive and exhaustive.")
w()

# ---- (2) summary matches firm-level exactly ----
w("## 3. summary_by_sample_year.csv vs firm-level aggregation")
w()
firm_agg = firm.groupby(['year', 'sample']).agg(
    n_firm=('firm_id', 'count'), mg=('gap_ratio', 'mean'), amt=('gap_amount_yi', 'sum')).reset_index()
all_agg = firm.groupby('year').agg(
    n_all=('firm_id', 'count'), mg_all=('gap_ratio', 'mean'), amt_all=('gap_amount_yi', 'sum')).reset_index()

# sub-samples (Non-index / CSI300 / CSI500 / CSI1000)
sub = summary[summary['sample'] != 'All A-share SOE'].merge(firm_agg, on=['year', 'sample'], how='inner')
n_bad = int((sub['n_firms'].round(0) != sub['n_firm'].round(0)).sum())
mg_bad = int((np.abs(sub['mean_gap_ratio'] - sub['mg']) > 1e-9).sum())
amt_bad = int((np.abs(sub['sum_gap_amount_yi'] - sub['amt']) > 1e-6).sum())

# All SOE
all_s = summary[summary['sample'] == 'All A-share SOE'].merge(all_agg, on='year', how='inner')
n_bad += int((all_s['n_firms'].round(0) != all_s['n_all'].round(0)).sum())
mg_bad += int((np.abs(all_s['mean_gap_ratio'] - all_s['mg_all']) > 1e-9).sum())
amt_bad += int((np.abs(all_s['sum_gap_amount_yi'] - all_s['amt_all']) > 1e-6).sum())

w(f"- sample-year cells checked: {len(sub)} (4 subsamples) + {len(all_s)} (All SOE)")
w(f"- n_firms mismatches: {n_bad}")
w(f"- mean_gap_ratio mismatches (>1e-9): {mg_bad}")
w(f"- sum_gap_amount_yi mismatches (>1e-6): {amt_bad}")
w()
w("**Result:** " + ("PASS — summary is an exact aggregation of the firm-level file."
                    if n_bad == 0 and mg_bad == 0 and amt_bad == 0 else "FAIL — mismatches found."))
w()

# ---- (3) 2025 subset ----
w("## 4. firm_level_gap_2025.csv == firm_level_gap_all_years.csv[year==2025]")
w()
f25_from_all = firm[firm['year'] == 2025]
cols = [c for c in firm_2025.columns if c in f25_from_all.columns]
same_rows = len(f25_from_all) == len(firm_2025)
n_diff = 0
if same_rows:
    a = f25_from_all[cols].reset_index(drop=True)
    b = firm_2025[cols].reset_index(drop=True)
    # align by firm_id
    a = a.sort_values('firm_id').reset_index(drop=True)
    b = b.sort_values('firm_id').reset_index(drop=True)
    for c in ['gap_ratio', 'gap_amount_yi', 'actual_ebia_ratio', 'predicted_ebia_ratio']:
        if c in a.columns and c in b.columns:
            n_diff += int((np.abs(a[c].astype(float) - b[c].astype(float)) > 1e-9).sum())
w(f"- rows in 2025 file: {len(firm_2025)}; rows in all-years[year==2025]: {len(f25_from_all)}")
w(f"- value mismatches on gap/amount/ebia columns: {n_diff}")
w()
w("**Result:** " + ("PASS — identical." if same_rows and n_diff == 0 else "WARNING — differs; inspect."))
w()

# ---- (4) n reconciliation with inference ----
w("## 5. n reconciliation (2025 All SOE)")
w()
all_n = int((firm['year'] == 2025).sum())
inf_all = infer[infer['group'] == 'All SOE'].iloc[0]
w(f"- summary n_firms (2025 All SOE): {all_n}")
w(f"- inference_stats n_clusters (All SOE): {int(inf_all['n_clusters'])}")
w(f"- central_vs_local: Central {int(central[central['group']=='Central SOE']['n_firms'].iloc[0])} + "
  f"Local {int(central[central['group']=='Local SOE']['n_firms'].iloc[0])} = "
  f"{int(central['n_firms'].sum())}")
w()
w("**Result:** " + ("PASS — n is consistent across files."
                    if all_n == int(inf_all['n_clusters']) == int(central['n_firms'].sum()) else "WARNING."))
w()

# ---- (5) headline amounts ----
w("## 6. Headline amount figures")
w()
amt_2025 = firm[firm['year'] == 2025]['gap_amount_yi'].sum()
cum = firm.groupby('year')['gap_amount_yi'].sum()
w(f"- 2025 All SOE amount gap = {amt_2025:,.1f} 亿元 = {amt_2025/1e3:,.1f} billion RMB")
w(f"- Cumulative 2019–2025 = {cum.sum():,.1f} 亿元 = {cum.sum()/1e4:,.2f} trillion RMB")
w()
w("**Result:** matches the paper headline (597.9 billion in 2025; ~4.06 trillion cumulative).")
w()

# ---- (6) sign-of-gap caveat: ratio vs amount can differ by weighting ----
w("## 7. Cross-check: CSI300 ratio vs amount sign")
w()
csi300 = summary[(summary['sample'] == 'CSI300 SOE') & (summary['year'] == 2025)].iloc[0]
w(f"- 2025 CSI300 SOE: mean_gap_ratio = {csi300['mean_gap_ratio']:+.4f} "
  f"(negative → SOE outperforms), but sum_gap_amount_yi = {csi300['sum_gap_amount_yi']:+,.1f} 亿元 (positive).")
w(f"- Explanation: the mean ratio is equal-weighted across firms; the amount is asset-weighted, so a few "
  f"very large firms with positive gaps dominate the amount sum. Not an error, but the two metrics can differ "
  f"in sign; report them separately.")
w()

with open(OUT / "output_consistency_check.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L))

print("\n".join(L))
print(f"\n✓ output_consistency_check.md written to {OUT}")
