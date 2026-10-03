#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision3 — Multi-year Common-Support inference (cluster bootstrap).

Adds SE / 95% CI / p-value to the 2019-2025 multi-year common-support means
(Full / CS / Persistent-support) reported in table_multiyear_common_support.csv.

Bootstrap design (as requested):
  - resample FIRMS (cluster = firm_id), with replacement, 2,000 reps, seed 42;
  - NO model retraining — gaps are already fitted (rolling ex-ante GBM);
  - for each rep, recompute the three metrics under the SAME definitions as task02:
        Full       = mean over t of cross-sectional mean gap
        CS         = mean over t of cross-sectional mean gap on good+boundary
        Persistent = mean of firm mean-gap over firms with SupportShare_i >= 0.8
  - point estimates are recomputed directly on the full sample (identical to task02);
    SE = std of bootstrap distribution; 95% CI = 2.5/97.5 percentiles;
    p-value = two-sided percentile test of H0: mean gap = 0.

Output (outputs/revision3/):
  table_multiyear_common_support_inference.csv
  table_multiyear_common_support_inference.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3, compute_long_panel, sample_mask

OUT3.mkdir(exist_ok=True)

RNG_SEED = 42
N_BOOT = 2000
YEARS = np.arange(2019, 2026)
SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']


# ---- point-estimate functions (identical definitions to task02) ----
def annual_mean(gap, year, mask=None):
    """mean over t of cross-sectional mean gap (optionally restricted by `mask`)."""
    vals = []
    for y in YEARS:
        m = (year == y) & (mask if mask is not None else np.ones(len(gap), dtype=bool))
        if m.any():
            vals.append(gap[m].mean())
    return float(np.mean(vals)) if vals else np.nan


def persistent_gap(share, firm_mean_gap):
    """mean firm mean-gap over firms with SupportShare_i >= 0.8."""
    sel = share >= 0.8
    return float(firm_mean_gap[sel].mean()) if sel.any() else np.nan


long = compute_long_panel()
rows = []
for sample in SAMPLES:
    sub = long[sample_mask(long, sample)].copy()
    # firm-year flat arrays, sorted by firm for fast resampling
    sub = sub.sort_values('firm_id').reset_index(drop=True)
    fid = sub['firm_id'].values
    gap = sub['gap'].values
    year = sub['year'].values
    cs_flag = sub['support'].isin(['good', 'boundary']).values

    # per-firm boundaries
    uniq, starts = np.unique(fid, return_index=True)
    counts = np.diff(np.append(starts, len(fid))).astype(int)
    n_firms = len(uniq)
    firm_of_row = np.repeat(np.arange(n_firms), counts)  # 0-based firm code per row

    # firm-level support_share and firm mean gap (for persistent metric)
    share = np.array([(sub['support'].values[starts[i]:starts[i] + counts[i]] == 'good').mean()
                      for i in range(n_firms)])
    firm_mean_gap = np.array([gap[starts[i]:starts[i] + counts[i]].mean()
                              for i in range(n_firms)])

    # ---- point estimates (full sample) ----
    pt_full = annual_mean(gap, year)
    pt_cs = annual_mean(gap, year, cs_flag)
    pt_pers = persistent_gap(share, firm_mean_gap)

    # ---- cluster bootstrap ----
    rng = np.random.default_rng(RNG_SEED)
    boot_full = np.full(N_BOOT, np.nan)
    boot_cs = np.full(N_BOOT, np.nan)
    boot_pers = np.full(N_BOOT, np.nan)

    for b in range(N_BOOT):
        codes = rng.integers(0, n_firms, size=n_firms)
        lens = counts[codes]
        block_start = np.repeat(np.cumsum(lens) - lens, lens)
        within = np.arange(lens.sum()) - block_start
        gidx = np.repeat(starts[codes], lens) + within   # global firm-year row indices

        g_b = gap[gidx]
        y_b = year[gidx]
        c_b = cs_flag[gidx]

        boot_full[b] = annual_mean(g_b, y_b)
        boot_cs[b] = annual_mean(g_b, y_b, c_b)

        sel = codes[share[codes] >= 0.8]
        boot_pers[b] = firm_mean_gap[sel].mean() if len(sel) else np.nan

    def summarize(boot):
        b = boot[~np.isnan(boot)]
        se = float(np.std(b, ddof=1)) if len(b) > 1 else np.nan
        lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan, np.nan)
        # two-sided percentile p-value for H0: mean = 0
        frac_le = float(np.mean(b <= 0))
        frac_ge = float(np.mean(b >= 0))
        p = min(1.0, 2.0 * min(frac_le, frac_ge))
        return se, lo, hi, p, len(b)

    full = summarize(boot_full)
    cs = summarize(boot_cs)
    pers = summarize(boot_pers)

    rows.append({
        'sample': sample,
        'n_firms': int(n_firms),
        'full_gap_pp': round(pt_full * 100, 3), 'full_SE_pp': round(full[0] * 100, 3),
        'full_CI_low_pp': round(full[1] * 100, 3), 'full_CI_high_pp': round(full[2] * 100, 3),
        'full_p': round(full[3], 4),
        'cs_gap_pp': round(pt_cs * 100, 3), 'cs_SE_pp': round(cs[0] * 100, 3),
        'cs_CI_low_pp': round(cs[1] * 100, 3), 'cs_CI_high_pp': round(cs[2] * 100, 3),
        'cs_p': round(cs[3], 4),
        'persistent_gap_pp': round(pt_pers * 100, 3), 'persistent_SE_pp': round(pers[0] * 100, 3),
        'persistent_CI_low_pp': round(pers[1] * 100, 3), 'persistent_CI_high_pp': round(pers[2] * 100, 3),
        'persistent_p': round(pers[3], 4),
        'n_boot_valid': int(full[4]),
    })

tab = pd.DataFrame(rows)
tab.to_csv(OUT3 / 'table_multiyear_common_support_inference.csv', index=False, encoding='utf-8-sig')

print('\n=== table_multiyear_common_support_inference (pp) ===')
cols = ['sample', 'full_gap_pp', 'full_CI_low_pp', 'full_CI_high_pp', 'full_p',
        'cs_gap_pp', 'cs_CI_low_pp', 'cs_CI_high_pp', 'cs_p',
        'persistent_gap_pp', 'persistent_CI_low_pp', 'persistent_CI_high_pp', 'persistent_p', 'n_firms']
print(tab[cols].to_string(index=False))

# ---- markdown ----
lines = ['# 多年 Common Support 推断（cluster bootstrap by firm_id, 2,000 reps, seed=42）\n']
lines.append('点估计 = 全样本直接计算（与 table_multiyear_common_support.csv 完全一致）；'
             'SE = bootstrap 标准差；95% CI = 2.5/97.5 分位；p = 双侧分位检验 H0: 平均 gap = 0。\n')
lines.append('`Full` / `CS` 为「逐年截面均值再跨年平均」；`Persistent` 为 SupportShare≥0.8 企业的 firm 均值 gap。'
             '所有 gap 以 pp（百分点）表示。\n')
lines.append('| Sample | Full (CI) | p | CS (CI) | p | Persistent (CI) | p | n_firms |')
lines.append('|---|---|---|---|---|---|---|---|')
for _, r in tab.iterrows():
    lines.append(
        f"| {r['sample']} | {r['full_gap_pp']:.2f} [{r['full_CI_low_pp']:.2f}, {r['full_CI_high_pp']:.2f}] | {r['full_p']:.4f} | "
        f"{r['cs_gap_pp']:.2f} [{r['cs_CI_low_pp']:.2f}, {r['cs_CI_high_pp']:.2f}] | {r['cs_p']:.4f} | "
        f"{r['persistent_gap_pp']:.2f} [{r['persistent_CI_low_pp']:.2f}, {r['persistent_CI_high_pp']:.2f}] | {r['persistent_p']:.4f} | {int(r['n_firms'])} |"
    )
(OUT3 / 'table_multiyear_common_support_inference.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'\n✓ {OUT3 / "table_multiyear_common_support_inference.csv"}')
print(f'✓ {OUT3 / "table_multiyear_common_support_inference.md"}')
