#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision3 — two figures for the paper.

  fig_dynamic_common_support_gap.png : 2x2 panels (All SOE / CSI300 / CSI500 / CSI1000),
       2019-2025 lines of Full gap (pp), CS gap (pp) and Good-support share (%).
  fig_persistent_support_distribution.png : 2x2 panels, histogram of per-firm
       SupportShare_i = GoodSupportYears_i / ObservedYears_i, per sample.

Gaps in percentage points; support share in %. Reads the already-computed
table_dynamic_common_support.csv and the master long panel (_audit3.compute_long_panel).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3, compute_long_panel, sample_mask

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    'axes.spines.top': False, 'axes.spines.right': False,
})

OUT3.mkdir(exist_ok=True)
SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']
YEARS = np.arange(2019, 2026)
COLORS = {'full': '#1f77b4', 'cs': '#d62728', 'good': '#7f7f7f'}


# ---------------- Figure 1: dynamic common support + gap ----------------
dyn = pd.read_csv(OUT3 / 'table_dynamic_common_support.csv')

fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
for ax, sample in zip(axes.ravel(), SAMPLES):
    s = dyn[dyn['sample'] == sample].sort_values('year')
    ax.plot(s['year'], s['full_gap_pp'], marker='o', ms=3.5, lw=1.6,
            color=COLORS['full'], label='Full gap')
    ax.plot(s['year'], s['cs_gap_pp'], marker='s', ms=3, lw=1.4, ls='--',
            color=COLORS['cs'], label='CS gap')
    ax.axhline(0, color='black', lw=0.7, alpha=0.5)
    ax.set_title(sample, fontweight='bold')
    ax.set_ylabel('Gap (pp)')
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], rotation=45, ha='right', fontsize=8)

    ax2 = ax.twinx()
    ax2.fill_between(s['year'], 0, s['good_support_pct'], color=COLORS['good'], alpha=0.18)
    ax2.plot(s['year'], s['good_support_pct'], color=COLORS['good'], lw=1.1, ls=':',
             marker='^', ms=3, label='Good-support %')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Good-support share (%)', color=COLORS['good'], fontsize=8)
    ax2.tick_params(axis='y', labelcolor=COLORS['good'], labelsize=8)

    # single shared legend per panel
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc='upper right', fontsize=7.5, framealpha=0.9)

fig.suptitle('动态共同支撑域与反事实缺口（2019–2025）', fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT3 / 'fig_dynamic_common_support_gap.png')
plt.close(fig)
print(f'✓ {OUT3 / "fig_dynamic_common_support_gap.png"}')


# ---------------- Figure 2: persistent support distribution ----------------
long = compute_long_panel()

fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, sharey=False)
for ax, sample in zip(axes.ravel(), SAMPLES):
    sub = long[sample_mask(long, sample)]
    share = (sub.groupby('firm_id')['support']
             .apply(lambda g: (g == 'good').mean()).values)
    share = share[~np.isnan(share)]
    ax.hist(share, bins=np.linspace(0, 1, 15), density=True,
            color=COLORS['full'], alpha=0.65, edgecolor='white')
    ax.axvline(0.8, color=COLORS['cs'], lw=1.4, ls='--', label='0.8 threshold')
    ax.set_title(f"{sample}（n={len(share)} firms）", fontweight='bold')
    ax.set_ylabel('Density')
    ax.set_xlabel('SupportShare$_{i}$ = Good-support years / observed years')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)

fig.suptitle('企业持续性共同支撑分布（SupportShare, per firm）', fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT3 / 'fig_persistent_support_distribution.png')
plt.close(fig)
print(f'✓ {OUT3 / "fig_persistent_support_distribution.png"}')
