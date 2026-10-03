#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raw_main_figures — the 3 Task-C figures (RAW_RAW main spec).

Reads the CSVs produced by raw_main_run.py (master RAW panel + summaries).
Writes:
  fig_summary_7year_raw.png              : 4-panel GBM Full + CS gap trend 2019-2025
  fig_dynamic_common_support_gap_raw.png : Full/CS gap + CS-share% overlay (v2 support)
  fig_industry_topbottom_trend_raw.png   : top/bottom-6 industries (All SOE)

No model retraining, no gap modification.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    'axes.spines.top': False, 'axes.spines.right': False,
})

SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']
SAMPLE_CN = {'All A-share SOE': 'All SOE', 'CSI300': '沪深300',
             'CSI500': '中证500', 'CSI1000': '中证1000'}
YEARS = np.arange(2019, 2026)
COLORS = {'full': '#1f77b4', 'cs': '#d62728', 'good': '#7f7f7f'}


summary = pd.read_csv(OUT / 'support_year_sample_summary_raw.csv')


# ============================================================================
# Figure 1: main 7-year summary (Full + CS gap, no share overlay)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
for ax, sample in zip(axes.ravel(), SAMPLES):
    s = summary[summary['sample'] == sample].sort_values('ForecastYear')
    ax.plot(s['ForecastYear'], s['Full_gap_pp'], marker='o', ms=3.5, lw=1.6,
            color=COLORS['full'], label='Full gap')
    ax.plot(s['ForecastYear'], s['CS_gap_pp'], marker='s', ms=3, lw=1.4, ls='--',
            color=COLORS['cs'], label='CS gap')
    ax.axhline(0, color='black', lw=0.7, alpha=0.5)
    ax.set_title(SAMPLE_CN[sample], fontweight='bold')
    ax.set_ylabel('Gap (pp)')
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9)

fig.suptitle('反事实缺口（RAW_RAW 主规格，2019–2025；GBM）', fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / 'fig_summary_7year_raw.png')
plt.close(fig)
print('✓ fig_summary_7year_raw.png')


# ============================================================================
# Figure 2: dynamic common support + gap (CS share % on secondary axis)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
for ax, sample in zip(axes.ravel(), SAMPLES):
    s = summary[summary['sample'] == sample].sort_values('ForecastYear')
    ax.plot(s['ForecastYear'], s['Full_gap_pp'], marker='o', ms=3.5, lw=1.6,
            color=COLORS['full'], label='Full gap')
    ax.plot(s['ForecastYear'], s['CS_gap_pp'], marker='s', ms=3, lw=1.4, ls='--',
            color=COLORS['cs'], label='CS gap')
    ax.axhline(0, color='black', lw=0.7, alpha=0.5)
    ax.set_title(SAMPLE_CN[sample], fontweight='bold')
    ax.set_ylabel('Gap (pp)')
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], rotation=45, ha='right', fontsize=8)

    ax2 = ax.twinx()
    ax2.fill_between(s['ForecastYear'], 0, s['CS_share'] * 100, color=COLORS['good'], alpha=0.18)
    ax2.plot(s['ForecastYear'], s['CS_share'] * 100, color=COLORS['good'], lw=1.1, ls=':',
             marker='^', ms=3, label='CS share %')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('CS share (%)', color=COLORS['good'], fontsize=8)
    ax2.tick_params(axis='y', labelcolor=COLORS['good'], labelsize=8)

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc='upper right', fontsize=7.5, framealpha=0.9)

fig.suptitle('动态共同支撑域与反事实缺口（RAW_RAW，2019–2025；v2 5-NN 支撑）',
             fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / 'fig_dynamic_common_support_gap_raw.png')
plt.close(fig)
print('✓ fig_dynamic_common_support_gap_raw.png')


# ============================================================================
# Figure 3: industry top/bottom trend (All SOE)
# ============================================================================

ind = pd.read_csv(OUT / 'industry_gap_by_year_raw.csv')
sub = ind[ind['Sample'] == 'All A-share SOE']
piv = sub.pivot_table(index='industry_l2', columns='ForecastYear',
                      values='Mean_gap_pp', aggfunc='mean')
n_ind = sub.groupby('industry_l2')['N'].sum()
keep = n_ind[n_ind >= 20].index
piv = piv.loc[piv.index.isin(keep)]
piv['_m'] = piv.mean(axis=1)
piv = piv.sort_values('_m', ascending=False)
top6 = piv.head(6).drop(columns='_m')
bot6 = piv.tail(6).drop(columns='_m')

colors_top = plt.cm.Reds(np.linspace(0.4, 0.9, 6))
colors_bot = plt.cm.Blues(np.linspace(0.4, 0.9, 6))

fig, ax = plt.subplots(figsize=(9, 5.5))
years = [int(c) for c in top6.columns]
for i, (ind_, row) in enumerate(top6.iterrows()):
    ax.plot(years, row.values, 'o-', color=colors_top[i], lw=1.8, ms=5, label=ind_)
for i, (ind_, row) in enumerate(bot6.iterrows()):
    ax.plot(years, row.values, 's--', color=colors_bot[i], lw=1.5, ms=5, label=ind_)
ax.axhline(0, color='#999', lw=0.8, alpha=0.5)
ax.set_xticks(years)
ax.set_xlabel('Forecast Year')
ax.set_ylabel('Gap (pp)')
ax.set_title('Top/Bottom 6 行业反事实缺口趋势（RAW_RAW，All SOE）', fontweight='bold')
ax.legend(frameon=True, fancybox=False, edgecolor='#CCC', facecolor='white',
          fontsize=6.5, ncol=2, loc='upper left')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'fig_industry_topbottom_trend_raw.png')
plt.close(fig)
print('✓ fig_industry_topbottom_trend_raw.png')
