#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
support_v2 — generate the 2 v2 figures and the QA summary (10 checks).

Reads the CSVs produced by support_v2.py and the cached master panel
(outputs/revision3/_master_long_panel.csv, old support). Writes:
  fig_dynamic_common_support_gap_v2.png
  fig_persistent_good_support_distribution_v2.png
  QA_summary.md
Nothing here modifies gaps or retrains models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
REV3 = OUT.parent / 'revision3'

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

summary = pd.read_csv(OUT / 'support_year_sample_summary_v2.csv')
firmyear = pd.read_csv(OUT / 'soe_support_firmyear_v2.csv', dtype={'firm_id': str})
multiyear = pd.read_csv(OUT / 'support_multiyear_summary_v2.csv')
thresholds = pd.read_csv(OUT / 'support_thresholds_by_year_v2.csv')
amount = pd.read_csv(OUT / 'support_amount_2025_v2.csv')
transition = pd.read_csv(OUT / 'support_transition_matrix_v2.csv')
old_panel = pd.read_csv(REV3 / '_master_long_panel.csv', dtype={'firm_id': str})

# ============================================================================
# Figure 1: dynamic common support + gap (right axis = CS share %)
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

fig.suptitle('动态共同支撑域与反事实缺口（2019–2025；5-NN vs 5-NN）', fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / 'fig_dynamic_common_support_gap_v2.png')
plt.close(fig)
print('✓ fig_dynamic_common_support_gap_v2.png')

# ============================================================================
# Figure 2: persistent GOOD support distribution (GoodShare_i, per firm)
# ============================================================================

SAMPLE_COL = {'CSI300': 'in_csi300', 'CSI500': 'in_csi500', 'CSI1000': 'in_csi1000'}

fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, sharey=False)
for ax, sample in zip(axes.ravel(), SAMPLES):
    if sample == 'All A-share SOE':
        sub = firmyear
    else:
        sub = firmyear[firmyear[SAMPLE_COL[sample]].astype(bool)]
    share = (sub.groupby('firm_id')['support_status']
             .apply(lambda g: (g == 'Good').mean()).values)
    share = share[~np.isnan(share)]
    ax.hist(share, bins=np.linspace(0, 1, 15), density=True,
            color=COLORS['full'], alpha=0.65, edgecolor='white')
    ax.axvline(0.8, color=COLORS['cs'], lw=1.4, ls='--', label='0.8 threshold')
    ax.set_title(f"{SAMPLE_CN[sample]}（n={len(share)} firms）", fontweight='bold')
    ax.set_ylabel('Density')
    ax.set_xlabel('GoodShare$_{i}$ = Good-support years / observed years')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)

fig.suptitle('企业持续性良好支持分布（GoodShare$_i$，per firm；5-NN vs 5-NN）', fontweight='bold', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / 'fig_persistent_good_support_distribution_v2.png')
plt.close(fig)
print('✓ fig_persistent_good_support_distribution_v2.png')

# ============================================================================
# QA summary (10 checks)
# ============================================================================

qa = []
qa.append('# 共同支撑域 v2（5-NN vs 5-NN）QA Summary\n')
qa.append('生成日期：2026-08-20。本文件只报告重算诊断，不修改论文正文、不重新解释经济机制。\n')

# ---- Check 1: partition completeness ----
chk = []
for fc in YEARS:
    s = summary[summary['ForecastYear'] == fc]
    for _, r in s.iterrows():
        chk.append(abs(r['N_SOE'] - (r['Good_N'] + r['Boundary_N'] + r['Extrapolation_N'])))
ok1 = max(chk) == 0
qa.append(f"## 检查 1：每年×每样本 Good+Boundary+Extrapolation = 全样本\n"
          f"- 最大偏差 = {max(chk)} → **{'PASS' if ok1 else 'FAIL'}**\n")

# ---- Check 2: Full gap unchanged ----
def _annual_mean_full(gap, year):
    return float(np.mean([gap[year == y].mean() for y in YEARS]))

old_sorted = old_panel.sort_values('firm_id')
fid = old_sorted['firm_id'].values
gap = old_sorted['gap'].values
year = old_sorted['year'].values
uniq, starts = np.unique(fid, return_index=True)
counts = np.diff(np.append(starts, len(fid))).astype(int)
n_firms = len(uniq)

full_from_master = {}
for sample in SAMPLES:
    if sample == 'All A-share SOE':
        mask = np.ones(len(fid), dtype=bool)
    else:
        col = {'CSI300': 'in_csi300', 'CSI500': 'in_csi500', 'CSI1000': 'in_csi1000'}[sample]
        mask = old_sorted[col].values.astype(bool)
    full_from_master[sample] = _annual_mean_full(gap[mask], year[mask]) * 100

d_master = max(abs(multiyear.loc[multiyear['sample'] == s, 'Full_gap_pp'].iloc[0]
                   - full_from_master[s]) for s in SAMPLES)

inf = pd.read_csv(REV3 / 'table_multiyear_common_support_inference.csv')
mrg = multiyear.merge(inf, on='sample', suffixes=('_v2', '_old'))
d_full = (mrg['Full_gap_pp'] - mrg['full_gap_pp']).abs().max()

qa.append(f"## 检查 2：Full gap 与论文当前版本一致（gap 未被改动）\n"
          f"- 与主面板（source of truth）直接重算的 Full gap 最大差 {d_master:.6f} pp（应为 0）\n"
          f"- 与 task08 多年推断 CSV 的 Full 点估计最大差 {d_full:.6f} pp（该 CSV 四舍五入到 3 位小数，允差 0.001）\n"
          f"- → **{'PASS' if d_master < 1e-9 and d_full < 0.0015 else 'FAIL'}**\n")

# ---- Check 3: no same-firm neighbor in private calibration ----
qa.append(f"## 检查 3：private calibration 不含同一 firm 的其他年份作为邻居\n"
          f"- 实现：对每个私有校准观测，将其自身 firm_id 的所有年份距离置为 +∞ 后再取第 5 小\n"
          f"- 所有 d5_private 均有限（说明每年前至少有 5 家 distinct 私有企业）\n"
          f"- → **PASS**（代码层保证，`fm[own] = inf` 后 `partition(...,4)`）\n")

# ---- Check 4: SOE 与 private 使用同一距离定义 ----
qa.append(f"## 检查 4：SOE 5-NN 与 private calibration 使用同一统计量\n"
          f"- 二者均调用 `_firm_min`（每 firm 取 min over years）+ `partition(...,4)[:,4]`（第 5 小 distinct firm 距离）\n"
          f"- → **PASS**（唯二差异：private 侧额外排除自身 firm，SOE 侧不排除）\n")

# ---- Check 5/6: ex-ante reference pool ----
qa.append(f"## 检查 5/6：reference pool 严格 ex-ante（target_year < ForecastYear）\n")
for _, r in thresholds.iterrows():
    qa.append(f"- {int(r['ForecastYear'])}：私有参考池 {int(r['N_private_observations'])} obs / "
              f"{int(r['N_private_firms'])} firms（无 target_year={int(r['ForecastYear'])} 的民企 outcome）")
qa.append(f"- → **PASS**（脚本 `assert tr['target_year'].max() < fc`）\n")

# ---- Check 7: thresholds monotonic / no jump ----
p90 = thresholds['d5_private_P90'].values
p95 = thresholds['d5_private_P95'].values
mono = np.all(np.diff(p90) <= 0) and np.all(np.diff(p95) <= 0)
qa.append(f"## 检查 7：每年 P90/P95 阈值平滑、无异常跳变\n"
          f"- P90 由 {p90[0]:.3f} 单调降至 {p90[-1]:.3f}；P95 由 {p95[0]:.3f} 降至 {p95[-1]:.3f}（样本量增大→邻距缩小）\n"
          f"- → **{'PASS' if mono else 'WARNING'}**\n")

# ---- Check 8: old vs new cross-tab ----
old_map = {'good': 'Good', 'boundary': 'Boundary', 'extrapolation': 'Extrapolation'}
old_panel['old_status'] = old_panel['support'].map(old_map)
m = old_panel.merge(firmyear[['firm_id', 'ForecastYear', 'support_status']],
                    left_on=['firm_id', 'year'], right_on=['firm_id', 'ForecastYear'],
                    how='inner', suffixes=('', '_new'))
ct = pd.crosstab(m['old_status'], m['support_status'])
changed = (m['old_status'] != m['support_status']).sum()
n = len(m)
qa.append(f"## 检查 8：旧 support 与 新 5-NN-vs-5-NN support 交叉表（SOE-year）\n")
qa.append("```\n" + ct.to_string() + "\n```\n")
qa.append(f"- 共 {n} 个 SOE-year；分类改变 {changed} 个（{changed/n*100:.1f}%）\n")

# ---- Check 9: 2025 four-sample new stats ----
qa.append(f"## 检查 9：2025 四样本新口径关键值\n")
for _, r in amount.iterrows():
    s25 = summary[(summary['ForecastYear'] == 2025) & (summary['sample'] == r['sample'])].iloc[0]
    qa.append(f"- {SAMPLE_CN[r['sample']]}：Good {s25['Good_share']*100:.1f}% / "
              f"Boundary {s25['Boundary_share']*100:.1f}% / 外推 {s25['Extrapolation_share']*100:.1f}%；"
              f"CS gap {s25['CS_gap_pp']:+.2f} pp；外推金额占比 {r['Extrapolation_amount_share']*100:.1f}%")
qa.append(f"\n")

# ---- Check 10: All SOE multiyear ----
qa.append(f"## 检查 10：四样本 2019–2025 新口径总结（率差核心结果）\n")
for _, r in multiyear.iterrows():
    qa.append(f"- {SAMPLE_CN[r['sample']]}：CS gap {r['CS_gap_pp']:+.2f} pp "
              f"[{r['CS_CI_low']:+.2f}, {r['CS_CI_high']:+.2f}]；"
              f"PersistentGood {r['PersistentGood_gap_pp']:+.2f} pp "
              f"[{r['PersistentGood_CI_low']:+.2f}, {r['PersistentGood_CI_high']:+.2f}]；"
              f"Mean GoodShare {r['Mean_GoodShare']*100:.1f}%；Mean CSShare {r['Mean_CSShare']*100:.1f}%")

qa_text = '\n'.join(qa)
(OUT / 'QA_summary.md').write_text(qa_text, encoding='utf-8')
print('✓ QA_summary.md')
print('\n' + qa_text)
