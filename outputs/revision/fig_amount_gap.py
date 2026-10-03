#!/usr/bin/env python3
"""金额 gap（亿元）分年份堆叠分解图 — 论文核心图（最终交付结果可视化）。"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = "/Users/violet/Desktop/DURF/模型/Linear-A"
OUT_DIR = PROJECT_DIR + "/final_results"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ---- 读取数据 ----
df = pd.read_csv(f"{PROJECT_DIR}/final_results/summary_by_sample_year.csv")
years = sorted(df['year'].unique())  # 2019..2025
order = ['CSI300 SOE', 'CSI500 SOE', 'CSI1000 SOE', 'Non-index SOE']  # 互斥分解
all_soe = df[df['sample'] == 'All A-share SOE'].set_index('year')['sum_gap_amount_yi']

# 颜色（学术风格）
colors = {
    'CSI300 SOE': '#2C3E50',   # 深蓝灰（大市值）
    'CSI500 SOE': '#5D6D7E',   # 中灰蓝
    'CSI1000 SOE': '#85929E',  # 浅灰蓝
    'Non-index SOE': '#B03A2E', # 砖红（非指数，突出）
}

# ---- 图 1：金额 gap 分年份（堆叠柱状 + All SOE 总额线）----
fig, ax = plt.subplots(figsize=(10.5, 6))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

x = np.arange(len(years))
bottom = np.zeros(len(years))
for samp in order:
    vals = df[df['sample'] == samp].set_index('year').reindex(years)['sum_gap_amount_yi'].values
    ax.bar(x, vals, bottom=bottom, width=0.62, color=colors[samp],
           edgecolor='white', linewidth=0.5, label=samp.replace(' SOE', ''))
    bottom += vals

# All SOE 总额线
tot = all_soe.reindex(years).values
ax.plot(x, tot, color='#C0392B', marker='o', markersize=6, linewidth=2.2,
        label='All SOE 合计', zorder=5)

for xi, v in zip(x, tot):
    ax.annotate(f'{v:,.0f}', (xi, v), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=8.5, fontweight='bold', color='#C0392B')

ax.axhline(0, color='#999', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
ax.set_xlabel('预测年份')
ax.set_ylabel('金额 gap（亿元）')
ax.set_title('反事实金额缺口（Gap × 资产总计）分年份互斥分解，2019–2025',
             fontweight='bold')
ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=9)
ax.grid(axis='y', alpha=0.25)
ax.margins(x=0.02)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_amount_gap_by_year.png", facecolor='white')
fig.savefig(f"{OUT_DIR}/fig_amount_gap_by_year.pdf", facecolor='white')
plt.close(fig)
print('✓ fig_amount_gap_by_year.png/pdf saved')

# ---- 图 2：比例 gap（pp）与金额 gap（亿元）并排 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
fig.patch.set_facecolor('white')

samples = ['All A-share SOE', 'CSI300 SOE', 'CSI500 SOE', 'CSI1000 SOE']
markers = {'All A-share SOE': 'o', 'CSI300 SOE': 's', 'CSI500 SOE': '^', 'CSI1000 SOE': 'd'}
line_cols = {'All A-share SOE': '#C0392B', 'CSI300 SOE': '#2C3E50',
             'CSI500 SOE': '#5D6D7E', 'CSI1000 SOE': '#85929E'}

ax = axes[0]; ax.set_facecolor('white')
for s in samples:
    sub = df[df['sample'] == s].set_index('year').reindex(years)
    ax.plot(x, sub['mean_gap_ratio'].values * 100, marker=markers[s], markersize=5,
            color=line_cols[s], linewidth=1.8, label=s.replace(' SOE', ''))
ax.axhline(0, color='#999', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
ax.set_xlabel('预测年份'); ax.set_ylabel('比例 gap（pp）')
ax.set_title('反事实比例缺口（firm 层面均值）', fontweight='bold')
ax.grid(axis='y', alpha=0.25); ax.legend(frameon=False, fontsize=8.5)

ax = axes[1]; ax.set_facecolor('white')
for s in samples:
    sub = df[df['sample'] == s].set_index('year').reindex(years)
    ax.plot(x, sub['sum_gap_amount_yi'].values, marker=markers[s], markersize=5,
            color=line_cols[s], linewidth=1.8, label=s.replace(' SOE', ''))
ax.axhline(0, color='#999', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
ax.set_xlabel('预测年份'); ax.set_ylabel('金额 gap（亿元）')
ax.set_title('反事实金额缺口（Gap × 资产总计）', fontweight='bold')
ax.grid(axis='y', alpha=0.25); ax.legend(frameon=False, fontsize=8.5)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_ratio_vs_amount_gap.png", facecolor='white')
fig.savefig(f"{OUT_DIR}/fig_ratio_vs_amount_gap.pdf", facecolor='white')
plt.close(fig)
print('✓ fig_ratio_vs_amount_gap.png/pdf saved')

# 打印关键数值（供论文引用）
print('\n=== 金额 gap 汇总（亿元）===')
piv = df.pivot_table(index='year', columns='sample', values='sum_gap_amount_yi')[samples + ['Non-index SOE']]
print(piv.round(0).to_string())
print('\n=== 2025 年各样本 ===')
print(df[df['year'] == 2025][['sample', 'n_firms', 'mean_gap_ratio', 'sum_gap_amount_yi']].round(4).to_string(index=False))
