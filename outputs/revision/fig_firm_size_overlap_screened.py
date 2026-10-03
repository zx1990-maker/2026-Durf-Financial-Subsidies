#!/usr/bin/env python3
"""企业规模（ln 总资产）分布重叠图 — 筛选后样本，统一 A 股 panel + 最新指数成分。"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import *   # 注意：run_counterfactual 将 OUTPUT_DIR 覆盖为 outputs/

# 重新赋值：确保输出到 final_results/（而非 run_counterfactual 中的 outputs/）
OUTPUT_DIR = PROJECT_DIR / "final_results"
INDEX_DIR = PROJECT_DIR / "input_index_weight"

SOE = {'中央国有企业', '地方国有企业'}
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print('Loading...')
all_a = load_all_a_shares(); all_a = clean_and_filter(all_a, 'A')
st_firms = set(all_a[all_a['firm_name'].str.contains('ST', na=False)]['firm_id'].unique())
all_a = all_a[~all_a['firm_id'].isin(st_firms)]
all_a = all_a[~(all_a['FinancialLeverage'] > 1.0)]
all_a = all_a[all_a['EBI_A'].abs() <= 0.5]

# 2025 指数成分（复用 common_support 逻辑）
idx_files = {
    'CSI300': ('沪深300-成分及权重-20260811.xlsx', '沪深300-成分进出记录-20260811.xlsx'),
    'CSI500': ('中证500-成分及权重-20260811.xlsx', '中证500-成分进出记录-20260811.xlsx'),
    'CSI1000': ('中证1000-成分及权重-20260811.xlsx', '中证1000-成分进出记录-20260811.xlsx'),
}
codes = {}; all_records = {}
for idx, (fw, fe) in idx_files.items():
    wb = openpyxl.load_workbook(INDEX_DIR/fw, read_only=True, data_only=True); ws = wb.active
    codes[idx] = {str(r[0]).strip() for r in ws.iter_rows(min_row=2, values_only=True) if r[0] and str(r[2]).strip() != '金融'}
    wb.close()
    wb2 = openpyxl.load_workbook(INDEX_DIR/fe, read_only=True, data_only=True); ws2 = wb2.active
    all_records[idx] = sorted([{'date': str(r[0])[:10], 'code': str(r[1]).strip(), 'action': str(r[7]).strip()}
                               for r in ws2.iter_rows(min_row=2, values_only=True) if r[0] and r[1] and r[7]],
                              key=lambda x: x['date'])
    wb2.close()

cur = codes['CSI300']; rev = sorted(all_records['CSI300'], key=lambda r: r['date'], reverse=True)
c300 = set(cur)
for rec in rev:
    if rec['date'] <= '2025-07-01': break
    if rec['action'] == '纳入': c300.discard(rec['code'])
    elif rec['action'] == '剔除': c300.add(rec['code'])
cur = codes['CSI500']; rev = sorted(all_records['CSI500'], key=lambda r: r['date'], reverse=True)
c500 = set(cur)
for rec in rev:
    if rec['date'] <= '2025-07-01': break
    if rec['action'] == '纳入': c500.discard(rec['code'])
    elif rec['action'] == '剔除': c500.add(rec['code'])
cur = codes['CSI1000']; rev = sorted(all_records['CSI1000'], key=lambda r: r['date'], reverse=True)
c1000 = set(cur)
for rec in rev:
    if rec['date'] <= '2025-07-01': break
    if rec['action'] == '纳入': c1000.discard(rec['code'])
    elif rec['action'] == '剔除': c1000.add(rec['code'])

# 样本（2025，FirmSize 有效）
priv = all_a[(all_a['ownership'] == '民营企业') & (all_a['target_year'] == 2025)].dropna(subset=['FirmSize'])
soe_all = all_a[(all_a['ownership'].isin(SOE)) & (all_a['target_year'] == 2025)].dropna(subset=['FirmSize'])

samples = [
    ('民营企业（训练基准）', priv['FirmSize'].values, '#2C3E50'),
    ('全 A 股 SOE', soe_all['FirmSize'].values, '#E74C3C'),
    ('沪深300 SOE', soe_all[soe_all['firm_id'].isin(c300)]['FirmSize'].values, '#27AE60'),
    ('中证500 SOE', soe_all[soe_all['firm_id'].isin(c500)]['FirmSize'].values, '#2980B9'),
    ('中证1000 SOE', soe_all[soe_all['firm_id'].isin(c1000)]['FirmSize'].values, '#8E44AD'),
]

# ---- KDE 重叠图 ----
fig, ax = plt.subplots(figsize=(8.5, 5.4))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

x_min, x_max = np.inf, -np.inf
for label, fs, color in samples:
    fs = fs[(~np.isnan(fs)) & (~np.isinf(fs))]
    if len(fs) < 5: continue
    kde = gaussian_kde(fs, bw_method='scott')
    xr = np.linspace(fs.min() - 0.5, fs.max() + 0.5, 500)
    if label == '民营企业（训练基准）':
        ax.fill_between(xr, kde(xr), color=color, alpha=0.10, zorder=2)
        ax.plot(xr, kde(xr), color=color, linewidth=2.2, label=label, zorder=5)
    else:
        ax.plot(xr, kde(xr), color=color, linewidth=1.6, alpha=0.85, label=label, zorder=3)
    x_min = min(x_min, fs.min()); x_max = max(x_max, fs.max())

ax.set_xlim(x_min - 0.3, x_max + 0.3)
ax.set_xlabel('Firm Size = ln(总资产)')
ax.set_ylabel('密度')
ax.set_title('企业规模分布：民营企业与各指数 SOE（2025，筛选后）', fontweight='bold')
ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', facecolor='white',
          loc='upper left', framealpha=0.95, fontsize=8.5)
ax.grid(True, axis='y', alpha=0.3)
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))

# n 标注
ann = '\n'.join([f"{l}：n={len(fs[(~np.isnan(fs)) & (~np.isinf(fs))])}" for l, fs, _ in samples])
ax.text(0.98, 0.97, ann, transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#DDDDDD', alpha=0.92))

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_firm_size_overlap_screened.png', facecolor='white')
fig.savefig(OUTPUT_DIR / 'fig_firm_size_overlap_screened.pdf', facecolor='white')
plt.close(fig)
print('✓ fig_firm_size_overlap_screened.png/pdf saved')

# 打印各样本 n 与关键统计
print('\n=== 企业规模分布（FirmSize，2025 筛选后）===')
rows = []
for label, fs, _ in samples:
    fs = fs[(~np.isnan(fs)) & (~np.isinf(fs))]
    rows.append({'样本': label, 'n': len(fs), '均值': np.mean(fs), '中位数': np.median(fs),
                 'P90': np.quantile(fs, 0.9), '超出民企P95%': np.mean(fs > np.quantile(priv['FirmSize'].values, 0.95))})
print(pd.DataFrame(rows).round(3).to_string(index=False))
