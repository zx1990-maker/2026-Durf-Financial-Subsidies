#!/usr/bin/env python3
"""描述性统计表（表1）：民营企业 vs SOE 的 9 个特征 + EBI/A，2025 预测期（特征年 t=2024）。

口径：完整案例（9 特征 + EBI/A 均非缺失），与估计样本一致。
输出：
  outputs/revision/descriptives_by_ownership.csv      —— 分所有权均值/SD/中位数/分位数
  outputs/revision/descriptives_by_ownership.tex      —— LaTeX booktabs 表格（可直接 include）
仅作统计汇总，不改动任何原始数据。
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "revision"

sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import load_all_a_shares, clean_and_filter, NUMERIC_FEATURES

SOE = {'中央国有企业', '地方国有企业'}
FEATURES = NUMERIC_FEATURES  # 9 个财务比率特征
TARGET = 'EBI_A'

print('Loading...')
all_a = load_all_a_shares()
all_a = clean_and_filter(all_a, 'A')

# 与最终一致的稳健性筛选
st_firms = set(all_a[all_a['firm_name'].str.contains('ST', na=False)]['firm_id'].unique())
all_a = all_a[~all_a['firm_id'].isin(st_firms)]
all_a = all_a[~(all_a['FinancialLeverage'] > 1.0)]
all_a = all_a[all_a['EBI_A'].abs() <= 0.5]

# 完整案例：9 特征 + 目标变量均非缺失（与估计样本一致）
FULL = FEATURES + [TARGET]
priv = all_a[(all_a['ownership'] == '民营企业') & (all_a['target_year'] == 2025)].dropna(subset=FULL).copy()
soe = all_a[(all_a['ownership'].isin(SOE)) & (all_a['target_year'] == 2025)].dropna(subset=FULL).copy()

# 训练样本规模（全年份，供表注）
priv_all = all_a[all_a['ownership'] == '民营企业'].dropna(subset=FULL)
n_priv_all_fy = len(priv_all)
n_priv_all_firms = priv_all['firm_id'].nunique()

rows = []
for c in FULL:
    ps = priv[c].astype(float)
    ss = soe[c].astype(float)
    rows.append({
        'variable': c,
        'priv_mean': ps.mean(), 'priv_sd': ps.std(ddof=1), 'priv_median': ps.median(),
        'priv_p10': ps.quantile(0.1), 'priv_p90': ps.quantile(0.9),
        'soe_mean': ss.mean(), 'soe_sd': ss.std(ddof=1), 'soe_median': ss.median(),
        'soe_p10': ss.quantile(0.1), 'soe_p90': ss.quantile(0.9),
    })

desc = pd.DataFrame(rows)
desc.to_csv(OUTPUT_DIR / 'descriptives_by_ownership.csv', index=False, encoding='utf-8-sig')

n_priv_2025 = priv['firm_id'].nunique()
n_soe_2025 = soe['firm_id'].nunique()

print('\n=== 描述性统计（2025 预测期，完整案例）===')
print(f'民营企业 2025（完整案例）：n = {len(priv)} obs, {n_priv_2025} firms')
print(f'SOE 2025（完整案例）：n = {len(soe)} obs, {n_soe_2025} firms')
print(f'民营企业训练样本（全年份完整案例）：{n_priv_all_fy} firm-year, {n_priv_all_firms} firms\n')
pd.set_option('display.width', 220)
print(desc.round(4).to_string(index=False))

# ---- 生成 LaTeX 表格（booktabs，均值 (SD)）----
labels = {
    'FirmSize': r'企业规模（$\ln$ 总资产）',
    'AssetTurnover': '资产周转率',
    'FinancialLeverage': '资产负债率',
    'FixedAssetsRatio': '固定资产比率',
    'CurrentRatio': '流动比率',
    'CapexRatio': '资本支出比率',
    'WorkingCapitalRatio': '营运资本比率',
    'MarketShare': '市场份额',
    'HHI': '行业集中度 HHI',
    'EBI_A': r'$EBI/A_{t+1}$',
}

lines = []
lines.append(r'\begin{table}[t]')
lines.append(r'\centering')
lines.append(r'\caption{描述性统计（2025 预测期，特征年 $t=2024$，完整案例）}')
lines.append(r'\label{tab:descriptives}')
lines.append(r'\small')
lines.append(r'\begin{tabular}{llcccc}')
lines.append(r'\toprule')
lines.append(r'变量 & 定义 & \multicolumn{2}{c}{民营企业} & \multicolumn{2}{c}{SOE} \\')
lines.append(r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}')
lines.append(r'& & 均值 & SD & 均值 & SD \\')
lines.append(r'\midrule')
for _, r in desc.iterrows():
    lab = labels.get(r['variable'], r['variable'])
    lines.append(
        f"{r['variable']} & {lab} & {r['priv_mean']:.3f} & {r['priv_sd']:.3f} "
        f"& {r['soe_mean']:.3f} & {r['soe_sd']:.3f} \\\\"
    )
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append('')
lines.append(
    r'\begin{tablenotes}'
    r'\footnotesize\raggedright'
    rf'注：样本为 2025 预测期（特征年 $t=2024$）完整案例，民营企业 $n={n_priv_2025}$ 家、SOE $n={n_soe_2025}$ 家；'
    rf'民营企业训练样本为全年份共 {n_priv_all_fy:,} 个 firm-year、{n_priv_all_firms:,} 家。'
    r'各特征的双样本标准化均值差（SMD）与共同支撑域诊断见\textcolor{black}{表~\ref{tab:cs}}。'
    r'\end{tablenotes}'
)
lines.append(r'\end{table}')

tex_table = '\n'.join(lines) + '\n'
(OUTPUT_DIR / 'descriptives_by_ownership.tex').write_text(tex_table, encoding='utf-8')
print(f'\n✓ descriptives_by_ownership.csv / .tex saved to {OUTPUT_DIR}')
