#!/usr/bin/env python3
"""Audit the amount-gap export for calculation correctness."""
import numpy as np, pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
OUT = PROJECT / "final_results"
df = pd.read_csv(OUT/'firm_level_gap_all_years.csv')

print('='*75)
print('  金额 Gap 数据出具审计')
print('='*75)
print(f'总行数: {len(df)}, 年份: {sorted(df["year"].unique())}')
print(f'样本: {list(df["sample"].unique())}')
print(f'列: {list(df.columns)}')
print()

issues = []
passes = []

# ============================================================================
# 1. 逐行公式验证：gap_ratio = predicted - actual
# ============================================================================
diff1 = (df['predicted_ebia'] - df['actual_ebia']) - df['gap_ratio']
max_err1 = np.abs(diff1).max()
if max_err1 < 1e-12:
    passes.append(f'✓ gap_ratio = predicted_ebia - actual_ebia（最大误差 {max_err1:.2e}）')
else:
    issues.append(f'✗ gap_ratio 公式不一致，最大误差 {max_err1:.2e}')

# ============================================================================
# 2. 逐行公式验证：gap_amount_yuan = gap_ratio × total_assets
# ============================================================================
diff2 = (df['gap_ratio'] * df['total_assets_target_yuan']) - df['gap_amount_yuan']
rel_err2 = (np.abs(diff2) / df['gap_amount_yuan'].abs().clip(lower=1e-10)).max()
if rel_err2 < 1e-6:
    passes.append(f'✓ gap_amount_yuan = gap_ratio × total_assets（相对误差 {rel_err2:.2e}）')
else:
    issues.append(f'✗ 金额 gap 公式不一致，相对误差 {rel_err2:.2e}')

# ============================================================================
# 3. 单位换算：gap_amount_yi = gap_amount_yuan / 1e8
# ============================================================================
diff3 = (df['gap_amount_yuan'] / 1e8) - df['gap_amount_yi']
max_err3 = np.abs(diff3).max()
if max_err3 < 1e-6:
    passes.append(f'✓ gap_amount_yi = gap_amount_yuan / 1e8（亿元换算正确）')
else:
    issues.append(f'✗ 亿元换算错误，最大误差 {max_err3:.2e}')

# ============================================================================
# 4. 唯一性：每个 (sample, year, firm_id) 唯一
# ============================================================================
dup = df.duplicated(subset=['sample','year','firm_id']).sum()
if dup == 0:
    passes.append('✓ 每个 (sample, year, firm_id) 唯一，无重复')
else:
    issues.append(f'✗ 有 {dup} 个重复的 (sample, year, firm_id)')

# ============================================================================
# 5. 资产总计 > 0
# ============================================================================
nonpos = (df['total_assets_target_yuan'] <= 0).sum()
if nonpos == 0:
    passes.append('✓ 所有 total_assets_target > 0')
else:
    issues.append(f'✗ {nonpos} 行 total_assets <= 0')

# ============================================================================
# 6. 比率范围检查：EBI/A 应在 (-0.5, 0.5)（筛选了 |EBI/A|<=0.5）
# ============================================================================
actual_range = (df['actual_ebia'].min(), df['actual_ebia'].max())
if actual_range[0] > -0.5 and actual_range[1] < 0.5:
    passes.append(f'✓ actual_ebia 范围 {actual_range[0]:.3f} ~ {actual_range[1]:.3f}（在筛选范围）')
else:
    issues.append(f'✗ actual_ebia 超出筛选范围: {actual_range}')

# ============================================================================
# 7. 样本重叠检查：All SOE 是否包含 CSI 子集（导致重复计算风险）
# ============================================================================
print()
print('-'*75)
print('  样本重叠分析（重要）')
print('-'*75)
for year in [2025]:
    all_ids = set(df[(df['sample']=='All A-share SOE')&(df['year']==year)]['firm_id'])
    c300_ids = set(df[(df['sample']=='CSI300 SOE')&(df['year']==year)]['firm_id'])
    c500_ids = set(df[(df['sample']=='CSI500 SOE')&(df['year']==year)]['firm_id'])
    c1000_ids = set(df[(df['sample']=='CSI1000 SOE')&(df['year']==year)]['firm_id'])
    print(f'  {year} 年:')
    print(f'    All SOE: {len(all_ids)} firms')
    print(f'    CSI300: {len(c300_ids)}, CSI500: {len(c500_ids)}, CSI1000: {len(c1000_ids)}')
    overlap_300 = len(all_ids & c300_ids)
    overlap_500 = len(all_ids & c500_ids)
    overlap_1000 = len(all_ids & c1000_ids)
    print(f'    All ∩ CSI300: {overlap_300}（{overlap_300/len(c300_ids)*100:.0f}% 的 CSI300 在 All 中）')
    print(f'    All ∩ CSI500: {overlap_500}')
    print(f'    All ∩ CSI1000: {overlap_1000}')
    # CSI 之间是否重叠（一个 firm 可能同时在 CSI300 和 CSI500？理论上不会）
    overlap_300_500 = len(c300_ids & c500_ids)
    print(f'    CSI300 ∩ CSI500: {overlap_300_500}（应为 0，指数互斥）')
    print()
    if overlap_300 > 0:
        print(f'  ⚠️ 重要：All SOE 包含 CSI300/500/1000 子集。')
        print(f'     若直接加总 4 个样本的金额 gap 会重复计算。')
        print(f'     "All SOE" 才是全市场总额，CSI 是子集分解。')

# ============================================================================
# 8. 列名语义检查
# ============================================================================
print()
print('-'*75)
print('  列名语义检查')
print('-'*75)
print('  ⚠️ "actual_ebia" 和 "predicted_ebia" 列实际是 EBI/A 比率（无量纲），不是金额。')
print('     建议改名为 actual_ebia_ratio / predicted_ebia_ratio 以避免混淆。')
print('     gap_ratio = predicted_ebia - actual_ebia（比率差）')
print('     gap_amount_yuan = gap_ratio × total_assets（金额，元）')

# ============================================================================
# 9. 随机抽查 20 行，用原始数据重算
# ============================================================================
print()
print('-'*75)
print('  随机抽查（20 行，从原始数据重算）')
print('-'*75)
np.random.seed(42)
sample_idx = np.random.choice(len(df), 20, replace=False)
err_count = 0
for i in sample_idx:
    row = df.iloc[i]
    recalc_amount = row['gap_ratio'] * row['total_assets_target_yuan']
    recalc_gap = row['predicted_ebia'] - row['actual_ebia']
    if abs(recalc_amount - row['gap_amount_yuan']) > 1e-6 or abs(recalc_gap - row['gap_ratio']) > 1e-12:
        err_count += 1
        print(f'  ✗ {row["firm_id"]} {int(row["year"])}: 重算不一致')
if err_count == 0:
    print(f'  ✓ 20 行抽查全部一致')

# ============================================================================
# 汇总
# ============================================================================
print()
print('='*75)
print('  审计结论')
print('='*75)
for p in passes:
    print(f'  {p}')
for i in issues:
    print(f'  {i}')
print()
if len(issues) == 0:
    print('  ✅ 金额 gap 计算过程正确，无计算错误。')
    print('  ⚠️ 注意：All SOE 与 CSI 子集重叠，加总需谨慎；列名建议优化。')
else:
    print(f'  ❌ 发现 {len(issues)} 个问题，需修正。')
