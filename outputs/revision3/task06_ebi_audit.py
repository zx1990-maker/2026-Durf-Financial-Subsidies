#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 6 (revision3) — EBI accounting-field audit.

Scans the raw Wind exports (0803/A_*_origin.xlsx) for the profit / interest / tax /
minority-interest fields actually present, and documents which field the production
pipeline uses for the EBI/A numerator.

Outputs (outputs/revision3/):
  ebi_accounting_field_audit.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from run_counterfactual import parse_xlsx_xml, extract_header_info

from _audit3 import OUT3  # noqa: E402  (after sys.path insert above)

OUT3.mkdir(exist_ok=True)

files = sorted(Path('0803').glob('A_*_origin.xlsx'))
field_files = {}
for fp in files:
    headers, _, _ = parse_xlsx_xml(fp)
    for col_letter, htext in headers.items():
        fname, _yr = extract_header_info(htext)
        field_files.setdefault(fname, set()).add(fp.name[:6])

all_fields = sorted(field_files)
profit_fields = [f for f in all_fields if any(
    k in f for k in ['净利润', '净利', '利润', '利息', '损益', '所得税', '少数', '股东', '营业利润', '利润总额'])]

has_consolidated_ni = any(('净利润' in f and '归属母公司' not in f and '归母' not in f) or '合并' in f for f in all_fields)
has_minority = any('少数股东' in f for f in all_fields)
has_parent_ni = any('归属母公司股东的净利润' in f or '归母净利润' in f for f in all_fields)

lines = ['# TASK 6 — EBI 会计口径审计\n']

lines.append('## 1. 当前使用的字段\n')
lines.append('- **原始 Wind 字段名**：`归属母公司股东的净利润`（net profit attributable to parent）。\n')
lines.append('- **Python dataframe 列名**：`run_counterfactual.py` 中 `classify_column()` 将其映射为语义类型 `net_profit`；`records_to_dataframe()` 用其与 `interest_expense`、`total_assets` 计算 `EBI_A`。\n')
lines.append('- **EBI/A 构造公式**：`EBI_A = (net_profit[t+1] + interest_expense[t+1]) / total_assets[t+1]`。\n')
lines.append('- **数据来源**：`0803/A_{2014..2024}_origin.xlsx`（Wind 终端导出）。\n')
lines.append('- **使用该变量的文件/函数**：`run_counterfactual.py` → `classify_column()`（第 186 行）与 `records_to_dataframe()`（第 325 行）。\n')

lines.append('## 2. 原始数据字段清单（决定性）\n')
lines.append(f'对全部 {len(files)} 个 `A_*_origin.xlsx` 做逐列扫描，共 {len(all_fields)} 个不同字段：\n')
for f in all_fields:
    lines.append(f'- `{f}`（{len(field_files[f])} 个文件）')
lines.append('')
lines.append('其中与利润/利息/税/少数股东相关的字段：\n')
if profit_fields:
    for f in profit_fields:
        lines.append(f'- `{f}`')
else:
    lines.append('- （无）')

lines.append('\n## 3. 是否存在 consolidated net income\n')
yn = lambda b: '存在' if b else '不存在'
lines.append(f'- **合并净利润 / 净利润**：**{yn(has_consolidated_ni)}**。')
lines.append(f'- **少数股东损益**：**{yn(has_minority)}**。')
lines.append(f'- **归属母公司股东的净利润**：**{yn(has_parent_ni)}**（唯一被使用的利润字段）。\n')

lines.append('\n## 4. 结论\n')
if has_consolidated_ni:
    lines.append('存在 consolidated net income，本应可做 `EBI^consolidated` robustness。\n')
else:
    lines.append(
        '原始数据**只包含单一利润序列**（归属母公司股东的净利润），**不存在** 合并净利润/净利润、'
        '少数股东损益、利润总额、营业利润、所得税费用。因此：\n'
    )
    lines.append('> Consolidated NI robustness cannot be implemented with available data.\n')
    lines.append(
        '故本任务**不生成** `table_ebi_definition_robustness.csv`（数据可得性限制，而非分析选择）。'
        '主指标 `EBI/A`（归母净利润口径）保持不变。\n'
    )

lines.append('\n## 5. 口径含义（非 EBIT）\n')
lines.append(
    '分子为「税后、归母」净利润加利息支出，故**不是** EBIT（缺税加回、缺少数股东损益加回）。'
    '它是「(税后) 归母权益 + 有息债务」口径的资产回报率变体。跨组有效税率或少数股东结构差异会机械地进入 gap，'
    '但本数据无法量化该影响。\n'
)

(OUT3 / 'ebi_accounting_field_audit.md').write_text('\n'.join(lines), encoding='utf-8')

print('\n'.join(lines[:40]))
print(f'\n✓ {OUT3 / "ebi_accounting_field_audit.md"}')
