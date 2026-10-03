#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision3 — Insert the two appendix tables (dynamic common support + amount-gap
decomposition) into the paper, add the new figures/files to the appendix lists, and
rename the appendix subsections so the new "补充表格" becomes 附录 B.

Reads table_dynamic_common_support.csv and table_amount_gap_support.csv (already
produced by task10) and generates LaTeX tables directly from them (no transcription).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3

PAPER = Path('/Users/violet/Desktop/DURF/模型/Linear-A/论文_中国国有企业反事实资产回报率研究.tex')

dyn = pd.read_csv(OUT3 / 'table_dynamic_common_support.csv')
amt = pd.read_csv(OUT3 / 'table_amount_gap_support.csv')

SAMPLE_CN = {
    'All A-share SOE': 'All SOE',
    'CSI300': '沪深300',
    'CSI500': '中证500',
    'CSI1000': '中证1000',
}


def esc(x):
    return str(x).replace('_', r'\_')


# ---------------- Appendix Table A1: dynamic common support ----------------
lines1 = []
for _, r in dyn.iterrows():
    lines1.append(
        f"{int(r['year'])} & {SAMPLE_CN[r['sample']]} & "
        f"{r['good_support_pct']:.1f} & {r['boundary_pct']:.1f} & {r['extrapolation_pct']:.1f} & "
        f"{r['roc_auc']:.3f} & {r['firmsize_smd']:.2f} & "
        f"{r['full_gap_pp']:+.2f} & {r['cs_gap_pp']:+.2f} & {int(r['n'])} \\\\"
    )
body1 = '\n'.join(lines1)

table1 = r"""\begin{table}[H]
\centering
\small
\caption{表 A1　逐年动态共同支撑域（2019--2025；gap 单位 pp）}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrr}
\toprule
年份 & 样本 & 良好支持\% & 边界\% & 外推\% & ROC-AUC & FirmSize SMD & Full gap & CS gap & n \\
\midrule
""" + body1 + r"""
\bottomrule
\end{tabular}}
\end{table}"""

# ---------------- Appendix Table A2: amount-gap decomposition ----------------
lines2 = []
for _, r in amt.iterrows():
    lines2.append(
        f"{int(r['year'])} & {SAMPLE_CN[r['sample']]} & "
        f"{r['total_amount_gap_yi']:.0f} & {r['good_amount_gap_yi']:.0f} & "
        f"{r['boundary_amount_gap_yi']:.0f} & {r['extrapolation_amount_gap_yi']:.0f} & "
        f"{r['extrapolation_amount_share'] * 100:.1f} & {r['extrapolation_asset_share'] * 100:.1f} & "
        f"{int(r['n_firms'])} \\\\"
    )
body2 = '\n'.join(lines2)

table2 = r"""\begin{table}[H]
\centering
\small
\caption{表 A2　逐年金额缺口按共同支撑分解（2019--2025；亿元）}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrr}
\toprule
年份 & 样本 & 总金额缺口 & good & boundary & 外推 & 外推金额占比\% & 外推资产占比\% & n \\
\midrule
""" + body2 + r"""
\bottomrule
\end{tabular}}
\end{table}"""

section = (
    '\\subsection{附录 B　补充表格}\n\n'
    '表 A1 与表 A2 分别给出 2019--2025 年逐年、逐样本的动态共同支撑域统计与金额缺口'
    '按共同支撑状态的分解；完整数据见附录 C 中的两个 CSV 文件。\n\n'
    + table1 + '\n\n' + table2 + '\n\n'
)

# ---------------- apply to paper ----------------
paper = PAPER.read_text(encoding='utf-8')

# 1. rename existing 附录 B -> 附录 C, 附录 C -> 附录 D
paper = paper.replace(
    '\\subsection{附录 B　核心结果文件}',
    '\\subsection{附录 C　核心结果文件}',
)
paper = paper.replace(
    '\\subsection{附录 C　复现代码}',
    '\\subsection{附录 D　复现代码}',
)

# 2. insert the new 附录 B before 附录 C
marker = '\\subsection{附录 C　核心结果文件}'
assert paper.count(marker) == 1, '附录 C marker not unique'
paper = paper.replace(marker, section + marker, 1)

# 3. add new CSVs to 附录 C (核心结果文件) list
old_file_row = '\\texttt{final\\_results/common\\_support\\_report.md} & 共同支撑域完整诊断 \\\\'
new_file_row = old_file_row + '\n'
new_file_row += '\\texttt{outputs/revision3/table\\_multiyear\\_common\\_support\\_inference.csv} & 多年共同支撑 bootstrap 推断（表 9） \\\\\n'
new_file_row += '\\texttt{outputs/revision3/table\\_industry\\_knn\\_multiyear.csv} & 同行业 KNN 多年平均（表 12） \\\\\n'
new_file_row += '\\texttt{outputs/revision3/table\\_dynamic\\_common\\_support.csv} & 逐年动态共同支撑（表 A1） \\\\\n'
new_file_row += '\\texttt{outputs/revision3/table\\_amount\\_gap\\_support.csv} & 逐年金额缺口按支撑分解（表 A2） \\\\'
assert paper.count(old_file_row) == 1, 'file-list anchor not unique'
paper = paper.replace(old_file_row, new_file_row, 1)

# 4. add new scripts to 附录 D (复现代码) list
old_code_row = '\\texttt{outputs/revision/fig\\_firm\\_size\\_overlap\\_screened.py} & 企业规模重叠图（筛选后，新增） \\\\'
new_code_row = old_code_row + '\n'
new_code_row += '\\texttt{outputs/revision3/task08\\_multiyear\\_cs\\_inference.py} & 多年共同支撑 cluster bootstrap \\\\\n'
new_code_row += '\\texttt{outputs/revision3/task09\\_industry\\_knn\\_multiyear.py} & 同行业 KNN 多年平均 \\\\\n'
new_code_row += '\\texttt{outputs/revision3/task10\\_export\\_final\\_tables.py} & 导出附录 CSV 表 \\\\\n'
new_code_row += '\\texttt{outputs/revision3/task11\\_make\\_figures.py} & 生成图 6/图 7 \\\\'
assert paper.count(old_code_row) == 1, 'code-list anchor not unique'
paper = paper.replace(old_code_row, new_code_row, 1)

PAPER.write_text(paper, encoding='utf-8')
print('✓ paper updated (appendix tables + lists)')
print(f'  paper lines now: {paper.count(chr(10))}')
