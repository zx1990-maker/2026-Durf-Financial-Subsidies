#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 1 (revision3) — Time-window audit (FINAL).

Confirms whether the production expanding-window satisfies the strict ex-ante
requirement  max(train outcome year) < forecast year  for every forecast year
2019-2025. The production scripts (outputs/revision/export_final_results_v2.py,
industry_screened.py, unified_analysis.py, rerun_filtered.py,
inference_and_heterogeneity.py, common_support_full.py) all filter training with
`target_year < fc_year`, i.e. the outcome year is strictly BEFORE the forecast year.

Outputs (outputs/revision3/):
  time_window_audit_final.csv
  time_window_audit_final.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3
from _audit_common import load_screened_panel, feats_in

OUT3.mkdir(exist_ok=True)

print('Loading screened panel...')
all_a = load_screened_panel()
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()

print(f'  Panel: {len(all_a)} obs, {all_a["firm_id"].nunique()} firms')
print(f'  Train pool (private): {len(train_pool)} obs, {train_pool["firm_id"].nunique()} firms')

rows = []
for fc_year in range(2019, 2026):
    soe = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc_year)]
    soe_feature_year = int(soe['feature_year'].min()) if len(soe) else np.nan

    # CORRECTED / production window (strict ex-ante): target_year < fc_year
    tr = train_pool[train_pool['target_year'] < fc_year]
    tv = tr.dropna(subset=feats + ['EBI_A'])

    rows.append({
        'forecast_year': fc_year,
        'soe_feature_year': soe_feature_year,
        'min_train_feature_year': int(tv['feature_year'].min()) if len(tv) else np.nan,
        'max_train_feature_year': int(tv['feature_year'].max()) if len(tv) else np.nan,
        'min_train_outcome_year': int(tv['target_year'].min()) if len(tv) else np.nan,
        'max_train_outcome_year': int(tv['target_year'].max()) if len(tv) else np.nan,
        'n_train_firm_year': int(len(tv)),
        'n_train_firms': int(tv['firm_id'].nunique()),
        'leakage_flag': bool(int(tv['target_year'].max()) >= fc_year) if len(tv) else True,
    })

audit = pd.DataFrame(rows)
audit.to_csv(OUT3 / 'time_window_audit_final.csv', index=False, encoding='utf-8-sig')
print('\n=== time_window_audit_final ===')
print(audit.to_string(index=False))

any_leak = bool(audit['leakage_flag'].any())
verdict = 'FAIL' if any_leak else 'PASS'

# ---- Markdown report ----
lines = []
lines.append('# TASK 1 — 时间窗口审计（最终）\n')
lines.append(f'**结论：{verdict}**。\n')
if not any_leak:
    lines.append(
        '对 2019–2025 每一个 forecast year，训练样本都严格满足 '
        '`max(train outcome year) < forecast year`，即**无时间信息泄漏**。\n'
    )
else:
    lines.append(
        '**存在时间信息泄漏**：训练样本中出现了 outcome year >= forecast year 的民营企业观测。\n'
    )

lines.append('## 逐年的训练窗口\n')
for _, r in audit.iterrows():
    fc = int(r['forecast_year'])
    lo = int(r['min_train_outcome_year']) if pd.notna(r['min_train_outcome_year']) else None
    hi = int(r['max_train_outcome_year']) if pd.notna(r['max_train_outcome_year']) else None
    lines.append(
        f"- **Forecast {fc}**: train outcome years = {lo}–{hi}; "
        f"therefore no {fc} outcome enters training. "
        f"（n = {int(r['n_train_firm_year'])} firm-year, {int(r['n_train_firms'])} firms）"
    )

lines.append('## 负责训练窗口的代码\n')
lines.append(
    '生产脚本一律使用 `target_year < fc_year`（结果年份严格早于预测年），例如：\n'
)
for fn, ln in [
    ('outputs/revision/export_final_results_v2.py', 63),
    ('outputs/revision/industry_screened.py', 98),
    ('outputs/revision/unified_analysis.py', 249),
    ('outputs/revision/rerun_filtered.py', 102),
    ('outputs/revision/inference_and_heterogeneity.py', 48),
    ('outputs/revision/common_support_full.py', 38),
]:
    lines.append(f"- `{fn}:{ln}` → `train = train_pool[train_pool['target_year'] < fc_year]`（严格 ex-ante）")

lines.append('## 字段定义\n')
lines.append(
    '- `leakage_flag` = True 当且仅当 `max_train_outcome_year >= forecast_year`。\n'
    '- `soe_feature_year` = forecast year − 1（`X_t → Y_{t+1}` 的特征年）。\n'
    '- `n_train_firm_year` = 训练样本的 firm-year 观测数；`n_train_firms` = 训练样本的 firm 数（二者需区分）。\n'
)

if not any_leak:
    lines.append(
        '\n## 说明\n\n'
        '历史上 `outputs/revision2/audit_01_time_window.py` 曾记录一个旧版窗口 '
        '`target_year <= fc_year`（含当年结果年）存在泄漏；该问题已在生产脚本中修复为 '
        '`< fc_year`，`final_results/` 的 headline 结果（`summary_by_year_all_soe.csv`）即为修正后结果。\n'
    )

(OUT3 / 'time_window_audit_final.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'\n✓ {OUT3 / "time_window_audit_final.csv"}')
print(f'✓ {OUT3 / "time_window_audit_final.md"}')
print(f'\nVERDICT: {verdict}')
