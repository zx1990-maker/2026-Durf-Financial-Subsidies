# TASK 1 — 时间窗口审计（最终）

**结论：PASS**。

对 2019–2025 每一个 forecast year，训练样本都严格满足 `max(train outcome year) < forecast year`，即**无时间信息泄漏**。

## 逐年的训练窗口

- **Forecast 2019**: train outcome years = 2016–2018; therefore no 2019 outcome enters training. （n = 5554 firm-year, 2160 firms）
- **Forecast 2020**: train outcome years = 2016–2019; therefore no 2020 outcome enters training. （n = 7704 firm-year, 2344 firms）
- **Forecast 2021**: train outcome years = 2016–2020; therefore no 2021 outcome enters training. （n = 10278 firm-year, 2728 firms）
- **Forecast 2022**: train outcome years = 2016–2021; therefore no 2022 outcome enters training. （n = 13258 firm-year, 3057 firms）
- **Forecast 2023**: train outcome years = 2016–2022; therefore no 2023 outcome enters training. （n = 16501 firm-year, 3266 firms）
- **Forecast 2024**: train outcome years = 2016–2023; therefore no 2024 outcome enters training. （n = 19827 firm-year, 3341 firms）
- **Forecast 2025**: train outcome years = 2016–2024; therefore no 2025 outcome enters training. （n = 23150 firm-year, 3344 firms）
## 负责训练窗口的代码

生产脚本一律使用 `target_year < fc_year`（结果年份严格早于预测年），例如：

- `outputs/revision/export_final_results_v2.py:63` → `train = train_pool[train_pool['target_year'] < fc_year]`（严格 ex-ante）
- `outputs/revision/industry_screened.py:98` → `train = train_pool[train_pool['target_year'] < fc_year]`（严格 ex-ante）
- `outputs/revision/unified_analysis.py:249` → `train = train_pool[train_pool['target_year'] < fc_year]`（严格 ex-ante）
- `outputs/revision/rerun_filtered.py:102` → `train = train_pool[train_pool['target_year'] < fc_year]`（严格 ex-ante）
- `outputs/revision/inference_and_heterogeneity.py:48` → `train = train_pool[train_pool['target_year'] < fc_year]`（严格 ex-ante）
- `outputs/revision/common_support_full.py:38` → `train = train_pool[train_pool['target_year'] < fc_year]`（严格 ex-ante）
## 字段定义

- `leakage_flag` = True 当且仅当 `max_train_outcome_year >= forecast_year`。
- `soe_feature_year` = forecast year − 1（`X_t → Y_{t+1}` 的特征年）。
- `n_train_firm_year` = 训练样本的 firm-year 观测数；`n_train_firms` = 训练样本的 firm 数（二者需区分）。


## 说明

历史上 `outputs/revision2/audit_01_time_window.py` 曾记录一个旧版窗口 `target_year <= fc_year`（含当年结果年）存在泄漏；该问题已在生产脚本中修复为 `< fc_year`，`final_results/` 的 headline 结果（`summary_by_year_all_soe.csv`）即为修正后结果。
