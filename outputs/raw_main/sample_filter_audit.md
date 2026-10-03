# 样本筛选审计（RAW_RAW 主规格，冻结设定未改动）

生成日期：2026-08-20。本文件只记录样本筛选的每一步 drop 计数，确认与本任务冻结设定一致；不修改原始数据、不修改任何现有结果。

## 筛选步骤（与论文最终代码 `_audit_common.load_screened_panel` 完全一致）

| 步骤 | 操作 | 剔除 firm-year | 剩余 firm-year |
|---|---|---|---|
| 0 | `load_all_a_shares()` + `clean_and_filter('A')`（含剔除金融行业、EBI/A 有效性、所有制回填） | — | 54,671 |
| 1 | 剔除 ST 公司（firm-level，名称含 ST） | 2,198 | 52,473 |
| 2 | 剔除 `FinancialLeverage > 1`（firm-year） | 140 | 52,333 |
| 3 | 剔除 `|EBI/A| > 0.5`（firm-year） | 129 | 52,204 |

## 最终样本构成

- 最终 firm-year：**52,204**（SOE 14,142 / 民营 32,979）。

- firm 数：SOE **1,351** 家 / 民营 **3,344** 家。

- SOE 预测样本（target_year 2019–2025）：**9,362** firm-year / 1,351 firms；其中 9 特征完整的 **9,078** 个 firm-year 进入预测，与 `soe_support_firmyear_v2.csv`（9,078）一致。


## 结论

三处筛选（ST、FinancialLeverage>1、|EBI/A|>0.5）与冻结设定一致，本任务未增删任何筛选条件。9 个 X 特征、P5/P95 feature-winsorization、median imputation、StandardScaler、二级行业 one-hot、严格 ex-ante expanding window、ForecastYear 2019–2025 均未改动。
