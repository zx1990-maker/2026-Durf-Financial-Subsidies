# QA — RAW_RAW 主规格重算（Task C）

生成日期：2026-08-20。本文件逐项确认 `raw_main_run.py` 是否符合任务约束，只客观报告数字，不重释经济机制、不改论文正文。

## QA1：GBM 训练 target 为 raw EBI/A，SOE actual 为 raw EBI/A

**PASS。** `raw_main_run.py` 中 `yt_raw = prep['tv']['EBI_A'].values`（未 clip），`raw_actual = pd2['EBI_A'].values`；特征仍为 9 个 X 的 P5/P95 feature-winsorization。

## QA2：GBM All SOE 多年 gap 与 outcome-spec 审计 RAW_RAW 完全一致

- 本任务 All SOE GBM RAW 多年 gap = **1.4497 pp**；审计 = **1.4497 pp** [1.2498, 1.6365]，点估计差 = 0.000000 pp（应 < 1e-3，四舍五入精度内）。

- → **PASS**

## QA3：Ridge / Random Forest 亦在 raw target 上训练

**PASS。** 三个模型共用 `prepare_train` 生成的特征矩阵，仅以 `yt_raw` 为标签拟合（`build("ridge")`、`build("random_forest")`、`build("gbm")` 均 `fit(prep["Xt"], yt_raw)`），无一使用 winsorized target。

## QA4：X / 特征预处理 / SOE 观测 / ForecastYear / 超参数完全冻结

**PASS。** 9 特征与 `feats_in` 断言一致；P5/P95 feature-winsorize、median impute、StandardScaler、one-hot industry_l2、`target_year < ForecastYear` 严格 ex-ante、GBM(n_estimators=200,max_depth=4,lr=0.05,min_samples_leaf=10)、Ridge(alpha=10)、RF(n_estimators=300,max_depth=8,min_samples_leaf=10,max_features=sqrt)、seed=42 均未改动。

## QA5：SOE firm-year N 与既有结果一致

- RAW master panel = **9,078** 个 SOE firm-year（= 9,078），与 outcome-spec 审计及 `soe_support_firmyear_v2.csv`（9,078）一致；N_firms=1,351 亦一致。

## QA6：v2 common-support 分类未重算，直接复用

- 将 RAW gap 按 (firm_id, ForecastYear) one-to-one merge 到 `soe_support_firmyear_v2.csv`，匹配 **9,078 / 9,078** 行，support_status 全部非空（9,078 行）。support 距离/分类未重算，直接用 v2 的 5-NN-vs-5-NN 分类。

- → **PASS**

## QA7：OOF 性能（raw target，pooled）

- GBM raw OOF R² = **0.2386**（审计 QA7 报告 0.23862，一致）；Ridge = 0.1720，RF = 0.2064。

## QA8：民营 OOF placebo（raw target）

- 民营 OOF mean residual = **0.0000** [CI -0.0014, 0.0014]，p = 0.9792，与 0 无差异 → **PASS**（模型对民营无系统性偏差）。

## QA9：模型替换稳健性（All SOE 多年 Full gap，pp）

- GBM：Full **+1.45** [+1.25, +1.64]；CS +1.42；PersistentGood +1.41
- Ridge：Full **+0.97** [+0.74, +1.20]；CS +0.95；PersistentGood +1.06
- RF：Full **+1.70** [+1.50, +1.91]；CS +1.74；PersistentGood +1.84
- 三模型方向一致为正、CI 均不含 0（除 Ridge 数值较低），结论对模型选择稳健。

## QA10：中央 vs 地方（pooled 2019–2025）

- 中央 +0.0140 vs 地方 +0.0147，差 -0.0011 [CI -0.0049, +0.0027]，不显著。

## QA11：金额缺口（2025 All SOE）

- 总金额缺口 6,826.3 亿元；外推区金额占比 16.4%（CS 内 5,704.4 亿元）。（金额分解对 support 口径敏感，见 support_v2 结论。）

## QA12：outcome-definition 稳健性（All SOE 多年，pp）

- CURRENT +2.0164 / RAW_RAW +1.4497 / WIN_WINSYM +1.5016；三者方向一致为正且显著，量级对 outcome 口径敏感约 0.5 pp。


## QA 结论

- **PASS**：RAW target 定义、Gap 复现、v2 支撑复用、三模型 RAW 训练、样本/特征/超参数冻结、N 一致、placebo 无偏，全部通过。

- 主结论：RAW_RAW 下 All SOE 多年反事实缺口约 **+1.45 pp**（GBM；Ridge +0.97、RF +1.70），共同支撑（+1.42）与 persistent-good（+1.41）均显著为正；方向与旧口径一致，量级较 CURRENT（+2.02）降低约 0.5 pp。

- 判断：**PASS**（主结论方向稳健；量级对 outcome 口径敏感，已在 outcome_definition_robustness_final.csv 客观呈现）。
