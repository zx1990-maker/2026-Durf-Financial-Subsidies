# Outcome / Target Definition Audit（机器学习 EBI/A 口径审计）

生成日期：2026-08-20。本文件只审计现有代码中 target / prediction / actual / gap 的真实定义，并给出关键代码位置；不改动任何代码、不重释经济机制。

**审计对象**：当前论文 GBM firm-year 预测与 gap 的真实生成代码。经确认，共有两条代码路径，二者口径一致：

1. 生产版（`final_results/`）：`run_counterfactual.py` → `winsorize_features_and_target()` + `run_model()`。
2. revision3 审计版（论文表 9/表 10 的多年数字来源）：`outputs/revision3/_audit3.py` → `compute_long_panel()`，其底层复用 `outputs/revision2/_audit_common.py` 的 `prepare_train()` / `predict_gap()`。

两条路径对 target / actual / gap 的处理**完全一致**（`_audit_common.py` 自述为「生产版流水线的逐位复现」）。

---

## Q1. 当前 GBM 的训练 target 是什么？

**答：Winsorized EBI/A（P5/P95 截断后的 EBI/A），不是 raw EBI/A。**

- revision3 版：`_audit_common.py:171-172` 计算 `yl,yh = tv['EBI_A'].quantile(0.05/0.95)` 并生成 `tv['EBI_A_w'] = tv['EBI_A'].clip(yl, yh)`；`:179` `yt = tv['EBI_A_w'].values` 作为训练标签。
- 生产版：`run_counterfactual.py:662-664` 生成 `train['EBI_A_winsor'] = train['EBI_A'].clip(y_lo, y_hi)`；`:740` `y_train = train_valid['EBI_A_winsor'].values`。

**关键代码位置**：
- `outputs/revision2/_audit_common.py:162-182`（`prepare_train`，其中 171-172、179）
- `run_counterfactual.py:650-683`（`winsorize_features_and_target`）、`run_counterfactual.py:732-740`（`run_model`）

---

## Q2. 如果 target 被 Winsorize，阈值如何计算？

**答**：

- **阈值**：P5 / P95（`quantile(0.05)`、`quantile(0.95)`）。
- **是否逐年动态**：**逐年动态**。`compute_long_panel`（`_audit3.py:90-92`）在 `for fc_year` 循环内对每个 ForecastYear 单独调用 `prepare_train(tr, feats)`，`tr = train_pool[train_pool['target_year'] < fc_year]`，故 P5/P95 每年按当年历史民企训练池单独计算。
- **使用哪些 observations**：`tv = train_df.dropna(subset=feats + ['EBI_A'])`（`_audit_common.py:165`），即当年历史民企训练池中 9 特征与 EBI/A 均非缺失的观测。
- **是否只使用 OutcomeYear < ForecastYear 的历史民企训练样本**：**是**。`_audit3.py:91` `tr = train_pool[train_pool['target_year'] < fc_year]`，严格 ex-ante（`target_year < fc_year`）。
- **是否存在使用 ForecastYear 当年或未来 outcome 的情况**：**否**。`target_year < fc_year` 排除了当年与未来 outcome；SOE outcome 未进入阈值计算（阈值只来自 `is_private` 训练池）。

---

## Q3. 当前 `GBM_predicted_EBIA` 对应什么 target？

**答：模型预测的是 E[Y^winsorized | X]，即被 Winsorize 后的 EBI/A 的条件期望，而非 E[Y^raw | X]。**

- 因为模型在 `EBI_A_w`（winsorized）上训练（见 Q1）。
- 生产版 `run_counterfactual.py:829` `det['predicted_EBI_A'] = preds`，其中 `preds = model.predict(X_pred)`（`:799`），`model` 拟合于 `y_train = EBI_A_winsor`。
- revision3 版中 `gap = preds - actuals`，`preds` 来自拟合于 `EBI_A_w` 的模型；故 `predicted_EBI_A = pred_w`（winsorized-target 预测值）。

---

## Q4. 当前 `actual_EBIA` 和 `gap_pp` 使用什么？

**答**：

- **SOE actual 未经 Winsorize**：`_audit_common.py:197` `actuals = pd2['EBI_A'].values`（raw EBI/A）；生产版 `run_counterfactual.py:761` `y_pred_actual = pred_valid['EBI_A'].values`、`:830` `det['actual_EBI_A'] = det['EBI_A'].values`。
- **gap 的真实计算公式**：`gap = predicted − actual = pred(E[Y^w|X]) − Y^raw`。`_audit_common.py:198` `gaps = preds - actuals`；生产版 `run_counterfactual.py:802` `gap = preds - y_pred_actual`。
- **是否存在 prediction target 与 actual outcome 口径不一致**：**存在**。模型在 winsorized EBI/A 上训练（预测 winsorized 值），但 gap 用 **raw SOE EBI/A** 作 actual 比较，即

  $$
  \text{gap}_{i,T} = \widehat{Y}^{winsorized}_{SOE,i,T} - Y^{raw}_{SOE,i,T}.
  $$

  生产版代码甚至在 `winsorize_features_and_target` 中已生成 `pred['EBI_A_winsor']`（`run_counterfactual.py:665`），但 `run_model` 计算 gap 时**未使用**该列，而是用了 raw `EBI_A`（`:761`），证实这是一处真实的、而非仅概念上的口径不一致。

---

## 结论（审计）

当前主规格（论文约 +2.02 pp 的 All-SOE 多年 gap）在 outcome 侧的定义为：

> **模型 target = winsorized EBI/A（私企 P5/P95、逐年、ex-ante）；SOE actual = raw EBI/A；gap = pred_w − raw。**

本审计不评价该口径的优劣，只客观记录。三个规格（CURRENT_REPLICATION / RAW_RAW / WIN_WINSYM）的对比见 `outcome_spec_*.csv` 与 `QA_outcome_spec.md`。
