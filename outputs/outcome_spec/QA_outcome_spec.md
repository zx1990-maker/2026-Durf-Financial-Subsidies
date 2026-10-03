# QA — Outcome-Definition Audit（三规格对比）

生成日期：2026-08-20。本文件逐项确认 `outcome_spec_run.py` 的三规格计算是否符合任务约束，并客观报告数字；不重释经济机制、不改论文正文。

---

## QA1：CURRENT_REPLICATION 是否复现论文当前 All SOE 2019–2025 ≈ 2.02 pp

**PASS。**

- All SOE 多年 annual-mean gap = **+2.0164 pp [1.815, 2.209]**，与论文表 9（task08 多年推断）的 +2.02 [1.82, 2.21] 逐位一致（含 bootstrap CI）。
- 与 `_master_long_panel.csv`（论文数字的 source of truth）逐年核对，**最大逐年绝对差异 = 2.7e-15 pp**（浮点精度，视为 0）。
- 四样本均复现：CSI300 +0.069、CSI500 +1.625、CSI1000 +1.884（与 task08 完全一致）。

## QA2：RAW_RAW 中 target 未做 outcome winsorization，SOE actual 为 raw EBI/A

**PASS。**

- 训练 target = `prep['tv']['EBI_A'].values`（raw，未 clip），代码 `outcome_spec_run.py` 中 `yt_raw = tv['EBI_A'].values`。
- SOE actual = `raw_actual = pd2['EBI_A'].values`（raw）。
- 特征（9 个 X）仍按现有 P5/P95 feature-winsorization 处理，未改动。

## QA3：WIN_WINSYM 中 private target 与 SOE actual 使用完全相同的 (L_T, U_T)，且阈值只来自 ex-ante 私企

**PASS。**

- `(L_T, U_T) = (tv['EBI_A'].quantile(0.05), tv['EBI_A'].quantile(0.95))`，只来自 `tr = train_pool[target_year < fc_year]`（严格 ex-ante 私企）。
- private target = `clip(EBI_A, L_T, U_T)`（即 `prep['yt']`）；SOE actual = `clip(raw SOE EBI/A, L_T, U_T)`，二者用同一阈值（脚本内 `assert` 已确认 `prep['yt'] == clip(tv.EBI_A, yl, yh)`）。
- 未用 SOE 分布、未用 ForecastYear 当年或未来 outcome 重算阈值。

## QA4：三规格的 X / 特征预处理 / SOE 观测 / ForecastYear / GBM 参数完全一致

**PASS。**

- 三个规格共用同一 `prepare_train(tr, feats)` 生成的特征预处理（9 特征 P5/P95 winsorize → median impute → StandardScaler → one-hot industry），仅 target 不同。
- 共用同一 SOE 观测（同一 `pd2`）、同一 `ForecastYear ∈ 2019–2025`、同一 GBM 参数（n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=10, random_state=42）。
- **Winsorization 只截断数值、不删除观测**：`clip()` 不减少行数；三规格 N 完全相等（见 QA5）。

## QA5：N_CURRENT == N_RAW == N_WIN 对每个 ForecastYear × Sample 成立

**PASS。**

- 对全部 7 年 × 4 样本 = 28 个组合逐一核对，三个规格的 N 全部相等，0 个不匹配。
- All SOE 各年 N：2019=1188, 2020=1241, 2021=1301, 2022=1331, 2023=1342, 2024=1339, 2025=1336（合计 9,078）。

## QA6：三规格 All SOE 多年 gap 与差值（单位 pp）

```
CURRENT:  +2.0164  [1.815, 2.209]
RAW_RAW:  +1.4497  [1.250, 1.637]
WIN_WINSYM: +1.5016  [1.360, 1.636]

RAW - CURRENT:  -0.5667 pp
WIN - CURRENT:  -0.5148 pp
RAW - WIN:      -0.0520 pp
```

（四样本完整值见 `outcome_spec_multiyear_summary.csv`。）

## QA7：RAW_RAW 相对 CURRENT 的 OOF 性能变化（pooled，96,272 私有 OOF 观测）

```
R2:   0.31006 → 0.23862   (Δ = -0.0714)
RMSE: 0.04234 → 0.06439   (Δ = +0.0221)
MAE:  0.03241 → 0.04098   (Δ = +0.0086)
Calibration slope: 0.99997 → 0.98049  (Δ = -0.0195)
```

即：改用 raw target 后，OOF R² 下降、RMSE/MAE 上升（raw target 含极端值、更难拟合），校准斜率略低于 1（更强的向均值收缩）；winsorized target 的 R²=0.310 部分来自截断本身。

## QA8：WIN_WINSYM 中 SOE actual 的 pooled clipping rate

**PASS。**

- All SOE 2019–2025 pooled：**总截断率 = 8.38%**（761 / 9,078）。
  - Lower-tail（raw < L_T）截断 686 个 = **7.56%**；
  - Upper-tail（raw > U_T）截断 75 个 = **0.83%**。
- 被截断观测的 raw actual 均值 ≈ **-0.071**（中位数 -0.069，即集中在极端亏损端）；其 CURRENT raw-gap 均值 ≈ **+11.3 pp**；其总资产占比 ≈ **2.9%**（以小规模 SOE 为主）。
- 逐年总截断率介于 6.36%–10.56% 之间（逐年明细见 `winsor_clipping_diagnostics.csv`）。

---

## QA 结论

- CURRENT 复现成功（QA1），三个规格在样本/特征/模型/时间切分上完全可比（QA4/QA5）。
- outcome 侧口径不一致确认为真实存在（见 `target_definition_audit.md` Q4），且对主结论数值有影响：**约 +2.02 pp → 一致口径下约 +1.45~1.50 pp**。
- 三个规格方向均为正且统计显著（CI 不含 0），但**量级对 outcome 处理敏感**（约 0.5 pp / ~25% 相对差异）。
