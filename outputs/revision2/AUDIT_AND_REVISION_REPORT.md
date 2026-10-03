# 代码审计、稳健性分析与增量修改 — 最终报告

**项目.** 中国国有企业反事实 EBI/A（息前利润/资产总计）研究
**审计范围.** `run_counterfactual.py` + `outputs/revision/` 生产脚本的代码正确性、稳健性与增量修改
**新结果目录.** `outputs/revision2/`（未修改原始数据、未覆盖既有结果）
**日期.** 2026-08-18

---

## 0. 总体结论（Overall Verdict）

| 维度 | 判定 |
|---|---|
| 总体 | **PASS WITH REVISION**（结论成立且被稳健性强化；但滚动窗口必须重估 + 两处披露性修订） |
| 必须重估（FAIL-REESTIMATE） | TASK 1 时间窗泄漏（`target_year <= fc_year` 纳入同年私有结果） |
| 需修订披露（PASS WITH REVISION） | TASK 7 EBI/A 定义标签；TASK 3/4 金额差距由共同支持外样本主导 |
| 全部通过（PASS） | TASK 2、5、6、8、9、10（稳健性/一致性） |

**一句话结论.** 审计发现一处真实的方法学缺陷（滚动训练窗含同年结果），但**该缺陷系统地 *低估*（attenuate）了核心结果**：改用严格 ex-ante 窗口后，各年 SOE 差距**不变号且普遍更大**（2025 年 +20.9%）。因此论文的 headline 结论（SOE 相对民营企业基准在 EBI/A 上表现更差）**不仅成立，而且被稳健性检验进一步强化**。没有发现任何会推翻结论的问题。

---

## 1. 任务级判定汇总

| 任务 | 主题 | 判定 | 关键结果 |
|---|---|---|---|
| TASK 0 | 项目结构 | PASS | 生产脚本 = `export_final_results_v2.py`；数据链完整映射 |
| TASK 1 | 时间窗泄漏 | **FAIL-REESTIMATE** | 7/7 年 `leakage_flag=True`；修正后差距更大 |
| TASK 2 | 私有 OOF 安慰剂 + 校准 | PASS | 私有残差 0.0001（p=0.91）→ 差距非机械伪影 |
| TASK 3 | 金额差距按共同支持分解 | PASS WITH REVISION | 2025 金额差距 64.5% 来自外推区 |
| TASK 4 | 全样本 vs 共同支持 | PASS | 比例差距稳健（2025：0.0159 vs 0.0153） |
| TASK 4A | 多年共同支持动态 | PASS | 支持状态稳定且粘性；外推区大企业主导金额 |
| TASK 5 | KNN 可比公司基准 | PASS | KNN 差距逐年为正，与 GBM 高相关 |
| TASK 6 | 行业内共同支持 | PASS | 差距不变（0.0149 vs 0.0159） |
| TASK 7 | EBI/A 定义审计 | PASS WITH REVISION | 归母净利润（税后、含少数股东后）非 EBIT |
| TASK 8 | 中央 vs 地方 | PASS | 差异小且基本不显著 |
| TASK 9 | GBM 聚类 bootstrap | PASS | 2025 差距 CI 不含 0 |
| TASK 10 | 输出一致性 | PASS | 0 处不一致；headline 金额核对无误 |

---

## 2. 详细发现

### 2.1 TASK 1 — 时间窗泄漏（唯一 FAIL-REESTIMATE）

**缺陷.** 生产脚本用 `train = train_pool[train_pool['target_year'] <= fc_year]` 构建训练集，把
**同年**（`target_year == fc_year`）的私有结果也纳入训练，违反严格 ex-ante 要求
`OutcomeYear_train < ForecastYear`。7 个预测年（2019–2025）**全部** `leakage_flag=True`。

**规模.** 2025 年训练集中同年私有观测 3,314 / 26,464 ≈ **12.5%**；随年份推进逐年增大。

**影响方向（关键）.** 泄漏使 headline 结果**偏小**。改用严格 ex-ante 窗口
（`target_year < fc_year`）后，All A-share SOE 的 GBM 差距：

| 预测年 | 原始（含同年） | 修正（严格 ex-ante） | 变化 |
|---|---|---|---|
| 2019 | 0.0190 | 0.0195 | +2.5% |
| 2020 | 0.0233 | 0.0247 | +6.4% |
| 2021 | 0.0155 | 0.0155 | +0.2% |
| 2022 | 0.0145 | 0.0203 | **+40.1%** |
| 2023 | 0.0135 | 0.0191 | **+40.8%** |
| 2024 | 0.0176 | 0.0229 | **+30.0%** |
| 2025 | 0.0159 | 0.0192 | **+20.9%** |

**结论.** 修正后差距在所有年份**不变号且 ≥ 原始值**，近年增幅尤为明显。该缺陷
*保守化*了结果，因此不推翻结论，反而强化之；但正式版本**必须**改用
`target_year < fc_year` 重估全部表格与图表。

**修正产物.** `time_window_audit.csv`（泄漏标记）、`time_window_original_vs_corrected.csv`（逐样本原始 vs 修正）。

---

### 2.2 TASK 2 — 私有 OOF 安慰剂与校准（PASS）

对**民营企业本身**用同一 GBM + GroupKFold（按 firm_id）做样本外预测，检验模型是否有偏：

- **安慰剂.** 私有 OOF 残差均值 = 0.0001（中位数 −0.0014），聚类标准误 0.0005，
  95% CI [−0.0010, +0.0011]，**p = 0.91**（n=26,464 obs，3,344 firms）。
  → 模型对私有企业无偏，SOE 的正差距**不是机械伪影**。
- **校准.** 实际 = −0.016 + 1.29×预测（R²=0.25）。斜率 >1、截距 <0 的轻微 miscalibration
  来自 winsorization + 无年份固定效应，但**该偏差同时作用于私有与 SOE 两组**，故对
  *差距* 的解释有效。

**产物.** `private_oof_placebo_summary.csv`、`private_oof_placebo_by_year.csv`、`private_oof_calibration.csv`、`fig_private_oof_calibration.png`。

---

### 2.3 TASK 3 + 4 — 金额差距分解与共同支持（PASS WITH REVISION）

2025 年金额差距按共同支持状态分解（`amount_gap_by_support_year.csv`）：

| 支持状态 | n | 金额差距（亿元） | 占比 |
|---|---|---|---|
| good（共同支持内） | 813 | 1,351.9 | 22.6% |
| boundary | 166 | 773.2 | 12.9% |
| **extrapolation（支持外）** | 357 | **3,854.0** | **64.5%** |

**发现.** 金额差距 headline（2025 年 5,979.1 亿元）约 **64.5%** 来自处于私有特征支持
**之外**（外推区）的 SOE 大企业。这些企业的 *比例* 差距并不大（外推区 mean gap 0.0149），
但因其资产规模巨大，`gap × 资产` 的金额被显著放大。

**比例差距稳健（TASK 4）.** 限制到共同支持后，比例差距几乎不变：

| 年 | 全样本 mean_gap | 共同支持 mean_gap | 差 |
|---|---|---|---|
| 2025 | 0.0159 | 0.0153 | −0.0006 |

**修订要求.** 金额差距（亿元）必须**伴随**共同支持分解一起报告，不得作为独立 headline；
比例差距（等权）是更稳健的核心指标。这印证论文已有的“沪深300 大企业共同支持弱”的警示。

**产物.** `amount_gap_by_support_year.csv`、`fig_amount_gap_support_decomposition.png`、`gap_full_vs_common_support.csv`。

---

### 2.4 TASK 4A — 多年共同支持动态（PASS）

- **逐年占比稳定.** good ≈ 60–67%，extrapolation ≈ 22–27%，七年无趋势性漂移。
- **状态粘性（转移矩阵，7,690 次转移）.** good→good 88.8%；extrapolation→extrapolation
  82.3%；boundary 最不稳定（38.1%→good，26.3%→extrapolation）。
- **企业持久性（n=1,351 firms）.** 574 家“始终在支持内”（share=1），236 家“从不在支持内”
  （share=0），中位企业在 5/7 年处于支持内。
- **关键.** “从不在支持内”的企业比例差距**最低**（多年均值 0.0098），但金额差距**最大**
  （多年年均 13.0 亿元，累计 3,073 亿元）——再次印证金额由支持外大企业主导。

**产物.** `common_support_by_year_2019_2025.csv`、`firm_persistent_support.csv`、`persistent_support_summary.csv`、`multiyear_common_support_gap.csv`、`common_support_transition_matrix.csv`、`multiyear_amount_gap_summary.csv`、`fig_dynamic_common_support_gap.png`、`fig_firm_support_share_distribution.png`。

---

### 2.5 TASK 5 — KNN 可比公司基准（PASS）

用透明的 5-NN / 10-NN 可比公司估计量替换 GBM（同一特征空间：winsorized 数值特征 +
industry_l2 one-hot），验证差距是否依赖 GBM：

| 预测年 | GBM gap | KNN5 gap | KNN10 gap |
|---|---|---|---|
| 2019 | 0.0190 | 0.0131 | 0.0136 |
| 2020 | 0.0233 | 0.0166 | 0.0178 |
| 2021 | 0.0155 | 0.0095 | 0.0108 |
| 2022 | 0.0145 | 0.0137 | 0.0140 |
| 2023 | 0.0135 | 0.0140 | 0.0141 |
| 2024 | 0.0176 | 0.0193 | 0.0187 |
| 2025 | 0.0159 | 0.0172 | 0.0168 |

KNN 差距**每年均为正**（0.0095–0.0193），与 GBM 差距高度相关（r=0.845 / 0.907）。
→ 差距**不是 GBM 特有伪影**，换用非参数匹配估计量结论不变。

**产物.** `knn_comparable_firm_level.csv`、`knn_comparable_summary_by_year.csv`、`fig_gbm_vs_knn5_gap.png`、`fig_gbm_vs_knn10_gap.png`。

---

### 2.6 TASK 6 — 行业内共同支持（PASS）

在**同一 industry_l2 内部**重算共同支持（要求 5-NN 邻居为同行业私有企业）：

- 2025 年 good 占比：行业内 63.9% **vs** 池化 60.9%（略微提高，因去除跨行业距离）；
  仅 2019 年有 3 家企业所在行业私有样本不足（<10）。
- 差距不变：2025 年 mean gap 全样本 0.0159 vs 行业内 good 0.0149。

→ 结果对同行业匹配稳健。

**产物.** `industry_restricted_common_support.csv`。

---

### 2.7 TASK 7 — EBI/A 定义审计（PASS WITH REVISION）

- **实际计算.** `EBI/A_{t+1} = (归母净利润_{t+1} + 利息支出_{t+1}) / 资产总计_{t+1}`。
  → 分子利润为**归母净利润**（税后、扣除少数股东损益后），**非 EBIT**，**非合并净利润**。
- **数据限制.** 原始数据仅含 `归属母公司股东的净利润`，**无** 合并净利润 / 所得税 / 少数股东
  损益字段，故“净利润 vs 归母净利润”的稳健性对比**无法运行**（不产出 `ebi_definition_robustness.csv`）。
- **两条方向性警示.** (1) 税后指标 → 若 SOE 有效税率更低，会*缩小*测得的 SOE 落后幅度
  （即报告差距可能是真实税前效率差距的**下界**）；(2) 归母口径排除少数股东 → 若 SOE 含更多
  少数股东权益的并表子公司，会*放大* SOE 落后幅度。

**修订要求.** 保留变量但**重新命名/标注**为“归母净利润加利息支出/总资产（税后）”，并披露
上述两条方向性 caveat。

**产物.** `ebi_variable_audit.md`。

---

### 2.8 TASK 8 — 中央 vs 地方（PASS）

中央与地方 SOE 差距之差（central − local）：

| 年 | central gap | local gap | diff | p |
|---|---|---|---|---|
| 2020 | 0.0186 | 0.0254 | −0.0068 | 0.037 |
| 2022 | 0.0097 | 0.0168 | −0.0070 | 0.021 |
| 2025 | 0.0156 | 0.0160 | −0.0004 | **0.88** |

除 2020、2022 年边际显著外，中央/地方差异**小且基本不显著**（2025 p=0.88）。n：2025 中央 436 + 地方 900。

**产物.** `central_local_difference_test.csv`、`fig_central_local_difference_by_year.png`。

---

### 2.9 TASK 9 — GBM 聚类 bootstrap（PASS）

2025 年 GBM 差距的企业级（聚类）bootstrap（200 次企业重抽样）：

| 样本 | n_firms | mean gap | bootstrap SE | 95% CI |
|---|---|---|---|---|
| All A-share SOE | 1,336 | 0.0159 | 0.0012 | [0.0135, 0.0183] |
| CSI1000 SOE | 343 | 0.0118 | 0.0020 | [0.0069, 0.0152] |

两个样本的 95% CI **均不含 0**（p<0.05）。analytic 聚类 SE 与 bootstrap SE 高度一致。

**产物.** `gbm_cluster_bootstrap_selected_samples.csv`。

---

### 2.10 TASK 10 — 输出一致性（PASS）

- `firm_level_gap_all_years.csv` → `summary_by_sample_year.csv`：**0** 处 n / mean_gap / amount 不一致。
- `firm_level_gap_2025.csv` == all-years[year==2025]：**0** 处不一致。
- n 一致：2025 All SOE = 1,336（summary = inference = central 436 + local 900）。
- headline 金额核对：2025 = **5,979.1 亿元 = 597.9 billion RMB**；累计 2019–2025 = **40,593.2 亿元 = 4.06 trillion RMB**。
- 注：2025 CSI300 SOE 比例差距为负（−0.0036，SOE 占优）但金额为正（+1,307 亿元）——等权 vs 资产加权口径差异，非错误。

**产物.** `output_consistency_check.md`。

---

## 3. “Results That Changed”（A/B/C/D 分类）

| 类别 | 定义 | 本审计涉及项 |
|---|---|---|
| **A**（推翻 headline 结论） | 符号/显著性反转 | **无** |
| **B**（量级变化，结论不变） | 量级明显变化但结论不变 | TASK 1 泄漏修正：SOE 差距系统性变大（2022/2023 +40%，2025 +21%），符号不变 |
| **C**（稳健性/解释性 nuance） | headline 不变，解释口径需调整 | TASK 3/4/4A：金额差距 64.5% 由支持外大企业主导；TASK 7：EBI/A 税后/归母口径 caveat |
| **D**（无变化/确认） | 结果不变 | TASK 2（安慰剂）、5（KNN）、6（行业）、8（央地）、9（bootstrap）、10（一致性） |

---

## 4. 修订建议（供论文修改）

1. **（必须）重估滚动窗口.** 将所有 `target_year <= fc_year` 改为 `target_year < fc_year`，
   重新产出全部 rolling 表格与图。预期核心结论不变且更稳健（差距更大）。
2. **（必须）重命名 EBI/A 并披露 caveat.** 标注为“归母净利润 + 利息支出 / 总资产（税后、非 EBIT）”，
   说明税差与少数股东两条方向性影响。
3. **（强烈建议）金额差距伴随共同支持分解报告.** 明确 headline 金额（4.06 万亿累计）约 60–65%
   由共同支持之外的大企业贡献；将等权比例差距作为更稳健的核心指标。
4. **（可选）在稳健性中补充 KNN 可比公司结果.** KNN5/KNN10 差距逐年为正且与 GBM 高相关，
   可直接回应“模型依赖”质疑。
5. **（可选）补充 200 次聚类 bootstrap CI** 作为非参数推断，替代/补充 analytic 聚类 SE。

---

## 5. 下一步

- 若 Wind 可提供合并净利润 / 所得税字段，补跑 `ebi_definition_robustness.csv`（净利润+利息、EBIT 口径）。
- 基于修正窗口重估后，更新 `final_results/`（保留旧版本备份以便 before/after 对比）。

---

## 6. 产物清单（outputs/revision2/）

**18 个 CSV + 3 个 MD + 1 个本报告 = 22 个数据/文档文件，7 个图**，全部独立存放于
`outputs/revision2/`，未改动任何原始数据或既有结果。可复现脚本 `audit_*.py` 同目录存放。
