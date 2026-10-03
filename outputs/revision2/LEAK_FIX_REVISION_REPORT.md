# 时间窗泄露修复与全链路修订报告

**日期**：2026-08-19
**范围**：代码 / 数据 / 图表 / 论文正文
**性质**：修订（不修改原始数据、不覆盖既有结果，所有新结果落入既有 `final_results/`、`outputs/revision/`、`outputs/revision2/` 目录）

---

## 0. 一句话结论

原始流水线存在**两个独立缺陷**：① 扩展窗口训练时用 `target_year <= fc_year` 过滤民营企业训练集，**把"与预测年同一年实现的民营 outcome"也纳入了训练**（时间窗前视泄露，违反严格 ex-ante 原则）；② 部分结果脚本在 `from run_counterfactual import *` 之后，被 `run_counterfactual.py` 顶层的 `OUTPUT_DIR = .../outputs` 覆盖了本应写入 `final_results/` 的输出目录，导致**结果被静默写入错误目录**。两者均已修复并全量重跑。修复后**核心结论方向不变**（gap 为正、显著、存在规模梯度、行业集中），但 **gap 的幅度系统性变大**（时间窗泄露此前低估了缺口）。

---

## 1. 缺陷定位与修复

### 缺陷 A：时间窗前视泄露（TASK 1，最高优先级）

- **泄露点**：`outputs/revision/*.py`、`outputs/revision2/audit_*.py` 中扩展窗口训练集过滤：
  - 修复前：`train_pool[train_pool['target_year'] <= fc_year]`（包含 outcome 与预测年同年的民营样本）。
  - 修复后：`train_pool[train_pool['target_year'] < fc_year]`（严格 ex-ante：`OutcomeYear_train < ForecastYear`）。
- **受影响脚本**：主流水线 9 个脚本（`src/01_main_pipeline.py` 等）+ 审计脚本 6 个（`audit_03_04` / `audit_04a` / `audit_05` / `audit_06` / `audit_08` / `audit_09`）。
- **量化影响**：以 2025 年预测期为例，泄露窗口包含 3,314 个"同年 outcome"民营 firm-year（原训练池 26,464 → 修复后 23,150，剔除的同年观测占原训练池 12.5%）。逐年剔除的同年观测数为 2019=2,150、2020=2,574、2021=2,980、2022=3,243、2023=3,326、2024=3,323、2025=3,314（见 `time_window_audit.csv`）。

### 缺陷 B：OUTPUT_DIR 静默覆盖

- **定位**：`run_counterfactual.py:31` 定义 `OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"`；下游脚本 `from run_counterfactual import *` 后，脚本内自定义的 `OUTPUT_DIR` 被导入覆盖。
- **后果**：修正后的结果一度被写入 `outputs/` 而非 `final_results/`，导致读旧目录时出现"重跑后字节级相同（0.015885）"的假象。
- **修复**：在受影响的 7 个脚本导入之后重新断言 `OUTPUT_DIR`。

---

## 2. 核心数字：修复前 vs 修复后

### 2.1 全 A 股 SOE 逐年 GBM gap（firm 层面均值，单位 pp）

| 年份 | 修复前（泄露） | 修复后 | Δ |
|------|--------------:|-------:|--:|
| 2019 | +1.90 | **+1.95** | +0.05 |
| 2020 | +2.33 | **+2.47** | +0.14 |
| 2021 | +1.55 | **+1.55** | +0.00 |
| 2022 | +1.45 | **+2.03** | +0.58 |
| 2023 | +1.35 | **+1.91** | +0.55 |
| 2024 | +1.76 | **+2.29** | +0.53 |
| 2025 | +1.59 | **+1.92** | +0.33 |
| **7 年均值** | **+1.70** | **+2.02** | **+0.32** |

> 泄露修复后 gap **方向（正）与显著性不变**，幅度在多数年份（尤其 2022–2025）系统性变大（+0.33 ~ +0.58pp）；2019–2021 年影响较小（0 ~ +0.14pp）。这与预期方向一致：泄露窗口把"同年已实现的民营 outcome"混入训练，抬高了反事实基准，从而**低估**了 SOE 的缺口；其影响在样本后段（同年观测占比更大、历史信息更少）更为明显。修复前（泄露）与修复后（ex-ante）的完整逐年对照见 `time_window_original_vs_corrected.csv`。

### 2.2 金额缺口（亿元）

| 口径 | 修复前 | 修复后 |
|------|-------:|-------:|
| 2025 年金额缺口 | 5,979 | **7,560** |
| 2019–2025 累计 | ~4.1 万亿 | **5.1 万亿（50,590）** |

### 2.3 推断统计（2025）

| 方法 | 样本 | 修复后点估计 | SE | 95% CI | 结论 |
|------|------|------------:|---:|--------|------|
| 聚类 SE（GBM） | All SOE | +1.92pp | 0.13 | [+1.67, +2.17] | 显著 |
| 聚类 SE（GBM） | 中央国企 | +1.84pp | 0.21 | [+1.42, +2.26] | 显著 |
| 聚类 SE（GBM） | 地方国企 | +1.96pp | 0.16 | [+1.64, +2.27] | 显著 |
| Bootstrap（Ridge，500） | All SOE | +1.63pp | 0.14 | [+1.35, +1.91] | 显著 |
| Bootstrap（Ridge，500） | 中证1000 | +0.81pp | 0.19 | [+0.45, +1.18] | 显著 |
| Bootstrap（Ridge，500） | 沪深300 | −0.43pp | 0.38 | [−1.15, +0.34] | 不显著 |
| Bootstrap（Ridge，500） | 中证500 | +0.26pp | 0.21 | [−0.13, +0.65] | 不显著 |

---

## 3. 14 个 Task 的发现与论文落点

| Task | 主题 | 结论 | 论文落点 |
|------|------|------|---------|
| 0 | 项目结构 | 结构清晰、原始数据未被修改 | 附录 C 复现代码清单 |
| **1** | **时间窗泄露** | **FAIL → 修复重估**，gap 系统性变大、方向不变 | 全文核心数字更新 |
| 2 | 民营 OOF 安慰剂 | pooled 平均残差 +0.0055pp，p=0.914（n=26,464 obs / 3,344 firms），民营基准近似无偏 | 新增 §5.7.5(1) |
| 3 | 金额缺口分解 | 金额缺口 62.2% 来自共同支撑域外（外推区） | §5.7.2、§5.6、结论 |
| 4 | 全样本 vs 共同支撑 | 全 A 股 gap +1.92pp → CS 内 +1.83pp，核心结论非外推驱动 | §5.7.2 表 9 |
| 4A | 多年共同支撑 | 支撑状态高度粘性（good→good 88.7%、外推→外推 81.3%） | §5.7.2 |
| 5 | KNN 基准 | KNN5/KNN10 gap 逐年为正，与 GBM 相关系数 0.85~0.91 | 新增 §5.7.5(3) |
| 6 | 行业受限共同支撑 | 行业内 CS 下 gap 仍为正且相近（2025 +1.73pp） | 附录 / 稳健性 |
| 7 | EBI 会计口径审计 | EBI 为**税后、归母**口径（归母净利润+利息支出），非税前 EBIT | 摘要、§3.3、结论"局限" |
| 8 | 中央 vs 地方 | 无显著差异（修复后 p=0.67） | §5.5 表 6 |
| 9 | GBM 聚类 bootstrap | 2025 All SOE bootstrap CI [0.017, 0.022]，排除零 | §5.7.4（与 Ridge bootstrap 一致） |
| 10 | 输出一致性 | 各脚本输出经核对一致（修复 OUTPUT_DIR 后） | — |
| 12 | 终审报告 | 见 `AUDIT_AND_REVISION_REPORT.md` | — |
| 13 | 文件校验 | 所有 CSV/图/报告均在位 | 附录 B |
| 14 | 终端总结 | 见本报告 | — |

### "Results That Changed" 分级

- **A（不变）**：核心结论方向、显著性、规模梯度、行业集中。
- **B（Task 1，重估）**：gap 幅度系统性变大（+0.3~0.45pp），金额缺口 5,979→7,560 亿元，累计 4.1→5.1 万亿。
- **C（Task 3/4/4A/7）**：金额缺口外推占比 64.5%→62.2%（修正后仍过半）；EBI 披露为税后归母口径。
- **D（Task 2/5/6/8/9/10，确认）**：民营安慰剂≈0、KNN 复现、行业 CS 稳健、央地无差异、聚类 bootstrap 一致、输出一致。

---

## 4. 修改文件清单

### 代码（修复泄露 + OUTPUT_DIR）

- `run_counterfactual.py`（OUTPUT_DIR 源头）
- 主流水线 9 个脚本：`src/01_main_pipeline.py`、`outputs/revision/rerun_filtered.py`、`export_final_results*.py`、`inference*.py`、`cluster_bootstrap.py`、`common_support_full.py`、`fig_amount_gap.py`、`industry_screened.py` 等（`<= fc_year` → `< fc_year` + 重新断言 `OUTPUT_DIR`）
- 审计脚本 6 个：`outputs/revision2/audit_03_04_amount_gap.py`、`audit_04a_multiyear_cs.py`、`audit_05_knn.py`、`audit_06_industry_cs.py`、`audit_08_central_local.py`、`audit_09_bootstrap.py`（`<= fc_year` → `< fc_year`）

### 数据（全部重跑后覆盖正确目录）

- `final_results/summary_by_sample_year.csv`、`inference_stats_2025.csv`、`central_vs_local_2025.csv`、`bootstrap_inference_2025.csv`、`cs_common_support.csv`、`cs_support_restricted_gap.csv`、`cs_propensity.csv`、`cs_smd.csv`、`summary_by_industry_2025.csv`
- `outputs/revision/filtered_rolling.csv`
- `outputs/revision2/knn_comparable_summary_by_year.csv`、`amount_gap_by_support_year.csv`、`multiyear_amount_gap_summary.csv`、`common_support_transition_matrix.csv`、`central_local_difference_test.csv`、`gbm_cluster_bootstrap_selected_samples.csv`

### 图表（重跑后覆盖）

- `final_results/fig_amount_gap_by_year.png`、`fig_ratio_vs_amount_gap.png`、`fig_firm_size_overlap_screened.png`、`fig_supp_size_gap_and_calibration.png`
- `outputs/revision/fig_summary_7year_screened.png`、`fig_industry_year_heatmap_screened.png`、`fig_industry_topbottom_trend_screened.png`
- `outputs/revision2/fig_amount_gap_support_decomposition.png`、`fig_dynamic_common_support_gap.png`、`fig_gbm_vs_knn5_gap.png`、`fig_gbm_vs_knn10_gap.png`、`fig_central_local_difference_by_year.png`

### 论文

- `论文_中国国有企业反事实资产回报率研究.md`（→ 同步重生成 `.html` / `.pdf`）
  - 摘要、引言、§3.3 EBI 定义、表 2 注、表 3、核心发现一、表 4、表 5、核心发现三、表 6、表 7、核心发现四/五、表 8、表 9、表 10、§5.7.2、§5.7.3、§5.7.4、新增 §5.7.5、结论。

---

## 5. 判定与下一步建议

### 总判定：**PASS（核心结论）／ WARNING（幅度与金额口径）**

- **PASS**：修复时间窗泄露后，全 A 股 SOE 的反事实资产回报率缺口在 2019–2025 年**仍为正且统计显著**（+1.5~+2.5pp），规模梯度（大市值≈0/负、中小市值显著为正）、行业集中（钢铁/煤炭/家电等重资产与周期行业最大）、中央-地方无差异，全部在样本筛选、共同支撑、模型类别、聚类推断、民营安慰剂、KNN 基准、聚类 bootstrap 多重检验下保持稳健。
- **WARNING（幅度）**：金额缺口（2025 年 7,560 亿元、7 年累计 5.1 万亿）约 **62.2% 来自共同支撑域外的外推**（大市值 SOE 缺乏可比民企对照），金额量级应视为**上界而非精确点值**；等权比例 gap（+1.92pp）才是更稳健的核心指标。
- **WARNING（口径）**：① EBI 为**税后、归母**口径，非税前 EBIT，与税前效率度量不可直接比较；② 金额缺口是"gap × 资产总计"的**资产加权求和**，反映资产规模而非当年新增补贴现金流，**不能解读为现值加总**；③ 本文 gap 是"以民营为基准的回报率差异"，**不自动等同于政府补贴或政策性负担**。
- **WARNING（披露精度，轻微）**：表 2 的 OOF 指标（GBM R²=+0.303）为**全样本 pooled**（对全部民营 firm-year 做一次 5 折 GroupKFold）计算，而 §5.2 引言的"2025 年预测期"表述暗示逐年滚动窗口。滚动 2025 窗口的 GBM OOF R²=0.2992（见 `filtered_rolling.csv`）。两者均为合法的"模型拟合"度量，差异仅 0.004，不影响任何结论，但表 2 的"2025 年预测期"措辞宜改为"全样本 pooled"以避免歧义。

### 下一步建议

1. **金额缺口的共同支撑稳健化**：对处于外推区的大市值 SOE，考虑按 FirmSize 分层匹配、或改用"支撑域内企业"单独报告金额缺口，避免外推主导金额量级。
2. **口径敏感性**：补充"税前 EBIT/A"与"归母 EBI/A"的双口径对照，量化税负与少数股东权益对 gap 的贡献。
3. **时变所有制**：当前所有制为静态快照（潜在 look-ahead 偏差），建议获取时变所有制字段，消除该偏差并做敏感性检验。
4. **因果识别**：在反事实预测之外，结合双重差分 / 断点设计（如国企改革试点、混改事件），进一步分解 gap 的来源（代理成本 vs 政策性负担 vs 市场势力）。
5. **跨市场与新股筛选**：纳入港股/中概股，补全新股与低面值筛选，检验结论的外部有效性。
