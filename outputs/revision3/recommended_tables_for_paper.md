# TASK 8 — 正文表格建议清单（Recommended Tables for Paper）

> 原则：正文最多保留 5 张核心表；其余稳健性/诊断表放附录。不要把全部表塞进正文。

---

## Main text（建议正文，≤ 5 张）

| # | 表 | 对应新文件 | 建议作用 |
|---|---|---|---|
| 1 | 基准逐年 gap 表（2019–2025，All SOE + CSI300/500/1000） | `final_results/summary_by_year_all_soe.csv` + `summary_by_sample_year.csv` | 主结果：逐年条件回报缺口 |
| 2 | 规模异质性表（按 FirmSize 分箱 gap） | `final_results/fig_supp_size_gap_and_calibration.png` 的底层数据 | 主结果的规模维度 |
| 3 | 多年 common-support 主表 | `outputs/revision3/table_multiyear_common_support.csv` | Full / CS / Persistent 多年平均 gap |
| 4 | 2025 common-support + amount 分解表 | `outputs/revision3/table_amount_gap_support_2025.csv` | 2025 金额分解（good/boundary/extrapolation） |
| 5 | KNN / OOF placebo 精简表 | `outputs/revision2/private_oof_placebo_summary.csv` + `outputs/revision3/industry_restricted_knn_summary.csv`（精简） | 可比企业 + 安慰剂方向一致性 |

---

## Appendix（建议放附录）

| 结果 | 对应新文件 |
|---|---|
| 完整 dynamic overlap（逐年 × 样本） | `outputs/revision3/table_dynamic_overlap_2019_2025.csv`（+ `table_dynamic_overlap_summary.csv`） |
| Support transition matrix | `outputs/revision2/common_support_transition_matrix.csv` |
| 行业内 KNN（完整） | `outputs/revision3/industry_restricted_knn_summary.csv` + `industry_restricted_knn_firm_level.csv` |
| EBI alternative definition | `outputs/revision3/ebi_accounting_field_audit.md`（consolidated NI 不可得） |
| Calibration audit | `outputs/revision3/calibration_definition_audit.md` |
| 逐年 amount support 分解（2019–2025 全量） | `outputs/revision3/table_amount_gap_support_decomposition.csv` |
| 完整 bootstrap 结果 | `final_results/bootstrap_inference_2025.csv`、`outputs/revision2/gbm_cluster_bootstrap_selected_samples.csv` |
| 中央 vs 地方差异检验 | `outputs/revision2/central_local_difference_test.csv` |
| 时间窗口审计 | `outputs/revision3/time_window_audit_final.csv` |

---

## 不建议放入正文

- 逐年 dynamic overlap 全表（信息密度低、适合附录）。
- 完整 transition matrix（辅助性）。
- 完整 bootstrap 明细（方法论细节，正文只写 SE/CI）。

---

## 措辞约束（与正文修改直接相关）

1. 不要称 extrapolation 金额为“上界”，只写：
   > amount-gap estimate is highly sensitive to observations outside the main private-sector support region.
2. 不要自动写“calibration slope > 1 保证 gap 是保守下界”，除非能证明；否则按 `calibration_definition_audit.md` 的表述弱化。
3. 不要将 AmountGap 自动解释为 subsidy / fiscal cost / welfare loss；不要将 2019–2025 名义金额求和解释为 present value。
