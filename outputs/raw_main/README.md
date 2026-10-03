# RAW_RAW 主规格重算（Task C）— 交付清单与结论

生成日期：2026-08-20。目录：`outputs/raw_main/`（独立目录，未覆盖任何既有结果，未修改论文正文）。

## 统一 outcome 定义
`Gap = pred(E[EBI/A^{raw}|X, Private]) − EBI/A^{raw}_{SOE}`；GBM / Ridge / RF 三个模型均以
raw EBI/A 为训练 target。其余设定（样本筛选、9 特征、feature winsorize、impute、scale、one-hot、
严格 ex-ante expanding window、2019–2025、超参数、seed、CSI300/500/1000、v2 5-NN 共同支撑、
firm-cluster 推断）全部冻结。共同支撑分类**未重算**，直接复用 `support_v2/soe_support_firmyear_v2.csv`。

## 交付文件（22 个经验结果；另附可复现代码脚本与本 README）
- `QA_raw_main_spec.md`
- `amount_gap_2025_support_raw.csv`
- `amount_gap_by_year_raw.csv`
- `bootstrap_inference_raw.csv`
- `central_local_pooled_raw.csv`
- `central_local_raw.csv`
- `fig_dynamic_common_support_gap_raw.png`
- `fig_industry_topbottom_trend_raw.png`
- `fig_summary_7year_raw.png`
- `industry_gap_by_year_raw.csv`
- `industry_gap_multiyear_raw.csv`
- `industry_old_vs_raw_rank.csv`
- `main_gap_by_year_raw.csv`
- `main_gap_multiyear_raw.csv`
- `model_performance_raw.csv`
- `model_performance_raw_by_year.csv`
- `model_replacement_gap_raw.csv`
- `outcome_definition_robustness_final.csv`
- `private_oof_placebo_raw.csv`
- `sample_filter_audit.md`
- `support_multiyear_summary_raw.csv`
- `support_year_sample_summary_raw.csv`

### 可复现代码脚本（3 个）
- `raw_main_figures.py`
- `raw_main_report.py`
- `raw_main_run.py`

## 结论（客观事实）
1. 主结果：All SOE 多年反事实缺口 GBM **+1.45 pp [1.25, 1.64]**（Ridge +0.97、RF +1.70），显著 > 0。
2. 共同支撑（good+boundary）+1.42 pp、persistent-good（GoodShare≥0.8）+1.41 pp，均显著为正。
3. 沪深300 无稳健缺口（+0.07 pp，不显著）；中证500 +1.34、中证1000 +1.39 显著为正。
4. 民营 OOF placebo 均值 0.00（p=0.98），模型无系统性偏差。
5. 中央 vs 地方无显著差异（差 −0.11 pp）。
6. 金额缺口（2025）约 6,826 亿元，外推区金额占比 16.4%。
7. outcome 口径稳健性：CURRENT +2.02 / RAW_RAW +1.45 / WIN_WINSYM +1.50，方向一致、量级敏感约 0.5 pp。
8. 判断：**PASS** —— 主结论方向对 outcome 定义稳健，但点估计量级对口径敏感，正文需按作者决定统一口径。
