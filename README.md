# Measuring Financial Subsidies to Chinese State-Owned Enterprises

[![Tests](https://github.com/zx1990-maker/2026-Durf-Financial-Subsidies/actions/workflows/test.yml/badge.svg)](https://github.com/zx1990-maker/2026-Durf-Financial-Subsidies/actions/workflows/test.yml)

An audit-first machine-learning study of the return concessions associated with financing Chinese A-share listed state-owned enterprises (SOEs). The project builds strictly out-of-time private-sector return benchmarks and makes extrapolation risk explicit with firm-level common-support diagnostics.

**Authors:** Zixuan Xin and Jiayu Xu

**Final paper:** [PDF](paper/quantifying-financial-subsidies-to-chinese-soes.pdf)

## Research question

How much lower is an SOE's realized return on assets than the return predicted for a comparable private firm?

Following the asset-return framework of Lucas, Garcia, and Solberg (2026), the project defines:

```text
financial subsidy rate = predicted private-firm EBI/A - realized SOE EBI/A
annual subsidy amount  = financial subsidy rate x book assets
```

EBI/A is earnings before interest divided by total assets. A positive estimate means that the SOE's realized asset return is below its private-sector benchmark. This is an asset-return opportunity-cost estimate, not a budget cash transfer or a causal privatization effect.

## Main findings

| Result | Estimate |
|---|---:|
| Average subsidy rate, 2019-2025 | **1.45 pp** |
| Firm-cluster bootstrap 95% CI | **[1.25, 1.64] pp** |
| Rate within common support | **1.42 pp** |
| Persistent-good-support rate | **1.41 pp** |
| 2025 subsidy amount | **CNY 682.6 billion** |
| 2025 amount from Good + Boundary support | **83.6%** |

The multiyear GBM estimates are 1.34 pp for CSI 500 SOEs and 1.39 pp for CSI 1000 SOEs. The CSI 300 estimate is -0.07 pp and statistically indistinguishable from zero; these very large firms also have substantially weaker private-firm overlap.

## What makes the estimate auditable

- **Strict temporal separation.** Each forecast year uses only private-firm outcomes realized in earlier years. The 2025 model never sees a 2025 outcome.
- **Grouped validation.** Five-fold `GroupKFold` keeps every observation for a firm in a single fold.
- **Multiple model classes.** Gradient boosting is the main specification; Ridge and random forest are robustness checks.
- **Calibration checks.** On pooled out-of-fold private-firm predictions, GBM achieves R2 = 0.239, RMSE = 0.0644, and a calibration slope of 0.98.
- **Explicit extrapolation labels.** A distinct-firm 5-nearest-neighbor diagnostic labels each SOE observation Good, Boundary, or Extrapolation using forecast-year-specific private-firm thresholds.
- **Dependence-aware inference.** Multiyear estimates use 2,000 firm-level cluster-bootstrap draws, preserving within-firm serial dependence.
- **Documented limitations.** Static ownership labels, licensed-data constraints, first-stage uncertainty, and limited overlap for the largest SOEs are stated rather than hidden.

## Repository map

| Path | Purpose |
|---|---|
| `paper/` | Final English paper |
| `run_counterfactual.py` | Wind-file parsing, cleaning, feature construction, and benchmark models |
| `src/01_main_pipeline.py` | Expanding-window analysis for 2019-2025 |
| `outputs/revision3/` | Final common-support, inference, table, and figure audit scripts |
| `outputs/outcome_spec/` | Raw-versus-winsorized outcome specification audit |
| `outputs/raw_main/` | Canonical raw-outcome specification checks |
| `tests/` | Unit tests for research-design invariants and input parsing |

Earlier `revision/` and `revision2/` folders are retained as an audit trail. The final paper and `revision3/` scripts are the canonical reference for reported results.

## Reproducibility

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Full analysis

```bash
python src/01_main_pipeline.py
python outputs/revision3/task10_export_final_tables.py
python outputs/revision3/task11_make_figures.py
```

The raw Wind Financial Terminal exports are licensed and therefore not included. To rerun the full pipeline, place the source files in the ignored `0803/` and `input_index_weight/` directories using the filenames referenced by the loader. The public repository supports code and design review; exact numerical reproduction additionally requires access to those licensed snapshots.

## Responsible interpretation

The model estimates a conditional benchmark difference. It does not identify the mechanism behind the gap, separate debt from equity concessions, or establish what would happen under privatization. External validity is limited to listed A-share firms, and estimates for very large SOEs require extra caution because common support is weaker.

## 中文摘要

本项目用严格样本外的机器学习模型估计中国 A 股上市国企相对于可比民营企业的资产回报差，并用 5-NN 共同支撑诊断明确区分插值与外推。2019-2025 年平均金融补贴率为 1.45 个百分点，2025 年估计金额为 6,826 亿元。完整定义、稳健性检验和解释边界见最终英文论文。
