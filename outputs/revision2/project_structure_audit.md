# TASK 0 — Project Structure Audit

**Purpose.** Map the full reproduction chain: which scripts produce which outputs, how
data flows from raw files to the final CSVs, and what the canonical column names are.
This is a *structural* audit — correctness issues found along the way are flagged but
are developed in the dedicated task reports (TASK 1, 7, 10, …).

---

## 1. Entry scripts and their roles

| Script | Role | Primary output(s) |
|---|---|---|
| `run_counterfactual.py` | **Shared library** — data loading, feature construction, model builders, `run_full_experiment()` | `outputs/counterfactual_summary.csv` (pooled training, *not* the paper's rolling numbers) |
| `src/01_main_pipeline.py` | Rolling (2019–2025), 3 models × 4 targets, A-share panel only | `outputs/revision/unified_rolling.csv`, `unified_master.md` |
| `outputs/revision/rerun_filtered.py` | Same rolling analysis on the **screened** sample | `outputs/revision/filtered_rolling.csv` |
| `outputs/revision/export_final_results.py` | v1 rolling GBM + amount gap (overlapping index samples) | `final_results/firm_level_gap_all_years.csv`, `summary_by_sample_2025.csv`, … |
| `outputs/revision/export_final_results_v2.py` | **v2 (production)** rolling GBM + amount gap, **mutually-exclusive** samples | `final_results/firm_level_gap_all_years.csv`, `firm_level_gap_2025.csv`, `summary_by_sample_year.csv`, `summary_by_industry_year.csv`, `summary_by_industry_2025.csv` |
| `outputs/revision/common_support_full.py` | 2025 single-year common-support diagnostics | `final_results/cs_smd.csv`, `cs_propensity.csv`, `cs_common_support.csv`, `cs_mahalanobis.csv`, `cs_support_restricted_gap.csv` |
| `outputs/revision/inference_and_heterogeneity.py` | 2025 clustered inference + central/local split | `final_results/inference_stats_2025.csv`, `central_vs_local_2025.csv`, `firm_level_prediction_2025.csv` |
| `outputs/revision/cluster_bootstrap.py` | Ridge cluster bootstrap (500 draws, 2025) | `final_results/bootstrap_inference_2025.csv` |
| `outputs/revision/export_private_oof.py` | Private-firm OOF predictions (placebo source) | `final_results/private_oof_predictions.csv`, `private_oof_firm_mean.csv` |
| `outputs/revision/industry_screened.py` | Industry × year gap + common-support figures | `outputs/revision/industry_year_firm_gaps_screened.csv`, `industry_year_cs_firm_gaps_screened.csv`, 6 figure files |
| `outputs/revision/fig_amount_gap.py` | Amount-gap figures (reads `summary_by_sample_year.csv`) | `final_results/fig_amount_gap_by_year.{png,pdf}`, `fig_ratio_vs_amount_gap.{png,pdf}` |
| `outputs/revision/fig_firm_size_overlap_screened.py` | Firm-size overlap figure | `final_results/fig_firm_size_overlap_screened.{png,pdf}` |
| `outputs/revision/export_descriptives.py` | Table-1 descriptives | `outputs/revision/descriptives_by_ownership.csv/.tex` |
| `outputs/revision/audit_*.py` | Prior audit scripts (ownership, sample construction, variables, gap sign) | `outputs/revision/*_audit.md`, `sample_flow.csv`, `spot_check_results.csv`, … |

**Which script produced the paper's headline numbers?** The mutually-exclusive
decomposition in `summary_by_sample_year.csv` (`Non-index / CSI300 / CSI500 / CSI1000`,
plus `All A-share SOE`) is produced by **`export_final_results_v2.py`** (v2). The earlier
`export_final_results.py` (v1) used overlapping index samples and was superseded.

---

## 2. Data loading pipeline

`run_counterfactual.load_all_a_shares()` reads **11** raw files
`0803/A_{year}_origin.xlsx` for `year = 2014…2024`. Each file is parsed by
`parse_xlsx_xml()` (a custom XML parser that bypasses openpyxl style corruption) and
converted by `records_to_dataframe(records, expected_feature_year=year,
expected_target_year=year+1)`. Hence each row is a firm-year with:

- `feature_year = t`  (features measured from fiscal-year **t** statements)
- `target_year = t + 1` (outcome measured from fiscal-year **t+1** statements)

The two special files `csi500_2024.xlsx` and `中证1000_2024_财务数据提取.xlsx` are parsed by
`load_csi500()` / `load_csi1000()` but are **not** used in the rolling production pipeline —
the rolling pipeline identifies CSI300/500/1000 members **only** from the constituent-code
files in `input_index_weight/` (see §3).

---

## 3. Index membership reconstruction

`input_index_weight/` contains six files (per index: a "成分及权重" and a "成分进出记录").
Constituent codes are reconstructed **backward** from the current (2026-08-11) list using
the entry/exit records, anchored at `{year}-07-01`:

- "纳入" after the anchor date → discard (was not yet in the index),
- "剔除" after the anchor date → add back.

`build_yearly_constituents()` / `load_yearly_codes()` produce `yearly_codes[index][year]`
for `year = 2019…2025`. This is a standard point-in-time reconstruction, but note it is an
**approximation** (it cannot account for index reconstitution exactly at a fiscal boundary;
membership is as of July 1 of each year). This is the same assumption used across all
production scripts.

---

## 4. Feature construction (`records_to_dataframe`)

**Outcome.** For a row with `feature_year = t`, `target_year = t+1`:

```
EBI/A_t+1 = (归属母公司股东的净利润_t+1  +  利息支出_t+1) / 资产总计_t+1
```

The numerator profit field is **归母净利润** (net profit attributable to parent), *not*
consolidated net income — see TASK 7. The outcome is winsorized/clipped later, but is
computed here in ratio form (a fraction, not a percentage).

**Nine numeric features (all measured at feature year t):**

| Feature | Definition |
|---|---|
| `FirmSize` | `ln(资产总计_t)` |
| `AssetTurnover` | `营业收入_t / 资产总计_t` |
| `FinancialLeverage` | `负债合计_t / 资产总计_t` |
| `FixedAssetsRatio` | `固定资产_t / 资产总计_t` |
| `CurrentRatio` | `流动资产合计_t / 流动负债合计_t` |
| `CapexRatio` | `资本性支出_t / 资产总计_t` |
| `WorkingCapitalRatio` | `(流动资产合计_t − 流动负债合计_t) / 资产总计_t` |
| `MarketShare` | `营业收入_t / Σ_j 营业收入_t` within `(feature_year, industry_l2)` |
| `HHI` | `Σ_j MarketShare²` within `(feature_year, industry_l2)` |

`MarketShare` and `HHI` are computed on the **full A-share panel** (market-wide
denominator) within each `(feature_year, industry_l2)` group — not just private firms.

**Categorical fixed effect.** `industry_l2` = second level of `所属Wind行业名称`
(split on `--`), one-hot encoded in the model. Financial industries
(`银行/保险/证券/多元金融/非银金融`) are excluded.

---

## 5. Screening and cleaning

Two layers:

1. **`clean_and_filter()` (library):** backfill 2014 ownership from later years, drop
   financials, drop missing EBI/A, drop `|EBI/A| > 1`, drop unclassified ownership.
2. **Production screen (in every `outputs/revision/*.py` script):** additionally
   (a) drop ST firms (firm-level, current stock name contains `ST`),
   (b) drop `FinancialLeverage > 1` (insolvent, firm-year),
   (c) drop `|EBI/A| > 0.5` (extreme outcome, firm-year).

The paper's numbers in `final_results/` reflect the **full two-layer screen**.

---

## 6. Model training

Three models, identical hyperparameters everywhere (`build_model` / `build`):

| Model | Parameters |
|---|---|
| GBM (main) | `GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=10)` |
| Ridge | `Ridge(alpha=10.0)` |
| Random Forest | `RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, max_features='sqrt')` |

`random_state = 42` throughout. Preprocessing: winsorize features and target at P5/P95
(based on the **training** distribution), median impute, `StandardScaler`, one-hot
`industry_l2` (`handle_unknown='ignore'`). Out-of-fold metrics use `GroupKFold` (≤5 folds)
grouped by `firm_id`.

---

## 7. Expanding window (the rolling analysis)

The production loop over `forecast_year = 2019…2025` filters the private training pool with

```python
train = train_pool[train_pool['target_year'] <= fc_year]
```

and predicts on SOE firms with `target_year == fc_year`. **This includes same-year private
outcomes** (`target_year == fc_year`) in the training set, violating the strict ex-ante
requirement `OutcomeYear_train < ForecastYear`. This is developed in **TASK 1** (leakage
flag = TRUE for all seven years; impact quantified there).

---

## 8. Common support, bootstrap, amount gap

- **Common support (2025).** `common_support_full.py` computes, relative to the private
  pool: univariate SMD (pooled-SD), out-of-range shares (P1/P99, P5/P95); OOF propensity
  score (ROC-AUC) via stratified 5-fold logistic regression; 5-NN distance in standardized
  feature space vs the **P90** of the private firms' own internal NN distance
  (`good / boundary / extrapolation`); Mahalanobis distance in PCA top-5 space.
- **Bootstrap.** `cluster_bootstrap.py` resamples **firms** (train and SOE) with
  replacement, 500 draws, Ridge only.
- **Amount gap.** `gap_ratio × 资产总计_target` (RMB yuan), reported in 亿元 (`/1e8`).

---

## 9. Canonical column names (from `records_to_dataframe`)

| Concept | Column name |
|---|---|
| Firm ID | `firm_id` (= `证券代码`, stock code) |
| Firm name | `firm_name` (= `证券简称`) |
| Ownership | `ownership` (= `企业所有制性质`); SOE = `中央国有企业`/`地方国有企业`, benchmark = `民营企业` |
| Industry (level 1 / 2) | `industry_l1`, `industry_l2` (from `所属Wind行业名称`) |
| Feature / target year | `feature_year`, `target_year` |
| Feature-year total assets | `资产总计` |
| **Target-year total assets** | `资产总计_target` (used as amount-gap weight) |
| Feature-year revenue | `营业收入` |
| Outcome | `EBI_A` (fraction) |
| Nine features | `FirmSize, AssetTurnover, FinancialLeverage, FixedAssetsRatio, CurrentRatio, CapexRatio, WorkingCapitalRatio, MarketShare, HHI` |
| Flags | `is_soe`, `is_private`, `central` (=`中央国有企业`), `local` (=`地方国有企业`) |

---

## 10. Core I/O inventory

**Inputs (read-only):**
- `0803/A_2014_origin.xlsx` … `A_2024_origin.xlsx` (11 files) — A-share financials
- `0803/csi500_2024.xlsx`, `0803/中证1000_2024_财务数据提取.xlsx` — legacy index files (not in rolling pipeline)
- `input_index_weight/` — 6 files: per-index 成分及权重 + 成分进出记录 (2026-08-11)

**Outputs (existing, not to be overwritten):**
- `final_results/*.csv` — the paper's core numbers (`firm_level_gap_all_years.csv`,
  `summary_by_sample_year.csv`, `summary_by_industry_year.csv`, `summary_by_industry_2025.csv`,
  `inference_stats_2025.csv`, `central_vs_local_2025.csv`, `bootstrap_inference_2025.csv`,
  `private_oof_predictions.csv`, `private_oof_firm_mean.csv`, `cs_*.csv`, …)
- `outputs/revision/*.csv/*.md/*.png/*.pdf` — intermediate rolling results, figures, prior audits

**Outputs of this audit (new, isolated):** everything under `outputs/revision2/`.
