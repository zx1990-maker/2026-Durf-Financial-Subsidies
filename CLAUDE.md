# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

A counterfactual econometric pipeline that estimates how A-share SOE (state-owned
enterprise) profitability *would* look if those firms behaved like non-SOE
benchmarks. It trains a Ridge model on non-SOE firms, then predicts on SOE firms;
the gap between predicted and actual return measures the implied SOE
under/over-performance (interpreted as the effect of financial subsidies).

Three benchmarks are compared for every run:
- **H_nonSOE** — Hong Kong (H-share) private firms
- **CSI_nonSOE** — mainland (CSI300) private firms
- **Pooled** — H + CSI private firms combined (adds a `market_A` dummy; SOE is predicted as if `market_A = 1`)

The primary target is **EBI/A** (`(EBIT − Taxes) / Total Assets`); ROA and ROE are
run as supplementary robustness targets. See `Feature_Selection.md` for the exact
variable definitions and `README.md` for data-format details and current results.

## Commands

```bash
pip install -r requirements.txt

# Yearly mode — RECOMMENDED. Uses data_input/{YYYY}/ folders (2018–2024),
# auto-merges ROA/ROE, runs EBI/A + ROA + ROE targets into outputs/{TARGET}/.
python scripts/run_benchmarks.py yearly
python scripts/run_benchmarks.py yearly --years 2023 2024   # subset of years
python scripts/run_benchmarks.py yearly --alpha 5.0 -v      # override hyperparams

# Legacy mode — single 2024→2025 period from the three original wide files.
python scripts/run_benchmarks.py legacy

# Panel mode — long-format data registered in data_input/manifest.json.
python scripts/run_benchmarks.py run --mode panel
python scripts/run_benchmarks.py register --file data_input/panel_2018.csv --years 2018,2019
python scripts/run_benchmarks.py validate --file data_input/panel.csv
```

There is no test suite, linter, or CI configured. Validate changes by running the
pipeline end-to-end (`yearly` is the canonical path) and inspecting `outputs/`.
Use `validate` to schema-check a panel file before a full run.

## Architecture

Everything funnels through `scripts/run_benchmarks.py`, which dispatches to one of
three run modes. All three modes converge on the **same long-format panel
DataFrame** and call the shared core `run_benchmarks_panel()`. The whole system is
config-driven by `config/default.yaml` — column names, feature lists, benchmark
group labels, fixed-effects toggles, winsor bounds, and matching methods all live
there, not in code.

### The unifying data contract: the long panel

Regardless of input format, data is reshaped to one row per (firm, year) with
canonical column names (`firm_id`, `year`, `market`, `benchmark_group`,
`industry_l1`, `industry_l2`, `EBI_A`, `资产总计`, plus the 9 numeric features).
`wide_to_long()` in `src/data_loader.py` is the bridge: each wide firm row becomes
a **feature-year row** (features populated, target NaN) and a **target-year row**
(target/assets populated, features NaN — later filled by lag). The `_row_type`
column tracks this distinction and gates where ROA/ROE values are allowed.

### Pipeline stages (in `run_benchmarks_panel`)

1. **Lag** — `build_lagged_features()` (`src/features.py`) shifts each numeric
   feature by one year within each firm → `{feature}_lag`. The model is strictly
   `X_{t-1} → Y_t`. All numeric features used by the model are the `_lag` versions.
2. **Split** — `split_train_pred()` separates SOE firms (the prediction target)
   from the non-SOE benchmark training sets, building the H/CSI/Pooled dict.
3. **Optional matching/weighting** — when `psm.enabled`, `src/psm.py` runs PSM,
   CEM, entropy balancing, and overlap weighting as *preprocessing* that reweights
   or subsets the non-SOE training pool toward SOE comparability. Each method
   produces its own summary row alongside the `baseline`.
4. **Fit + predict** — `src/model.py` builds an sklearn `Pipeline`
   (impute → scale → OHE → Ridge). Winsorization is applied per-fold/per-fit using
   **train-distribution quantiles only** (no leakage). CV uses firm-level
   `GroupKFold` to prevent the same firm leaking across folds.
5. **Gaps** — `compute_gap_columns()` (`src/utils.py`): `return_gap = predicted −
   actual`, `implied_gap_amount = return_gap × assets`, positive-only variant too.
6. **Common support** — `compute_support_flags()` flags SOE rows whose lagged
   covariates fall outside the benchmark's P1–P99 range; a `common_support`
   summary subset is emitted alongside the full sample.
7. **Output** — `src/outputs.py` `write_all_outputs()` writes the CSV/Excel set
   described in `README.md`.

### Module map (`src/`)

- `data_loader.py` — config loading, all three input loaders, wide→long bridge,
  industry filter, train/pred split, ROA/ROE wide-file parsing & merge.
- `features.py` — lagging, fixed-effects dummy encoding, winsorization, common-support.
- `model.py` — pipeline construction, `cv_metrics` (GroupKFold), `fit_predict`.
- `psm.py` — the four matching/weighting methods.
- `outputs.py` — summary/group/diagnostic builders and file writers.
- `utils.py` — numeric coercion (handles `,` `%` `--`), weighted average, gaps.

## Things to know before editing

- **Config is the source of truth.** Adding a feature, changing a benchmark label,
  toggling a fixed effect, or excluding an industry is a `config/default.yaml`
  edit. Code reads names from config (`config["data"][...]`, `config["features"][...]`).
- **Chinese column names are load-bearing.** Identifiers like `证券代码`,
  `资产总计`, `二级行业`, `企业所有制性质`, and ownership values
  (`地方国有企业`, `中央国有企业`, `民营企业`) are matched literally against the raw
  data. The `非SOE` benchmark defaults to `民营企业` only.
- **Financial firms are filtered by `二级行业`** (not L1) against
  `exclude_industries`, applied at load time in every mode.
- **Yearly mode auto-detects old vs new per-year format.** New = two mixed files
  (`csi_features_{year}.csv`, `h_features_{year}.csv`) with explicit ownership
  classification; old = three separated files assigned by source. Group assignment
  logic differs between the two — see `load_yearly_data()`.
- **Winsorization must stay train-only.** Bounds are computed from the training
  distribution and applied to both train and pred; never fit bounds on pred or on
  the full pool, or the counterfactual leaks.
- **`outputs/` is committed** and overwritten on each run. The per-target
  subdirectories (`outputs/EBI_A/`, `ROA/`, `ROE/`) are produced only by `yearly`.
