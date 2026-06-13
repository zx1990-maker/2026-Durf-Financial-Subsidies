#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Output generation: summaries, predictions, group statistics, diagnostics.

Preserves the exact column schemas from the original single-period script
while adding panel-specific fields (train_year_range, n_unique_firms, etc.).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import weighted_avg, compute_gap_columns

BASE = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Summarize predictions
# ---------------------------------------------------------------------------

def summarize_predictions(tmp, benchmark, info, winsor_label, target_col="EBI_A",
                          weight_col="资产总计", pred_col="predicted_EBI_A",
                          actual_col="actual_EBI_A", sample_label="full"):
    """
    Aggregate summary statistics for a set of predictions.

    Parameters
    ----------
    tmp : pd.DataFrame
        Firm-level predictions with gap columns.
    benchmark : str
        Benchmark name.
    info : dict
        Metadata from fit_predict (n_train, target_lower, target_upper).
    winsor_label : str
        e.g. "5-95".
    target_col, weight_col, pred_col, actual_col : str
        Column name aliases.

    Returns
    -------
    dict of summary statistics.
    """
    total_assets = tmp[weight_col].sum()
    aw_pred = weighted_avg(tmp, pred_col, weight_col)
    aw_act = weighted_avg(tmp, actual_col, weight_col)

    row = {
        "sample": sample_label,
        "winsor": winsor_label,
        "benchmark": benchmark,
        "n_train_valid_target": info["n_train"],
        "n_soe_predicted": len(tmp),
        "actual_mean": tmp[actual_col].mean(),
        "predicted_mean": tmp[pred_col].mean(),
        "mean_gap": tmp["return_gap_pred_minus_actual"].mean(),
        "median_gap": tmp["return_gap_pred_minus_actual"].median(),
        "positive_gap_share": (tmp["return_gap_pred_minus_actual"] > 0).mean(),
        "actual_asset_weighted": aw_act,
        "predicted_asset_weighted": aw_pred,
        "asset_weighted_gap": aw_pred - aw_act,
        "net_implied_gap_amount": tmp["implied_gap_amount"].sum(),
        "positive_only_gap_amount": tmp["positive_gap_amount"].sum(),
        "target_lower": info["target_lower"],
        "target_upper": info["target_upper"],
    }
    return row


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------

def group_stats(pred_long, group_col, weight_col="资产总计", actual_col="actual_EBI_A",
                pred_col="predicted_EBI_A"):
    """
    Compute per-group counterfactual statistics.

    Parameters
    ----------
    pred_long : pd.DataFrame
        Long-format predictions with 'benchmark' and group_col columns.
    group_col : str
        Column to group by (e.g. "企业所有制性质", "二级行业").
    weight_col, actual_col, pred_col : str

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for (benchmark, grp), g in pred_long.groupby(["benchmark", group_col], dropna=False):
        rows.append({
            "benchmark": benchmark,
            group_col: grp,
            "n_firms": len(g),
            "actual_mean": g[actual_col].mean(),
            "predicted_mean": g[pred_col].mean(),
            "mean_gap": g["return_gap_pred_minus_actual"].mean(),
            "median_gap": g["return_gap_pred_minus_actual"].median(),
            "positive_gap_share": (g["return_gap_pred_minus_actual"] > 0).mean(),
            "actual_asset_weighted": weighted_avg(g, actual_col, weight_col),
            "predicted_asset_weighted": weighted_avg(g, pred_col, weight_col),
            "asset_weighted_gap": (
                weighted_avg(g, pred_col, weight_col)
                - weighted_avg(g, actual_col, weight_col)
            ),
            "net_implied_gap_amount": g["implied_gap_amount"].sum(),
            "positive_only_gap_amount": g["positive_gap_amount"].sum(),
            "total_assets": g[weight_col].sum(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Support diagnostics
# ---------------------------------------------------------------------------

def support_diagnostics(train_df, pred_df, benchmark, numeric_cols,
                        industry_l2_col="二级行业"):
    """
    Common-support diagnostics: check how many SOE firms fall outside
    the benchmark feature distribution.

    Parameters
    ----------
    train_df : pd.DataFrame
        Benchmark (non-SOE) training data.
    pred_df : pd.DataFrame
        SOE prediction data.
    benchmark : str
        Benchmark name.
    numeric_cols : list of str
        Numeric feature columns to check.
    industry_l2_col : str

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for c in numeric_cols:
        if c not in train_df.columns or c not in pred_df.columns:
            continue
        train_s = train_df[c].dropna().astype(float)
        pred_s = pred_df[c].astype(float)
        p05, p95 = train_s.quantile(0.05), train_s.quantile(0.95)
        mn, mx = train_s.min(), train_s.max()
        rows.append({
            "benchmark": benchmark,
            "feature": c,
            "train_min": mn,
            "train_p05": p05,
            "train_p95": p95,
            "train_max": mx,
            "soe_outside_minmax_count": int(((pred_s < mn) | (pred_s > mx)).sum()),
            "soe_outside_p05_p95_count": int(((pred_s < p05) | (pred_s > p95)).sum()),
            "soe_outside_p05_p95_share": float(((pred_s < p05) | (pred_s > p95)).mean()),
        })

    # Industry coverage
    soe_ind = set(pred_df[industry_l2_col].dropna())
    train_ind = set(train_df[industry_l2_col].dropna())
    missing = sorted(soe_ind - train_ind)
    if missing:
        rows.append({
            "benchmark": benchmark,
            "feature": "二级行业 coverage",
            "soe_outside_minmax_count": int(pred_df[industry_l2_col].isin(missing).sum()),
            "soe_outside_p05_p95_count": int(pred_df[industry_l2_col].isin(missing).sum()),
            "soe_outside_p05_p95_share": float(pred_df[industry_l2_col].isin(missing).mean()),
            "missing_industries_in_benchmark": ", ".join(missing),
            "n_missing_industries": len(missing),
            "n_benchmark_industries": len(train_ind),
            "n_soe_industries": len(soe_ind),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data summary
# ---------------------------------------------------------------------------

def data_summary(panel, config):
    """
    Generate a unified data summary (replaces static target_distribution,
    feature_distribution, and benchmark_diagnostics CSVs).

    Parameters
    ----------
    panel : pd.DataFrame
        Full long-format panel.
    config : dict

    Returns
    -------
    pd.DataFrame
    """
    target_col = config["data"]["target_col"]
    bg = config["data"]["benchmark_groups"]
    num_features = config["features"]["numeric"]

    rows = []
    for group_label, group_df in panel.groupby("benchmark_group"):
        sub = group_df.dropna(subset=[target_col])
        if len(sub) == 0:
            continue
        row = {
            "sample": group_label,
            "n_total": len(group_df),
            "n_valid_target": len(sub),
            "target_mean": sub[target_col].mean(),
            "target_median": sub[target_col].median(),
            "target_p05": sub[target_col].quantile(0.05),
            "target_p95": sub[target_col].quantile(0.95),
            "target_min": sub[target_col].min(),
            "target_max": sub[target_col].max(),
        }
        # Feature distribution stats
        for f in num_features:
            if f in sub.columns:
                row[f"{f}_mean"] = sub[f].mean()
                row[f"{f}_median"] = sub[f].median()
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Write all outputs
# ---------------------------------------------------------------------------

def write_all_outputs(summaries_list, cv_rows_list, pred_long, panel,
                      benchmark_train_data, pred_df, config):
    """
    Write all output files (CSV + optional Excel workbook).

    Parameters
    ----------
    summaries_list : list of pd.DataFrame
        Summary DataFrames from each winsor level.
    cv_rows_list : list of pd.DataFrame
        CV metrics DataFrames from each winsor level.
    pred_long : pd.DataFrame
        Long-format predictions.
    panel : pd.DataFrame
        Full panel data.
    benchmark_train_data : dict
        Maps benchmark name -> training DataFrame.
    pred_df : pd.DataFrame
        SOE prediction DataFrame.
    config : dict
    """
    out_dir = BASE / config["output"]["dir"]
    out_dir.mkdir(exist_ok=True)
    enc = config["output"]["encoding"]
    num_features = config["features"]["numeric"]
    num_cols = [f"{f}_lag" for f in num_features]

    # --- Summaries ---
    all_summaries = pd.concat(summaries_list, ignore_index=True)
    main_summary = all_summaries[all_summaries["winsor"] == "5-95"]
    main_summary.to_csv(out_dir / "summary_main_5_95.csv", index=False, encoding=enc)
    all_summaries.to_csv(out_dir / "summary_robustness.csv", index=False, encoding=enc)

    # --- CV metrics ---
    all_cv = pd.concat(cv_rows_list, ignore_index=True)
    all_cv.to_csv(out_dir / "cv_metrics_robustness.csv", index=False, encoding=enc)

    # --- Predictions (long) ---
    pred_long.to_csv(out_dir / "predictions_main_long.csv", index=False, encoding=enc)

    # --- Predictions (compact) ---
    pred_base_cols = [
        "证券代码", "证券简称", "一级行业", "二级行业",
        "企业所有制性质", "资产总计",
    ]
    # Fall back to panel column names
    id_cols_config = config["data"]["id_cols"]
    firm_id_col = id_cols_config.get("firm_id", "证券代码")
    firm_name_col = id_cols_config.get("firm_name", "证券简称")
    l1_col = config["data"]["industry_l1_col"]
    l2_col = config["data"]["industry_l2_col"]
    soe_col = config["data"]["soe_col"]
    weight_col = config["data"]["weight_col"]
    actual_col = "actual_EBI_A"

    compact_base = [firm_id_col, firm_name_col, l1_col, l2_col, soe_col, weight_col]
    # Map to actual column names in pred_long
    compact_base_actual = [c for c in compact_base if c in pred_long.columns]
    if actual_col in pred_long.columns:
        compact_base_actual.append(actual_col)

    pivot_cols = compact_base_actual + [
        "benchmark", "predicted_EBI_A", "return_gap_pred_minus_actual",
        "implied_gap_amount", "positive_gap_amount",
    ]
    pivot_cols = [c for c in pivot_cols if c in pred_long.columns]

    pred_pivot = pred_long[pivot_cols].pivot_table(
        index=[c for c in compact_base_actual if c != "benchmark"],
        columns="benchmark",
        values=[c for c in [
            "predicted_EBI_A", "return_gap_pred_minus_actual",
            "implied_gap_amount", "positive_gap_amount",
        ] if c in pred_long.columns],
        aggfunc="first",
    ).reset_index()

    pred_pivot.columns = [
        c[0] if c[1] == "" else f"{c[0]}__{c[1]}"
        for c in pred_pivot.columns.to_flat_index()
    ]
    pred_pivot.to_csv(out_dir / "predictions_main_compact.csv", index=False, encoding=enc)

    # --- Group statistics ---
    if soe_col in pred_long.columns:
        group_stats(pred_long, soe_col, weight_col=weight_col).to_csv(
            out_dir / "group_ownership.csv", index=False, encoding=enc)
    if l2_col in pred_long.columns:
        group_stats(pred_long, l2_col, weight_col=weight_col).to_csv(
            out_dir / "group_industry.csv", index=False, encoding=enc)

    # --- Support diagnostics ---
    support_dfs = []
    for bname, bdf in benchmark_train_data.items():
        sd = support_diagnostics(bdf, pred_df, bname, num_cols, industry_l2_col=l2_col)
        support_dfs.append(sd)
    pd.concat(support_dfs, ignore_index=True).to_csv(
        out_dir / "support_diagnostics.csv", index=False, encoding=enc)

    # --- Data summary ---
    data_summary(panel, config).to_csv(
        out_dir / "data_summary.csv", index=False, encoding=enc)

    # --- Excel workbook ---
    if config["output"].get("include_excel", True):
        _write_excel(all_summaries, all_cv, pred_long, pred_pivot, out_dir, config)


def _write_excel(all_summaries, all_cv, pred_long, pred_pivot, out_dir, config):
    """Write multi-sheet Excel workbook."""
    import re

    xl_path = out_dir / "three_benchmark_counterfactual_results.xlsx"
    l1_col = config["data"]["industry_l1_col"]
    l2_col = config["data"]["industry_l2_col"]
    soe_col = config["data"]["soe_col"]
    weight_col = config["data"]["weight_col"]

    with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
        # README
        pd.DataFrame({
            "项目": ["模型", "目标变量", "Benchmarks", "预测框架", "Winsorization"],
            "说明": [
                "Ridge(alpha=10) with FEs",
                "EBI/A (t)",
                "H non-SOE / CSI non-SOE / Pooled",
                "X_{t-1} → Y_t (panel data)",
                "5%-95% (main) + robustness checks",
            ],
        }).to_excel(writer, sheet_name="README", index=False)

        all_summaries.to_excel(writer, sheet_name="Summary", index=False)
        all_cv.to_excel(writer, sheet_name="CV_Metrics", index=False)
        pred_pivot.to_excel(writer, sheet_name="Predictions_Compact", index=False)

        if soe_col in pred_long.columns:
            group_stats(pred_long, soe_col, weight_col=weight_col).to_excel(
                writer, sheet_name="Group_Ownership", index=False)
        if l2_col in pred_long.columns:
            group_stats(pred_long, l2_col, weight_col=weight_col).to_excel(
                writer, sheet_name="Group_Industry", index=False)
