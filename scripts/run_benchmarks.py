#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Three-Benchmark Counterfactual EBI/A Model — CLI Entry Point.

Usage:
  python scripts/run_benchmarks.py legacy          # Run with existing wide-format data
  python scripts/run_benchmarks.py run --mode panel # Run with new panel data
  python scripts/run_benchmarks.py register ...     # Register panel data files
  python scripts/run_benchmarks.py validate ...     # Validate panel data schema
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.data_loader import (
    load_config, load_legacy_data, load_panel_data,
    load_panel_file, validate_panel_schema,
    register_file, split_train_pred, load_yearly_data,
    merge_roa_roe,
)
from src.features import build_lagged_features, compute_support_flags
from src.model import build_pipeline, cv_metrics, fit_predict
from src.psm import (
    run_psm_matching, run_cem_matching,
    run_entropy_balancing, run_overlap_weighting,
)
from src.outputs import (
    summarize_predictions, group_stats, support_diagnostics,
    data_summary, write_all_outputs,
)
from src.utils import compute_gap_columns, weighted_avg


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks_panel(panel, config, winsor=(0.05, 0.95)):
    """
    Run all three benchmarks using panel data.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel data (from legacy bridge or panel files).
    config : dict
    winsor : tuple

    Returns
    -------
    summaries_df, cv_df, pred_long_df
    """
    target_col = config["data"]["target_col"]
    weight_col = config["data"]["weight_col"]
    num_features = config["features"]["numeric"]
    num_cols_lag = [f"{f}_lag" for f in num_features]
    l1_col = config["data"]["industry_l1_col"]
    l2_col = config["data"]["industry_l2_col"]
    market_col = config["data"]["market_col"]
    soe_col = config["data"]["soe_col"]
    bg = config["data"]["benchmark_groups"]

    # --- 1. Build lagged features ---
    panel_lag = build_lagged_features(panel, num_features, id_col="firm_id", year_col="year")

    # --- 2. Split into train/pred ---
    train_dict, pred_df = split_train_pred(panel_lag, config)

    # --- 3. Keep only rows with valid target for training ---
    # In legacy mode, target is only in year=target_year rows
    # In panel mode, target is in all rows
    for key in train_dict:
        train_dict[key] = train_dict[key][pd.notna(train_dict[key][target_col])].copy()

    # For prediction, we only care about rows with target (to compare actual vs predicted)
    pred_df = pred_df[pd.notna(pred_df[target_col])].copy()

    # --- 4. Prepare for each benchmark ---
    # Map benchmark name → training DataFrame
    benchmarks = {
        bg["h_non_soe"]: train_dict[bg["h_non_soe"]],
        bg["csi_non_soe"]: train_dict[bg["csi_non_soe"]],
        bg["pooled"]: train_dict[bg["pooled"]],
    }

    summaries = []
    cv_rows = []
    pred_parts = []

    winsor_label = f"{int(winsor[0]*100)}-{int(winsor[1]*100)}"

    for bname, train_df in benchmarks.items():
        # Determine categorical columns
        # For Pooled benchmark, add market dummy to distinguish H vs CSI
        cat_cols = [l1_col, l2_col]
        is_pooled = bname.startswith("Pooled")

        if is_pooled:
            # Add market_A binary dummy: 1 = A-share/CSI, 0 = H-share
            # This controls for A/H market environment in pooled benchmark.
            # When predicting SOE, market_A = 1 (A-share counterfactual).
            train_df = train_df.copy()
            train_df["market_A"] = (train_df[market_col].astype(str).str.upper() != "H").astype(float)
            num_cols_used = num_cols_lag + ["market_A"]

            pred_df_pooled = pred_df.copy()
            pred_df_pooled["market_A"] = 1.0  # SOE = A-share environment
        else:
            pred_df_pooled = pred_df.copy()
            num_cols_used = num_cols_lag

        # --- Matching/Weighting methods ---
        psm_cfg_block = config.get("psm", {})
        methods_to_run = psm_cfg_block.get("methods", ["psm"]) if psm_cfg_block.get("_active", False) else []
        method_results = {}

        train_original = train_df.copy()

        for method in methods_to_run:
            if method == "psm":
                mtrain, minfo = run_psm_matching(
                    train_original.copy(), pred_df_pooled.copy(),
                    covariate_cols=num_cols_used,
                    caliper_sd=psm_cfg_block.get("caliper_sd", 0.5),
                    classifier=psm_cfg_block.get("classifier", "gbm"),
                    random_state=psm_cfg_block.get("random_state", 42),
                )
                print(f"    PSM: {len(train_original)}->{len(mtrain)} nsOE "
                      f"(SMD max: {minfo.get('smd_before_max',0):.3f}->{minfo.get('smd_after_max',0):.3f})")
                method_results["psm"] = (mtrain, minfo)

            elif method == "cem":
                mtrain, minfo = run_cem_matching(
                    train_original.copy(), pred_df_pooled.copy(),
                    exact_cols=["year", l2_col],
                    coarsen_cols=[c for c in num_cols_used if c in train_original.columns],
                    coarsen_bins=5,
                )
                print(f"    CEM: {minfo['n_before']}->{minfo['n_after']} nsOE "
                      f"({minfo['n_strata']} strata, "
                      f"SMD: {minfo.get('smd_before_max',0):.3f}->{minfo.get('smd_after_max',0):.3f})")
                method_results["cem"] = (mtrain, minfo)

            elif method == "entropy":
                mtrain, minfo = run_entropy_balancing(
                    train_original.copy(), pred_df_pooled.copy(),
                    balance_cols=[c for c in num_cols_used if c in train_original.columns],
                )
                print(f"    EB: ESS={minfo.get('ess',0):.0f} (ratio={minfo.get('ess_ratio',0):.2f})")
                method_results["entropy"] = (mtrain, minfo)

            elif method == "overlap":
                mtrain, minfo = run_overlap_weighting(
                    train_original.copy(), pred_df_pooled.copy(),
                    covariate_cols=num_cols_used,
                    classifier=psm_cfg_block.get("classifier", "gbm"),
                    random_state=psm_cfg_block.get("random_state", 42),
                )
                print(f"    OW: ESS={minfo.get('ess',0):.0f} (ratio={minfo.get('ess_ratio',0):.2f})")
                method_results["overlap"] = (mtrain, minfo)

        # Run baseline + each method
        all_methods = [("baseline", train_original, None)] + [
            (name, mtrain, minfo) for name, (mtrain, minfo) in method_results.items()
        ]

        for method_label, use_train, method_info in all_methods:
            cv = cv_metrics(
                use_train, cat_cols, config,
                numeric_cols=num_cols_used, winsor=winsor,
                groups_col="firm_id",
            )
            cv_row = {"winsor": winsor_label, "benchmark": bname, "model": "Ridge",
                      "method": method_label}
            cv_row.update(cv)
            cv_rows.append(cv_row)

            _, preds, info = fit_predict(
                use_train, pred_df_pooled, cat_cols, config,
                numeric_cols=num_cols_used, winsor=winsor,
            )

            tmp = pred_df_pooled.copy()
            tmp["benchmark"] = bname
            tmp["predicted_EBI_A"] = preds
            tmp["actual_EBI_A"] = tmp[target_col]

            wcol = weight_col if weight_col in tmp.columns else "资产总计"
            if wcol not in tmp.columns:
                for candidate in ["资产总计", "assets", "Assets"]:
                    if candidate in tmp.columns:
                        wcol = candidate
                        break

            tmp = compute_gap_columns(tmp, "predicted_EBI_A", "actual_EBI_A", wcol)

            cs_cfg_block2 = config.get("common_support", {})
            if cs_cfg_block2.get("enabled", True) and method_label == "baseline":
                cs_vars = cs_cfg_block2.get("core_variables", [
                    "FirmSize_lag", "FinancialLeverage_lag", "AssetTurnover_lag",
                    "FixedAssetsRatio_lag", "CapexRatio_lag", "HHI_lag",
                    "MarketShare_lag", "CurrentRatio_lag",
                ])
                cs_bounds = tuple(cs_cfg_block2.get("bounds", [0.01, 0.99]))
                tmp, cs_info = compute_support_flags(
                    train_original.copy(), tmp, cs_vars, bounds=cs_bounds)
                tmp["benchmark"] = bname
            elif method_label == "baseline":
                cs_info = None

            if method_label == "baseline":
                pred_parts.append(tmp)

            summaries.append(summarize_predictions(
                tmp, bname, info, winsor_label,
                target_col=target_col, weight_col=wcol,
                sample_label=method_label,
            ))

            if method_label == "baseline" and cs_info is not None and cs_info["n_supported"] > 0:
                tmp_cs = tmp[tmp["_support_flag"] == 1].copy()
                summaries.append(summarize_predictions(
                    tmp_cs, bname, info, winsor_label,
                    target_col=target_col, weight_col=wcol,
                    sample_label="common_support",
                ))

    return pd.DataFrame(summaries), pd.DataFrame(cv_rows), pd.concat(pred_parts, ignore_index=True)


def run_legacy_mode(config):
    """Run benchmarks in legacy mode (original wide-format data)."""
    print("=" * 60)
    print("Legacy 模式：使用现有 wide-format 数据运行")
    print("=" * 60)

    panel = load_legacy_data(config)
    print(f"加载数据: {len(panel)} 行 (long format)")

    # Run main + robustness winsor levels
    main_winsor = tuple(config["winsorization"]["main"])
    robust_winsors = [tuple(w) for w in config["winsorization"]["robustness"]]

    all_summaries = []
    all_cv = []

    for i, w in enumerate([main_winsor] + robust_winsors):
        label = f"{int(w[0]*100)}-{int(w[1]*100)}"
        print(f"  运行 winsor={label} ...")
        s, cv, pred_long = run_benchmarks_panel(panel, config, winsor=w)
        all_summaries.append(s)
        all_cv.append(cv)
        if i == 0:
            main_pred = pred_long
            main_summary = s

    # --- Prepare benchmark_train_data for diagnostics ---
    panel_lag = build_lagged_features(panel, config["features"]["numeric"],
                                       id_col="firm_id", year_col="year")
    train_dict, pred_df = split_train_pred(panel_lag, config)
    target_col = config["data"]["target_col"]
    for key in train_dict:
        train_dict[key] = train_dict[key][pd.notna(train_dict[key][target_col])].copy()
    pred_df = pred_df[pd.notna(pred_df[target_col])].copy()

    # Map to output-friendly benchmark names
    bg = config["data"]["benchmark_groups"]
    benchmark_train_data = {
        "H_nonSOE_only": train_dict[bg["h_non_soe"]],
        "CSI_nonSOE_only": train_dict[bg["csi_non_soe"]],
        "Pooled_H_plus_CSI_nonSOE": train_dict[bg["pooled"]],
    }

    # --- Write outputs ---
    print("\n写入输出文件...")
    write_all_outputs(
        all_summaries, all_cv, main_pred, panel,
        benchmark_train_data, pred_df, config,
    )
    print("\n✓ Legacy 模式运行完成")
    print(main_summary.to_string(index=False))


def run_panel_mode(config, file_paths=None):
    """Run benchmarks in panel mode (new long-format data)."""
    print("=" * 60)
    print("Panel 模式：使用 long-format panel 数据运行")
    print("=" * 60)

    # Load panel data (with industry filter applied automatically)
    panel = load_panel_data(config, file_paths=file_paths)

    print(f"加载数据: {len(panel)} 行, {panel['firm_id'].nunique()} 家企业")
    print(f"年份范围: {sorted(panel['year'].dropna().unique())}")

    # Validate schema
    validation = validate_panel_schema(panel)
    for msg in validation["messages"]:
        print(f"  {msg}")
    if not validation["valid"]:
        print("⚠ Schema 校验失败，请修正后重试。")
        return

    # Run benchmarks
    main_winsor = tuple(config["winsorization"]["main"])
    robust_winsors = [tuple(w) for w in config["winsorization"]["robustness"]]

    all_summaries = []
    all_cv = []

    for i, w in enumerate([main_winsor] + robust_winsors):
        label = f"{int(w[0]*100)}-{int(w[1]*100)}"
        print(f"  运行 winsor={label} ...")
        s, cv, pred_long = run_benchmarks_panel(panel, config, winsor=w)
        all_summaries.append(s)
        all_cv.append(cv)
        if i == 0:
            main_pred = pred_long
            main_summary = s

    # Benchmark train data for diagnostics
    panel_lag = build_lagged_features(panel, config["features"]["numeric"],
                                       id_col="firm_id", year_col="year")
    train_dict, pred_df = split_train_pred(panel_lag, config)
    target_col = config["data"]["target_col"]
    for key in train_dict:
        train_dict[key] = train_dict[key][pd.notna(train_dict[key][target_col])].copy()
    pred_df = pred_df[pd.notna(pred_df[target_col])].copy()

    bg = config["data"]["benchmark_groups"]
    benchmark_train_data = {
        "H_nonSOE_only": train_dict[bg["h_non_soe"]],
        "CSI_nonSOE_only": train_dict[bg["csi_non_soe"]],
        "Pooled_H_plus_CSI_nonSOE": train_dict[bg["pooled"]],
    }

    # Write outputs
    print("\n写入输出文件...")
    write_all_outputs(
        all_summaries, all_cv, main_pred, panel,
        benchmark_train_data, pred_df, config,
    )
    print("\n✓ Panel 模式运行完成")
    print(main_summary.to_string(index=False))


def run_yearly_mode(config, years=None):
    """Run benchmarks for all targets (EBI/A + ROA + ROE)."""
    print("=" * 60)
    print("Yearly 模式：使用年份文件夹数据运行")
    print("=" * 60)

    # Load year-foldered data
    panel = load_yearly_data(config, years=years)

    # Merge ROA/ROE data
    panel = merge_roa_roe(panel, config)

    print(f"加载数据: {len(panel)} 行 (long format), "
          f"{panel['firm_id'].nunique()} 家企业")

    # Run all targets
    targets_cfg = config["data"].get("targets", {
        "EBI_A": {"col": "EBI_A", "label": "EBI/A", "primary": True},
    })

    base_out_dir = config["output"]["dir"]

    for target_key, target_info in targets_cfg.items():
        target_col = target_info["col"]
        target_label = target_info["label"]
        is_primary = target_info.get("primary", False)

        # Activate matching/weighting methods
        psm_enabled = config.get("psm", {}).get("enabled", False)
        config["psm"]["_active"] = psm_enabled

        star = " ★" if is_primary else ""
        method_str = " [+ PSM/CEM/EB/OW]" if psm_enabled else ""
        print(f"\n{'─' * 50}")
        print(f"  目标变量: {target_label} ({target_col}){star}{method_str}")
        print(f"{'─' * 50}")

        config["data"]["target_col"] = target_col
        config["data"]["weight_col"] = "资产总计"
        print(f"  有效观测: {panel[target_col].notna().sum()} 行有 {target_label} 值")

        main_winsor = tuple(config["winsorization"]["main"])
        robust_winsors = [tuple(w) for w in config["winsorization"]["robustness"]]

        all_summaries = []
        all_cv = []

        for i, w in enumerate([main_winsor] + robust_winsors):
            label = f"{int(w[0]*100)}-{int(w[1]*100)}"
            print(f"    运行 winsor={label} ...")
            s, cv, pred_long = run_benchmarks_panel(panel, config, winsor=w)
            all_summaries.append(s)
            all_cv.append(cv)
            if i == 0:
                main_pred = pred_long
                main_summary = s

        panel_lag = build_lagged_features(panel, config["features"]["numeric"],
                                           id_col="firm_id", year_col="year")
        train_dict, pred_df = split_train_pred(panel_lag, config)
        for key in train_dict:
            train_dict[key] = train_dict[key][pd.notna(train_dict[key][target_col])].copy()
        pred_df = pred_df[pd.notna(pred_df[target_col])].copy()

        bg = config["data"]["benchmark_groups"]
        benchmark_train_data = {
            "H_nonSOE_only": train_dict[bg["h_non_soe"]],
            "CSI_nonSOE_only": train_dict[bg["csi_non_soe"]],
            "Pooled_H_plus_CSI_nonSOE": train_dict[bg["pooled"]],
        }

        out_subdir = f"{base_out_dir}/{target_key}"
        config["output"]["dir"] = out_subdir
        print(f"\n  写入输出文件 → {out_subdir}/")
        write_all_outputs(
            all_summaries, all_cv, main_pred, panel,
            benchmark_train_data, pred_df, config,
        )
        print(f"  ✓ {target_label} 完成")
        print(main_summary.to_string(index=False))

    # Restore output dir
    config["output"]["dir"] = base_out_dir
    config["data"]["target_col"] = "EBI_A"
    print(f"\n✓ Yearly 模式全部完成 ({len(targets_cfg)} 个目标变量)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Three-Benchmark Counterfactual EBI/A Model (Panel Data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmarks.py legacy
  python scripts/run_benchmarks.py run --mode panel
  python scripts/run_benchmarks.py register --file data_input/panel_2018.csv --years 2018
  python scripts/run_benchmarks.py validate --file data_input/panel.csv
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Commands")

    # --- legacy ---
    leg = sub.add_parser("legacy", help="Run with original wide-format data")
    leg.add_argument("--config", default=None, help="Config YAML path")
    leg.add_argument("--output-dir", default=None, help="Override output directory")

    # --- run ---
    run = sub.add_parser("run", help="Run benchmark pipeline")
    run.add_argument("--mode", choices=["legacy", "panel"], default="panel",
                     help="Data mode (default: panel)")
    run.add_argument("--config", default=None, help="Config YAML path")
    run.add_argument("--file", nargs="*", default=None,
                     help="Panel data file(s) to use (bypasses manifest)")
    run.add_argument("--output-dir", default=None, help="Override output directory")
    run.add_argument("--alpha", type=float, default=None, help="Override Ridge alpha")
    run.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # --- register ---
    reg = sub.add_parser("register", help="Register panel data file(s) in manifest")
    reg.add_argument("--file", required=True, help="Data file path")
    reg.add_argument("--years", required=True,
                     help="Comma-separated years, e.g. '2018,2019,2020'")
    reg.add_argument("--config", default=None, help="Config YAML path")

    # --- yearly ---
    yr = sub.add_parser("yearly", help="Run with yearly folder data (e.g. 2023)")
    yr.add_argument("--years", nargs="*", type=int, default=None,
                    help="Years to load (e.g. 2023). Default: auto-detect all.")
    yr.add_argument("--config", default=None, help="Config YAML path")
    yr.add_argument("--output-dir", default=None, help="Override output directory")
    yr.add_argument("--alpha", type=float, default=None, help="Override Ridge alpha")
    yr.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # --- validate ---
    val = sub.add_parser("validate", help="Validate panel data schema")
    val.add_argument("--file", required=True, help="Panel data file to validate")
    val.add_argument("--config", default=None, help="Config YAML path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Load config
    config = load_config(args.config)

    # Override output dir if specified
    if hasattr(args, "output_dir") and args.output_dir:
        config["output"]["dir"] = args.output_dir

    # Override alpha if specified
    if hasattr(args, "alpha") and args.alpha:
        config["model"]["params"]["alpha"] = args.alpha

    # Dispatch
    if args.command == "legacy":
        run_legacy_mode(config)

    elif args.command == "run":
        if args.mode == "legacy":
            run_legacy_mode(config)
        else:
            run_panel_mode(config, file_paths=args.file)

    elif args.command == "register":
        years = [int(y.strip()) for y in args.years.split(",")]
        manifest = register_file(args.file, years)
        print(f"✓ 已注册: {args.file}")
        print(f"  年份: {years}")
        print(f"  Manifest: {len(manifest['files'])} 个文件")

    elif args.command == "yearly":
        run_yearly_mode(config, years=args.years)

    elif args.command == "validate":
        df = load_panel_file(args.file)
        result = validate_panel_schema(df)
        print(f"文件: {args.file}")
        print(f"行数: {len(df)}, 列数: {len(df.columns)}")
        for msg in result["messages"]:
            print(f"  {msg}")
        if result["missing"]:
            print(f"  缺少列: {result['missing']}")
        if result["valid"]:
            print("✓ Schema 校验通过")
        else:
            print("✗ Schema 校验失败")


if __name__ == "__main__":
    main()
