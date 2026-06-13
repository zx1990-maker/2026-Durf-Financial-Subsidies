#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading: legacy wide-format bridge and panel data interface.

Supports two modes:
  - legacy: reads the 3 existing wide-format files, normalizes, converts to panel
  - panel:  reads long-format panel CSV/Excel via manifest or direct path
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from .utils import to_numeric_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Load YAML configuration, resolving paths relative to project root."""
    if path is None:
        path = BASE / "config" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Legacy: wide-to-long bridge
# ---------------------------------------------------------------------------

# Columns shared across all legacy files after normalization
LEGACY_COMMON_COLS = [
    "市场", "证券代码", "证券简称", "一级行业", "二级行业", "行业全路径", "企业所有制性质",
    "资产总计_2024", "营业收入_2024", "归母净利润_2025", "利息支出_2025", "资产总计_2025",
    "EBI_A_2025", "FirmSize_2024", "MarketShare_2024_二级行业", "HHI_2024_二级行业",
    "AssetTurnover_2024", "WorkingCapitalRatio_2024", "FixedAssetsRatio_2024",
    "FinancialLeverage_2024", "CurrentRatio_2024", "CapexRatio_2024",
]

# Map legacy column suffixes to feature names and target name
LEGACY_FEATURE_YEARS = [2024]
LEGACY_TARGET_YEAR = 2025

# Columns that carry a year suffix and map to feature names
# Map feature names to their column name *without* year suffix.
# For features like MarketShare_YYYY_二级行业, the template is f"{base}_{year}_{suffix}"
LEGACY_NUMERIC_FEATURES = [
    # (feature_name, column_base, column_suffix_after_year)
    # Column name = f"{column_base}_{year}_{column_suffix}" if suffix else f"{column_base}_{year}"
    ("FirmSize",           "FirmSize",           ""),
    ("MarketShare",        "MarketShare",         "二级行业"),
    ("HHI",                "HHI",                 "二级行业"),
    ("AssetTurnover",      "AssetTurnover",       ""),
    ("WorkingCapitalRatio","WorkingCapitalRatio", ""),
    ("FixedAssetsRatio",   "FixedAssetsRatio",    ""),
    ("FinancialLeverage",  "FinancialLeverage",   ""),
    ("CurrentRatio",       "CurrentRatio",        ""),
    ("CapexRatio",         "CapexRatio",          ""),
]

# Columns that map to target/assets with year suffix
LEGACY_TARGET_COL = "EBI_A"
LEGACY_ASSETS_COL = "资产总计"
LEGACY_REVENUE_COL = "营业收入"
LEGACY_NET_PROFIT_COL = "归母净利润"
LEGACY_INTEREST_COL = "利息支出"


def _normalize_old(df, market_label):
    """Normalize CSI SOE or H non-SOE CSV (original normalize_old logic)."""
    df = df.copy()
    if "EBI_A_2025" not in df.columns and "EBI/A_2025" in df.columns:
        df["EBI_A_2025"] = to_numeric_series(df["EBI/A_2025"])
    df["市场"] = market_label

    if "行业全路径" in df.columns:
        if "一级行业" not in df.columns:
            df["一级行业"] = df["行业全路径"].astype(str).str.split("--").str[0]
        if "二级行业" not in df.columns:
            df["二级行业"] = df["行业全路径"].astype(str).str.split("--").str[1]

    for c in LEGACY_COMMON_COLS:
        if c in df.columns and c not in [
            "市场", "证券代码", "证券简称", "一级行业", "二级行业",
            "行业全路径", "企业所有制性质",
        ]:
            df[c] = to_numeric_series(df[c])
    return df[LEGACY_COMMON_COLS].copy()


def _normalize_csi_ns(df):
    """Normalize CSI non-SOE Excel data (original normalize_csi_ns logic)."""
    df = df.copy().rename(columns={"EBI/A_2025": "EBI_A_2025"})
    df["市场"] = "CSI"
    df["一级行业"] = df["行业全路径"].astype(str).str.split("--").str[0]
    for c in LEGACY_COMMON_COLS:
        if c in df.columns and c not in [
            "市场", "证券代码", "证券简称", "一级行业", "二级行业",
            "行业全路径", "企业所有制性质",
        ]:
            df[c] = to_numeric_series(df[c])
    return df[LEGACY_COMMON_COLS].copy()


def wide_to_long(df_wide, feature_years=None, target_year=None):
    """
    Convert legacy wide-format data to long panel format.

    A single firm row with columns FirmSize_2024, EBI_A_2025, etc.
    becomes two rows:
      - year = feature_year:  numeric features populated, target = NaN
      - year = target_year:   target + assets populated, numeric features = NaN
                              (they will be filled by lag in the panel pipeline)

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format data (output of _normalize_old / _normalize_csi_ns).
    feature_years : list of int
        Years whose suffix columns supply features (e.g. [2024]).
    target_year : int
        Year whose suffix columns supply target/assets (e.g. 2025).

    Returns
    -------
    pd.DataFrame in long format with columns:
        firm_id, firm_name, market, soe_label, industry_l1, industry_l2,
        industry_full, year, benchmark_group,
        EBI_A, 资产总计, 营业收入, 归母净利润, 利息支出,
        FirmSize, MarketShare, HHI, AssetTurnover, WorkingCapitalRatio,
        FixedAssetsRatio, FinancialLeverage, CurrentRatio, CapexRatio
    """
    if feature_years is None:
        feature_years = LEGACY_FEATURE_YEARS
    if target_year is None:
        target_year = LEGACY_TARGET_YEAR

    rows = []
    for _, firm in df_wide.iterrows():
        firm_base = {
            "firm_id": firm.get("证券代码", None),
            "firm_name": firm.get("证券简称", None),
            "市场": firm.get("市场", None),
            "企业所有制性质": firm.get("企业所有制性质", None),
            "一级行业": firm.get("一级行业", None),
            "二级行业": firm.get("二级行业", None),
            "行业全路径": firm.get("行业全路径", None),
        }

        # Feature year row(s)
        for fy in feature_years:
            row = {**firm_base, "year": fy}
            for feat_name, col_base, col_suffix in LEGACY_NUMERIC_FEATURES:
                if col_suffix:
                    col = f"{col_base}_{fy}_{col_suffix}"
                else:
                    col = f"{col_base}_{fy}"
                row[feat_name] = firm.get(col, np.nan) if col in firm.index else np.nan
            # Assets / revenue for feature year
            for lbl, legacy_base in [
                ("资产总计", LEGACY_ASSETS_COL),
                ("营业收入", LEGACY_REVENUE_COL),
            ]:
                col = f"{legacy_base}_{fy}"
                row[lbl] = firm.get(col, np.nan) if col in firm.index else np.nan
            # Target not available for feature year
            row["EBI_A"] = np.nan
            row["归母净利润"] = np.nan
            row["利息支出"] = np.nan
            row["_row_type"] = "feature_year"
            rows.append(row)

        # Target year row
        row_t = {**firm_base, "year": target_year}
        for feat_name in LEGACY_NUMERIC_FEATURES:
            row_t[feat_name] = np.nan  # filled by lag
        row_t["EBI_A"] = firm.get(f"EBI_A_{target_year}", np.nan)
        row_t["资产总计"] = firm.get(f"资产总计_{target_year}", np.nan)
        row_t["营业收入"] = np.nan
        row_t["归母净利润"] = firm.get(f"归母净利润_{target_year}", np.nan)
        row_t["利息支出"] = firm.get(f"利息支出_{target_year}", np.nan)
        row_t["_row_type"] = "target_year"
        rows.append(row_t)

    long = pd.DataFrame(rows)
    return long


def load_legacy_data(config):
    """
    Load all three legacy files, normalize, and convert to long panel format.

    Returns
    -------
    panel : pd.DataFrame
        Long-format panel with all firms (SOE + non-SOE) from all three files.
        Contains columns: firm_id, firm_name, market, soe_label,
        industry_l1, industry_l2, industry_full, year, EBI_A, 资产总计,
        FirmSize, MarketShare, HHI, AssetTurnover, WorkingCapitalRatio,
        FixedAssetsRatio, FinancialLeverage, CurrentRatio, CapexRatio.
    """
    lc = config["data"]["legacy"]
    feat_years = lc.get("feature_years", LEGACY_FEATURE_YEARS)
    tgt_year = lc.get("target_year", LEGACY_TARGET_YEAR)

    # CSI SOE (target group)
    csi_soe_path = BASE / lc["csi_soe_path"]
    csi_soe_raw = pd.read_csv(csi_soe_path)
    csi_soe = _normalize_old(csi_soe_raw, "CSI")
    csi_soe["_source"] = "SOE_target"

    # H non-SOE (benchmark)
    h_ns_path = BASE / lc["h_ns_path"]
    h_ns_raw = pd.read_csv(h_ns_path)
    h_ns = _normalize_old(h_ns_raw, "H")
    h_ns["_source"] = "H_nonSOE"

    # CSI non-SOE (benchmark)
    csi_ns_path = BASE / lc["csi_ns_path"]
    csi_ns_sheet = lc.get("csi_ns_sheet", "计算结果_数值版")
    csi_ns_raw = pd.read_excel(csi_ns_path, sheet_name=csi_ns_sheet)
    csi_ns = _normalize_csi_ns(csi_ns_raw)
    csi_ns["_source"] = "CSI_nonSOE"

    # Convert each to long format and stack
    panels = []
    for src_label, df_wide in [
        ("SOE_target", csi_soe),
        ("H_nonSOE", h_ns),
        ("CSI_nonSOE", csi_ns),
    ]:
        long = wide_to_long(df_wide, feature_years=feat_years, target_year=tgt_year)
        long["benchmark_group"] = src_label
        panels.append(long)

    panel = pd.concat(panels, ignore_index=True)

    # Apply industry filter
    exclude = config["data"].get("exclude_industries", [])
    if exclude:
        panel = filter_industries(panel, exclude, industry_col="二级行业")

    return panel


# ---------------------------------------------------------------------------
# Yearly folder data loading
# ---------------------------------------------------------------------------

def _normalize_yearly(df, market_label, feature_year, target_year):
    """
    Normalize a yearly CSV file with flexible year suffixes.

    Similar to _normalize_old but handles the year dynamically (e.g., 2023→2024).
    """
    df = df.copy()

    # Rename EBI/A column if needed
    ebi_col = f"EBI_A_{target_year}"
    ebi_alt = f"EBI/A_{target_year}"
    if ebi_col not in df.columns and ebi_alt in df.columns:
        df[ebi_col] = to_numeric_series(df[ebi_alt])

    df["市场"] = market_label

    # Extract industry columns from industry full path
    if "行业全路径" in df.columns:
        if "一级行业" not in df.columns:
            df["一级行业"] = df["行业全路径"].astype(str).str.split("--").str[0]
        if "二级行业" not in df.columns:
            df["二级行业"] = df["行业全路径"].astype(str).str.split("--").str[1]

    # Define expected columns dynamically
    id_cols = ["市场", "证券代码", "证券简称", "一级行业", "二级行业",
               "行业全路径", "企业所有制性质"]
    num_suffixes_f = [  # feature-year columns
        f"资产总计_{feature_year}", f"营业收入_{feature_year}",
    ]
    num_suffixes_t = [  # target-year columns
        f"归母净利润_{target_year}", f"利息支出_{target_year}",
        f"资产总计_{target_year}", ebi_col,
    ]
    feature_ratio_cols = [
        f"FirmSize_{feature_year}",
        f"MarketShare_{feature_year}_二级行业",
        f"HHI_{feature_year}_二级行业",
        f"AssetTurnover_{feature_year}",
        f"WorkingCapitalRatio_{feature_year}",
        f"FixedAssetsRatio_{feature_year}",
        f"FinancialLeverage_{feature_year}",
        f"CurrentRatio_{feature_year}",
        f"CapexRatio_{feature_year}",
    ]

    all_cols = id_cols + num_suffixes_f + num_suffixes_t + feature_ratio_cols

    # Convert numeric columns
    for c in all_cols:
        if c in df.columns and c not in id_cols:
            df[c] = to_numeric_series(df[c])

    # Return only columns that exist
    keep_cols = [c for c in all_cols if c in df.columns]
    return df[keep_cols].copy()


def load_yearly_data(config, years=None):
    """
    Load wide-format data from yearly subfolders, filter financials,
    split into SOE/non-SOE, and convert to panel format.

    Each year folder should contain:
      - csi_features_YYYY.csv  (all CSI firms: SOE + non-SOE)
      - h_features_YYYY.csv    (all H firms)

    Parameters
    ----------
    config : dict
    years : list of int, optional
        Years to load. If None, loads all available year folders.

    Returns
    -------
    panel : pd.DataFrame
        Long-format panel with benchmark_group assigned.
    """
    yc = config["data"].get("yearly", {})
    base_dir = BASE / yc.get("base_dir", "data_input")

    if years is None:
        # Auto-detect year folders
        years = sorted([
            int(d.name) for d in base_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

    if not years:
        raise ValueError(f"在 {base_dir} 中未找到年份文件夹。")

    print(f"加载年份: {years}")

    soe_types = config["data"].get("soe_types", ["地方国有企业", "中央国有企业"])
    exclude = config["data"].get("exclude_industries", [])

    all_panels = []

    for year in years:
        year_dir = base_dir / str(year)
        csi_new = year_dir / f"csi_features_{year}.csv"
        h_new = year_dir / f"h_features_{year}.csv"

        # --- Detect format: new (2 mixed files) vs old (3 separated files) ---
        if csi_new.exists() and h_new.exists():
            # New format: all firms mixed per market
            feature_year = year
            target_year = year + 1
            csi_raw = pd.read_csv(csi_new)
            h_raw = pd.read_csv(h_new)
            csi_df = _normalize_yearly(csi_raw, "CSI", feature_year, target_year)
            h_df = _normalize_yearly(h_raw, "H", feature_year, target_year)
            is_old_format = False

        else:
            # Old format: separate SOE, H non-SOE, CSI non-SOE files
            csi_soe_file = year_dir / "csi——features.csv"
            h_ns_file = year_dir / "h——features.csv"
            csi_ns_file = year_dir / f"csi_ns_{year}_features_{year+1}_EBIA_二级行业HHI_结果(1).xlsx"

            if not (csi_soe_file.exists() and h_ns_file.exists()):
                print(f"  ⚠ {year} 年数据不完整（新旧格式均未找到），跳过")
                continue

            # Detect years from column suffixes in old format
            feature_year = year
            target_year = year + 1

            print(f"  {year}: 检测到旧格式，转换为新格式...")

            # Load old-format files
            csi_soe_raw = pd.read_csv(csi_soe_file)
            h_ns_raw = pd.read_csv(h_ns_file)

            csi_soe_df = _normalize_old(csi_soe_raw, "CSI")
            h_ns_df = _normalize_old(h_ns_raw, "H")

            if csi_ns_file.exists():
                csi_ns_raw = pd.read_excel(csi_ns_file, sheet_name="计算结果_数值版")
                csi_ns_df = _normalize_csi_ns(csi_ns_raw)
            else:
                csi_ns_df = pd.DataFrame(columns=csi_soe_df.columns)

            # Mark source for old-format group assignment
            csi_soe_df["_src"] = "SOE_target"
            csi_ns_df["_src"] = "CSI_nonSOE"
            h_ns_df["_src"] = "H_nonSOE"
            is_old_format = True

            # Merge CSI SOE + CSI non-SOE into one CSI DataFrame
            csi_df = pd.concat([csi_soe_df, csi_ns_df], ignore_index=True)
            h_df = h_ns_df

        # Combine and filter
        combined = pd.concat([csi_df, h_df], ignore_index=True)
        n_before = len(combined)

        # Apply industry filter
        if exclude:
            combined = filter_industries(combined, exclude, industry_col="二级行业")

        n_after = len(combined)
        if n_before != n_after:
            print(f"  {year}: {n_before} → {n_after} 行 (过滤金融类后)")

        # Assign benchmark_group
        if is_old_format:
            # Old format: group by file source (按文件来源)
            combined["benchmark_group"] = combined["_src"]
            combined = combined.drop(columns=["_src"])
        else:
            # New format: explicit SOE/non-SOE classification
            non_soe_types = config["data"].get("non_soe_types", ["民营企业"])

            def _assign_group(row):
                own = row.get("企业所有制性质", "")
                market = row.get("市场", "")
                if own in soe_types and market == "CSI":
                    return "SOE_target"
                elif own in non_soe_types and market == "H":
                    return "H_nonSOE"
                elif own in non_soe_types and market == "CSI":
                    return "CSI_nonSOE"
                else:
                    return "_excluded"
            combined["benchmark_group"] = combined.apply(_assign_group, axis=1)

        # Remove excluded firms (H SOE, 公众企业, 外资企业, 集体企业, etc.)
        n_excluded = (combined["benchmark_group"] == "_excluded").sum()
        if n_excluded > 0:
            excluded_types = combined[combined["benchmark_group"] == "_excluded"]["企业所有制性质"].value_counts().to_dict()
            print(f"  {year}: 剔除 {n_excluded} 行 (非目标/非民营: {excluded_types})")
        combined = combined[combined["benchmark_group"] != "_excluded"].copy()

        # Convert to long panel format
        long = wide_to_long(
            combined,
            feature_years=[feature_year],
            target_year=target_year,
        )
        # wide_to_long sets benchmark_group from the input, but it's overridden
        # by the column we just set. However wide_to_long doesn't pass through
        # benchmark_group from the wide df. Let's re-merge it.
        # Actually, wide_to_long creates new rows without benchmark_group.
        # We need to add it back from the original combined df.
        # Use firm_id + year to merge benchmark_group back.
        # But wide_to_long uses firm_id and year, and benchmark_group is per firm.
        # Let's add benchmark_group after wide_to_long by joining on firm_id.
        firm_groups = combined[["证券代码", "benchmark_group"]].drop_duplicates()
        firm_groups = firm_groups.rename(columns={"证券代码": "firm_id"})
        long = long.merge(firm_groups, on="firm_id", how="left")

        all_panels.append(long)

    if not all_panels:
        raise ValueError("未成功加载任何年份的数据。")

    panel = pd.concat(all_panels, ignore_index=True)

    # Report
    for grp in ["SOE_target", "H_nonSOE", "CSI_nonSOE"]:
        n = len(panel[panel["benchmark_group"] == grp])
        print(f"  {grp}: {n} 行")

    return panel


# ---------------------------------------------------------------------------
# Industry filter
# ---------------------------------------------------------------------------

def filter_industries(df, exclude_list, industry_col="二级行业"):
    """
    Remove rows whose industry (secondary/二级行业) is in the exclude list.

    Parameters
    ----------
    df : pd.DataFrame
    exclude_list : list of str
        Industry names to exclude (e.g. ['银行', '保险', '证券']).
    industry_col : str
        Column name for industry classification.

    Returns
    -------
    pd.DataFrame with financial firms removed.
    """
    if industry_col not in df.columns:
        return df

    before = len(df)
    mask = ~df[industry_col].astype(str).isin(exclude_list)
    df_filtered = df[mask].copy()
    after = len(df_filtered)
    dropped = before - after
    if dropped > 0:
        print(f"  已过滤金融类公司: {dropped} 行 "
              f"(二级行业 ∈ {exclude_list})")
    return df_filtered


# ---------------------------------------------------------------------------
# Panel mode
# ---------------------------------------------------------------------------

PANEL_REQUIRED_COLS = [
    "firm_id", "firm_name", "year", "market", "benchmark_group",
    "industry_l1", "industry_l2", "EBI_A", "资产总计",
    "FirmSize", "MarketShare", "HHI", "AssetTurnover",
    "WorkingCapitalRatio", "FixedAssetsRatio", "FinancialLeverage",
    "CurrentRatio", "CapexRatio",
]

PANEL_OPTIONAL_COLS = [
    "soe_label", "industry_full", "营业收入", "归母净利润", "利息支出",
    "ROA", "ROE", "revenue", "net_profit", "interest_expense",
]


def validate_panel_schema(df):
    """
    Validate that a DataFrame conforms to the panel schema.

    Returns
    -------
    dict with keys: valid (bool), missing (list), extra (list), messages (list)
    """
    result = {"valid": True, "missing": [], "extra": [], "messages": []}
    existing = set(df.columns)

    missing = [c for c in PANEL_REQUIRED_COLS if c not in existing]
    if missing:
        result["valid"] = False
        result["missing"] = missing
        result["messages"].append(f"缺少必需列: {missing}")

    # Check year range
    if "year" in existing:
        years = sorted(df["year"].dropna().unique())
        result["messages"].append(f"检测到年份: {years}")
        if len(years) >= 2:
            year_gaps = set(range(years[0], years[-1] + 1)) - set(years)
            if year_gaps:
                result["messages"].append(f"⚠ 年份不连续，缺少: {sorted(year_gaps)}")

    # Check benchmark groups
    if "benchmark_group" in existing:
        groups = df["benchmark_group"].dropna().unique()
        result["messages"].append(f"检测到分组: {sorted(groups)}")

    # Check for null firm_ids
    if "firm_id" in existing:
        n_null = df["firm_id"].isna().sum()
        if n_null:
            result["valid"] = False
            result["messages"].append(f"firm_id 有 {n_null} 个缺失值")

    return result


def load_panel_file(file_path):
    """Load a single panel data file (CSV or Excel)."""
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = BASE / file_path

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")


def load_panel_from_files(file_paths):
    """Load and concatenate multiple panel files."""
    frames = []
    for fp in file_paths:
        df = load_panel_file(fp)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_panel_from_manifest(manifest_path=None):
    """
    Load panel data from all files registered in manifest.json.

    manifest.json format:
    {
      "files": [
        {"path": "data_input/panel/panel_2018_2020.csv", "years": [2018, 2019, 2020]},
        {"path": "data_input/panel/panel_2021.csv", "years": [2021]}
      ]
    }
    """
    if manifest_path is None:
        manifest_path = BASE / "data_input" / "manifest.json"

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest 文件不存在: {manifest_path}\n"
            f"请先用 'register' 命令注册数据文件，或使用 '--file' 直接指定文件。"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    file_paths = [entry["path"] for entry in manifest.get("files", [])]
    if not file_paths:
        raise ValueError("Manifest 中没有注册任何文件。")

    return load_panel_from_files(file_paths)


def load_panel_data(config, file_paths=None):
    """
    Unified panel data loader with industry filter.

    Parameters
    ----------
    config : dict
    file_paths : list of str, optional
        Direct file paths (bypasses manifest).

    Returns
    -------
    pd.DataFrame (filtered)
    """
    if file_paths:
        panel = load_panel_from_files(file_paths)
    else:
        panel = load_panel_from_manifest()

    # Apply industry filter
    exclude = config["data"].get("exclude_industries", [])
    if exclude:
        panel = filter_industries(panel, exclude, industry_col="二级行业")

    return panel


def register_file(file_path, years, manifest_path=None):
    """
    Register a data file in the manifest.

    Parameters
    ----------
    file_path : str
        Path to the data file (relative to project root).
    years : list of int
        Years covered by this file.
    manifest_path : str, optional
        Path to manifest.json.
    """
    if manifest_path is None:
        manifest_path = BASE / "data_input" / "manifest.json"

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"files": []}

    # Avoid duplicate entries
    existing_paths = {e["path"] for e in manifest["files"]}
    if file_path not in existing_paths:
        manifest["files"].append({
            "path": file_path,
            "years": years,
        })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# Train/predict split
# ---------------------------------------------------------------------------

def split_train_pred(panel, config):
    """
    Split panel into training (non-SOE benchmarks) and prediction (SOE target) sets.

    Returns
    -------
    train_dict : dict
        Keys are benchmark names, values are DataFrames (non-SOE firms).
    pred_df : pd.DataFrame
        SOE firms to predict on.
    """
    soe_types = config["data"].get("soe_types", ["地方国有企业", "中央国有企业"])
    bg = config["data"]["benchmark_groups"]

    panel = panel.copy()

    # Determine SOE status
    soe_col = config["data"]["soe_col"]
    if soe_col in panel.columns:
        panel["is_soe"] = panel[soe_col].isin(soe_types)
    elif "benchmark_group" in panel.columns:
        panel["is_soe"] = panel["benchmark_group"] == bg["soe_target"]
    else:
        raise ValueError("无法确定 SOE 状态：缺少 soe_label 列和 benchmark_group 列。")

    pred_df = panel[panel["is_soe"]].copy()

    # Training benchmarks
    h_ns = panel[panel["benchmark_group"] == bg["h_non_soe"]].copy()
    csi_ns = panel[panel["benchmark_group"] == bg["csi_non_soe"]].copy()

    train_dict = {
        bg["h_non_soe"]: h_ns,
        bg["csi_non_soe"]: csi_ns,
        bg["pooled"]: pd.concat([h_ns, csi_ns], ignore_index=True),
    }

    return train_dict, pred_df


# ---------------------------------------------------------------------------
# ROA / ROE supplementary data
# ---------------------------------------------------------------------------

def _load_wide_ratio_file(file_path):
    """
    Load a wide-format ROA or ROE CSV file with multi-line Wind headers.

    These files have 20+ header rows with verbose column names like
    "总资产报酬率ROA [报告期] 2025年报 [单位] %". This function extracts
    the year from each column header and returns a clean DataFrame with
    columns: 证券代码, 证券简称, {2015: val, 2016: val, ...}.

    Values are converted from percentage strings to decimal floats (÷100).
    """
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = BASE / file_path

    # Read raw to find the header row
    raw = pd.read_csv(file_path, encoding="utf-8", header=None, dtype=str)

    # Find the row that contains 证券代码 (the real header)
    header_row = None
    for i, row in raw.iterrows():
        if "证券代码" in row.values:
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"无法在 {file_path} 中找到表头行。")

    # Read again with correct header
    df = pd.read_csv(file_path, encoding="utf-8", header=header_row, dtype=str)

    # Drop non-data rows
    df = df[df["证券代码"].notna() & (df["证券代码"] != "")].copy()
    df = df[~df["证券代码"].str.contains("数据来源|Wind|单位", na=False)].copy()

    # Build year-column mapping
    year_map = {}
    for col in df.columns:
        import re
        match = re.search(r"(\d{4})", col)
        if match:
            year_map[col] = int(match.group(1))

    # Reshape to long format
    id_vars = ["证券代码", "证券简称"]
    value_vars = [c for c in df.columns if c in year_map]

    long = df.melt(
        id_vars=[c for c in id_vars if c in df.columns],
        value_vars=value_vars,
        var_name="year_col",
        value_name="value",
    )
    long["year"] = long["year_col"].map(year_map)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # Drop NaN values (empty cells in the raw data)
    long = long.dropna(subset=["value"])

    return long[["证券代码", "year", "value"]].copy()


def merge_roa_roe(panel, config):
    """
    Load ROA and ROE data from wide-format files, convert to long,
    and merge into the panel DataFrame.

    Merge key: firm_id = 证券代码, year = year.
    ROA/ROE values are divided by 100 (percentage → decimal).

    Adds columns 'ROA' and 'ROE' to the panel.
    """
    rr_cfg = config["data"].get("roa_roe", {})
    if not rr_cfg:
        return panel

    panel = panel.copy()

    # --- ROA ---
    roa_frames = []
    for key in ["csi_roa_path", "h_roa_path"]:
        if key in rr_cfg and rr_cfg[key]:
            fpath = BASE / rr_cfg[key] if not Path(rr_cfg[key]).is_absolute() else Path(rr_cfg[key])
            if fpath.exists():
                roa_long = _load_wide_ratio_file(fpath)
                roa_frames.append(roa_long)

    if roa_frames:
        roa_all = pd.concat(roa_frames, ignore_index=True)
        roa_all = roa_all.rename(columns={"value": "ROA_raw"})
        roa_all["ROA"] = roa_all["ROA_raw"] / 100.0  # % → decimal
        panel = panel.merge(
            roa_all[["证券代码", "year", "ROA"]],
            left_on=["firm_id", "year"],
            right_on=["证券代码", "year"],
            how="left",
        )
        panel = panel.drop(columns=["证券代码"], errors="ignore")
        # ROA only valid for target_year rows (same framework as EBI/A)
        if "_row_type" in panel.columns:
            panel.loc[panel["_row_type"] != "target_year", "ROA"] = np.nan
        n_roa = panel["ROA"].notna().sum()
        print(f"  已合并 ROA: {n_roa} 行有值 (仅 target_year)")

    # --- ROE ---
    roe_frames = []
    for key in ["csi_roe_path", "h_roe_path"]:
        if key in rr_cfg and rr_cfg[key]:
            fpath = BASE / rr_cfg[key] if not Path(rr_cfg[key]).is_absolute() else Path(rr_cfg[key])
            if fpath.exists():
                roe_long = _load_wide_ratio_file(fpath)
                roe_frames.append(roe_long)

    if roe_frames:
        roe_all = pd.concat(roe_frames, ignore_index=True)
        roe_all = roe_all.rename(columns={"value": "ROE_raw"})
        roe_all["ROE"] = roe_all["ROE_raw"] / 100.0  # % → decimal
        panel = panel.merge(
            roe_all[["证券代码", "year", "ROE"]],
            left_on=["firm_id", "year"],
            right_on=["证券代码", "year"],
            how="left",
        )
        panel = panel.drop(columns=["证券代码"], errors="ignore")
        if "_row_type" in panel.columns:
            panel.loc[panel["_row_type"] != "target_year", "ROE"] = np.nan
        n_roe = panel["ROE"].notna().sum()
        print(f"  已合并 ROE: {n_roe} 行有值 (仅 target_year)")

    return panel
