#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature engineering: lagged features, fixed effects, winsorization.

Pipeline:
  1. Build t-1 lagged features (groupby firm_id, shift(1))
  2. Split into train (non-SOE) and predict (SOE) sets
  3. Winsorize numeric features and target based on train distribution
  4. Encode fixed effects (year, industry, interactions, market dummy)
  5. Assemble final X, y matrices
"""

import numpy as np
import pandas as pd


def build_lagged_features(panel, numeric_cols, id_col="firm_id", year_col="year"):
    """
    Create t-1 lagged features for each firm.

    For each numeric column f, creates f_lag = f shifted by 1 year within each firm.
    Rows where lag is NaN (first observation of a firm) are retained;
    they will be imputed downstream.

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel, sorted by firm_id + year.
    numeric_cols : list of str
        Column names to lag.
    id_col : str
        Firm identifier column.
    year_col : str
        Year column.

    Returns
    -------
    pd.DataFrame with original columns plus '{col}_lag' columns.
    """
    df = panel.sort_values([id_col, year_col]).copy()
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_lag"] = df.groupby(id_col)[col].shift(1)
        else:
            df[f"{col}_lag"] = np.nan
    return df


def _get_dummy_columns(df, col, prefix=None, drop_first=True):
    """
    Return a DataFrame of dummy variables for a categorical column.

    drop_first=True avoids perfect multicollinearity (reference category).
    """
    if prefix is None:
        prefix = col
    dummies = pd.get_dummies(df[col].astype(str), prefix=prefix, drop_first=drop_first)
    return dummies.astype(float)


def encode_fixed_effects(df, fe_config, market_col="market"):
    """
    Encode fixed effects as dummy columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: year, industry_l1, industry_l2, market.
    fe_config : dict
        Fixed effects configuration with boolean flags:
        year, industry_l1, industry_l2, industry_l1_x_year,
        industry_l2_x_year, market_dummy.

    Returns
    -------
    fe_df : pd.DataFrame
        Dummy-encoded fixed effects (no intercept column).
    fe_names : list of str
        Column names of the FE dummies.
    """
    parts = []
    year_col = "year"
    l1_col = "industry_l1"
    l2_col = "industry_l2"

    # Year FE
    if fe_config.get("year", True) and year_col in df.columns:
        d = _get_dummy_columns(df, year_col, prefix="year")
        parts.append(d)

    # Industry L1 FE
    if fe_config.get("industry_l1", True) and l1_col in df.columns:
        d = _get_dummy_columns(df, l1_col, prefix="ind_l1")
        parts.append(d)

    # Industry L2 FE
    if fe_config.get("industry_l2", True) and l2_col in df.columns:
        d = _get_dummy_columns(df, l2_col, prefix="ind_l2")
        parts.append(d)

    # Industry L1 × Year FE
    if fe_config.get("industry_l1_x_year", True) and l1_col in df.columns and year_col in df.columns:
        df_tmp = df.copy()
        df_tmp["ind_l1_x_year"] = df_tmp[l1_col].astype(str) + "__" + df_tmp[year_col].astype(str)
        d = _get_dummy_columns(df_tmp, "ind_l1_x_year", prefix="l1xy")
        parts.append(d)

    # Industry L2 × Year FE
    if fe_config.get("industry_l2_x_year", False) and l2_col in df.columns and year_col in df.columns:
        df_tmp = df.copy()
        df_tmp["ind_l2_x_year"] = df_tmp[l2_col].astype(str) + "__" + df_tmp[year_col].astype(str)
        d = _get_dummy_columns(df_tmp, "ind_l2_x_year", prefix="l2xy")
        parts.append(d)

    # Market dummy (H=1, CSI=0)
    if fe_config.get("market_dummy", True) and market_col in df.columns:
        d = pd.DataFrame({"market_H": (df[market_col].astype(str).str.upper() == "H").astype(float)})
        parts.append(d)

    if parts:
        fe_df = pd.concat(parts, axis=1)
    else:
        fe_df = pd.DataFrame(index=df.index)

    return fe_df, list(fe_df.columns)


def winsorize_features(train_df, pred_df, numeric_cols, bounds=(0.05, 0.95)):
    """
    Clip numeric features in train and pred based on train quantiles.

    Parameters
    ----------
    train_df, pred_df : pd.DataFrame
    numeric_cols : list of str
        Column names to winsorize.
    bounds : tuple of (lower, upper)

    Returns
    -------
    train_df, pred_df (clipped copies), lower_bounds dict, upper_bounds dict
    """
    lower, upper = bounds
    train = train_df.copy()
    pred = pred_df.copy()
    lower_bounds = {}
    upper_bounds = {}

    for col in numeric_cols:
        if col not in train.columns:
            continue
        train.loc[:, col] = pd.to_numeric(train[col], errors="coerce")
        pred.loc[:, col] = pd.to_numeric(pred[col], errors="coerce")
        lo = train[col].quantile(lower)
        hi = train[col].quantile(upper)
        lower_bounds[col] = lo
        upper_bounds[col] = hi
        train.loc[:, col] = train[col].clip(lo, hi)
        pred.loc[:, col] = pred[col].clip(lo, hi)

    return train, pred, lower_bounds, upper_bounds


def build_feature_matrix(train_df, pred_df, config):
    """
    Build X_train, y_train, X_pred, y_pred_actual for model fitting.

    This is the main orchestrator for feature engineering.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (non-SOE firms in long format with lagged features).
    pred_df : pd.DataFrame
        Prediction data (SOE firms in long format with lagged features).
    config : dict
        Full configuration.

    Returns
    -------
    X_train : pd.DataFrame
    y_train : pd.Series
    X_pred : pd.DataFrame
    y_pred_actual : pd.Series (actual EBI_A values for comparison)
    info : dict with metadata (n_train, feature_names, winsor_bounds, etc.)
    """
    fe_cfg = config["features"]["fixed_effects"]
    num_features = config["features"]["numeric"]
    cat_features = config["features"].get("categorical", ["industry_l1", "industry_l2"])
    target_col = config["data"]["target_col"]
    winsor_cfg = config["winsorization"]["main"]
    market_col = config["data"]["market_col"]
    l1_col = config["data"]["industry_l1_col"]
    l2_col = config["data"]["industry_l2_col"]

    # Use lagged versions of numeric features
    num_cols_lag = [f"{f}_lag" for f in num_features]

    # Ensure required columns exist
    for col in num_cols_lag:
        if col not in train_df.columns:
            train_df[col] = np.nan
            pred_df[col] = np.nan

    # --- Keep only rows with valid target for training ---
    train = train_df[pd.notna(train_df[target_col])].reset_index(drop=True).copy()
    pred = pred_df.copy()

    # --- Winsorize numeric features ---
    train, pred, feat_lower, feat_upper = winsorize_features(
        train, pred, num_cols_lag, bounds=winsor_cfg
    )

    # --- Winsorize target ---
    y_lower = train[target_col].quantile(winsor_cfg[0])
    y_upper = train[target_col].quantile(winsor_cfg[1])
    y_train = train[target_col].clip(y_lower, y_upper).astype(float)
    y_pred_actual = pred[target_col].astype(float).values

    # --- Fixed effects encoding ---
    # Rename columns to match what encode_fixed_effects expects
    train_fe = train.rename(columns={l1_col: "industry_l1", l2_col: "industry_l2", market_col: "market"})
    pred_fe = pred.rename(columns={l1_col: "industry_l1", l2_col: "industry_l2", market_col: "market"})

    # Add year column if not present (legacy mode)
    if "year" not in train_fe.columns:
        train_fe["year"] = config["data"]["legacy"]["target_year"]
        pred_fe["year"] = config["data"]["legacy"]["target_year"]

    # Use raw (non-lagged) categorical columns for industry FEs
    # Map categorical column names from config to actual DataFrame columns
    cat_cols_in_df = []
    for c in cat_features:
        if c in train_fe.columns:
            cat_cols_in_df.append(c)
        elif c in train.columns:
            cat_cols_in_df.append(c)

    # Encode fixed effects
    train_fe_df, fe_names = encode_fixed_effects(train_fe, fe_cfg, market_col="market")
    pred_fe_df, _ = encode_fixed_effects(pred_fe, fe_cfg, market_col="market")

    # Align columns (pred may have different FEs than train)
    train_fe_df, pred_fe_df = train_fe_df.align(pred_fe_df, join="left", axis=1, fill_value=0)

    # --- Assemble X ---
    X_train_num = train[num_cols_lag].copy()
    X_pred_num = pred[num_cols_lag].copy()

    # Align numeric column names
    X_train_num.columns = [f"num__{c}" for c in X_train_num.columns]
    X_pred_num.columns = [f"num__{c}" for c in X_pred_num.columns]

    # Add categorical dummy features
    for c in cat_cols_in_df:
        train_dummies = pd.get_dummies(train[c].astype(str), prefix=f"cat__{c}", drop_first=True).astype(float)
        pred_dummies = pd.get_dummies(pred[c].astype(str), prefix=f"cat__{c}", drop_first=True).astype(float)
        train_dummies, pred_dummies = train_dummies.align(pred_dummies, join="left", axis=1, fill_value=0)
        X_train_num = pd.concat([X_train_num, train_dummies], axis=1)
        X_pred_num = pd.concat([X_pred_num, pred_dummies], axis=1)

    # Add FE columns
    train_fe_df.columns = [f"fe__{c}" for c in train_fe_df.columns]
    pred_fe_df.columns = [f"fe__{c}" for c in pred_fe_df.columns]

    X_train = pd.concat([X_train_num, train_fe_df], axis=1)
    X_pred = pd.concat([X_pred_num, pred_fe_df], axis=1)

    # Align train and pred columns
    X_train, X_pred = X_train.align(X_pred, join="left", axis=1, fill_value=0)

    info = {
        "n_train": len(train),
        "n_pred": len(pred),
        "n_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
        "fe_names": fe_names,
        "target_lower": y_lower,
        "target_upper": y_upper,
    }

    return X_train, y_train, X_pred, y_pred_actual, info


def build_feature_matrix_simple(train_df, pred_df, config):
    """
    Simplified feature matrix builder compatible with the original approach.

    Uses sklearn ColumnTransformer for a cleaner pipeline integration.
    Uses raw numeric features + categorical FEs encoded via OHE.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with columns for numeric features and categorical columns.
    pred_df : pd.DataFrame
        Prediction data (SOE firms).
    config : dict
        Full configuration.

    Returns
    -------
    X_train : pd.DataFrame
    y_train : pd.Series
    X_pred : pd.DataFrame
    info : dict
    """
    fe_cfg = config["features"]["fixed_effects"]
    num_features = config["features"]["numeric"]
    winsor_cfg = config["winsorization"]["main"]
    target_col = config["data"]["target_col"]
    market_col = config["data"]["market_col"]
    l1_col = config["data"]["industry_l1_col"]
    l2_col = config["data"]["industry_l2_col"]

    # Use lagged numeric features
    num_cols_lag = [f"{f}_lag" for f in num_features]
    for col in num_cols_lag:
        if col not in train_df.columns:
            train_df[col] = np.nan
            pred_df[col] = np.nan

    train = train_df[pd.notna(train_df[target_col])].reset_index(drop=True).copy()
    pred = pred_df.copy()

    # Winsorize numeric features
    train, pred, feat_lower, feat_upper = winsorize_features(
        train, pred, num_cols_lag, bounds=winsor_cfg
    )

    # Winsorize target
    y_lower = train[target_col].quantile(winsor_cfg[0])
    y_upper = train[target_col].quantile(winsor_cfg[1])
    y_train = train[target_col].clip(y_lower, y_upper).astype(float)

    # Prepare FE columns
    train_fe = train.rename(columns={l1_col: "industry_l1", l2_col: "industry_l2", market_col: "market"})
    pred_fe = pred.rename(columns={l1_col: "industry_l1", l2_col: "industry_l2", market_col: "market"})

    if "year" not in train_fe.columns:
        train_fe["year"] = config["data"]["legacy"]["target_year"]
        pred_fe["year"] = config["data"]["legacy"]["target_year"]

    # Encode FEs as dummies and attach to train/pred
    train_fe_df, fe_names = encode_fixed_effects(train_fe, fe_cfg, market_col="market")
    pred_fe_df, _ = encode_fixed_effects(pred_fe, fe_cfg, market_col="market")
    train_fe_df, pred_fe_df = train_fe_df.align(pred_fe_df, join="left", axis=1, fill_value=0)

    # Combine numeric + FE
    X_train = pd.concat([train[num_cols_lag].reset_index(drop=True), train_fe_df.reset_index(drop=True)], axis=1)
    X_pred = pd.concat([pred[num_cols_lag].reset_index(drop=True), pred_fe_df.reset_index(drop=True)], axis=1)
    X_train, X_pred = X_train.align(X_pred, join="left", axis=1, fill_value=0)

    info = {
        "n_train": len(train),
        "n_features": X_train.shape[1],
        "target_lower": y_lower,
        "target_upper": y_upper,
    }

    return X_train, y_train, X_pred, info


# ---------------------------------------------------------------------------
# Common support filtering
# ---------------------------------------------------------------------------

def compute_support_flags(train_df, pred_df, core_vars, bounds=(0.01, 0.99)):
    """
    Flag prediction observations that fall outside the benchmark's common support.

    For each core variable, computes [P_lower, P_upper] from train_df.
    A prediction row is "in support" only if ALL core variables fall inside
    their respective intervals.

    Parameters
    ----------
    train_df : pd.DataFrame
        Benchmark training data (non-SOE).
    pred_df : pd.DataFrame
        SOE prediction data.
    core_vars : list of str
        Column names to check (should exist in both train_df and pred_df).
    bounds : tuple of (lower, upper)
        Percentile bounds (e.g. (0.01, 0.99) or (0.05, 0.95)).

    Returns
    -------
    pred_df : pd.DataFrame
        Copy of pred_df with added columns:
          _support_flag: 1 = all core vars in bounds, 0 = at least one violation
          _support_violations: count of violated variables
          _support_detail: comma-separated list of violated variable names
    info : dict
        Per-variable violation counts and shares.
    """
    lower, upper = bounds
    pred = pred_df.copy()
    pred["_support_flag"] = 1
    pred["_support_violations"] = 0
    pred["_support_detail"] = ""

    violation_info = {}
    n_total = len(pred)

    for var in core_vars:
        if var not in train_df.columns or var not in pred.columns:
            violation_info[var] = {"n_violations": 0, "share": 0.0, "lower": None, "upper": None}
            continue

        train_vals = train_df[var].dropna().astype(float)
        if len(train_vals) == 0:
            continue

        lo = train_vals.quantile(lower)
        hi = train_vals.quantile(upper)
        pred_vals = pred[var].astype(float)
        outside = (pred_vals < lo) | (pred_vals > hi)
        n_out = int(outside.sum())

        violation_info[var] = {
            "n_violations": n_out,
            "share": n_out / n_total if n_total > 0 else 0.0,
            "lower": float(lo),
            "upper": float(hi),
        }

        if n_out > 0:
            pred.loc[outside, "_support_flag"] = 0
            pred.loc[outside, "_support_violations"] += 1
            # Append variable name to detail
            pred.loc[outside, "_support_detail"] = (
                pred.loc[outside, "_support_detail"].str.strip()
                .where(lambda s: s == "", lambda s: s + ", ")
                + var
            )

    info = {
        "n_total": n_total,
        "n_supported": int(pred["_support_flag"].sum()),
        "n_unsupported": int((pred["_support_flag"] == 0).sum()),
        "share_supported": float(pred["_support_flag"].mean()),
        "violation_info": violation_info,
        "bounds_used": bounds,
    }

    return pred, info
