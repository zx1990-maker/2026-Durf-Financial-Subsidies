#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions: numeric conversion, weighted averages, gap calculations.
"""

import numpy as np
import pandas as pd


def to_numeric_series(s):
    """Convert a series to numeric, handling commas, percent signs, and placeholders."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan, "--": np.nan})
        .astype(float)
    )


def weighted_avg(g, val_col, weight_col="assets"):
    """Compute weighted average of val_col using weight_col."""
    w = g[weight_col].astype(float)
    v = g[val_col].astype(float)
    return np.nan if w.sum() == 0 else float((v * w).sum() / w.sum())


def compute_gap_columns(df, pred_col, actual_col, assets_col):
    """
    Compute counterfactual gap columns.

    return_gap = predicted - actual
    implied_gap_amount = return_gap × assets
    positive_gap_amount = max(return_gap, 0) × assets
    """
    df = df.copy()
    df["return_gap_pred_minus_actual"] = df[pred_col] - df[actual_col]
    df["implied_gap_amount"] = df["return_gap_pred_minus_actual"] * df[assets_col]
    df["positive_gap_amount"] = np.maximum(df["return_gap_pred_minus_actual"], 0) * df[assets_col]
    return df
