#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model: pipeline construction, cross-validation, fit/predict.

Supports Ridge/Lasso/ElasticNet with configurable parameters.
CV strategies: group (firm-level GroupKFold) or random (row-level KFold).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .features import build_feature_matrix_simple, winsorize_features


def _get_ohe():
    """Get OneHotEncoder with compatible parameters across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _get_model(config):
    """Build a model instance from configuration."""
    mtype = config["model"]["type"]
    params = config["model"]["params"].copy()
    alpha = params.pop("alpha", 10.0)
    random_state = params.pop("random_state", 42)

    if mtype == "ridge":
        return Ridge(alpha=alpha, random_state=random_state, **params)
    elif mtype == "lasso":
        return Lasso(alpha=alpha, random_state=random_state, **params)
    elif mtype == "elasticnet":
        return ElasticNet(alpha=alpha, random_state=random_state, **params)
    else:
        raise ValueError(f"不支持的模型类型: {mtype}")


def build_pipeline(numeric_cols, cat_cols, config):
    """
    Build an sklearn Pipeline: preprocessing (impute + scale + OHE) + model.

    Parameters
    ----------
    numeric_cols : list of str
        Numeric feature column names.
    cat_cols : list of str
        Categorical feature column names (will be one-hot encoded).
    config : dict
        Full configuration.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", _get_ohe()),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("prep", preprocessor),
        ("model", _get_model(config)),
    ])


def cv_metrics(train_df, cat_cols, config, numeric_cols=None, winsor=None,
               n_splits=None, groups_col=None):
    """
    Run cross-validation and return performance metrics.

    For panel data, uses GroupKFold on firm_id to prevent leakage.
    For legacy single-period data, uses standard KFold.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with numeric and categorical columns.
    cat_cols : list of str
        Categorical column names.
    config : dict
        Full configuration.
    numeric_cols : list of str, optional
        Numeric feature columns. If None, uses config.
    winsor : tuple, optional
        (lower, upper) winsor bounds. If None, uses config.
    n_splits : int, optional
        Number of CV folds. If None, uses config.
    groups_col : str, optional
        Column name for group-based CV (firm_id). Only used if
        cv.strategy == "group".

    Returns
    -------
    dict of CV metrics.
    """
    target_col = config["data"]["target_col"]
    cv_cfg = config["cv"]

    if numeric_cols is None:
        numeric_cols = [f"{f}_lag" for f in config["features"]["numeric"]]
    if winsor is None:
        winsor = config["winsorization"]["main"]
    if n_splits is None:
        n_splits = cv_cfg["n_splits"]

    train = train_df[pd.notna(train_df[target_col])].reset_index(drop=True).copy()

    n_splits = min(n_splits, len(train))
    strategy = cv_cfg.get("strategy", "random")

    if strategy == "group" and groups_col and groups_col in train.columns:
        kf = GroupKFold(n_splits=n_splits)
        split_on = train[groups_col]
    else:
        kf = KFold(n_splits=n_splits, shuffle=cv_cfg.get("shuffle", True),
                   random_state=cv_cfg.get("random_state", 42))
        split_on = train

    preds = np.full(len(train), np.nan)
    y_true = train[target_col].astype(float).values

    for tr_idx, te_idx in kf.split(split_on, groups=split_on if strategy == "group" else None):
        tr = train.iloc[tr_idx].copy()
        te = train.iloc[te_idx].copy()

        # Winsorize features
        tr, te, _, _ = winsorize_features(tr, te, numeric_cols, bounds=winsor)

        # Winsorize target
        y_lower = tr[target_col].quantile(winsor[0])
        y_upper = tr[target_col].quantile(winsor[1])
        y_tr = tr[target_col].clip(y_lower, y_upper).astype(float)

        pipe = build_pipeline(numeric_cols, cat_cols, config)
        pipe.fit(tr[numeric_cols + cat_cols], y_tr)
        preds[te_idx] = pipe.predict(te[numeric_cols + cat_cols])

    y_w = pd.Series(y_true).clip(
        train[target_col].quantile(winsor[0]),
        train[target_col].quantile(winsor[1]),
    ).values

    return {
        "n_train": len(train),
        "cv_rmse_raw_y": float(np.sqrt(mean_squared_error(y_true, preds))),
        "cv_mae_raw_y": float(mean_absolute_error(y_true, preds)),
        "cv_r2_raw_y": float(r2_score(y_true, preds)),
        "cv_rmse_winsor_y": float(np.sqrt(mean_squared_error(y_w, preds))),
        "cv_mae_winsor_y": float(mean_absolute_error(y_w, preds)),
        "cv_r2_winsor_y": float(r2_score(y_w, preds)),
    }


def fit_predict(train_df, pred_df, cat_cols, config, numeric_cols=None, winsor=None):
    """
    Fit model on training data and predict on prediction data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    pred_df : pd.DataFrame
        Prediction data (SOE firms).
    cat_cols : list of str
        Categorical columns for the pipeline.
    config : dict
        Full configuration.
    numeric_cols : list of str, optional
        Numeric feature columns.
    winsor : tuple, optional
        (lower, upper) winsor bounds.

    Returns
    -------
    pipe : fitted sklearn Pipeline
    predictions : np.ndarray
    info : dict with n_train, target_lower, target_upper
    """
    target_col = config["data"]["target_col"]

    if numeric_cols is None:
        numeric_cols = [f"{f}_lag" for f in config["features"]["numeric"]]
    if winsor is None:
        winsor = config["winsorization"]["main"]

    train = train_df[pd.notna(train_df[target_col])].reset_index(drop=True).copy()
    pred = pred_df.copy()

    # Ensure numeric columns exist in both train and pred
    for col in numeric_cols:
        if col not in train.columns:
            train[col] = np.nan
        if col not in pred.columns:
            pred[col] = np.nan

    train, pred, _, _ = winsorize_features(train, pred, numeric_cols, bounds=winsor)

    y_lower = train[target_col].quantile(winsor[0])
    y_upper = train[target_col].quantile(winsor[1])
    y = train[target_col].clip(y_lower, y_upper).astype(float)

    # Ensure cat columns exist
    for col in cat_cols:
        if col not in train.columns:
            train[col] = "missing"
        if col not in pred.columns:
            pred[col] = "missing"

    pipe = build_pipeline(numeric_cols, cat_cols, config)
    pipe.fit(train[numeric_cols + cat_cols], y)
    predictions = pipe.predict(pred[numeric_cols + cat_cols])

    info = {
        "n_train": len(train),
        "target_lower": y_lower,
        "target_upper": y_upper,
    }
    return pipe, predictions, info
