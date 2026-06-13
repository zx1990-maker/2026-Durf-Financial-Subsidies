#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Propensity Score Matching (PSM) for benchmark sample selection.

PSM is used as a PRE-processing step: before training the Ridge prediction
model, we match non-SOE firms to SOE firms on propensity score, so that
the training sample is more comparable to the prediction target.

Workflow:
  1. Pool SOE + non-SOE, add binary treatment indicator (is_SOE)
  2. Estimate P(is_SOE | X) using GradientBoostingClassifier
  3. Nearest-neighbor matching on logit propensity score with caliper
  4. Balance check (Standardized Mean Difference)
  5. Return matched non-SOE subset for downstream model training
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def _impute_na(X):
    """Simple median imputation for propensity score estimation."""
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    return X


def estimate_propensity(X, y_treat, classifier="gbm", random_state=42):
    """
    Estimate propensity scores.

    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix (imputed).
    y_treat : pd.Series or np.array
        Binary treatment indicator (1 = SOE, 0 = non-SOE).
    classifier : str
        "gbm" for GradientBoosting or "logit" for LogisticRegression.
    random_state : int

    Returns
    -------
    ps : np.array of propensity scores P(SOE=1).
    ps_logit : np.array of logit(ps).
    model : fitted classifier.
    """
    X_imputed = _impute_na(X)

    if classifier == "gbm":
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, random_state=random_state,
        )
    else:
        model = LogisticRegression(
            solver="saga", penalty="l1", C=1.0,
            max_iter=2000, random_state=random_state,
        )

    model.fit(X_imputed, y_treat)
    ps = model.predict_proba(X_imputed)[:, 1]

    # Clip to avoid logit extremes
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    ps_logit = np.log(ps / (1 - ps))

    return ps, ps_logit, model


def match_nearest_neighbor(ps_logit_treat, ps_logit_control,
                           control_indices, caliper=None, replace=True):
    """
    Nearest-neighbor matching on logit propensity score.

    Parameters
    ----------
    ps_logit_treat : np.array
        Logit PS for treatment (SOE) observations.
    ps_logit_control : np.array
        Logit PS for control (non-SOE) observations.
    control_indices : np.array
        Integer indices of control observations (for mapping back).
    caliper : float or None
        Max allowed distance in logit PS. If None, no caliper.
    replace : bool
        If True, allow one control to match multiple treatments (1:n).
        If False, each control used at most once (1:1).

    Returns
    -------
    matched_idx : np.array of control indices that were matched.
    matched_weights : np.array of match counts per control.
    n_unmatched_treat : int
    """
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(ps_logit_control.reshape(-1, 1))
    distances, indices = nn.kneighbors(ps_logit_treat.reshape(-1, 1))

    match_counts = {}
    n_unmatched = 0

    for i in range(len(ps_logit_treat)):
        d = distances[i][0]
        c_idx = control_indices[indices[i][0]]
        if caliper is not None and d > caliper:
            n_unmatched += 1
            continue
        match_counts[c_idx] = match_counts.get(c_idx, 0) + 1

    matched_idx = np.array(sorted(match_counts.keys()))
    weights = np.array([match_counts[i] for i in matched_idx])

    return matched_idx, weights, n_unmatched


def compute_smd(df, vars, treat_col):
    """
    Compute Standardized Mean Difference for balance diagnostics.

    SMD = (mean_treat - mean_control) / sqrt((var_treat + var_control) / 2)

    Returns dict: {var: smd_value}.
    """
    treat = df[df[treat_col] == 1]
    control = df[df[treat_col] == 0]
    smd = {}
    for v in vars:
        if v not in df.columns:
            continue
        mt = treat[v].mean()
        mc = control[v].mean()
        vt = treat[v].var()
        vc = control[v].var()
        denom = np.sqrt((vt + vc) / 2)
        if denom == 0:
            smd[v] = 0.0
        else:
            smd[v] = float((mt - mc) / denom)
    return smd


def run_psm_matching(train_df, pred_df, covariate_cols,
                     caliper_sd=0.2, classifier="gbm", random_state=42):
    """
    Full PSM pipeline for one benchmark.

    1. Pool SOE (pred_df) + non-SOE (train_df)
    2. Estimate propensity score
    3. Match each SOE to nearest non-SOE on logit PS
    4. Balance check
    5. Return matched non-SOE subset + diagnostics

    Parameters
    ----------
    train_df : pd.DataFrame
        Non-SOE training data (benchmark).
    pred_df : pd.DataFrame
        SOE prediction data (target).
    covariate_cols : list of str
        Covariate column names for propensity estimation.
    caliper_sd : float
        Caliper = caliper_sd × SD(logit PS) of the control group.
    classifier : str
        "gbm" or "logit".
    random_state : int

    Returns
    -------
    train_matched : pd.DataFrame
        Subset of train_df with matched non-SOE firms only.
    psm_info : dict
        Diagnostics: n_before, n_after, n_unmatched, smd_before, smd_after, etc.
    """
    # --- 1. Pool data ---
    train = train_df.copy()
    pred = pred_df.copy()
    train["_is_soe"] = 0
    pred["_is_soe"] = 1
    pooled = pd.concat([train, pred], ignore_index=True)

    # Available covariates
    cov_cols_present = [c for c in covariate_cols if c in pooled.columns]
    if len(cov_cols_present) == 0:
        return train, {"error": "No covariate columns found", "n_before": len(train)}

    X = pooled[cov_cols_present]
    y_treat = pooled["_is_soe"].astype(int)

    # --- 2. Estimate propensity ---
    ps, ps_logit, model = estimate_propensity(X, y_treat, classifier, random_state)

    pooled["_ps"] = ps
    pooled["_ps_logit"] = ps_logit

    treat_mask = pooled["_is_soe"] == 1
    control_mask = pooled["_is_soe"] == 0

    ps_logit_treat = ps_logit[treat_mask.values]
    ps_logit_control = ps_logit[control_mask.values]
    control_indices = np.where(control_mask.values)[0]

    # --- 3. Matching ---
    caliper = caliper_sd * np.std(ps_logit_control) if caliper_sd else None

    matched_idx, weights, n_unmatched = match_nearest_neighbor(
        ps_logit_treat, ps_logit_control, control_indices,
        caliper=caliper, replace=True,
    )

    n_treat = len(ps_logit_treat)
    n_control = len(ps_logit_control)
    n_matched = len(matched_idx)
    total_weight = int(weights.sum()) if len(weights) > 0 else 0

    # --- 4. Balance check ---
    smd_before = compute_smd(pooled, cov_cols_present, "_is_soe")

    pooled_matched = pd.concat([
        pooled.iloc[matched_idx],
        pooled[treat_mask],
    ], ignore_index=True)
    smd_after = compute_smd(pooled_matched, cov_cols_present, "_is_soe")

    # --- 5. Return matched non-SOE subset (with replacement weights) ---
    # Repeat each matched control by its weight
    train_rows = []
    for idx, w in zip(matched_idx, weights):
        if idx < len(train):
            train_rows.extend([train.iloc[idx]] * int(w))
    train_matched = pd.DataFrame(train_rows).reset_index(drop=True) if train_rows else train.iloc[:0].copy()

    smd_improved = sum(
        1 for v in cov_cols_present
        if abs(smd_after.get(v, 0)) < abs(smd_before.get(v, 0))
    )

    psm_info = {
        "n_treat": n_treat,
        "n_control_before": n_control,
        "n_control_matched": n_matched,
        "n_control_weighted": total_weight,
        "n_unmatched_treat": n_unmatched,
        "match_rate": n_matched / n_control if n_control > 0 else 0,
        "caliper": caliper,
        "smd_before": smd_before,
        "smd_after": smd_after,
        "smd_before_max": max(abs(v) for v in smd_before.values()) if smd_before else None,
        "smd_after_max": max(abs(v) for v in smd_after.values()) if smd_after else None,
        "smd_vars_improved": smd_improved,
        "smd_vars_total": len(cov_cols_present),
        "ps_distribution": {
            "treat_mean": float(ps[treat_mask].mean()),
            "control_mean": float(ps[control_mask].mean()),
            "matched_control_mean": float(ps[matched_idx].mean()),
        },
    }

    return train_matched, psm_info


# ---------------------------------------------------------------------------
# CEM (Coarsened Exact Matching)
# ---------------------------------------------------------------------------

def run_cem_matching(train_df, pred_df, exact_cols, coarsen_cols, coarsen_bins=5):
    """
    Coarsened Exact Matching.

    Exact match on: year, industry_l2, market (exact_cols).
    Coarsen: FirmSize, Leverage, etc. into k bins.
    Keep only cells with >= 1 SOE and >= 1 non-SOE.

    Returns matched non-SOE subset + diagnostics.
    """
    train = train_df.copy()
    pred = pred_df.copy()
    train["_is_soe"] = 0
    pred["_is_soe"] = 1
    pooled = pd.concat([train, pred], ignore_index=True)

    # Build stratum key
    pooled["_stratum"] = ""
    for col in exact_cols:
        if col in pooled.columns:
            pooled["_stratum"] += pooled[col].astype(str) + "|"

    for col in coarsen_cols:
        if col in pooled.columns:
            vals = pooled[col].dropna()
            if len(vals) < 10:
                continue
            try:
                cuts = pd.qcut(vals, q=min(coarsen_bins, len(vals.unique())),
                               duplicates="drop", labels=False)
                pooled.loc[vals.index, "_stratum"] += f"{col}_{cuts.astype(str)}|"
            except Exception:
                pass

    # Count SOE and non-SOE per stratum
    strata_counts = pooled.groupby("_stratum")["_is_soe"].agg(["sum", "count", "mean"])
    valid_strata = strata_counts[
        (strata_counts["sum"] >= 1) &  # at least 1 SOE
        (strata_counts["count"] > strata_counts["sum"])  # at least 1 non-SOE
    ].index

    # Keep only valid strata
    pooled_valid = pooled[pooled["_stratum"].isin(valid_strata)]
    train_cem = pooled_valid[pooled_valid["_is_soe"] == 0].copy()
    pred_cem = pooled_valid[pooled_valid["_is_soe"] == 1].copy()

    n_before = len(train)
    n_after = len(train_cem)
    n_soe_before = len(pred)
    n_soe_after = len(pred_cem)
    n_strata = len(valid_strata)

    # SMD before and after
    cov_cols = [c for c in coarsen_cols if c in pooled.columns]
    smd_before = compute_smd(pooled, cov_cols, "_is_soe")
    smd_after = compute_smd(pooled_valid, cov_cols, "_is_soe")

    cem_info = {
        "n_before": n_before,
        "n_after": n_after,
        "n_soe_before": n_soe_before,
        "n_soe_after": n_soe_after,
        "retention_rate": n_after / n_before if n_before > 0 else 0,
        "soe_retention_rate": n_soe_after / n_soe_before if n_soe_before > 0 else 0,
        "n_strata": n_strata,
        "smd_before": smd_before,
        "smd_after": smd_after,
        "smd_before_max": max(abs(v) for v in smd_before.values()) if smd_before else None,
        "smd_after_max": max(abs(v) for v in smd_after.values()) if smd_after else None,
    }

    return train_cem, cem_info


# ---------------------------------------------------------------------------
# Entropy Balancing
# ---------------------------------------------------------------------------

def run_entropy_balancing(train_df, pred_df, balance_cols, max_iter=200):
    """
    Entropy balancing: reweight non-SOE firms so that weighted means
    of balance_cols exactly match SOE means.

    Uses iterative raking/logistic weighting.
    Returns train_df with added _entropy_weight column.
    """
    train = train_df.copy()
    pred = pred_df.copy()

    # Available balance columns
    cols = [c for c in balance_cols if c in train.columns and c in pred.columns]
    if len(cols) == 0:
        train["_entropy_weight"] = 1.0
        return train, {"error": "No balance columns available"}

    # Compute SOE means
    soe_means = {}
    for c in cols:
        soe_means[c] = pred[c].mean()

    # Initialize weights
    w = np.ones(len(train))
    X = train[cols].fillna(train[cols].median()).values

    # Simple iterative raking: adjust weights to match each moment in turn
    lr = 0.05
    for iteration in range(max_iter):
        max_adj = 0.0
        for j, c in enumerate(cols):
            current_mean = np.average(X[:, j], weights=w)
            target = soe_means[c]
            if abs(target) < 1e-10 and abs(current_mean) < 1e-10:
                continue
            if abs(target) > 1e-10:
                adj = np.exp(lr * (target - current_mean) / abs(target))
            else:
                adj = np.exp(lr * (target - current_mean))
            adj = np.clip(adj, 0.1, 10.0)
            w = w * adj
            w = w / w.sum() * len(train)
            max_adj = max(max_adj, abs(adj - 1.0).max())
        if max_adj < 1e-4:
            break

    train["_entropy_weight"] = w

    # Compute effective sample size
    ess = w.sum() ** 2 / (w ** 2).sum()

    # Balance check
    balance_after = {}
    for c in cols:
        balance_after[c] = np.average(X[:, j] if c == cols[-1] else
                                      train[c].fillna(train[c].median()).values,
                                      weights=w)

    eb_info = {
        "n_train": len(train),
        "ess": float(ess),
        "ess_ratio": float(ess / len(train)),
        "balance_after": balance_after,
        "soe_targets": {c: float(soe_means[c]) for c in cols[:5]},
    }

    return train, eb_info


# ---------------------------------------------------------------------------
# Overlap Weighting
# ---------------------------------------------------------------------------

def run_overlap_weighting(train_df, pred_df, covariate_cols,
                          classifier="gbm", random_state=42):
    """
    Overlap weighting (Li, Morgan, Zaslavsky 2018):
      w_i = 1 - e(X_i)  for SOE
      w_i = e(X_i)      for non-SOE

    Automatically downweights extreme propensity scores.
    Returns train_df with _overlap_weight column.
    """
    train = train_df.copy()
    pred = pred_df.copy()
    train["_is_soe"] = 0
    pred["_is_soe"] = 1
    pooled = pd.concat([train, pred], ignore_index=True)

    cov_cols_present = [c for c in covariate_cols if c in pooled.columns]
    if len(cov_cols_present) == 0:
        train["_overlap_weight"] = 1.0
        return train, {"error": "No covariate columns"}

    X = pooled[cov_cols_present].fillna(pooled[cov_cols_present].median())
    y = pooled["_is_soe"].astype(int)

    _, ps, _ = estimate_propensity(X, y, classifier, random_state)
    pooled["_ps"] = np.clip(ps, 1e-6, 1 - 1e-6)

    # Overlap weights
    pooled["_ow"] = np.where(
        pooled["_is_soe"] == 1,
        1.0 - pooled["_ps"],   # SOE: weight = 1 - e(x)
        pooled["_ps"],          # non-SOE: weight = e(x)
    )

    ow_train = pooled[pooled["_is_soe"] == 0].copy()
    ow_train["_overlap_weight"] = ow_train["_ow"]

    ess = ow_train["_ow"].sum() ** 2 / (ow_train["_ow"] ** 2).sum()

    ow_info = {
        "n_train": len(train),
        "ess": float(ess),
        "ess_ratio": float(ess / len(train)),
        "ps_distribution": {
            "soe_mean": float(pooled[pooled["_is_soe"] == 1]["_ps"].mean()),
            "nsoe_mean": float(pooled[pooled["_is_soe"] == 0]["_ps"].mean()),
        },
    }

    return ow_train, ow_info
