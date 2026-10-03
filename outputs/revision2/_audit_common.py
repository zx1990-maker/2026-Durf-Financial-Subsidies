#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared audit module for outputs/revision2/.

Replicates the PRODUCTION data pipeline EXACTLY (same screening, same index
reconstruction, same model hyperparameters, same winsorization/imputation/
scaling/one-hot scheme) so that every robustness result in revision2/ is
directly comparable to the paper's numbers in final_results/.

This module does NOT modify raw data and does NOT overwrite existing results —
it only reads 0803/ and input_index_weight/ and writes to outputs/revision2/.
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "revision2"
INDEX_DIR = PROJECT_DIR / "input_index_weight"

sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import (
    load_all_a_shares, clean_and_filter, NUMERIC_FEATURES,
    SOE_TYPES, NON_SOE_TYPE, EXCLUDE_INDUSTRIES_L2, RANDOM_STATE, RIDGE_ALPHA,
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold, cross_val_predict

SOE = {'中央国有企业', '地方国有企业'}
RANDOM_STATE = 42
RIDGE_ALPHA = 10.0


# ============================================================================
# Data loading (EXACT production screen)
# ============================================================================

def load_screened_panel():
    """Load A-share panel + apply the production screening:
    (1) drop ST firms (firm-level), (2) drop FinancialLeverage > 1 (firm-year),
    (3) drop |EBI/A| > 0.5 (firm-year). Returns panel with is_soe/is_private."""
    all_a = load_all_a_shares()
    all_a = clean_and_filter(all_a, 'A')
    st_firms = set(all_a[all_a['firm_name'].str.contains('ST', na=False)]['firm_id'].unique())
    all_a = all_a[~all_a['firm_id'].isin(st_firms)]
    all_a = all_a[~(all_a['FinancialLeverage'] > 1.0)]
    all_a = all_a[all_a['EBI_A'].abs() <= 0.5]
    all_a['is_soe'] = all_a['ownership'].isin(SOE)
    all_a['is_private'] = all_a['ownership'] == '民营企业'
    return all_a


def load_yearly_codes():
    """Backward-reconstruct index constituent code sets per year (2019-2025),
    matching the production scripts exactly."""
    idx_files = {
        'CSI300': ('沪深300-成分及权重-20260811.xlsx', '沪深300-成分进出记录-20260811.xlsx'),
        'CSI500': ('中证500-成分及权重-20260811.xlsx', '中证500-成分进出记录-20260811.xlsx'),
        'CSI1000': ('中证1000-成分及权重-20260811.xlsx', '中证1000-成分进出记录-20260811.xlsx'),
    }
    codes = {}
    all_records = {}
    for idx, (fw, fe) in idx_files.items():
        wb = openpyxl.load_workbook(INDEX_DIR / fw, read_only=True, data_only=True)
        ws = wb.active
        codes[idx] = {str(r[0]).strip() for r in ws.iter_rows(min_row=2, values_only=True)
                      if r[0] and str(r[2]).strip() != '金融'}
        wb.close()
        wb2 = openpyxl.load_workbook(INDEX_DIR / fe, read_only=True, data_only=True)
        ws2 = wb2.active
        all_records[idx] = sorted(
            [{'date': str(r[0])[:10], 'code': str(r[1]).strip(), 'action': str(r[7]).strip()}
             for r in ws2.iter_rows(min_row=2, values_only=True) if r[0] and r[1] and r[7]],
            key=lambda x: x['date'])
        wb2.close()
    yearly_codes = {}
    for idx in ['CSI300', 'CSI500', 'CSI1000']:
        cur = codes[idx]
        rev = sorted(all_records[idx], key=lambda r: r['date'], reverse=True)
        yearly_codes[idx] = {}
        for y in range(2019, 2026):
            td = f'{y}-07-01'
            cs = set(cur)
            for rec in rev:
                if rec['date'] <= td:
                    break
                if rec['action'] == '纳入':
                    cs.discard(rec['code'])
                elif rec['action'] == '剔除':
                    cs.add(rec['code'])
            yearly_codes[idx][y] = cs
    return yearly_codes


def feats_in(all_a):
    return [f for f in NUMERIC_FEATURES if f in all_a.columns]


# ============================================================================
# Models + inference
# ============================================================================

def build(mt):
    if mt == 'ridge':
        return Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    elif mt == 'random_forest':
        return RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                     max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)
    else:  # gbm
        return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         min_samples_leaf=10, random_state=RANDOM_STATE)


def cluster_se(gaps, firm_ids):
    """Firm-level clustered standard error of the mean gap."""
    gaps = np.asarray(gaps)
    firm_ids = np.asarray(firm_ids)
    u = np.unique(firm_ids)
    if len(u) <= 1:
        return np.nan, len(u)
    fm = np.array([gaps[firm_ids == f].mean() for f in u])
    se = np.std(fm, ddof=1) / np.sqrt(len(u))
    return se, len(u)


def cluster_t_test(gaps_a, firm_ids_a, gaps_b, firm_ids_b):
    """Two-sample difference test with firm-level clustering (Welch-style).
    Returns (diff, se, ci_lo, ci_hi, t, p)."""
    gaps_a = np.asarray(gaps_a); firm_ids_a = np.asarray(firm_ids_a)
    gaps_b = np.asarray(gaps_b); firm_ids_b = np.asarray(firm_ids_b)
    def firm_means(g, f):
        u = np.unique(f)
        return np.array([g[f == x].mean() for x in u])
    ma = firm_means(gaps_a, firm_ids_a)
    mb = firm_means(gaps_b, firm_ids_b)
    diff = ma.mean() - mb.mean()
    se_a = ma.std(ddof=1) / np.sqrt(len(ma)) if len(ma) > 1 else np.nan
    se_b = mb.std(ddof=1) / np.sqrt(len(mb)) if len(mb) > 1 else np.nan
    se = np.sqrt(se_a ** 2 + se_b ** 2)
    t = diff / se if se > 0 else np.nan
    df = min(len(ma), len(mb)) - 1
    from scipy import stats as scipy_stats
    p = 2 * (1 - scipy_stats.t.cdf(abs(t), df)) if not np.isnan(t) and df > 0 else np.nan
    return diff, se, diff - 1.96 * se, diff + 1.96 * se, t, p


# ============================================================================
# Training-transform / predict helpers (production fit-and-predict flow)
# ============================================================================

def prepare_train(train_df, feats):
    """Winsorize (P5/P95), median-impute, standardize, one-hot industry.
    Returns a dict `prep` holding everything needed for predict()."""
    tv = train_df.dropna(subset=feats + ['EBI_A']).copy()
    qlo = {}; qhi = {}
    for c in feats:
        lo, hi = tv[c].quantile(0.05), tv[c].quantile(0.95)
        tv[c + '_w'] = tv[c].clip(lo, hi)
        qlo[c] = lo; qhi[c] = hi
    yl, yh = tv['EBI_A'].quantile(0.05), tv['EBI_A'].quantile(0.95)
    tv['EBI_A_w'] = tv['EBI_A'].clip(yl, yh)
    ncw = [c + '_w' for c in feats]
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    Xt = scl.fit_transform(imp.fit_transform(tv[ncw].values))
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    Xt = np.hstack([Xt, ohe.fit_transform(tv[['industry_l2']].fillna('Unknown'))])
    yt = tv['EBI_A_w'].values
    prep = dict(tv=tv, Xt=Xt, yt=yt, imp=imp, scl=scl, ohe=ohe, ncw=ncw,
                feats=feats, qlo=qlo, qhi=qhi)
    return prep


def predict_gap(prep, model, pdf):
    """Predict on pdf using a fitted prep; returns (preds, gaps, pd2, actuals)."""
    feats = prep['feats']; ncw = prep['ncw']
    pv = pdf.copy()
    for c in feats:
        pv[c + '_w'] = pv[c].clip(prep['qlo'][c], prep['qhi'][c])
    pd2 = pv.dropna(subset=ncw).copy()
    if len(pd2) == 0:
        return None, None, None, None
    Xp = np.hstack([prep['scl'].transform(prep['imp'].transform(pd2[ncw].values)),
                    prep['ohe'].transform(pd2[['industry_l2']].fillna('Unknown'))])
    preds = model.predict(Xp)
    actuals = pd2['EBI_A'].values
    gaps = preds - actuals
    return preds, gaps, pd2, actuals


# ============================================================================
# Common-support (5-NN vs P90 threshold on private internal NN distance)
# ============================================================================

def common_support_mask(private_df, soe_df, feats, pct=0.90):
    """Return a boolean mask over soe_df (aligned to soe_df's full length) marking
    'good support' firms, using the production 5-NN / P90 rule."""
    priv = private_df.dropna(subset=feats)
    mask = np.zeros(len(soe_df), dtype=bool)
    if len(priv) < 10 or len(soe_df) == 0:
        return mask
    soe_valid = soe_df[feats].notna().all(axis=1)
    soe = soe_df[soe_valid]
    if len(soe) == 0:
        return mask
    scl_nn = StandardScaler()
    X_priv = scl_nn.fit_transform(priv[feats].values)
    nn_priv = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_priv)
    priv_dist = nn_priv.kneighbors(X_priv)[0][:, 1]
    p90 = np.quantile(priv_dist, pct)
    X_soe = scl_nn.transform(soe[feats].values)
    nn_soe = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(X_priv)
    soe_dist = nn_soe.kneighbors(X_soe)[0][:, 0]
    good = soe_dist <= p90
    valid_pos = np.where(soe_valid.values)[0]
    mask[valid_pos] = good
    return mask
