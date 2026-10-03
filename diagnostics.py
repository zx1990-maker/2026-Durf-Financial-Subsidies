#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全A股反事实预测 — 四项诊断分析
=================================
一、系统 Bias 检验（OOF calibration + SOE-like 子样本）
二、共同支持检验（SMD + propensity score + 最近邻距离 + PCA）
三、Placebo Gap 检验（匹配式 placebo + empirical p-value）
四、Bootstrap 置信区间（cluster bootstrap + BCa）

所有模型选择/调参仅在民企样本内部完成。
不允许使用 SOE 实际 EBI/A 选模型、挑特征或调参。
"""

import zipfile, re, os, sys, warnings, json, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# 0. Configuration — 请在此处确认/修改列名
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent / "0803"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAP = 500   # 建议 1000-2000，当前设为 500 以控制运行时间
N_PLACEBO = 500     # 建议 1000+，当前设为 500

# Column names — 如需修改请在此处集中替换
COL_FIRM_ID = 'firm_id'
COL_YEAR = 'target_year'
COL_OWNERSHIP = 'ownership'
COL_INDUSTRY = 'industry_l2'
COL_TARGET = 'EBI_A'
COL_TARGET_W = 'EBI_A_winsor'
COL_ASSETS = '资产总计'
COL_ASSETS_TARGET = '资产总计_target'
COL_REVENUE = '营业收入'
COL_SOE_FLAG = 'is_soe'

SOE_TYPES = {'中央国有企业', '地方国有企业'}
NON_SOE_TYPE = '民营企业'
EXCLUDE_INDUSTRIES = {'银行', '保险', '证券', '多元金融', '非银金融'}

NUMERIC_FEATURES = [
    'FirmSize', 'AssetTurnover', 'FinancialLeverage', 'FixedAssetsRatio',
    'CurrentRatio', 'CapexRatio', 'WorkingCapitalRatio', 'MarketShare', 'HHI',
]

# Index group labels
CSI500_LABEL = 'CSI500_SOE'
CSI1000_LABEL = 'CSI1000_SOE'

# ============================================================================
# 1. Data Loading (reuses run_counterfactual parsing infrastructure)
# ============================================================================

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict, StratifiedGroupKFold
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              roc_auc_score, average_precision_score, roc_curve)
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats as scipy_stats
from scipy.spatial.distance import mahalanobis

# Import parsing functions from run_counterfactual
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_counterfactual import (
    parse_file_to_records, records_to_dataframe, load_all_a_shares,
    load_csi500, load_csi1000, clean_and_filter, backfill_ownership,
    NUMERIC_FEATURES as _NF, SOE_TYPES as _ST, NON_SOE_TYPE as _NST,
    EXCLUDE_INDUSTRIES_L2 as _EI, WINSOR_BOUNDS, RIDGE_ALPHA,
)

WINSOR_BOUNDS = (0.05, 0.95)
RIDGE_ALPHA = 10.0


def winsorize_by_train(train_df, pred_df, numeric_features, bounds=(0.05, 0.95)):
    """Winsorize features and target based on training data quantiles."""
    train = train_df.copy()
    pred = pred_df.copy()
    lower, upper = bounds

    # Target
    y_lo = train[COL_TARGET].quantile(lower)
    y_hi = train[COL_TARGET].quantile(upper)
    train[COL_TARGET_W] = train[COL_TARGET].clip(y_lo, y_hi)
    pred[COL_TARGET_W] = pred[COL_TARGET].clip(y_lo, y_hi)

    # Features
    for col in numeric_features:
        if col not in train.columns:
            continue
        valid = train[col].dropna()
        if len(valid) == 0:
            continue
        lo = valid.quantile(lower)
        hi = valid.quantile(upper)
        train[col + '_winsor'] = train[col].clip(lo, hi)
        pred[col + '_winsor'] = pred[col].clip(lo, hi)

    return train, pred


def prepare_features(train_df, pred_df, numeric_features, use_industry=True):
    """Prepare X, y matrices with imputation, scaling, and industry FE."""
    num_cols_w = [f + '_winsor' for f in numeric_features
                  if f + '_winsor' in train_df.columns]

    train_valid = train_df.dropna(subset=num_cols_w + [COL_TARGET_W]).copy()
    pred_valid = pred_df.dropna(subset=num_cols_w).copy()

    X_train_num = train_valid[num_cols_w].values
    y_train = train_valid[COL_TARGET_W].values

    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(imp.fit_transform(X_train_num))

    if use_industry and COL_INDUSTRY in train_valid.columns:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ind_train = ohe.fit_transform(train_valid[[COL_INDUSTRY]].fillna('Unknown'))
        X_train = np.hstack([X_train_num, ind_train])

        X_pred_num = scaler.transform(imp.transform(pred_valid[num_cols_w].values))
        ind_pred = ohe.transform(pred_valid[[COL_INDUSTRY]].fillna('Unknown'))
        X_pred = np.hstack([X_pred_num, ind_pred])
    else:
        X_train = X_train_num
        X_pred = scaler.transform(imp.transform(pred_valid[num_cols_w].values))

    # Feature names
    feature_names = num_cols_w.copy()
    if use_industry and COL_INDUSTRY in train_valid.columns:
        feature_names += list(ohe.get_feature_names_out([COL_INDUSTRY]))

    return X_train, y_train, X_pred, train_valid, pred_valid, feature_names


def generate_oof_predictions(train_df, numeric_features, n_splits=5):
    """
    Generate strict OOF predictions on private firm training set.
    Each firm's predictions come from models NOT trained on that firm.
    """
    num_cols_w = [f + '_winsor' for f in numeric_features
                  if f + '_winsor' in train_df.columns]
    train_valid = train_df.dropna(subset=num_cols_w + [COL_TARGET_W]).copy()

    X_num = train_valid[num_cols_w].values
    y = train_valid[COL_TARGET_W].values
    groups = train_valid[COL_FIRM_ID].values

    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_num = scaler.fit_transform(imp.fit_transform(X_num))

    if COL_INDUSTRY in train_valid.columns:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ind = ohe.fit_transform(train_valid[[COL_INDUSTRY]].fillna('Unknown'))
        X = np.hstack([X_num, ind])
    else:
        X = X_num

    # Use actual n_splits capped by number of unique firms
    n_unique_firms = len(set(groups))
    actual_splits = min(n_splits, n_unique_firms)
    if actual_splits < 2:
        actual_splits = 2

    gkf = GroupKFold(n_splits=actual_splits)
    model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)

    oof_preds = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        model.fit(X_tr, y_tr)
        oof_preds[test_idx] = model.predict(X_te)
        fold_ids[test_idx] = fold_idx

    # Also fit on full training for SOE predictions
    model_full = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    model_full.fit(X, y)

    results = train_valid[[COL_FIRM_ID, COL_YEAR, COL_OWNERSHIP, COL_INDUSTRY,
                            COL_TARGET, COL_ASSETS, COL_ASSETS_TARGET,
                            COL_REVENUE]].copy()
    results['oof_predicted'] = oof_preds
    results['fold'] = fold_ids
    results['residual'] = results['oof_predicted'] - results[COL_TARGET]

    # Add features for later use
    for f in NUMERIC_FEATURES:
        if f in train_valid.columns:
            results[f] = train_valid[f].values

    return results, model_full, imp, scaler


# ============================================================================
# 2. Main Analysis Pipeline
# ============================================================================

def load_all_data():
    """Load and prepare all data."""
    print("=" * 70)
    print("  加载数据")
    print("=" * 70)

    print("\n[1] 全A股数据...")
    all_a = load_all_a_shares()
    all_a = clean_and_filter(all_a, "全A股")

    print("\n[2] CSI500数据...")
    csi500 = load_csi500()
    csi500 = clean_and_filter(csi500, "CSI500")

    print("\n[3] CSI1000数据...")
    csi1000 = load_csi1000()
    csi1000 = clean_and_filter(csi1000, "CSI1000")

    # Classify
    for df in [all_a, csi500, csi1000]:
        df[COL_SOE_FLAG] = df[COL_OWNERSHIP].isin(SOE_TYPES)

    # Training: private firms only
    train_full = all_a[all_a[COL_OWNERSHIP] == NON_SOE_TYPE].copy()

    # SOE targets from index files (use original file data)
    csi500_soe = csi500[csi500[COL_SOE_FLAG]].copy()
    csi1000_soe = csi1000[csi1000[COL_SOE_FLAG]].copy()

    # CSI300 SOE from all_a
    csi300_path = DATA_DIR.parent / "outputs" / "csi300_constituents.json"
    if csi300_path.exists():
        with open(csi300_path) as f:
            csi300_codes = set(json.load(f))
    else:
        csi300_codes = set()
    csi300_soe = all_a[(all_a[COL_SOE_FLAG]) &
                       (all_a[COL_YEAR] == 2025) &
                       (all_a[COL_FIRM_ID].isin(csi300_codes))].copy()

    print(f"\n  训练集 (民企): {len(train_full)} obs, {train_full[COL_FIRM_ID].nunique()} firms")
    print(f"  CSI300 SOE: {len(csi300_soe)} firms")
    print(f"  CSI500 SOE: {len(csi500_soe)} firms")
    print(f"  CSI1000 SOE: {len(csi1000_soe)} firms")

    return train_full, csi300_soe, csi500_soe, csi1000_soe


def run_full_diagnostics():
    """Run all four diagnostic modules."""
    t0 = time.time()

    # ---- Load data ----
    train_full, csi300_soe, csi500_soe, csi1000_soe = load_all_data()

    # ---- Winsorize based on training ----
    train_w, _ = winsorize_by_train(train_full, train_full.head(1), NUMERIC_FEATURES, WINSOR_BOUNDS)
    # Winsorize SOE targets based on training quantiles
    _, csi500_w = winsorize_by_train(train_full, csi500_soe, NUMERIC_FEATURES, WINSOR_BOUNDS)
    _, csi1000_w = winsorize_by_train(train_full, csi1000_soe, NUMERIC_FEATURES, WINSOR_BOUNDS)
    _, csi300_w = winsorize_by_train(train_full, csi300_soe, NUMERIC_FEATURES, WINSOR_BOUNDS)

    # ---- Generate OOF predictions on private firms ----
    print("\n" + "=" * 70)
    print("  生成 OOF 预测 (strict, by firm_id)")
    print("=" * 70)
    oof_results, model_full, imp, scaler = generate_oof_predictions(train_w, NUMERIC_FEATURES)
    print(f"  OOF predictions: {len(oof_results)} obs, "
          f"{oof_results[COL_FIRM_ID].nunique()} firms, "
          f"{oof_results['fold'].nunique()} folds")

    # Residual stats
    residuals = oof_results['residual'].values
    print(f"  OOF ME: {np.mean(residuals):+.4f}")
    print(f"  OOF MAE: {mean_absolute_error(oof_results[COL_TARGET], oof_results['oof_predicted']):.4f}")
    print(f"  OOF RMSE: {np.sqrt(mean_squared_error(oof_results[COL_TARGET], oof_results['oof_predicted'])):.4f}")
    print(f"  Pos residual: {np.mean(residuals > 0):.1%}")
    print(f"  Neg residual: {np.mean(residuals < 0):.1%}")

    # ---- Predict SOE targets ----
    # Prepare full training features
    num_cols_w = [f + '_winsor' for f in NUMERIC_FEATURES if f + '_winsor' in train_w.columns]
    train_valid = train_w.dropna(subset=num_cols_w + [COL_TARGET_W])

    X_train_num = imp.transform(scaler.transform(
        SimpleImputer(strategy='median').fit_transform(train_valid[num_cols_w].values) if False
        else StandardScaler().fit_transform(
            SimpleImputer(strategy='median').fit_transform(train_valid[num_cols_w].values)
        )
    ))

    # Actually, let's use the model_full already fitted in generate_oof_predictions
    # We need to predict on SOE targets using the same preprocessing

    def predict_soe(model, imp, scaler, train_w, soe_w):
        """Predict SOE targets using fitted model and preprocessing."""
        num_cols_w_local = [f + '_winsor' for f in NUMERIC_FEATURES
                            if f + '_winsor' in train_w.columns]
        soe_valid = soe_w.dropna(subset=num_cols_w_local)

        # Re-fit preprocessor on full training
        imp2 = SimpleImputer(strategy='median')
        imp2.fit(train_w[num_cols_w_local])
        scaler2 = StandardScaler()
        scaler2.fit(imp2.transform(train_w[num_cols_w_local]))

        X_soe_num = scaler2.transform(imp2.transform(soe_valid[num_cols_w_local].values))

        if COL_INDUSTRY in train_w.columns:
            ohe2 = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ohe2.fit(train_w[[COL_INDUSTRY]].fillna('Unknown'))
            ind_soe = ohe2.transform(soe_valid[[COL_INDUSTRY]].fillna('Unknown'))
            X_soe = np.hstack([X_soe_num, ind_soe])
        else:
            X_soe = X_soe_num

        # Re-fit model on full training
        # For the full pipeline, refit
        X_train_num2 = scaler2.transform(imp2.transform(train_w.dropna(
            subset=num_cols_w_local)[num_cols_w_local].values))
        y_train2 = train_w.dropna(subset=num_cols_w_local)[COL_TARGET_W].values

        if COL_INDUSTRY in train_w.columns:
            ohe2_train = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ohe2_train.fit(train_w[[COL_INDUSTRY]].fillna('Unknown'))
            ind_train2 = ohe2_train.transform(
                train_w.dropna(subset=num_cols_w_local)[[COL_INDUSTRY]].fillna('Unknown'))
            X_train_full = np.hstack([X_train_num2, ind_train2])
        else:
            X_train_full = X_train_num2

        m = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
        m.fit(X_train_full, y_train2)

        preds = m.predict(X_soe)
        result = soe_valid[[COL_FIRM_ID, COL_YEAR, COL_OWNERSHIP, COL_INDUSTRY,
                             COL_TARGET, COL_ASSETS, COL_ASSETS_TARGET, COL_REVENUE]].copy()
        result['predicted_EBI_A'] = preds
        result['gap'] = result['predicted_EBI_A'] - result[COL_TARGET]

        for f in NUMERIC_FEATURES:
            if f in soe_valid.columns:
                result[f] = soe_valid[f].values

        return result

    print("\n  预测 SOE 目标...")
    csi500_preds = predict_soe(None, None, None, train_w, csi500_w)
    csi500_preds['index_group'] = 'CSI500'
    csi1000_preds = predict_soe(None, None, None, train_w, csi1000_w)
    csi1000_preds['index_group'] = 'CSI1000'
    # Also CSI300 for completeness
    csi300_preds = predict_soe(None, None, None, train_w, csi300_w) if len(csi300_w) > 0 else None
    if csi300_preds is not None:
        csi300_preds['index_group'] = 'CSI300'

    soe_all = pd.concat([df for df in [csi500_preds, csi1000_preds, csi300_preds]
                         if df is not None], ignore_index=True)

    print(f"  CSI500 SOE gap: {csi500_preds['gap'].mean():+.4f}")
    print(f"  CSI1000 SOE gap: {csi1000_preds['gap'].mean():+.4f}")

    # Save OOF and SOE predictions
    oof_results.to_csv(OUTPUT_DIR / 'oof_private_predictions.csv', index=False, encoding='utf-8-sig')
    soe_all.to_csv(OUTPUT_DIR / 'soe_predictions.csv', index=False, encoding='utf-8-sig')

    # ========================================================================
    # MODULE 1: System Bias Diagnostics
    # ========================================================================
    print("\n" + "=" * 70)
    print("  一、系统 Bias 检验")
    print("=" * 70)

    run_module_1_bias(oof_results, train_w, csi500_preds, csi1000_preds)

    # ========================================================================
    # MODULE 2: Common Support Diagnostics
    # ========================================================================
    print("\n" + "=" * 70)
    print("  二、共同支持检验")
    print("=" * 70)

    run_module_2_support(train_w, csi500_w, csi1000_w, csi300_w,
                         csi500_preds, csi1000_preds, csi300_preds)

    # ========================================================================
    # MODULE 3: Placebo Gap Test
    # ========================================================================
    print("\n" + "=" * 70)
    print("  三、Placebo Gap 检验")
    print("=" * 70)

    placebo_results = run_module_3_placebo(train_w, csi500_preds, csi1000_preds,
                                           train_full)

    # ========================================================================
    # MODULE 4: Bootstrap Confidence Intervals
    # ========================================================================
    print("\n" + "=" * 70)
    print("  四、Bootstrap 置信区间")
    print("=" * 70)

    bootstrap_results = run_module_4_bootstrap(train_w, csi500_preds, csi1000_preds,
                                               placebo_results)

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("  最终汇总")
    print("=" * 70)

    generate_final_summary(oof_results, csi500_preds, csi1000_preds,
                           placebo_results, bootstrap_results)

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  输出目录: {OUTPUT_DIR}")

    return {
        'oof': oof_results,
        'csi500': csi500_preds,
        'csi1000': csi1000_preds,
        'placebo': placebo_results,
        'bootstrap': bootstrap_results,
    }


# ============================================================================
# MODULE 1: System Bias Diagnostics
# ============================================================================

def run_module_1_bias(oof_results, train_w, csi500_preds, csi1000_preds):
    """System bias diagnostics on private firm OOF predictions."""
    residuals = oof_results['residual'].values
    actual = oof_results[COL_TARGET].values
    predicted = oof_results['oof_predicted'].values
    firms = oof_results[COL_FIRM_ID].values

    # ---- 1.1 Overall OOF metrics ----
    me = np.mean(residuals)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    resid_std = np.std(residuals, ddof=1)
    resid_median = np.median(residuals)

    # Cluster-robust SE for ME
    unique_firms = np.unique(firms)
    firm_means = np.array([residuals[firms == f].mean() for f in unique_firms])
    n_firms = len(unique_firms)
    cluster_se = np.std(firm_means, ddof=1) / np.sqrt(n_firms)

    t_stat = me / cluster_se
    p_val = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n_firms - 1))
    ci_95 = (me - 1.96 * cluster_se, me + 1.96 * cluster_se)

    # Quantiles
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    quantiles = np.quantile(residuals, qs)

    print(f"\n  --- 1.1 总体 OOF 指标 ---")
    print(f"  OOF ME: {me:+.4f}")
    print(f"  Clustered SE: {cluster_se:.4f}")
    print(f"  95% CI: [{ci_95[0]:+.4f}, {ci_95[1]:+.4f}]")
    print(f"  t-stat: {t_stat:.4f}, p-value: {p_val:.4f}")
    print(f"  OOF MAE: {mae:.4f}")
    print(f"  OOF RMSE: {rmse:.4f}")
    print(f"  Residual SD: {resid_std:.4f}")
    print(f"  Residual median: {resid_median:+.4f}")
    for q_val, q_name in zip(quantiles, qs):
        print(f"  {q_name:.0%} quantile: {q_val:+.4f}")
    print(f"  Pos residual: {np.mean(residuals > 0):.1%}")
    print(f"  Neg residual: {np.mean(residuals < 0):.1%}")

    # ---- 1.2 Calibration regression ----
    X_cal = np.column_stack([np.ones(len(actual)), predicted])
    beta_hat = np.linalg.lstsq(X_cal, actual, rcond=None)[0]
    alpha_hat, beta_hat_val = beta_hat[0], beta_hat[1]

    # Cluster-robust SE for calibration
    cal_resid = actual - (alpha_hat + beta_hat_val * predicted)
    n_obs = len(cal_resid)
    # Sandwich estimator with firm clustering
    Qxx_inv = np.linalg.inv(X_cal.T @ X_cal / n_obs)
    # Cluster-robust meat
    meat = np.zeros((2, 2))
    for f in unique_firms:
        mask = firms == f
        X_f = X_cal[mask]
        e_f = cal_resid[mask]
        score_f = X_f.T @ e_f
        meat += np.outer(score_f, score_f)
    vcov_cluster = Qxx_inv @ (meat / n_obs) @ Qxx_inv / n_obs
    se_alpha = np.sqrt(vcov_cluster[0, 0])
    se_beta = np.sqrt(vcov_cluster[1, 1])

    # Test alpha=0
    t_alpha = alpha_hat / se_alpha
    p_alpha = 2 * (1 - scipy_stats.t.cdf(abs(t_alpha), df=n_firms - 1))
    # Test beta=1
    t_beta = (beta_hat_val - 1) / se_beta
    p_beta = 2 * (1 - scipy_stats.t.cdf(abs(t_beta), df=n_firms - 1))
    # Joint test (Wald)
    from scipy.stats import chi2
    R = np.array([[1, 0], [0, 1]])
    r = np.array([0, 1])
    diff = np.array([alpha_hat, beta_hat_val]) - r
    wald = diff @ np.linalg.inv(vcov_cluster) @ diff
    p_joint = 1 - chi2.cdf(wald, 2)

    print(f"\n  --- 1.2 Calibration: actual = alpha + beta * predicted ---")
    print(f"  alpha = {alpha_hat:.4f} (SE={se_alpha:.4f}, p={p_alpha:.4f})")
    print(f"  beta = {beta_hat_val:.4f} (SE={se_beta:.4f}, p(β=1)={p_beta:.4f})")
    print(f"  Joint test (α=0, β=1): Wald={wald:.4f}, p={p_joint:.4f}")

    # ---- 1.3 Subgroup bias ----
    print(f"\n  --- 1.3 分组 Bias ---")

    # Add derived features
    oof_results['residual'] = residuals
    oof_results['size_group'] = pd.qcut(oof_results[COL_ASSETS].rank(method='first'), 5,
                                         labels=['Q1_small', 'Q2', 'Q3', 'Q4', 'Q5_large'])
    oof_results['leverage'] = oof_results['FinancialLeverage'] if 'FinancialLeverage' in oof_results.columns else np.nan
    if 'FinancialLeverage' in oof_results.columns:
        oof_results['lev_group'] = pd.qcut(oof_results['FinancialLeverage'].rank(method='first'), 4,
                                            labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
    oof_results['profitability'] = oof_results[COL_TARGET]

    subgroups = {
        'year': COL_YEAR,
        'industry': COL_INDUSTRY,
        'size': 'size_group',
    }
    if 'lev_group' in oof_results.columns:
        subgroups['leverage'] = 'lev_group'

    subgroup_table = []
    for sg_name, sg_col in subgroups.items():
        if sg_col not in oof_results.columns:
            continue
        for sg_val, sg_data in oof_results.groupby(sg_col):
            subgroup_table.append({
                'subgroup': sg_name,
                'value': str(sg_val),
                'n': len(sg_data),
                'ME': sg_data['residual'].mean(),
                'MAE': mean_absolute_error(sg_data[COL_TARGET], sg_data['oof_predicted']),
                'RMSE': np.sqrt(mean_squared_error(sg_data[COL_TARGET], sg_data['oof_predicted'])),
            })

    sg_df = pd.DataFrame(subgroup_table)
    sg_df.to_csv(OUTPUT_DIR / 'subgroup_bias.csv', index=False, encoding='utf-8-sig')
    print(f"  分组 bias 已保存 ({len(sg_df)} 行)")

    # ---- 1.4 SOE-like private firms ----
    print(f"\n  --- 1.4 SOE-like 民企子样本 ---")
    soe_like_bias(oof_results, train_w, csi500_preds, csi1000_preds)

    # Save overall bias summary
    bias_summary = {
        'OOF_ME': me, 'OOF_MAE': mae, 'OOF_RMSE': rmse,
        'Residual_SD': resid_std, 'Residual_median': resid_median,
        'ME_clustered_SE': cluster_se, 'ME_95CI_lower': ci_95[0], 'ME_95CI_upper': ci_95[1],
        'ME_t_stat': t_stat, 'ME_p_value': p_val,
        'calib_alpha': alpha_hat, 'calib_beta': beta_hat_val,
        'calib_p_alpha0': p_alpha, 'calib_p_beta1': p_beta, 'calib_p_joint': p_joint,
    }
    for q_val, q_name in zip(quantiles, qs):
        bias_summary[f'residual_q{q_name:.0%}'] = q_val
    bias_summary['pos_residual_pct'] = np.mean(residuals > 0)
    bias_summary['neg_residual_pct'] = np.mean(residuals < 0)

    pd.Series(bias_summary).to_csv(OUTPUT_DIR / 'bias_summary.csv', encoding='utf-8-sig')
    print(f"  Bias 摘要已保存")

    return bias_summary


def soe_like_bias(oof_results, train_w, csi500_preds, csi1000_preds):
    """Create SOE-like private firm subsample and compute bias."""
    # Build combined dataset for propensity score estimation
    private_marked = train_w.dropna(subset=[f + '_winsor' for f in NUMERIC_FEATURES
                                             if f + '_winsor' in train_w.columns]).copy()
    private_marked['is_soe_sample'] = 0

    soe_combined = pd.concat([csi500_preds, csi1000_preds], ignore_index=True)

    # Features for propensity score
    ps_features = [f for f in NUMERIC_FEATURES if f in private_marked.columns]

    # Use log(assets) for better scaling
    if COL_ASSETS in private_marked.columns:
        private_marked['log_assets'] = np.log(private_marked[COL_ASSETS].clip(lower=1))
        if 'log_assets' not in soe_combined.columns and COL_ASSETS in soe_combined.columns:
            soe_combined['log_assets'] = np.log(soe_combined[COL_ASSETS].clip(lower=1))
        ps_features = ps_features + ['log_assets']

    # Combine
    combined = pd.concat([
        private_marked[[COL_FIRM_ID] + [c for c in ps_features if c in private_marked.columns]],
        soe_combined[[COL_FIRM_ID] + [c for c in ps_features if c in soe_combined.columns]].assign(is_soe_sample=1)
    ], ignore_index=True).fillna(0)

    X_ps = combined[[c for c in ps_features if c in combined.columns]].values
    y_ps = combined['is_soe_sample'].values

    # Logistic regression for propensity score
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_ps, y_ps)
    ps_scores = lr.predict_proba(X_ps)[:, 1]

    # Get private firm propensity scores
    private_ps = ps_scores[combined['is_soe_sample'] == 0]
    private_firms = combined.loc[combined['is_soe_sample'] == 0, COL_FIRM_ID].values
    soe_ps = ps_scores[combined['is_soe_sample'] == 1]

    # SOE-like: top 20% propensity score among private firms
    ps_threshold = np.quantile(private_ps, 0.80)
    soe_like_firms = private_firms[private_ps >= ps_threshold]
    soe_like_mask = oof_results[COL_FIRM_ID].isin(soe_like_firms)
    soe_like_results = oof_results[soe_like_mask]

    soe_like_me = soe_like_results['residual'].mean()
    soe_like_mae = mean_absolute_error(soe_like_results[COL_TARGET],
                                        soe_like_results['oof_predicted'])
    soe_like_rmse = np.sqrt(mean_squared_error(soe_like_results[COL_TARGET],
                                                soe_like_results['oof_predicted']))
    # Cluster SE
    sl_firms = soe_like_results[COL_FIRM_ID].values
    sl_unique = np.unique(sl_firms)
    sl_means = np.array([soe_like_results['residual'].values[sl_firms == f].mean()
                         for f in sl_unique])
    sl_se = np.std(sl_means, ddof=1) / np.sqrt(len(sl_unique))
    sl_ci = (soe_like_me - 1.96 * sl_se, soe_like_me + 1.96 * sl_se)

    print(f"  SOE-like 民企 n={len(soe_like_results)}, firms={len(sl_unique)}")
    print(f"  SOE-like ME: {soe_like_me:+.4f} (SE={sl_se:.4f})")
    print(f"  SOE-like 95% CI: [{sl_ci[0]:+.4f}, {sl_ci[1]:+.4f}]")
    print(f"  SOE-like MAE: {soe_like_mae:.4f}, RMSE: {soe_like_rmse:.4f}")

    # Also do nearest-neighbor matching
    from sklearn.neighbors import NearestNeighbors
    # Standardize features
    scaler_ps = StandardScaler()
    X_ps_scaled = scaler_ps.fit_transform(X_ps)
    X_priv_scaled = X_ps_scaled[combined['is_soe_sample'] == 0]
    X_soe_scaled = X_ps_scaled[combined['is_soe_sample'] == 1]

    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X_priv_scaled)
    distances, indices = nn.kneighbors(X_soe_scaled)

    # Matched private firms
    matched_firms = set()
    for idx_row in indices:
        for idx in idx_row:
            matched_firms.add(private_firms[idx])
    matched_mask = oof_results[COL_FIRM_ID].isin(matched_firms)
    matched_results = oof_results[matched_mask]

    matched_me = matched_results['residual'].mean()
    mf_unique = np.unique(matched_results[COL_FIRM_ID].values)
    mf_means = np.array([matched_results['residual'].values[
        matched_results[COL_FIRM_ID].values == f].mean() for f in mf_unique])
    mf_se = np.std(mf_means, ddof=1) / np.sqrt(len(mf_unique))
    mf_ci = (matched_me - 1.96 * mf_se, matched_me + 1.96 * mf_se)

    print(f"\n  NN-matched 民企 n={len(matched_results)}, firms={len(mf_unique)}")
    print(f"  NN-matched ME: {matched_me:+.4f} (SE={mf_se:.4f})")
    print(f"  NN-matched 95% CI: [{mf_ci[0]:+.4f}, {mf_ci[1]:+.4f}]")

    # Save
    soe_like_summary = {
        'soe_like_n': len(soe_like_results),
        'soe_like_firms': len(sl_unique),
        'soe_like_ME': soe_like_me,
        'soe_like_SE': sl_se,
        'soe_like_95CI_lower': sl_ci[0],
        'soe_like_95CI_upper': sl_ci[1],
        'soe_like_MAE': soe_like_mae,
        'soe_like_RMSE': soe_like_rmse,
        'nn_matched_n': len(matched_results),
        'nn_matched_firms': len(mf_unique),
        'nn_matched_ME': matched_me,
        'nn_matched_SE': mf_se,
        'nn_matched_95CI_lower': mf_ci[0],
        'nn_matched_95CI_upper': mf_ci[1],
    }
    pd.Series(soe_like_summary).to_csv(OUTPUT_DIR / 'soe_like_bias.csv', encoding='utf-8-sig')

    print(f"\n  --- 1.5 Bias 结论 ---")
    overall_me = oof_results['residual'].mean()
    print(f"  总体 OOF ME = {overall_me:+.4f}")
    print(f"  模型总体{'存在' if abs(overall_me / (oof_results['residual'].std() / np.sqrt(len(oof_results)))) > 1.96 else '不存在'}显著系统偏差")
    print(f"  SOE-like 民企 ME = {soe_like_me:+.4f} (总体={overall_me:+.4f})")
    csi500_gap = csi500_preds['gap'].mean()
    csi1000_gap = csi1000_preds['gap'].mean()
    print(f"  CSI500 SOE gap = {csi500_gap:+.4f}, SOE-like bias = {soe_like_me:+.4f}")
    print(f"  Gap 能否被 bias 解释: {'是' if abs(csi500_gap) < abs(soe_like_me) + sl_se else '否'}")

    return soe_like_summary


# ============================================================================
# MODULE 2: Common Support Diagnostics
# ============================================================================

def run_module_2_support(train_w, csi500_w, csi1000_w, csi300_w,
                         csi500_preds, csi1000_preds, csi300_preds):
    """Common support diagnostics."""
    num_cols = [f for f in NUMERIC_FEATURES if f in train_w.columns]

    # ---- 2.1 Univariate SMD ----
    print(f"\n  --- 2.1 单变量重叠 (SMD) ---")
    smd_results = []
    for col in num_cols:
        priv = train_w[col].dropna()
        soe5 = csi500_w[col].dropna() if len(csi500_w) > 0 else pd.Series(dtype=float)
        soe10 = csi1000_w[col].dropna() if len(csi1000_w) > 0 else pd.Series(dtype=float)

        pooled_sd = np.sqrt((priv.var() + soe5.var()) / 2) if len(soe5) > 0 else priv.std()

        for soe_name, soe_data in [('CSI500', soe5), ('CSI1000', soe10)]:
            if len(soe_data) == 0:
                continue
            smd = (soe_data.mean() - priv.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            pct_outside_1_99 = np.mean((soe_data < priv.quantile(0.01)) |
                                       (soe_data > priv.quantile(0.99)))
            pct_outside_minmax = np.mean((soe_data < priv.min()) |
                                         (soe_data > priv.max()))
            smd_results.append({
                'variable': col,
                'group': soe_name,
                'priv_mean': priv.mean(),
                'soe_mean': soe_data.mean(),
                'SMD': smd,
                '|SMD|>0.10': abs(smd) > 0.10 if not np.isnan(smd) else False,
                '|SMD|>0.25': abs(smd) > 0.25 if not np.isnan(smd) else False,
                'pct_outside_P1_P99': pct_outside_1_99,
                'pct_outside_minmax': pct_outside_minmax,
            })

    smd_df = pd.DataFrame(smd_results)
    smd_df.to_csv(OUTPUT_DIR / 'smd_results.csv', index=False, encoding='utf-8-sig')

    # Flag problematic variables
    large_smd = smd_df[smd_df['|SMD|>0.25']]
    print(f"  |SMD|>0.25 的变量: {len(large_smd)}")
    for _, r in large_smd.iterrows():
        print(f"    {r['variable']} ({r['group']}): SMD={r['SMD']:+.3f}")

    # ---- 2.2 Propensity Score Overlap ----
    print(f"\n  --- 2.2 Propensity Score 重叠 ---")
    ps_features = [f for f in NUMERIC_FEATURES if f in train_w.columns]
    if COL_ASSETS in train_w.columns:
        train_w_copy = train_w.copy()
        train_w_copy['log_assets'] = np.log(train_w_copy[COL_ASSETS].clip(lower=1))
        ps_features = ps_features + ['log_assets']

    # Build training: private=0, SOE=1
    all_soe_for_ps = pd.concat([csi500_preds, csi1000_preds], ignore_index=True)
    if 'log_assets' not in all_soe_for_ps.columns and COL_ASSETS in all_soe_for_ps.columns:
        all_soe_for_ps = all_soe_for_ps.copy()
        all_soe_for_ps['log_assets'] = np.log(all_soe_for_ps[COL_ASSETS].clip(lower=1))

    ps_cols = [c for c in ps_features if c in train_w.columns and c in all_soe_for_ps.columns]
    priv_data = train_w[ps_cols].dropna()
    soe_data = all_soe_for_ps[ps_cols].dropna()

    X_ps = np.vstack([priv_data.values, soe_data.values])
    y_ps = np.hstack([np.zeros(len(priv_data)), np.ones(len(soe_data))])

    # OOF propensity scores via cross-validation
    n_splits_ps = min(5, len(priv_data) // 100)
    if n_splits_ps >= 2:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits_ps, shuffle=True, random_state=RANDOM_STATE)
        oof_ps = np.full(len(X_ps), np.nan)
        for tr_idx, te_idx in skf.split(X_ps, y_ps):
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            lr.fit(X_ps[tr_idx], y_ps[tr_idx])
            oof_ps[te_idx] = lr.predict_proba(X_ps[te_idx])[:, 1]
    else:
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_ps, y_ps)
        oof_ps = lr.predict_proba(X_ps)[:, 1]

    private_ps = oof_ps[y_ps == 0]
    soe_ps_all = oof_ps[y_ps == 1]

    # ROC-AUC
    try:
        auc_roc = roc_auc_score(y_ps, oof_ps)
        auc_pr = average_precision_score(y_ps, oof_ps)
    except:
        auc_roc, auc_pr = np.nan, np.nan

    print(f"  OOF ROC-AUC: {auc_roc:.4f}")
    print(f"  OOF PR-AUC: {auc_pr:.4f}")
    print(f"  {'AUC 较高，SOE与民企容易被协变量区分' if auc_roc > 0.75 else 'AUC 适中'}")

    # Common support
    ps_min = private_ps.min()
    ps_max = private_ps.max()
    ps_p01 = np.quantile(private_ps, 0.01)
    ps_p99 = np.quantile(private_ps, 0.99)
    soe_outside = np.mean((soe_ps_all < ps_min) | (soe_ps_all > ps_max))
    soe_outside_trimmed = np.mean((soe_ps_all < ps_p01) | (soe_ps_all > ps_p99))

    print(f"  Private PS range: [{ps_min:.4f}, {ps_max:.4f}]")
    print(f"  Private PS 1%-99%: [{ps_p01:.4f}, {ps_p99:.4f}]")
    print(f"  SOE outside private PS range: {soe_outside:.1%}")
    print(f"  SOE outside trimmed PS range: {soe_outside_trimmed:.1%}")

    # ---- 2.3 Nearest-neighbor distances ----
    print(f"\n  --- 2.3 最近邻距离 ---")
    scaler_nn = StandardScaler()
    X_nn_scaled = scaler_nn.fit_transform(X_ps)
    X_priv_nn = X_nn_scaled[y_ps == 0]
    X_soe_nn = X_nn_scaled[y_ps == 1]

    # Private internal distances (leave-one-out)
    nn_priv = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn_priv.fit(X_priv_nn)
    priv_dist, _ = nn_priv.kneighbors(X_priv_nn)
    priv_dist = priv_dist[:, 1]  # Distance to nearest other private firm

    # SOE to private distances
    nn_soe = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn_soe.fit(X_priv_nn)
    soe_dist, soe_indices = nn_soe.kneighbors(X_soe_nn)
    soe_dist_1nn = soe_dist[:, 0]
    soe_dist_5nn = soe_dist.mean(axis=1)

    # Thresholds from private distribution
    p90 = np.quantile(priv_dist, 0.90)
    p95 = np.quantile(priv_dist, 0.95)

    support_labels = []
    for d in soe_dist_1nn:
        if d <= p90:
            support_labels.append('良好支持')
        elif d <= p95:
            support_labels.append('边界')
        else:
            support_labels.append('外推风险')

    support_counts = Counter(support_labels)
    print(f"  民企内部距离: median={np.median(priv_dist):.4f}, P90={p90:.4f}, P95={p95:.4f}")
    print(f"  SOE 最近邻距离: median={np.median(soe_dist_1nn):.4f}")
    for k, v in support_counts.items():
        print(f"    {k}: {v} ({v/len(support_labels):.1%})")

    # ---- 2.4 PCA ----
    print(f"\n  --- 2.4 PCA 外推诊断 ---")
    pca = PCA(n_components=min(5, X_nn_scaled.shape[1]))
    pca.fit(X_priv_nn)
    priv_pca = pca.transform(X_priv_nn)
    soe_pca = pca.transform(X_soe_nn)

    # Mahalanobis distance in PCA space
    try:
        priv_cov_inv = np.linalg.inv(np.cov(priv_pca.T))
        priv_center = priv_pca.mean(axis=0)
        soe_mahalanobis = np.array([
            np.sqrt((sp - priv_center).T @ priv_cov_inv @ (sp - priv_center))
            for sp in soe_pca
        ])
        priv_mahalanobis = np.array([
            np.sqrt((pp - priv_center).T @ priv_cov_inv @ (pp - priv_center))
            for pp in priv_pca
        ])
        mah_p90 = np.quantile(priv_mahalanobis, 0.90)
        mah_p95 = np.quantile(priv_mahalanobis, 0.95)
        soe_outside_mah_p90 = np.mean(soe_mahalanobis > mah_p90)
        soe_outside_mah_p95 = np.mean(soe_mahalanobis > mah_p95)
        print(f"  PCA空间马氏距离: 民企P90={mah_p90:.2f}, P95={mah_p95:.2f}")
        print(f"  SOE超出民企P90: {soe_outside_mah_p90:.1%}, 超出P95: {soe_outside_mah_p95:.1%}")
    except Exception as e:
        print(f"  马氏距离计算失败: {e}")

    # Isolation Forest
    iso = IsolationForest(random_state=RANDOM_STATE, contamination=0.1)
    iso.fit(X_priv_nn)
    soe_if_scores = iso.decision_function(X_soe_nn)
    soe_if_outlier = iso.predict(X_soe_nn)
    print(f"  Isolation Forest: SOE 中被标记为异常的: {np.mean(soe_if_outlier == -1):.1%}")

    # ---- 2.5 Support-restricted gap ----
    print(f"\n  --- 2.5 支持集限制后的 gap ---")
    all_soe_df = pd.concat([csi500_preds, csi1000_preds], ignore_index=True)

    # Add support info to SOE dataframe
    all_soe_df = all_soe_df.iloc[:len(support_labels)]  # Align
    all_soe_df['support_label'] = support_labels
    all_soe_df['nn_distance'] = soe_dist_1nn
    all_soe_df['propensity_score'] = soe_ps_all

    restrictions = {
        '全部 SOE': slice(None),
        '良好支持 (dist≤P90)': all_soe_df['support_label'] == '良好支持',
        'trimmed PS (P1-P99)': (all_soe_df['propensity_score'] >= ps_p01) &
                               (all_soe_df['propensity_score'] <= ps_p99),
        '去除 NN dist top 5%': all_soe_df['nn_distance'] <= np.quantile(soe_dist_1nn, 0.95),
        '去除 NN dist top 10%': all_soe_df['nn_distance'] <= np.quantile(soe_dist_1nn, 0.90),
    }

    support_table = []
    for name, mask in restrictions.items():
        sub = all_soe_df[mask]
        if len(sub) == 0:
            continue
        support_table.append({
            'restriction': name,
            'n': len(sub),
            'mean_gap': sub['gap'].mean(),
            'median_gap': sub['gap'].median(),
            'gap_sd': sub['gap'].std(),
        })
        print(f"  {name}: n={len(sub)}, gap={sub['gap'].mean():+.4f}, median={sub['gap'].median():+.4f}")
        # Also by index
        for idx_grp in ['CSI500', 'CSI1000']:
            idx_sub = sub[sub.get('index_group', '') == idx_grp]
            if len(idx_sub) > 0:
                print(f"    {idx_grp}: n={len(idx_sub)}, gap={idx_sub['gap'].mean():+.4f}")

    support_df = pd.DataFrame(support_table)
    support_df.to_csv(OUTPUT_DIR / 'support_restricted_gap.csv', index=False, encoding='utf-8-sig')

    # ---- 2.6 Conclusions ----
    print(f"\n  --- 2.6 共同支持结论 ---")
    n_good = support_counts.get('良好支持', 0)
    n_extrap = support_counts.get('外推风险', 0)
    n_total = len(support_labels)
    print(f"  良好支持: {n_good}/{n_total} ({n_good/n_total:.1%})")
    print(f"  外推风险: {n_extrap}/{n_total} ({n_extrap/n_total:.1%})")
    print(f"  CSI500 vs CSI1000 支持差异: 见 support_restricted_gap.csv")


# ============================================================================
# MODULE 3: Placebo Gap Test
# ============================================================================

def run_module_3_placebo(train_w, csi500_preds, csi1000_preds, train_full):
    """Matched placebo gap test."""
    print(f"\n  Placebo 迭代次数: {N_PLACEBO}")
    print(f"  (计算量较大，可能需要数分钟)")

    num_cols_w = [f + '_winsor' for f in NUMERIC_FEATURES if f + '_winsor' in train_w.columns]
    train_valid = train_w.dropna(subset=num_cols_w + [COL_TARGET_W])

    # Get SOE characteristics for matching
    csi500_soe_combined = csi500_preds
    csi1000_soe_combined = csi1000_preds

    # Matching dimensions: year, size, industry
    # We'll match on deciles of total assets within industry×year
    train_valid_copy = train_valid.copy()
    if COL_ASSETS in train_valid_copy.columns:
        train_valid_copy['size_decile'] = pd.qcut(train_valid_copy[COL_ASSETS].rank(method='first'),
                                                   10, labels=False)

    # For each SOE group, define target counts
    csi500_n = len(csi500_soe_combined)
    csi1000_n = len(csi1000_soe_combined)

    # Pre-compute private firm pools by industry×size for stratified sampling
    if COL_INDUSTRY in train_valid_copy.columns:
        strata_pools = {}
        for (ind, sz), grp in train_valid_copy.groupby([COL_INDUSTRY, 'size_decile'] if
                                                        'size_decile' in train_valid_copy.columns
                                                        else [COL_INDUSTRY]):
            strata_pools[(ind, sz)] = grp[COL_FIRM_ID].unique()

    # Placebo for CSI500
    print(f"\n  --- Placebo: CSI500 (target n={csi500_n}) ---")
    placebo_gaps_500 = run_placebo_iterations(
        train_valid_copy, csi500_soe_combined, csi500_n,
        NUMERIC_FEATURES, strata_pools, label='CSI500'
    )

    # Placebo for CSI1000
    print(f"\n  --- Placebo: CSI1000 (target n={csi1000_n}) ---")
    placebo_gaps_1000 = run_placebo_iterations(
        train_valid_copy, csi1000_soe_combined, csi1000_n,
        NUMERIC_FEATURES, strata_pools, label='CSI1000'
    )

    # ---- 3.3 Placebo inference ----
    G_real_500 = csi500_preds['gap'].mean()
    G_real_1000 = csi1000_preds['gap'].mean()

    results = {}
    for label, placebo_gaps, G_real in [
        ('CSI500', placebo_gaps_500, G_real_500),
        ('CSI1000', placebo_gaps_1000, G_real_1000)
    ]:
        pg = np.array(placebo_gaps)
        pg = pg[~np.isnan(pg)]  # Remove failed iterations

        placebo_mean = np.mean(pg)
        placebo_sd = np.std(pg, ddof=1)
        placebo_qs = np.quantile(pg, [0.025, 0.05, 0.50, 0.95, 0.975])

        # Empirical p-value (one-sided: placebo >= real)
        p_one_sided = (1 + np.sum(pg >= G_real)) / (len(pg) + 1)
        p_two_sided = (1 + np.sum(np.abs(pg) >= np.abs(G_real))) / (len(pg) + 1)

        # z-score
        z_placebo = (G_real - placebo_mean) / placebo_sd if placebo_sd > 0 else np.nan

        # Bias-corrected gap
        G_corrected = G_real - placebo_mean

        # Test H0: placebo_mean = 0
        t_placebo_zero = placebo_mean / (placebo_sd / np.sqrt(len(pg)))
        p_placebo_zero = 2 * (1 - scipy_stats.t.cdf(abs(t_placebo_zero), df=len(pg) - 1))

        print(f"\n  [{label}]")
        print(f"  Real gap: {G_real:+.4f}")
        print(f"  Placebo mean: {placebo_mean:+.4f} (SD={placebo_sd:.4f})")
        print(f"  Placebo median: {placebo_qs[2]:+.4f}")
        print(f"  Placebo 95% CI: [{placebo_qs[0]:+.4f}, {placebo_qs[4]:+.4f}]")
        print(f"  Empirical p (one-sided): {p_one_sided:.4f}")
        print(f"  Empirical p (two-sided): {p_two_sided:.4f}")
        print(f"  z_placebo: {z_placebo:+.4f}")
        print(f"  Placebo mean = 0 test: t={t_placebo_zero:.4f}, p={p_placebo_zero:.4f}")
        print(f"  Bias-corrected gap: {G_corrected:+.4f}")

        results[label] = {
            'real_gap': G_real,
            'placebo_mean': placebo_mean,
            'placebo_sd': placebo_sd,
            'placebo_median': placebo_qs[2],
            'placebo_95CI_lower': placebo_qs[0],
            'placebo_95CI_upper': placebo_qs[4],
            'p_one_sided': p_one_sided,
            'p_two_sided': p_two_sided,
            'z_placebo': z_placebo,
            'bias_corrected_gap': G_corrected,
            'placebo_mean_zero_p': p_placebo_zero,
        }

    # Save placebo distributions
    pd.DataFrame({'CSI500_placebo_gap': placebo_gaps_500,
                  'CSI1000_placebo_gap': placebo_gaps_1000[:len(placebo_gaps_500)]}
                 ).to_csv(OUTPUT_DIR / 'placebo_distributions.csv', index=False, encoding='utf-8-sig')

    # Save summary
    placebo_summary = pd.DataFrame(results).T
    placebo_summary.to_csv(OUTPUT_DIR / 'placebo_summary.csv', encoding='utf-8-sig')

    return results


def run_placebo_iterations(train_valid, soe_data, target_n, numeric_features,
                           strata_pools, label='', n_iter=N_PLACEBO):
    """Run placebo iterations with stratified matching."""
    gaps = []
    n_actual = min(n_iter, 200) if n_iter > 200 else n_iter  # Reduce for speed if needed
    n_actual = n_iter  # Use full count

    # Pre-compute SOE industry distribution for matching
    if COL_INDUSTRY in soe_data.columns:
        soe_ind_dist = soe_data[COL_INDUSTRY].value_counts(normalize=True)

    num_cols_w = [f + '_winsor' for f in numeric_features
                  if f + '_winsor' in train_valid.columns]

    for b in range(n_iter):
        if (b + 1) % 200 == 0:
            print(f"    {label} placebo: {b+1}/{n_iter}")

        try:
            # Sample placebo firms matching SOE characteristics
            available = train_valid.copy()

            # Stratified sampling by industry
            if COL_INDUSTRY in soe_data.columns and COL_INDUSTRY in available.columns:
                placebo_firms = []
                for ind, prop in soe_ind_dist.items():
                    n_ind = max(1, int(target_n * prop))
                    ind_pool = available[available[COL_INDUSTRY] == ind]
                    if len(ind_pool) >= n_ind:
                        sampled = ind_pool.sample(n=n_ind, random_state=RANDOM_STATE + b)
                        placebo_firms.append(sampled[COL_FIRM_ID].values)
                    elif len(ind_pool) > 0:
                        placebo_firms.append(ind_pool[COL_FIRM_ID].values)
                placebo_fids = np.concatenate(placebo_firms)[:target_n]
            else:
                placebo_fids = available[COL_FIRM_ID].sample(
                    n=min(target_n, available[COL_FIRM_ID].nunique()),
                    random_state=RANDOM_STATE + b
                ).values

            # Split: training excludes placebo firms
            train_mask = ~available[COL_FIRM_ID].isin(placebo_fids)
            placebo_data = available[available[COL_FIRM_ID].isin(placebo_fids)]

            if len(placebo_data) < 10:
                continue

            # Train on remaining private firms
            train_subset = available[train_mask]
            X_tr_num = SimpleImputer(strategy='median').fit_transform(train_subset[num_cols_w].values)
            scaler_b = StandardScaler().fit(X_tr_num)
            X_tr = scaler_b.transform(SimpleImputer(strategy='median').fit_transform(
                train_subset[num_cols_w].values))
            y_tr = train_subset[COL_TARGET_W].values

            if COL_INDUSTRY in train_subset.columns:
                ohe_b = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                ind_tr = ohe_b.fit_transform(train_subset[[COL_INDUSTRY]].fillna('Unknown'))
                X_tr = np.hstack([X_tr, ind_tr])

            model_b = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
            model_b.fit(X_tr, y_tr)

            # Predict placebo
            X_pl_num = scaler_b.transform(SimpleImputer(strategy='median').fit_transform(
                placebo_data[num_cols_w].values))
            if COL_INDUSTRY in train_subset.columns:
                ind_pl = ohe_b.transform(placebo_data[[COL_INDUSTRY]].fillna('Unknown'))
                X_pl = np.hstack([X_pl_num, ind_pl])
            else:
                X_pl = X_pl_num

            preds = model_b.predict(X_pl)
            gap_b = np.mean(preds - placebo_data[COL_TARGET].values)
            gaps.append(gap_b)

        except Exception as e:
            gaps.append(np.nan)

    return gaps


# ============================================================================
# MODULE 4: Bootstrap Confidence Intervals
# ============================================================================

def run_module_4_bootstrap(train_w, csi500_preds, csi1000_preds, placebo_results):
    """Cluster bootstrap confidence intervals."""
    print(f"\n  Bootstrap 迭代次数: {N_BOOTSTRAP}")
    print(f"  (计算量较大，包含完整 refit)")

    train_valid = train_w.dropna(
        subset=[f + '_winsor' for f in NUMERIC_FEATURES if f + '_winsor' in train_w.columns] +
               [COL_TARGET_W])

    private_firms = train_valid[COL_FIRM_ID].unique()
    soe500_firms = csi500_preds[COL_FIRM_ID].unique()
    soe1000_firms = csi1000_preds[COL_FIRM_ID].unique()

    n_private = len(private_firms)
    n_soe500 = len(soe500_firms)
    n_soe1000 = len(soe1000_firms)

    # Bootstrap
    boot_gaps_500 = []
    boot_gaps_1000 = []
    boot_deltas = []

    for b in range(N_BOOTSTRAP):
        if (b + 1) % 200 == 0:
            print(f"    Bootstrap: {b+1}/{N_BOOTSTRAP}")

        try:
            # Sample firms with replacement
            boot_priv_fids = np.random.choice(private_firms, size=n_private, replace=True)
            boot_soe500_fids = np.random.choice(soe500_firms, size=n_soe500, replace=True)
            boot_soe1000_fids = np.random.choice(soe1000_firms, size=n_soe1000, replace=True)

            # Build bootstrap datasets
            boot_train = pd.concat([
                train_valid[train_valid[COL_FIRM_ID] == f] for f in boot_priv_fids
            ], ignore_index=True)

            boot_soe500 = pd.concat([
                csi500_preds[csi500_preds[COL_FIRM_ID] == f] for f in boot_soe500_fids
            ], ignore_index=True)

            boot_soe1000 = pd.concat([
                csi1000_preds[csi1000_preds[COL_FIRM_ID] == f] for f in boot_soe1000_fids
            ], ignore_index=True)

            # Winsorize based on bootstrap training
            for col in NUMERIC_FEATURES:
                if col not in boot_train.columns:
                    continue
                lo = boot_train[col].quantile(0.05)
                hi = boot_train[col].quantile(0.95)
                boot_train[col + '_winsor'] = boot_train[col].clip(lo, hi)
                if col in boot_soe500.columns:
                    boot_soe500[col + '_winsor'] = boot_soe500[col].clip(lo, hi)
                if col in boot_soe1000.columns:
                    boot_soe1000[col + '_winsor'] = boot_soe1000[col].clip(lo, hi)

            y_lo = boot_train[COL_TARGET].quantile(0.05)
            y_hi = boot_train[COL_TARGET].quantile(0.95)
            boot_train[COL_TARGET_W] = boot_train[COL_TARGET].clip(y_lo, y_hi)

            # Prepare features
            num_cols_w = [f + '_winsor' for f in NUMERIC_FEATURES
                          if f + '_winsor' in boot_train.columns]

            boot_train_v = boot_train.dropna(subset=num_cols_w + [COL_TARGET_W])
            if len(boot_train_v) < 50:
                continue

            # Fit model
            X_tr = StandardScaler().fit_transform(
                SimpleImputer(strategy='median').fit_transform(boot_train_v[num_cols_w].values))
            y_tr = boot_train_v[COL_TARGET_W].values

            if COL_INDUSTRY in boot_train_v.columns:
                ohe_b = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                X_tr = np.hstack([X_tr, ohe_b.fit_transform(
                    boot_train_v[[COL_INDUSTRY]].fillna('Unknown'))])

            model_b = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
            model_b.fit(X_tr, y_tr)

            # Predict
            for soe_data, gap_list in [(boot_soe500, boot_gaps_500),
                                        (boot_soe1000, boot_gaps_1000)]:
                soe_v = soe_data.dropna(subset=num_cols_w)
                if len(soe_v) == 0:
                    continue
                X_soe = StandardScaler().fit_transform(
                    SimpleImputer(strategy='median').fit_transform(soe_v[num_cols_w].values))
                if COL_INDUSTRY in boot_train_v.columns:
                    X_soe = np.hstack([X_soe, ohe_b.transform(
                        soe_v[[COL_INDUSTRY]].fillna('Unknown'))])
                preds = model_b.predict(X_soe)
                gap_list.append(np.mean(preds - soe_v[COL_TARGET].values))

            # Delta
            if len(boot_gaps_500) > len(boot_deltas) and len(boot_gaps_1000) > len(boot_deltas):
                boot_deltas.append(boot_gaps_500[-1] - boot_gaps_1000[-1])

        except Exception as e:
            continue

    # ---- Compute bootstrap statistics ----
    print(f"\n  --- Bootstrap 结果 ---")
    boot_results = {}

    for label, boot_gaps in [('CSI500', boot_gaps_500), ('CSI1000', boot_gaps_1000)]:
        bg = np.array(boot_gaps)
        bg = bg[~np.isnan(bg)]

        point_est = np.mean(bg)
        boot_se = np.std(bg, ddof=1)
        perc_ci = np.quantile(bg, [0.025, 0.975])

        # BCa CI
        try:
            from scipy.stats import norm
            # Bias correction
            z0 = norm.ppf(np.mean(bg < point_est))
            # Acceleration (jackknife approximation)
            n_boot = len(bg)
            # Simplified BCa
            bca_ci = perc_ci  # Fallback
            alpha_vals = [0.025, 0.975]
            bca_ci = np.array([
                np.quantile(bg, norm.cdf(z0 + (z0 + norm.ppf(a)) / (1 - 0.1 * (z0 + norm.ppf(a)))))
                if not np.isnan(z0) else perc_ci[i]
                for i, a in enumerate(alpha_vals)
            ])
        except:
            bca_ci = perc_ci

        # p-value (two-sided: is 0 in CI?)
        p_val = np.mean(np.abs(bg) >= np.abs(point_est)) if point_est != 0 else 1.0

        print(f"\n  [{label}]")
        print(f"  Point estimate: {point_est:+.4f}")
        print(f"  Bootstrap SE: {boot_se:.4f}")
        print(f"  95% percentile CI: [{perc_ci[0]:+.4f}, {perc_ci[1]:+.4f}]")
        print(f"  95% BCa CI: [{bca_ci[0]:+.4f}, {bca_ci[1]:+.4f}]")
        print(f"  p-value (two-sided): {p_val:.4f}")
        print(f"  Contains zero: {'YES' if perc_ci[0] <= 0 <= perc_ci[1] else 'NO'}")

        boot_results[label] = {
            'point_estimate': point_est,
            'boot_se': boot_se,
            'ci_95_lower': perc_ci[0],
            'ci_95_upper': perc_ci[1],
            'bca_ci_lower': bca_ci[0],
            'bca_ci_upper': bca_ci[1],
            'p_value': p_val,
        }

    # Delta
    if len(boot_deltas) > 0:
        bd = np.array(boot_deltas)
        bd = bd[~np.isnan(bd)]
        delta_est = np.mean(bd)
        delta_ci = np.quantile(bd, [0.025, 0.975])
        delta_p = np.mean(np.abs(bd) >= np.abs(delta_est)) if delta_est != 0 else 1.0
        print(f"\n  [Delta: CSI500 - CSI1000]")
        print(f"  Estimate: {delta_est:+.4f}")
        print(f"  95% CI: [{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}]")
        print(f"  p-value: {delta_p:.4f}")
        boot_results['delta_500_minus_1000'] = {
            'point_estimate': delta_est,
            'ci_95_lower': delta_ci[0],
            'ci_95_upper': delta_ci[1],
            'p_value': delta_p,
        }

    # Save
    pd.DataFrame({
        'CSI500_gap': boot_gaps_500 + [np.nan] * (len(boot_gaps_1000) - len(boot_gaps_500)),
        'CSI1000_gap': boot_gaps_1000,
    }).to_csv(OUTPUT_DIR / 'bootstrap_distributions.csv', index=False, encoding='utf-8-sig')

    boot_summary = pd.DataFrame(boot_results).T
    boot_summary.to_csv(OUTPUT_DIR / 'bootstrap_summary.csv', encoding='utf-8-sig')

    return boot_results


# ============================================================================
# Final Summary
# ============================================================================

def generate_final_summary(oof_results, csi500_preds, csi1000_preds,
                           placebo_results, bootstrap_results):
    """Generate final summary table."""

    print(f"\n  --- 最终汇总表 ---")

    rows = []
    for label, soe_data, real_gap in [
        ('CSI500 SOE', csi500_preds, csi500_preds['gap'].mean()),
        ('CSI1000 SOE', csi1000_preds, csi1000_preds['gap'].mean()),
    ]:
        pl = placebo_results.get(label.split()[0], {})
        bt = bootstrap_results.get(label.split()[0], {})

        rows.append({
            'Gap definition/group': label,
            'Point estimate': f"{real_gap:+.4f}",
            'OOF target-like bias': f"{oof_results['residual'].mean():+.4f}",
            'Placebo mean': f"{pl.get('placebo_mean', np.nan):+.4f}",
            'Bias-corrected gap': f"{pl.get('bias_corrected_gap', np.nan):+.4f}",
            'Bootstrap SE': f"{bt.get('boot_se', np.nan):.4f}",
            '95% CI': f"[{bt.get('ci_95_lower', np.nan):+.4f}, {bt.get('ci_95_upper', np.nan):+.4f}]",
            'Empirical placebo p': f"{pl.get('p_two_sided', np.nan):.4f}",
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / 'final_summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))

    # Overall conclusions
    print(f"\n  --- 总体判断 ---")
    csi500_gap = csi500_preds['gap'].mean()
    csi1000_gap = csi1000_preds['gap'].mean()

    bt500 = bootstrap_results.get('CSI500', {})
    bt1000 = bootstrap_results.get('CSI1000', {})

    ci500_low = bt500.get('ci_95_lower', np.nan)
    ci500_high = bt500.get('ci_95_upper', np.nan)
    ci1000_low = bt1000.get('ci_95_lower', np.nan)
    ci1000_high = bt1000.get('ci_95_upper', np.nan)

    print(f"  1. CSI500 gap 95% CI 是否包含0: "
          f"{'YES (不显著)' if (ci500_low <= 0 <= ci500_high) else 'NO (显著)'}")
    print(f"  2. CSI1000 gap 95% CI 是否包含0: "
          f"{'YES (不显著)' if (ci1000_low <= 0 <= ci1000_high) else 'NO (显著)'}")
    print(f"  3. OOF bias = {oof_results['residual'].mean():+.4f}")
    print(f"  4. Placebo mean CSI500 = {placebo_results.get('CSI500', {}).get('placebo_mean', np.nan):+.4f}")
    print(f"  5. Placebo mean CSI1000 = {placebo_results.get('CSI1000', {}).get('placebo_mean', np.nan):+.4f}")


if __name__ == '__main__':
    results = run_full_diagnostics()
    print("\n" + "=" * 70)
    print("  四项诊断完成")
    print("=" * 70)
