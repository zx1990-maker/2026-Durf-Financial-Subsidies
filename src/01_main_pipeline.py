#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Analysis — A-share Panel as Sole Data Source
=====================================================
All financial data from A-share panel (2014-2024). Index constituents from
input_index_weight files used ONLY for stock code identification.

Outputs:
  unified_2025.csv         — 2025 single-year: 3 models × 4 targets
  unified_rolling.csv      — rolling 2019-2025: 3 models × 4 targets × 7 years
  unified_master.md        — publication-ready tables
"""

import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "revision"
INDEX_DIR = PROJECT_DIR / "input_index_weight"

sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import (
    load_all_a_shares, clean_and_filter, backfill_ownership,
    NUMERIC_FEATURES, SOE_TYPES, NON_SOE_TYPE,
    EXCLUDE_INDUSTRIES_L2, WINSOR_BOUNDS, RIDGE_ALPHA, RANDOM_STATE,
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================================
# 1. Load index constituent CODES only (no financial data)
# ============================================================================

def load_constituent_codes():
    """Load current constituent codes from 成分及权重 files (codes only)."""
    fname_map = {
        'CSI300': '沪深300-成分及权重-20260811.xlsx',
        'CSI500': '中证500-成分及权重-20260811.xlsx',
        'CSI1000': '中证1000-成分及权重-20260811.xlsx',
    }
    current_codes = {}
    for idx, fname in fname_map.items():
        fpath = INDEX_DIR / fname
        wb = openpyxl.load_workbook(fpath, read_only=True, data_only=True)
        ws = wb.active
        codes = set()
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0]:
                codes.add(str(row[0]).strip())
        wb.close()
        current_codes[idx] = codes
        print(f"  {idx}: {len(codes)} current constituents")
    return current_codes


def load_entry_exit():
    """Load entry/exit records for all three indices."""
    fname_map = {
        'CSI300': '沪深300-成分进出记录-20260811.xlsx',
        'CSI500': '中证500-成分进出记录-20260811.xlsx',
        'CSI1000': '中证1000-成分进出记录-20260811.xlsx',
    }
    all_records = {}
    for idx, fname in fname_map.items():
        fpath = INDEX_DIR / fname
        wb = openpyxl.load_workbook(fpath, read_only=True, data_only=True)
        ws = wb.active
        records = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] and row[1] and row[7]:
                records.append({
                    'date': str(row[0])[:10],
                    'code': str(row[1]).strip(),
                    'action': str(row[7]).strip(),
                })
        wb.close()
        records.sort(key=lambda r: r['date'])
        all_records[idx] = records
        print(f"  {idx}: {len(records)} entry/exit records")
    return all_records


def build_yearly_constituents(current_codes, all_records):
    """Build constituent code sets for each year (2019-2025) using backward
    reconstruction from current list. Returns dict: index -> year -> set of codes."""
    yearly = {}
    for idx in ['CSI300', 'CSI500', 'CSI1000']:
        current = current_codes[idx]
        records = all_records[idx]
        rev_records = sorted(records, key=lambda r: r['date'], reverse=True)
        yearly[idx] = {}
        for year in range(2019, 2026):
            target_date = f"{year}-07-01"
            codes = set(current)
            for rec in rev_records:
                if rec['date'] <= target_date:
                    break
                if rec['action'] == '纳入':
                    codes.discard(rec['code'])
                elif rec['action'] == '剔除':
                    codes.add(rec['code'])
            yearly[idx][year] = codes
        print(f"  {idx}: years 2019-2025, n constituents: "
              f"{', '.join(str(len(yearly[idx][y])) for y in range(2019,2026))}")
    return yearly


# ============================================================================
# 2. Model fitting (single interface)
# ============================================================================

def build_model(model_type):
    if model_type == 'ridge':
        return Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    elif model_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)
    elif model_type == 'gbm':
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, random_state=RANDOM_STATE)


def cluster_se(gaps, firm_ids):
    u = np.unique(firm_ids)
    fm = np.array([gaps[firm_ids == f].mean() for f in u])
    se = np.std(fm, ddof=1) / np.sqrt(len(u)) if len(u) > 1 else np.nan
    return se, len(u)


def fit_and_predict(train_df, pred_df, model_type, num_cols):
    """Train on train_df, predict on pred_df. All data from A-share panel."""
    tv = train_df.dropna(subset=num_cols + ['EBI_A']).copy()
    if len(tv) < 50:
        return None

    # Winsorize based on training
    for c in num_cols:
        lo, hi = tv[c].quantile(0.05), tv[c].quantile(0.95)
        tv[c + '_w'] = tv[c].clip(lo, hi)
    yl, yh = tv['EBI_A'].quantile(0.05), tv['EBI_A'].quantile(0.95)
    tv['EBI_A_w'] = tv['EBI_A'].clip(yl, yh)

    ncw = [c + '_w' for c in num_cols]
    Xt = tv[ncw].values; yt = tv['EBI_A_w'].values; grp = tv['firm_id'].values

    imp = SimpleImputer(strategy='median'); scl = StandardScaler()
    Xt = scl.fit_transform(imp.fit_transform(Xt))

    ohe = None
    if 'industry_l2' in tv.columns:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        Xt = np.hstack([Xt, ohe.fit_transform(tv[['industry_l2']].fillna('Unknown'))])

    nf = tv['firm_id'].nunique(); ns = min(5, nf)
    if ns >= 2:
        gkf = GroupKFold(n_splits=ns)
        oof_p = cross_val_predict(build_model(model_type), Xt, yt,
                                  cv=gkf.split(Xt, yt, groups=grp),
                                  n_jobs=-1 if model_type == 'random_forest' else 1)
        oof_r2 = r2_score(yt, oof_p)
        oof_rmse = np.sqrt(mean_squared_error(yt, oof_p))
        oof_mae = mean_absolute_error(yt, oof_p)
    else:
        oof_r2 = oof_rmse = oof_mae = np.nan

    m = build_model(model_type); m.fit(Xt, yt)
    tr_rmse = np.sqrt(mean_squared_error(yt, m.predict(Xt)))

    # Predict
    pv = pred_df.copy()
    for c in num_cols:
        if c in pv.columns:
            lo, hi = tv[c].quantile(0.05), tv[c].quantile(0.95)
            pv[c + '_w'] = pv[c].clip(lo, hi)
    pd2 = pv.dropna(subset=ncw)
    if len(pd2) == 0:
        return None

    Xp = scl.transform(imp.transform(pd2[ncw].values))
    if ohe is not None:
        Xp = np.hstack([Xp, ohe.transform(pd2[['industry_l2']].fillna('Unknown'))])

    preds = m.predict(Xp); actuals = pd2['EBI_A'].values; gaps = preds - actuals
    se, nc = cluster_se(gaps, pd2['firm_id'].values)
    mg = np.mean(gaps)

    return {
        'n_train_obs': len(tv), 'n_train_firms': nf,
        'n_pred': len(pd2), 'n_pred_firms': nc,
        'oof_r2': oof_r2, 'oof_rmse': oof_rmse, 'oof_mae': oof_mae,
        'train_rmse': tr_rmse,
        'pred_mean': np.mean(preds), 'actual_mean': np.mean(actuals),
        'mean_gap': mg, 'clustered_se': se,
        'ci_95_lower': mg - 1.96*se if not np.isnan(se) else np.nan,
        'ci_95_upper': mg + 1.96*se if not np.isnan(se) else np.nan,
    }


# ============================================================================
# 3. Main
# ============================================================================

def main():
    print("=" * 60)
    print("  Unified Analysis — A-share Panel Only")
    print("=" * 60)

    # --- Load A-share panel (ONCE) ---
    print("\n[1] Loading A-share panel...")
    all_a = load_all_a_shares()
    all_a = clean_and_filter(all_a, "全A股")
    all_a['is_soe'] = all_a['ownership'].isin(SOE_TYPES)
    all_a['is_private'] = all_a['ownership'] == NON_SOE_TYPE
    print(f"  Panel: {len(all_a)} obs, {all_a['firm_id'].nunique()} firms")
    print(f"  Target years: {sorted(all_a['target_year'].unique())}")

    # --- Load index constituent codes ---
    print("\n[2] Loading index constituent codes...")
    current_codes = load_constituent_codes()
    all_records = load_entry_exit()
    yearly_codes = build_yearly_constituents(current_codes, all_records)

    num_cols = [f for f in NUMERIC_FEATURES if f in all_a.columns]
    models = ['ridge', 'random_forest', 'gbm']
    target_years = sorted(all_a['target_year'].unique())
    forecast_years = [y for y in target_years if y >= 2019]

    results_2025 = []
    results_rolling = []

    # ================================================================
    # PART A: Rolling window (2019-2025)
    # ================================================================
    print("\n[3] Expanding-window rolling analysis (2019-2025)...")

    for fc_year in forecast_years:
        train_data = all_a[(all_a['is_private']) & (all_a['target_year'] < fc_year)].copy()  # 严格 ex-ante：结果年份 < 预测年
        train_yrs = sorted(train_data['target_year'].unique())

        # Build targets from A-share panel only
        soe_all = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc_year)].copy()
        targets = {'All A-share SOE': soe_all}

        for idx in ['CSI300', 'CSI500', 'CSI1000']:
            codes = yearly_codes[idx].get(fc_year, set())
            targets[f'{idx} SOE'] = soe_all[soe_all['firm_id'].isin(codes)].copy()

        for mt in models:
            for tn, pd_df in targets.items():
                if len(pd_df) < 3:
                    continue
                r = fit_and_predict(train_data, pd_df, mt, num_cols)
                if r is None:
                    continue
                results_rolling.append({
                    'forecast_year': fc_year, 'model': mt, 'target': tn,
                    'training_years': f"{train_yrs[0]}–{train_yrs[-1]}", **r,
                })

        n_all = len(soe_all)
        n300 = len(targets.get('CSI300 SOE', pd.DataFrame()))
        n500 = len(targets.get('CSI500 SOE', pd.DataFrame()))
        n1000 = len(targets.get('CSI1000 SOE', pd.DataFrame()))
        print(f"  {fc_year}: train={train_yrs[0]}–{train_yrs[-1]} "
              f"({len(train_data)} obs), SOE: All={n_all}, 300={n300}, 500={n500}, 1000={n1000}")

    df_rolling = pd.DataFrame(results_rolling)
    df_rolling.to_csv(OUTPUT_DIR / 'unified_rolling.csv', index=False, encoding='utf-8-sig')
    print(f"  ✓ unified_rolling.csv ({len(df_rolling)} rows)")

    # ================================================================
    # Produce tables
    # ================================================================
    lines = []
    def w(s): lines.append(s)

    w("# Unified Analysis — A-share Panel as Sole Data Source")
    w("")
    w("All financial data from A-share panel (A_2014–A_2024_origin.xlsx).")
    w("Index constituent codes from input_index_weight files used only for identification.")
    w("")

    # All SOE rolling table
    w("## Table 1: All A-share SOE — Rolling Gap by Model")
    w("")
    w("| Year | Model | N Train | N SOE | Gap | SE | 95% CI | OOS R² |")
    w("|------|-------|---------|-------|-----|----|--------|--------|")
    sub = df_rolling[df_rolling['target'] == 'All A-share SOE'].sort_values(['forecast_year','model'])
    for _, r in sub.iterrows():
        ci = f"[{r['ci_95_lower']:+.4f}, {r['ci_95_upper']:+.4f}]"
        w(f"| {int(r['forecast_year'])} | {r['model']} | {int(r['n_train_obs']):,} | {int(r['n_pred'])} | **{r['mean_gap']:+.4f}** | {r['clustered_se']:.4f} | {ci} | {r['oof_r2']:+.4f} |")
    w("")

    # Per-index tables
    for idx in ['CSI300', 'CSI500', 'CSI1000']:
        target_name = f'{idx} SOE'
        sub_idx = df_rolling[df_rolling['target'] == target_name].sort_values(['forecast_year','model'])
        w(f"## Table: {idx} SOE — Rolling Gap by Model")
        w("")
        w("| Year | Model | N | Gap | SE | 95% CI |")
        w("|------|-------|---|-----|----|--------|")
        for _, r in sub_idx.iterrows():
            ci = f"[{r['ci_95_lower']:+.4f}, {r['ci_95_upper']:+.4f}]"
            w(f"| {int(r['forecast_year'])} | {r['model']} | {int(r['n_pred'])} | **{r['mean_gap']:+.4f}** | {r['clustered_se']:.4f} | {ci} |")
        w("")

    # Ridge summary
    w("## Ridge Summary: Gap by Target × Year")
    w("")
    w("| Year | All SOE | CSI300 | CSI500 | CSI1000 |")
    w("|------|---------|--------|--------|---------|")
    ridge = df_rolling[df_rolling['model'] == 'ridge']
    for year in forecast_years:
        row = f"| {year} |"
        for tn in ['All A-share SOE', 'CSI300 SOE', 'CSI500 SOE', 'CSI1000 SOE']:
            r = ridge[(ridge['forecast_year'] == year) & (ridge['target'] == tn)]
            if len(r) > 0:
                row += f" **{r.iloc[0]['mean_gap']:+.4f}** ({int(r.iloc[0]['n_pred'])}) |"
            else:
                row += " — |"
        w(row)
    w("")

    with open(OUTPUT_DIR / 'unified_master.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print("  ✓ unified_master.md written")

    # ================================================================
    # Print key tables to stdout
    # ================================================================
    print("\n" + "=" * 80)
    print("  RIDGE SUMMARY: Gap by Target × Year")
    print("=" * 80)
    print(f"{'Year':<6} {'All SOE':>14} {'CSI300':>14} {'CSI500':>14} {'CSI1000':>14}")
    print("-" * 62)
    for year in forecast_years:
        row = f"{year:<6}"
        for tn in ['All A-share SOE', 'CSI300 SOE', 'CSI500 SOE', 'CSI1000 SOE']:
            r = ridge[(ridge['forecast_year'] == year) & (ridge['target'] == tn)]
            if len(r) > 0:
                row += f" {r.iloc[0]['mean_gap']:>+8.4f} ({int(r.iloc[0]['n_pred']):>3d})"
            else:
                row += f" {'—':>12}"
        print(row)

    print("\n✓ Unified analysis complete")


if __name__ == '__main__':
    main()
