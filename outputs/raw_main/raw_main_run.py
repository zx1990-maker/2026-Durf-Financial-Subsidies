#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raw_main_run — Task C: regenerate all paper empirical results with RAW_RAW as the
unified main specification.

Unified outcome definition:
    target      = RAW private EBI/A        (no outcome winsorization)
    SOE actual  = RAW SOE EBI/A
    gap         = pred_raw - raw           (GBM / Ridge / RF all on raw target)

All other settings are FROZEN (identical to the paper's final code): SOE/private
definitions, sample screening, the 9 X features, P5/P95 FEATURE-winsorization,
median imputation, StandardScaler, one-hot industry_l2, strict time-OOS expanding
window (target_year < ForecastYear), ForecastYear 2019-2025, GBM/Ridge/RF
hyperparameters, seed, CSI300/500/1000 membership, and firm-cluster inference.

IMPORTANT: this module does NOT recompute common-support distance/status. It merges
the RAW_RAW firm-year gap onto the already-confirmed v2 support classification
(outputs/support_v2/soe_support_firmyear_v2.csv, 5-NN vs 5-NN distinct-firm).

Writes everything to outputs/raw_main/ (new directory; nothing overwritten).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'revision2'))
sys.path.insert(0, str(BASE.parent / 'revision3'))

from _audit_common import (  # noqa: E402
    load_screened_panel, load_yearly_codes, feats_in, build, prepare_train,
    cluster_se, cluster_t_test,
)
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

OUT = BASE
OUT.mkdir(exist_ok=True)

YEARS = list(range(2019, 2026))
SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']
SAMPLE_COL = {'CSI300': 'in_csi300', 'CSI500': 'in_csi500', 'CSI1000': 'in_csi1000'}
INDEXES = ['CSI300', 'CSI500', 'CSI1000']
MODELS = ['GBM', 'Ridge', 'RF']
MODEL_TYPE = {'GBM': 'gbm', 'Ridge': 'ridge', 'RF': 'random_forest'}
RNG_SEED = 42
N_BOOT = 2000
V2_SUPPORT = BASE.parent / 'support_v2' / 'soe_support_firmyear_v2.csv'


# ============================================================================
# load
# ============================================================================

all_a = load_screened_panel()
feats = feats_in(all_a)
assert list(feats) == ['FirmSize', 'AssetTurnover', 'FinancialLeverage',
                       'FixedAssetsRatio', 'CurrentRatio', 'CapexRatio',
                       'WorkingCapitalRatio', 'MarketShare', 'HHI'], feats
train_pool = all_a[all_a['is_private']].copy()
yearly_codes = load_yearly_codes()


# ============================================================================
# OOF helper (production GroupKFold(5)-by-firm, raw target)
# ============================================================================

def oof_metrics(prep, y, model_type):
    X = prep['Xt']
    groups = prep['tv']['firm_id'].values
    n_firms = len(np.unique(groups))
    n_splits = min(5, n_firms)
    if n_splits < 2:
        return None
    model_cv = build(model_type)
    cv_preds = cross_val_predict(Pipeline([('model', model_cv)]), X, y,
                                 cv=GroupKFold(n_splits).split(X, y, groups=groups))
    resid = cv_preds - y
    lr = LinearRegression().fit(cv_preds.reshape(-1, 1), y)
    return {
        'OOF_N': int(len(y)),
        'R2': float(r2_score(y, cv_preds)),
        'RMSE': float(np.sqrt(mean_squared_error(y, cv_preds))),
        'MAE': float(mean_absolute_error(y, cv_preds)),
        'Mean_residual': float(resid.mean()),
        'Calibration_intercept': float(lr.intercept_),
        'Calibration_slope': float(lr.coef_[0]),
        'cv_preds': cv_preds,
        'y': y,
    }


# ============================================================================
# build RAW master firm-year panel (GBM / Ridge / RF on raw target)
# ============================================================================

CACHE = OUT / '_master_raw_panel.csv'

if CACHE.exists():
    raw = pd.read_csv(CACHE, dtype={'firm_id': str})
    print(f'Loaded cached master RAW panel: {len(raw)} rows')
else:
    parts = []
    oof_pool = {m: {'preds': [], 'y': []} for m in MODELS}
    oof_by_year = []

    for fc in YEARS:
        tr = train_pool[train_pool['target_year'] < fc]
        assert tr['target_year'].max() < fc, f'ex-ante violation at {fc}'
        prep = prepare_train(tr, feats)
        yt_raw = prep['tv']['EBI_A'].values   # RAW target (no outcome winsorization)

        soe = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc)].copy()

        # ---- preprocess SOE features EXACTLY as predict_gap does ----
        pv = soe.copy()
        for c in feats:
            pv[c + '_w'] = pv[c].clip(prep['qlo'][c], prep['qhi'][c])
        pd2 = pv.dropna(subset=prep['ncw']).copy()
        X_soe = np.hstack([
            prep['scl'].transform(prep['imp'].transform(pd2[prep['ncw']].values)),
            prep['ohe'].transform(pd2[['industry_l2']].fillna('Unknown')),
        ])
        raw_actual = pd2['EBI_A'].values

        # ---- index membership ----
        for idx in INDEXES:
            codes = yearly_codes[idx][fc]
            pd2[f'in_{idx.lower()}'] = pd2['firm_id'].apply(lambda f: f in codes)

        part = {
            'firm_id': pd2['firm_id'].astype(str).values,
            'ForecastYear': fc,
            'firm_name': pd2['firm_name'].values,
            'ownership': pd2['ownership'].values,
            'industry_l2': pd2['industry_l2'].values,
            'EBI_A': raw_actual,
            'total_assets': pd2['资产总计_target'].values,
            'in_csi300': pd2['in_csi300'].values,
            'in_csi500': pd2['in_csi500'].values,
            'in_csi1000': pd2['in_csi1000'].values,
        }

        # ---- fit each model on RAW target, predict RAW gap ----
        for model_name in MODELS:
            m = build(MODEL_TYPE[model_name])
            m.fit(prep['Xt'], yt_raw)
            preds = m.predict(X_soe)
            part[f'gap_{model_name.lower()}'] = preds - raw_actual

        parts.append(pd.DataFrame(part))

        # ---- rolling OOF (raw target) per model ----
        for model_name in MODELS:
            om = oof_metrics(prep, yt_raw, MODEL_TYPE[model_name])
            if om is not None:
                oof_by_year.append({'Model': model_name, 'ForecastYear': fc,
                                    'OOF_N': om['OOF_N'], 'R2': om['R2'],
                                    'RMSE': om['RMSE'], 'MAE': om['MAE'],
                                    'Mean_residual': om['Mean_residual'],
                                    'Calibration_intercept': om['Calibration_intercept'],
                                    'Calibration_slope': om['Calibration_slope']})
                oof_pool[model_name]['preds'].append(om['cv_preds'])
                oof_pool[model_name]['y'].append(om['y'])
        print(f'  {fc} done')

    raw = pd.concat(parts, ignore_index=True)
    raw['amount_gap_gbm'] = raw['gap_gbm'] * raw['total_assets'] / 1e8
    raw.to_csv(CACHE, index=False, encoding='utf-8-sig')

    # ---- cache OOF results for the report script ----
    oof_by_year_df = pd.DataFrame(oof_by_year)
    oof_by_year_df.to_csv(OUT / '_oof_by_year_raw_cache.csv', index=False, encoding='utf-8-sig')
    np.savez(OUT / '_oof_pool_raw_cache.npz',
             **{f'{m}_preds': np.concatenate(oof_pool[m]['preds']) for m in MODELS},
             **{f'{m}_y': np.concatenate(oof_pool[m]['y']) for m in MODELS})

# ---- merge v2 support classification (no recompute) ----
v2 = pd.read_csv(V2_SUPPORT, dtype={'firm_id': str})
raw = raw.merge(v2[['firm_id', 'ForecastYear', 'support_status']],
                on=['firm_id', 'ForecastYear'], how='left', validate='one_to_one')
assert raw['support_status'].notna().all(), 'v2 support merge incomplete'

# central / local flags
raw['central'] = (raw['ownership'] == '中央国有企业').values
raw['local'] = (raw['ownership'] == '地方国有企业').values


# ============================================================================
# helpers
# ============================================================================

def annual_mean(gap, year, mask=None):
    vals = []
    for y in YEARS:
        mm = (year == y)
        if mask is not None:
            mm = mm & mask
        if mm.any():
            vals.append(gap[mm].mean())
    return float(np.mean(vals)) if vals else np.nan


def cluster_bootstrap_full(gap, year, fid, seed=RNG_SEED, n_boot=N_BOOT):
    """Firm-cluster bootstrap of the annual-mean gap (Full metric)."""
    order = np.argsort(fid, kind='stable')
    fid_s = fid[order]; gap_s = gap[order]; year_s = year[order]
    uniq, starts = np.unique(fid_s, return_index=True)
    counts = np.diff(np.append(starts, len(fid_s))).astype(int)
    n_firms = len(uniq)
    pt = annual_mean(gap_s, year_s)
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        codes = rng.integers(0, n_firms, size=n_firms)
        lens = counts[codes]
        block_start = np.repeat(np.cumsum(lens) - lens, lens)
        within = np.arange(lens.sum()) - block_start
        gidx = np.repeat(starts[codes], lens) + within
        boot[b] = annual_mean(gap_s[gidx], year_s[gidx])
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    return pt, lo, hi, n_firms, int(len(gap_s))


def cluster_bootstrap_support(sub, gapcol, seed=RNG_SEED, n_boot=N_BOOT):
    """Full / CS / PersistentGood cluster bootstrap (v2 support)."""
    sub = sub.sort_values('firm_id').reset_index(drop=True)
    fid = sub['firm_id'].values
    gap = sub[gapcol].values
    year = sub['ForecastYear'].values
    cs_flag = sub['support_status'].isin(['Good', 'Boundary']).values

    uniq, starts = np.unique(fid, return_index=True)
    counts = np.diff(np.append(starts, len(fid))).astype(int)
    n_firms = len(uniq)

    goodshare = np.array([
        (sub['support_status'].values[starts[i]:starts[i] + counts[i]] == 'Good').mean()
        for i in range(n_firms)])
    csshare = np.array([
        np.isin(sub['support_status'].values[starts[i]:starts[i] + counts[i]],
                ['Good', 'Boundary']).mean()
        for i in range(n_firms)])
    firm_mean_gap = np.array([gap[starts[i]:starts[i] + counts[i]].mean()
                              for i in range(n_firms)])

    pt_full = annual_mean(gap, year)
    pt_cs = annual_mean(gap, year, cs_flag)
    sel_pers = goodshare >= 0.8
    pt_pers = float(firm_mean_gap[sel_pers].mean()) if sel_pers.any() else np.nan

    rng = np.random.default_rng(seed)
    boot_full = np.full(n_boot, np.nan)
    boot_cs = np.full(n_boot, np.nan)
    boot_pers = np.full(n_boot, np.nan)
    for b in range(n_boot):
        codes = rng.integers(0, n_firms, size=n_firms)
        lens = counts[codes]
        block_start = np.repeat(np.cumsum(lens) - lens, lens)
        within = np.arange(lens.sum()) - block_start
        gidx = np.repeat(starts[codes], lens) + within
        g_b = gap[gidx]; y_b = year[gidx]; c_b = cs_flag[gidx]
        boot_full[b] = annual_mean(g_b, y_b)
        boot_cs[b] = annual_mean(g_b, y_b, c_b)
        sel = codes[goodshare[codes] >= 0.8]
        boot_pers[b] = firm_mean_gap[sel].mean() if len(sel) else np.nan

    def summ(b):
        b = b[~np.isnan(b)]
        se = float(np.std(b, ddof=1)) if len(b) > 1 else np.nan
        lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan, np.nan)
        p = min(1.0, 2.0 * min(float(np.mean(b <= 0)), float(np.mean(b >= 0))))
        return se, lo, hi, p

    s_full = summ(boot_full); s_cs = summ(boot_cs); s_pers = summ(boot_pers)
    return {
        'Full_gap_pp': pt_full * 100, 'Full_SE': s_full[0] * 100,
        'Full_CI_low': s_full[1] * 100, 'Full_CI_high': s_full[2] * 100, 'Full_p': s_full[3],
        'CS_gap_pp': pt_cs * 100, 'CS_SE': s_cs[0] * 100,
        'CS_CI_low': s_cs[1] * 100, 'CS_CI_high': s_cs[2] * 100, 'CS_p': s_cs[3],
        'PersistentGood_gap_pp': pt_pers * 100 if not np.isnan(pt_pers) else np.nan,
        'PersistentGood_SE': s_pers[0] * 100,
        'PersistentGood_CI_low': s_pers[1] * 100, 'PersistentGood_CI_high': s_pers[2] * 100,
        'PersistentGood_p': s_pers[3],
        'Mean_GoodShare': float(np.mean(goodshare)),
        'Mean_CSShare': float(np.mean(csshare)),
        'N_firms': n_firms,
        'N_PersistentGood_firms': int(sel_pers.sum()),
    }


def sample_subset(df, sample):
    if sample == 'All A-share SOE':
        return df
    return df[df[SAMPLE_COL[sample]].astype(bool)]


# ============================================================================
# D1/D2: model performance (rolling OOF, raw target) — cached from build loop
# ============================================================================

def pooled_metrics(preds_list, y_list):
    preds = np.concatenate(preds_list)
    y = np.concatenate(y_list)
    resid = preds - y
    lr = LinearRegression().fit(preds.reshape(-1, 1), y)
    return {
        'OOF_N': int(len(y)),
        'R2': float(r2_score(y, preds)),
        'RMSE': float(np.sqrt(mean_squared_error(y, preds))),
        'MAE': float(mean_absolute_error(y, preds)),
        'Mean_residual': float(resid.mean()),
        'Calibration_intercept': float(lr.intercept_),
        'Calibration_slope': float(lr.coef_[0]),
    }


oof_by_year_df = pd.read_csv(OUT / '_oof_by_year_raw_cache.csv')
pool = np.load(OUT / '_oof_pool_raw_cache.npz', allow_pickle=True)
perf_rows = []
for m in MODELS:
    pm = pooled_metrics([pool[f'{m}_preds']], [pool[f'{m}_y']])
    perf_rows.append({'Model': m, 'ForecastYear': 'Pooled', **pm})
perf = pd.DataFrame(perf_rows)
perf = perf[['Model', 'ForecastYear', 'OOF_N', 'R2', 'RMSE', 'MAE',
             'Mean_residual', 'Calibration_intercept', 'Calibration_slope']]
perf.to_csv(OUT / 'model_performance_raw.csv', index=False, encoding='utf-8-sig')

by_year = oof_by_year_df[['Model', 'ForecastYear', 'OOF_N', 'R2', 'RMSE', 'MAE',
                          'Mean_residual', 'Calibration_intercept', 'Calibration_slope']]
by_year.to_csv(OUT / 'model_performance_raw_by_year.csv', index=False, encoding='utf-8-sig')
print('\n=== model_performance_raw (pooled OOF, raw target) ===')
print(perf.round(4).to_string(index=False))


# ============================================================================
# D3: main_gap_by_year_raw (Year x Sample x Model)
# ============================================================================

gap_by_year_rows = []
for model_name in MODELS:
    gcol = f'gap_{model_name.lower()}'
    for fc in YEARS:
        for sample in SAMPLES:
            sub = sample_subset(raw, sample)
            sub = sub[sub['ForecastYear'] == fc]
            if len(sub) == 0:
                continue
            g = sub[gcol].values
            fid = sub['firm_id'].values
            se, n_f = cluster_se(g, fid)
            m = float(g.mean())
            gap_by_year_rows.append({
                'Model': model_name, 'ForecastYear': fc, 'Sample': sample,
                'N': int(len(sub)),
                'Mean_gap_pp': m * 100,
                'Firm_clustered_SE_pp': se * 100 if not np.isnan(se) else np.nan,
                'CI95_low_pp': (m - 1.96 * se) * 100 if not np.isnan(se) else np.nan,
                'CI95_high_pp': (m + 1.96 * se) * 100 if not np.isnan(se) else np.nan,
            })
gap_by_year = pd.DataFrame(gap_by_year_rows)
gap_by_year.to_csv(OUT / 'main_gap_by_year_raw.csv', index=False, encoding='utf-8-sig')


# ============================================================================
# D4: main_gap_multiyear_raw (Sample x Model, cluster bootstrap Full)
# ============================================================================

multiyear_rows = []
for model_name in MODELS:
    gcol = f'gap_{model_name.lower()}'
    for sample in SAMPLES:
        sub = sample_subset(raw, sample)
        pt, lo, hi, n_firms, n_fy = cluster_bootstrap_full(
            sub[gcol].values, sub['ForecastYear'].values, sub['firm_id'].values)
        multiyear_rows.append({
            'Model': model_name, 'Sample': sample,
            'N_firmyears': n_fy, 'N_firms': n_firms,
            'Mean_gap_pp': pt * 100, 'CI95_low_pp': lo * 100, 'CI95_high_pp': hi * 100,
        })
multiyear = pd.DataFrame(multiyear_rows)
multiyear.to_csv(OUT / 'main_gap_multiyear_raw.csv', index=False, encoding='utf-8-sig')
print('\n=== main_gap_multiyear_raw (GBM All SOE should be ~+1.4497 pp) ===')
print(multiyear[multiyear['Model'] == 'GBM'].round(4).to_string(index=False))


# ============================================================================
# D6: support_year_sample_summary_raw (GBM RAW gap + v2 support)
# ============================================================================

summary_rows = []
for fc in YEARS:
    for sample in SAMPLES:
        s = sample_subset(raw, sample)
        s = s[s['ForecastYear'] == fc]
        n = len(s)
        g = s['support_status'] == 'Good'
        b = s['support_status'] == 'Boundary'
        cs = g | b
        summary_rows.append({
            'ForecastYear': fc, 'sample': sample, 'N_SOE': n,
            'Good_N': int(g.sum()), 'Boundary_N': int(b.sum()),
            'Extrapolation_N': int((~cs).sum()),
            'Good_share': g.mean(), 'Boundary_share': b.mean(),
            'Extrapolation_share': (~cs).mean(), 'CS_share': cs.mean(),
            'Full_gap_pp': s['gap_gbm'].mean() * 100,
            'Good_gap_pp': s.loc[g, 'gap_gbm'].mean() * 100 if g.sum() else np.nan,
            'CS_gap_pp': s.loc[cs, 'gap_gbm'].mean() * 100 if cs.sum() else np.nan,
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / 'support_year_sample_summary_raw.csv', index=False, encoding='utf-8-sig')


# ============================================================================
# D7: support_multiyear_summary_raw (GBM RAW gap + v2 support, cluster bootstrap)
# ============================================================================

multiyear_support_rows = []
for sample in SAMPLES:
    sub = sample_subset(raw, sample).copy()
    r = cluster_bootstrap_support(sub, 'gap_gbm')
    r['sample'] = sample
    multiyear_support_rows.append(r)
ms_df = pd.DataFrame(multiyear_support_rows)
ms_cols = ['sample', 'Full_gap_pp', 'Full_CI_low', 'Full_CI_high',
           'CS_gap_pp', 'CS_CI_low', 'CS_CI_high',
           'PersistentGood_gap_pp', 'PersistentGood_CI_low', 'PersistentGood_CI_high',
           'Mean_GoodShare', 'Mean_CSShare', 'N_firms', 'N_PersistentGood_firms']
ms_df = ms_df[ms_cols]
ms_df.to_csv(OUT / 'support_multiyear_summary_raw.csv', index=False, encoding='utf-8-sig')
print('\n=== support_multiyear_summary_raw (GBM) ===')
print(ms_df.round(4).to_string(index=False))


# ============================================================================
# D10: industry heterogeneity (GBM RAW gap)
# ============================================================================

# by year x sample x industry
ind_rows = []
for fc in YEARS:
    for sample in SAMPLES:
        s = sample_subset(raw, sample)
        s = s[s['ForecastYear'] == fc]
        for ind, g in s.groupby('industry_l2'):
            ind_rows.append({'ForecastYear': fc, 'Sample': sample,
                             'industry_l2': ind, 'N': len(g),
                             'Mean_gap_pp': g['gap_gbm'].mean() * 100})
ind_year = pd.DataFrame(ind_rows)
ind_year.to_csv(OUT / 'industry_gap_by_year_raw.csv', index=False, encoding='utf-8-sig')

# multiyear (industries with n>=20 total obs over 2019-2025, All SOE)
ind_multi_rows = []
for sample in SAMPLES:
    s = sample_subset(raw, sample)
    for ind, g in s.groupby('industry_l2'):
        if len(g) >= 20:
            ind_multi_rows.append({'Sample': sample, 'industry_l2': ind,
                                   'N_total': len(g),
                                   'Mean_gap_pp': g['gap_gbm'].mean() * 100})
ind_multi = pd.DataFrame(ind_multi_rows)
ind_multi.to_csv(OUT / 'industry_gap_multiyear_raw.csv', index=False, encoding='utf-8-sig')

# old (CURRENT gap) vs raw rank, All SOE
old_panel = pd.read_csv(BASE.parent / 'revision3' / '_master_long_panel.csv',
                        dtype={'firm_id': str})
old_all = old_panel[['industry_l2', 'gap']].copy()
raw_all = raw[['industry_l2', 'gap_gbm']].copy()
old_multi = old_all.groupby('industry_l2').agg(
    N_total=('gap', 'count'), old_gap_pp=('gap', 'mean')).reset_index()
raw_multi = raw_all.groupby('industry_l2').agg(
    N_total=('gap_gbm', 'count'), raw_gap_pp=('gap_gbm', 'mean')).reset_index()
old_multi = old_multi[old_multi['N_total'] >= 20]
raw_multi = raw_multi[raw_multi['N_total'] >= 20]
rank = old_multi.merge(raw_multi, on='industry_l2', suffixes=('_old', '_raw'))
rank['old_gap_pp'] = rank['old_gap_pp'] * 100
rank['raw_gap_pp'] = rank['raw_gap_pp'] * 100
rank = rank.sort_values('raw_gap_pp', ascending=False).reset_index(drop=True)
rank['old_rank'] = rank['old_gap_pp'].rank(ascending=False, method='first').astype(int)
rank['raw_rank'] = rank['raw_gap_pp'].rank(ascending=False, method='first').astype(int)
rank['rank_change'] = rank['old_rank'] - rank['raw_rank']
rank = rank[['industry_l2', 'N_total_old', 'N_total_raw', 'old_gap_pp', 'raw_gap_pp',
             'old_rank', 'raw_rank', 'rank_change']]
rank.to_csv(OUT / 'industry_old_vs_raw_rank.csv', index=False, encoding='utf-8-sig')


# ============================================================================
# D11: central / local (GBM RAW gap)
# ============================================================================

# per-year difference test
cl_rows = []
for fc in YEARS:
    s = raw[raw['ForecastYear'] == fc]
    cen = s[s['central']]
    loc = s[s['local']]
    c_m, c_se, c_lo, c_hi, c_nf = np.nan, np.nan, np.nan, np.nan, np.nan
    l_m, l_se, l_lo, l_hi, l_nf = np.nan, np.nan, np.nan, np.nan, np.nan
    if len(cen) >= 3:
        c_se, c_nf = cluster_se(cen['gap_gbm'].values, cen['firm_id'].values)
        c_m = float(cen['gap_gbm'].mean())
        c_lo = c_m - 1.96 * c_se if not np.isnan(c_se) else np.nan
        c_hi = c_m + 1.96 * c_se if not np.isnan(c_se) else np.nan
    if len(loc) >= 3:
        l_se, l_nf = cluster_se(loc['gap_gbm'].values, loc['firm_id'].values)
        l_m = float(loc['gap_gbm'].mean())
        l_lo = l_m - 1.96 * l_se if not np.isnan(l_se) else np.nan
        l_hi = l_m + 1.96 * l_se if not np.isnan(l_se) else np.nan
    diff, d_se, d_lo, d_hi, t, p = (np.nan,) * 6
    if len(cen) >= 3 and len(loc) >= 3:
        diff, d_se, d_lo, d_hi, t, p = cluster_t_test(
            cen['gap_gbm'].values, cen['firm_id'].values,
            loc['gap_gbm'].values, loc['firm_id'].values)
    cl_rows.append({
        'year': fc,
        'central_n': len(cen), 'central_gap': c_m, 'central_se': c_se,
        'central_ci_lower': c_lo, 'central_ci_upper': c_hi,
        'local_n': len(loc), 'local_gap': l_m, 'local_se': l_se,
        'local_ci_lower': l_lo, 'local_ci_upper': l_hi,
        'diff_central_minus_local': diff, 'diff_se': d_se,
        'diff_ci_lower': d_lo, 'diff_ci_upper': d_hi, 't_stat': t, 'p_value': p,
    })
cl_df = pd.DataFrame(cl_rows)
cl_df.to_csv(OUT / 'central_local_raw.csv', index=False, encoding='utf-8-sig')

# pooled (2019-2025) central vs local
cl_pooled_rows = []
for grp_name, grp in [('Central SOE', raw[raw['central']]),
                      ('Local SOE', raw[raw['local']])]:
    se, n_f = cluster_se(grp['gap_gbm'].values, grp['firm_id'].values)
    m = float(grp['gap_gbm'].mean())
    cl_pooled_rows.append({
        'group': grp_name, 'n_obs': len(grp), 'n_firms': n_f,
        'actual_mean': float(grp['EBI_A'].mean()),
        'predicted_mean': float((grp['EBI_A'] + grp['gap_gbm']).mean()),
        'coefficient': m, 'clustered_se': se,
        'ci_95_lower': m - 1.96 * se if not np.isnan(se) else np.nan,
        'ci_95_upper': m + 1.96 * se if not np.isnan(se) else np.nan,
    })
diff, d_se, d_lo, d_hi, t, p = cluster_t_test(
    raw[raw['central']]['gap_gbm'].values, raw[raw['central']]['firm_id'].values,
    raw[raw['local']]['gap_gbm'].values, raw[raw['local']]['firm_id'].values)
cl_pooled_rows.append({
    'group': 'Central - Local', 'n_obs': None, 'n_firms': None,
    'actual_mean': None, 'predicted_mean': None,
    'coefficient': diff, 'clustered_se': d_se,
    'ci_95_lower': d_lo, 'ci_95_upper': d_hi,
})
cl_pooled = pd.DataFrame(cl_pooled_rows)
cl_pooled.to_csv(OUT / 'central_local_pooled_raw.csv', index=False, encoding='utf-8-sig')
print('\n=== central_local_pooled_raw ===')
print(cl_pooled.round(4).to_string(index=False))


# ============================================================================
# D12: amount gap decomposition (RAW GBM gap, v2 support)
# ============================================================================

CATS = ['Good', 'Boundary', 'Extrapolation']

def decompose(sub):
    total_firms = len(sub)
    total_assets = sub['total_assets'].sum()
    total_gap = sub['amount_gap_gbm'].sum()
    out = {'total_firms': total_firms, 'total_assets': total_assets,
           'total_amount_gap': total_gap}
    for c in CATS:
        d = sub[sub['support_status'] == c]
        out[f'{c.lower()}_firms'] = len(d)
        out[f'{c.lower()}_assets'] = d['total_assets'].sum()
        out[f'{c.lower()}_amount_gap'] = d['amount_gap_gbm'].sum()
    for c in CATS:
        out[f'{c.lower()}_asset_share'] = out[f'{c.lower()}_assets'] / total_assets if total_assets else np.nan
        out[f'{c.lower()}_amount_share'] = out[f'{c.lower()}_amount_gap'] / total_gap if total_gap else np.nan
    out['good_plus_boundary_amount_gap'] = out['good_amount_gap'] + out['boundary_amount_gap']
    out['good_plus_boundary_asset_share'] = (out['good_assets'] + out['boundary_assets']) / total_assets if total_assets else np.nan
    out['good_plus_boundary_amount_share'] = out['good_plus_boundary_amount_gap'] / total_gap if total_gap else np.nan
    return out


amt_rows = []
for year in YEARS:
    for sample in SAMPLES:
        sub = sample_subset(raw, sample)
        sub = sub[sub['ForecastYear'] == year]
        if len(sub) == 0:
            continue
        d = decompose(sub)
        d['year'] = year
        d['sample'] = sample
        amt_rows.append(d)
amt = pd.DataFrame(amt_rows)
amt_cols = ['year', 'sample', 'total_firms', 'good_firms', 'boundary_firms', 'extrapolation_firms',
            'total_assets', 'good_assets', 'boundary_assets', 'extrapolation_assets',
            'good_asset_share', 'boundary_asset_share', 'extrapolation_asset_share',
            'total_amount_gap', 'good_amount_gap', 'boundary_amount_gap', 'extrapolation_amount_gap',
            'good_amount_share', 'boundary_amount_share', 'extrapolation_amount_share',
            'good_plus_boundary_amount_gap', 'good_plus_boundary_asset_share',
            'good_plus_boundary_amount_share']
amt = amt[amt_cols]
amt.to_csv(OUT / 'amount_gap_by_year_raw.csv', index=False, encoding='utf-8-sig')

# 2025 support decomposition (like support_amount_2025_v2)
amt25_rows = []
for sample in SAMPLES:
    s = sample_subset(raw, sample)
    s = s[s['ForecastYear'] == 2025]
    g = s[s['support_status'] == 'Good']
    b = s[s['support_status'] == 'Boundary']
    e = s[s['support_status'] == 'Extrapolation']
    cs = s[s['support_status'].isin(['Good', 'Boundary'])]
    total_amt = s['amount_gap_gbm'].sum()
    amt25_rows.append({
        'sample': sample,
        'N_good': len(g), 'N_boundary': len(b), 'N_extrapolation': len(e),
        'Good_gap_pp': g['gap_gbm'].mean() * 100 if len(g) else np.nan,
        'Boundary_gap_pp': b['gap_gbm'].mean() * 100 if len(b) else np.nan,
        'Extrapolation_gap_pp': e['gap_gbm'].mean() * 100 if len(e) else np.nan,
        'Good_amount_yi': g['amount_gap_gbm'].sum(),
        'Boundary_amount_yi': b['amount_gap_gbm'].sum(),
        'Extrapolation_amount_yi': e['amount_gap_gbm'].sum(),
        'Good_amount_share': g['amount_gap_gbm'].sum() / total_amt if total_amt else np.nan,
        'Boundary_amount_share': b['amount_gap_gbm'].sum() / total_amt if total_amt else np.nan,
        'Extrapolation_amount_share': e['amount_gap_gbm'].sum() / total_amt if total_amt else np.nan,
        'Total_amount_yi': total_amt,
        'CS_amount_yi': cs['amount_gap_gbm'].sum(),
    })
amt25 = pd.DataFrame(amt25_rows)
amt25.to_csv(OUT / 'amount_gap_2025_support_raw.csv', index=False, encoding='utf-8-sig')
print('\n=== amount_gap_2025_support_raw (All SOE) ===')
print(amt25[amt25['sample'] == 'All A-share SOE'].round(3).to_string(index=False))


# ============================================================================
# D13: private OOF placebo (GBM, raw target, single full-pool GroupKFold)
# ============================================================================

from scipy import stats as scipy_stats  # noqa: E402

train = all_a[all_a['is_private']].copy()
prep_priv = prepare_train(train, feats)
tv_priv = prep_priv['tv']
X_priv = prep_priv['Xt']; y_priv_raw = tv_priv['EBI_A'].values
groups_priv = tv_priv['firm_id'].values
gkf = GroupKFold(n_splits=5)
model_priv = build('gbm')
oof_priv = cross_val_predict(model_priv, X_priv, y_priv_raw,
                             cv=gkf.split(X_priv, y_priv_raw, groups=groups_priv))
resid_priv = oof_priv - y_priv_raw


def ci_of(gaps, fids):
    se, nf = cluster_se(gaps, fids)
    m = float(np.mean(gaps))
    lo = m - 1.96 * se if not np.isnan(se) else np.nan
    hi = m + 1.96 * se if not np.isnan(se) else np.nan
    t = m / se if se and se > 0 else np.nan
    p = 2 * (1 - scipy_stats.t.cdf(abs(t), nf - 1)) if not np.isnan(t) and nf > 1 else np.nan
    return m, se, lo, hi, nf, p


m_p, se_p, lo_p, hi_p, nf_p, p_p = ci_of(resid_priv, groups_priv)
placebo = pd.DataFrame([{
    'group': 'Private OOF placebo',
    'n_obs': len(resid_priv), 'n_firms': nf_p,
    'mean_residual': m_p, 'median_residual': float(np.median(resid_priv)),
    'clustered_se': se_p, 'ci_95_lower': lo_p, 'ci_95_upper': hi_p, 'p_value': p_p,
    'oof_r2': r2_score(y_priv_raw, oof_priv),
    'oof_rmse': np.sqrt(mean_squared_error(y_priv_raw, oof_priv)),
    'oof_mae': mean_absolute_error(y_priv_raw, oof_priv),
}])
placebo.to_csv(OUT / 'private_oof_placebo_raw.csv', index=False, encoding='utf-8-sig')
print('\n=== private_oof_placebo_raw ===')
print(placebo.round(4).to_string(index=False))


# ============================================================================
# D14: model replacement (GBM/Ridge/RF Full + CS + Persistent, v2 support)
# ============================================================================

mr_rows = []
for model_name in MODELS:
    gcol = f'gap_{model_name.lower()}'
    for sample in SAMPLES:
        sub = sample_subset(raw, sample).copy()
        r = cluster_bootstrap_support(sub, gcol)
        mr_rows.append({'Model': model_name, 'Sample': sample,
                        'Full_gap_pp': r['Full_gap_pp'],
                        'Full_CI_low': r['Full_CI_low'], 'Full_CI_high': r['Full_CI_high'],
                        'CS_gap_pp': r['CS_gap_pp'],
                        'CS_CI_low': r['CS_CI_low'], 'CS_CI_high': r['CS_CI_high'],
                        'PersistentGood_gap_pp': r['PersistentGood_gap_pp'],
                        'PersistentGood_CI_low': r['PersistentGood_CI_low'],
                        'PersistentGood_CI_high': r['PersistentGood_CI_high'],
                        'N_firms': r['N_firms'],
                        'N_PersistentGood_firms': r['N_PersistentGood_firms']})
mr = pd.DataFrame(mr_rows)
mr.to_csv(OUT / 'model_replacement_gap_raw.csv', index=False, encoding='utf-8-sig')
print('\n=== model_replacement_gap_raw (Full gap, All SOE) ===')
print(mr[mr['Sample'] == 'All A-share SOE'].round(4).to_string(index=False))


# ============================================================================
# D15: bootstrap inference (firm-cluster bootstrap, Full/CS/Persistent, all models)
# ============================================================================

boot_inf_rows = []
for model_name in MODELS:
    gcol = f'gap_{model_name.lower()}'
    for sample in SAMPLES:
        sub = sample_subset(raw, sample).copy()
        r = cluster_bootstrap_support(sub, gcol)
        for metric in ['Full', 'CS', 'PersistentGood']:
            boot_inf_rows.append({
                'Model': model_name, 'Sample': sample, 'Metric': metric,
                'point_pp': r[f'{metric}_gap_pp'],
                'SE_pp': r[f'{metric}_SE'],
                'CI_low_pp': r[f'{metric}_CI_low'],
                'CI_high_pp': r[f'{metric}_CI_high'],
                'p_value': r[f'{metric}_p'],
                'n_boot': N_BOOT,
            })
boot_inf = pd.DataFrame(boot_inf_rows)
boot_inf.to_csv(OUT / 'bootstrap_inference_raw.csv', index=False, encoding='utf-8-sig')


# ============================================================================
# D16: outcome-definition robustness (reformat already-computed 3-spec audit)
# ============================================================================

osrc = pd.read_csv(BASE.parent / 'outcome_spec' / 'outcome_spec_multiyear_summary.csv')
ob = osrc[['Specification', 'Sample', 'N_firmyears', 'N_firms',
           'Mean_gap_pp', 'CI95_low_pp', 'CI95_high_pp']].copy()
ob = ob.rename(columns={'Specification': 'Specification', 'Mean_gap_pp': 'Mean_gap_pp'})
ob.to_csv(OUT / 'outcome_definition_robustness_final.csv', index=False, encoding='utf-8-sig')
print('\n=== outcome_definition_robustness_final (All SOE) ===')
print(ob[ob['Sample'] == 'All A-share SOE'].round(4).to_string(index=False))


print('\n✓ raw_main_run.py complete. All CSVs written to outputs/raw_main/')
