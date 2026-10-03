#!/usr/bin/env python3
"""Full common-support / overlap diagnostics on screened sample (2025)."""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
import openpyxl

warnings.filterwarnings('ignore')
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / "final_results"
INDEX_DIR = PROJECT_DIR / "input_index_weight"

sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import *
OUTPUT_DIR = PROJECT_DIR / "final_results"  # 重新断言：import * 会把 OUTPUT_DIR 覆盖为 outputs/
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

RANDOM_STATE=42; SOE={'中央国有企业','地方国有企业'}

# ============================================================================
# 1. Load + screen
# ============================================================================
print('Loading...')
all_a = load_all_a_shares(); all_a = clean_and_filter(all_a, 'A')
st_firms = set(all_a[all_a['firm_name'].str.contains('ST', na=False)]['firm_id'].unique())
all_a = all_a[~all_a['firm_id'].isin(st_firms)]
all_a = all_a[~(all_a['FinancialLeverage'] > 1.0)]
all_a = all_a[all_a['EBI_A'].abs() <= 0.5]
all_a['is_soe'] = all_a['ownership'].isin(SOE)
all_a['is_private'] = all_a['ownership']=='民营企业'

# 训练集（民企，全部年份用于训练）
train = all_a[(all_a['is_private']) & (all_a['target_year'] < 2025)].dropna(subset=NUMERIC_FEATURES + ['EBI_A'])  # 严格 ex-ante

# 预测集（SOE 2025）
soe_2025 = all_a[(all_a['is_soe'])&(all_a['target_year']==2025)]

# 指数成分
idx_files = {
    'CSI300': ('沪深300-成分及权重-20260811.xlsx', '沪深300-成分进出记录-20260811.xlsx'),
    'CSI500': ('中证500-成分及权重-20260811.xlsx', '中证500-成分进出记录-20260811.xlsx'),
    'CSI1000': ('中证1000-成分及权重-20260811.xlsx', '中证1000-成分进出记录-20260811.xlsx'),
}
codes={}; all_records={}
for idx,(fw,fe) in idx_files.items():
    wb=openpyxl.load_workbook(INDEX_DIR/fw, read_only=True, data_only=True); ws=wb.active
    codes[idx]={str(r[0]).strip() for r in ws.iter_rows(min_row=2, values_only=True) if r[0] and str(r[2]).strip()!='金融'}
    wb.close()
    wb2=openpyxl.load_workbook(INDEX_DIR/fe, read_only=True, data_only=True); ws2=wb2.active
    all_records[idx]=sorted([{'date':str(r[0])[:10],'code':str(r[1]).strip(),'action':str(r[7]).strip()}
                             for r in ws2.iter_rows(min_row=2, values_only=True) if r[0] and r[1] and r[7]],
                            key=lambda x:x['date'])
    wb2.close()
yearly_codes={}
for idx in ['CSI300','CSI500','CSI1000']:
    cur=codes[idx]; rev=sorted(all_records[idx],key=lambda r:r['date'],reverse=True)
    yearly_codes[idx]={}
    for y in range(2019,2026):
        td=f'{y}-07-01'; cs=set(cur)
        for rec in rev:
            if rec['date']<=td: break
            if rec['action']=='纳入': cs.discard(rec['code'])
            elif rec['action']=='剔除': cs.add(rec['code'])
        yearly_codes[idx][y]=cs

# SOE 样本
targets = {
    'All A-share SOE': soe_2025,
    'CSI300 SOE': soe_2025[soe_2025['firm_id'].isin(yearly_codes['CSI300'][2025])],
    'CSI500 SOE': soe_2025[soe_2025['firm_id'].isin(yearly_codes['CSI500'][2025])],
    'CSI1000 SOE': soe_2025[soe_2025['firm_id'].isin(yearly_codes['CSI1000'][2025])],
}

feats = NUMERIC_FEATURES

# 有效观测数
print('\n=== 有效观测数（2025，筛选后）===')
n_summary = [{'group': 'Private (train)', 'n': len(train), 'n_firms': train['firm_id'].nunique()}]
for name, df in targets.items():
    valid = df.dropna(subset=feats)
    n_summary.append({'group': name, 'n': len(valid), 'n_firms': valid['firm_id'].nunique()})
n_df = pd.DataFrame(n_summary)
print(n_df.to_string(index=False))

# ============================================================================
# 2. 单变量 SMD
# ============================================================================
print('\n=== 单变量 SMD ===')
priv = train[feats]
smd_rows = []
for name, df in targets.items():
    soe = df.dropna(subset=feats)[feats]
    for f in feats:
        p = priv[f].dropna(); s = soe[f].dropna()
        if len(p)==0 or len(s)==0: continue
        pooled = np.sqrt((p.var() + s.var())/2)
        smd = (s.mean() - p.mean())/pooled if pooled>0 else np.nan
        pct_outside_1_99 = np.mean((s < p.quantile(0.01)) | (s > p.quantile(0.99)))
        pct_outside_5_95 = np.mean((s < p.quantile(0.05)) | (s > p.quantile(0.95)))
        smd_rows.append({
            'sample': name, 'feature': f,
            'priv_mean': p.mean(), 'soe_mean': s.mean(),
            'SMD': smd, 'pct_outside_P1P99': pct_outside_1_99,
            'pct_outside_P5P95': pct_outside_5_95,
        })
smd_df = pd.DataFrame(smd_rows)
smd_df.to_csv(OUTPUT_DIR/'cs_smd.csv', index=False, encoding='utf-8-sig')
print(smd_df[['sample','feature','SMD','pct_outside_P5P95']].round(3).to_string(index=False))

# ============================================================================
# 3. Propensity score（民企 vs 每个 SOE 样本）
# ============================================================================
print('\n=== Propensity Score ===')
ps_rows = []
for name, df in targets.items():
    soe = df.dropna(subset=feats)
    if len(soe) < 20: continue
    X_priv = priv.values; X_soe = soe[feats].values
    X = np.vstack([X_priv, X_soe])
    y = np.hstack([np.zeros(len(X_priv)), np.ones(len(X_soe))])
    scl = StandardScaler(); Xs = scl.fit_transform(X)
    # OOF propensity
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_ps = np.zeros(len(X))
    for tr, te in skf.split(Xs, y):
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(Xs[tr], y[tr]); oof_ps[te] = lr.predict_proba(Xs[te])[:,1]
    auc = roc_auc_score(y, oof_ps)
    priv_ps = oof_ps[y==0]; soe_ps = oof_ps[y==1]
    p01, p99 = np.quantile(priv_ps, [0.01, 0.99])
    p_min, p_max = priv_ps.min(), priv_ps.max()
    outside_trimmed = np.mean((soe_ps < p01)|(soe_ps > p99))
    outside_range = np.mean((soe_ps < p_min)|(soe_ps > p_max))
    ps_rows.append({
        'sample': name, 'ROC_AUC': auc,
        'SOE_outside_P1P99': outside_trimmed, 'SOE_outside_minmax': outside_range,
        'priv_ps_mean': priv_ps.mean(), 'soe_ps_mean': soe_ps.mean(),
    })
ps_df = pd.DataFrame(ps_rows)
ps_df.to_csv(OUTPUT_DIR/'cs_propensity.csv', index=False, encoding='utf-8-sig')
print(ps_df.round(4).to_string(index=False))

# ============================================================================
# 4. 最近邻距离 + Common support 分类
# ============================================================================
print('\n=== 最近邻距离 + Common Support ===')
scl_nn = StandardScaler()
X_priv_nn = scl_nn.fit_transform(priv.values)
nn_priv = NearestNeighbors(n_neighbors=2, metric='euclidean'); nn_priv.fit(X_priv_nn)
priv_dist = nn_priv.kneighbors(X_priv_nn)[0][:,1]
p50, p90, p95 = np.quantile(priv_dist, [0.50, 0.90, 0.95])

cs_rows = []
for name, df in targets.items():
    soe = df.dropna(subset=feats)
    if len(soe)==0: continue
    X_soe_nn = scl_nn.transform(soe[feats].values)
    nn_soe = NearestNeighbors(n_neighbors=5, metric='euclidean'); nn_soe.fit(X_priv_nn)
    soe_dist = nn_soe.kneighbors(X_soe_nn)[0][:,0]
    good = np.mean(soe_dist <= p90)
    boundary = np.mean((soe_dist > p90) & (soe_dist <= p95))
    extrap = np.mean(soe_dist > p95)
    cs_rows.append({
        'sample': name, 'n': len(soe),
        'n_good_support': int(np.sum(soe_dist <= p90)),
        'n_boundary': int(np.sum((soe_dist>p90)&(soe_dist<=p95))),
        'n_extrapolation': int(np.sum(soe_dist>p95)),
        'pct_good_support': good, 'pct_boundary': boundary, 'pct_extrapolation': extrap,
        'median_soe_nn_dist': np.median(soe_dist),
    })
cs_df = pd.DataFrame(cs_rows)
cs_df.to_csv(OUTPUT_DIR/'cs_common_support.csv', index=False, encoding='utf-8-sig')
print(f'民企内部距离: P50={p50:.4f}, P90={p90:.4f}, P95={p95:.4f}')
print(cs_df.round(4).to_string(index=False))

# ============================================================================
# 5. Mahalanobis distance（PCA 空间）
# ============================================================================
print('\n=== Mahalanobis distance（PCA 前 5 主成分）===')
pca = PCA(n_components=min(5, len(feats)))
pca.fit(X_priv_nn)
priv_pca = pca.transform(X_priv_nn)
cov_inv = np.linalg.inv(np.cov(priv_pca.T))
priv_center = priv_pca.mean(axis=0)
priv_mahal = np.array([np.sqrt((p-priv_center).T @ cov_inv @ (p-priv_center)) for p in priv_pca])
mah_p90, mah_p95 = np.quantile(priv_mahal, [0.90, 0.95])

mahal_rows = []
for name, df in targets.items():
    soe = df.dropna(subset=feats)
    if len(soe)==0: continue
    soe_pca = pca.transform(scl_nn.transform(soe[feats].values))
    soe_mahal = np.array([np.sqrt((p-priv_center).T @ cov_inv @ (p-priv_center)) for p in soe_pca])
    mahal_rows.append({
        'sample': name, 'n': len(soe),
        'median_mahal': np.median(soe_mahal),
        'pct_outside_P90': np.mean(soe_mahal > mah_p90),
        'pct_outside_P95': np.mean(soe_mahal > mah_p95),
    })
mahal_df = pd.DataFrame(mahal_rows)
mahal_df.to_csv(OUTPUT_DIR/'cs_mahalanobis.csv', index=False, encoding='utf-8-sig')
print(f'民企 Mahalanobis: P90={mah_p90:.2f}, P95={mah_p95:.2f}')
print(mahal_df.round(4).to_string(index=False))

# ============================================================================
# 6. Support-restricted gap 对比
# ============================================================================
print('\n=== Support-restricted Gap 对比（GBM, 2025）===')
from sklearn.ensemble import GradientBoostingRegressor
tv = train.dropna(subset=feats+['EBI_A'])
for c in feats: lo,hi=tv[c].quantile(0.05),tv[c].quantile(0.95); tv[c+'_w']=tv[c].clip(lo,hi)
yl,yh=tv['EBI_A'].quantile(0.05),tv['EBI_A'].quantile(0.95); tv['EBI_A_w']=tv['EBI_A'].clip(yl,yh)
ncw=[c+'_w' for c in feats]
imp=SimpleImputer(strategy='median'); scl_m=StandardScaler()
Xt=scl_m.fit_transform(imp.fit_transform(tv[ncw].values))
ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
Xt=np.hstack([Xt,ohe.fit_transform(tv[['industry_l2']].fillna('Unknown'))])
m=GradientBoostingRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,min_samples_leaf=10,random_state=RANDOM_STATE)
m.fit(Xt, tv['EBI_A_w'].values)

gap_rows = []
for name, df in targets.items():
    soe = df.dropna(subset=feats)
    if len(soe)==0: continue
    X_soe_nn = scl_nn.transform(soe[feats].values)
    nn_soe = NearestNeighbors(n_neighbors=5, metric='euclidean'); nn_soe.fit(X_priv_nn)
    soe_dist = nn_soe.kneighbors(X_soe_nn)[0][:,0]
    cs_mask = soe_dist <= p90

    # predict all SOE
    pv = soe.copy()
    for c in feats: lo,hi=tv[c].quantile(0.05),tv[c].quantile(0.95); pv[c+'_w']=pv[c].clip(lo,hi)
    Xp = np.hstack([scl_m.transform(imp.transform(pv[ncw].values)), ohe.transform(pv[['industry_l2']].fillna('Unknown'))])
    preds = m.predict(Xp)
    gaps = preds - pv['EBI_A'].values

    gap_rows.append({
        'sample': name, 'restriction': 'All', 'n': len(soe), 'mean_gap': np.mean(gaps), 'median_gap': np.median(gaps),
    })
    gap_rows.append({
        'sample': name, 'restriction': 'Common support', 'n': int(cs_mask.sum()), 'mean_gap': np.mean(gaps[cs_mask]), 'median_gap': np.median(gaps[cs_mask]),
    })
gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv(OUTPUT_DIR/'cs_support_restricted_gap.csv', index=False, encoding='utf-8-sig')
print(gap_df.round(4).to_string(index=False))

print('\n✓ 全部 common-support 诊断结果已导出到 final_results/')
