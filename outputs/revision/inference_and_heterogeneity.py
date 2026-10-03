#!/usr/bin/env python3
"""Inference stats + Central/Local split + supplementary plots (screened 2025)."""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / "final_results"
INDEX_DIR = PROJECT_DIR / "input_index_weight"

sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import *
OUTPUT_DIR = PROJECT_DIR / "final_results"  # 重新断言：import * 会把 OUTPUT_DIR 覆盖为 outputs/
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats as scipy_stats

RANDOM_STATE=42; SOE={'中央国有企业','地方国有企业'}

plt.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['Arial Unicode MS','PingFang SC','Hiragino Sans GB','STHeiti','DejaVu Sans'],
    'font.size':9, 'axes.titlesize':11, 'axes.labelsize':10,
    'figure.dpi':300, 'savefig.dpi':300,
    'savefig.bbox':'tight', 'savefig.pad_inches':0.15,
    'axes.spines.top':False, 'axes.spines.right':False,
})

# ============================================================================
# 1. Load + screen + predict (GBM, 2025)
# ============================================================================
print('Loading...')
all_a = load_all_a_shares(); all_a = clean_and_filter(all_a, 'A')
st_firms = set(all_a[all_a['firm_name'].str.contains('ST', na=False)]['firm_id'].unique())
all_a = all_a[~all_a['firm_id'].isin(st_firms)]
all_a = all_a[~(all_a['FinancialLeverage'] > 1.0)]
all_a = all_a[all_a['EBI_A'].abs() <= 0.5]
all_a['is_soe'] = all_a['ownership'].isin(SOE)
all_a['is_private'] = all_a['ownership']=='民营企业'
all_a['central'] = all_a['ownership']=='中央国有企业'
all_a['local'] = all_a['ownership']=='地方国有企业'

train = all_a[(all_a['is_private']) & (all_a['target_year'] < 2025)]  # 严格 ex-ante：2025 预测只用 <2025 结果
feats = NUMERIC_FEATURES
tv = train.dropna(subset=feats+['EBI_A']).copy()
for c in feats: lo,hi=tv[c].quantile(0.05),tv[c].quantile(0.95); tv[c+'_w']=tv[c].clip(lo,hi)
yl,yh=tv['EBI_A'].quantile(0.05),tv['EBI_A'].quantile(0.95); tv['EBI_A_w']=tv['EBI_A'].clip(yl,yh)
ncw=[c+'_w' for c in feats]
imp=SimpleImputer(strategy='median'); scl=StandardScaler()
Xt=scl.fit_transform(imp.fit_transform(tv[ncw].values))
ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
Xt=np.hstack([Xt,ohe.fit_transform(tv[['industry_l2']].fillna('Unknown'))])
m=GradientBoostingRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,min_samples_leaf=10,random_state=RANDOM_STATE)
m.fit(Xt, tv['EBI_A_w'].values)

soe_2025 = all_a[(all_a['is_soe'])&(all_a['target_year']==2025)]
pv = soe_2025.copy()
for c in feats: lo,hi=tv[c].quantile(0.05),tv[c].quantile(0.95); pv[c+'_w']=pv[c].clip(lo,hi)
pd2 = pv.dropna(subset=ncw)
Xp = np.hstack([scl.transform(imp.transform(pd2[ncw].values)), ohe.transform(pd2[['industry_l2']].fillna('Unknown'))])
preds = m.predict(Xp)
pd2 = pd2.copy()
pd2['predicted'] = preds
pd2['gap'] = preds - pd2['EBI_A'].values
pd2['FirmSize_raw'] = pd2['FirmSize']

# ============================================================================
# 2. Inference stats (clustered SE by firm_id)
# ============================================================================
def clustered_inference(gaps, firm_ids):
    gaps = np.array(gaps); firm_ids = np.array(firm_ids)
    unique = np.unique(firm_ids)
    firm_means = np.array([gaps[firm_ids==f].mean() for f in unique])
    n_clusters = len(unique)
    point = np.mean(gaps)
    se = np.std(firm_means, ddof=1)/np.sqrt(n_clusters) if n_clusters>1 else np.nan
    ci_lo = point - 1.96*se; ci_hi = point + 1.96*se
    t_stat = point/se if se>0 else np.nan
    p_val = 2*(1-scipy_stats.t.cdf(abs(t_stat), df=n_clusters-1)) if not np.isnan(t_stat) else np.nan
    return {'coefficient': point, 'clustered_se': se, 'n_clusters': n_clusters,
            'ci_95_lower': ci_lo, 'ci_95_upper': ci_hi, 't_stat': t_stat, 'p_value': p_val}

infer_rows = []
for grp_name, grp in [('All SOE', pd2), ('Central SOE', pd2[pd2['central']]), ('Local SOE', pd2[pd2['local']])]:
    r = clustered_inference(grp['gap'].values, grp['firm_id'].values)
    r['group'] = grp_name
    infer_rows.append(r)
infer_df = pd.DataFrame(infer_rows)
infer_df.to_csv(OUTPUT_DIR/'inference_stats_2025.csv', index=False, encoding='utf-8-sig')
print('\n=== 推断统计（clustered by firm_id, 2025）===')
print(infer_df[['group','coefficient','clustered_se','n_clusters','ci_95_lower','ci_95_upper','t_stat','p_value']].round(4).to_string(index=False))

# ============================================================================
# 3. Central vs Local split
# ============================================================================
print('\n=== 中央 vs 地方 SOE ===')
split_rows = []
for grp_name, grp in [('Central SOE', pd2[pd2['central']]), ('Local SOE', pd2[pd2['local']])]:
    r = clustered_inference(grp['gap'].values, grp['firm_id'].values)
    r['group'] = grp_name
    r['n_firms'] = grp['firm_id'].nunique()
    r['actual_mean'] = grp['EBI_A'].mean()
    r['predicted_mean'] = grp['predicted'].mean()
    split_rows.append(r)
split_df = pd.DataFrame(split_rows)
split_df.to_csv(OUTPUT_DIR/'central_vs_local_2025.csv', index=False, encoding='utf-8-sig')
print(split_df[['group','n_firms','actual_mean','predicted_mean','coefficient','clustered_se','ci_95_lower','ci_95_upper','p_value']].round(4).to_string(index=False))

# ============================================================================
# 4. Supplementary plots: size-gap scatter + prediction vs actual
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

# --- size-gap binned plot ---
ax = axes[0]; ax.set_facecolor('white')
# bin by FirmSize decile
pd2['size_decile'] = pd.qcut(pd2['FirmSize_raw'].rank(method='first'), 10, labels=False)
binned = pd2.groupby('size_decile').agg(
    size_mean=('FirmSize_raw','mean'), gap_mean=('gap','mean'),
    gap_se=('gap', lambda x: np.std(x)/np.sqrt(len(x)))).reset_index()
ax.errorbar(binned['size_mean'], binned['gap_mean']*100, yerr=binned['gap_se']*100*1.96,
            fmt='o-', color='#2C3E50', capsize=3, markersize=6, linewidth=1.8)
ax.axhline(0, color='#999', linestyle='--', linewidth=0.8)
ax.set_xlabel('FirmSize = ln(Total Assets)')
ax.set_ylabel('Mean Gap (pp)')
ax.set_title('Gap by Firm Size (10 bins)', fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

# --- prediction vs actual ---
ax = axes[1]; ax.set_facecolor('white')
ax.scatter(pd2['EBI_A']*100, pd2['predicted']*100, s=12, alpha=0.4, color='#2980B9', edgecolor='none')
lo = min(pd2['EBI_A'].min(), pd2['predicted'].min())*100
hi = max(pd2['EBI_A'].max(), pd2['predicted'].max())*100
ax.plot([lo,hi],[lo,hi], '--', color='#E74C3C', linewidth=1.2, label='45° line')
ax.set_xlabel('Actual EBI/A (%)')
ax.set_ylabel('Predicted EBI/A (%)')
ax.set_title('Predicted vs Actual (GBM, 2025)', fontweight='bold')
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR/'fig_supp_size_gap_and_calibration.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(OUTPUT_DIR/'fig_supp_size_gap_and_calibration.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print('\n✓ fig_supp_size_gap_and_calibration.png/pdf saved')

# 保存 firm 级数据（含 central/local, predicted, actual, gap, FirmSize）
pd2[['firm_id','firm_name','ownership','central','local','industry_l2','FirmSize_raw','EBI_A','predicted','gap']].to_csv(
    OUTPUT_DIR/'firm_level_prediction_2025.csv', index=False, encoding='utf-8-sig')
print('✓ firm_level_prediction_2025.csv saved')

print('\n✓ 推断统计 + 中央/地方拆分 + 补充图完成')
