#!/usr/bin/env python3
"""Cluster bootstrap for 2025 gap (Ridge, 500 iterations)."""
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
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE=42; SOE={'中央国有企业','地方国有企业'}

print('Loading...')
all_a = load_all_a_shares(); all_a = clean_and_filter(all_a, 'A')
st_firms = set(all_a[all_a['firm_name'].str.contains('ST', na=False)]['firm_id'].unique())
all_a = all_a[~all_a['firm_id'].isin(st_firms)]
all_a = all_a[~(all_a['FinancialLeverage'] > 1.0)]
all_a = all_a[all_a['EBI_A'].abs() <= 0.5]
all_a['is_soe'] = all_a['ownership'].isin(SOE)
all_a['is_private'] = all_a['ownership']=='民营企业'

# 训练集（完整案例）
train = all_a[(all_a['is_private']) & (all_a['target_year'] < 2025)]  # 严格 ex-ante：2025 预测只用 <2025 结果
feats = NUMERIC_FEATURES
tv = train.dropna(subset=feats+['EBI_A']).copy()
soe_2025 = all_a[(all_a['is_soe'])&(all_a['target_year']==2025)].dropna(subset=feats)

# 指数
idx_files = {'CSI300':('沪深300-成分及权重-20260811.xlsx','沪深300-成分进出记录-20260811.xlsx'),
             'CSI500':('中证500-成分及权重-20260811.xlsx','中证500-成分进出记录-20260811.xlsx'),
             'CSI1000':('中证1000-成分及权重-20260811.xlsx','中证1000-成分进出记录-20260811.xlsx')}
codes={}; all_records={}
for idx,(fw,fe) in idx_files.items():
    wb=openpyxl.load_workbook(INDEX_DIR/fw, read_only=True, data_only=True); ws=wb.active
    codes[idx]={str(r[0]).strip() for r in ws.iter_rows(min_row=2, values_only=True) if r[0] and str(r[2]).strip()!='金融'}
    wb.close()
    wb2=openpyxl.load_workbook(INDEX_DIR/fe, read_only=True, data_only=True); ws2=wb2.active
    all_records[idx]=sorted([{'date':str(r[0])[:10],'code':str(r[1]).strip(),'action':str(r[7]).strip()}
                             for r in ws2.iter_rows(min_row=2, values_only=True) if r[0] and r[1] and r[7]], key=lambda x:x['date'])
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

# 样本定义
targets = {
    'All A-share SOE': soe_2025,
    'CSI300 SOE': soe_2025[soe_2025['firm_id'].isin(yearly_codes['CSI300'][2025])],
    'CSI500 SOE': soe_2025[soe_2025['firm_id'].isin(yearly_codes['CSI500'][2025])],
    'CSI1000 SOE': soe_2025[soe_2025['firm_id'].isin(yearly_codes['CSI1000'][2025])],
}

# 预计算：每个 firm 的特征（训练集）
train_firms = tv['firm_id'].unique()
soe_firms_by_target = {k: v['firm_id'].unique() for k,v in targets.items()}

N_BOOT = 500
np.random.seed(RANDOM_STATE)

def fit_predict_bootstrap(boot_train_firms, soe_df, tv_full):
    """训练 Ridge on bootstrap 训练集，预测 soe_df，返回 mean gap."""
    boot_train = tv_full[tv_full['firm_id'].isin(boot_train_firms)].copy()
    if len(boot_train) < 100: return np.nan
    # winsorize
    for c in feats:
        lo,hi = boot_train[c].quantile(0.05), boot_train[c].quantile(0.95); boot_train[c+'_w']=boot_train[c].clip(lo,hi)
    yl,yh = boot_train['EBI_A'].quantile(0.05), boot_train['EBI_A'].quantile(0.95); boot_train['EBI_A_w']=boot_train['EBI_A'].clip(yl,yh)
    ncw=[c+'_w' for c in feats]
    imp=SimpleImputer(strategy='median'); scl=StandardScaler()
    Xt=scl.fit_transform(imp.fit_transform(boot_train[ncw].values))
    ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    Xt=np.hstack([Xt,ohe.fit_transform(boot_train[['industry_l2']].fillna('Unknown'))])
    m=Ridge(alpha=10, random_state=RANDOM_STATE); m.fit(Xt, boot_train['EBI_A_w'].values)
    # predict
    pv = soe_df.copy()
    for c in feats:
        lo,hi = boot_train[c].quantile(0.05), boot_train[c].quantile(0.95); pv[c+'_w']=pv[c].clip(lo,hi)
    pd2 = pv.dropna(subset=ncw)
    Xp = np.hstack([scl.transform(imp.transform(pd2[ncw].values)), ohe.transform(pd2[['industry_l2']].fillna('Unknown'))])
    preds = m.predict(Xp)
    gaps = preds - pd2['EBI_A'].values
    return np.mean(gaps)

# 对每个目标跑 bootstrap
boot_results = {}
for name, soe_df in targets.items():
    if len(soe_df) < 10: continue
    soe_firms = soe_firms_by_target[name]
    gaps = []
    for b in range(N_BOOT):
        boot_train_firms = np.random.choice(train_firms, size=len(train_firms), replace=True)
        boot_soe = soe_df[soe_df['firm_id'].isin(np.random.choice(soe_firms, size=len(soe_firms), replace=True))]
        g = fit_predict_bootstrap(boot_train_firms, boot_soe, tv)
        gaps.append(g)
        if (b+1) % 100 == 0: print(f'  {name}: {b+1}/{N_BOOT}')
    gaps = np.array([g for g in gaps if not np.isnan(g)])
    point = np.mean(gaps); se = np.std(gaps, ddof=1)
    ci = np.quantile(gaps, [0.025, 0.975])
    boot_results[name] = {'point_estimate': point, 'bootstrap_se': se,
                          'ci_95_lower': ci[0], 'ci_95_upper': ci[1],
                          'n_bootstrap': len(gaps)}
    print(f'  {name}: point={point:+.4f}, boot SE={se:.4f}, CI=[{ci[0]:+.4f},{ci[1]:+.4f}]')

boot_df = pd.DataFrame(boot_results).T.reset_index().rename(columns={'index':'sample'})
boot_df.to_csv(OUTPUT_DIR/'bootstrap_inference_2025.csv', index=False, encoding='utf-8-sig')
print('\n✓ bootstrap_inference_2025.csv saved')
print(boot_df.round(4).to_string(index=False))
