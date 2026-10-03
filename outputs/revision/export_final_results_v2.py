#!/usr/bin/env python3
"""Export final results v2 — fixed column names + mutually-exclusive samples."""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
import openpyxl

warnings.filterwarnings('ignore')
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / "final_results"
OUTPUT_DIR.mkdir(exist_ok=True)
INDEX_DIR = PROJECT_DIR / "input_index_weight"

sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import *
OUTPUT_DIR = PROJECT_DIR / "final_results"  # 重新断言：import * 会把 OUTPUT_DIR 覆盖为 outputs/
from sklearn.ensemble import GradientBoostingRegressor
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
train_pool = all_a[all_a['is_private']]

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

feats=[f for f in NUMERIC_FEATURES if f in all_a.columns]

firm_rows = []
for fc_year in range(2019, 2026):
    train = train_pool[train_pool['target_year']<fc_year].copy()  # 严格 ex-ante：结果年份 < 预测年
    tv = train.dropna(subset=feats+['EBI_A']).copy()
    for c in feats: lo,hi=tv[c].quantile(0.05),tv[c].quantile(0.95); tv[c+'_w']=tv[c].clip(lo,hi)
    yl,yh=tv['EBI_A'].quantile(0.05),tv['EBI_A'].quantile(0.95); tv['EBI_A_w']=tv['EBI_A'].clip(yl,yh)
    ncw=[c+'_w' for c in feats]
    imp=SimpleImputer(strategy='median'); scl=StandardScaler()
    Xt=scl.fit_transform(imp.fit_transform(tv[ncw].values))
    ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    Xt=np.hstack([Xt,ohe.fit_transform(tv[['industry_l2']].fillna('Unknown'))])
    yt=tv['EBI_A_w'].values
    m = GradientBoostingRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,min_samples_leaf=10,random_state=RANDOM_STATE)
    m.fit(Xt, yt)

    soe_all = all_a[(all_a['is_soe'])&(all_a['target_year']==fc_year)]
    # 指数并集
    idx_union = set().union(*[yearly_codes[i][fc_year] for i in ['CSI300','CSI500','CSI1000']])
    non_idx = soe_all[~soe_all['firm_id'].isin(idx_union)]

    # 定义互斥样本：Non-index + CSI300 + CSI500 + CSI1000 = All
    targets = {
        'Non-index SOE': non_idx,
        'CSI300 SOE': soe_all[soe_all['firm_id'].isin(yearly_codes['CSI300'][fc_year])],
        'CSI500 SOE': soe_all[soe_all['firm_id'].isin(yearly_codes['CSI500'][fc_year])],
        'CSI1000 SOE': soe_all[soe_all['firm_id'].isin(yearly_codes['CSI1000'][fc_year])],
    }

    # 预测所有 SOE（含 index 标记）
    pv_all = soe_all.copy()
    for c in feats: lo,hi=tv[c].quantile(0.05),tv[c].quantile(0.95); pv_all[c+'_w']=pv_all[c].clip(lo,hi)
    pd_all = pv_all.dropna(subset=ncw)
    Xp_all = np.hstack([scl.transform(imp.transform(pd_all[ncw].values)),
                        ohe.transform(pd_all[['industry_l2']].fillna('Unknown'))])
    preds_all = m.predict(Xp_all)

    # 给每个 SOE 标 index 归属
    def index_label(fid):
        labels = []
        for i in ['CSI300','CSI500','CSI1000']:
            if fid in yearly_codes[i][fc_year]: labels.append(i)
        return '+'.join(labels) if labels else 'Non-index'

    for i in range(len(pd_all)):
        fid = pd_all['firm_id'].iloc[i]
        actual_ratio = pd_all['EBI_A'].iloc[i]
        pred_ratio = preds_all[i]
        assets = pd_all['资产总计_target'].iloc[i]
        gap_ratio = pred_ratio - actual_ratio
        amount_yuan = gap_ratio * assets
        firm_rows.append({
            'year': fc_year,
            'firm_id': fid,
            'firm_name': pd_all['firm_name'].iloc[i] if 'firm_name' in pd_all.columns else '',
            'industry_l2': pd_all['industry_l2'].iloc[i],
            'ownership': pd_all['ownership'].iloc[i],
            'index_membership': index_label(fid),
            'total_assets_target_yuan': assets,
            'actual_ebia_ratio': actual_ratio,
            'predicted_ebia_ratio': pred_ratio,
            'actual_ebi_yuan': actual_ratio * assets,   # 实际 EBI（元）
            'predicted_ebi_yuan': pred_ratio * assets,  # 预测 EBI（元）
            'gap_ratio': gap_ratio,
            'gap_amount_yuan': amount_yuan,
            'gap_amount_yi': amount_yuan / 1e8,
        })
    print(f'  {fc_year}: All SOE={len(soe_all)}, Non-index={len(non_idx)}, '
          f'300={len(targets["CSI300 SOE"])}, 500={len(targets["CSI500 SOE"])}, 1000={len(targets["CSI1000 SOE"])}')

firm_df = pd.DataFrame(firm_rows)
firm_df.to_csv(OUTPUT_DIR/'firm_level_gap_all_years.csv', index=False, encoding='utf-8-sig')
firm_df[firm_df['year']==2025].to_csv(OUTPUT_DIR/'firm_level_gap_2025.csv', index=False, encoding='utf-8-sig')
print(f'\n✓ firm_level CSVs saved ({len(firm_df)} rows)')

# ============================================================================
# 汇总（互斥样本）
# ============================================================================
# 分样本 × 年份（用 index_membership 列，互斥）
sample_map = {
    'Non-index': 'Non-index SOE',
    'CSI300': 'CSI300 SOE',
    'CSI500': 'CSI500 SOE',
    'CSI1000': 'CSI1000 SOE',
    'CSI300+CSI500': 'CSI300+CSI500 SOE',
    'CSI300+CSI1000': 'CSI300+CSI1000 SOE',
    'CSI500+CSI1000': 'CSI500+CSI1000 SOE',
    'CSI300+CSI500+CSI1000': 'CSI300+CSI500+CSI1000 SOE',
}
firm_df['sample'] = firm_df['index_membership'].map(sample_map)

# 全市场（All）作为单独汇总
all_group = firm_df.groupby('year').agg(
    n_firms=('firm_id','count'),
    mean_gap_ratio=('gap_ratio','mean'),
    sum_gap_amount_yi=('gap_amount_yi','sum'),
    mean_gap_amount_yi=('gap_amount_yi','mean'),
).reset_index()
all_group['sample'] = 'All A-share SOE'

sub_group = firm_df.groupby(['sample','year']).agg(
    n_firms=('firm_id','count'),
    mean_gap_ratio=('gap_ratio','mean'),
    sum_gap_amount_yi=('gap_amount_yi','sum'),
    mean_gap_amount_yi=('gap_amount_yi','mean'),
).reset_index()

sum_sample_year = pd.concat([all_group, sub_group], ignore_index=True)
sum_sample_year.to_csv(OUTPUT_DIR/'summary_by_sample_year.csv', index=False, encoding='utf-8-sig')
print('✓ summary_by_sample_year.csv saved')

# 分行业 × 年份（All SOE）
sum_ind_year = firm_df.groupby(['industry_l2','year']).agg(
    n_firms=('firm_id','count'),
    mean_gap_ratio=('gap_ratio','mean'),
    sum_gap_amount_yi=('gap_amount_yi','sum'),
).reset_index()
sum_ind_year.to_csv(OUTPUT_DIR/'summary_by_industry_year.csv', index=False, encoding='utf-8-sig')
print('✓ summary_by_industry_year.csv saved')

# 分行业 2025
sum_ind_2025 = firm_df[firm_df['year']==2025].groupby('industry_l2').agg(
    n_firms=('firm_id','count'),
    mean_gap_ratio=('gap_ratio','mean'),
    median_gap_ratio=('gap_ratio','median'),
    sum_gap_amount_yi=('gap_amount_yi','sum'),
    mean_gap_amount_yi=('gap_amount_yi','mean'),
).reset_index().sort_values('sum_gap_amount_yi', ascending=False)
sum_ind_2025.to_csv(OUTPUT_DIR/'summary_by_industry_2025.csv', index=False, encoding='utf-8-sig')
print('✓ summary_by_industry_2025.csv saved')

# ============================================================================
# 验证互斥性
# ============================================================================
print('\n=== 验证样本互斥性 ===')
for year in [2025]:
    sub = firm_df[firm_df['year']==year]
    all_n = len(sub)
    n_non = (sub['index_membership']=='Non-index').sum()
    n_300 = sub['index_membership'].str.contains('CSI300').sum()
    n_500 = sub['index_membership'].str.contains('CSI500').sum()
    n_1000 = sub['index_membership'].str.contains('CSI1000').sum()
    print(f'  {year}: All={all_n}, Non-index={n_non}, 含CSI300={n_300}, 含CSI500={n_500}, 含CSI1000={n_1000}')
    print(f'  互斥分解: Non-index + CSI300 + CSI500 + CSI1000 = {n_non + n_300 + n_500 + n_1000}（应≈All，但重叠指数 firm 会重复计）')

print('\n✓ v2 导出完成')
