#!/usr/bin/env python3
"""Screened-sample industry × year gap + CS version, all figures."""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_DIR / "input_index_weight"

sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import *
OUTPUT_DIR = Path(__file__).resolve().parent  # 重新断言：import * 会把 OUTPUT_DIR 覆盖为 outputs/
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE=42; SOE={'中央国有企业','地方国有企业'}

plt.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['Arial Unicode MS','PingFang SC','Hiragino Sans GB','STHeiti','DejaVu Sans'],
    'font.size':9, 'axes.titlesize':11, 'axes.labelsize':9,
    'figure.dpi':300, 'savefig.dpi':300,
    'savefig.bbox':'tight', 'savefig.pad_inches':0.15,
})

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
train_pool = all_a[all_a['is_private']]
print(f'Train pool: {len(train_pool)} obs, {train_pool["firm_id"].nunique()} firms')

# Index codes
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

def cs_mask(train_df, pred_df):
    tv=train_df[feats].dropna(); pv=pred_df[feats].dropna()
    if len(tv)<10 or len(pv)==0: return np.zeros(len(pred_df),dtype=bool)
    scl=StandardScaler(); Xt=scl.fit_transform(tv.values); Xp=scl.transform(pv.values)
    nn_priv=NearestNeighbors(n_neighbors=2,metric='euclidean'); nn_priv.fit(Xt)
    priv_d=nn_priv.kneighbors(Xt)[0][:,1]; p90=np.quantile(priv_d,0.90)
    nn_soe=NearestNeighbors(n_neighbors=5,metric='euclidean'); nn_soe.fit(Xt)
    soe_d=nn_soe.kneighbors(Xp)[0][:,0]
    mask=np.zeros(len(pred_df),dtype=bool)
    valid_idx=pred_df[feats].dropna().index
    for i,idx in enumerate(valid_idx):
        if soe_d[i]<=p90: mask[list(pred_df.index).index(idx)]=True
    return mask

# ============================================================================
# 2. Compute firm-level gap for all years/samples, plus CS mask
# ============================================================================
records = []  # full sample
records_cs = []  # common support

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

    soe_all = all_a[(all_a['is_soe'])&(all_a['target_year']==fc_year)]
    targets = {'All A-share SOE': soe_all}
    for idx in ['CSI300','CSI500','CSI1000']:
        targets[f'{idx} SOE'] = soe_all[soe_all['firm_id'].isin(yearly_codes[idx][fc_year])]

    m = GradientBoostingRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,min_samples_leaf=10,random_state=RANDOM_STATE)
    m.fit(Xt, yt)
    for tn, pdf in targets.items():
        pv = pdf.copy()
        for c in feats: lo,hi=tv[c].quantile(0.05),tv[c].quantile(0.95); pv[c+'_w']=pv[c].clip(lo,hi)
        pd2 = pv.dropna(subset=ncw)
        if len(pd2)==0: continue
        Xp = np.hstack([scl.transform(imp.transform(pd2[ncw].values)),
                        ohe.transform(pd2[['industry_l2']].fillna('Unknown'))])
        preds = m.predict(Xp); gaps = preds - pd2['EBI_A'].values
        for fid, ind, gap in zip(pd2['firm_id'], pd2['industry_l2'], gaps):
            records.append({'year': fc_year, 'target': tn, 'industry': ind, 'gap': gap})

        # CS subset
        if tn in ['All A-share SOE', 'CSI1000 SOE']:
            mask = cs_mask(train_pool, pdf)
            pdf_cs = pdf[mask]
            pv_cs = pdf_cs.copy()
            for c in feats: lo,hi=tv[c].quantile(0.05),tv[c].quantile(0.95); pv_cs[c+'_w']=pv_cs[c].clip(lo,hi)
            pd_cs = pv_cs.dropna(subset=ncw)
            if len(pd_cs)>0:
                Xp_cs = np.hstack([scl.transform(imp.transform(pd_cs[ncw].values)),
                                   ohe.transform(pd_cs[['industry_l2']].fillna('Unknown'))])
                preds_cs = m.predict(Xp_cs); gaps_cs = preds_cs - pd_cs['EBI_A'].values
                for fid, ind, gap in zip(pd_cs['firm_id'], pd_cs['industry_l2'], gaps_cs):
                    records_cs.append({'year': fc_year, 'target': tn, 'industry': ind, 'gap': gap})
    print(f'  {fc_year} done')

fr = pd.DataFrame(records)
fr_cs = pd.DataFrame(records_cs)
fr.to_csv(OUTPUT_DIR/'industry_year_firm_gaps_screened.csv', index=False, encoding='utf-8-sig')
fr_cs.to_csv(OUTPUT_DIR/'industry_year_cs_firm_gaps_screened.csv', index=False, encoding='utf-8-sig')
print('✓ firm-level CSVs saved')

# ============================================================================
# 3. Figure: full-sample heatmap (4 panels)
# ============================================================================
agg = fr.groupby(['target','year','industry']).agg(n=('gap','count'), mean_gap=('gap','mean')).reset_index()
targets_4 = ['All A-share SOE', 'CSI300 SOE', 'CSI500 SOE', 'CSI1000 SOE']
cmap = plt.cm.RdBu_r; vmax = 0.05

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('white')
for ti, target in enumerate(targets_4):
    ax = axes[ti//2][ti%2]; ax.set_facecolor('white')
    sub = agg[agg['target']==target]
    piv = sub.pivot_table(index='industry', columns='year', values='mean_gap', aggfunc='mean')
    n_ind = sub.groupby('industry')['n'].sum()
    keep = n_ind[n_ind>=20].index
    piv = piv.loc[piv.index.isin(keep)]
    piv['_m'] = piv.mean(axis=1); piv = piv.sort_values('_m', ascending=False).drop(columns='_m')
    years = [c for c in piv.columns]
    data = np.ma.masked_invalid(piv.values)
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(years))); ax.set_xticklabels([int(y) for y in years], fontsize=8)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(target, fontweight='bold')
    for i in range(len(piv.index)):
        for j in range(len(years)):
            v = piv.iloc[i,j]
            if not np.isnan(v):
                c='white' if abs(v)>0.025 else 'black'
                ax.text(j, i, f'{v*100:.1f}', ha='center', va='center', fontsize=6.5, color=c)
    ax.set_xlabel('Forecast Year')
    cbar=fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('GBM Gap (pp)', fontsize=8)
fig.suptitle('Industry-Level Counterfactual Gap by Year (GBM, SCREENED, 2019–2025)', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR/'fig_industry_year_heatmap_screened.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(OUTPUT_DIR/'fig_industry_year_heatmap_screened.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print('✓ fig_industry_year_heatmap_screened')

# ============================================================================
# 4. Figure: top/bottom trend (4 panels)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor('white')
colors_top = plt.cm.Reds(np.linspace(0.4, 0.9, 6))
colors_bot = plt.cm.Blues(np.linspace(0.4, 0.9, 6))
for ti, target in enumerate(targets_4):
    ax = axes[ti//2][ti%2]; ax.set_facecolor('white')
    sub = fr[fr['target']==target]
    piv = sub.pivot_table(index='industry', columns='year', values='gap', aggfunc='mean')
    n_ind = sub.groupby('industry')['gap'].count()
    keep = n_ind[n_ind>=20].index
    piv = piv.loc[piv.index.isin(keep)]
    piv['_m'] = piv.mean(axis=1); piv = piv.sort_values('_m', ascending=False)
    top6 = piv.head(6).drop(columns='_m'); bot6 = piv.tail(6).drop(columns='_m')
    years = [int(c) for c in top6.columns]
    for i,(ind,row) in enumerate(top6.iterrows()):
        ax.plot(years, row.values*100, 'o-', color=colors_top[i], linewidth=1.8, markersize=5, label=ind)
    for i,(ind,row) in enumerate(bot6.iterrows()):
        ax.plot(years, row.values*100, 's--', color=colors_bot[i], linewidth=1.5, markersize=5, label=ind)
    ax.axhline(0, color='#999', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xticks(years); ax.set_xlabel('Forecast Year'); ax.set_ylabel('GBM Gap (pp)')
    ax.set_title(target, fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='#CCC', facecolor='white', fontsize=6.5, ncol=2, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
fig.suptitle('Top/Bottom 6 Industries by Gap Over Time (GBM, SCREENED)', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR/'fig_industry_topbottom_trend_screened.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(OUTPUT_DIR/'fig_industry_topbottom_trend_screened.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print('✓ fig_industry_topbottom_trend_screened')

# ============================================================================
# 5. Figure: CS heatmap (2 panels: All SOE + CSI1000)
# ============================================================================
agg_cs = fr_cs.groupby(['target','year','industry']).agg(n=('gap','count'), mean_gap=('gap','mean')).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('white')
for ti, target in enumerate(['All A-share SOE', 'CSI1000 SOE']):
    ax = axes[ti]; ax.set_facecolor('white')
    sub = agg_cs[agg_cs['target']==target]
    piv = sub.pivot_table(index='industry', columns='year', values='mean_gap', aggfunc='mean')
    n_ind = sub.groupby('industry')['n'].sum()
    keep = n_ind[n_ind>=15].index
    piv = piv.loc[piv.index.isin(keep)]
    piv['_m'] = piv.mean(axis=1); piv = piv.sort_values('_m', ascending=False).drop(columns='_m')
    years = [c for c in piv.columns]
    data = np.ma.masked_invalid(piv.values)
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(years))); ax.set_xticklabels([int(y) for y in years], fontsize=8)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(f'{target} (Common Support)', fontweight='bold')
    for i in range(len(piv.index)):
        for j in range(len(years)):
            v = piv.iloc[i,j]
            if not np.isnan(v):
                c='white' if abs(v)>0.025 else 'black'
                ax.text(j, i, f'{v*100:.1f}', ha='center', va='center', fontsize=6.5, color=c)
    ax.set_xlabel('Forecast Year')
    cbar=fig.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('GBM Gap (pp)', fontsize=8)
fig.suptitle('Common-Support Industry Gap by Year (GBM, SCREENED)', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR/'fig_industry_year_cs_heatmap_screened.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(OUTPUT_DIR/'fig_industry_year_cs_heatmap_screened.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print('✓ fig_industry_year_cs_heatmap_screened')

# ============================================================================
# 6. Figure: CS top/bottom trend (2 panels)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
fig.patch.set_facecolor('white')
for ti, target in enumerate(['All A-share SOE', 'CSI1000 SOE']):
    ax = axes[ti]; ax.set_facecolor('white')
    sub = fr_cs[fr_cs['target']==target]
    piv = sub.pivot_table(index='industry', columns='year', values='gap', aggfunc='mean')
    n_ind = sub.groupby('industry')['gap'].count()
    keep = n_ind[n_ind>=15].index
    piv = piv.loc[piv.index.isin(keep)]
    piv['_m'] = piv.mean(axis=1); piv = piv.sort_values('_m', ascending=False)
    top6 = piv.head(6).drop(columns='_m'); bot6 = piv.tail(6).drop(columns='_m')
    years = [int(c) for c in top6.columns]
    for i,(ind,row) in enumerate(top6.iterrows()):
        ax.plot(years, row.values*100, 'o-', color=colors_top[i], linewidth=1.8, markersize=5, label=ind)
    for i,(ind,row) in enumerate(bot6.iterrows()):
        ax.plot(years, row.values*100, 's--', color=colors_bot[i], linewidth=1.5, markersize=5, label=ind)
    ax.axhline(0, color='#999', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xticks(years); ax.set_xlabel('Forecast Year'); ax.set_ylabel('GBM Gap (pp)')
    ax.set_title(f'{target} (Common Support)', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='#CCC', facecolor='white', fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
fig.suptitle('Common-Support Top/Bottom 6 Industries by Gap (GBM, SCREENED)', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR/'fig_industry_year_cs_trend_screened.png', dpi=300, facecolor='white', edgecolor='none')
fig.savefig(OUTPUT_DIR/'fig_industry_year_cs_trend_screened.pdf', facecolor='white', edgecolor='none')
plt.close(fig)
print('✓ fig_industry_year_cs_trend_screened')
print('\n✓ All screened industry figures complete')
