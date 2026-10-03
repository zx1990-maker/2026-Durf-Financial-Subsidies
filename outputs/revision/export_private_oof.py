#!/usr/bin/env python3
"""Private-firm OOF predictions (firm-level), GBM, GroupKFold by firm_id."""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / "final_results"

sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE=42

print('Loading...')
all_a = load_all_a_shares(); all_a = clean_and_filter(all_a, 'A')
# 筛选（与最终一致）
st_firms = set(all_a[all_a['firm_name'].str.contains('ST', na=False)]['firm_id'].unique())
all_a = all_a[~all_a['firm_id'].isin(st_firms)]
all_a = all_a[~(all_a['FinancialLeverage'] > 1.0)]
all_a = all_a[all_a['EBI_A'].abs() <= 0.5]

train = all_a[all_a['ownership']=='民营企业'].copy()
feats = NUMERIC_FEATURES
tv = train.dropna(subset=feats+['EBI_A']).copy()

# Winsorize（基于全训练集，与之前报告一致）
for c in feats:
    lo,hi = tv[c].quantile(0.05), tv[c].quantile(0.95); tv[c+'_w'] = tv[c].clip(lo,hi)
yl,yh = tv['EBI_A'].quantile(0.05), tv['EBI_A'].quantile(0.95); tv['EBI_A_w'] = tv['EBI_A'].clip(yl,yh)
ncw = [c+'_w' for c in feats]

imp = SimpleImputer(strategy='median'); scl = StandardScaler()
X = scl.fit_transform(imp.fit_transform(tv[ncw].values))
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X = np.hstack([X, ohe.fit_transform(tv[['industry_l2']].fillna('Unknown'))])
y = tv['EBI_A_w'].values
groups = tv['firm_id'].values

# GroupKFold by firm_id, 5 folds
gkf = GroupKFold(n_splits=5)
model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  min_samples_leaf=10, random_state=RANDOM_STATE)
oof = cross_val_predict(model, X, y, cv=gkf.split(X, y, groups=groups))

# 输出 firm-level
out = tv[['firm_id','firm_name','target_year','industry_l2','industry_l1','EBI_A']].copy()
out['year'] = out['target_year']
out['actual_ebia_ratio'] = out['EBI_A']
out['oof_predicted_ebia_ratio'] = oof
out['residual'] = oof - out['EBI_A'].values
out = out.drop(columns=['target_year','EBI_A'])
out = out[['firm_id','firm_name','year','industry_l1','industry_l2',
           'actual_ebia_ratio','oof_predicted_ebia_ratio','residual']]

out.to_csv(OUTPUT_DIR/'private_oof_predictions.csv', index=False, encoding='utf-8-sig')
print(f'✓ private_oof_predictions.csv ({len(out)} rows)')

# 汇总指标
print('\n=== OOF 指标（GBM, 民企, GroupKFold by firm_id）===')
print(f'n = {len(out)} obs, {out["firm_id"].nunique()} firms')
print(f'OOF R² = {r2_score(y, oof):+.4f}')
print(f'OOF RMSE = {np.sqrt(mean_squared_error(y, oof)):.4f}')
print(f'OOF MAE = {mean_absolute_error(y, oof):.4f}')
print(f'OOF ME = {np.mean(oof - y):+.4f}')
# calibration
Xc = np.column_stack([np.ones(len(y)), oof])
b = np.linalg.lstsq(Xc, y, rcond=None)[0]
print(f'Calibration: actual = {b[0]:+.4f} + {b[1]:.4f} × predicted')

# firm-level 汇总（每个 firm 的均值）
firm_mean = out.groupby('firm_id').agg(
    firm_name=('firm_name','first'),
    n_years=('year','count'),
    mean_actual=('actual_ebia_ratio','mean'),
    mean_predicted=('oof_predicted_ebia_ratio','mean'),
    mean_residual=('residual','mean'),
).reset_index()
firm_mean.to_csv(OUTPUT_DIR/'private_oof_firm_mean.csv', index=False, encoding='utf-8-sig')
print(f'✓ private_oof_firm_mean.csv ({len(firm_mean)} firms)')
