#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 2 — Private-firm OOF placebo + calibration.

Falsification test: apply the SAME model (GBM, GroupKFold by firm_id) to the
PRIVATE benchmark firms themselves (out-of-fold). If the model is unbiased,
the private OOF gap (pred - actual) should be centered on ~0. A large SOE gap
alongside a ~0 private placebo gap strengthens the interpretation that the SOE
gap is a real return differential, not a mechanical model artifact.

Calibration: regress actual = a + b * predicted; a well-calibrated model has
a ~ 0 and b ~ 1.

Outputs (outputs/revision2/):
  private_oof_placebo_summary.csv
  private_oof_placebo_by_year.csv
  private_oof_calibration.csv
  fig_private_oof_calibration.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit_common import load_screened_panel, feats_in, build, cluster_se, OUTPUT_DIR
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats as scipy_stats

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print('Loading screened panel...')
all_a = load_screened_panel()
feats = feats_in(all_a)
train = all_a[all_a['is_private']].copy()

# Same prep as production export_private_oof.py
from _audit_common import prepare_train
prep = prepare_train(train, feats)
tv = prep['tv']

X = prep['Xt']; y = prep['yt']; groups = tv['firm_id'].values
gkf = GroupKFold(n_splits=5)
model = build('gbm')
oof = cross_val_predict(model, X, y, cv=gkf.split(X, y, groups=groups))

# NOTE: y is winsorized EBI_A; residual computed against winsorized target,
# matching the model's training objective.
resid = oof - y
out = pd.DataFrame({
    'firm_id': tv['firm_id'].values,
    'year': tv['target_year'].values,
    'actual_ebia_ratio': tv['EBI_A'].values,      # un-winsorized actual (for calibration)
    'actual_ebia_w': y,                            # winsorized actual (training target)
    'oof_predicted': oof,
    'residual': resid,
})

# ============================================================================
# 1. Placebo summary (pooled)
# ============================================================================
def ci_of(gaps, fids):
    se, nf = cluster_se(gaps, fids)
    m = float(np.mean(gaps))
    lo = m - 1.96 * se if not np.isnan(se) else np.nan
    hi = m + 1.96 * se if not np.isnan(se) else np.nan
    t = m / se if se and se > 0 else np.nan
    p = 2 * (1 - scipy_stats.t.cdf(abs(t), nf - 1)) if not np.isnan(t) and nf > 1 else np.nan
    return m, se, lo, hi, nf, p

m, se, lo, hi, nf, p = ci_of(resid, out['firm_id'].values)
summary = pd.DataFrame([{
    'group': 'Private OOF placebo',
    'n_obs': len(out), 'n_firms': nf,
    'mean_residual': m, 'median_residual': float(np.median(resid)),
    'clustered_se': se, 'ci_95_lower': lo, 'ci_95_upper': hi, 'p_value': p,
    'oof_r2': r2_score(y, oof), 'oof_rmse': np.sqrt(mean_squared_error(y, oof)),
    'oof_mae': mean_absolute_error(y, oof),
}])
summary.to_csv(OUTPUT_DIR / 'private_oof_placebo_summary.csv', index=False, encoding='utf-8-sig')
print('\n=== Placebo summary (private OOF) ===')
print(summary.round(4).to_string(index=False))

# ============================================================================
# 2. Placebo by year
# ============================================================================
by_year = []
for yr in sorted(out['year'].unique()):
    sub = out[out['year'] == yr]
    mm, sse, llo, hhi, nnf, pp = ci_of(sub['residual'].values, sub['firm_id'].values)
    by_year.append({'year': int(yr), 'n_obs': len(sub), 'n_firms': nnf,
                    'mean_residual': mm, 'clustered_se': sse,
                    'ci_95_lower': llo, 'ci_95_upper': hhi})
by_year = pd.DataFrame(by_year)
by_year.to_csv(OUTPUT_DIR / 'private_oof_placebo_by_year.csv', index=False, encoding='utf-8-sig')
print('\n=== Placebo by year ===')
print(by_year.round(4).to_string(index=False))

# ============================================================================
# 3. Calibration: actual = a + b * predicted
# ============================================================================
def calib(df, label):
    x = np.column_stack([np.ones(len(df)), df['oof_predicted'].values])
    b = np.linalg.lstsq(x, df['actual_ebia_ratio'].values, rcond=None)[0]
    pred_lin = x @ b
    ss_res = np.sum((df['actual_ebia_ratio'].values - pred_lin) ** 2)
    ss_tot = np.sum((df['actual_ebia_ratio'].values - df['actual_ebia_ratio'].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'scope': label, 'intercept_a': b[0], 'slope_b': b[1], 'r2': r2, 'n': len(df)}

cal_rows = [calib(out, 'pooled')]
for yr in sorted(out['year'].unique()):
    cal_rows.append(calib(out[out['year'] == yr], f'year_{yr}'))
cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(OUTPUT_DIR / 'private_oof_calibration.csv', index=False, encoding='utf-8-sig')
print('\n=== Calibration (actual = a + b*predicted) ===')
print(cal_df.round(4).to_string(index=False))

# ============================================================================
# 4. Calibration figure
# ============================================================================
fig, ax = plt.subplots(figsize=(6.5, 6))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
a_ = out['actual_ebia_ratio'] * 100
p_ = out['oof_predicted'] * 100
ax.scatter(p_, a_, s=6, alpha=0.25, color='#2980B9', edgecolor='none', label='firm-year')
lo = min(a_.min(), p_.min()); hi = max(a_.max(), p_.max())
ax.plot([lo, hi], [lo, hi], '--', color='#E74C3C', linewidth=1.2, label='45° line')
# OLS line
xlin = np.linspace(lo, hi, 50)
b0, b1 = cal_df.iloc[0]['intercept_a'], cal_df.iloc[0]['slope_b']
ax.plot(xlin, (b0 + b1 * (xlin / 100)) * 100, '-', color='#2C3E50', linewidth=1.8,
        label=f'OLS fit: y = {b0:+.3f} + {b1:.3f}x')
ax.set_xlabel('OOF predicted EBI/A (%)')
ax.set_ylabel('Actual EBI/A (%)')
ax.set_title('Private-firm OOF calibration (GBM, GroupKFold)', fontweight='bold')
ax.legend(frameon=False, fontsize=8.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_private_oof_calibration.png', dpi=300, facecolor='white')
plt.close(fig)
print('\n✓ fig_private_oof_calibration.png saved')

print('\n✓ audit_02_placebo.py complete')
