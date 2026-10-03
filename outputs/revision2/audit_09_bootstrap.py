#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 9 — GBM cluster bootstrap (200 draws) for the 2025 gap.

Firm-level (cluster) bootstrap: resample FIRMS with replacement 200 times and
recompute the mean gap, to obtain a non-parametric standard error and percentile
CI for the 2025 gap. Two samples: All A-share SOE and CSI1000 SOE.

Outputs (outputs/revision2/):
  gbm_cluster_bootstrap_selected_samples.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit_common import (
    load_screened_panel, load_yearly_codes, feats_in, build, cluster_se,
    prepare_train, predict_gap, OUTPUT_DIR,
)

RNG_SEED = 20250816
N_DRAWS = 200

print('Loading screened panel...')
all_a = load_screened_panel()
yearly_codes = load_yearly_codes()
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()

FC_YEAR = 2025
tr = train_pool[train_pool['target_year'] < FC_YEAR]
prep = prepare_train(tr, feats)
m = build('gbm'); m.fit(prep['Xt'], prep['yt'])

soe = all_a[(all_a['is_soe']) & (all_a['target_year'] == FC_YEAR)].copy()
_, gaps, pd2, _ = predict_gap(prep, m, soe)
pd2 = pd2.copy()
pd2['gap'] = gaps

csi1000 = set(yearly_codes['CSI1000'][FC_YEAR])
pd2['in_csi1000'] = pd2['firm_id'].astype(str).isin(csi1000)

rng = np.random.default_rng(RNG_SEED)

rows = []
for label, sub in [('All A-share SOE', pd2), ('CSI1000 SOE', pd2[pd2['in_csi1000']])]:
    gaps_all = sub['gap'].values
    firms = sub['firm_id'].values
    uf = np.unique(firms)
    firm_gaps = np.array([gaps_all[firms == f].mean() for f in uf])

    # analytic cluster SE for comparison
    a_se, a_nf = cluster_se(gaps_all, firms)

    # cluster bootstrap: resample firms (with replacement), mean of firm-level means
    boot_means = np.empty(N_DRAWS)
    for b in range(N_DRAWS):
        idx = rng.integers(0, len(uf), size=len(uf))
        boot_means[b] = firm_gaps[idx].mean()

    boot_mean = boot_means.mean()
    boot_se = boot_means.std(ddof=1)
    ci_lo = np.quantile(boot_means, 0.025)
    ci_hi = np.quantile(boot_means, 0.975)

    rows.append({
        'sample': label, 'year': FC_YEAR,
        'n_obs': len(sub), 'n_firms': len(uf),
        'mean_gap': float(np.mean(gaps_all)),
        'analytic_cluster_se': float(a_se) if not np.isnan(a_se) else np.nan,
        'bootstrap_mean': float(boot_mean),
        'bootstrap_se': float(boot_se),
        'bootstrap_ci_95_lower': float(ci_lo),
        'bootstrap_ci_95_upper': float(ci_hi),
        'n_draws': N_DRAWS,
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / 'gbm_cluster_bootstrap_selected_samples.csv', index=False, encoding='utf-8-sig')

print('\n=== 2025 GBM gap: cluster bootstrap (200 firm-resampling draws) ===')
for _, r in df.iterrows():
    print(f"\n{r['sample']}: n={r['n_obs']} obs, {r['n_firms']} firms")
    print(f"  mean gap        = {r['mean_gap']:.4f}")
    print(f"  analytic SE     = {r['analytic_cluster_se']:.4f}")
    print(f"  bootstrap SE    = {r['bootstrap_se']:.4f}")
    print(f"  bootstrap 95% CI= [{r['bootstrap_ci_95_lower']:.4f}, {r['bootstrap_ci_95_upper']:.4f}]")
    print(f"  CI excludes 0   = {'YES (p<0.05)' if r['bootstrap_ci_95_lower'] > 0 else 'NO'}")

print('\n✓ audit_09_bootstrap.py complete')
