#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 5 (revision3) — Industry-restricted KNN comparable-firm robustness.

AUDIT RESULT: the existing KNN (outputs/revision2/audit_05_knn.py) is NOT strictly
industry-restricted. It fits NearestNeighbors on the GBM feature matrix Xt, which is
the scaled 9 numeric features CONCATENATED with the industry_l2 one-hot. The one-hot
only adds a distance penalty for a different industry; it does NOT forbid cross-industry
neighbours. See knn_code_audit.md.

This script therefore adds a STRICT industry-restricted KNN:
  - candidate private firms = SAME industry_l2 only, same ex-ante window (target_year < fc_year);
  - distance on the 9 numeric features, standardized on the PRIVATE training sample only;
  - benchmark = mean actual EBI/A of the k nearest same-industry private firms;
  - if N < k -> use all available (record actual_k); if N = 0 -> benchmark missing (no cross-industry).

Outputs (outputs/revision3/):
  industry_restricted_knn_summary.csv
  industry_restricted_knn_firm_level.csv
  knn_code_audit.md
  industry_restricted_knn.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3, compute_long_panel, sample_mask
from _audit_common import load_screened_panel, feats_in, build, prepare_train, predict_gap
from sklearn.neighbors import NearestNeighbors

OUT3.mkdir(exist_ok=True)

all_a = load_screened_panel()
feats = feats_in(all_a)
train_pool = all_a[all_a['is_private']].copy()
ncw = [c + '_w' for c in feats]

SAMPLES = ['All A-share SOE', 'CSI300', 'CSI500', 'CSI1000']


def industry_knn_bench(priv_num, priv_actual, priv_ind, soe_num, soe_ind, k):
    """Mean actual EBI/A of the k nearest SAME-industry private firms (NaN if none)."""
    bench = np.full(len(soe_num), np.nan)
    actual_k = np.zeros(len(soe_num), dtype=int)
    for i in range(len(soe_num)):
        mask = priv_ind == soe_ind[i]
        if mask.sum() == 0:
            continue
        pn = priv_num[mask]
        pa = priv_actual[mask]
        d = np.linalg.norm(pn - soe_num[i], axis=1)
        kk = int(min(k, len(pa)))
        nn = np.argsort(d)[:kk]
        bench[i] = pa[nn].mean()
        actual_k[i] = kk
    return bench, actual_k


firm_rows = []
for fc_year in range(2019, 2026):
    tr = train_pool[train_pool['target_year'] < fc_year]
    prep = prepare_train(tr, feats)
    if len(prep['tv']) < 50:
        continue

    tv = prep['tv']
    priv_actual = tv['EBI_A'].values                     # raw (un-winsorized) actual
    priv_ind = tv['industry_l2'].values
    P_num = prep['scl'].transform(prep['imp'].transform(tv[ncw].values))   # scaled 9 numeric
    Xt = prep['Xt']                                      # numeric + industry one-hot

    m = build('gbm'); m.fit(prep['Xt'], prep['yt'])
    soe = all_a[(all_a['is_soe']) & (all_a['target_year'] == fc_year)].copy()
    preds, gaps_gbm, pd2, _ = predict_gap(prep, m, soe)
    if pd2 is None or len(pd2) == 0:
        continue

    pd2 = pd2.copy()
    soe_actual = pd2['EBI_A'].values
    soe_ind = pd2['industry_l2'].values
    S_num = prep['scl'].transform(prep['imp'].transform(pd2[ncw].values))
    Xp = np.hstack([S_num, prep['ohe'].transform(pd2[['industry_l2']].fillna('Unknown'))])

    # unrestricted KNN5 (cross-industry, on numeric + one-hot)
    nn5 = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(Xt)
    _, idx5 = nn5.kneighbors(Xp)
    gap_knn5_unr = priv_actual[idx5].mean(axis=1) - soe_actual

    for k in [5, 10]:
        bench, actual_k = industry_knn_bench(P_num, priv_actual, priv_ind, S_num, soe_ind, k)
        pd2[f'gap_knn{k}_ind'] = bench - soe_actual
        pd2[f'actual_k{k}'] = actual_k

    pd2['gap_gbm'] = gaps_gbm
    pd2['gap_knn5_unr'] = gap_knn5_unr

    for _, r in pd2.iterrows():
        firm_rows.append({
            'year': fc_year, 'firm_id': r['firm_id'], 'industry_l2': r['industry_l2'],
            'gap_gbm': r['gap_gbm'], 'gap_knn5_unrestricted': r['gap_knn5_unr'],
            'gap_knn5_industry': r['gap_knn5_ind'], 'gap_knn10_industry': r['gap_knn10_ind'],
            'actual_k5': r['actual_k5'], 'actual_k10': r['actual_k10'],
        })

firm = pd.DataFrame(firm_rows)
firm['firm_id'] = firm['firm_id'].astype(str)

# attach sample flags from the master long panel
long = compute_long_panel()
flag = long[['year', 'firm_id', 'in_csi300', 'in_csi500', 'in_csi1000']].drop_duplicates()
firm = firm.merge(flag, on=['year', 'firm_id'], how='left')

firm.to_csv(OUT3 / 'industry_restricted_knn_firm_level.csv', index=False, encoding='utf-8-sig')

# ---- summary per (year, sample) ----
summary_rows = []
for year in range(2019, 2026):
    for sample in SAMPLES:
        s = firm[firm['year'] == year]
        s = s[sample_mask(s, sample)]
        if len(s) == 0:
            continue
        valid = s[s['gap_knn5_industry'].notna()]
        n_valid = len(valid)
        corr = valid['gap_gbm'].corr(valid['gap_knn5_industry']) if n_valid >= 3 else np.nan
        direction = np.mean(np.sign(valid['gap_gbm']) == np.sign(valid['gap_knn5_industry'])) if n_valid else np.nan
        summary_rows.append({
            'year': year, 'sample': sample,
            'GBM_gap': float(s['gap_gbm'].mean()),
            'unrestricted_KNN5_gap': float(s['gap_knn5_unrestricted'].mean()),
            'industry_KNN5_gap': float(valid['gap_knn5_industry'].mean()) if n_valid else np.nan,
            'industry_KNN10_gap': float(s[s['gap_knn10_industry'].notna()]['gap_knn10_industry'].mean()) if s['gap_knn10_industry'].notna().any() else np.nan,
            'corr_GBM_industryKNN5': float(corr) if not np.isnan(corr) else np.nan,
            'direction_agreement': float(direction) if not np.isnan(direction) else np.nan,
            'n_valid': n_valid,
            'n_total': len(s),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT3 / 'industry_restricted_knn_summary.csv', index=False, encoding='utf-8-sig')
print('\n=== industry_restricted_knn_summary ===')
print(summary.round(4).to_string(index=False))

# ---- knn_code_audit.md ----
audit_md = [
    '# TASK 5 — KNN 代码审计\n',
    '## 现有 KNN 是否严格限制同行业？\n',
    '**否。** 现有 KNN（`outputs/revision2/audit_05_knn.py`）**不**严格限制同行业。\n',
    '- **文件**：`outputs/revision2/audit_05_knn.py`\n',
    '- **函数**：主循环中 `NearestNeighbors(n_neighbors=k).fit(Xt)`，其中 `Xt = prep[\'Xt\']`\n',
    '- **行业限制条件**：无硬约束。`Xt` 是「9 个 winsorize+标准化数值特征」与「industry_l2 one-hot」拼接后的矩阵；one-hot 只作为距离的一个分量（不同行业会增加欧氏距离），但**不禁止**跨行业邻居。\n',
    '- **actual_k 处理**：现有代码**不**记录 `actual_k`，也**不**做「同行业不足 k 家」的处理——它始终取全样本的 k 近邻（可跨行业）。\n',
    '\n## 结论与处理\n',
    '现有 KNN 是「全市场（可跨行业）可比企业」benchmark，不是「同行业可比企业」benchmark。',
    '因此本任务新增**严格同行业 KNN**（`task05_industry_knn.py`）：候选民企必须 `industry_l2` 相同，',
    '距离用主模型同一套 9 个数值特征（scaler 只由民营训练样本拟合），',
    '同行业民企不足 k 家用全部（记录 `actual_k`），同行业无民企则 benchmark 缺失且不跨行业。\n',
    '\n## 新增输出\n',
    '- `industry_restricted_knn_summary.csv`（year × sample）\n',
    '- `industry_restricted_knn_firm_level.csv`（firm-year，含 `actual_k5`/`actual_k10`）\n',
]
(OUT3 / 'knn_code_audit.md').write_text('\n'.join(audit_md), encoding='utf-8')

# ---- summary md ----
lines = ['# TASK 5 — 同行业内 KNN Comparable-Firm Robustness\n']
lines.append('现有 KNN 不严格限制同行业（见 `knn_code_audit.md`），故新增严格同行业 KNN。\n')
lines.append('| Year | Sample | GBM gap | Unrestricted KNN5 | Industry KNN5 | Industry KNN10 | corr(GBM, IndKNN5) | Direction agree | n_valid |')
lines.append('|---|---|---|---|---|---|---|---|---|')
for _, r in summary.iterrows():
    lines.append(
        f"| {int(r['year'])} | {r['sample']} | {r['GBM_gap']:.4f} | {r['unrestricted_KNN5_gap']:.4f} | "
        f"{r['industry_KNN5_gap']:.4f} | {r['industry_KNN10_gap']:.4f} | "
        f"{r['corr_GBM_industryKNN5']:.3f} | {r['direction_agreement']:.3f} | {int(r['n_valid'])} |"
    )
(OUT3 / 'industry_restricted_knn.md').write_text('\n'.join(lines), encoding='utf-8')

print(f'\n✓ {OUT3 / "industry_restricted_knn_summary.csv"}')
print(f'✓ {OUT3 / "industry_restricted_knn_firm_level.csv"}')
print(f'✓ {OUT3 / "knn_code_audit.md"}')
print(f'✓ {OUT3 / "industry_restricted_knn.md"}')
