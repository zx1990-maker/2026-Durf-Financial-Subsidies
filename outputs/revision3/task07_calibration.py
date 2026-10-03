#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK 7 (revision3) — Calibration definition audit.

The paper reports two GBM calibration slopes:
  1.04  (winsorized target)  — §4 Table 2
  1.29  (raw target)         — §6.2

This script reproduces the private-firm OOF predictions (GBM, GroupKFold by firm_id,
winsorized P5/P95 target) and re-estimates BOTH calibration regressions to document
exactly which y each slope uses, the intercept, R^2 and n, and whether they are
directly comparable.

Outputs (outputs/revision3/):
  calibration_definition_audit.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit3 import OUT3
from _audit_common import load_screened_panel, feats_in, build, prepare_train
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

OUT3.mkdir(exist_ok=True)

all_a = load_screened_panel()
feats = feats_in(all_a)
train = all_a[all_a['is_private']].copy()

prep = prepare_train(train, feats)
tv = prep['tv']
X = prep['Xt']
y_w = prep['yt']                                   # winsorized target (training objective)
y_raw = tv['EBI_A'].values                         # raw target
groups = tv['firm_id'].values

gkf = GroupKFold(n_splits=5)
model = build('gbm')
oof = cross_val_predict(model, X, y_w, cv=gkf.split(X, y_w, groups=groups))


def ols(x, y):
    Xc = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(Xc, y, rcond=None)[0]
    pred = Xc @ b
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return b[0], b[1], r2


a_w, b_w, r2_w = ols(oof, y_w)       # winsorized-target calibration (slope ~1.04)
a_r, b_r, r2_r = ols(oof, y_raw)     # raw-target calibration (slope ~1.29)

n = len(y_raw)

print(f'Winsorized-target calibration: alpha={a_w:+.4f}, beta={b_w:.4f}, R2={r2_w:.4f}, n={n}')
print(f'Raw-target calibration:        alpha={a_r:+.4f}, beta={b_r:.4f}, R2={r2_r:.4f}, n={n}')

lines = ['# TASK 7 — Calibration 表述审计\n']

lines.append('## 1. 两个斜率分别使用哪个 y\n')
lines.append('两处斜率都来自**同一组**民营 OOF 预测值（GBM，GroupKFold by firm_id，5 折，随机种子 42，'
             '训练目标为 P5–P95 winsorize 后的 EBI/A），区别仅在于被回归的**左侧变量（实际值）**：\n')
lines.append('| 斜率 | 左侧变量 y | 右侧变量 x | 截距 α | R² | n |')
lines.append('|---|---|---|---|---|---|')
lines.append(f'| **{b_w:.2f}** | winsorized 目标 `EBI_A_w`（训练目标） | OOF 预测值 | {a_w:+.4f} | {r2_w:.4f} | {n} |')
lines.append(f'| **{b_r:.2f}** | 原始（未缩尾）实际值 `EBI_A` | OOF 预测值 | {a_r:+.4f} | {r2_r:.4f} | {n} |')

lines.append('\n## 2. 两个 regression 的 alpha\n')
lines.append(f'- winsorized 口径：α = {a_w:+.4f}\n- raw 口径：α = {a_r:+.4f}\n')

lines.append('\n## 3. 样本量\n')
lines.append(f'- 两个回归均为民营 firm-year 样本，n = {n}（同一 OOF 样本，pooled）。\n')

lines.append('\n## 4. 是否可以直接比较\n')
lines.append(
    '**否。** 两个斜率不是同一回归：1.04 是对 winsorize 后目标回归，1.29 是对原始（未缩尾）实际值回归。'
    '二者不可直接比较，也不应被混用为一个“校准斜率”。论文 §4 表 2（β=1.04）与 §6.2（slope 1.29）分别对应上述两种口径。\n'
)

lines.append('\n## 5. 关于“保守下界”的结论\n')
lines.append(
    '论文 §6.2 有表述“斜率大于 1 表明……报告中的 gap 是校准前的保守下界”。'
    '该结论**未被代码严格证明**，理由：\n'
)
lines.append(
    '- 校准回归（β>1 的 shrinkage）是在**民营**训练样本上（in-sample OOF）估计的，描述的是模型对民营分布的压缩；\n'
    '- SOE gap = f̂(X_SOE) − Y_SOE 是对**另一个群体**的样本外预测，shrinkage 究竟使 gap 被高估还是低估，'
    '取决于 SOE 相对民营条件均值的位置与进入 SOE 的选择机制，代码中没有形式化证明；\n'
    '- 因此不能自动得出“gap 是保守下界”。\n'
)
lines.append('\n> Do not interpret calibration slope > 1 as a guaranteed lower bound on SOE gap.\n')

(OUT3 / 'calibration_definition_audit.md').write_text('\n'.join(lines), encoding='utf-8')
print(f'\n✓ {OUT3 / "calibration_definition_audit.md"}')
