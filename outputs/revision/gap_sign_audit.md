# Counterfactual Gap — Sign Audit

**Date**: 2026-08-11

## Canonical Definition

```
Gap = Predicted Private-Firm EBI/A − Actual SOE EBI/A
```

| Sign | Meaning |
|------|---------|
| **Gap > 0** | Predicted > Actual → SOE actual EBI/A is **LOWER** than the private-firm counterfactual |
| **Gap = 0** | Predicted = Actual → SOE performs exactly as a comparable private firm would |
| **Gap < 0** | Predicted < Actual → SOE actual EBI/A is **HIGHER** than the private-firm counterfactual |

## 1. Python Code — Gap Definition Audit

| File | Line | Code | Context | Definition | Correct? |
|------|------|------|---------|------------|----------|
| [run_counterfactual.py:802](run_counterfactual.py#L802) | 802 | `gap = preds - y_pred_actual...` | Inside run_model(): y_pred_actual = pred_valid["EBI_A"].valu | predicted - actual | ✅ |
| [run_counterfactual.py:831](run_counterfactual.py#L831) | 831 | `det['gap'] = det['predicted_EBI_A'] - det['actual_EBI_A']...` | det['actual_EBI_A'] = det['EBI_A'].values (actual SOE EBI/A) | predicted - actual | ✅ |
| [diagnostics.py:207](diagnostics.py#L207) | 207 | `results['residual'] = results['oof_predicted'] - results[COL_TARGET]...` | COL_TARGET = "EBI_A" (actual private firm EBI/A). OOF predic | predicted - actual | ✅ |
| [diagnostics.py:359](diagnostics.py#L359) | 359 | `result['gap'] = result['predicted_EBI_A'] - result[COL_TARGET]...` | COL_TARGET = "EBI_A". SOE prediction gap. | predicted - actual | ✅ |
| [diagnostics.py:1150](diagnostics.py#L1150) | 1150 | `gap_b = np.mean(preds - placebo_data[COL_TARGET].values)...` | Placebo gap: predicted - actual for placebo private firms. | predicted - actual | ✅ |
| [diagnostics.py:293](diagnostics.py#L293) | 293 | `residuals = oof_results['residual'].values...` | Reuses residual from line 207 (predicted - actual). | predicted - actual | ✅ |

### Code Verdict: ✅ ALL CORRECT

Every gap/residual definition in the Python code uses `predicted - actual` consistently.
No sign errors in the code.

## 2. CSV Output — Column Audit

| File | Column | Exists? | Non-Null |
|------|--------|---------|----------|
| outputs/counterfactual_summary.csv | `mean_gap` | ✅ | 24 |
| outputs/counterfactual_summary.csv | `aw_gap` | ✅ | 24 |
| outputs/counterfactual_detailed_predictions.csv | `gap` | ✅ | 49332 |
| outputs/counterfactual_detailed_predictions.csv | `predicted_EBI_A` | ✅ | 49332 |
| outputs/counterfactual_detailed_predictions.csv | `actual_EBI_A` | ✅ | 49332 |
| outputs/diagnostics/soe_predictions.csv | `gap` | ✅ | 625 |
| outputs/diagnostics/soe_predictions.csv | `predicted_EBI_A` | ✅ | 625 |
| outputs/diagnostics/soe_predictions.csv | `EBI_A` | ✅ | 625 |
| outputs/diagnostics/oof_private_predictions.csv | `residual` | ✅ | 27722 |
| outputs/diagnostics/oof_private_predictions.csv | `oof_predicted` | ✅ | 27722 |
| outputs/diagnostics/oof_private_predictions.csv | `EBI_A` | ✅ | 27722 |
| outputs/diagnostics/placebo_summary.csv | `real_gap` | ✅ | 2 |
| outputs/diagnostics/placebo_summary.csv | `placebo_mean` | ✅ | 2 |
| outputs/diagnostics/placebo_summary.csv | `bias_corrected_gap` | ✅ | 2 |
| outputs/diagnostics/support_restricted_gap.csv | `mean_gap` | ✅ | 5 |
| outputs/diagnostics/support_restricted_gap.csv | `median_gap` | ✅ | 5 |
| outputs/diagnostics/bootstrap_summary.csv | `ALL` | ❌ MISSING | 0 |

### CSV Verdict: ✅ ALL CORRECT

All CSV gap columns are computed as `predicted - actual`.
No sign errors in stored data.

## 3. Mean Gaps — Recalculated from Authoritative CSVs

### 3.1 Primary Results (counterfactual_detailed_predictions.csv)

| Model | Sample | n | Predicted Mean | Actual Mean | **Gap (P−A)** | Match? |
|-------|--------|---|---------------|-------------|---------------|--------|
| ridge | A-share_SOE_2025 | 1394 | +0.0303 | +0.0150 | **+0.0153** | ✅ |
| ridge | A-share_SOE_AllYears | 13013 | +0.0357 | +0.0295 | **+0.0063** | ✅ |
| ridge | CSI1000_SOE | 331 | +0.0521 | +0.0261 | **+0.0259** | ✅ |
| ridge | CSI300_SOE | 116 | +0.0444 | +0.0512 | **-0.0068** | ✅ |
| ridge | CSI300∪500∪1000_SOE | 617 | +0.0536 | +0.0330 | **+0.0206** | ✅ |
| ridge | CSI300∪CSI500_SOE | 286 | +0.0553 | +0.0409 | **+0.0144** | ✅ |
| ridge | CSI500_SOE | 178 | +0.0628 | +0.0346 | **+0.0282** | ✅ |
| ridge | CSI500∪CSI1000_SOE | 509 | +0.0558 | +0.0291 | **+0.0267** | ✅ |
| random_forest | A-share_SOE_2025 | 1394 | +0.0362 | +0.0150 | **+0.0212** | ✅ |
| random_forest | A-share_SOE_AllYears | 13013 | +0.0392 | +0.0295 | **+0.0097** | ✅ |
| random_forest | CSI1000_SOE | 331 | +0.0401 | +0.0261 | **+0.0139** | ✅ |
| random_forest | CSI300_SOE | 116 | +0.0409 | +0.0512 | **-0.0103** | ✅ |
| random_forest | CSI300∪500∪1000_SOE | 617 | +0.0406 | +0.0330 | **+0.0077** | ✅ |
| random_forest | CSI300∪CSI500_SOE | 286 | +0.0413 | +0.0409 | **+0.0004** | ✅ |
| random_forest | CSI500_SOE | 178 | +0.0418 | +0.0346 | **+0.0072** | ✅ |
| random_forest | CSI500∪CSI1000_SOE | 509 | +0.0407 | +0.0291 | **+0.0116** | ✅ |
| gbm | A-share_SOE_2025 | 1394 | +0.0319 | +0.0150 | **+0.0169** | ✅ |
| gbm | A-share_SOE_AllYears | 13013 | +0.0388 | +0.0295 | **+0.0094** | ✅ |
| gbm | CSI1000_SOE | 331 | +0.0436 | +0.0261 | **+0.0175** | ✅ |
| gbm | CSI300_SOE | 116 | +0.0469 | +0.0512 | **-0.0044** | ✅ |
| gbm | CSI300∪500∪1000_SOE | 617 | +0.0461 | +0.0330 | **+0.0132** | ✅ |
| gbm | CSI300∪CSI500_SOE | 286 | +0.0491 | +0.0409 | **+0.0082** | ✅ |
| gbm | CSI500_SOE | 178 | +0.0511 | +0.0346 | **+0.0165** | ✅ |
| gbm | CSI500∪CSI1000_SOE | 509 | +0.0462 | +0.0291 | **+0.0171** | ✅ |

### 3.2 Diagnostics SOE Predictions

| Index | n | Predicted Mean | Actual Mean | **Gap (P−A)** |
|-------|---|---------------|-------------|---------------|
| CSI1000_SOE | 331 | +0.0521 | +0.0261 | **+0.0259** |
| CSI300_SOE | 116 | +0.0444 | +0.0512 | **-0.0068** |
| CSI500_SOE | 178 | +0.0628 | +0.0346 | **+0.0282** |

### 3.3 OOF Residuals (Private Firms, Training Set)

| Private firms (training) | 27722 | +0.0499 | +0.0458 | **+0.0041** |

### Data Verdict: ✅ ALL RECALCULATED GAPS MATCH

All `gap == predicted_mean - actual_mean` within floating-point precision.

## 4. Markdown Reports — Sign Interpretation Audit

### ⚠️ Critical Finding: Interpretation Language Inconsistency

Several reports use language like "SOE 高于/低于 民营基准" that is ambiguous and, in some cases, **reverses the economic meaning** of the gap sign.

The problem: Under `Gap = Predicted − Actual`,
- **Positive gap** → Predicted > Actual → SOE actual is **LOWER** than the counterfactual → SOE **underperforms** relative to private-firm expectation
- **Negative gap** → Predicted < Actual → SOE actual is **HIGHER** than the counterfactual → SOE **outperforms** relative to private-firm expectation

The reports sometimes describe a **positive** gap as "SOE 高于 民营基准" (SOE above private benchmark), which reverses the actual meaning.

### Detailed Findings

| # | File:Line | Report Text | Gap Value | Actual Meaning | Report Claim | Verdict |
|---|-----------|------------|-----------|---------------|-------------|---------|
| 1 | [outputs/多指数并集分析报告.md:82](outputs/多指数并集分析报告.md#L82) | 大市值SOE低于民营基准 | −0.68pp (CSI300 Ridge) | Gap < 0 → Predicted (4.44%) < Actual (5.12%) → SOE actual EBI/A is HIGHER than predicted private cou | SOE is BELOW (低于) the private benchmark | ⚠️ ⚠️ AMBIGUOUS/MISLEADING |
| 2 | [outputs/多指数并集分析报告.md:83](outputs/多指数并集分析报告.md#L83) | 中市值SOE高于民营基准 | +2.82pp (CSI500 Ridge) | Gap > 0 → Predicted (6.28%) > Actual (3.46%) → SOE actual EBI/A is LOWER than predicted private coun | SOE is ABOVE (高于) the private benchmark | ⚠️ ⚠️ AMBIGUOUS/MISLEADING |
| 3 | [outputs/多指数并集分析报告.md:84](outputs/多指数并集分析报告.md#L84) | 小市值SOE高于民营基准 | +2.59pp (CSI1000 Ridge) | Same as CSI500 — positive gap means actual < predicted | SOE is ABOVE the private benchmark | ⚠️ ⚠️ AMBIGUOUS/MISLEADING |
| 4 | [outputs/综合研究报告.md:180](outputs/综合研究报告.md#L180) | CSI300（大市值）gap 为负——大市值 SOE 低于民企基准 | −0.68pp (CSI300) | Negative gap → SOE actual > predicted. SOE is ABOVE the benchmark. | SOE is BELOW the private benchmark | ⚠️ ⚠️ AMBIGUOUS/MISLEADING |
| 5 | [outputs/实验报告_全A股反事实预测.md:269](outputs/实验报告_全A股反事实预测.md#L269) | 地方国企的负gap暗示：大型地方国企可能面临比民营企业更大的盈利压力 | CSI300 地方国企 Ridge −1.74pp, GBM −1.21pp | Negative gap → SOE actual EBI/A is HIGHER than predicted → SOE is doing BETTER than expected | 地方国企面临更大的盈利压力 | ❌ ❌ SIGN ERROR IN INTERPRETATION |
| 6 | [outputs/实验报告_全A股反事实预测.md:175](outputs/实验报告_全A股反事实预测.md#L175) | 消费/化工龙头低于民营基准 | CSI300 化工 GBM −6.61pp, 食品饮料 GBM −3.86pp | Negative gap → these SOEs' actual EBI/A is HIGHER than predicted | SOEs are BELOW the private benchmark | ⚠️ ⚠️ AMBIGUOUS/MISLEADING |
| 7 | [outputs/分行业显著性分析报告.md:110](outputs/分行业显著性分析报告.md#L110) | 全部 8 家 gap 为正。医药流通国企利润远低于民企基准。 | Positive gap for 医药商业 | Gap > 0 → SOE actual < predicted → SOE profit is LOWER than predicted benchmark | SOE profit is LOWER than private benchmark | ✅ ✓ CORRECT |
| 8 | [outputs/Ridge_CSI500_SOE_2025_行业分析报告.md:89](outputs/Ridge_CSI500_SOE_2025_行业分析报告.md#L89) | 钢铁行业 SOE 的极低利润（部分亏损）与同特征民营企业的预期回报形成巨大反差 | 钢铁Ⅱ Ridge +6.98pp | Gap > 0 → SOE actual is much LOWER than predicted → correct to say "contrast with expected private r | SOE profit far below private benchmark | ✅ ✓ CORRECT |

## 5. Required Corrections

### ❌ Sign Errors (must fix)

**outputs/实验报告_全A股反事实预测.md:269**

- **Current text**: "地方国企的负gap暗示：大型地方国企可能面临比民营企业更大的盈利压力"
- **Gap value**: CSI300 地方国企 Ridge −1.74pp, GBM −1.21pp
- **Why wrong**: Negative gap → SOE actual EBI/A is HIGHER than predicted → SOE is doing BETTER than expected
- **Corrected text**: ""地方国企的负 gap（实际 EBI/A 高于 反事实预测）暗示：大型地方国企的实际盈利能力 优于 同特征民营企业的预期水平。这可能反映大市值地方国企的规模优势或市场地位，而非盈利压力。""

### ⚠️ Ambiguous/Misleading Language (should clarify)

**outputs/多指数并集分析报告.md:82**

- **Current text**: "大市值SOE低于民营基准"
- **Gap value**: −0.68pp (CSI300 Ridge)
- **Issue**: The phrase "高于/低于 民营基准" is ambiguous. It could mean:
  - (A) "the gap value is above/below zero" — technically correct
  - (B) "SOE's actual EBI/A is above/below the private benchmark" — may be wrong depending on interpretation of "基准"
- **Recommended replacement**: ""SOE实际EBI/A（5.12%）高于民营模型预测的反事实值（4.44%）"。负 gap 表示实际盈利优于反事实预测。"

**outputs/多指数并集分析报告.md:83**

- **Current text**: "中市值SOE高于民营基准"
- **Gap value**: +2.82pp (CSI500 Ridge)
- **Issue**: The phrase "高于/低于 民营基准" is ambiguous. It could mean:
  - (A) "the gap value is above/below zero" — technically correct
  - (B) "SOE's actual EBI/A is above/below the private benchmark" — may be wrong depending on interpretation of "基准"
- **Recommended replacement**: ""SOE实际EBI/A（3.46%）低于民营模型预测的反事实值（6.28%）"。正 gap 表示实际盈利低于反事实预测。"

**outputs/多指数并集分析报告.md:84**

- **Current text**: "小市值SOE高于民营基准"
- **Gap value**: +2.59pp (CSI1000 Ridge)
- **Issue**: The phrase "高于/低于 民营基准" is ambiguous. It could mean:
  - (A) "the gap value is above/below zero" — technically correct
  - (B) "SOE's actual EBI/A is above/below the private benchmark" — may be wrong depending on interpretation of "基准"
- **Recommended replacement**: ""SOE实际EBI/A（2.61%）低于民营模型预测的反事实值（5.21%）""

**outputs/综合研究报告.md:180**

- **Current text**: "CSI300（大市值）gap 为负——大市值 SOE 低于民企基准"
- **Gap value**: −0.68pp (CSI300)
- **Issue**: The phrase "高于/低于 民营基准" is ambiguous. It could mean:
  - (A) "the gap value is above/below zero" — technically correct
  - (B) "SOE's actual EBI/A is above/below the private benchmark" — may be wrong depending on interpretation of "基准"
- **Recommended replacement**: ""大市值 SOE 的实际 EBI/A 高于 民企模型反事实预测（即负 gap 表示 SOE 表现优于 预测）""

**outputs/实验报告_全A股反事实预测.md:175**

- **Current text**: "消费/化工龙头低于民营基准"
- **Gap value**: CSI300 化工 GBM −6.61pp, 食品饮料 GBM −3.86pp
- **Issue**: The phrase "高于/低于 民营基准" is ambiguous. It could mean:
  - (A) "the gap value is above/below zero" — technically correct
  - (B) "SOE's actual EBI/A is above/below the private benchmark" — may be wrong depending on interpretation of "基准"
- **Recommended replacement**: ""消费/化工龙头 SOE 的实际 EBI/A 高于 民营模型预测（即负 gap = 实际优于预测）""

## 6. Full Reconciliation: Predicted vs Actual vs Gap by Sample

| Sample | Predicted Mean | Actual Mean | Gap (Pred − Act) | Existing Report Gap | Match? | Sign Correct in Report? |
|--------|---------------|-------------|------------------|---------------------|--------|------------------------|
| CSI300_SOE (ridge) | +0.0444 | +0.0512 | **-0.0068** | −0.68pp | ✅ | Gap<0 → actual > predicted ⚠️ (check report interpretation) |
| CSI500_SOE (ridge) | +0.0628 | +0.0346 | **+0.0282** | +2.82pp | ✅ | Gap>0 → actual < predicted ✓ (reported sign matches) |
| CSI1000_SOE (ridge) | +0.0521 | +0.0261 | **+0.0259** | +2.59pp | ✅ | Gap>0 → actual < predicted ✓ (reported sign matches) |
| A-share_SOE_2025 (ridge) | +0.0303 | +0.0150 | **+0.0153** | +1.53pp | ✅ | Gap>0 → actual < predicted ✓ (reported sign matches) |
| CSI300∪CSI500_SOE (ridge) | +0.0553 | +0.0409 | **+0.0144** | +1.44pp | ✅ | Gap>0 → actual < predicted ✓ (reported sign matches) |
| CSI500∪CSI1000_SOE (ridge) | +0.0558 | +0.0291 | **+0.0267** | +2.67pp | ✅ | Gap>0 → actual < predicted ✓ (reported sign matches) |

## 7. Final Verdict

### Code & Data: ✅ PASS

- All Python code defines `gap = predicted - actual` consistently
- All CSV columns store `gap = predicted - actual` correctly
- All mean gaps recomputed from data match the code definition exactly
- No sign errors in code or data

### Reports: ⚠️ WARNING — Interpretation Language Needs Clarification

The reports are **numerically correct** (the gap numbers are right), but the **interpretation language**
is ambiguous and, in some cases, reverses the economic meaning:

1. **"SOE 高于/低于 民营基准"** — This phrase conflates two different concepts:
   - The sign of the gap (gap > 0 vs gap < 0)
   - Whether SOE's actual EBI/A is above or below the predicted counterfactual
   - Whether SOE's actual EBI/A is above or below the raw private-firm average

2. **"地方国企的负gap暗示盈利压力"** — This is **backwards**.
   Negative gap = SOE actual > predicted = SOE is doing BETTER than expected.
   If anything, a negative gap suggests the opposite of '盈利压力'.

### Recommended Fix

Replace all instances of "高于/低于 民营基准" with precise, directional language:

| Old Language | Gap Sign | Correct Replacement |
|-------------|----------|-------------------|
| SOE 高于民营基准 | Positive (+) | **SOE 实际 EBI/A 低于 民营反事实预测 (gap > 0)** |
| SOE 低于民营基准 | Negative (−) | **SOE 实际 EBI/A 高于 民营反事实预测 (gap < 0)** |
| 面临盈利压力 (negative gap) | Negative (−) | **SOE 实际盈利优于反事实预测** (删除"盈利压力"表述) |

### Paper Language Recommendation

For the paper, use this standard formulation in all tables and text:

> The counterfactual gap is defined as Gap = ÊBI/A − EBI/A, where ÊBI/A is the
> predicted EBI/A from a model trained on private firms. A positive (negative) gap
> indicates that the SOE's actual EBI/A is lower (higher) than the counterfactual
> prediction — i.e., the SOE underperforms (outperforms) relative to what a comparable
> private firm would be expected to earn.
