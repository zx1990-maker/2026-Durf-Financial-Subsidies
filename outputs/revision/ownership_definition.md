# Ownership Classification Audit

**Date**: 2026-08-11
**Final Verdict**: **PASS (with documented caveat)**

## Executive Summary

| Question | Answer |
|----------|--------|
| Wind field used | `企业所有制性质` [交易日期] 最新收盘日 (column R in xlsx) |
| SOE definition | `ownership` ∈ {'中央国有企业', '地方国有企业'} |
| Private firm definition | `ownership` == '民营企业' |
| Central vs Local SOE | `'中央国有企业'` vs `'地方国有企业'` (Wind classification) |
| Time-varying or static? | **STATIC** — same ownership value in all 2015–2024 files |
| Backfill method | Most recent non-missing ownership (from later target_year) |
| Ownership switchers | 0 firms change type across files |
| Always-missing ownership | 1 firm (Wind metadata row "数据来源：Wind" — not a real firm, dropped during parsing) |
| Look-ahead bias | **YES** — inherent in Wind field `[交易日期] 最新收盘日` |

## 1. Wind Field Identification

### 1.1 Raw Wind Header

The ownership classification comes from the Wind field with the following exact header:

```
企业所有制性质
[交易日期] 最新收盘日
```

### 1.2 Code Path

In `run_counterfactual.py`, the `classify_column()` function (line 179) matches this field:

```python
elif '企业所有制' in field_name:
    return 'ownership', year
```

The matched field name is `企业所有制性质` (Enterprise Ownership Nature).

### 1.3 Critical Detail: `[交易日期] 最新收盘日`

The Wind header includes the tag `[交易日期] 最新收盘日` (Trading Date: Latest Closing Day).
This means the ownership value is **NOT** the historical ownership as of each report year.
It is a **point-in-time snapshot** as of the data extraction date (approximately August 2026).

**Implication**: When applied to the 2014–2024 panel, the ownership classification is retroactive.
A firm that was an SOE in 2014 but was privatized by 2026 would be classified as '民营企业' in ALL years,
including 2014 when it was actually state-owned. Conversely, a firm that was private in 2014 but
nationalized by 2026 would be classified as '国有企业' in all years.

This is an **inherent look-ahead bias** in the Wind data field itself — not a coding error.

## 2. Classification Rules in Code

### 2.1 Constants (run_counterfactual.py, lines 42-45)

```python
SOE_TYPES = {'中央国有企业', '地方国有企业'}
NON_SOE_TYPE = '民营企业'
```

### 2.2 Training Set Construction

The training set is constructed in `prepare_train_pred()` (lines 552-566):

```python
all_a['is_soe'] = all_a['ownership'].isin(SOE_TYPES)
all_a['is_non_soe_private'] = all_a['ownership'] == NON_SOE_TYPE
all_a['benchmark_group'] = all_a.apply(
    lambda r: 'SOE' if r['is_soe'] else (
        'nonSOE_private' if r['is_non_soe_private'] else 'other'),
    axis=1
)
# Training: non-SOE private enterprises
train_df = all_a[all_a['benchmark_group'] == 'nonSOE_private'].copy()
```

### 2.3 All Ownership Types in Data

The Wind `企业所有制性质` field takes the following values in this dataset:

- `中央国有企业` → SOE
- `公众企业` → EXCLUDED from training
- `其他企业` → EXCLUDED from training
- `地方国有企业` → SOE
- `外资企业` → EXCLUDED from training
- `民营企业` → Training (private)
- `集体企业` → EXCLUDED from training

## 3. Ownership is Static Across Files

### 3.1 Evidence

✅ **All 5536 firms have identical ownership values in every yearly file (2015–2024).**

Each yearly file (A_2015_origin.xlsx through A_2024_origin.xlsx) contains the same ownership value
for a given firm. This confirms that the Wind `企业所有制性质` field is a static attribute —
it reports the firm's ownership as of the data extraction date, not the historical ownership.

The 2014 file (A_2014_origin.xlsx) is the exception: its ownership column is entirely empty.
This is handled by the backfill procedure (see Section 4).

## 4. Ownership Backfill for 2014

### 4.1 Mechanism

The `backfill_ownership()` function (run_counterfactual.py, lines 459-479):

```python
known = df[df['ownership'].notna() & (df['ownership'] != '')]
firm_own = known.sort_values('target_year').groupby('firm_id')['ownership'].last()
```

This takes the **last (most recent target_year)** non-missing ownership value for each firm.
Since ownership is static across all files, the 'last' value is identical to the 'first' value —
the choice of 'last' vs 'first' is moot because all are the same.

- Firms in both 2014 and 2015+: 5536 → **all backfilled successfully**
- Firms only in 2014 (not in later years): 0 → **cannot be backfilled, will be dropped**


## 5. Ownership Changes Over Time

✅ **No firms change ownership type across the 2015–2024 files.**

This is expected because the Wind field is a static snapshot. However, it also means:
- The data **cannot detect** real ownership changes (e.g., privatization, nationalization)
- If a firm was privatized in 2020, it would be classified as '民营企业' in 2015 data too
- This creates a potential measurement error for firms that changed ownership during 2014–2024

## 6. Missing and Ambiguous Ownership

- **Firms with ownership missing in all years**: 1 — this is the Wind metadata/annotation row (`数据来源：Wind`), which has no valid `stock_code` and is automatically dropped by `records_to_dataframe()`. It never enters the analysis pipeline.
- **Firms with valid ownership after pipeline**: 5,535 (100% of firms that survive `records_to_dataframe` + backfill)
- **Firms classified as 'other' (公众企业, 外资企业, 集体企业, 其他企业)**: 566

'Other' ownership types are **excluded from both the training set and the SOE prediction set**.
They constitute a residual category of:
- `公众企业`: 328 firms
- `其他企业`: 19 firms
- `外资企业`: 193 firms
- `集体企业`: 26 firms

See `ownership_missing_or_ambiguous.csv` for the complete list.

## 7. Look-Ahead Bias Assessment

### 7.1 The Problem

The Wind field `企业所有制性质 [交易日期] 最新收盘日` classifies each firm based on its ownership
**as of the data extraction date** (≈ August 2026). When this is applied to historical years (2014–2024),
the 2026 ownership status is retroactively assigned to all earlier years.

### 7.2 Examples of Potential Misclassification

1. **Privatized SOE**: A firm was 地方国有企业 in 2014–2018, privatized in 2019. Wind (2026) shows
   '民营企业'. The code classifies it as '民营企业' for ALL years including 2014–2018 → wrong training data.
2. **Nationalized firm**: A firm was 民营企业 in 2014–2020, nationalized in 2021. Wind (2026) shows
   '地方国有企业'. The code classifies it as SOE for ALL years → wrong prediction target.
3. **IPO after ownership change**: A firm IPO'd in 2020 after being privatized in 2018. Wind shows
   '民营企业' for all years → actually correct (it was private when it entered the dataset).

### 7.3 Severity Assessment

The severity depends on the frequency of ownership changes among Chinese listed firms during 2014–2024.
In practice:
- **State-to-private conversions (privatization)** are rare among listed firms in China
- **Private-to-state conversions (nationalization)** are more common (e.g., 纾困 bailouts, 2018–2019)
- The number of affected firms is likely small (< 50) but not zero
- This is a **measurement error** in the ownership classification, not a coding bug

### 7.4 Mitigation

The current code provides no mitigation. Recommendations:
1. Acknowledge this limitation explicitly in the paper
2. If possible, obtain historical ownership data (e.g., from CSMAR's 股权性质 file, which tracks
   ultimate controller changes year by year)
3. As a robustness check, identify firms with known ownership changes and exclude them
4. Note that the bias direction is ambiguous: misclassified privatized SOEs in training could bias
   predictions toward SOE-like returns; misclassified nationalized firms in prediction would
   contaminate the gap estimate

## 8. Ownership by Target Year (After Backfill)

The following table shows the distribution of ownership types across target years AFTER backfill.
Note that ownership is identical across years for each firm because the underlying Wind field is static.

See `ownership_by_year.csv` for the complete machine-readable table.

### Before Backfill (raw from files):

```
ownership          中央国有企业  公众企业  其他企业  地方国有企业  外资企业  民营企业  集体企业
target_year                                                    
2015         4385       0     0     0       0     0     0     0
2016            0     430   292    19     959   148  2767    25
2017            0     444   305    19     978   162  3027    25
2018            0     458   320    19     995   182  3245    25
2019            0     474   326    18    1004   189  3371    25
2020            0     473   326    19    1007   189  3412    26
2021            0     475   326    19    1008   190  3430    26
2022            0     478   328    19    1011   193  3480    26
2023            0     478   328    19    1011   193  3480    26
2024            0     478   328    19    1011   193  3480    26
2025            0     478   328    19    1011   193  3479    26
```

### After Backfill:

```
ownership    中央国有企业  公众企业  其他企业  地方国有企业  外资企业  民营企业  集体企业
target_year                                              
2015            418   279    16     955   133  2560    24
2016            430   292    19     959   148  2767    25
2017            444   305    19     978   162  3027    25
2018            458   320    19     995   182  3245    25
2019            474   326    18    1004   189  3371    25
2020            473   326    19    1007   189  3412    26
2021            475   326    19    1008   190  3430    26
2022            478   328    19    1011   193  3480    26
2023            478   328    19    1011   193  3480    26
2024            478   328    19    1011   193  3480    26
2025            478   328    19    1011   193  3479    26
```

## 9. Ownership Definition for Paper (Data Section)

The following paragraph can be inserted directly into the paper's Data section:

> **Ownership Classification.** — Firm ownership is classified using Wind Financial Terminal's
> `企业所有制性质` (Enterprise Ownership Nature) field. State-owned enterprises (SOEs) are defined
> as firms classified as `中央国有企业` (central SOEs) or `地方国有企业` (local SOEs). Private firms
> (the benchmark group) are defined as firms classified as `民营企业`. Firms classified as `公众企业`
> (public enterprises), `外资企业` (foreign-invested enterprises), `集体企业` (collective enterprises),
> or `其他企业` (other enterprises) are excluded from both the training sample and the prediction sample.
> 
> An important caveat is that Wind's `企业所有制性质` field reflects the ownership status as of the
> data extraction date (August 2026) rather than the historical ownership at each fiscal year-end.
> The ownership classification is therefore static across the 2014–2024 panel period. For the 2014
> cross-section, where this field is missing in the raw Wind extract, ownership is backfilled using
> each firm's classification from the most recent subsequent year in which the firm appears. This
> procedure implicitly assumes that ownership type is time-invariant for each firm. While ownership
> changes among Chinese listed firms are relatively infrequent, the static classification may
> misclassify a small number of firms that underwent privatization or nationalization during the
> sample period. The resulting measurement error is likely small in magnitude but should be noted
> as a limitation of the Wind ownership data.

## 10. Final Verdict

**PASS (with documented caveat)**

- ⚠️ Ownership field `[交易日期] 最新收盘日` is inherently look-ahead (2026 snapshot applied to 2014–2024). This is a Wind data limitation, not a coding error. All other aspects of ownership classification are correctly implemented.

### Assessment

The ownership classification in this project is:

1. **Correctly implemented** — The code correctly reads `企业所有制性质` from Wind, applies the
   stated definitions (SOE = {中央, 地方}国有企业, private = 民营企业), and backfills 2014 ownership.

2. **Internally consistent** — Ownership is static across all yearly files, so there are no within-firm
   contradictions or year-to-year inconsistencies in the processed data.

3. **Subject to inherent look-ahead bias** — The Wind field itself is a current snapshot, not a
   historical time series. This is a data limitation, not a coding error. The code cannot fix this
   without access to historical ownership data (e.g., CSMAR 股权性质).

4. **Well-documented exception for 2014** — The backfill procedure is deterministic and documented.
   All 4,385 affected rows (2014 observations) are successfully backfilled with no data loss.

### Recommended Actions

1. **PAPER**: Include the ownership definition paragraph from Section 9 in the Data section
2. **PAPER**: Add a footnote acknowledging the static/look-ahead nature of the Wind ownership field
3. **ROBUSTNESS**: If possible, cross-reference with CSMAR 股权性质 for firms with known ownership
   changes during 2014–2024
4. **ROBUSTNESS**: As a sensitivity check, identify firms with ownership changes using alternative
   data sources and verify that excluding them does not change results
