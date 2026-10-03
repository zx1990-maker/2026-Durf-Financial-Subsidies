# Variable Construction Audit

**Date**: 2026-08-11

## Target Variable

### EBI/A (t+1)

| Property | Value |
|----------|-------|
| **Report Definition** | (归母净利润_t+1 + 利息支出_t+1) / 资产总计_t+1 |
| **Code Formula** | `(np_t + ie_t) / ta_t` where np_t = `net_profit[t+1]`, ie_t = `interest_expense[t+1]`, ta_t = `total_assets[t+1]` |
| **Raw Field: Numerator 1** | `归属母公司股东的净利润` [报告期] (t+1)年报, [报表类型] 合并报表, [单位] 元 |
| **Raw Field: Numerator 2** | `利息支出` [报告期] (t+1)年报, [报表类型] 合并报表, [币种] 原始币种, [单位] 元 |
| **Raw Field: Denominator** | `资产总计` [报告期] (t+1)年报, [报表类型] 合并报表, [单位] 元 |
| **Year Alignment** | All components from t+1 (target year) ✓ |
| **Consolidation** | 合并报表 (consolidated) for all components ✓ |
| **Unit** | Numerator in 元, denominator in 元 → ratio is unitless |
| **Zero Denominator** | Rows with `total_assets[t+1] == 0` or NaN are dropped in `records_to_dataframe()` |
| **Negative Denominator** | Dropped at `ta_t == 0` check; negative assets are extremely rare (accounting error) |
| **Match Report?** | ✅ **MATCH** |
| **Issues** | None |

## Feature Variables (all measured at t)

### FirmSize

| Property | Value |
|----------|-------|
| **Report Definition** | ln(资产总计_t) |
| **Actual Code Formula** | `np.log(ta_f) where ta_f = total_assets[t]` |
| **Raw Field (Numerator)** | 资产总计 [报告期] t年报, [报表类型] 合并报表, [单位] 元 |
| **Raw Field (Denominator)** | N/A (log transform) |
| **Year Alignment** | t (feature year) ✓ |
| **Consolidation** | 合并报表 ✓ |
| **Unit** | ln(元) |
| **Zero/NaN Handling** | ta_f > 0 required; non-positive → NaN |
| **Issues** | Column duplication: 2015 file has 资产总计 in BOTH Col I and Col K (identical values — verified, no data issue) |

### AssetTurnover

| Property | Value |
|----------|-------|
| **Report Definition** | 营业收入_t / 资产总计_t |
| **Actual Code Formula** | `rev_f / ta_f` |
| **Raw Field (Numerator)** | 营业收入 [报告期] t年报, [报表类型] 合并报表, [单位] 元 |
| **Raw Field (Denominator)** | 资产总计 [报告期] t年报 (same as FirmSize) |
| **Year Alignment** | t (feature year) ✓ |
| **Consolidation** | 合并报表 ✓ |
| **Unit** | unitless ratio |
| **Zero/NaN Handling** | ta_f > 0; if NaN or ≤ 0 → NaN |
| **Issues** | None |

### FinancialLeverage

| Property | Value |
|----------|-------|
| **Report Definition** | 负债合计_t / 资产总计_t |
| **Actual Code Formula** | `tl_f / ta_f` |
| **Raw Field (Numerator)** | 负债合计 [报告期] t年报, [报表类型] 合并报表, [单位] 元 |
| **Raw Field (Denominator)** | 资产总计 [报告期] t年报 |
| **Year Alignment** | t (feature year) ✓ |
| **Consolidation** | 合并报表 ✓ |
| **Unit** | unitless ratio |
| **Zero/NaN Handling** | ta_f > 0; if NaN or ≤ 0 → NaN |
| **Issues** | None |

### FixedAssetsRatio

| Property | Value |
|----------|-------|
| **Report Definition** | 固定资产净值_t / 资产总计_t |
| **Actual Code Formula** | `nfa_f / ta_f` |
| **Raw Field (Numerator)** | 固定资产-净值 [报告期] t年报, [单位] 元 |
| **Raw Field (Denominator)** | 资产总计 [报告期] t年报 |
| **Year Alignment** | t (feature year) ✓ |
| **Consolidation** | ⚠️ 固定资产-净值 header does NOT specify [报表类型] 合并报表. Missing consolidation tag. |
| **Unit** | unitless ratio |
| **Zero/NaN Handling** | ta_f > 0; net fixed assets can be 0 (service firms). NaN rate = 19.3% |
| **Issues** | ⚠️ 1. Uses NET fixed assets (净值), not gross (原值). This should be stated in paper. 2. Missing [报表类型] tag — likely consolidated but not explicitly tagged. 3. 19.3% NaN rate — single largest source of sample attrition in complete-case filter. |

### CurrentRatio

| Property | Value |
|----------|-------|
| **Report Definition** | 流动资产_t / 流动负债_t |
| **Actual Code Formula** | `ca_f / cl_f` |
| **Raw Field (Numerator)** | 流动资产合计 [报告期] t年报, [报表类型] 合并报表, [币种] 原始币种, [单位] 元 |
| **Raw Field (Denominator)** | 流动负债合计 [报告期] t年报, [报表类型] 合并报表, [单位] 元 |
| **Year Alignment** | t (feature year) ✓ |
| **Consolidation** | 合并报表 ✓ (both) |
| **Unit** | unitless ratio |
| **Zero/NaN Handling** | cl_f > 0; if NaN or ≤ 0 → NaN |
| **Issues** | None |

### CapexRatio

| Property | Value |
|----------|-------|
| **Report Definition** | 资本性支出_t / 资产总计_t |
| **Actual Code Formula** | `capex_f / ta_f` |
| **Raw Field (Numerator)** | 资本性支出 [报告期] t年报, [报表类型] 合并报表, [币种] 原始币种, [单位] 元 |
| **Raw Field (Denominator)** | 资产总计 [报告期] t年报 |
| **Year Alignment** | t (feature year) ✓ |
| **Consolidation** | 合并报表 ✓ |
| **Unit** | unitless ratio |
| **Zero/NaN Handling** | ta_f > 0 |
| **Issues** | None |

### WorkingCapitalRatio

| Property | Value |
|----------|-------|
| **Report Definition** | (流动资产_t − 流动负债_t) / 资产总计_t |
| **Actual Code Formula** | `(ca_f - cl_f) / ta_f` |
| **Raw Field (Numerator)** | 流动资产合计 [报告期] t年报 − 流动负债合计 [报告期] t年报 (see CurrentRatio) |
| **Raw Field (Denominator)** | 资产总计 [报告期] t年报 |
| **Year Alignment** | t (feature year) ✓ |
| **Consolidation** | 合并报表 ✓ |
| **Unit** | unitless ratio |
| **Zero/NaN Handling** | ta_f > 0 |
| **Issues** | None. |

### MarketShare

| Property | Value |
|----------|-------|
| **Report Definition** | 企业营收_t / 同行业同年度所有企业总营收_t |
| **Actual Code Formula** | `firm_revenue / sum(all firm revenues in same feature_year × industry_l2 group)` |
| **Raw Field (Numerator)** | 营业收入 [报告期] t年报 (same as AssetTurnover numerator) |
| **Raw Field (Denominator)** | Sum of 营业收入 across ALL firms in the same (feature_year, industry_l2) that survived records_to_dataframe() |
| **Year Alignment** | t (feature year) ✓ |
| **Consolidation** | 合并报表 ✓ |
| **Unit** | unitless (share, 0–1) |
| **Zero/NaN Handling** | Group total_rev > 0 required |
| **Issues** | ⚠️ UNIVERSE: Denominator includes ALL ownership types (民营 + SOE + 公众 + 外资 + 集体 + 其他). This means a private firm's MarketShare is relative to the ENTIRE industry including SOEs. If the counterfactual question is about private-firm return-generating process, this is correct — the firm competes against all firms in the industry. However, the computation happens per yearly file, meaning firms dropped for missing t+1 assets are excluded from the denominator. This is a minor inconsistency. |

### HHI

| Property | Value |
|----------|-------|
| **Report Definition** | 行业赫芬达尔指数 (sum of squared MarketShare within industry-year) |
| **Actual Code Formula** | `sum(MarketShare²) for each (feature_year, industry_l2) group` |
| **Raw Field (Numerator)** | Derived from MarketShare (see above) |
| **Raw Field (Denominator)** | N/A |
| **Year Alignment** | t (feature year) ✓ |
| **Consolidation** | N/A (derived) |
| **Unit** | unitless (0–1, where 1 = monopoly) |
| **Zero/NaN Handling** | Group total_rev > 0 required |
| **Issues** | ⚠️ SAME UNIVERSE ISSUE as MarketShare. HHI is computed using ALL firms in the industry-year, including SOEs. A private firm's HHI reflects the competitive structure of the entire industry, not the private-only segment. This is arguably correct for the counterfactual question, but should be documented. Also: HHI is IDENTICAL for all firms in the same (feature_year, industry_l2) — it is an industry-level variable, not firm-level. This is intentional. |

## Definition Match Summary

| Variable | Report Definition | Actual Code Definition | Raw Fields | Match? | Issues |
|----------|------------------|----------------------|------------|--------|--------|
| FirmSize | ln(资产总计_t) | np.log(ta_f) where ta_f = total_assets[t] | 资产总计 [报告期] t年报, [报表类型] 合并报表, [单位] 元 | MATCH |  None |
| AssetTurnover | 营业收入_t / 资产总计_t | rev_f / ta_f | 营业收入 [报告期] t年报, [报表类型] 合并报表, [单位] 元 | MATCH |  None |
| FinancialLeverage | 负债合计_t / 资产总计_t | tl_f / ta_f | 负债合计 [报告期] t年报, [报表类型] 合并报表, [单位] 元 | MATCH |  None |
| FixedAssetsRatio | 固定资产净值_t / 资产总计_t | nfa_f / ta_f | 固定资产-净值 [报告期] t年报, [单位] 元 | MATCH* | ⚠️ See notes |
| CurrentRatio | 流动资产_t / 流动负债_t | ca_f / cl_f | 流动资产合计 [报告期] t年报, [报表类型] 合并报表, [币种] 原始币种, [单位] 元 | MATCH |  None |
| CapexRatio | 资本性支出_t / 资产总计_t | capex_f / ta_f | 资本性支出 [报告期] t年报, [报表类型] 合并报表, [币种] 原始币种, [单位] 元 | MATCH |  None |
| WorkingCapitalRatio | (流动资产_t − 流动负债_t) / 资产总计_t | (ca_f - cl_f) / ta_f | 流动资产合计 [报告期] t年报 − 流动负债合计 [报告期] t年报 (see CurrentRatio) | MATCH |  None |
| MarketShare | 企业营收_t / 同行业同年度所有企业总营收_t | firm_revenue / sum(all firm revenues in same feature_year × industry_l2 group) | 营业收入 [报告期] t年报 (same as AssetTurnover numerator) | MATCH* | ⚠️ See notes |
| HHI | 行业赫芬达尔指数 (sum of squared MarketShare within industry-year) | sum(MarketShare²) for each (feature_year, industry_l2) group | Derived from MarketShare (see above) | MATCH* | ⚠️ See notes |

## Random Spot-Check Results

**160 variable-level comparisons across 20 firm-years**

| Match Type | Count |
|-----------|-------|
| EXACT | 159 |
| BOTH_NaN | 1 |

✅ **All spot-checks match exactly** — manual recalculation from raw Wind fields produces identical values to the processed data.

## MarketShare / HHI — Universe Audit

### MarketShare Denominator Composition

- MarketShare sum-to-one check: 0 groups deviate from 1.0 (tolerance 0.001)
- Ownership types in MarketShare denominator: {'民营企业', '地方国有企业', '中央国有企业', '公众企业', '其他企业'}
- Sample group: year=2021, ind=硬件设备, n=594, total_rev=3,649,613,093,460, sum(MarketShare)=1.000000
- MarketShare computed on: 57701 obs (all types), of which 35731 are private

### Key Question: What firms are in the denominator?

MarketShare is computed in `records_to_dataframe()` for each yearly file independently:

```python
for (fy, ind2), group in df.groupby(['feature_year', 'industry_l2']):
    total_rev = group['营业收入'].sum()
    shares = group['营业收入'] / total_rev
```

The denominator (`total_rev`) includes:
- ✅ ALL ownership types (民营 + SOE + 公众 + 外资 + 集体 + 其他)
- ✅ ALL firms in the yearly cross-section that survived `records_to_dataframe()`
- ❌ Does NOT include firms dropped for missing t+1 assets (these are excluded from the dataframe before MarketShare is computed)

**Implication**: A private firm's MarketShare is relative to the entire industry (including SOEs),
but the denominator slightly understates total industry revenue because firms with missing t+1
assets are excluded. The magnitude of this understatement is small (≤3% of firms in later years, up to 20% in 2014→2015).

## Column Duplication Issue

In the 2015 file, `资产总计 [报告期] 2015年报` appears in TWO columns: I and K.
- Col I: 4385 values parsed
- Col K: 4385 values parsed
- Mismatches between I and K: **0**

✅ The two columns are **identical** — this is a true duplicate in the Wind extract, not a data error.

In the parsing code, the later column (K) overwrites the earlier (I) for `record['data']['total_assets'][2015]`.
Since the values are identical, this has no effect on computed variables.

## Winsorization

- **Bounds**: (0.05, 0.95) (5th to 95th percentile)
- **Reference distribution**: Training set (private firms only)
- **Applied to**: EBI/A + all 9 features
- **Timing**: After train/pred split, BEFORE model fitting
- **Implementation**: `winsorize_features_and_target()` in `run_counterfactual.py`
- **Stored as**: `{variable}_winsor` columns; original values preserved in `{variable}` columns

⚠️ **Important**: Winsorization bounds are computed from the TRAINING (private) distribution and applied to BOTH training and prediction (SOE) data. This means SOE features/targets are clipped at private-firm quantiles. This is the correct approach for counterfactual prediction (the model should not see SOE distribution information), but it means SOE values outside the private-firm P5–P95 range are clipped.

## Final Verdict

**WARNING** — Issues found: FixedAssetsRatio, MarketShare, HHI

### Issues Requiring Attention

#### FixedAssetsRatio

⚠️ 1. Uses NET fixed assets (净值), not gross (原值). This should be stated in paper. 2. Missing [报表类型] tag — likely consolidated but not explicitly tagged. 3. 19.3% NaN rate — single largest source of sample attrition in complete-case filter.

#### MarketShare

⚠️ UNIVERSE: Denominator includes ALL ownership types (民营 + SOE + 公众 + 外资 + 集体 + 其他). This means a private firm's MarketShare is relative to the ENTIRE industry including SOEs. If the counterfactual question is about private-firm return-generating process, this is correct — the firm competes against all firms in the industry. However, the computation happens per yearly file, meaning firms dropped for missing t+1 assets are excluded from the denominator. This is a minor inconsistency.

#### HHI

⚠️ SAME UNIVERSE ISSUE as MarketShare. HHI is computed using ALL firms in the industry-year, including SOEs. A private firm's HHI reflects the competitive structure of the entire industry, not the private-only segment. This is arguably correct for the counterfactual question, but should be documented. Also: HHI is IDENTICAL for all firms in the same (feature_year, industry_l2) — it is an industry-level variable, not firm-level. This is intentional.

### Documentation Recommendations

1. **FixedAssetsRatio**: Specify in the paper that 'net' (净值) fixed assets are used, not gross (原值).
2. **MarketShare / HHI**: Document that the denominator includes ALL firm types (not private-only),
   which is correct for measuring competitive position but should be stated explicitly.
3. **Winsorization**: Clarify that winsorization uses private-firm quantiles, applied to both
   training and prediction samples.
4. **固定资产-净值**: Note the missing [报表类型] tag and confirm with Wind that this field
   uses consolidated statements (it almost certainly does in practice).
