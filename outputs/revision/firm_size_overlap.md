# Firm-Size Overlap Analysis

**Date**: 2026-08-11

## Figure 1: Kernel Density of Firm Size

![KDE Overlap](fig_firm_size_kde.png)

## Figure 2: Empirical CDF of Firm Size

![ECDF Overlap](fig_firm_size_ecdf.png)

## Summary Statistics

| Sample | n | Mean | Median | SD | P10 | P25 | P75 | P90 |
|--------|---|------|--------|----|-----|-----|-----|-----|
| A-share Private | 3,465 | 21.88 | 21.73 | 1.12 | 20.63 | 21.14 | 22.50 | 23.37 |
| A-share SOE (All) | 1,406 | 23.07 | 22.92 | 1.54 | 21.26 | 21.93 | 24.02 | 25.06 |
| CSI500 SOE | 178 | 24.19 | 24.22 | 1.02 | 22.84 | 23.49 | 24.89 | 25.41 |
| CSI1000 SOE | 334 | 23.35 | 23.40 | 0.97 | 22.12 | 22.68 | 23.98 | 24.56 |
| CSI300 SOE | 117 | 25.72 | 25.77 | 1.35 | 24.08 | 24.71 | 26.60 | 27.49 |
| H-share Private | 1,443 | 22.66 | 22.14 | 2.86 | 19.02 | 20.83 | 24.48 | 26.99 |

## Standardized Mean Differences (SMD)

| Sample | Private Mean | SOE Mean | SMD | \|SMD\| > 0.25? |
|--------|-------------|----------|-----|------------------|
| A-share SOE (All) | 21.88 | 23.07 | +0.881 | ⚠️ YES |
| CSI300 SOE | 21.88 | 25.72 | +3.090 | ⚠️ YES |
| CSI500 SOE | 21.88 | 24.19 | +2.150 | ⚠️ YES |
| CSI1000 SOE | 21.88 | 23.35 | +1.395 | ⚠️ YES |

## Overlap Assessment

The private-firm FirmSize distribution spans ln(assets) ∈ [18.1, 27.4], with a central 90% range of [20.3, 23.9].

### A-share SOE (All)

- Mean FirmSize: 23.07 (private: 21.88)
- Median FirmSize: 22.92 (private: 21.73)
- % outside private P1–P99: **7.8%**
- % outside private P5–P95: **28.8%**

### CSI300 SOE

- Mean FirmSize: 25.72 (private: 21.88)
- Median FirmSize: 25.77 (private: 21.73)
- % outside private P1–P99: **58.1%**
- % outside private P5–P95: **93.2%**

### CSI500 SOE

- Mean FirmSize: 24.19 (private: 21.88)
- Median FirmSize: 24.22 (private: 21.73)
- % outside private P1–P99: **11.2%**
- % outside private P5–P95: **60.7%**

### CSI1000 SOE

- Mean FirmSize: 23.35 (private: 21.88)
- Median FirmSize: 23.40 (private: 21.73)
- % outside private P1–P99: **2.4%**
- % outside private P5–P95: **28.1%**

### Is the overlap sufficient?

The answer depends on the index and the threshold applied:

1. **CSI300 SOE**: These are the largest A-share firms. Their FirmSize distribution
   is shifted substantially right of the private-firm distribution. A large fraction
   of CSI300 SOEs are larger than the 95th percentile of private firms. Common
   support for CSI300 SOEs in the private-firm size distribution is limited.

2. **CSI500 SOE**: These are mid-cap firms. Their FirmSize distribution shows
   better overlap with the private-firm distribution, though a rightward shift
   remains visible. A moderate fraction falls outside the private P5–P95 range.

3. **CSI1000 SOE**: These are smaller-cap firms. Their FirmSize distribution has
   the best overlap with private firms among the three indices. The majority of
   CSI1000 SOEs fall within the private-firm size range.

4. **All A-share SOE**: The combined SOE distribution is bimodal or right-skewed
   relative to private firms, reflecting the mixture of very large central SOEs
   and smaller local SOEs.

### Are CSI300/500/1000 differences size-driven?

The three CSI indices are explicitly constructed by market capitalization ranking:
- CSI300: Top 300 by market cap (largest firms)
- CSI500: Next 500 (mid-cap)
- CSI1000: Next 1000 (small-cap)

Since market capitalization is highly correlated with total assets (FirmSize),
the observed gap pattern — CSI300 gap negative, CSI500/CSI1000 gap positive —
is **mechanically correlated with FirmSize**. The counterfactual model controls
for FirmSize as a feature, so the size effect is partialled out in the gap
estimation. However, the strong size-dependence of index membership means that
comparing gaps across indices is, to some extent, comparing gaps across size
strata. This is a descriptive observation about the index construction, not a
causal statement about firm size and SOE performance.

### Note on Common Support

FirmSize is only one of nine features in the model. The multivariate common
support assessment (Task 2 diagnostics) provides a more complete picture. The
univariate FirmSize overlap shown here is a useful diagnostic but does not, by
itself, determine whether the counterfactual predictions are interpolations or
extrapolations.
