# Support-Robustness Comparison

**Date**: 2026-08-11

## Specifications

| Spec | Training | Prediction |
|------|----------|-----------|
| **A (Full)** | All private firms (27,722 complete-case obs, 3,471 firms) | All SOEs |
| **B (Supported SOEs)** | All private firms (same model as A) | Only common-support SOEs (5-NN dist ≤ P90) |
| **C (Support-local)** | Private firms in SOE support region | Only common-support SOEs |

Common support defined as: 5-nearest-neighbor distance from SOE to private firms
in standardized 9-feature space ≤ 90th percentile of private-to-private distances.

## Results

| Spec | Sample | N Train | N SOE | Pred Mean | Actual Mean | **Gap** | SE | 95% CI | CV R² |
|------|--------|---------|-------|-----------|-------------|---------|----|--------|-------|
| A (Full) | All A-share SOE | 27,722 | 1406 | +0.0295 | +0.0150 | **+0.0145** | 0.0015 | [+0.0116, +0.0174] | +0.203 |
| B (Supported SOEs) | All A-share SOE | 27,722 | 870 | +0.0269 | +0.0135 | **+0.0134** | 0.0018 | [+0.0098, +0.0170] | +0.203 |
| C (Support-local) | All A-share SOE | 25,037 | 870 | +0.0271 | +0.0135 | **+0.0136** | 0.0018 | [+0.0100, +0.0172] | +0.214 |
| A (Full) | CSI300 SOE | 27,722 | 117 | +0.0426 | +0.0512 | **-0.0086** | 0.0043 | [-0.0170, -0.0002] | +0.203 |
| B (Supported SOEs) | CSI300 SOE | 27,722 | 15 | +0.0468 | +0.0764 | **-0.0296** | 0.0120 | [-0.0531, -0.0061] | +0.203 |
| C (Support-local) | CSI300 SOE | 24,950 | 15 | +0.0529 | +0.0764 | **-0.0235** | 0.0122 | [-0.0475, +0.0004] | +0.216 |
| A (Full) | CSI500 SOE | 27,722 | 178 | +0.0619 | +0.0346 | **+0.0273** | 0.0023 | [+0.0227, +0.0318] | +0.203 |
| A (Full) | CSI1000 SOE | 27,722 | 334 | +0.0505 | +0.0261 | **+0.0243** | 0.0022 | [+0.0199, +0.0287] | +0.203 |
| B (Supported SOEs) | CSI1000 SOE | 27,722 | 105 | +0.0455 | +0.0283 | **+0.0171** | 0.0052 | [+0.0070, +0.0273] | +0.203 |
| C (Support-local) | CSI1000 SOE | 24,962 | 105 | +0.0501 | +0.0283 | **+0.0218** | 0.0052 | [+0.0116, +0.0320] | +0.217 |

## Gap Stability Across Specifications

### All A-share SOE

| Spec | N SOE | Gap | SE | 95% CI | Δ from Spec A |
|------|-------|-----|----|--------|---------------|
| A (Full) | 1406 | **+0.0145** | 0.0015 | [+0.0116, +0.0174] | +0.0000 |
| B (Supported SOEs) | 870 | **+0.0134** | 0.0018 | [+0.0098, +0.0170] | -0.0011 |
| C (Support-local) | 870 | **+0.0136** | 0.0018 | [+0.0100, +0.0172] | -0.0009 |

### CSI1000 SOE

| Spec | N SOE | Gap | SE | 95% CI | Δ from Spec A |
|------|-------|-----|----|--------|---------------|
| A (Full) | 334 | **+0.0243** | 0.0022 | [+0.0199, +0.0287] | +0.0000 |
| B (Supported SOEs) | 105 | **+0.0171** | 0.0052 | [+0.0070, +0.0273] | -0.0072 |
| C (Support-local) | 105 | **+0.0218** | 0.0052 | [+0.0116, +0.0320] | -0.0025 |

### CSI300 SOE

| Spec | N SOE | Gap | SE | 95% CI | Δ from Spec A |
|------|-------|-----|----|--------|---------------|
| A (Full) | 117 | **-0.0086** | 0.0043 | [-0.0170, -0.0002] | +0.0000 |
| B (Supported SOEs) | 15 | **-0.0296** | 0.0120 | [-0.0531, -0.0061] | -0.0210 |
| C (Support-local) | 15 | **-0.0235** | 0.0122 | [-0.0475, +0.0004] | -0.0149 |

### CSI500 SOE

| Spec | N SOE | Gap | SE | 95% CI | Δ from Spec A |
|------|-------|-----|----|--------|---------------|
| A (Full) | 178 | **+0.0273** | 0.0023 | [+0.0227, +0.0318] | +0.0000 |

## Key Findings

### 1. Does support restriction change the point estimate?

- **All A-share SOE**: Spec A gap = +0.0145 (n=1406), Spec B gap = +0.0134 (n=870), |Δ| = 0.0011
- **All A-share SOE**: Spec A gap = +0.0145 (n=1406), Spec C gap = +0.0136 (n=870), |Δ| = 0.0009
- **CSI1000 SOE**: Spec A gap = +0.0243 (n=334), Spec B gap = +0.0171 (n=105), |Δ| = 0.0072
- **CSI1000 SOE**: Spec A gap = +0.0243 (n=334), Spec C gap = +0.0218 (n=105), |Δ| = 0.0025
- **CSI300 SOE**: Spec A gap = -0.0086 (n=117), Spec B gap = -0.0296 (n=15), |Δ| = 0.0210
- **CSI300 SOE**: Spec A gap = -0.0086 (n=117), Spec C gap = -0.0235 (n=15), |Δ| = 0.0149

### 2. Does support restriction mainly increase standard error?

- **All A-share SOE**: SE_ratio (B/A) = 1.24× (clusters: 1394 → 870)
- **CSI1000 SOE**: SE_ratio (B/A) = 2.31× (clusters: 331 → 105)
- **CSI300 SOE**: SE_ratio (B/A) = 2.80× (clusters: 116 → 15)

### 3. Which indices depend primarily on extrapolation?

- **All A-share SOE**: 870/1406 (61.9%) in common support → 38% extrapolation → **MODERATE extrapolation dependence**
- **CSI1000 SOE**: 105/334 (31.4%) in common support → 69% extrapolation → **HEAVY extrapolation dependence**
- **CSI300 SOE**: 15/117 (12.8%) in common support → 87% extrapolation → **HEAVY extrapolation dependence**

## Caveats

1. **Common support defined univariately on features, not outcome model**: The NN-distance criterion identifies SOEs whose feature vector is well-represented in the training data. It does not guarantee that the outcome model is well-calibrated for these observations.
2. **Specification C training set size varies**: The support-local model may have substantially fewer training observations than Spec A/B, increasing estimation variance.
3. **Ridge only**: Only Ridge regression is used. Tree-based models (RF, GBM) may respond differently to support restrictions due to their different extrapolation behavior.
4. **No model selection based on results**: All three specifications use pre-specified Ridge(α=10). No specification is preferred over others based on gap magnitude or significance.
