# Asset Beta Control Variable Experiment

## 1. Objective

We introduced **Asset Beta** as an additional control variable to better capture firm-level business risk.

- Equity Beta is affected by capital structure.
- Asset Beta removes the leverage effect and is therefore closer to operating risk.
- The experiment tests whether differences between H-share private firms and CSI 300 state-owned enterprises are partly explained by differences in risk.

The main questions are:

> Does adding Asset Beta improve predictive performance?  
> Does it materially change the counterfactual gap?

## 2. Experimental Design

### Data

- Asset Beta coverage: **2022–2024**
- Equity Beta estimated from weekly stock returns over **2020–2024**
- Asset Beta calculated using the **Hamada adjustment**
- Minimum requirement: **104 valid weekly return observations**

| Item | Value |
|---|---:|
| Asset Beta coverage | 2022–2024 |
| Valid Beta observations | 1,611 / 2,469 (65.2%) |
| Main reasons for missing Beta | Insufficient weekly observations or low estimation quality |
| Prediction period | 2023–2025 |

The baseline model uses seven years of data, while the Asset Beta experiment covers only three years. This substantially reduces the training sample.

## 3. Model Comparison

### Baseline
- Excludes Asset Beta
- Uses the full available sample

### Restricted
- Excludes Asset Beta
- Uses only observations for which Asset Beta is available
- Separates the effect of sample restriction from the effect of adding Beta

### + Asset Beta
- Uses the same restricted sample
- Adds Asset Beta as an explanatory variable

## 4. Results

### 4.1 Training Sample

| Model | Training Observations |
|---|---:|
| Baseline | 1,139 |
| Restricted | 130 |
| + Asset Beta | 130 |

The usable training sample falls by approximately **89%**.

### 4.2 Predictive Performance

Adding Asset Beta did not improve prediction:

- Cross-validation R² remained negative.
- Winsorized RMSE was almost unchanged.
- In-sample R² increased only slightly.

Overall, Asset Beta provided little additional predictive value.

### 4.3 Counterfactual Gap

| Model | Mean Gap |
|---|---:|
| Baseline | +0.75 pp |
| Restricted | −1.18 pp |
| + Asset Beta | −1.02 pp |

The change from Baseline to Restricted was approximately **1.93 percentage points**, while the additional change after including Asset Beta was only **0.16 percentage points**.

This indicates that most of the gap change was caused by the restricted sample rather than by Asset Beta itself.

### 4.4 Asset Beta Coefficient

The Ridge coefficient on Asset Beta was approximately **−0.026**.

This implies that a one-unit increase in Asset Beta is associated with an average decline of about **2.6 percentage points in EBI/A**.

However:

- the effect is modest,
- it contributes little to overall prediction,
- and its negative sign is not consistent with the conventional expectation that higher systematic risk should be associated with higher returns.

This may reflect H-share market characteristics or measurement noise in the Beta estimates.

## 5. Interim Conclusion

The current evidence suggests:

1. Asset Beta does not materially improve predictive performance.
2. Most changes in the counterfactual gap are driven by sample restriction.
3. The available sample is too small to support strong conclusions.

The main limitation is that only three years are covered and the usable H-share training sample contains only 130 observations, which is insufficient for a model with more than 40 features.

Therefore, Asset Beta should not currently be used as a core control variable in the main model.

## 6. Next Steps

- Retain the current baseline model as the main specification.
- Use Asset Beta as a robustness check.
- Obtain longer historical price series and extend Beta coverage to at least **2019–2025**, ideally longer.
- Re-estimate the model once the Beta sample is sufficiently large.
- Reassess whether Asset Beta improves prediction or changes the counterfactual gap.

## Final Takeaway

Asset Beta was successfully constructed, but under the current data constraints it does not meaningfully improve model performance.

The observed gap changes are mainly explained by the reduced sample rather than by the Beta variable itself.

> Asset Beta is best treated as a robustness variable, not as a core control in the current main model.
