# TASK 10 — Output Consistency Check

## 1. firm_level schema

Columns of `firm_level_gap_all_years.csv`:
`year, firm_id, firm_name, industry_l2, ownership, index_membership, total_assets_target_yuan, actual_ebia_ratio, predicted_ebia_ratio, actual_ebi_yuan, predicted_ebi_yuan, gap_ratio, gap_amount_yuan, gap_amount_yi`

## 2. firm-level -> summary reconciliation (2019–2025)

| year | All n | Σ(4 sub) n | Δn | All mean_gap | All Σamount(亿) | Σ(4 sub) amount | Δamount |
|---|---|---|---|---|---|---|---|
| 2019 | 1188 | 1188 | 0 | +0.0190 | 6,962.7 | 6,962.7 | 0.0 |
| 2020 | 1241 | 1241 | 0 | +0.0233 | 8,682.5 | 8,682.5 | 0.0 |
| 2021 | 1301 | 1301 | 0 | +0.0155 | 4,649.2 | 4,649.2 | 0.0 |
| 2022 | 1331 | 1331 | 0 | +0.0145 | 4,307.9 | 4,307.9 | 0.0 |
| 2023 | 1342 | 1342 | 0 | +0.0135 | 5,051.5 | 5,051.5 | -0.0 |
| 2024 | 1339 | 1339 | 0 | +0.0176 | 4,960.2 | 4,960.2 | 0.0 |
| 2025 | 1336 | 1336 | 0 | +0.0159 | 5,979.1 | 5,979.1 | -0.0 |

Δn and Δamount should both be 0 if the decomposition is mutually exclusive and exhaustive.

## 3. summary_by_sample_year.csv vs firm-level aggregation

- sample-year cells checked: 28 (4 subsamples) + 7 (All SOE)
- n_firms mismatches: 0
- mean_gap_ratio mismatches (>1e-9): 0
- sum_gap_amount_yi mismatches (>1e-6): 0

**Result:** PASS — summary is an exact aggregation of the firm-level file.

## 4. firm_level_gap_2025.csv == firm_level_gap_all_years.csv[year==2025]

- rows in 2025 file: 1336; rows in all-years[year==2025]: 1336
- value mismatches on gap/amount/ebia columns: 0

**Result:** PASS — identical.

## 5. n reconciliation (2025 All SOE)

- summary n_firms (2025 All SOE): 1336
- inference_stats n_clusters (All SOE): 1336
- central_vs_local: Central 436 + Local 900 = 1336

**Result:** PASS — n is consistent across files.

## 6. Headline amount figures

- 2025 All SOE amount gap = 5,979.1 亿元 = 6.0 billion RMB
- Cumulative 2019–2025 = 40,593.2 亿元 = 4.06 trillion RMB

**Result:** matches the paper headline (597.9 billion in 2025; ~4.06 trillion cumulative).

## 7. Cross-check: CSI300 ratio vs amount sign

- 2025 CSI300 SOE: mean_gap_ratio = -0.0036 (negative → SOE outperforms), but sum_gap_amount_yi = +1,307.3 亿元 (positive).
- Explanation: the mean ratio is equal-weighted across firms; the amount is asset-weighted, so a few very large firms with positive gaps dominate the amount sum. Not an error, but the two metrics can differ in sign; report them separately.
