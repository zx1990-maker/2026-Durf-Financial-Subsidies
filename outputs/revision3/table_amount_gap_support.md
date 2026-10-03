# TASK 4 — Amount Gap 的支持域分解（亿元）

`AmountGap_it = Gap_it × Assets_it`，单位为亿元（`资产总计_target / 1e8`）。逐年 rolling ex-ante GBM + 5-NN common support 分类后，按 good / boundary / extrapolation 分解。

## 2025 年重点（per sample）

| Sample | Total amount gap | Good | Boundary | Extrapolation | Extrapolation share of amount | Extrapolation share of assets |
|---|---|---|---|---|---|---|
| All A-share SOE | 7559.50 | 1617.20 | 1241.44 | 4700.85 | 62.2% | 78.1% |
| CSI300 | 2155.33 | -36.37 | -44.18 | 2235.88 | 103.7% | 95.6% |
| CSI500 | 1783.15 | 216.54 | 446.44 | 1120.18 | 62.8% | 57.6% |
| CSI1000 | 1197.20 | 485.33 | 225.99 | 485.87 | 40.6% | 39.6% |

## 说明

注意：extrapolation 部分**不**称为“上界”。它仅表示：amount-gap estimate is highly sensitive to observations outside the main private-sector support region。


## 2019–2025 逐年（All SOE）

| Year | Total | Good | Boundary | Extrapolation | Good+boundary | Extrapolation share of amount |
|---|---|---|---|---|---|---|
| 2019 | 7221.7 | 1502.6 | 326.4 | 5392.6 | 1829.1 | 74.7% |
| 2020 | 10467.8 | 2287.5 | 392.8 | 7787.5 | 2680.3 | 74.4% |
| 2021 | 5061.5 | 1781.4 | 329.0 | 2951.2 | 2110.3 | 58.3% |
| 2022 | 5867.4 | 1690.0 | 1100.5 | 3076.9 | 2790.5 | 52.4% |
| 2023 | 6458.6 | 1907.9 | 514.9 | 4035.8 | 2422.8 | 62.5% |
| 2024 | 7953.2 | 2274.2 | 511.4 | 5167.6 | 2785.6 | 65.0% |
| 2025 | 7559.5 | 1617.2 | 1241.4 | 4700.9 | 2858.6 | 62.2% |