# TASK 2 — 多年 Common Support 主表（2019–2025）

逐年 rolling ex-ante GBM（`target_year < fc_year`）得到 firm-year gap，逐年做 5-NN common-support 分类（good / boundary / extrapolation），再跨年汇总；**未**把 2019–2025 pooled 后重算统一 support。

`CS` = good + boundary；`Persistent-support` = SupportShare_i ≥ 0.8；`persistent-extrapolation` = SupportShare_i ≤ 0.2。gap 单位为比例（EBI/A 之差），表内 ×100 即百分点。

| Sample | Full annual mean gap | CS annual mean gap | Good-only mean gap | Firm-level full | Firm-level CS | Persistent gap | Avg good-support % | Persistent % | N firms |
|---|---|---|---|---|---|---|---|---|---|
| All A-share SOE | 0.0202 | 0.0210 | 0.0211 | 0.0196 | 0.0206 | 0.0187 | 64.4 | 52.8 | 1351 |
| CSI300 | 0.0007 | -0.0106 | -0.0136 | 0.0036 | 0.0021 | 0.0079 | 26.5 | 20.2 | 163 |
| CSI500 | 0.0162 | 0.0154 | 0.0146 | 0.0173 | 0.0170 | 0.0155 | 50.2 | 41.2 | 374 |
| CSI1000 | 0.0188 | 0.0200 | 0.0205 | 0.0215 | 0.0231 | 0.0236 | 69.0 | 59.0 | 583 |

## 客观总结

All SOE 的多年平均 gap：full 样本 0.0202、common-support（good+boundary）0.0210、persistent-support（SupportShare≥0.8）企业 0.0187。三者方向一致，说明仅保留长期具有民营企业可比对象的国企后，条件回报缺口仍存在。