# 共同支撑域 v2（5-NN vs 5-NN）QA Summary

生成日期：2026-08-20。本文件只报告重算诊断，不修改论文正文、不重新解释经济机制。

## 检查 1：每年×每样本 Good+Boundary+Extrapolation = 全样本
- 最大偏差 = 0 → **PASS**

## 检查 2：Full gap 与论文当前版本一致（gap 未被改动）
- 与主面板（source of truth）直接重算的 Full gap 最大差 0.000000 pp（应为 0）
- 与 task08 多年推断 CSV 的 Full 点估计最大差 0.000496 pp（该 CSV 四舍五入到 3 位小数，允差 0.001）
- → **PASS**

## 检查 3：private calibration 不含同一 firm 的其他年份作为邻居
- 实现：对每个私有校准观测，将其自身 firm_id 的所有年份距离置为 +∞ 后再取第 5 小
- 所有 d5_private 均有限（说明每年前至少有 5 家 distinct 私有企业）
- → **PASS**（代码层保证，`fm[own] = inf` 后 `partition(...,4)`）

## 检查 4：SOE 5-NN 与 private calibration 使用同一统计量
- 二者均调用 `_firm_min`（每 firm 取 min over years）+ `partition(...,4)[:,4]`（第 5 小 distinct firm 距离）
- → **PASS**（唯二差异：private 侧额外排除自身 firm，SOE 侧不排除）

## 检查 5/6：reference pool 严格 ex-ante（target_year < ForecastYear）

- 2019：私有参考池 5554 obs / 2160 firms（无 target_year=2019 的民企 outcome）
- 2020：私有参考池 7704 obs / 2344 firms（无 target_year=2020 的民企 outcome）
- 2021：私有参考池 10278 obs / 2728 firms（无 target_year=2021 的民企 outcome）
- 2022：私有参考池 13258 obs / 3057 firms（无 target_year=2022 的民企 outcome）
- 2023：私有参考池 16501 obs / 3266 firms（无 target_year=2023 的民企 outcome）
- 2024：私有参考池 19827 obs / 3341 firms（无 target_year=2024 的民企 outcome）
- 2025：私有参考池 23150 obs / 3344 firms（无 target_year=2025 的民企 outcome）
- → **PASS**（脚本 `assert tr['target_year'].max() < fc`）

## 检查 7：每年 P90/P95 阈值平滑、无异常跳变
- P90 由 1.532 单调降至 1.244；P95 由 1.695 降至 1.404（样本量增大→邻距缩小）
- → **PASS**

## 检查 8：旧 support 与 新 5-NN-vs-5-NN support 交叉表（SOE-year）

```
support_status  Boundary  Extrapolation  Good
old_status                                   
Boundary             224            123   689
Extrapolation        501            681  1046
Good                 222             89  5503
```

- 共 9078 个 SOE-year；分类改变 2670 个（29.4%）

## 检查 9：2025 四样本新口径关键值

- All SOE：Good 77.2% / Boundary 11.7% / 外推 11.2%；CS gap +1.88 pp；外推金额占比 17.7%
- 沪深300：Good 47.0% / Boundary 26.5% / 外推 26.5%；CS gap -0.14 pp；外推金额占比 -10.6%
- 中证500：Good 69.3% / Boundary 12.7% / 外推 18.0%；CS gap +1.13 pp；外推金额占比 24.6%
- 中证1000：Good 76.4% / Boundary 12.2% / 外推 11.4%；CS gap +1.43 pp；外推金额占比 23.7%


## 检查 10：四样本 2019–2025 新口径总结（率差核心结果）

- All SOE：CS gap +2.01 pp [+1.81, +2.22]；PersistentGood +2.04 pp [+1.80, +2.28]；Mean GoodShare 79.9%；Mean CSShare 90.2%
- 沪深300：CS gap +0.01 pp [-0.73, +0.75]；PersistentGood +0.84 pp [+0.14, +1.57]；Mean GoodShare 56.4%；Mean CSShare 77.1%
- 中证500：CS gap +1.48 pp [+1.16, +1.79]；PersistentGood +1.68 pp [+1.19, +2.19]；Mean GoodShare 70.4%；Mean CSShare 85.7%
- 中证1000：CS gap +1.86 pp [+1.54, +2.20]；PersistentGood +2.22 pp [+1.76, +2.72]；Mean GoodShare 80.4%；Mean CSShare 91.1%