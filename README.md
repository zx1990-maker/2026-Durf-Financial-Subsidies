# 三套 benchmark 反事实 EBI/A 模型代码包（Panel Data 版本）

比较三套 non-SOE benchmark 对 A 股 SOE 企业的反事实 EBI/A 预测：

1. **H_nonSOE** — H 股非 SOE benchmark
2. **CSI_nonSOE** — CSI300 非 SOE benchmark
3. **Pooled** — H 股 + CSI300 非 SOE pooled benchmark

## 项目结构

```
three_benchmark_model_package/
├── config/
│   └── default.yaml              # 所有可配置参数
├── src/
│   ├── utils.py                  # 工具函数（数值转换、加权平均、gap 计算）
│   ├── data_loader.py            # 数据加载（yearly / legacy / panel 三种模式）
│   ├── features.py               # 特征工程（滞后特征、固定效应、winsorization）
│   ├── model.py                  # 模型（Ridge pipeline、CV、fit/predict）
│   └── outputs.py                # 输出生成（CSV + Excel）
├── scripts/
│   └── run_benchmarks.py         # CLI 入口
├── data_input/
│   ├── 2018/                     # 2018 年数据
│   │   ├── csi_features_2018.csv
│   │   └── h_features_2018.csv
│   ├── 2019/                     # ...
│   ├── ...                       # 2018–2024 各年文件夹
│   └── 2024/                     # 2024 年（兼容旧格式自动转换）
├── outputs/                      # 输出文件
├── requirements.txt
└── README.md
```

## 数据格式

### Yearly 模式（推荐，2018–2024）

每年一个子文件夹，包含两个 CSV 文件：

| 文件 | 内容 |
|------|------|
| `csi_features_YYYY.csv` | CSI 市场全部企业（SOE + 非SOE 混合） |
| `h_features_YYYY.csv` | H 股市场全部企业 |

列结构（年后缀为当年，如 `_2023`、`_2024`）：

| 列 | 说明 |
|----|------|
| 市场 | CSI / H |
| 证券代码 | 股票代码 |
| 证券简称 | 企业名称 |
| 一级行业 | 行业大类 |
| 二级行业 | 行业细分（**按此列过滤金融**） |
| 行业全路径 | 完整行业路径（`--` 分隔） |
| 企业所有制性质 | 地方国有企业 / 中央国有企业 / 民营企业 等 |
| 资产总计_YYYY | 总资产（特征年） |
| 营业收入_YYYY | 营业收入（特征年） |
| 归母净利润_(YYYY+1) | 归母净利润（目标年） |
| 利息支出_(YYYY+1) | 利息支出（目标年） |
| 资产总计_(YYYY+1) | 总资产（目标年，权重变量） |
| EBI_A_(YYYY+1) | EBI/A（目标变量） |
| FirmSize_YYYY | ln(总资产) |
| MarketShare_YYYY_二级行业 | 二级行业市场份额 |
| HHI_YYYY_二级行业 | 二级行业 HHI |
| AssetTurnover_YYYY | 资产周转率 |
| WorkingCapitalRatio_YYYY | 营运资本比率 |
| FixedAssetsRatio_YYYY | 固定资产比率 |
| FinancialLeverage_YYYY | 财务杠杆 |
| CurrentRatio_YYYY | 流动比率 |
| CapexRatio_YYYY | 资本支出比率 |

### 2024 旧格式兼容

2024 年文件夹如为旧格式（`csi——features.csv` / `h——features.csv` / `.xlsx`），加载器会**自动检测并转换**为统一格式。

## 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# === Yearly 模式（推荐，使用 2018–2024 全部年份）===
python scripts/run_benchmarks.py yearly

# 指定年份
python scripts/run_benchmarks.py yearly --years 2023 2024

# === Legacy 模式（仅用原始 2024→2025 单期数据）===
python scripts/run_benchmarks.py legacy

# === 覆盖超参数 ===
python scripts/run_benchmarks.py yearly --alpha 5.0 --verbose
```

## 模型设定

| 参数 | 设定 |
|------|------|
| 模型 | Ridge(alpha=10) |
| 预测框架 | X_{t-1} → Y_t（滞后一期特征预测当期 EBI/A） |
| 预测样本 | 2019–2025（使用 2018–2024 特征） |
| 目标变量 | EBI/A（t 期） |
| 数值特征 | FirmSize, MarketShare, HHI, AssetTurnover, WorkingCapitalRatio, FixedAssetsRatio, FinancialLeverage, CurrentRatio, CapexRatio（均为 t-1 滞后值） |
| 类别特征 | 一级行业, 二级行业 |
| Pooled 额外特征 | 市场虚拟变量（H vs CSI） |
| Winsorization | 5%–95%（主结果），另输出 2.5%–97.5%、1%–99% 稳健性 |
| CV 策略 | Firm-level GroupKFold（5 折，防止同企业跨年泄露） |

## 金融行业过滤

按 `二级行业` 过滤，默认排除以下行业（可在 `config/default.yaml` 中修改）：

```
银行、保险、证券、多元金融、金融、非银金融
```

## 数据概览（2018–2024）

| 分组 | 企业数 | 说明 |
|------|--------|------|
| SOE_target | 118 | CSI A 股 SOE（中央国有 + 地方国有） |
| H_nonSOE | ~281 | H 股非 SOE benchmark |
| CSI_nonSOE | ~130 | CSI 非 SOE benchmark |
| **总计** | **~529** | 每年过滤金融后约 666 行 |

## 主要结果（2018–2024 Panel, 5%-95% winsor）

| Benchmark | 训练样本 | 实际均值 | 预测均值 | Mean Gap | 资产加权 Gap | 正向占比 |
|-----------|----------|----------|----------|----------|-------------|----------|
| H_nonSOE | 1,139 | 0.0573 | 0.0648 | +0.75pp | +2.26pp | 59.6% |
| CSI_nonSOE | 685 | 0.0573 | 0.0533 | −0.41pp | −0.54pp | 42.7% |
| **Pooled** | **1,824** | 0.0573 | 0.0804 | **+2.31pp** | **+3.88pp** | **70.2%** |

## 输出文件

| 文件 | 说明 |
|------|------|
| `summary_main_5_95.csv` | 主汇总结果 |
| `summary_robustness.csv` | 稳健性汇总（3 种 winsor） |
| `cv_metrics_robustness.csv` | 交叉验证指标 |
| `predictions_main_long.csv` | 企业级预测（长格式，含年份） |
| `predictions_main_compact.csv` | 企业级预测（宽格式，3 benchmark 并列） |
| `group_ownership.csv` | 按所有制分组统计 |
| `group_industry.csv` | 按二级行业分组统计 |
| `support_diagnostics.csv` | 共同支撑检验 |
| `data_summary.csv` | 数据概览 |
| `three_benchmark_counterfactual_results.xlsx` | 多工作表 Excel 汇总 |

## Gap 定义

```text
return_gap = predicted_EBI_A − actual_EBI_A
implied_gap_amount = return_gap × 资产总计
positive_gap_amount = max(return_gap, 0) × 资产总计
```

## 配置说明

编辑 `config/default.yaml` 可调整：

- `exclude_industries` — 需过滤的二级行业
- `features.fixed_effects` — 固定效应开关（year / industry / industry×year / market）
- `model.params.alpha` — Ridge 正则化强度
- `winsorization` — Winsorization 分位数
- `cv.strategy` — 交叉验证策略（`group` 或 `random`）
