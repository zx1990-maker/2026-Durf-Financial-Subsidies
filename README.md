# 中国 SOE 反事实 EBI/A 研究

使用全 A 股民营企业训练模型，估计国有上市公司（SOE）在"民营企业收益生成过程"下的反事实 EBI/A（资产回报率），并测量 Gap = 预测值 − 实际值。

## 核心结果

- **全 A 股 SOE 的 gap 持续为正**（GBM 主模型约 +1.4 ~ +2.3pp，2019–2025 年 7 年滚动窗口，统计显著）
- **规模梯度**：大市值 SOE（CSI300）gap 为负或接近零；中小市值 SOE（CSI500/CSI1000）gap 为正
- **行业集中**：钢铁、煤炭、家电等重资产行业 gap 最大（+3~7pp）；交通运输、公用事业接近零
- **稳健**：剔除 ST/资不抵债/极端盈亏企业后结论不变；common support 限制、模型类别、归一化方法均不影响结论

## 关键定义

```
Gap = Predicted Private-Firm EBI/A − Actual SOE EBI/A
```

- Gap > 0：SOE 实际 EBI/A 低于民企反事实预测
- Gap < 0：SOE 实际 EBI/A 高于民企反事实预测

## 数据

| 数据 | 位置 | 说明 |
|------|------|------|
| 全 A 股 | `0803/` | 2014–2024，11 年 panel，5,535 firms |
| 指数成分股 | `input_index_weight/` | CSI300/CSI500/CSI1000（2019–2025 进出记录） |

数据来自 Wind 金融终端（2026 年 8 月提取）。

## 方法

- **主模型**：GBM（n=200, depth=4, lr=0.05）
- **稳健性**：Ridge（α=10）、Random Forest
- **特征**：9 个财务比率（FirmSize, AssetTurnover, FinancialLeverage, FixedAssetsRatio, CurrentRatio, CapexRatio, WorkingCapitalRatio, MarketShare, HHI）+ 行业 FE
- **交叉验证**：GroupKFold by firm_id（5 折）
- **样本**：剔除金融、ST、资不抵债、极端盈亏（|EBI/A| > 50%）

## ⚠️ 关键数据约定

**MarketShare/HHI 必须在全 A 股市场 universe 计算**（denominator = 全市场行业营收），不能在指数子样本中计算。早期版本在 CSI500 专属文件中计算导致 denominator 被低估，gap 被系统性夸大（CSI500 +2.73pp → 修正后 +0.10pp）。

## 文件结构

```
Linear-A/
├── 0803/                          # 原始 Wind 数据（勿改）
├── input_index_weight/            # 指数成分股（勿改）
├── run_counterfactual.py          # 数据加载 + 特征构造
├── diagnostics.py                 # 四项诊断分析
├── src/
│   ├── 01_main_pipeline.py        # 主流水线
│   └── README.md
├── outputs/
│   ├── csi300_constituents.json   # 数据依赖
│   ├── diagnostics/               # 民企内部诊断
│   └── revision/                  # ★ 最终结果
│       ├── filtered_rolling.csv   # 筛选后 7 年 rolling（84 行）
│       ├── unified_rolling.csv    # 筛选前（对比）
│       ├── fig_*_screened.*       # 筛选后图表
│       └── *.md                   # 审计报告
├── FINAL_REPORT.md                # ★ 最终报告（Markdown）
├── FINAL_REPORT.pdf               # ★ 最终报告（PDF，11 页）
└── convert_to_pdf.py              # md → PDF 转换脚本
```

## 复现

```bash
# 1. 主流水线（统一 A 股 panel，MarketShare 全市场计算）
python src/01_main_pipeline.py

# 2. 筛选后重跑（剔除 ST/资不抵债/极端盈亏）
python outputs/revision/rerun_filtered.py

# 3. 分行业可视化（筛选后）
python outputs/revision/industry_screened.py

# 4. 报告转 PDF
python convert_to_pdf.py
```

## 报告

- **最终报告**：[FINAL_REPORT.md](FINAL_REPORT.md) 或 [FINAL_REPORT.pdf](FINAL_REPORT.pdf)
- **审计报告**：`outputs/revision/` 下的 sample_audit.md、ownership_definition.md、variable_audit.md、gap_sign_audit.md、support_robustness.md、model_comparison.md

## 论文写作要点

1. 主模型 GBM（拟合最佳 + 校准完美 β=1.04），Ridge/RF 作稳健性（RF 校准失真 β=1.46，不推荐）
2. MarketShare/HHI 全市场计算
3. 训练样本 27,722 obs / 3,471 firms 是"完整案例"口径
4. 所有制是静态快照（look-ahead caveat）
5. 新股、低面值筛选因数据缺失未执行
