# 代码结构说明

本目录整理最终交付代码。所有脚本可复现，依赖以下数据目录：

- `../0803/` — 原始 Wind 数据（11 年全 A 股 + CSI500/CSI1000）
- `../input_index_weight/` — 指数成分股进出记录

## 脚本

| 位置 | 作用 | 说明 |
|------|------|------|
| `src/01_main_pipeline.py` | 统一分析流水线 | 统一用 A 股 panel，MarketShare/HHI 全市场计算 |
| `../run_counterfactual.py` | 数据加载 + 特征构造 | 原始解析函数（XML → firm-year → 特征） |
| `../diagnostics.py` | 四项诊断分析 | 系统偏差/共同支持/Placebo/Bootstrap |
| `../outputs/revision/rerun_filtered.py` | 筛选后重跑 | 剔除 ST/资不抵债/极端盈亏 |
| `../outputs/revision/industry_screened.py` | 分行业可视化 | 筛选后分行业 gap 图 |

## 运行方式

```bash
# 主分析（rolling 2019-2025，3 模型 × 4 样本）
python src/01_main_pipeline.py

# 筛选后重跑
python outputs/revision/rerun_filtered.py

# 分行业可视化
python outputs/revision/industry_screened.py
```

## 关键约定

1. **Gap = Predicted − Actual**（正 gap = SOE 实际低于民企反事实预测）
2. **训练集** = 民营企业（完整案例，剔除金融/ST/资不抵债/极端盈亏）
3. **特征** = 9 个财务比率 + 行业 FE，均测于 t 年，预测 t+1
4. **MarketShare/HHI 必须全市场计算**（denominator = 全 A 股行业营收）
5. **超参数预先固定**，从未用 SOE EBI/A 选模型
6. **主模型 GBM**，Ridge/RF 作稳健性

## 最终结果

- `../outputs/revision/filtered_rolling.csv` — 筛选后 7 年 rolling（84 行）
- `../FINAL_REPORT.pdf` — 最终报告
