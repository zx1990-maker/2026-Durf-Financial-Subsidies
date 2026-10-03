# 中国国有企业的反事实资产回报率缺口（LaTeX 版）

基于民营企业基准的机器学习估计 · 中文论文 LaTeX 源文件与编译成品。

## 目录结构

```
latex/
├── main.tex                     # 论文主源文件（ctexart，需 xelatex 编译）
├── main.pdf                     # 已编译的 PDF（19 页）
├── figures/                     # 9 张图（PDF 矢量格式）
│   ├── fig1_summary_7year.pdf           # 逐年 gap（4 样本 × 3 模型）
│   ├── fig2_industry_heatmap.pdf        # 行业 × 年份 gap 热力图
│   ├── fig3_topbottom_trend.pdf         # 前 6 / 后 6 行业趋势
│   ├── fig4_amount_gap_by_year.pdf      # 金额缺口分年份分解
│   ├── fig5_ratio_vs_amount.pdf         # 比例 vs 金额缺口
│   ├── figA1_firm_size_overlap.pdf      # 企业规模重叠（§4.4）
│   ├── figA2_size_gap_calibration.pdf   # 规模-gap 分箱 + 校准散点（附录）
│   ├── figA3_cs_heatmap.pdf             # Common-Support 行业热力图（附录）
│   └── figA4_cs_trend.pdf               # Common-Support 行业趋势（附录）
├── Makefile                     # 编译脚本
├── README.md                    # 本说明
├── export_descriptives.py       # 描述性统计表（表1）复现脚本
└── descriptives_by_ownership.csv # 表1 的数据来源
```

## 编译方法

本论文使用 **xelatex + ctex** 编译（中文支持）。推荐安装 [TinyTeX](https://yihui.org/tinytex/)
（`curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh`）或 MacTeX，并安装以下宏包：

```bash
tlmgr install ctex booktabs threeparttable multirow caption enumitem setspace float xcolor geometry hyperref
```

随后在 `latex/` 目录下执行：

```bash
make            # 或：xelatex -interaction=nonstopmode main.tex （运行两次）
```

## 投稿前需自行修改

1. **作者信息**：`main.tex` 中 `\author{辛紫瑄}` 与 `\thanks{...}` 为占位信息，请替换为真实姓名、单位与基金致谢。
2. **参考文献**：文末 29 条参考文献已依据 `source/` 目录下 `reference 1/2/3.zip` 三份压缩包中的原文 PDF 提取整理（作者、标题、期刊、卷期页码与 DOI 均与原文核对）。个别期刊（如 *Corporate Governance*、*Public Choice*、*Management and Organization Review*）的期号未在题名页显示，按“卷、页码”著录，正式投稿前请补全期号并核验 DOI 的最新版本。
3. **中文字体**：`main.tex` 已设定 `\setCJKmainfont{PingFang SC}`；若在其他操作系统编译，请改为本机可用的中文字体（如 Windows 的 `SimSun`、Linux 的 `Noto Serif CJK SC`）。

## 与结果的可复现性

- 论文中所有数字均来自 `final_results/` 下的结果 CSV，与 `outputs/revision/` 下的复现脚本一一对应。
- 表 1（描述性统计）由 `export_descriptives.py` 从原始数据生成，输出 `descriptives_by_ownership.csv`。
- 图件的生成脚本见论文附录 C（复现代码清单）。

## 关于口径的重要说明（实事求是）

- Gap = 反事实预测值 − 实际值，度量的是“以民营企业为基准的资产回报率差异”，**不直接等同于政府补贴或政策性负担的金额**。
- 沪深 300 大市值 SOE 的共同支撑域严重不足（良好支持仅 13.7%），其 gap 估计依赖外推，精确幅度应谨慎对待。
