# Overleaf 上传清单（LaTeX 版论文）

> 本清单用于把 LaTeX 版论文投到 Overleaf 编译。生成日期：2026-08-19。

## 一、一键上传

上传 **`论文_latex_overleaf.zip`**（已由 `build_overleaf_package.py` 生成）。

压缩包内已包含全部编译所需文件，目录结构与本地一致：

```
论文_中国国有企业反事实资产回报率研究.tex        ← 主文件
final_results/
  fig_firm_size_overlap_screened.png
  fig_amount_gap_by_year.png
  fig_ratio_vs_amount_gap.png
  fig_supp_size_gap_and_calibration.png
outputs/revision/
  fig_summary_7year_screened.png
  fig_industry_year_heatmap_screened.png
  fig_industry_topbottom_trend_screened.png
  fig_industry_year_cs_heatmap_screened.png
  fig_industry_year_cs_trend_screened.png
```

共 10 个文件（1 个 `.tex` + 9 张图），约 5.7 MB。图片为 ASCII 文件名，`.tex` 中文文件名已用 UTF-8 标志存储，Overleaf 可正常解压。

## 二、编译设置

| 项 | 值 |
|----|----|
| 编译器 | **XeLaTeX**（文档类为 `ctexart`，需 XeLaTeX/LuaLaTeX，不能用 pdfLaTeX） |
| 主文件 | `论文_中国国有企业反事实资产回报率研究.tex` |
| 编译次数 | **两遍**（第二遍生成交叉引用，虽然本文无 `\label`/`\ref`，跑一遍即可） |
| 宏包依赖 | `ctex`、`amsmath`、`booktabs`、`graphicx`、`float`、`caption`、`hyperref`、`textcomp`（Overleaf 的 TeX Live 全量自带，无需手动安装） |

> 若 Overleaf 解压后中文主文件名显示为乱码，改为在 Overleaf 项目里把主文件重命名为任意 ASCII 名（如 `main.tex`）即可，文档标题由 `\title{…}` 决定，不受文件名影响。

## 三、重新生成压缩包（可复现）

```bash
python3 build_overleaf_package.py
```

脚本自动从 `.tex` 里解析 `\includegraphics{…}` 路径、校验图片存在、并用 UTF-8 安全方式打包。

## 四、⚠️ 需要清理的问题（与论文内容无关，但影响仓库健康）

以下是在打包过程中发现的、与论文正确性无关但建议处理的工程问题：

1. **【严重】Git 仓库根目录在 `~`（家目录），不在本项目目录。**
   `git rev-parse --show-toplevel` = `/Users/violet`（而非 `/Users/violet/Desktop/DURF/模型/Linear-A`）。
   这导致：① `git status` 在扫描整个家目录（出现 `.cups/`、`.Trash/`、`Documents/` 权限告警）；② 暂存区里有大量 `~/.vscode/extensions/tonybaloney.vscode-pets-1.35.0/*` 的删除记录（误操作）。当前项目文件（代码、论文）**都还没被版本控制**（`git ls-files` 为空）。
   **建议**：在确认无误后，将仓库根改为项目目录（见下方命令），或删除 `~/.git` 后在项目目录重新 `git init`。

2. **【重复目录】`final_results 2/`**：是 `final_results/` 的过期副本（少 5 个文件：`_path_test*.csv`、`_sandbox_write_test.csv`、`summary_by_sample_2025.csv`、`summary_by_year_all_soe.csv`）。疑为拖拽误复制，可删除。

3. **【调试残留】`final_results/_path_test.csv`、`_path_test2.csv`、`_sandbox_write_test.csv`**：是修复 OUTPUT_DIR 覆盖 bug 时留下的路径写测试文件，可删除。

4. **【编译产物】`__pycache__/`、`.DS_Store`、`final_results.zip`**：均为缓存/归档，已加入 `.gitignore`。

### 修正 Git 仓库根目录（供参考，未经确认不会自动执行）

```bash
# 方案 A：把当前误建的 ~/.git 仓库迁移到项目目录（保留唯一那次提交）
#   —— 更简单、更安全的做法是 B。
# 方案 B：直接丢弃 ~/.git，在项目目录重新初始化
#   rm -rf ~/.git            # ⚠️ 会删除家目录下的 .git（含那次 commit），请先确认
#   cd /Users/violet/Desktop/DURF/模型/Linear-A
#   git init
#   git add -A && git commit -m "初始提交：SOE 反事实资产回报率研究"
```

> 已提供 `.gitignore`（见项目根目录），在仓库根改到项目目录后即生效，会排除 `0803/`（原始 Wind 数据）、`final_results/`、`outputs/` 下的生成结果、备份与缓存，仅保留代码（含 `outputs/revision*/**/*.py`）与论文文稿（`.md`/`.tex`/`.html`）。
