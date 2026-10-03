# =============================================================================
#  论文 LaTeX 编译脚本
#  需求：xelatex（须含 ctex 宏包，推荐 TinyTeX 或 MacTeX）
#  用法：make           —— 编译生成 main.pdf（自动运行两次以解析交叉引用）
#        make clean     —— 清理中间文件
# =============================================================================

MAIN = main
PDF  = $(MAIN).pdf

all: $(PDF)

$(PDF): $(MAIN).tex $(wildcard figures/*.pdf)
	xelatex -interaction=nonstopmode $(MAIN).tex
	xelatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.fls *.fdb_latexmk

.PHONY: all clean
