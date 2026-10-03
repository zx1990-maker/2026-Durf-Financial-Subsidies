#!/usr/bin/env python3
"""Convert 论文_中国国有企业反事实资产回报率研究.md to styled HTML, then PDF via Chrome headless.

复用 convert_to_pdf.py 的样式与 Chrome 转换逻辑，额外增加 LaTeX 数学公式 → HTML 的预处理。
"""
import re
import markdown
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
md_file = PROJECT / '论文_中国国有企业反事实资产回报率研究.md'
html_file = PROJECT / '论文_中国国有企业反事实资产回报率研究.html'
pdf_file = PROJECT / '论文_中国国有企业反事实资产回报率研究.pdf'


def latex_to_html(s: str) -> str:
    """将 LaTeX 数学片段转为可读 HTML（下标/上标/符号）。"""
    # 1. \text{X} → X
    s = re.sub(r'\\text\{([^{}]+)\}', r'\1', s)
    # 2. 下标 _{...} → <sub>...</sub>，单字符 _x → <sub>x</sub>
    s = re.sub(r'_\{([^{}]+)\}', lambda m: f'<sub>{m.group(1)}</sub>', s)
    s = re.sub(r'_([a-zA-Z0-9])', lambda m: f'<sub>{m.group(1)}</sub>', s)
    # 3. 上标 ^{...} → <sup>...</sup>
    s = re.sub(r'\^\{([^{}]+)\}', lambda m: f'<sup>{m.group(1)}</sup>', s)
    s = re.sub(r'\^([a-zA-Z0-9])', lambda m: f'<sup>{m.group(1)}</sup>', s)
    # 4. \sqrt{X} → √(X)
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'√(\1)', s)
    # 5. \frac{A}{B} → (A)/(B)
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1) / (\2)', s)
    # 6. 符号与希腊字母
    s = s.replace(r'\hat{f}', 'f̂').replace(r'\bar{X}', 'X̄')
    s = s.replace(r'\sum', 'Σ').replace(r'\sigma', 'σ')
    s = s.replace(r'\times', ' × ').replace(r'\cdot', '·').replace(r'\approx', '≈')
    s = s.replace(r'\leq', '≤').replace(r'\geq', '≥').replace(r'\in', '∈').replace(r'\ne', '≠')
    s = s.replace(r'\pm', '±').replace(r'\to', '→')
    # 7. 残留花括号清理
    s = s.replace('{', '').replace('}', '')
    return s


def render_math(text: str) -> str:
    """块级 $$...$$ 与行内 $...$ 的公式预处理。"""
    # 块级公式 → 居中 div
    def block_repl(m):
        return f'\n<div class="math-block">{latex_to_html(m.group(1).strip())}</div>\n'
    text = re.sub(r'\$\$(.+?)\$\$', block_repl, text, flags=re.DOTALL)
    # 行内公式 → span
    def inline_repl(m):
        return f'<span class="math-inline">{latex_to_html(m.group(1).strip())}</span>'
    text = re.sub(r'\$(.+?)\$', inline_repl, text)
    return text


# 读取 + 公式预处理 + markdown 转换
text = md_file.read_text(encoding='utf-8')
text = render_math(text)
body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])

css = """
<style>
@page { margin: 20mm; }
body {
  font-family: -apple-system, 'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'Microsoft YaHei', sans-serif;
  font-size: 11pt;
  line-height: 1.65;
  color: #1a1a1a;
  max-width: 210mm;
  margin: 0 auto;
  padding: 20mm;
  background: #fff;
}
h1 { font-size: 20pt; border-bottom: 2px solid #2C3E50; padding-bottom: 6px; color: #2C3E50; }
h2 { font-size: 15pt; border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #2C3E50; margin-top: 26px; }
h3 { font-size: 12.5pt; color: #34495E; margin-top: 18px; }
h4 { font-size: 11.5pt; color: #555; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0;
  font-size: 8.5pt;
}
th, td {
  border: 1px solid #ccc;
  padding: 5px 8px;
  text-align: left;
}
th { background: #f0f3f5; font-weight: bold; }
tr:nth-child(even) { background: #fafbfc; }
img { max-width: 100%; height: auto; margin: 10px 0; }
code {
  background: #f5f5f5;
  padding: 1px 4px;
  border-radius: 3px;
  font-family: 'SF Mono', Menlo, monospace;
  font-size: 9.5pt;
}
pre {
  background: #f5f5f5;
  padding: 10px;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 9.5pt;
}
pre code { background: none; padding: 0; }
blockquote {
  border-left: 3px solid #2C3E50;
  margin: 10px 0;
  padding: 5px 15px;
  background: #f8f9fa;
  color: #444;
}
strong { color: #000; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
.math-block {
  text-align: center;
  margin: 12px 0;
  font-size: 11pt;
}
.math-inline { font-family: inherit; }
sub { font-size: 72%; line-height: 0; }
sup { font-size: 72%; line-height: 0; }
</style>
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
{css}
</head>
<body>
{body}
</body>
</html>"""

html_file.write_text(html, encoding='utf-8')
print(f'✓ HTML saved: {html_file}')

# Convert HTML to PDF via Chrome headless
import subprocess
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
url = html_file.as_uri()
result = subprocess.run([
    chrome, '--headless', '--disable-gpu', '--no-pdf-header-footer',
    '--print-to-pdf=' + str(pdf_file), url
], capture_output=True, text=True, timeout=120)

if pdf_file.exists():
    size = pdf_file.stat().st_size / 1024
    print(f'✓ PDF saved: {pdf_file} ({size:.0f} KB)')
else:
    print('❌ PDF generation failed')
    print(result.stderr[-2000:] if result.stderr else '')
