#!/usr/bin/env python3
"""Convert FINAL_REPORT.md to styled HTML, then PDF via Chrome headless."""
import markdown
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
md_file = PROJECT / 'FINAL_REPORT.md'
html_file = PROJECT / 'FINAL_REPORT.html'
pdf_file = PROJECT / 'FINAL_REPORT.pdf'

# Read markdown
text = md_file.read_text(encoding='utf-8')

# Convert to HTML
body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])

# CSS for academic Chinese report
css = """
<style>
@page { margin: 20mm; }
body {
  font-family: -apple-system, 'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'Microsoft YaHei', sans-serif;
  font-size: 11pt;
  line-height: 1.6;
  color: #1a1a1a;
  max-width: 210mm;
  margin: 0 auto;
  padding: 20mm;
  background: #fff;
}
h1 { font-size: 20pt; border-bottom: 2px solid #2C3E50; padding-bottom: 6px; color: #2C3E50; }
h2 { font-size: 15pt; border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #2C3E50; margin-top: 24px; }
h3 { font-size: 12.5pt; color: #34495E; margin-top: 18px; }
h4 { font-size: 11.5pt; color: #555; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0;
  font-size: 9pt;
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
