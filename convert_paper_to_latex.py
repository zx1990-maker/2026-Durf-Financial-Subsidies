#!/usr/bin/env python3
"""Convert 论文_中国国有企业反事实资产回报率研究.md → 论文_..._.tex (ctexart + xelatex).

Produces a complete, compilable LaTeX source for the Chinese academic paper.
Mirrors the style of convert_paper_to_pdf.py but targets LaTeX instead of HTML.

Compile with (requires a TeX distribution with ctex + a CJK font):
    xelatex 论文_中国国有企业反事实资产回报率研究.tex
    xelatex 论文_中国国有企业反事实资产回报率研究.tex   # twice for cross-refs
"""
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
md_file = PROJECT / '论文_中国国有企业反事实资产回报率研究.md'
tex_file = PROJECT / '论文_中国国有企业反事实资产回报率研究.tex'

text = md_file.read_text(encoding='utf-8')

# ---------------------------------------------------------------------------
# Phase 1: protect math spans so later text transforms never touch them.
#   $$ ... $$  ->  block math
#   $ ... $    ->  inline math
# ---------------------------------------------------------------------------
math_blocks = []  # list of (is_block, content)


def _protect_block(m):
    math_blocks.append((True, m.group(1).strip()))
    return f'\x00MB{len(math_blocks) - 1}\x00'


def _protect_inline(m):
    math_blocks.append((False, m.group(1).strip()))
    return f'\x00MI{len(math_blocks) - 1}\x00'


text = re.sub(r'\$\$(.+?)\$\$', _protect_block, text, flags=re.DOTALL)
text = re.sub(r'\$(.+?)\$', _protect_inline, text, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Unicode -> LaTeX (text mode).  Math is already protected, so these apply only
# to prose / table cells / captions.
# ---------------------------------------------------------------------------
UNICODE_MAP = {
    '–': '--',                     # en dash
    '—': '---',                    # em dash
    '−': '-',                      # minus sign (U+2212) -> hyphen (safe in text & tabular)
    '×': r'$\times$',              # multiplication sign
    '→': r'$\to$',                 # right arrow
    '§': r'\S',                    # section sign
    '²': r'\textsuperscript{2}',   # superscript two
    'Ⅱ': 'II',                     # Roman numeral two (industry suffixes)
    '≈': r'$\approx$',             # almost equal
    '°': r'\textdegree{}',          # degree (textcomp; no '^' so _escape_specials is safe)
    '∉': r'$\notin$',              # not element of
    'Σ': r'$\Sigma$',              # capital sigma
    '≤': r'$\le$',                 # less-or-equal
    'Δ': r'$\Delta$',              # capital delta
    '±': r'$\pm$',                 # plus-minus
    '′': "'",                      # prime (appendix A′)
    'β': r'$\beta$',               # beta
    'α': r'$\alpha$',              # alpha
    '·': r'$\cdot$',               # middle dot
}
for _u, _r in UNICODE_MAP.items():
    text = text.replace(_u, _r)


# ---------------------------------------------------------------------------
# Inline transforms (apply per text piece / table cell).
# ---------------------------------------------------------------------------
def _escape_specials(s: str) -> str:
    # Escape LaTeX-special characters in TEXT mode.  Math is protected by
    # placeholders, so these only hit prose / table cells / captions.
    # NOTE: do NOT escape '\' here — the source prose has no literal backslash
    # (only math does, and that's protected), and escaping it would corrupt the
    # \textbf/\emph/\texttt commands introduced just above.
    s = s.replace('%', r'\%')
    s = s.replace('#', r'\#')
    s = s.replace('&', r'\&')
    s = s.replace('_', r'\_')
    s = s.replace('^', r'\textasciicircum{}')
    s = s.replace('~', r'\textasciitilde{}')
    return s


def _restore_math(s: str) -> str:
    def rep(m):
        idx = int(m.group(1))
        is_block, content = math_blocks[idx]
        return f'\\[{content}\\]' if is_block else f'\\({content}\\)'
    s = re.sub(r'\x00MB(\d+)\x00', rep, s)
    s = re.sub(r'\x00MI(\d+)\x00', rep, s)
    return s


def inline_convert(s: str) -> str:
    """Apply all inline transforms to a text piece (no line-structure handling)."""
    s = s.replace(r'\*', r'\textasteriskcentered{}')  # markdown escaped asterisk
    # inline code
    s = re.sub(r'`([^`]+)`', lambda m: r'\texttt{' + m.group(1) + '}', s)
    # bold
    s = re.sub(r'\*\*(.+?)\*\*', lambda m: r'\textbf{' + m.group(1) + '}', s)
    # italic
    s = re.sub(r'\*([^*\n]+?)\*', lambda m: r'\emph{' + m.group(1) + '}', s)
    s = _escape_specials(s)
    s = _restore_math(s)
    return s


# ---------------------------------------------------------------------------
# Table rendering.
# ---------------------------------------------------------------------------
def _cell_align(cell: str) -> str:
    # right-align a column if a majority of its (non-empty) data cells are numeric.
    return 'r' if re.match(r'^[+\-0-9.()（−]', cell.strip()) else 'l'


def render_table(table_lines, caption=None):
    """table_lines: raw markdown pipe-table lines (including header + separator)."""
    rows = []
    for ln in table_lines:
        body = ln.strip().strip('|')
        rows.append([c.strip() for c in body.split('|')])
    header = rows[0]
    sep = rows[1] if len(rows) > 1 else []
    data = rows[2:] if len(rows) > 2 else []

    ncol = len(header)
    align = []
    for j in range(ncol):
        # explicit alignment from separator (:---: / ---: / :---)
        hint = sep[j].strip() if j < len(sep) else ''
        if hint.startswith(':') and hint.endswith(':'):
            align.append('c')
        elif hint.endswith(':'):
            align.append('r')
        elif hint.startswith(':'):
            align.append('l')
        else:
            cells = [r[j] for r in data if j < len(r) and r[j].strip()]
            n_num = sum(1 for c in cells if _cell_align(c) == 'r')
            align.append('r' if cells and n_num / len(cells) >= 0.5 else 'l')
    colspec = ''.join(align)

    def cell(c):
        return inline_convert(c)

    lines = []
    wide = ncol >= 6
    lines.append('\\begin{table}[H]' if caption else '\\begin{center}')
    lines.append('\\centering')
    lines.append('\\footnotesize' if wide else '\\small')
    if caption:
        cap = caption.strip().strip('*').strip()
        lines.append(f'\\caption{{{inline_convert(cap)}}}')
    lines.append(f'\\begin{{tabular}}{{{colspec}}}')
    lines.append('\\toprule')
    lines.append(' & '.join(cell(h) for h in header) + ' \\\\')
    lines.append('\\midrule')
    for r in data:
        r = (r + [''] * ncol)[:ncol]
        lines.append(' & '.join(cell(c) for c in r) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}' if caption else '\\end{center}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Structural pass (line by line).
# ---------------------------------------------------------------------------
lines = text.split('\n')
out = []
title = None
date = None
pending_caption = None
i = 0

in_list = None  # None | 'itemize' | 'enumerate'
while i < len(lines):
    raw = lines[i]
    s = raw.strip()

    if not s:
        i += 1
        continue

    # ---- list items (unordered `- ` or ordered `1. `) ----
    is_ul = s.startswith('- ')
    m_ol = re.match(r'^(\d+)\.\s+(.*)$', s)
    if is_ul or m_ol:
        kind = 'itemize' if is_ul else 'enumerate'
        content = s[2:].strip() if is_ul else m_ol.group(2).strip()
        if in_list != kind:
            if in_list is not None:
                out.append('\\end{' + in_list + '}')
            out.append('\\begin{' + kind + '}')
            in_list = kind
        out.append('  \\item ' + inline_convert(content))
        i += 1
        continue

    # ---- non-list line: close any open list first ----
    if in_list is not None:
        out.append('\\end{' + in_list + '}')
        in_list = None

    # table block
    if s.startswith('|'):
        block = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            block.append(lines[i].strip())
            i += 1
        out.append(render_table(block, pending_caption))
        pending_caption = None
        continue

    # title
    if s.startswith('# '):
        title = inline_convert(s[2:].strip())
        i += 1
        continue
    # version/date
    if s.startswith('**版本**') or s.startswith('**\\textbf{版本}'):
        m = re.search(r'版本\*?\*?[：:]\s*(.+)', s)
        if m:
            date = m.group(1).strip()
        i += 1
        continue

    # headings
    if s.startswith('#### '):
        out.append('\\subsubsection{' + inline_convert(s[5:].strip()) + '}')
        i += 1
        continue
    if s.startswith('### '):
        out.append('\\subsection{' + inline_convert(s[4:].strip()) + '}')
        i += 1
        continue
    if s.startswith('## '):
        out.append('\\section{' + inline_convert(s[3:].strip()) + '}')
        i += 1
        continue

    # horizontal rule
    if s == '---':
        i += 1
        continue

    # figure
    m = re.match(r'!\[(.*?)\]\((.*?)\)', s)
    if m:
        alt = m.group(1)
        path = m.group(2)
        out.append('\\begin{figure}[H]')
        out.append('\\centering')
        out.append(f'\\includegraphics[width=0.9\\textwidth]{{{path}}}')
        out.append(f'\\caption{{{inline_convert(alt)}}}')
        out.append('\\end{figure}')
        i += 1
        continue

    # bold table/figure caption line (store for the next table)
    if s.startswith('**表') and s.endswith('**'):
        pending_caption = s
        i += 1
        continue

    # ordinary paragraph / note
    out.append(inline_convert(s))
    i += 1

# close any trailing list
if in_list is not None:
    out.append('\\end{' + in_list + '}')

# ---------------------------------------------------------------------------
# Assemble document.
# ---------------------------------------------------------------------------
body = '\n\n'.join(out)

preamble = r"""\documentclass[11pt,a4paper]{ctexart}

\usepackage{amsmath,amssymb}
\usepackage{textcomp}   % \textdegree, \textasciitilde, \textasciicircum
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\usepackage[margin=2.5cm]{geometry}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}

% Tables / figures rendered with [H] (here) keep them adjacent to their text.

\title{""" + (title or '') + r"""}
\author{辛紫瑄}
\date{""" + (date or '') + r"""}

\begin{document}

\maketitle
"""

# The 摘要 is a normal `## 摘要` section in the .md, so it is emitted into the
# body as \section{摘要} rather than wrapped in a LaTeX abstract environment.

doc = preamble + body + r"""

\end{document}
"""

tex_file.write_text(doc, encoding='utf-8')
print(f'✓ LaTeX saved: {tex_file} ({len(doc.splitlines())} lines)')
