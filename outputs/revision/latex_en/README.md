# The Counterfactual Return-on-Assets Gap of State-Owned Enterprises in China (English version)

English version of the Chinese paper `latex/main.tex`, with **identical numbers** and a full
English rewrite (not a literal translation). Compiled output: `main_en.pdf` (22 pages).

## Directory structure

```
latex_en/
├── main_en.tex                 # English paper source (article class, xelatex)
├── main_en.pdf                 # compiled PDF (22 pages)
├── figures/                    # 9 figures (PDF, English labels, shared with Chinese version)
├── Makefile                    # build script
└── README.md                   # this file
```

## Compilation

```bash
xelatex -interaction=nonstopmode main_en.tex   # run twice
# or: make
```

Required packages (installed via `tlmgr`): `booktabs threeparttable multirow tabularx makecell
caption float xcolor enumitem setspace hyperref graphicx amsmath`.

Note: `makecell` is used for multi-line table headers and may need
`tlmgr install makecell` on a fresh TinyTeX install.

## What was changed relative to the Chinese version

1. **Full English rewrite.** The prose follows English academic writing conventions (topic
   sentences, hedged claims, shorter clauses) rather than a sentence-by-sentence translation.
   All numbers, signs, units, and statistical results are carried over exactly.
2. **Currency units.** Amounts are reported as RMB 100 million in the tables (as in the Chinese
   original) and as RMB billion / trillion in the prose headlines (RMB 597.9 billion in 2025;
   RMB 4.1 trillion cumulative over 2019–2025).
3. **Margin overflow fixed.** Three tables in the Chinese version overflowed the text width
   (data sources +135pt, sample construction +108pt, model specification +48pt). All tables now
   use `tabularx` with `X`-columns or explicit `p{}` columns so no table exceeds the margin.
   The English version compiles with **0 overfull boxes** (verified via the log and a
   programmatic content-vs-margin check on every page).

## Before submission (same caveats as the Chinese version)

1. **Author info.** `\author{Zixuan Xin}` and `\thanks{...}` are placeholders; replace with the
   real name, affiliation, and funding acknowledgements.
2. **References.** The 30 entries were compiled from the author-provided PDFs in `source/`. For a
   few journals (e.g., *Corporate Governance*, *Public Choice*, *Management and Organization
   Review*), the issue number is not shown on the title page; verify volume/issue/pages/DOI
   before submission.
3. **Interpretation.** The gap is a return differential relative to a private-enterprise
   benchmark, **not** directly an amount of government subsidy or policy burden (see the caveat
   in Section 4.1 and the pointer to Lucas, Garcia, and Solberg, 2026).
