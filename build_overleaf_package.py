#!/usr/bin/env python3
"""Build a self-contained Overleaf upload package for the LaTeX paper.

Collects 论文_中国国有企业反事实资产回报率研究.tex plus every image it
references via \\includegraphics, and zips them preserving the relative
directory layout (final_results/…, outputs/revision/…).

Uses Python's zipfile (NOT the macOS `zip` CLI) because zipfile sets the
UTF-8 filename flag (bit 11) on the archive, so the Chinese .tex filename
survives a round-trip through Overleaf's unzip intact.
"""
import re
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TEX = PROJECT / '论文_中国国有企业反事实资产回报率研究.tex'
ZIP = PROJECT / '论文_latex_overleaf.zip'

tex = TEX.read_text(encoding='utf-8')
images = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)

missing = [p for p in images if not (PROJECT / p).exists()]
if missing:
    raise SystemExit('Missing images:\n  ' + '\n  '.join(missing))

files = [TEX.name] + images
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(TEX, TEX.name)  # Chinese name → UTF-8 flag set automatically
    for img in images:
        z.write(PROJECT / img, img)  # preserve final_results/… , outputs/revision/…

# Verify the UTF-8 filename flag was set on the Chinese .tex entry.
with zipfile.ZipFile(ZIP) as z:
    info = z.getinfo(TEX.name)
    assert info.flag_bits & 0x800, 'UTF-8 filename flag not set!'

print(f'✓ {ZIP.name}: {len(files)} files')
print(f'  main: {TEX.name}')
print(f'  images ({len(images)}):')
for img in images:
    print(f'    - {img}')
print(f'  size: {ZIP.stat().st_size / 1e6:.2f} MB')
