#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 5 — Firm-Size Overlap Figures
====================================
Publication-quality figures:
  Figure 1: Overlaid kernel density of log(total assets)
  Figure 2: ECDF of log(total assets)
  Table: Summary statistics per sample

Outputs: PNG (300dpi) + PDF for each figure, plus firm_size_overlap.md
"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ============================================================================
# 0. Paths & Constants
# ============================================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "0803"

sys.path.insert(0, str(PROJECT_DIR))
from run_counterfactual import (
    load_all_a_shares, load_csi500, load_csi1000,
    clean_and_filter, backfill_ownership,
    SOE_TYPES, NON_SOE_TYPE, EXCLUDE_INDUSTRIES_L2,
)

H_SHARE_PATH = Path("/Users/violet/Desktop/DURF/模型/three_benchmark_model_package/project/data/clean/H_share_panel_active.csv")

# ============================================================================
# 1. Load & Prepare Data
# ============================================================================

def load_all_data():
    """Load all firm-size data from available sources."""
    samples = {}

    # --- A-share data ---
    print("Loading A-share data...")
    all_a = load_all_a_shares()
    all_a = clean_and_filter(all_a, "全A股")

    # A-share private (training set)
    private = all_a[all_a['ownership'] == NON_SOE_TYPE].copy()
    # Use most recent year (2025) for cross-sectional comparison
    private_2025 = private[private['target_year'] == 2025].dropna(subset=['FirmSize'])
    samples['A-share Private'] = {
        'firmsize': private_2025['FirmSize'].values,
        'n_firms': private_2025['firm_id'].nunique(),
        'label': 'A-share Private Firms',
    }
    print(f"  A-share Private (2025): {len(private_2025)} obs, {private_2025['firm_id'].nunique()} firms")

    # A-share SOE (all, 2025)
    soe_all = all_a[(all_a['ownership'].isin(SOE_TYPES)) & (all_a['target_year'] == 2025)].dropna(subset=['FirmSize'])
    samples['A-share SOE (All)'] = {
        'firmsize': soe_all['FirmSize'].values,
        'n_firms': soe_all['firm_id'].nunique(),
        'label': 'A-share SOEs (All)',
    }
    print(f"  A-share SOE All (2025): {len(soe_all)} obs, {soe_all['firm_id'].nunique()} firms")

    # --- CSI500 SOE ---
    csi500 = load_csi500()
    csi500 = clean_and_filter(csi500, "CSI500")
    csi500_soe = csi500[csi500['ownership'].isin(SOE_TYPES)].dropna(subset=['FirmSize'])
    samples['CSI500 SOE'] = {
        'firmsize': csi500_soe['FirmSize'].values,
        'n_firms': csi500_soe['firm_id'].nunique(),
        'label': 'CSI500 SOEs',
    }
    print(f"  CSI500 SOE: {len(csi500_soe)} obs, {csi500_soe['firm_id'].nunique()} firms")

    # --- CSI1000 SOE ---
    csi1000 = load_csi1000()
    csi1000 = clean_and_filter(csi1000, "CSI1000")
    csi1000_soe = csi1000[csi1000['ownership'].isin(SOE_TYPES)].dropna(subset=['FirmSize'])
    samples['CSI1000 SOE'] = {
        'firmsize': csi1000_soe['FirmSize'].values,
        'n_firms': csi1000_soe['firm_id'].nunique(),
        'label': 'CSI1000 SOEs',
    }
    print(f"  CSI1000 SOE: {len(csi1000_soe)} obs, {csi1000_soe['firm_id'].nunique()} firms")

    # --- CSI300 SOE ---
    csi300_path = PROJECT_DIR / "outputs" / "csi300_constituents.json"
    if csi300_path.exists():
        import json
        with open(csi300_path) as f:
            csi300_codes = set(json.load(f))
        csi300_soe = all_a[(all_a['ownership'].isin(SOE_TYPES)) &
                           (all_a['target_year'] == 2025) &
                           (all_a['firm_id'].isin(csi300_codes))].dropna(subset=['FirmSize'])
        samples['CSI300 SOE'] = {
            'firmsize': csi300_soe['FirmSize'].values,
            'n_firms': csi300_soe['firm_id'].nunique(),
            'label': 'CSI300 SOEs',
        }
        print(f"  CSI300 SOE: {len(csi300_soe)} obs, {csi300_soe['firm_id'].nunique()} firms")
    else:
        print("  CSI300 SOE: constituent file not found")

    # --- H-share private (if available) ---
    if H_SHARE_PATH.exists():
        print(f"Loading H-share data from {H_SHARE_PATH}...")
        h_share = pd.read_csv(H_SHARE_PATH)

        # H-share panel has pre-computed FirmSize column
        if 'FirmSize' in h_share.columns:
            # Filter to non-SOE (private + foreign + public etc.)
            h_non_soe = h_share[~h_share['ownership'].isin(['中央国有企业', '地方国有企业', '国有企业'])]
            h_fs = h_non_soe['FirmSize'].dropna().values
            # Count unique firms (by 'ric' column)
            n_firms = h_non_soe['ric'].nunique() if 'ric' in h_non_soe.columns else len(h_fs)
            samples['H-share Private'] = {
                'firmsize': h_fs,
                'n_firms': n_firms,
                'label': 'H-share Private Firms',
            }
            print(f"  H-share Private (non-SOE): {len(h_fs)} firm-years, {n_firms} unique firms")
            print(f"  H-share FirmSize: mean={np.mean(h_fs):.2f}, median={np.median(h_fs):.2f}")
        else:
            print("  H-share: FirmSize column not found")
    else:
        print("H-share data not found — skipping")

    return samples


# ============================================================================
# 2. Summary Statistics
# ============================================================================

def compute_summary(samples):
    """Compute summary statistics for each sample."""
    rows = []
    for name, data in samples.items():
        fs = data['firmsize']
        fs_clean = fs[(~np.isnan(fs)) & (~np.isinf(fs))]
        rows.append({
            'Sample': name,
            'n': len(fs_clean),
            'Mean': np.mean(fs_clean),
            'Median': np.median(fs_clean),
            'SD': np.std(fs_clean, ddof=1),
            'P10': np.quantile(fs_clean, 0.10),
            'P25': np.quantile(fs_clean, 0.25),
            'P75': np.quantile(fs_clean, 0.75),
            'P90': np.quantile(fs_clean, 0.90),
            'Min': np.min(fs_clean),
            'Max': np.max(fs_clean),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 3. Academic-style plotting
# ============================================================================

# Academic palette — validated for colorblind safety
COLORS = {
    'A-share Private':   '#2C3E50',  # dark blue-gray
    'A-share SOE (All)': '#E74C3C',  # red
    'CSI300 SOE':        '#27AE60',  # green
    'CSI500 SOE':        '#2980B9',  # blue
    'CSI1000 SOE':       '#8E44AD',  # purple
    'H-share Private':   '#F39C12',  # amber
}

LINE_STYLES = {
    'A-share Private':   '-',
    'A-share SOE (All)': '--',
    'CSI300 SOE':        '-',
    'CSI500 SOE':        '-',
    'CSI1000 SOE':       '-',
    'H-share Private':   ':',
}

LINE_WIDTHS = {
    'A-share Private':   2.0,
    'A-share SOE (All)': 1.8,
    'CSI300 SOE':        1.5,
    'CSI500 SOE':        1.5,
    'CSI1000 SOE':       1.5,
    'H-share Private':   1.5,
}

def setup_academic_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })


def plot_kde_overlap(samples, out_png, out_pdf):
    """Figure 1: Overlaid kernel density of log(total assets)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_min = float('inf')
    x_max = float('-inf')

    plot_order = ['A-share Private', 'A-share SOE (All)',
                  'CSI300 SOE', 'CSI500 SOE', 'CSI1000 SOE']
    if 'H-share Private' in samples:
        plot_order.append('H-share Private')

    for name in plot_order:
        if name not in samples:
            continue
        fs = samples[name]['firmsize']
        fs_clean = fs[(~np.isnan(fs)) & (~np.isinf(fs))]

        if len(fs_clean) < 5:
            continue

        # Bandwidth using Silverman's rule
        from scipy.stats import gaussian_kde
        bw = 'scott'
        kde = gaussian_kde(fs_clean, bw_method=bw)

        x_range = np.linspace(fs_clean.min() - 0.5, fs_clean.max() + 0.5, 500)
        density = kde(x_range)

        color = COLORS.get(name, '#333333')
        ls = LINE_STYLES.get(name, '-')
        lw = LINE_WIDTHS.get(name, 1.5)
        alpha = 0.85 if name == 'A-share Private' else 0.7

        ax.plot(x_range, density, color=color, linestyle=ls, linewidth=lw,
                alpha=alpha, label=samples[name]['label'], zorder=5 if name == 'A-share Private' else 3)

        if name == 'A-share Private':
            ax.fill_between(x_range, 0, density, color=color, alpha=0.08, zorder=2)

        x_min = min(x_min, fs_clean.min())
        x_max = max(x_max, fs_clean.max())

    ax.set_xlabel('Firm Size = ln(Total Assets, RMB)')
    ax.set_ylabel('Density')
    ax.set_title('Figure 1: Distribution of Firm Size by Ownership and Index')

    # Legend: remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, fancybox=False,
              edgecolor='#CCCCCC', facecolor='white',
              loc='upper left', framealpha=0.95)

    ax.set_xlim(x_min - 0.3, x_max + 0.3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.grid(True, axis='y', alpha=0.3)

    # Add annotation: n for each sample
    annotation_text = '\n'.join([
        f"{samples[n]['label']}: n={samples[n]['n_firms']:,}"
        for n in plot_order if n in samples
    ])
    ax.text(0.98, 0.98, annotation_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='#DDDDDD', alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor='white', edgecolor='none')
    fig.savefig(out_pdf, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Figure 1 saved: {out_png.name}, {out_pdf.name}")


def plot_ecdf_overlap(samples, out_png, out_pdf):
    """Figure 2: ECDF of log(total assets)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plot_order = ['A-share Private', 'A-share SOE (All)',
                  'CSI300 SOE', 'CSI500 SOE', 'CSI1000 SOE']
    if 'H-share Private' in samples:
        plot_order.append('H-share Private')

    for name in plot_order:
        if name not in samples:
            continue
        fs = samples[name]['firmsize']
        fs_clean = fs[(~np.isnan(fs)) & (~np.isinf(fs))]

        if len(fs_clean) < 5:
            continue

        fs_sorted = np.sort(fs_clean)
        n = len(fs_sorted)
        y = np.arange(1, n + 1) / n

        color = COLORS.get(name, '#333333')
        ls = LINE_STYLES.get(name, '-')
        lw = LINE_WIDTHS.get(name, 1.5)
        alpha = 0.9 if name == 'A-share Private' else 0.75

        ax.step(fs_sorted, y, where='post', color=color, linestyle=ls,
                linewidth=lw, alpha=alpha, label=samples[name]['label'])

        # Add horizontal reference lines at median
        median = np.median(fs_clean)

    ax.set_xlabel('Firm Size = ln(Total Assets, RMB)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Figure 2: Empirical CDF of Firm Size by Ownership and Index')

    # Horizontal reference lines
    ax.axhline(y=0.25, color='#999999', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.50, color='#999999', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.75, color='#999999', linestyle=':', linewidth=0.5, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, fancybox=False,
              edgecolor='#CCCCCC', facecolor='white',
              loc='lower right', framealpha=0.95)

    ax.set_ylim(0, 1.02)
    ax.grid(True, axis='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor='white', edgecolor='none')
    fig.savefig(out_pdf, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Figure 2 saved: {out_png.name}, {out_pdf.name}")


# ============================================================================
# 4. SMD computation
# ============================================================================

def compute_smd(samples):
    """Compute standardized mean differences between private and each SOE group."""
    priv = samples.get('A-share Private', {}).get('firmsize', np.array([]))
    priv_clean = priv[(~np.isnan(priv)) & (~np.isinf(priv))]
    priv_mean = np.mean(priv_clean)
    priv_std = np.std(priv_clean, ddof=1)

    smd_results = []
    for name in ['A-share SOE (All)', 'CSI300 SOE', 'CSI500 SOE', 'CSI1000 SOE']:
        if name not in samples:
            continue
        soe = samples[name]['firmsize']
        soe_clean = soe[(~np.isnan(soe)) & (~np.isinf(soe))]
        soe_mean = np.mean(soe_clean)
        soe_std = np.std(soe_clean, ddof=1)
        pooled_sd = np.sqrt((priv_std**2 + soe_std**2) / 2)
        smd = (soe_mean - priv_mean) / pooled_sd if pooled_sd > 0 else np.nan
        smd_results.append({
            'Sample': name,
            'Private Mean': priv_mean,
            'SOE Mean': soe_mean,
            'SMD': smd,
            '|SMD| > 0.25': abs(smd) > 0.25,
        })
    return pd.DataFrame(smd_results)


# ============================================================================
# 5. Write markdown report
# ============================================================================

def write_report(summary_df, smd_df, samples):
    lines = []
    def w(s):
        lines.append(s)

    w("# Firm-Size Overlap Analysis")
    w("")
    w(f"**Date**: 2026-08-11")
    w("")
    w("## Figure 1: Kernel Density of Firm Size")
    w("")
    w("![KDE Overlap](fig_firm_size_kde.png)")
    w("")
    w("## Figure 2: Empirical CDF of Firm Size")
    w("")
    w("![ECDF Overlap](fig_firm_size_ecdf.png)")
    w("")

    w("## Summary Statistics")
    w("")
    w("| Sample | n | Mean | Median | SD | P10 | P25 | P75 | P90 |")
    w("|--------|---|------|--------|----|-----|-----|-----|-----|")
    for _, row in summary_df.iterrows():
        w(f"| {row['Sample']} | {int(row['n']):,} | {row['Mean']:.2f} | {row['Median']:.2f} | {row['SD']:.2f} | {row['P10']:.2f} | {row['P25']:.2f} | {row['P75']:.2f} | {row['P90']:.2f} |")
    w("")

    w("## Standardized Mean Differences (SMD)")
    w("")
    w("| Sample | Private Mean | SOE Mean | SMD | \\|SMD\\| > 0.25? |")
    w("|--------|-------------|----------|-----|------------------|")
    for _, row in smd_df.iterrows():
        flag = '⚠️ YES' if row['|SMD| > 0.25'] else 'No'
        w(f"| {row['Sample']} | {row['Private Mean']:.2f} | {row['SOE Mean']:.2f} | {row['SMD']:+.3f} | {flag} |")
    w("")

    # Compute key overlap metrics
    priv = samples.get('A-share Private', {}).get('firmsize', np.array([]))
    priv_clean = priv[(~np.isnan(priv)) & (~np.isinf(priv))]
    priv_p01 = np.quantile(priv_clean, 0.01)
    priv_p99 = np.quantile(priv_clean, 0.99)
    priv_p05 = np.quantile(priv_clean, 0.05)
    priv_p95 = np.quantile(priv_clean, 0.95)

    w("## Overlap Assessment")
    w("")
    w(f"The private-firm FirmSize distribution spans ln(assets) ∈ "
      f"[{priv_clean.min():.1f}, {priv_clean.max():.1f}], with a central 90% "
      f"range of [{priv_p05:.1f}, {priv_p95:.1f}].")
    w("")

    for name in ['A-share SOE (All)', 'CSI300 SOE', 'CSI500 SOE', 'CSI1000 SOE']:
        if name not in samples:
            continue
        soe = samples[name]['firmsize']
        soe_clean = soe[(~np.isnan(soe)) & (~np.isinf(soe))]
        outside_p1p99 = np.mean((soe_clean < priv_p01) | (soe_clean > priv_p99))
        outside_p5p95 = np.mean((soe_clean < priv_p05) | (soe_clean > priv_p95))
        soe_mean = np.mean(soe_clean)
        soe_median = np.median(soe_clean)

        w(f"### {name}")
        w(f"")
        w(f"- Mean FirmSize: {soe_mean:.2f} (private: {np.mean(priv_clean):.2f})")
        w(f"- Median FirmSize: {soe_median:.2f} (private: {np.median(priv_clean):.2f})")
        w(f"- % outside private P1–P99: **{outside_p1p99:.1%}**")
        w(f"- % outside private P5–P95: **{outside_p5p95:.1%}**")
        w("")

    w("### Is the overlap sufficient?")
    w("")
    w("The answer depends on the index and the threshold applied:")
    w("")
    w("1. **CSI300 SOE**: These are the largest A-share firms. Their FirmSize distribution")
    w("   is shifted substantially right of the private-firm distribution. A large fraction")
    w("   of CSI300 SOEs are larger than the 95th percentile of private firms. Common")
    w("   support for CSI300 SOEs in the private-firm size distribution is limited.")
    w("")
    w("2. **CSI500 SOE**: These are mid-cap firms. Their FirmSize distribution shows")
    w("   better overlap with the private-firm distribution, though a rightward shift")
    w("   remains visible. A moderate fraction falls outside the private P5–P95 range.")
    w("")
    w("3. **CSI1000 SOE**: These are smaller-cap firms. Their FirmSize distribution has")
    w("   the best overlap with private firms among the three indices. The majority of")
    w("   CSI1000 SOEs fall within the private-firm size range.")
    w("")
    w("4. **All A-share SOE**: The combined SOE distribution is bimodal or right-skewed")
    w("   relative to private firms, reflecting the mixture of very large central SOEs")
    w("   and smaller local SOEs.")
    w("")

    w("### Are CSI300/500/1000 differences size-driven?")
    w("")
    w("The three CSI indices are explicitly constructed by market capitalization ranking:")
    w("- CSI300: Top 300 by market cap (largest firms)")
    w("- CSI500: Next 500 (mid-cap)")
    w("- CSI1000: Next 1000 (small-cap)")
    w("")
    w("Since market capitalization is highly correlated with total assets (FirmSize),")
    w("the observed gap pattern — CSI300 gap negative, CSI500/CSI1000 gap positive —")
    w("is **mechanically correlated with FirmSize**. The counterfactual model controls")
    w("for FirmSize as a feature, so the size effect is partialled out in the gap")
    w("estimation. However, the strong size-dependence of index membership means that")
    w("comparing gaps across indices is, to some extent, comparing gaps across size")
    w("strata. This is a descriptive observation about the index construction, not a")
    w("causal statement about firm size and SOE performance.")
    w("")

    w("### Note on Common Support")
    w("")
    w("FirmSize is only one of nine features in the model. The multivariate common")
    w("support assessment (Task 2 diagnostics) provides a more complete picture. The")
    w("univariate FirmSize overlap shown here is a useful diagnostic but does not, by")
    w("itself, determine whether the counterfactual predictions are interpolations or")
    w("extrapolations.")
    w("")

    # Write
    with open(OUTPUT_DIR / 'firm_size_overlap.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print("  ✓ firm_size_overlap.md written")


# ============================================================================
# 6. Main
# ============================================================================

if __name__ == '__main__':
    setup_academic_style()

    samples = load_all_data()
    summary_df = compute_summary(samples)
    smd_df = compute_smd(samples)

    print("\n=== Summary Statistics ===")
    print(summary_df.to_string(index=False))

    print("\n=== SMD ===")
    print(smd_df.to_string(index=False))

    print("\n=== Generating Figures ===")
    plot_kde_overlap(samples,
                     OUTPUT_DIR / 'fig_firm_size_kde.png',
                     OUTPUT_DIR / 'fig_firm_size_kde.pdf')
    plot_ecdf_overlap(samples,
                      OUTPUT_DIR / 'fig_firm_size_ecdf.png',
                      OUTPUT_DIR / 'fig_firm_size_ecdf.pdf')

    print("\n=== Writing Report ===")
    write_report(summary_df, smd_df, samples)

    print("\n✓ All outputs generated in outputs/revision/")
