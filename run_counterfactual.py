#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全A股 → 中证500/300 反事实预测模型

实验设计：
  训练集：全A股非SOE（民营企业）2014-2024 panel
  预测集：中证500 SOE 2025 + 中证300 SOE（从全A股中识别）

  模型：Ridge(α=10) + RandomForest + GBM（复现0728方法）
  目标：EBI/A = (归母净利润_t+1 + 利息支出_t+1) / 资产总计_t+1

  特征（t期）：
    FirmSize, AssetTurnover, FinancialLeverage, FixedAssetsRatio,
    CurrentRatio, CapexRatio, WorkingCapitalRatio, MarketShare, HHI
"""

import zipfile, re, os, sys, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# 0. Configuration
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent / "0803"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# A-share files
A_FILES = {year: DATA_DIR / f"A_{year}_origin.xlsx" for year in range(2014, 2025)}
CSI500_FILE = DATA_DIR / "csi500_2024.xlsx"
CSI1000_FILE = DATA_DIR / "中证1000_2024_财务数据提取.xlsx"

# Exclude financial industries (same as project standard)
EXCLUDE_INDUSTRIES_L2 = {'银行', '保险', '证券', '多元金融', '非银金融'}

# SOE types
SOE_TYPES = {'中央国有企业', '地方国有企业'}
# Non-SOE benchmark: only 民营企业
NON_SOE_TYPE = '民营企业'

# Winsorization
WINSOR_BOUNDS = (0.05, 0.95)

# Model params
RANDOM_STATE = 42
RIDGE_ALPHA = 10.0

# ============================================================================
# 1. XML Parser for corrupted xlsx files
# ============================================================================

def parse_xlsx_xml(filepath):
    """
    Parse an xlsx file via XML (bypasses openpyxl style corruption).

    Returns (headers, data_rows, shared_strings) where:
      headers: list of (col_letter, header_text) tuples
      data_rows: list of dicts {col_letter: (value, is_string)}
      shared_strings: list of str
    """
    with zipfile.ZipFile(filepath, 'r') as z:
        # Parse shared strings
        with z.open('xl/sharedStrings.xml') as f:
            import xml.etree.ElementTree as ET
            tree = ET.parse(f)
        ns = {'ns': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        strings = []
        for si in tree.findall('.//ns:si', ns):
            t_elem = si.find('ns:t', ns)
            if t_elem is not None and t_elem.text:
                strings.append(t_elem.text)
            else:
                strings.append('')

        # Parse sheet
        with z.open('xl/worksheets/sheet1.xml') as f:
            sheet_tree = ET.fromstring(f.read().decode('utf-8'))

        rows = sheet_tree.findall('.//ns:row', ns)

    # Parse header row
    header_row = rows[0]
    headers = {}  # col_letter -> full_header_text
    for cell in header_row.findall('ns:c', ns):
        ref = cell.get('r')
        col_letter = re.match(r'^[A-Z]+', ref).group(0)
        cell_type = cell.get('t', '')
        v_elem = cell.find('ns:v', ns)
        value = v_elem.text if v_elem is not None else ''
        if cell_type == 's' and value:
            idx = int(value)
            header_text = strings[idx] if idx < len(strings) else ''
        else:
            header_text = value or ''
        headers[col_letter] = header_text

    # Parse data rows
    data_rows = []
    for row in rows[1:]:
        row_data = {}
        for cell in row.findall('ns:c', ns):
            ref = cell.get('r')
            col_letter = re.match(r'^[A-Z]+', ref).group(0)
            cell_type = cell.get('t', '')
            v_elem = cell.find('ns:v', ns)
            value = v_elem.text if v_elem is not None else ''
            is_string = (cell_type == 's')
            if is_string and value:
                idx = int(value)
                row_data[col_letter] = (strings[idx] if idx < len(strings) else '', True)
            else:
                row_data[col_letter] = (value or '', False)
        data_rows.append(row_data)

    return headers, data_rows, strings


def extract_header_info(header_text):
    """
    Extract field name and report year from a Wind header.

    Examples:
      '资产总计\\n[报告期] 2015年报\\n...' → ('资产总计', 2015)
      '营业收入(同比增长率)\\n[报告期] 2014年报\\n...' → ('营业收入(同比增长率)', 2014)
      '所属Wind行业名称(2024)\\n...' → ('所属Wind行业名称', 2024)
      '企业所有制性质\\n...' → ('企业所有制性质', None)
    """
    # Split by newline or bracket
    first_line = header_text.split('\n')[0].strip()

    # Extract year from the full header
    year = None
    # Try [报告期] YYYY年报
    m = re.search(r'\[报告期\]\s*(\d{4})', header_text)
    if m:
        year = int(m.group(1))
    else:
        # Try parenthesized year
        m = re.search(r'\((\d{4})\)', first_line)
        if m:
            year = int(m.group(1))

    # Clean field name
    field_name = first_line

    return field_name, year


def classify_column(header_text):
    """
    Classify a column into a semantic type.

    Returns (field_type, report_year)
    field_type is one of:
      'stock_code', 'stock_name', 'ownership', 'industry_cn', 'industry_en',
      'net_profit', 'interest_expense', 'total_assets', 'revenue',
      'revenue_growth', 'current_assets', 'current_liabilities',
      'capex', 'total_liabilities', 'net_fixed_assets',
      'listing_status', 'list_date', 'delist_date', 'unknown'
    """
    field_name, year = extract_header_info(header_text)

    if '证券代码' in field_name:
        return 'stock_code', year
    elif '证券简称' in field_name:
        return 'stock_name', year
    elif '证券存续状态' in field_name:
        return 'listing_status', year
    elif '上市日期' in field_name:
        return 'list_date', year
    elif '摘牌日期' in field_name:
        return 'delist_date', year
    elif '企业所有制' in field_name:
        return 'ownership', year
    elif '所属Wind行业' in field_name:
        if '英文' in header_text:
            return 'industry_en', year
        else:
            return 'industry_cn', year
    elif '归属母公司股东的净利润' in field_name or '归母净利润' in field_name:
        return 'net_profit', year
    elif '利息支出' in field_name:
        return 'interest_expense', year
    elif '资产总计' in field_name:
        return 'total_assets', year
    elif '营业收入(同比增长率)' in field_name or '营业收入同比增长率' in field_name:
        return 'revenue_growth', year
    elif '营业收入' in field_name:
        return 'revenue', year
    elif '流动资产合计' in field_name or '流动资产' in field_name:
        return 'current_assets', year
    elif '流动负债合计' in field_name or '流动负债' in field_name:
        return 'current_liabilities', year
    elif '资本性支出' in field_name:
        return 'capex', year
    elif '负债合计' in field_name:
        return 'total_liabilities', year
    elif '固定资产' in field_name:
        return 'net_fixed_assets', year
    else:
        return 'unknown', year


def parse_file_to_records(filepath):
    """
    Parse an xlsx file and return a list of per-firm records with semantic fields.

    Each record: {
      'stock_code': str,
      'stock_name': str,
      'ownership': str,
      'industry_cn': str,  # e.g. "金融--银行--商业银行--多元化银行"
      'industry_en': str,
      'data': {field_type: {year: value}}
    }
    """
    headers, data_rows, strings = parse_xlsx_xml(filepath)

    # Classify each column
    col_map = {}  # col_letter -> (field_type, report_year)
    for col_letter, header_text in headers.items():
        ftype, year = classify_column(header_text)
        col_map[col_letter] = (ftype, year)

    # Parse data rows
    records = []
    for row_data in data_rows:
        record = {
            'stock_code': '',
            'stock_name': '',
            'ownership': '',
            'industry_cn': '',
            'industry_en': '',
            'listing_status': '',
            'list_date': None,
            'delist_date': None,
            'data': {}  # field_type -> {year: value}
        }

        for col_letter, (value, is_string) in row_data.items():
            if col_letter not in col_map:
                continue
            ftype, year = col_map[col_letter]

            if ftype == 'stock_code':
                record['stock_code'] = value
            elif ftype == 'stock_name':
                record['stock_name'] = value
            elif ftype == 'ownership':
                record['ownership'] = value
            elif ftype == 'industry_cn':
                record['industry_cn'] = value
            elif ftype == 'industry_en':
                record['industry_en'] = value
            elif ftype == 'listing_status':
                record['listing_status'] = value
            elif ftype in ('net_profit', 'interest_expense', 'total_assets', 'revenue',
                          'revenue_growth', 'current_assets', 'current_liabilities',
                          'capex', 'total_liabilities', 'net_fixed_assets'):
                if ftype not in record['data']:
                    record['data'][ftype] = {}
                try:
                    record['data'][ftype][year] = float(value) if value else np.nan
                except ValueError:
                    record['data'][ftype][year] = np.nan

        # Skip empty rows
        if not record['stock_code']:
            continue

        records.append(record)

    return records


def records_to_dataframe(records, expected_feature_year, expected_target_year):
    """
    Convert parsed records to a clean DataFrame with computed features.

    For each firm row:
      feature_year = t (e.g., 2014)
      target_year = t+1 (e.g., 2015)

    Returns DataFrame with columns:
      firm_id, firm_name, market, industry_l1, industry_l2, feature_year, target_year,
      ownership, 资产总计, EBI_A, FirmSize, AssetTurnover, FinancialLeverage,
      FixedAssetsRatio, CurrentRatio, CapexRatio, WorkingCapitalRatio,
      MarketShare, HHI
    """
    rows = []

    for rec in records:
        data = rec['data']

        # Get feature year and target year data
        ta_f = data.get('total_assets', {}).get(expected_feature_year, np.nan)
        ta_t = data.get('total_assets', {}).get(expected_target_year, np.nan)
        rev_f = data.get('revenue', {}).get(expected_feature_year, np.nan)
        np_t = data.get('net_profit', {}).get(expected_target_year, np.nan)
        ie_t = data.get('interest_expense', {}).get(expected_target_year, np.nan)
        ca_f = data.get('current_assets', {}).get(expected_feature_year, np.nan)
        cl_f = data.get('current_liabilities', {}).get(expected_feature_year, np.nan)
        capex_f = data.get('capex', {}).get(expected_feature_year, np.nan)
        tl_f = data.get('total_liabilities', {}).get(expected_feature_year, np.nan)
        nfa_f = data.get('net_fixed_assets', {}).get(expected_feature_year, np.nan)

        # Industry parsing
        industry_cn = rec.get('industry_cn', '')
        parts = industry_cn.split('--')
        ind_l1 = parts[0] if len(parts) > 0 else ''
        ind_l2 = parts[1] if len(parts) > 1 else ''

        # Skip if no target year assets (can't compute EBI/A)
        if pd.isna(ta_t) or ta_t == 0:
            continue

        # Compute EBI/A — require both net profit and interest expense
        if pd.notna(np_t) and pd.notna(ie_t):
            ebi_a = (np_t + ie_t) / ta_t
        else:
            ebi_a = np.nan

        # Compute feature-year ratios
        # FirmSize
        if pd.notna(ta_f) and ta_f > 0:
            firm_size = np.log(ta_f)
        else:
            firm_size = np.nan

        # AssetTurnover
        if pd.notna(rev_f) and pd.notna(ta_f) and ta_f > 0:
            asset_turnover = rev_f / ta_f
        else:
            asset_turnover = np.nan

        # FinancialLeverage
        if pd.notna(tl_f) and pd.notna(ta_f) and ta_f > 0:
            fin_lev = tl_f / ta_f
        else:
            fin_lev = np.nan

        # FixedAssetsRatio
        if pd.notna(nfa_f) and pd.notna(ta_f) and ta_f > 0:
            fixed_assets_ratio = nfa_f / ta_f
        else:
            fixed_assets_ratio = np.nan

        # CurrentRatio
        if pd.notna(ca_f) and pd.notna(cl_f) and cl_f > 0:
            current_ratio = ca_f / cl_f
        else:
            current_ratio = np.nan

        # CapexRatio
        if pd.notna(capex_f) and pd.notna(ta_f) and ta_f > 0:
            capex_ratio = capex_f / ta_f
        else:
            capex_ratio = np.nan

        # WorkingCapitalRatio
        if pd.notna(ca_f) and pd.notna(cl_f) and pd.notna(ta_f) and ta_f > 0:
            wc_ratio = (ca_f - cl_f) / ta_f
        else:
            wc_ratio = np.nan

        rows.append({
            'firm_id': rec['stock_code'],
            'firm_name': rec['stock_name'],
            'market': 'CSI',
            'ownership': rec['ownership'],
            'industry_cn': industry_cn,
            'industry_l1': ind_l1,
            'industry_l2': ind_l2,
            'feature_year': expected_feature_year,
            'target_year': expected_target_year,
            '资产总计': ta_f,
            '资产总计_target': ta_t,
            '营业收入': rev_f,
            'EBI_A': ebi_a,
            'FirmSize': firm_size,
            'AssetTurnover': asset_turnover,
            'FinancialLeverage': fin_lev,
            'FixedAssetsRatio': fixed_assets_ratio,
            'CurrentRatio': current_ratio,
            'CapexRatio': capex_ratio,
            'WorkingCapitalRatio': wc_ratio,
        })

    df = pd.DataFrame(rows)

    # Compute MarketShare and HHI within each (feature_year, industry_l2)
    df['MarketShare'] = np.nan
    df['HHI'] = np.nan

    for (fy, ind2), group in df.groupby(['feature_year', 'industry_l2']):
        total_rev = group['营业收入'].sum()
        if total_rev > 0:
            shares = group['营业收入'] / total_rev
            df.loc[group.index, 'MarketShare'] = shares
            df.loc[group.index, 'HHI'] = (shares ** 2).sum()

    return df


# ============================================================================
# 2. Load all data
# ============================================================================

def load_all_a_shares():
    """Load and parse all A-share files, return combined DataFrame."""
    all_dfs = []

    for year in range(2014, 2025):
        filepath = A_FILES[year]
        if not filepath.exists():
            print(f"  ⚠ 文件不存在: {filepath}")
            continue

        print(f"  解析 {filepath.name} (feature_year={year}, target_year={year+1})...")
        records = parse_file_to_records(filepath)
        df = records_to_dataframe(records, expected_feature_year=year,
                                  expected_target_year=year+1)
        df['_source_file'] = filepath.name
        all_dfs.append(df)
        print(f"    → {len(df)} valid rows, {df['firm_id'].nunique()} firms")

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def load_csi500():
    """Load and parse the CSI500 file."""
    print(f"  解析 csi500_2024.xlsx (feature_year=2024, target_year=2025)...")
    records = parse_file_to_records(CSI500_FILE)
    df = records_to_dataframe(records, expected_feature_year=2024,
                              expected_target_year=2025)
    return df


def load_csi1000():
    """Load and parse the CSI1000 file."""
    print(f"  解析 中证1000_2024_财务数据提取.xlsx (feature_year=2024, target_year=2025)...")
    records = parse_file_to_records(CSI1000_FILE)
    df = records_to_dataframe(records, expected_feature_year=2024,
                              expected_target_year=2025)
    return df


# ============================================================================
# 3. Data cleaning and filtering
# ============================================================================

def backfill_ownership(df):
    """
    Backfill missing ownership (2014 file has empty ownership column).
    Use the most common ownership from later years for each firm_id.
    """
    # Build ownership lookup from firms with known ownership
    known = df[df['ownership'].notna() & (df['ownership'] != '')]
    # For each firm, get the most recent/latest ownership (most reliable)
    firm_own = known.sort_values('target_year').groupby('firm_id')['ownership'].last()

    n_filled = 0
    for idx in df[df['ownership'].isna() | (df['ownership'] == '')].index:
        fid = df.loc[idx, 'firm_id']
        if fid in firm_own.index:
            df.loc[idx, 'ownership'] = firm_own[fid]
            n_filled += 1

    if n_filled > 0:
        print(f"    回填所有制: {n_filled} 行 (来自后续年份)")

    return df


def clean_and_filter(df, label=""):
    """
    Apply standard cleaning:
    1. Backfill missing ownership
    2. Remove financial industries
    3. Drop rows with missing EBI/A
    4. Drop extreme EBI/A outliers
    """
    n0 = len(df)
    print(f"\n  [{label}] 原始: {n0} 行, {df['firm_id'].nunique()} firms")

    # Backfill ownership (important for 2014)
    df = backfill_ownership(df)

    # Remove financials
    df = df[~df['industry_l2'].isin(EXCLUDE_INDUSTRIES_L2)].copy()
    n1 = len(df)
    dropped_fin = n0 - n1
    if dropped_fin > 0:
        print(f"    剔除金融: {dropped_fin} 行")

    # Check ownership distribution
    own_dist = df['ownership'].value_counts()
    print(f"    所有制分布: {dict(own_dist)}")

    # Drop rows with NaN EBI/A
    df = df[df['EBI_A'].notna()].copy()
    n2 = len(df)
    print(f"    有效 EBI/A: {n2} 行 → {df['firm_id'].nunique()} firms")

    # Drop extreme EBI/A outliers (>100% or <-100%)
    df = df[(df['EBI_A'] > -1.0) & (df['EBI_A'] < 1.0)].copy()
    n3 = len(df)
    print(f"    EBI/A 在(-1,1): {n3} 行")

    # Drop rows with '' ownership (still unclassified after backfill)
    df = df[df['ownership'].notna() & (df['ownership'] != '')].copy()
    n4 = len(df)
    if n4 < n3:
        print(f"    剔除未知所有制: {n3 - n4} 行")

    return df


# ============================================================================
# 4. Feature engineering
# ============================================================================

NUMERIC_FEATURES = [
    'FirmSize', 'AssetTurnover', 'FinancialLeverage', 'FixedAssetsRatio',
    'CurrentRatio', 'CapexRatio', 'WorkingCapitalRatio', 'MarketShare', 'HHI',
]


def winsorize_series(s, lower=0.05, upper=0.95):
    """Clip a series at given quantiles."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi), lo, hi


def prepare_train_pred(all_a, csi500_df, csi1000_df=None):
    """
    Split A-share data into training (non-SOE) and merge with prediction targets.

    Returns:
      train_df: all A-share non-SOE firms
      pred_dfs: dict of prediction DataFrames
    """
    # Classify SOE vs non-SOE in all A-shares
    all_a['is_soe'] = all_a['ownership'].isin(SOE_TYPES)
    all_a['is_non_soe_private'] = all_a['ownership'] == NON_SOE_TYPE
    all_a['benchmark_group'] = all_a.apply(
        lambda r: 'SOE' if r['is_soe'] else ('nonSOE_private' if r['is_non_soe_private'] else 'other'),
        axis=1
    )

    print(f"\n  分类结果:")
    for grp in ['nonSOE_private', 'SOE', 'other']:
        cnt = (all_a['benchmark_group'] == grp).sum()
        firms = all_a[all_a['benchmark_group'] == grp]['firm_id'].nunique()
        print(f"    {grp}: {cnt} obs, {firms} firms")

    # Training: non-SOE private enterprises
    train_df = all_a[all_a['benchmark_group'] == 'nonSOE_private'].copy()

    # Prediction targets
    pred_dfs = {}

    # --- CSI500 SOE ---
    csi500_df['is_soe'] = csi500_df['ownership'].isin(SOE_TYPES)
    csi500_soe_pred = csi500_df[csi500_df['is_soe']].copy()
    pred_dfs['CSI500_SOE'] = csi500_soe_pred

    # --- Load CSI300 constituent list ---
    import json
    csi300_codes_path = DATA_DIR.parent / "outputs" / "csi300_constituents.json"
    if csi300_codes_path.exists():
        with open(csi300_codes_path) as f:
            csi300_codes = set(json.load(f))
        print(f"    加载CSI300成分股: {len(csi300_codes)} codes")
    else:
        # Fallback: extract from original project
        old_data_dir = Path("/Users/violet/Desktop/DURF/模型/three_benchmark_model_package/data_input")
        csi300_codes = set()
        for year_dir in old_data_dir.iterdir():
            if not year_dir.is_dir():
                continue
            csi_file = year_dir / f"csi_features_{year_dir.name}.csv"
            if csi_file.exists():
                try:
                    tmp = pd.read_csv(csi_file)
                    csi300_codes.update(tmp['证券代码'].dropna().unique())
                except:
                    pass
        print(f"    提取CSI300成分股: {len(csi300_codes)} codes (从旧项目)")

    # --- CSI300 SOE 2025 ---
    csi300_soe_2025 = all_a[(all_a['benchmark_group'] == 'SOE') &
                             (all_a['target_year'] == 2025) &
                             (all_a['firm_id'].isin(csi300_codes))].copy()
    pred_dfs['CSI300_SOE'] = csi300_soe_2025

    # --- CSI1000 SOE (new!) ---
    if csi1000_df is not None:
        csi1000_df['is_soe'] = csi1000_df['ownership'].isin(SOE_TYPES)
        csi1000_soe_pred = csi1000_df[csi1000_df['is_soe']].copy()
        pred_dfs['CSI1000_SOE'] = csi1000_soe_pred

    # --- Union targets (built from original index files, NOT all_a) ---
    # CSI300 data comes from all_a (no separate CSI300 file), but CSI500/CSI1000
    # have their own data files. Build unions by concatenating original sources
    # to ensure same firm uses same feature data as in individual index targets.

    # CSI300 ∪ CSI500: concat CSI300 (from all_a) + CSI500 (from csi500 file), dedupe
    union_300_500 = pd.concat([csi300_soe_2025, csi500_soe_pred], ignore_index=True)
    union_300_500 = union_300_500.drop_duplicates(subset=['firm_id'], keep='first')
    pred_dfs['CSI300∪CSI500_SOE'] = union_300_500

    # CSI500 ∪ CSI1000: concat CSI500 + CSI1000 (both from original files), dedupe
    union_500_1000 = pd.concat([csi500_soe_pred, csi1000_soe_pred], ignore_index=True)
    union_500_1000 = union_500_1000.drop_duplicates(subset=['firm_id'], keep='first')
    pred_dfs['CSI500∪CSI1000_SOE'] = union_500_1000

    # CSI300 ∪ CSI500 ∪ CSI1000: concat all three, dedupe
    union_all = pd.concat([csi300_soe_2025, csi500_soe_pred, csi1000_soe_pred], ignore_index=True)
    union_all = union_all.drop_duplicates(subset=['firm_id'], keep='first')
    pred_dfs['CSI300∪500∪1000_SOE'] = union_all

    # --- A-share SOE 2025 (from all_a, as reference for full coverage) ---
    all_soe_2025 = all_a[(all_a['benchmark_group'] == 'SOE') &
                         (all_a['target_year'] == 2025)].copy()
    pred_dfs['A-share_SOE_2025'] = all_soe_2025

    # --- Full A-share SOE panel (all years) for reference ---
    all_soe_pred = all_a[all_a['benchmark_group'] == 'SOE'].copy()
    pred_dfs['A-share_SOE_AllYears'] = all_soe_pred

    for name, pdf in pred_dfs.items():
        if len(pdf) > 0:
            print(f"    预测集 {name}: {len(pdf)} obs, {pdf['firm_id'].nunique()} firms, "
                  f"EBI/A mean={pdf['EBI_A'].mean():.4f}")
        else:
            print(f"    预测集 {name}: EMPTY")

    return train_df, pred_dfs


def winsorize_features_and_target(train_df, pred_df, numeric_features, bounds=(0.05, 0.95)):
    """
    Winsorize features based on training distribution.
    Also winsorize target (EBI/A).
    """
    train = train_df.copy()
    pred = pred_df.copy()
    bounds_info = {}

    lower, upper = bounds

    # Winsorize target (EBI/A) based on training distribution
    y_lo = train['EBI_A'].quantile(lower)
    y_hi = train['EBI_A'].quantile(upper)
    train['EBI_A_winsor'] = train['EBI_A'].clip(y_lo, y_hi)
    pred['EBI_A_winsor'] = pred['EBI_A'].clip(y_lo, y_hi)
    bounds_info['target'] = (y_lo, y_hi)

    # Winsorize features
    for col in numeric_features:
        if col not in train.columns:
            continue
        valid_train = train[col].dropna()
        valid_pred = pred[col].dropna()
        if len(valid_train) == 0:
            continue

        lo = valid_train.quantile(lower)
        hi = valid_train.quantile(upper)
        train[col + '_winsor'] = train[col].clip(lo, hi)
        pred[col + '_winsor'] = pred[col].clip(lo, hi)
        bounds_info[col] = (lo, hi)

    return train, pred, bounds_info


# ============================================================================
# 5. Models
# ============================================================================

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def build_model(model_type='ridge', random_state=42):
    """Build a model instance."""
    if model_type == 'ridge':
        return Ridge(alpha=RIDGE_ALPHA, random_state=random_state)
    elif model_type == 'lasso':
        return Lasso(alpha=0.01, max_iter=5000, random_state=random_state)
    elif model_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            max_features='sqrt', random_state=random_state, n_jobs=-1
        )
    elif model_type == 'gbm':
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")


def run_model(train_df, pred_df, numeric_features, model_type='ridge',
              use_industry_fe=True, random_state=42, train_label='',
              pred_label=''):
    """
    Train model on training set, predict on prediction set.

    Returns dict with results.
    """
    # Prepare data
    num_cols_w = [f + '_winsor' for f in numeric_features
                  if f + '_winsor' in train_df.columns]

    train_valid = train_df.dropna(subset=num_cols_w + ['EBI_A_winsor']).copy()
    pred_valid = pred_df.dropna(subset=num_cols_w).copy()

    if len(train_valid) < 50:
        print(f"    ⚠ 训练样本不足 ({len(train_valid)})，跳过")
        return None

    X_train_num = train_valid[num_cols_w].values
    y_train = train_valid['EBI_A_winsor'].values

    # Impute + scale
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(imp.fit_transform(X_train_num))

    # Industry FE
    if use_industry_fe and 'industry_l2' in train_valid.columns:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ind_train = ohe.fit_transform(train_valid[['industry_l2']].fillna('Unknown'))
        X_train = np.hstack([X_train_num, ind_train])

        # Prediction
        X_pred_num = scaler.transform(imp.transform(pred_valid[num_cols_w].values))
        ind_pred = ohe.transform(pred_valid[['industry_l2']].fillna('Unknown'))
        X_pred = np.hstack([X_pred_num, ind_pred])
    else:
        X_train = X_train_num
        X_pred = scaler.transform(imp.transform(pred_valid[num_cols_w].values))

    y_pred_actual = pred_valid['EBI_A'].values

    # CV metrics (GroupKFold by firm_id)
    groups = train_valid['firm_id'].values
    n_splits = min(5, len(set(groups)))

    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        model_cv = build_model(model_type, random_state)
        try:
            cv_preds = cross_val_predict(
                Pipeline([('model', model_cv)]),
                X_train, y_train,
                cv=gkf.split(X_train, y_train, groups=groups)
            )
            cv_r2 = r2_score(y_train, cv_preds)
            cv_rmse = np.sqrt(mean_squared_error(y_train, cv_preds))
            cv_mae = mean_absolute_error(y_train, cv_preds)
        except Exception as e:
            cv_r2 = np.nan
            cv_rmse = np.nan
            cv_mae = np.nan
    else:
        cv_r2 = np.nan
        cv_rmse = np.nan
        cv_mae = np.nan

    # Fit on full training set
    model = build_model(model_type, random_state)
    model.fit(X_train, y_train)

    # ---- In-sample training error (for gap-vs-noise assessment) ----
    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_mae = mean_absolute_error(y_train, train_preds)
    train_std = np.std(y_train)

    # Predict
    preds = model.predict(X_pred)

    # Compute gap
    gap = preds - y_pred_actual
    weight = pred_valid['资产总计_target'].values if '资产总计_target' in pred_valid.columns else None

    if weight is not None and len(weight) > 0 and not np.all(np.isnan(weight)):
        valid_mask = ~np.isnan(weight) & (weight > 0)
        aw_gap = np.average(gap[valid_mask], weights=weight[valid_mask]) if valid_mask.any() else np.nan
    else:
        aw_gap = np.nan

    mean_gap = np.mean(gap)
    pos_share = np.mean(gap > 0)

    # Feature importance (for tree models)
    if model_type in ('random_forest', 'gbm') and hasattr(model, 'feature_importances_'):
        feature_names = num_cols_w.copy()
        if use_industry_fe and 'industry_l2' in train_valid.columns:
            feature_names += list(ohe.get_feature_names_out(['industry_l2']))
        importances = model.feature_importances_
        # Get top 10
        top_idx = np.argsort(importances)[-10:][::-1]
        fi_dict = {feature_names[i]: importances[i] for i in top_idx if i < len(feature_names)}
    else:
        fi_dict = {}

    # Build detailed predictions DataFrame
    det = pred_valid[['firm_id', 'firm_name', 'ownership', 'industry_l1',
                       'industry_l2', 'EBI_A', '资产总计_target']].copy()
    det['predicted_EBI_A'] = preds
    det['actual_EBI_A'] = det['EBI_A'].values
    det['gap'] = det['predicted_EBI_A'] - det['actual_EBI_A']
    det['pred_label'] = pred_label
    det['model'] = model_type

    result = {
        'model': model_type,
        'train_label': train_label,
        'pred_label': pred_label,
        'n_train': len(train_valid),
        'n_train_firms': train_valid['firm_id'].nunique(),
        'n_pred': len(pred_valid),
        'n_pred_firms': pred_valid['firm_id'].nunique(),
        'cv_r2': cv_r2,
        'cv_rmse': cv_rmse,
        'cv_mae': cv_mae,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_ebia_std': train_std,
        'mean_gap': mean_gap,
        'aw_gap': aw_gap,
        'pos_share': pos_share,
        'soe_actual_mean': np.mean(y_pred_actual),
        'soe_pred_mean': np.mean(preds),
        'soe_actual_median': np.median(y_pred_actual),
        'soe_pred_median': np.median(preds),
        'feature_importance': fi_dict,
        'predictions': det,
    }

    return result


# ============================================================================
# 6. Main experiment
# ============================================================================

def run_full_experiment():
    print("=" * 70)
    print("  全A股 → 中证500/300/1000 反事实预测模型（含并集目标）")
    print("=" * 70)

    # ---- Step 1: Load A-share data ----
    print("\n[Step 1] 加载全A股数据 (2014-2024)...")
    all_a = load_all_a_shares()
    print(f"  总计: {len(all_a)} 行, {all_a['firm_id'].nunique()} firms")
    print(f"  年份覆盖: {sorted(all_a['target_year'].dropna().unique())}")
    print(f"  所有制: {dict(all_a['ownership'].value_counts())}")

    # ---- Step 2: Load CSI500 data ----
    print("\n[Step 2] 加载中证500数据...")
    csi500 = load_csi500()
    print(f"  总计: {len(csi500)} 行, {csi500['firm_id'].nunique()} firms")
    print(f"  所有制: {dict(csi500['ownership'].value_counts())}")

    # ---- Step 2b: Load CSI1000 data ----
    print("\n[Step 2b] 加载中证1000数据...")
    csi1000 = load_csi1000()
    print(f"  总计: {len(csi1000)} 行, {csi1000['firm_id'].nunique()} firms")
    print(f"  所有制: {dict(csi1000['ownership'].value_counts())}")

    # ---- Step 3: Clean and filter ----
    print("\n[Step 3] 数据清洗...")
    all_a = clean_and_filter(all_a, "全A股")
    csi500 = clean_and_filter(csi500, "CSI500")
    csi1000 = clean_and_filter(csi1000, "CSI1000")

    # ---- Step 4: Split train/pred ----
    print("\n[Step 4] 划分训练/预测集...")
    train_df, pred_dfs = prepare_train_pred(all_a, csi500, csi1000)

    # Remove "other" ownership types from train (公众企业, 外资企业, 集体企业)
    # Keep only 民营企业 as per project convention
    real_train = train_df[train_df['ownership'] == NON_SOE_TYPE].copy()
    print(f"  训练集 final: {len(real_train)} obs, {real_train['firm_id'].nunique()} firms")

    # Also keep multi-year structure: each row has feature_year and target_year
    # The data already has (feature_year, target_year) for each firm
    # We need lag = True for prediction

    # ---- Step 5: Winsorize and run models ----
    print("\n[Step 5] 模型训练与预测...")

    result_rows = []
    detailed_predictions = []

    for pred_name, pred_df in pred_dfs.items():
        if len(pred_df) == 0:
            print(f"\n  {pred_name}: 无预测样本，跳过")
            continue

        print(f"\n{'─' * 60}")
        print(f"  预测目标: {pred_name}")
        print(f"{'─' * 60}")

        # Winsorize based on training data
        train_w, pred_w, bounds = winsorize_features_and_target(
            real_train, pred_df, NUMERIC_FEATURES, bounds=WINSOR_BOUNDS
        )

        # Show data quality
        print(f"    训练集: {len(train_w)} obs, "
              f"EBI/A mean={train_w['EBI_A'].mean():.4f}, "
              f"median={train_w['EBI_A'].median():.4f}")
        print(f"    预测集: {len(pred_w)} obs, "
              f"EBI/A mean={pred_w['EBI_A'].mean():.4f}, "
              f"median={pred_w['EBI_A'].median():.4f}")

        # Run all models
        for model_type in ['ridge', 'random_forest', 'gbm']:
            result = run_model(
                train_w, pred_w, NUMERIC_FEATURES,
                model_type=model_type,
                use_industry_fe=True,
                random_state=RANDOM_STATE,
                train_label='A-share non-SOE',
                pred_label=pred_name,
            )

            if result is None:
                continue

            result_rows.append(result)

            gap_rmse_ratio = abs(result['mean_gap']) / result['train_rmse'] if result['train_rmse'] > 0 else float('nan')
            print(f"    {model_type:15s}: N_train={result['n_train']:5d}  "
                  f"CV R²={result['cv_r2']:+.4f}  "
                  f"TrainRMSE={result['train_rmse']:.4f}  "
                  f"MeanGap={result['mean_gap']:+.4f}  "
                  f"AW_Gap={result['aw_gap']:+.4f}  "
                  f"|Gap|/RMSE={gap_rmse_ratio:.2f}")

            if result['feature_importance']:
                top3 = list(result['feature_importance'].items())[:5]
                for feat, imp in top3:
                    print(f"           {feat}: {imp:.3f}")

            # Collect detailed predictions from result
            if 'predictions' in result:
                detailed_predictions.append(result['predictions'])

    # ---- Step 6: Compile and save results ----
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)

    results_df = pd.DataFrame(result_rows)

    # Summary table
    print("\n  全A股 → 反事实预测 Gap（不同模型×预测目标）:")
    print(f"  {'Model':<15s} {'Target':<20s} {'N_train':>7s} {'N_pred':>6s} "
          f"{'CV_R²':>8s} {'TrainRMSE':>10s} {'MeanGap':>9s} {'AW_Gap':>9s} "
          f"{'|Gap|/RMSE':>11s} {'Sig?':>5s}")
    print(f"  {'─'*15} {'─'*20} {'─'*7} {'─'*6} {'─'*8} {'─'*10} {'─'*9} {'─'*9} "
          f"{'─'*11} {'─'*5}")
    for _, r in results_df.iterrows():
        gr = abs(r['mean_gap']) / r['train_rmse'] if r['train_rmse'] > 0 else float('nan')
        sig = 'NO' if gr < 1.0 else 'YES'
        print(f"  {r['model']:<15s} {r['pred_label']:<20s} "
              f"{r['n_train']:7.0f} {r['n_pred']:6.0f} "
              f"{r['cv_r2']:8.4f} {r['train_rmse']:10.4f} {r['mean_gap']:9.4f} "
              f"{r['aw_gap']:9.4f} {gr:11.3f} {sig:>5s}")

    # ---- Step 7: Output analysis by ownership type and industry ----
    # (for Ridge, RF, GBM as selected models)
    if detailed_predictions:
        det_all = pd.concat(detailed_predictions, ignore_index=True)

        SELECTED_MODELS = ['ridge', 'random_forest', 'gbm']

        for model_name in SELECTED_MODELS:
            model_sub = det_all[det_all['model'] == model_name]
            if len(model_sub) == 0:
                continue

            print(f"\n{'─'*60}")
            print(f"  [{model_name.upper()}] 按所有制分组:")
            print(f"{'─'*60}")
            for pred_name, sub in model_sub.groupby('pred_label'):
                print(f"\n  [{pred_name}]")
                for own_type, own_sub in sub.groupby('ownership'):
                    if len(own_sub) >= 3:
                        print(f"    {own_type}: n={len(own_sub)}, "
                              f"actual={own_sub['actual_EBI_A'].mean():.4f}, "
                              f"pred={own_sub['predicted_EBI_A'].mean():.4f}, "
                              f"gap={own_sub['gap'].mean():.4f}")

        for model_name in SELECTED_MODELS:
            model_sub = det_all[det_all['model'] == model_name]
            if len(model_sub) == 0:
                continue

            print(f"\n{'─'*60}")
            print(f"  [{model_name.upper()}] 按行业分组 (Gap top/bottom 5):")
            print(f"{'─'*60}")
            for pred_name, sub in model_sub.groupby('pred_label'):
                print(f"\n  [{pred_name}]")
                ind_gap = sub.groupby('industry_l2').agg(
                    n=('gap', 'count'),
                    mean_gap=('gap', 'mean'),
                    actual=('actual_EBI_A', 'mean'),
                    predicted=('predicted_EBI_A', 'mean'),
                ).sort_values('mean_gap')

                if len(ind_gap) > 0:
                    print(f"    Bottom 5 (most negative gap):")
                    for idx, row in ind_gap.head(5).iterrows():
                        print(f"      {idx}: n={row['n']:.0f}, gap={row['mean_gap']:+.4f}")
                    print(f"    Top 5 (most positive gap):")
                    for idx, row in ind_gap.tail(5).iterrows():
                        print(f"      {idx}: n={row['n']:.0f}, gap={row['mean_gap']:+.4f}")

        # ---- Save detailed analysis CSVs ----
        # Ownership summary per model per target
        own_rows = []
        for model_name in SELECTED_MODELS:
            model_sub = det_all[det_all['model'] == model_name]
            for pred_name, sub in model_sub.groupby('pred_label'):
                for own_type, own_sub in sub.groupby('ownership'):
                    if len(own_sub) >= 3:
                        own_rows.append({
                            'model': model_name,
                            'target': pred_name,
                            'ownership': own_type,
                            'n': len(own_sub),
                            'actual_mean': own_sub['actual_EBI_A'].mean(),
                            'pred_mean': own_sub['predicted_EBI_A'].mean(),
                            'mean_gap': own_sub['gap'].mean(),
                        })
        if own_rows:
            own_df = pd.DataFrame(own_rows)
            own_df.to_csv(OUTPUT_DIR / 'ownership_breakdown_by_model.csv',
                         index=False, encoding='utf-8-sig')
            print(f"\n  ✓ 所有制拆分: {OUTPUT_DIR / 'ownership_breakdown_by_model.csv'}")

        # Industry summary per model per target
        ind_rows = []
        for model_name in SELECTED_MODELS:
            model_sub = det_all[det_all['model'] == model_name]
            for pred_name, sub in model_sub.groupby('pred_label'):
                ind_gap = sub.groupby('industry_l2').agg(
                    n=('gap', 'count'),
                    mean_gap=('gap', 'mean'),
                    actual_mean=('actual_EBI_A', 'mean'),
                    pred_mean=('predicted_EBI_A', 'mean'),
                ).reset_index()
                ind_gap['model'] = model_name
                ind_gap['target'] = pred_name
                ind_rows.append(ind_gap)
        if ind_rows:
            ind_df = pd.concat(ind_rows, ignore_index=True)
            ind_df.to_csv(OUTPUT_DIR / 'industry_breakdown_by_model.csv',
                         index=False, encoding='utf-8-sig')
            print(f"  ✓ 行业拆分: {OUTPUT_DIR / 'industry_breakdown_by_model.csv'}")

        # ---- Common support analysis ----
        print("\n  共同支撑分析 (Common Support):")
        cs_vars = [f for f in NUMERIC_FEATURES if f + '_winsor' in real_train.columns]
        for pred_name, pred_df in pred_dfs.items():
            if len(pred_df) == 0:
                continue
            print(f"\n  [{pred_name}]")
            for var in cs_vars:
                train_vals = real_train[var].dropna()
                pred_vals = pred_df[var].dropna()
                if len(train_vals) == 0 or len(pred_vals) == 0:
                    continue

                p01, p99 = train_vals.quantile(0.01), train_vals.quantile(0.99)
                p05, p95 = train_vals.quantile(0.05), train_vals.quantile(0.95)
                outside_p1p99 = ((pred_vals < p01) | (pred_vals > p99)).mean()
                outside_p5p95 = ((pred_vals < p05) | (pred_vals > p95)).mean()

                if outside_p5p95 > 0.05:  # Only show problematic variables
                    print(f"    {var}: {outside_p5p95:.1%} SOE outside P5-P95, "
                          f"{outside_p1p99:.1%} outside P1-P99")

    # ---- Save outputs ----
    results_df.to_csv(OUTPUT_DIR / 'counterfactual_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n  ✓ 结果已保存: {OUTPUT_DIR / 'counterfactual_summary.csv'}")

    if detailed_predictions:
        det_all.to_csv(OUTPUT_DIR / 'counterfactual_detailed_predictions.csv',
                       index=False, encoding='utf-8-sig')
        print(f"  ✓ 详细预测: {OUTPUT_DIR / 'counterfactual_detailed_predictions.csv'}")

    # Also save the train/pred summary data
    data_summary = {
        'train_obs': len(real_train),
        'train_firms': real_train['firm_id'].nunique(),
        'train_years': str(sorted(real_train['target_year'].dropna().unique())),
        'train_ebia_mean': real_train['EBI_A'].mean(),
    }
    for pred_name, pdf in pred_dfs.items():
        data_summary[f'{pred_name}_obs'] = len(pdf)
        data_summary[f'{pred_name}_firms'] = pdf['firm_id'].nunique() if len(pdf) > 0 else 0
        data_summary[f'{pred_name}_ebia_mean'] = pdf['EBI_A'].mean() if len(pdf) > 0 else np.nan

    pd.Series(data_summary).to_csv(OUTPUT_DIR / 'data_summary.csv', encoding='utf-8-sig')
    print(f"  ✓ 数据摘要: {OUTPUT_DIR / 'data_summary.csv'}")

    return results_df


if __name__ == '__main__':
    results = run_full_experiment()
    print("\n✓ 实验完成")
