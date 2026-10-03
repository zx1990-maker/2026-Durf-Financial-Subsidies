#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 — Private-Firm Training Sample Construction Audit
=========================================================
Traces every step from raw Wind xlsx files → final 27,722 obs / 3,471 firms.
Does NOT modify any original data or existing code.
Outputs: sample_flow.csv, duplicate_firm_years.csv, sample_audit.md
"""

import zipfile, re, os, sys, warnings, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# 0. Configuration — mirrors run_counterfactual.py exactly
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "0803"
OUTPUT_DIR = Path(__file__).resolve().parent  # outputs/revision/
OUTPUT_DIR.mkdir(exist_ok=True)

A_FILES = {year: DATA_DIR / f"A_{year}_origin.xlsx" for year in range(2014, 2025)}

EXCLUDE_INDUSTRIES_L2 = {'银行', '保险', '证券', '多元金融', '非银金融'}
SOE_TYPES = {'中央国有企业', '地方国有企业'}
NON_SOE_TYPE = '民营企业'

# ============================================================================
# 1. XML Parser — EXACT copy from run_counterfactual.py
# ============================================================================

def parse_xlsx_xml(filepath):
    with zipfile.ZipFile(filepath, 'r') as z:
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

        with z.open('xl/worksheets/sheet1.xml') as f:
            sheet_tree = ET.fromstring(f.read().decode('utf-8'))

        rows = sheet_tree.findall('.//ns:row', ns)

    header_row = rows[0]
    headers = {}
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
    first_line = header_text.split('\n')[0].strip()
    year = None
    m = re.search(r'\[报告期\]\s*(\d{4})', header_text)
    if m:
        year = int(m.group(1))
    else:
        m = re.search(r'\((\d{4})\)', first_line)
        if m:
            year = int(m.group(1))
    field_name = first_line
    return field_name, year


def classify_column(header_text):
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
    headers, data_rows, strings = parse_xlsx_xml(filepath)
    col_map = {}
    for col_letter, header_text in headers.items():
        ftype, year = classify_column(header_text)
        col_map[col_letter] = (ftype, year)

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
            'data': {}
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

        if not record['stock_code']:
            continue
        records.append(record)

    return records


def records_to_dataframe(records, expected_feature_year, expected_target_year):
    rows = []
    for rec in records:
        data = rec['data']
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

        industry_cn = rec.get('industry_cn', '')
        parts = industry_cn.split('--')
        ind_l1 = parts[0] if len(parts) > 0 else ''
        ind_l2 = parts[1] if len(parts) > 1 else ''

        if pd.isna(ta_t) or ta_t == 0:
            continue

        if pd.notna(np_t) and pd.notna(ie_t):
            ebi_a = (np_t + ie_t) / ta_t
        else:
            ebi_a = np.nan

        if pd.notna(ta_f) and ta_f > 0:
            firm_size = np.log(ta_f)
        else:
            firm_size = np.nan

        if pd.notna(rev_f) and pd.notna(ta_f) and ta_f > 0:
            asset_turnover = rev_f / ta_f
        else:
            asset_turnover = np.nan

        if pd.notna(tl_f) and pd.notna(ta_f) and ta_f > 0:
            fin_lev = tl_f / ta_f
        else:
            fin_lev = np.nan

        if pd.notna(nfa_f) and pd.notna(ta_f) and ta_f > 0:
            fixed_assets_ratio = nfa_f / ta_f
        else:
            fixed_assets_ratio = np.nan

        if pd.notna(ca_f) and pd.notna(cl_f) and cl_f > 0:
            current_ratio = ca_f / cl_f
        else:
            current_ratio = np.nan

        if pd.notna(capex_f) and pd.notna(ta_f) and ta_f > 0:
            capex_ratio = capex_f / ta_f
        else:
            capex_ratio = np.nan

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
# 2. Audit Pipeline — step-by-step tracking
# ============================================================================

def run_audit():
    """Full audit of training sample construction."""
    flow_steps = []  # Collects step-level counts
    anomalies = []   # Collects all anomalies found

    # ------------------------------------------------------------------
    # STEP 0: Verify source files exist
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  Task 1 — Private-Firm Training Sample Construction Audit")
    print("=" * 70)

    print("\n[Step 0] Source file verification")
    for year in range(2014, 2025):
        fp = A_FILES[year]
        exists = fp.exists()
        size_mb = fp.stat().st_size / (1024*1024) if exists else 0
        print(f"  {fp.name}: {'EXISTS' if exists else 'MISSING'} ({size_mb:.1f} MB)")
        if not exists:
            anomalies.append(f"MISSING_FILE: {fp.name}")

    # ------------------------------------------------------------------
    # STEP 1: Parse raw records from each file (before records_to_dataframe)
    # ------------------------------------------------------------------
    print("\n[Step 1] Parse raw records from each xlsx file")

    step1_data = {}  # year -> list of records
    step1_counts = {}

    for year in range(2014, 2025):
        filepath = A_FILES[year]
        if not filepath.exists():
            continue
        records = parse_file_to_records(filepath)
        stock_codes = set(r['stock_code'] for r in records if r['stock_code'])
        # Check for empty stock_code
        empty_codes = sum(1 for r in records if not r['stock_code'])
        # Check ownership distribution
        owns = Counter(r['ownership'] for r in records)
        # Check for duplicate stock_codes within same file
        codes_list = [r['stock_code'] for r in records if r['stock_code']]
        dupes_in_file = [c for c, cnt in Counter(codes_list).items() if cnt > 1]

        step1_data[year] = records
        step1_counts[year] = {
            'n_records': len(records),
            'n_unique_codes': len(stock_codes),
            'n_empty_code': empty_codes,
            'ownership_dist': dict(owns),
            'duplicates_in_file': dupes_in_file,
        }
        print(f"  {filepath.name}: {len(records)} records, {len(stock_codes)} unique codes, "
              f"{empty_codes} empty codes, ownership={dict(owns)}")
        if dupes_in_file:
            print(f"    ⚠ DUPLICATE stock_codes in same file: {dupes_in_file}")
            anomalies.append(f"DUPLICATE_IN_FILE: {filepath.name} has {len(dupes_in_file)} duplicate stock codes: {dupes_in_file}")

    # ------------------------------------------------------------------
    # STEP 2: records_to_dataframe (creates firm-year with EBI/A + features)
    # ------------------------------------------------------------------
    print("\n[Step 2] Convert records to firm-year DataFrames (records_to_dataframe)")

    step2_dfs = {}
    step2_stats = []

    for year in range(2014, 2025):
        if year not in step1_data:
            continue
        records = step1_data[year]
        df = records_to_dataframe(records,
                                  expected_feature_year=year,
                                  expected_target_year=year+1)
        df['_source_file'] = f"A_{year}_origin.xlsx"
        step2_dfs[year] = df

        # Track what was dropped in records_to_dataframe:
        # - Rows where 资产总计_target (t+1) is NaN or 0
        n_records = step1_counts[year]['n_records']
        n_df = len(df)
        dropped_in_conversion = n_records - n_df

        n_firms = df['firm_id'].nunique()
        n_missing_ebia = df['EBI_A'].isna().sum()
        n_valid_ebia = df['EBI_A'].notna().sum()

        step2_stats.append({
            'year': year,
            'n_records_in': n_records,
            'n_firm_years_out': n_df,
            'n_firms_out': n_firms,
            'dropped_no_t1_assets': dropped_in_conversion,
            'n_ebia_nan': n_missing_ebia,
            'n_ebia_valid': n_valid_ebia,
        })
        print(f"  {year}→{year+1}: {n_records} records → {n_df} firm-years ({n_firms} firms), "
              f"dropped {dropped_in_conversion} (no t+1 assets), "
              f"EBI/A NaN={n_missing_ebia}, valid={n_valid_ebia}")

    # ------------------------------------------------------------------
    # STEP 3: Concatenate all years
    # ------------------------------------------------------------------
    print("\n[Step 3] Concatenate all years → all_a_raw")

    all_dfs_list = [step2_dfs[y] for y in sorted(step2_dfs.keys())]
    all_a_raw = pd.concat(all_dfs_list, ignore_index=True)
    n3_fy = len(all_a_raw)
    n3_f = all_a_raw['firm_id'].nunique()

    flow_steps.append({
        'step': '0_concat_all',
        'description': 'Concatenate 11 A-share files (2014–2024), after converting to firm-year with t+1 outcomes',
        'n_firms': n3_f,
        'n_firm_years': n3_fy,
        'dropped_firms': 0,
        'dropped_firm_years': 0,
    })

    print(f"  Combined: {n3_fy} firm-years, {n3_f} unique firms")
    print(f"  Years covered (target_year): {sorted(all_a_raw['target_year'].dropna().unique())}")
    print(f"  Ownership distribution: {dict(all_a_raw['ownership'].value_counts(dropna=False))}")

    # Check initial uniqueness
    dup_check = all_a_raw.groupby(['firm_id', 'target_year']).size()
    dup_pairs = dup_check[dup_check > 1]
    if len(dup_pairs) > 0:
        print(f"  ⚠ INITIAL DUPLICATES: {len(dup_pairs)} duplicate (firm_id, target_year) pairs")
        anomalies.append(f"INITIAL_DUPLICATES: {len(dup_pairs)} duplicate (firm_id, target_year) pairs before any filtering")

    # Check for missing firm_id
    missing_fid = all_a_raw['firm_id'].isna().sum() + (all_a_raw['firm_id'] == '').sum()
    if missing_fid > 0:
        print(f"  ⚠ MISSING firm_id: {missing_fid} rows")
        anomalies.append(f"MISSING_FIRM_ID: {missing_fid} rows with empty/NaN firm_id after concatenation")

    # ------------------------------------------------------------------
    # STEP 4: Backfill ownership (2014 has empty ownership)
    # ------------------------------------------------------------------
    print("\n[Step 4] Backfill ownership for firms with missing ownership")

    df_before_backfill = all_a_raw.copy()
    n_missing_own_before = (df_before_backfill['ownership'].isna() | (df_before_backfill['ownership'] == '')).sum()

    # Replicate backfill logic
    known = all_a_raw[all_a_raw['ownership'].notna() & (all_a_raw['ownership'] != '')]
    firm_own = known.sort_values('target_year').groupby('firm_id')['ownership'].last()

    n_filled = 0
    for idx in all_a_raw[all_a_raw['ownership'].isna() | (all_a_raw['ownership'] == '')].index:
        fid = all_a_raw.loc[idx, 'firm_id']
        if fid in firm_own.index:
            all_a_raw.loc[idx, 'ownership'] = firm_own[fid]
            n_filled += 1

    n_missing_own_after = (all_a_raw['ownership'].isna() | (all_a_raw['ownership'] == '')).sum()

    print(f"  Ownership missing before backfill: {n_missing_own_before}")
    print(f"  Ownership backfilled: {n_filled} rows")
    print(f"  Ownership still missing after backfill: {n_missing_own_after}")

    if n_missing_own_after > 0:
        still_missing_firms = all_a_raw[
            all_a_raw['ownership'].isna() | (all_a_raw['ownership'] == '')
        ]['firm_id'].unique()
        print(f"  Firms still with missing ownership: {len(still_missing_firms)}")
        anomalies.append(f"OWNERSHIP_STILL_MISSING: {n_missing_own_after} rows / {len(still_missing_firms)} firms still have missing ownership after backfill")

    # ------------------------------------------------------------------
    # STEP 5: Remove financial industries
    # ------------------------------------------------------------------
    print("\n[Step 5] Remove financial industries")

    n_before_fin = len(all_a_raw)
    f_before_fin = all_a_raw['firm_id'].nunique()

    fin_mask = all_a_raw['industry_l2'].isin(EXCLUDE_INDUSTRIES_L2)
    fin_rows = all_a_raw[fin_mask]
    fin_firms = fin_rows['firm_id'].nunique()

    all_a_no_fin = all_a_raw[~fin_mask].copy()

    n_after_fin = len(all_a_no_fin)
    f_after_fin = all_a_no_fin['firm_id'].nunique()

    dropped_fin_fy = n_before_fin - n_after_fin

    print(f"  Before: {n_before_fin} firm-years, {f_before_fin} firms")
    print(f"  Removed: {dropped_fin_fy} firm-years ({fin_firms} firms) — financial industries")
    print(f"  After: {n_after_fin} firm-years, {f_after_fin} firms")
    print(f"  Removed industries: {fin_rows['industry_l2'].value_counts().to_dict()}")

    flow_steps.append({
        'step': '1_remove_financial',
        'description': 'Remove financial industries (银行, 保险, 证券, 多元金融, 非银金融)',
        'n_firms': f_after_fin,
        'n_firm_years': n_after_fin,
        'dropped_firms': f_before_fin - f_after_fin,
        'dropped_firm_years': dropped_fin_fy,
    })

    # Track firms that exist ONLY in financial industries
    fin_only_firms = set(fin_rows['firm_id'].unique()) - set(all_a_no_fin['firm_id'].unique())
    if len(fin_only_firms) > 0:
        print(f"  Firms exclusively in financial industries (fully dropped): {len(fin_only_firms)}")
        anomalies.append(f"FIN_ONLY_FIRMS: {len(fin_only_firms)} firms appear only in financial industries and are completely dropped")

    # ------------------------------------------------------------------
    # STEP 6: Drop NaN EBI/A
    # ------------------------------------------------------------------
    print("\n[Step 6] Drop rows with NaN EBI/A")

    n_before_ebia = len(all_a_no_fin)
    f_before_ebia = all_a_no_fin['firm_id'].nunique()

    all_a_valid_ebia = all_a_no_fin[all_a_no_fin['EBI_A'].notna()].copy()
    dropped_ebia_fy = n_before_ebia - len(all_a_valid_ebia)

    # Count firms that lose ALL their rows
    firms_before = set(all_a_no_fin['firm_id'].unique())
    firms_after = set(all_a_valid_ebia['firm_id'].unique())
    firms_lost_all = firms_before - firms_after

    n_after_ebia = len(all_a_valid_ebia)
    f_after_ebia = all_a_valid_ebia['firm_id'].nunique()

    print(f"  Before: {n_before_ebia} firm-years, {f_before_ebia} firms")
    print(f"  Dropped: {dropped_ebia_fy} firm-years (NaN EBI/A)")
    print(f"  Firms losing ALL rows: {len(firms_lost_all)}")
    print(f"  After: {n_after_ebia} firm-years, {f_after_ebia} firms")

    if len(firms_lost_all) > 0:
        anomalies.append(f"FIRMS_LOST_ALL_EBIA: {len(firms_lost_all)} firms have no valid EBI/A in any year")

    flow_steps.append({
        'step': '2_drop_nan_ebia',
        'description': 'Drop rows where EBI/A is NaN (missing net profit or interest expense in t+1)',
        'n_firms': f_after_ebia,
        'n_firm_years': n_after_ebia,
        'dropped_firms': f_before_ebia - f_after_ebia,
        'dropped_firm_years': dropped_ebia_fy,
    })

    # ------------------------------------------------------------------
    # STEP 7: Drop extreme EBI/A outliers
    # ------------------------------------------------------------------
    print("\n[Step 7] Drop extreme EBI/A outliers (outside (-1, +1))")

    n_before_ext = len(all_a_valid_ebia)
    f_before_ext = all_a_valid_ebia['firm_id'].nunique()

    extreme_mask = (all_a_valid_ebia['EBI_A'] <= -1.0) | (all_a_valid_ebia['EBI_A'] >= 1.0)
    n_extreme = extreme_mask.sum()
    extreme_firms = all_a_valid_ebia[extreme_mask]['firm_id'].nunique()

    all_a_clean_ebia = all_a_valid_ebia[(all_a_valid_ebia['EBI_A'] > -1.0) & (all_a_valid_ebia['EBI_A'] < 1.0)].copy()

    n_after_ext = len(all_a_clean_ebia)
    f_after_ext = all_a_clean_ebia['firm_id'].nunique()

    # Show extreme values
    extreme_vals = all_a_valid_ebia[extreme_mask][['firm_id', 'firm_name', 'target_year', 'EBI_A', 'ownership']]
    print(f"  Before: {n_before_ext} firm-years, {f_before_ext} firms")
    print(f"  Dropped: {n_extreme} firm-years ({extreme_firms} firms) — EBI/A outside (-1, +1)")
    print(f"  After: {n_after_ext} firm-years, {f_after_ext} firms")
    if n_extreme > 0:
        print(f"  Extreme EBI/A examples:")
        for _, row in extreme_vals.head(5).iterrows():
            print(f"    {row['firm_id']} {row['firm_name']} year={row['target_year']} EBI/A={row['EBI_A']:.4f} own={row['ownership']}")
        anomalies.append(f"EXTREME_EBIA: {n_extreme} rows with EBI/A outside (-1,+1), from {extreme_firms} firms")

    flow_steps.append({
        'step': '3_drop_extreme_ebia',
        'description': 'Drop rows with EBI/A ≤ -1.0 or EBI/A ≥ +1.0',
        'n_firms': f_after_ext,
        'n_firm_years': n_after_ext,
        'dropped_firms': f_before_ext - f_after_ext,
        'dropped_firm_years': n_extreme,
    })

    # ------------------------------------------------------------------
    # STEP 8: Drop unknown ownership (still empty after backfill)
    # ------------------------------------------------------------------
    print("\n[Step 8] Drop rows with empty ownership (unclassified after backfill)")

    n_before_own = len(all_a_clean_ebia)
    f_before_own = all_a_clean_ebia['firm_id'].nunique()

    empty_own_mask = all_a_clean_ebia['ownership'].isna() | (all_a_clean_ebia['ownership'] == '')
    n_empty_own = empty_own_mask.sum()

    all_a_clean_own = all_a_clean_ebia[all_a_clean_ebia['ownership'].notna() & (all_a_clean_ebia['ownership'] != '')].copy()

    n_after_own = len(all_a_clean_own)
    f_after_own = all_a_clean_own['firm_id'].nunique()

    dropped_own_fy = n_before_own - n_after_own
    firms_lost_own = set(all_a_clean_ebia[empty_own_mask]['firm_id'].unique()) - set(all_a_clean_own['firm_id'].unique())

    print(f"  Before: {n_before_own} firm-years, {f_before_own} firms")
    print(f"  Dropped: {dropped_own_fy} firm-years — empty ownership")
    print(f"  After: {n_after_own} firm-years, {f_after_own} firms")

    if n_empty_own > 0:
        empty_own_firms = all_a_clean_ebia[empty_own_mask]['firm_id'].unique()
        print(f"  Firms with empty ownership: {len(empty_own_firms)}")
        anomalies.append(f"EMPTY_OWNERSHIP_DROPPED: {dropped_own_fy} rows from {len(empty_own_firms)} firms dropped due to empty ownership after backfill")

    flow_steps.append({
        'step': '4_drop_empty_ownership',
        'description': 'Drop rows where ownership is still empty/NaN after backfill',
        'n_firms': f_after_own,
        'n_firm_years': n_after_own,
        'dropped_firms': f_before_own - f_after_own,
        'dropped_firm_years': dropped_own_fy,
    })

    # ------------------------------------------------------------------
    # STEP 9: Filter to 民营企业 ONLY (the training sample)
    # ------------------------------------------------------------------
    print("\n[Step 9] Filter to 民营企业 (private firms) — the training sample")

    # Show full ownership distribution before filtering
    own_dist_full = all_a_clean_own['ownership'].value_counts()
    print(f"  Full ownership distribution:")
    for k, v in own_dist_full.items():
        print(f"    {k}: {v} firm-years, {all_a_clean_own[all_a_clean_own['ownership']==k]['firm_id'].nunique()} firms")

    private_train = all_a_clean_own[all_a_clean_own['ownership'] == NON_SOE_TYPE].copy()
    n_private_fy = len(private_train)
    n_private_f = private_train['firm_id'].nunique()

    n_soe_fy = (all_a_clean_own['ownership'].isin(SOE_TYPES)).sum()
    n_soe_f = all_a_clean_own[all_a_clean_own['ownership'].isin(SOE_TYPES)]['firm_id'].nunique()

    n_other_fy = n_after_own - n_private_fy - n_soe_fy
    n_other_f = all_a_clean_own[~(all_a_clean_own['ownership'].isin(SOE_TYPES)) &
                                  (all_a_clean_own['ownership'] != NON_SOE_TYPE)]['firm_id'].nunique()

    print(f"  民营企业 (TRAINING): {n_private_fy} firm-years, {n_private_f} firms")
    print(f"  SOE (中央+地方): {n_soe_fy} firm-years, {n_soe_f} firms")
    print(f"  Other (公众/外资/集体等): {n_other_fy} firm-years, {n_other_f} firms")

    flow_steps.append({
        'step': '5_filter_private',
        'description': 'Keep only 民营企业 as training sample',
        'n_firms': n_private_f,
        'n_firm_years': n_private_fy,
        'dropped_firms': f_after_own - n_private_f,
        'dropped_firm_years': n_after_own - n_private_fy,
    })

    # ==================================================================
    # COMPREHENSIVE CHECKS ON THE FINAL TRAINING SAMPLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE CHECKS ON FINAL TRAINING SAMPLE")
    print("=" * 70)

    # --- Check 1: (firm_id, target_year) uniqueness ---
    print("\n[Check 1] (firm_id, target_year) uniqueness")
    dup_check = private_train.groupby(['firm_id', 'target_year']).size()
    dup_pairs = dup_check[dup_check > 1]
    n_duplicate_pairs = len(dup_pairs)
    n_duplicate_rows = dup_pairs.sum() - n_duplicate_pairs

    if n_duplicate_pairs > 0:
        print(f"  ❌ FAIL: {n_duplicate_pairs} duplicate (firm_id, target_year) pairs ({n_duplicate_rows} excess rows)")
        # Extract all duplicate records
        dup_fids = dup_pairs.index.tolist()
        dup_records = private_train[
            private_train.set_index(['firm_id', 'target_year']).index.isin(dup_fids)
        ].sort_values(['firm_id', 'target_year'])
        dup_records.to_csv(OUTPUT_DIR / 'duplicate_firm_years.csv', index=False, encoding='utf-8-sig')
        print(f"  → Saved {len(dup_records)} duplicate records to duplicate_firm_years.csv")
        anomalies.append(f"DUPLICATE_FIRM_YEARS: {n_duplicate_pairs} duplicate (firm_id, target_year) pairs with {n_duplicate_rows} excess rows")
    else:
        print(f"  ✓ PASS: All (firm_id, target_year) pairs are unique")
        # Create empty duplicate file
        pd.DataFrame(columns=['firm_id', 'firm_name', 'target_year', 'EBI_A', 'ownership', '_source_file']).to_csv(
            OUTPUT_DIR / 'duplicate_firm_years.csv', index=False, encoding='utf-8-sig')

    # --- Check 2: Missing firm_id ---
    print("\n[Check 2] Missing firm_id")
    missing_fid = private_train['firm_id'].isna().sum() + (private_train['firm_id'] == '').sum()
    if missing_fid > 0:
        print(f"  ❌ FAIL: {missing_fid} rows with missing firm_id")
        anomalies.append(f"MISSING_FIRM_ID_FINAL: {missing_fid} rows")
    else:
        print(f"  ✓ PASS: No missing firm_id")

    # --- Check 3: firm_id format (should be 6-digit numeric string) ---
    print("\n[Check 3] firm_id format check")
    non_standard_fids = private_train[~private_train['firm_id'].str.match(r'^\d{6}$')]
    if len(non_standard_fids) > 0:
        print(f"  ⚠ WARNING: {len(non_standard_fids)} rows with non-standard firm_id format")
        print(f"  Examples: {non_standard_fids['firm_id'].unique()[:10]}")
        anomalies.append(f"NONSTANDARD_FIRM_ID: {len(non_standard_fids)} rows, {non_standard_fids['firm_id'].nunique()} unique non-standard IDs")
    else:
        print(f"  ✓ PASS: All firm_ids match 6-digit format")

    # --- Check 4: ticker changes (same firm_name with different firm_id or vice versa) ---
    print("\n[Check 4] Ticker/name consistency check")
    name_to_ids = private_train.groupby('firm_name')['firm_id'].apply(lambda x: list(set(x)))
    multi_id_names = {k: v for k, v in name_to_ids.items() if len(v) > 1}
    if len(multi_id_names) > 0:
        print(f"  ⚠ WARNING: {len(multi_id_names)} firm names associated with multiple firm_ids (possible ticker changes/name reuse)")
        for name, ids in list(multi_id_names.items())[:10]:
            print(f"    '{name}': {ids}")
        anomalies.append(f"MULTI_ID_SAME_NAME: {len(multi_id_names)} firm names map to multiple firm_ids")
    else:
        print(f"  ✓ PASS: Each firm_name maps to exactly one firm_id")

    id_to_names = private_train.groupby('firm_id')['firm_name'].apply(lambda x: list(set(x)))
    multi_name_ids = {k: v for k, v in id_to_names.items() if len(v) > 1}
    if len(multi_name_ids) > 0:
        print(f"  ⚠ WARNING: {len(multi_name_ids)} firm_ids associated with multiple names (name changes over time)")
        for fid, names in list(multi_name_ids.items())[:10]:
            print(f"    {fid}: {names}")
        anomalies.append(f"NAME_CHANGE: {len(multi_name_ids)} firm_ids changed names over time")

    # --- Check 5: Panel balance ---
    print("\n[Check 5] Panel balance")
    years_range = sorted(private_train['target_year'].dropna().unique())
    print(f"  Target years covered: {years_range}")
    print(f"  Expected: 2015–2025 (11 years)")

    firm_year_counts = private_train.groupby('firm_id')['target_year'].nunique()
    print(f"  Firms with 1 year:  {(firm_year_counts == 1).sum()}")
    print(f"  Firms with 2-4 years: {((firm_year_counts >= 2) & (firm_year_counts <= 4)).sum()}")
    print(f"  Firms with 5-8 years: {((firm_year_counts >= 5) & (firm_year_counts <= 8)).sum()}")
    print(f"  Firms with 9-10 years: {((firm_year_counts >= 9) & (firm_year_counts <= 10)).sum()}")
    print(f"  Firms with all 11 years: {(firm_year_counts == 11).sum()}")
    print(f"  Mean years per firm: {firm_year_counts.mean():.1f}")
    print(f"  Median years per firm: {firm_year_counts.median():.0f}")

    n_balanced = (firm_year_counts == 11).sum()
    if n_balanced < n_private_f:
        print(f"  ⚠ NOTE: Only {n_balanced}/{n_private_f} ({n_balanced/n_private_f:.1%}) firms present in all 11 years")
        print(f"  → Panel is UNBALANCED — survivorship concern exists")

    # --- Check 6: Year-by-year firm counts ---
    print("\n[Check 6] Year-by-year composition")
    yearly = private_train.groupby('target_year').agg(
        n_firms=('firm_id', 'nunique'),
        n_obs=('firm_id', 'count'),
        ebia_mean=('EBI_A', 'mean'),
        ebia_median=('EBI_A', 'median'),
        ebia_std=('EBI_A', 'std'),
    )
    for yr, row in yearly.iterrows():
        print(f"  {int(yr)}: {int(row['n_obs']):5d} obs, {int(row['n_firms']):4d} firms, "
              f"EBI/A mean={row['ebia_mean']:+.4f}, median={row['ebia_median']:+.4f}, sd={row['ebia_std']:.4f}")

    # --- Check 7: Feature coverage ---
    print("\n[Check 7] Feature missingness in final training sample")
    feature_cols = ['FirmSize', 'AssetTurnover', 'FinancialLeverage', 'FixedAssetsRatio',
                    'CurrentRatio', 'CapexRatio', 'WorkingCapitalRatio', 'MarketShare', 'HHI']
    for col in feature_cols:
        if col in private_train.columns:
            n_miss = private_train[col].isna().sum()
            pct_miss = n_miss / len(private_train) * 100
            if pct_miss > 0:
                print(f"  {col}: {n_miss} NaN ({pct_miss:.1f}%)")
        else:
            print(f"  {col}: COLUMN NOT FOUND")

    # --- Check 8: EBI/A distribution summary ---
    print("\n[Check 8] EBI/A distribution in training sample")
    ebia = private_train['EBI_A'].dropna()
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for q in qs:
        print(f"  P{int(q*100):2d}: {np.quantile(ebia, q):+.4f}")

    # --- Check 9: Extreme outliers within accepted range ---
    print("\n[Check 9] Near-boundary EBI/A values")
    near_neg = private_train[(private_train['EBI_A'] > -1.0) & (private_train['EBI_A'] < -0.5)]
    near_pos = private_train[(private_train['EBI_A'] < 1.0) & (private_train['EBI_A'] > 0.5)]
    print(f"  EBI/A in (-1.0, -0.5): {len(near_neg)} obs from {near_neg['firm_id'].nunique()} firms")
    print(f"  EBI/A in (+0.5, +1.0): {len(near_pos)} obs from {near_pos['firm_id'].nunique()} firms")

    # --- Check 10: Industry coverage ---
    print("\n[Check 10] Industry coverage")
    n_industries = private_train['industry_l2'].nunique()
    print(f"  Unique industry_l2: {n_industries}")
    top_inds = private_train['industry_l2'].value_counts().head(10)
    print(f"  Top 10 industries by obs:")
    for ind, cnt in top_inds.items():
        print(f"    {ind}: {cnt} obs, {private_train[private_train['industry_l2']==ind]['firm_id'].nunique()} firms")

    # --- Check 11: Survivorship concern — entry/exit patterns ---
    print("\n[Check 11] Entry/exit patterns (survivorship)")
    min_year_by_firm = private_train.groupby('firm_id')['target_year'].min()
    max_year_by_firm = private_train.groupby('firm_id')['target_year'].max()

    new_firms_by_year = min_year_by_firm.value_counts().sort_index()
    exit_firms_by_year = max_year_by_firm.value_counts().sort_index()
    print(f"  First appearance by target_year:")
    for yr, cnt in new_firms_by_year.items():
        print(f"    {int(yr)}: {cnt} firms first appear")
    print(f"  Last appearance by target_year:")
    for yr, cnt in exit_firms_by_year.items():
        print(f"    {int(yr)}: {cnt} firms last appear")

    # Exit before 2025 = attrition
    exited_before_end = (max_year_by_firm < 2025).sum()
    print(f"  Firms exiting before 2025: {exited_before_end} ({exited_before_end/n_private_f:.1%})")
    if exited_before_end > 0:
        print(f"  → SURVIVORSHIP CONCERN: {exited_before_end} firms do not survive to end of panel")
        anomalies.append(f"SURVIVORSHIP: {exited_before_end}/{n_private_f} firms exit before 2025")

    # Entered after 2015
    entered_after_start = (min_year_by_firm > 2015).sum()
    print(f"  Firms entering after 2015: {entered_after_start} ({entered_after_start/n_private_f:.1%})")

    if entered_after_start > 0:
        anomalies.append(f"LATE_ENTRY: {entered_after_start}/{n_private_f} firms enter after 2015")

    # ==================================================================
    # REPRODUCIBILITY CHECK
    # ==================================================================
    print("\n" + "=" * 70)
    print("  REPRODUCIBILITY CHECK")
    print("=" * 70)
    print(f"\n  EXPECTED: 27,722 firm-year observations, 3,471 firms")
    print(f"  ACTUAL:   {n_private_fy} firm-year observations, {n_private_f} firms")
    print(f"  DIFF:     {n_private_fy - 27722:+d} obs, {n_private_f - 3471:+d} firms")

    exact_match = (n_private_fy == 27722) and (n_private_f == 3471)
    if exact_match:
        print(f"\n  ✓ EXACT MATCH — 27,722 / 3,471 reproduced exactly")
        repro_status = "PASS — Exact reproduction"
    else:
        print(f"\n  ❌ MISMATCH — Observed {n_private_fy}/{n_private_f} vs expected 27,722/3,471")
        repro_status = f"FAIL — Difference: obs={n_private_fy-27722:+d}, firms={n_private_f-3471:+d}"

    # ==================================================================
    # SAVE OUTPUTS
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SAVING OUTPUTS")
    print("=" * 70)

    # A. sample_flow.csv
    flow_df = pd.DataFrame(flow_steps)
    flow_df.to_csv(OUTPUT_DIR / 'sample_flow.csv', index=False, encoding='utf-8-sig')
    print(f"  ✓ sample_flow.csv ({len(flow_df)} steps)")

    # B. duplicate_firm_years.csv (already saved above if duplicates found)
    print(f"  ✓ duplicate_firm_years.csv")

    # C. sample_audit.md
    write_audit_report(
        flow_df, n_private_fy, n_private_f, exact_match,
        firm_year_counts, yearly, anomalies, private_train,
        n_duplicate_pairs, repro_status,
        exited_before_end, entered_after_start, n_balanced,
        multi_id_names, multi_name_ids,
    )
    print(f"  ✓ sample_audit.md")

    print(f"\n{'='*70}")
    print(f"  AUDIT COMPLETE — {repro_status}")
    print(f"{'='*70}")

    return private_train, flow_df, anomalies


def write_audit_report(flow_df, n_fy, n_f, exact_match, firm_year_counts, yearly,
                       anomalies, private_train, n_dupes, repro_status,
                       exited_before_end, entered_after_start, n_balanced,
                       multi_id_names, multi_name_ids):
    """Write the comprehensive audit markdown report."""

    lines = []
    def w(s):
        lines.append(s)

    w("# Private-Firm Training Sample Construction Audit")
    w("")
    w(f"**Date**: 2026-08-11")
    w(f"**Auditor**: Automated pipeline audit script")
    w(f"**Final Result**: **{repro_status}**")
    w("")

    # --- Executive Summary ---
    w("## Executive Summary")
    w("")
    if exact_match:
        w(f"- ✅ **Exact reproduction confirmed**: 27,722 firm-year observations, 3,471 unique private firms")
    else:
        w(f"- ❌ **Reproduction failed**: Expected 27,722 / 3,471, got {n_fy} / {n_f}")
    w(f"- **Duplicate (firm_id, year) pairs**: {n_dupes} ({'PASS' if n_dupes == 0 else 'FAIL'})")
    w(f"- **Survivorship concern**: {'YES' if exited_before_end > 0 else 'NO'} — {exited_before_end} firms exit before 2025, {entered_after_start} enter after 2015")
    w(f"- **Panel type**: Unbalanced — only {n_balanced}/{n_f} ({n_balanced/n_f*100:.1f}%) firms present in all 11 years")
    n_anomalies = len([a for a in anomalies if not a.startswith('SURVIVORSHIP') and not a.startswith('LATE_ENTRY')])
    w(f"- **Anomalies found**: {len(anomalies)} total, {n_anomalies} critical")
    w("")
    w(f"**Final verdict**: {'**PASS**' if exact_match and n_dupes == 0 else '**WARNING**' if exact_match else '**FAIL**'}")
    w("")

    # --- Sample Construction Steps ---
    w("## 1. Sample Construction Process")
    w("")
    w("### 1.1 Data Sources")
    w("")
    w("| File | Feature Year (t) | Target Year (t+1) | Format |")
    w("|------|-----------------|-------------------|--------|")
    for year in range(2014, 2025):
        w(f"| `0803/A_{year}_origin.xlsx` | {year} | {year+1} | Wind xlsx (XML-parsed) |")
    w("")
    w("### 1.2 Step-by-Step Construction")
    w("")
    w("| Step | Description | Firms | Firm-Years | Δ Firms | Δ Firm-Years |")
    w("|------|-------------|-------|------------|---------|-------------|")
    for _, row in flow_df.iterrows():
        w(f"| {row['step']} | {row['description']} | {int(row['n_firms']):,} | {int(row['n_firm_years']):,} | {int(row['dropped_firms']):,} | {int(row['dropped_firm_years']):,} |")
    w("")

    # --- Detailed Step Explanations ---
    w("### 1.3 Detailed Step Explanations")
    w("")
    w("**Step 0 — XML Parsing**: Each `.xlsx` file is parsed via direct XML extraction from the zip archive (`xl/sharedStrings.xml` + `xl/worksheets/sheet1.xml`). This bypasses openpyxl style corruption. Column headers are classified into semantic types (`stock_code`, `ownership`, `net_profit`, etc.) by matching Wind header patterns. Report years are extracted from `[报告期] YYYY` tags.")
    w("")
    w("**Step 0→1 — records_to_dataframe**: Records with missing `资产总计` in the target year (t+1) are dropped at this stage — they cannot compute EBI/A. Each record becomes one firm-year row with:")
    w("- `firm_id = stock_code` (6-digit numeric string)")
    w("- `EBI/A = (归母净利润_t+1 + 利息支出_t+1) / 资产总计_t+1`")
    w("- 9 feature ratios computed from t-year financial data")
    w("- `MarketShare` and `HHI` computed within each (feature_year, industry_l2) group")
    w("")
    w("**Step 1 — Concatenation**: All 11 yearly DataFrames are concatenated with `pd.concat(..., ignore_index=True)`.")
    w("")
    w("**Step 4 — Ownership Backfill**: The 2014 file has an empty `ownership` column for all rows. Ownership is backfilled using each firm's most recent known ownership from later years (`sort_values('target_year').groupby('firm_id')['ownership'].last()`). Firms that never appear in any later year remain with empty ownership and are dropped in Step 8.")
    w("")
    w("**Step 5 — Financial Industry Exclusion**: Industries `{银行, 保险, 证券, 多元金融, 非银金融}` are dropped. These are Wind二级行业 codes. Financial firms have fundamentally different balance sheet structures (high leverage, different asset composition), making their EBI/A not comparable to non-financial firms.")
    w("")
    w("**Step 6 — NaN EBI/A Drop**: Rows where either `归母净利润_t+1` or `利息支出_t+1` is missing are dropped. This is the largest single drop. These are firms with incomplete Wind data for the target year.")
    w("")
    w("**Step 7 — Extreme EBI/A Censoring**: EBI/A values ≤ −100% or ≥ +100% are dropped. These are economically implausible and likely data errors or extreme one-time events (massive write-downs, restructuring).")
    w("")
    w("**Step 8 — Unknown Ownership Drop**: After backfill, any remaining rows with empty ownership are dropped. These are typically firms that only appear in 2014 and have no later-year ownership reference.")
    w("")
    w("**Step 9 — Private Firm Filter**: Only firms with `ownership == '民营企业'` are retained as the training sample. SOE (`中央国有企业`, `地方国有企业`) and other types (`公众企业`, `外资企业`, `集体企业`, etc.) are excluded.")
    w("")

    # --- Uniqueness Check ---
    w("## 2. Uniqueness Check")
    w("")
    if n_dupes == 0:
        w("✅ **PASS**: All `(firm_id, target_year)` pairs are unique. No duplicate firm-year observations.")
    else:
        w(f"❌ **FAIL**: {n_dupes} duplicate `(firm_id, target_year)` pairs found. See `duplicate_firm_years.csv` for details.")
    w("")

    # --- Ticker/Name Changes ---
    w("## 3. Identifier Consistency")
    w("")
    if len(multi_id_names) > 0:
        w(f"⚠️ **WARNING**: {len(multi_id_names)} firm names are associated with multiple `firm_id` values. This can occur when:")
        w("- A company changes its stock code (rare in A-shares)")
        w("- Different companies share the same abbreviated name")
        w("- Wind data has name mapping errors")
        w("")
        w("Examples:")
        for name, ids in list(multi_id_names.items())[:5]:
            w(f"- `{name}` → {ids}")
    else:
        w("✅ Each firm_name maps to exactly one firm_id.")
    w("")

    if len(multi_name_ids) > 0:
        w(f"ℹ️ **INFO**: {len(multi_name_ids)} firm_ids have name changes over time. This is normal — companies may change their registered name (e.g., after restructuring). Using `firm_id` (stock code) as the panel identifier is correct.")
    w("")

    # --- Survivorship ---
    w("## 4. Survivorship Analysis")
    w("")
    w(f"- **Firms present in all 11 years (2015–2025)**: {n_balanced}/{n_f} ({n_balanced/n_f*100:.1f}%)")
    w(f"- **Firms exiting before 2025**: {exited_before_end} ({exited_before_end/n_f*100:.1f}%)")
    w(f"- **Firms entering after 2015**: {entered_after_start} ({entered_after_start/n_f*100:.1f}%)")
    w(f"- **Mean years per firm**: {firm_year_counts.mean():.1f}")
    w(f"- **Median years per firm**: {firm_year_counts.median():.0f}")
    w("")
    w("### Survivorship Concern Assessment")
    w("")
    if exited_before_end / n_f > 0.3:
        w("⚠️ **SUBSTANTIAL**: More than 30% of firms exit before the panel end. This creates a potential survivorship bias — firms that survive to 2025 may be systematically different from those that exit earlier (delisting, M&A, bankruptcy). The training sample over-represents survivors.")
    elif exited_before_end / n_f > 0.1:
        w("⚠️ **MODERATE**: Between 10-30% of firms exit before the panel end. Some survivorship bias may exist but the panel retains substantial coverage across all years.")
    else:
        w("✅ **LOW**: Less than 10% of firms exit before the panel end. Survivorship concern is minimal.")
    w("")
    w("**Note**: The model is trained on ALL available private firm-years, including firms that later exit. Prediction targets (SOEs in 2025) only include surviving firms. The training sample's inclusion of non-survivors is actually DESIRABLE — it prevents the model from learning only survivor characteristics.")
    w("")

    # --- Anomalies ---
    w("## 5. Anomalies Found")
    w("")
    if len(anomalies) == 0:
        w("✅ No anomalies found.")
    else:
        for a in anomalies:
            w(f"- ⚠️ {a}")
    w("")

    # --- Year-by-Year Summary ---
    w("## 6. Year-by-Year Training Sample Composition")
    w("")
    w("| Target Year | Observations | Unique Firms | Mean EBI/A | Median EBI/A | Std EBI/A |")
    w("|-------------|-------------|-------------|------------|-------------|-----------|")
    for yr, row in yearly.iterrows():
        w(f"| {int(yr)} | {int(row['n_obs']):,} | {int(row['n_firms']):,} | {row['ebia_mean']:+.4f} | {row['ebia_median']:+.4f} | {row['ebia_std']:.4f} |")
    w("")

    # --- EBI/A Distribution ---
    w("## 7. EBI/A Distribution (Training Sample)")
    w("")
    ebia = private_train['EBI_A'].dropna()
    w("| Percentile | EBI/A |")
    w("|-----------|-------|")
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        w(f"| P{q:2d} | {np.quantile(ebia, q/100):+.4f} |")
    w(f"| **Mean** | **{ebia.mean():+.4f}** |")
    w(f"| **Std** | **{ebia.std():.4f}** |")
    w(f"| **N** | **{len(ebia):,}** |")
    w("")

    # --- Paper-Ready Sample Construction Table ---
    w("## 8. Sample Construction Table (Paper-Ready)")
    w("")
    w("| Step | Description | Firm-Years | Firms |")
    w("|------|-------------|-----------|-------|")
    for _, row in flow_df.iterrows():
        if row['step'] == '0_concat_all':
            w(f"| (1) | All A-share non-financial firm-years with valid t+1 assets, 2014–2024 | {int(row['n_firm_years']):,} | {int(row['n_firms']):,} |")
        elif row['step'] == '1_remove_financial':
            w(f"| (2) | Less: Financial industry firms | ({int(row['dropped_firm_years']):,}) | ({int(row['dropped_firms']):,}) |")
        elif row['step'] == '2_drop_nan_ebia':
            w(f"| (3) | Less: Missing EBI/A (incomplete t+1 financials) | ({int(row['dropped_firm_years']):,}) | ({int(row['dropped_firms']):,}) |")
        elif row['step'] == '3_drop_extreme_ebia':
            w(f"| (4) | Less: EBI/A outside (−100%, +100%) | ({int(row['dropped_firm_years']):,}) | ({int(row['dropped_firms']):,}) |")
        elif row['step'] == '4_drop_empty_ownership':
            w(f"| (5) | Less: Unclassified ownership after backfill | ({int(row['dropped_firm_years']):,}) | ({int(row['dropped_firms']):,}) |")
        elif row['step'] == '5_filter_private':
            w(f"| (6) | Less: Non-private ownership types (SOE, public, foreign, collective) | ({int(row['dropped_firm_years']):,}) | ({int(row['dropped_firms']):,}) |")
    w(f"| | **Final private-firm training sample** | **{n_fy:,}** | **{n_f:,}** |")
    w("")
    w("*Notes: This table reports the construction of the private-firm training sample used to estimate the counterfactual EBI/A return-generating process. All data are from Wind Financial Terminal. The panel covers fiscal years 2014–2024 (feature years) with outcomes measured in 2015–2025 (target years). EBI/A = (net profit attributable to parent + interest expense) / total assets. A firm-year is included only if total assets in t+1 is non-missing and positive. Industry classification follows the 2024 Wind二级行业 standard.*")
    w("")

    # --- Final Verdict ---
    w("## 9. Final Verdict")
    w("")
    if exact_match and n_dupes == 0:
        w(f"**PASS** — The training sample of {n_fy:,} firm-year observations ({n_f:,} unique private firms) has been exactly reproduced from raw Wind data files. All processing steps are deterministic and documented. No duplicate firm-year pairs exist. The sample is ready for model training.")
    elif exact_match:
        w(f"**WARNING** — The training sample counts match ({n_fy:,} / {n_f:,}) but {n_dupes} duplicate firm-year pairs were found. These duplicates should be investigated and resolved before model training.")
    else:
        w(f"**FAIL** — The training sample could not be exactly reproduced. Expected 27,722 / 3,471, got {n_fy:,} / {n_f:,}. The discrepancy should be traced to the specific processing step where it first appears.")
    w("")
    w("### Recommended Next Steps")
    w("")
    if n_dupes > 0:
        w("1. Investigate and resolve duplicate (firm_id, target_year) pairs")
    if not exact_match:
        w("1. Trace the exact step where the count divergence occurs")
        w("2. Check for differences in Wind data file versions")
    w("3. Document whether the existing results (27,722 / 3,471) should be updated")
    w("4. Add a deduplication step to the preprocessing pipeline")
    w("")

    # Write
    report_path = OUTPUT_DIR / 'sample_audit.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    df, flow, anomalies = run_audit()
