#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run three-benchmark counterfactual EBI/A models.

Benchmarks:
1. H non-SOE only
2. CSI300 non-SOE only
3. Pooled H + CSI300 non-SOE

Main model:
Ridge(alpha=10), one-hot industries, 2024 features -> 2025 EBI/A.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data_input"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

CSI_SOE_PATH = DATA / "csi——features.csv"
H_NS_PATH = DATA / "h——features.csv"
CSI_NS_PATH = DATA / "csi_ns_2024_features_2025_EBIA_二级行业HHI_结果(1).xlsx"

TARGET_COL = "EBI_A_2025"
NUM_COLS = [
    "FirmSize_2024",
    "MarketShare_2024_二级行业",
    "HHI_2024_二级行业",
    "AssetTurnover_2024",
    "WorkingCapitalRatio_2024",
    "FixedAssetsRatio_2024",
    "FinancialLeverage_2024",
    "CurrentRatio_2024",
    "CapexRatio_2024",
]
BASE_CAT_COLS = ["一级行业", "二级行业"]

COMMON_COLS = [
    "市场", "证券代码", "证券简称", "一级行业", "二级行业", "行业全路径", "企业所有制性质",
    "资产总计_2024", "营业收入_2024", "归母净利润_2025", "利息支出_2025", "资产总计_2025",
    "EBI_A_2025", "FirmSize_2024", "MarketShare_2024_二级行业", "HHI_2024_二级行业",
    "AssetTurnover_2024", "WorkingCapitalRatio_2024", "FixedAssetsRatio_2024",
    "FinancialLeverage_2024", "CurrentRatio_2024", "CapexRatio_2024",
]


def to_numeric_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan, "--": np.nan})
        .astype(float)
    )


def normalize_old(df, market_label):
    df = df.copy()
    if "EBI_A_2025" not in df.columns and "EBI/A_2025" in df.columns:
        df["EBI_A_2025"] = to_numeric_series(df["EBI/A_2025"])
    df["市场"] = market_label

    if "行业全路径" in df.columns:
        if "一级行业" not in df.columns:
            df["一级行业"] = df["行业全路径"].astype(str).str.split("--").str[0]
        if "二级行业" not in df.columns:
            df["二级行业"] = df["行业全路径"].astype(str).str.split("--").str[1]

    for c in COMMON_COLS:
        if c in df.columns and c not in ["市场", "证券代码", "证券简称", "一级行业", "二级行业", "行业全路径", "企业所有制性质"]:
            df[c] = to_numeric_series(df[c])
    return df[COMMON_COLS].copy()


def normalize_csi_ns(df):
    df = df.copy().rename(columns={"EBI/A_2025": "EBI_A_2025"})
    df["市场"] = "CSI"
    df["一级行业"] = df["行业全路径"].astype(str).str.split("--").str[0]
    for c in COMMON_COLS:
        if c in df.columns and c not in ["市场", "证券代码", "证券简称", "一级行业", "二级行业", "行业全路径", "企业所有制性质"]:
            df[c] = to_numeric_series(df[c])
    return df[COMMON_COLS].copy()


def get_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(cat_cols):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_COLS),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", get_ohe())]), cat_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", preprocess), ("model", Ridge(alpha=10.0, random_state=42))])


def clip_features_by_train(train, pred, winsor):
    lower = train[NUM_COLS].quantile(winsor[0], numeric_only=True)
    upper = train[NUM_COLS].quantile(winsor[1], numeric_only=True)
    train = train.copy()
    pred = pred.copy()
    for df in [train, pred]:
        for c in NUM_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower[c], upper[c])
    return train, pred, lower, upper


def fit_predict(train_df, pred_df, cat_cols, winsor=(0.05, 0.95)):
    train = train_df[pd.notna(train_df[TARGET_COL])].reset_index(drop=True).copy()
    pred = pred_df.copy()
    train, pred, lower, upper = clip_features_by_train(train, pred, winsor)

    y_lower = train[TARGET_COL].quantile(winsor[0])
    y_upper = train[TARGET_COL].quantile(winsor[1])
    y = train[TARGET_COL].clip(y_lower, y_upper).astype(float)

    pipe = build_pipeline(cat_cols)
    pipe.fit(train[NUM_COLS + cat_cols], y)
    predictions = pipe.predict(pred[NUM_COLS + cat_cols])
    info = {
        "n_train": len(train),
        "target_lower": y_lower,
        "target_upper": y_upper,
    }
    return pipe, predictions, info


def cv_metrics(train_df, cat_cols, winsor=(0.05, 0.95), n_splits=5):
    train = train_df[pd.notna(train_df[TARGET_COL])].reset_index(drop=True).copy()
    kf = KFold(n_splits=min(n_splits, len(train)), shuffle=True, random_state=42)
    preds = np.full(len(train), np.nan)
    y_true = train[TARGET_COL].astype(float).values

    for tr_idx, te_idx in kf.split(train):
        tr = train.iloc[tr_idx].copy()
        te = train.iloc[te_idx].copy()
        tr, te, _, _ = clip_features_by_train(tr, te, winsor)

        y_lower = tr[TARGET_COL].quantile(winsor[0])
        y_upper = tr[TARGET_COL].quantile(winsor[1])
        y_tr = tr[TARGET_COL].clip(y_lower, y_upper).astype(float)

        pipe = build_pipeline(cat_cols)
        pipe.fit(tr[NUM_COLS + cat_cols], y_tr)
        preds[te_idx] = pipe.predict(te[NUM_COLS + cat_cols])

    y_w = pd.Series(y_true).clip(train[TARGET_COL].quantile(winsor[0]), train[TARGET_COL].quantile(winsor[1])).values
    return {
        "n_train": len(train),
        "cv_rmse_raw_y": float(np.sqrt(mean_squared_error(y_true, preds))),
        "cv_mae_raw_y": float(mean_absolute_error(y_true, preds)),
        "cv_r2_raw_y": float(r2_score(y_true, preds)),
        "cv_rmse_winsor_y": float(np.sqrt(mean_squared_error(y_w, preds))),
        "cv_mae_winsor_y": float(mean_absolute_error(y_w, preds)),
        "cv_r2_winsor_y": float(r2_score(y_w, preds)),
    }


def weighted_avg(g, val_col, weight_col="资产总计_2025"):
    w = g[weight_col].astype(float)
    v = g[val_col].astype(float)
    return np.nan if w.sum() == 0 else float((v * w).sum() / w.sum())


def summarize_predictions(tmp, benchmark, info, winsor_label):
    total_assets = tmp["资产总计_2025"].sum()
    aw_pred = weighted_avg(tmp, "predicted_EBI_A_2025")
    aw_act = weighted_avg(tmp, "actual_EBI_A_2025")
    return {
        "winsor": winsor_label,
        "benchmark": benchmark,
        "n_train_valid_target": info["n_train"],
        "n_soe_predicted": len(tmp),
        "actual_mean": tmp["actual_EBI_A_2025"].mean(),
        "predicted_mean": tmp["predicted_EBI_A_2025"].mean(),
        "mean_gap": tmp["return_gap_pred_minus_actual"].mean(),
        "median_gap": tmp["return_gap_pred_minus_actual"].median(),
        "positive_gap_share": (tmp["return_gap_pred_minus_actual"] > 0).mean(),
        "actual_asset_weighted": aw_act,
        "predicted_asset_weighted": aw_pred,
        "asset_weighted_gap": aw_pred - aw_act,
        "net_implied_gap_amount": tmp["implied_gap_amount"].sum(),
        "positive_only_gap_amount": tmp["positive_gap_amount"].sum(),
        "target_lower": info["target_lower"],
        "target_upper": info["target_upper"],
    }


def group_stats(pred_long, group_col):
    rows = []
    for (benchmark, grp), g in pred_long.groupby(["benchmark", group_col], dropna=False):
        rows.append({
            "benchmark": benchmark,
            group_col: grp,
            "n_firms": len(g),
            "actual_mean": g["actual_EBI_A_2025"].mean(),
            "predicted_mean": g["predicted_EBI_A_2025"].mean(),
            "mean_gap": g["return_gap_pred_minus_actual"].mean(),
            "median_gap": g["return_gap_pred_minus_actual"].median(),
            "positive_gap_share": (g["return_gap_pred_minus_actual"] > 0).mean(),
            "actual_asset_weighted": weighted_avg(g, "actual_EBI_A_2025"),
            "predicted_asset_weighted": weighted_avg(g, "predicted_EBI_A_2025"),
            "asset_weighted_gap": weighted_avg(g, "predicted_EBI_A_2025") - weighted_avg(g, "actual_EBI_A_2025"),
            "net_implied_gap_amount": g["implied_gap_amount"].sum(),
            "positive_only_gap_amount": g["positive_gap_amount"].sum(),
            "total_assets": g["资产总计_2025"].sum(),
        })
    return pd.DataFrame(rows)


def support_diagnostics(train_df, pred_df, benchmark):
    rows = []
    for c in NUM_COLS:
        train_s = train_df[c].dropna().astype(float)
        pred_s = pred_df[c].astype(float)
        p05, p95 = train_s.quantile(0.05), train_s.quantile(0.95)
        mn, mx = train_s.min(), train_s.max()
        rows.append({
            "benchmark": benchmark,
            "feature": c,
            "train_min": mn,
            "train_p05": p05,
            "train_p95": p95,
            "train_max": mx,
            "soe_outside_minmax_count": int(((pred_s < mn) | (pred_s > mx)).sum()),
            "soe_outside_p05_p95_count": int(((pred_s < p05) | (pred_s > p95)).sum()),
            "soe_outside_p05_p95_share": float(((pred_s < p05) | (pred_s > p95)).mean()),
        })
    soe_ind = set(pred_df["二级行业"].dropna())
    train_ind = set(train_df["二级行业"].dropna())
    missing = sorted(soe_ind - train_ind)
    rows.append({
        "benchmark": benchmark,
        "feature": "二级行业 coverage",
        "soe_outside_minmax_count": int(pred_df["二级行业"].isin(missing).sum()),
        "soe_outside_p05_p95_count": int(pred_df["二级行业"].isin(missing).sum()),
        "soe_outside_p05_p95_share": float(pred_df["二级行业"].isin(missing).mean()),
        "missing_industries_in_benchmark": ", ".join(missing),
        "n_missing_industries": len(missing),
        "n_benchmark_industries": len(train_ind),
        "n_soe_industries": len(soe_ind),
    })
    return pd.DataFrame(rows)


def run_benchmarks(csi_soe, h_ns, csi_ns, winsor=(0.05, 0.95)):
    h_ns = h_ns.copy()
    csi_ns = csi_ns.copy()
    csi_soe = csi_soe.copy()
    h_ns["BenchmarkMarket"] = "H_nonSOE"
    csi_ns["BenchmarkMarket"] = "CSI_nonSOE"
    csi_soe["BenchmarkMarket"] = "SOE_target"

    benchmarks = {
        "H_nonSOE_only": h_ns,
        "CSI_nonSOE_only": csi_ns,
        "Pooled_H_plus_CSI_nonSOE": pd.concat([h_ns, csi_ns], ignore_index=True),
    }

    summaries = []
    cv_rows = []
    pred_long = []

    label = f"{int(winsor[0]*100)}-{int(winsor[1]*100)}"

    for bname, bdf in benchmarks.items():
        cat_cols = BASE_CAT_COLS + (["BenchmarkMarket"] if bname.startswith("Pooled") else [])
        pred_df = csi_soe.copy()
        if "BenchmarkMarket" in cat_cols:
            # Pooled 模型下，将 SOE 的反事实市场环境设置为 CSI non-SOE。
            pred_df["BenchmarkMarket"] = "CSI_nonSOE"

        cv = cv_metrics(bdf, cat_cols, winsor=winsor)
        cv.update({"winsor": label, "benchmark": bname, "model": "Ridge"})
        cv_rows.append(cv)

        _, preds, info = fit_predict(bdf, pred_df, cat_cols, winsor=winsor)
        tmp = pred_df.copy()
        tmp["benchmark"] = bname
        tmp["predicted_EBI_A_2025"] = preds
        tmp["actual_EBI_A_2025"] = tmp[TARGET_COL]
        tmp["return_gap_pred_minus_actual"] = tmp["predicted_EBI_A_2025"] - tmp["actual_EBI_A_2025"]
        tmp["implied_gap_amount"] = tmp["return_gap_pred_minus_actual"] * tmp["资产总计_2025"]
        tmp["positive_gap_amount"] = np.maximum(tmp["return_gap_pred_minus_actual"], 0) * tmp["资产总计_2025"]
        pred_long.append(tmp)
        summaries.append(summarize_predictions(tmp, bname, info, label))

    return pd.DataFrame(summaries), pd.DataFrame(cv_rows), pd.concat(pred_long, ignore_index=True)


def main():
    csi_soe = normalize_old(pd.read_csv(CSI_SOE_PATH), "CSI")
    h_ns = normalize_old(pd.read_csv(H_NS_PATH), "H")
    csi_ns = normalize_csi_ns(pd.read_excel(CSI_NS_PATH, sheet_name="计算结果_数值版"))

    main_summary, main_cv, main_pred = run_benchmarks(csi_soe, h_ns, csi_ns, winsor=(0.05, 0.95))
    s25, cv25, _ = run_benchmarks(csi_soe, h_ns, csi_ns, winsor=(0.025, 0.975))
    s1, cv1, _ = run_benchmarks(csi_soe, h_ns, csi_ns, winsor=(0.01, 0.99))

    main_summary.to_csv(OUT / "summary_main_5_95.csv", index=False, encoding="utf-8-sig")
    pd.concat([main_summary, s25, s1], ignore_index=True).to_csv(OUT / "summary_robustness.csv", index=False, encoding="utf-8-sig")
    pd.concat([main_cv, cv25, cv1], ignore_index=True).to_csv(OUT / "cv_metrics_robustness.csv", index=False, encoding="utf-8-sig")
    main_pred.to_csv(OUT / "predictions_main_long.csv", index=False, encoding="utf-8-sig")

    pred_base_cols = ["证券代码", "证券简称", "一级行业", "二级行业", "企业所有制性质", "资产总计_2025", "actual_EBI_A_2025"]
    pred_pivot = main_pred[
        pred_base_cols + ["benchmark", "predicted_EBI_A_2025", "return_gap_pred_minus_actual", "implied_gap_amount", "positive_gap_amount"]
    ].pivot_table(
        index=pred_base_cols,
        columns="benchmark",
        values=["predicted_EBI_A_2025", "return_gap_pred_minus_actual", "implied_gap_amount", "positive_gap_amount"],
        aggfunc="first",
    ).reset_index()
    pred_pivot.columns = [c[0] if c[1] == "" else f"{c[0]}__{c[1]}" for c in pred_pivot.columns.to_flat_index()]
    pred_pivot.to_csv(OUT / "predictions_main_compact.csv", index=False, encoding="utf-8-sig")

    group_stats(main_pred, "企业所有制性质").to_csv(OUT / "group_ownership.csv", index=False, encoding="utf-8-sig")
    group_stats(main_pred, "二级行业").to_csv(OUT / "group_industry.csv", index=False, encoding="utf-8-sig")

    support = pd.concat([
        support_diagnostics(h_ns, csi_soe, "H_nonSOE_only"),
        support_diagnostics(csi_ns, csi_soe, "CSI_nonSOE_only"),
        support_diagnostics(pd.concat([h_ns, csi_ns], ignore_index=True), csi_soe, "Pooled_H_plus_CSI_nonSOE"),
    ], ignore_index=True)
    support.to_csv(OUT / "support_diagnostics.csv", index=False, encoding="utf-8-sig")

    print(main_summary)


if __name__ == "__main__":
    main()
