#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCRI Baseline (Benchmark) — Logistic Regression (no-industry friendly)
"""
from __future__ import annotations

import argparse, json, os
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, \
                            precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCRI Logistic Baseline (no-industry ok)")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--id-col", type=str, default="coid")
    p.add_argument("--date-col", type=str, default="mdate")
    p.add_argument("--tcri-col", type=str, default="tcri")
    p.add_argument("--industry-col", type=str, default="industry")
    p.add_argument("--tau", type=int, default=7)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--train-start", type=str, default="2014-01-01")
    p.add_argument("--train-end", type=str, default="2021-12-31")
    p.add_argument("--valid-start", type=str, default="2022-01-01")
    p.add_argument("--valid-end", type=str, default="2022-12-31")
    p.add_argument("--test-start", type=str, default="2023-01-01")
    p.add_argument("--test-end", type=str, default="2024-12-31")
    p.add_argument("--target-precision", type=float, default=0.5)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()

def _map_tcri_value(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).strip().upper()
    if s == "D": return 10.0
    try: return float(int(s))
    except: return np.nan

def load_and_prepare(csv_path: str, id_col: str, date_col: str, industry_col: Optional[str], tcri_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)
    df["year"] = df[date_col].dt.year
    df["quarter"] = df[date_col].dt.to_period("Q").astype(str)
    df[tcri_col] = df[tcri_col].apply(_map_tcri_value)
    return df

def create_label_next_period(df: pd.DataFrame, id_col: str, tcri_col: str, tau: int, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["tcri_future"] = df.groupby(id_col)[tcri_col].shift(-horizon)
    df["y"] = (df["tcri_future"] >= float(tau)).astype("Int64")
    df = df[~df["y"].isna()].copy()
    df["y"] = df["y"].astype(int)
    return df

def time_split(df: pd.DataFrame, date_col: str, train_start: str, train_end: str,
               valid_start: str, valid_end: str, test_start: str, test_end: str):
    t0, t1 = pd.to_datetime(train_start), pd.to_datetime(train_end)
    v0, v1 = pd.to_datetime(valid_start), pd.to_datetime(valid_end)
    s0, s1 = pd.to_datetime(test_start), pd.to_datetime(test_end)
    train = df[(df[date_col] >= t0) & (df[date_col] <= t1)].copy()
    valid = df[(df[date_col] >= v0) & (df[date_col] <= v1)].copy()
    test  = df[(df[date_col] >= s0) & (df[date_col] <= s1)].copy()
    return train, valid, test

def detect_feature_columns(df: pd.DataFrame, id_col: str, date_col: str, industry_col: Optional[str], tcri_col: str):
    exclude = {id_col, date_col, tcri_col, "tcri_future", "y", "year", "quarter", "scr"}
    if industry_col:
        exclude.add(industry_col)
    present = set(df.columns)
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols: List[str] = []
    if industry_col and industry_col in present:
        cat_cols.append(industry_col)
    return num_cols, cat_cols

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1); inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask): continue
        conf = y_prob[mask].mean(); acc = y_true[mask].mean(); w = mask.mean()
        ece += w * abs(acc - conf)
    return float(ece)

def best_f1_threshold(y_true, y_prob):
    pr, rc, th = precision_recall_curve(y_true, y_prob)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    idx = np.nanargmax(f1[:-1]) if len(th) > 0 else 0
    return float(th[idx]) if len(th) > 0 else 0.5

def threshold_at_precision(y_true, y_prob, target_precision=0.5):
    pr, rc, th = precision_recall_curve(y_true, y_prob)
    valid = np.where(pr[:-1] >= target_precision)[0]
    if len(valid) == 0: return 0.5
    return float(th[valid[0]])

def compute_metrics(y_true, y_prob, threshold, target_precision=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": float(expected_calibration_error(y_true, y_prob, n_bins=10)),
        "threshold": float(threshold),
    }
    t_rp = threshold_at_precision(y_true, y_prob, target_precision)
    y_rp = (y_prob >= t_rp).astype(int)
    metrics[f"recall_at_p>={target_precision:.2f}"] = float(recall_score(y_true, y_rp, zero_division=0))
    metrics[f"threshold_at_p>={target_precision:.2f}"] = float(t_rp)
    return metrics

def main():
    args = parse_args(); os.makedirs(args.outdir, exist_ok=True)
    industry_col = (args.industry_col or "").strip()
    industry_col = industry_col if industry_col else None

    df = load_and_prepare(args.csv, args.id_col, args.date_col, industry_col, args.tcri_col)
    df = create_label_next_period(df, args.id_col, args.tcri_col, args.tau, args.horizon)
    train, valid, test = time_split(df, args.date_col, args.train_start, args.train_end,
                                    args.valid_start, args.valid_end, args.test_start, args.test_end)
    num_cols, cat_cols = detect_feature_columns(train, args.id_col, args.date_col, industry_col, args.tcri_col)

    numerical = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())])
    transformers = [("num", numerical, num_cols)]
    if len(cat_cols) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")

    Cs = [0.1, 1.0, 10.0]; best_C=None; best_score=-np.inf; best_model=None
    for C in Cs:
        logit = LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                                   max_iter=2000, random_state=args.random_state, solver="lbfgs")
        pipe = Pipeline([("pre", pre), ("clf", logit)])
        pipe.fit(train[num_cols + cat_cols], train["y"])
        p_valid = pipe.predict_proba(valid[num_cols + cat_cols])[:,1]
        pr_auc = average_precision_score(valid["y"], p_valid)
        if pr_auc > best_score: best_score, best_C, best_model = pr_auc, C, pipe

    print(f"[INFO] Selected C={best_C} by PR-AUC on validation ({best_score:.4f})")
    p_valid_best = best_model.predict_proba(valid[num_cols + cat_cols])[:,1]
    t_f1 = best_f1_threshold(valid["y"].values, p_valid_best)
    t_rp = threshold_at_precision(valid["y"].values, p_valid_best, args.target_precision)
    print(f"[INFO] Thresholds on validation: t_f1={t_f1:.4f}, t_at_P>={args.target_precision:.2f} = {t_rp:.4f}")

    p_test_raw = pd.Series(best_model.predict_proba(test[num_cols + cat_cols])[:,1], index=test.index)
    platt = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
    platt.fit(valid[num_cols + cat_cols], valid["y"])
    p_test_platt = pd.Series(platt.predict_proba(test[num_cols + cat_cols])[:,1], index=test.index)
    isot = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    isot.fit(valid[num_cols + cat_cols], valid["y"])
    p_test_isot = pd.Series(isot.predict_proba(test[num_cols + cat_cols])[:,1], index=test.index)

    feature_names = best_model.named_steps["pre"].get_feature_names_out()
    coef = best_model.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    metrics = {
        "raw": compute_metrics(test["y"].values, p_test_raw.values, threshold=t_f1, target_precision=args.target_precision),
        "platt": compute_metrics(test["y"].values, p_test_platt.values, threshold=t_f1, target_precision=args.target_precision),
        "isotonic": compute_metrics(test["y"].values, p_test_isot.values, threshold=t_f1, target_precision=args.target_precision),
        "thresholds": {"t_f1": t_f1, "t_at_precision": t_rp, "precision_target": args.target_precision},
        "valid_pr_auc_for_bestC": float(best_score), "chosen_C": float(best_C),
    }

    def _agg_slice(df_key: str, probs_map, test_df):
        if df_key not in test_df.columns: return pd.DataFrame()
        rows = []
        # Use .loc for explicit indexing to avoid potential SettingWithCopyWarning and ensure index alignment
        for g, sub in test_df.groupby(df_key):
            if sub["y"].nunique() < 2: continue
            row = {"slice": f"{df_key}={g}", "n": len(sub), "pos": int(sub["y"].sum())}
            # Filter probabilities using the index of the subgroup
            for mname, probs in probs_map.items():
                ps = probs.loc[sub.index]
                m = compute_metrics(sub["y"].values, ps.values, threshold=t_f1, target_precision=args.target_precision)
                for k, v in m.items():
                    row[f"{mname}.{k}"] = v
            rows.append(row)
        if not rows:
            cols = ["slice", "n", "pos"]
            for mname, stats in metrics.items():
                if isinstance(stats, dict) and mname in probs_map:
                    cols.extend([f"{mname}.{k}" for k in stats.keys()])
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows).sort_values("slice")

    probs_map = {"raw": p_test_raw, "platt": p_test_platt, "isotonic": p_test_isot}
    by_quarter = _agg_slice("quarter", probs_map, test)
    by_industry = _agg_slice(industry_col, probs_map, test) if industry_col else pd.DataFrame()

    preds = test[[args.id_col, args.date_col, "y"]].copy()
    preds.rename(columns={"y":"y_true"}, inplace=True)
    preds["p_raw"] = p_test_raw.values; preds["p_platt"] = p_test_platt.values; preds["p_isotonic"] = p_test_isot.values
    os.makedirs(args.outdir, exist_ok=True)
    preds.to_csv(os.path.join(args.outdir, "predictions_test.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(args.outdir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    coef_df.to_csv(os.path.join(args.outdir, "feature_weights.csv"), index=False, encoding="utf-8-sig")
    by_quarter.to_csv(os.path.join(args.outdir, "metrics_by_quarter.csv"), index=False, encoding="utf-8-sig")
    if industry_col and not by_industry.empty:
        by_industry.to_csv(os.path.join(args.outdir, "metrics_by_industry.csv"), index=False, encoding="utf-8-sig")

    print("\n=== Test Overall Metrics (threshold = t_f1 on valid) ===")
    for name in ["raw", "platt", "isotonic"]:
        m = metrics[name]
        print(f"[{name}] PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  "
              f"Recall@P>={args.target_precision:.2f}={m[f'recall_at_p>={args.target_precision:.2f}']:.4f}  "
              f"Brier={m['brier']:.4f}  ECE={m['ece']:.4f}")
    print("\nTop 10 feature weights by |coefficient|:")
    print(coef_df.head(10).to_string(index=False))
    print(f"\nArtifacts saved under: {os.path.abspath(args.outdir)}")
    files = ['predictions_test.csv','metrics_summary.json','metrics_by_quarter.csv','feature_weights.csv']
    if industry_col and not by_industry.empty: files.append('metrics_by_industry.csv')
    print('Files:', ', '.join(files))

if __name__ == "__main__":
    main()
