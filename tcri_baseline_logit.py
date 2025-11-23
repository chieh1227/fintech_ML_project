#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCRI Baseline (Benchmark) — Logistic Regression pipeline entry point.

This script now orchestrates three stages:
1. Data ingestion / label creation (src.data_prep)
2. Model training and calibration (src.modeling.logistic_pipeline)
3. Evaluation + artifact export (src.reporting)
"""
from __future__ import annotations

import argparse
import os

from src.data_prep import create_label_next_period, detect_feature_columns, load_and_prepare, time_split
from src.metrics import best_f1_threshold, threshold_at_precision
from src.modeling.logistic_pipeline import extract_feature_weights, predict_with_calibration, select_best_logistic_model
from src.reporting import aggregate_slice_metrics, evaluate_predictions, prepare_prediction_frame, save_artifacts


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


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    industry_col = (args.industry_col or "").strip() or None
    df = load_and_prepare(args.csv, args.id_col, args.date_col, industry_col, args.tcri_col)
    df = create_label_next_period(df, args.id_col, args.tcri_col, args.tau, args.horizon)
    train, valid, test = time_split(
        df,
        args.date_col,
        args.train_start,
        args.train_end,
        args.valid_start,
        args.valid_end,
        args.test_start,
        args.test_end,
    )
    num_cols, cat_cols = detect_feature_columns(train, args.id_col, args.date_col, industry_col, args.tcri_col)

    Cs = [0.1, 1.0, 10.0]
    best_model, best_C, best_score = select_best_logistic_model(
        train, valid, num_cols, cat_cols, Cs=Cs, random_state=args.random_state
    )
    valid_features = valid[num_cols + cat_cols]
    test_features = test[num_cols + cat_cols]
    p_valid_best = best_model.predict_proba(valid_features)[:, 1]
    t_f1 = best_f1_threshold(valid["y"].values, p_valid_best)
    t_rp = threshold_at_precision(valid["y"].values, p_valid_best, args.target_precision)
    print(f"[INFO] Selected C={best_C} by PR-AUC on validation ({best_score:.4f})")
    print(f"[INFO] Thresholds on validation: t_f1={t_f1:.4f}, t_at_P>={args.target_precision:.2f} = {t_rp:.4f}")

    calibrated_probs = predict_with_calibration(best_model, valid_features, valid["y"], test_features)
    metrics = evaluate_predictions(test["y"].values, calibrated_probs, threshold=t_f1, target_precision=args.target_precision)
    metrics["thresholds"] = {"t_f1": t_f1, "t_at_precision": t_rp, "precision_target": args.target_precision}
    metrics["valid_pr_auc_for_bestC"] = float(best_score)
    metrics["chosen_C"] = float(best_C)

    by_quarter = aggregate_slice_metrics(
        test, calibrated_probs, "quarter", threshold=t_f1, target_precision=args.target_precision
    )
    by_industry = aggregate_slice_metrics(
        test, calibrated_probs, industry_col, threshold=t_f1, target_precision=args.target_precision
    )
    preds = prepare_prediction_frame(test, args.id_col, args.date_col, calibrated_probs)
    feature_weights = extract_feature_weights(best_model)
    save_artifacts(preds, metrics, feature_weights, by_quarter, by_industry, args.outdir)

    print("\n=== Test Overall Metrics (threshold = t_f1 on valid) ===")
    for name in ["raw", "platt", "isotonic"]:
        m = metrics[name]
        key = f"recall_at_p>={args.target_precision:.2f}"
        print(
            f"[{name}] PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  "
            f"Recall@P>={args.target_precision:.2f}={m[key]:.4f}  Brier={m['brier']:.4f}  ECE={m['ece']:.4f}"
        )
    print("\nTop 10 feature weights by |coefficient|:")
    print(feature_weights.head(10).to_string(index=False))
    print(f"\nArtifacts saved under: {os.path.abspath(args.outdir)}")
    files = ["predictions_test.csv", "metrics_summary.json", "metrics_by_quarter.csv", "feature_weights.csv"]
    if not by_industry.empty:
        files.append("metrics_by_industry.csv")
    print("Files:", ", ".join(files))


if __name__ == "__main__":
    main()

