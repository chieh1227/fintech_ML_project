"""Aggregate metrics and persist experiment artifacts."""
from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd

from .metrics import compute_metrics


def evaluate_predictions(y_true, probabilities: Dict[str, pd.Series], threshold: float, target_precision: float):
    results = {}
    for name, probs in probabilities.items():
        results[name] = compute_metrics(y_true, probs.values, threshold=threshold, target_precision=target_precision)
    return results


def aggregate_slice_metrics(test_df: pd.DataFrame, probabilities: Dict[str, pd.Series], column: str, threshold: float, target_precision: float):
    if not column or column not in test_df.columns:
        return pd.DataFrame()
    rows = []
    for g, sub in test_df.groupby(column):
        if sub["y"].nunique() < 2:
            continue
        row = {"slice": f"{column}={g}", "n": len(sub), "pos": int(sub["y"].sum())}
        for name, probs in probabilities.items():
            ps = probs.loc[sub.index]
            metrics = compute_metrics(sub["y"].values, ps.values, threshold=threshold, target_precision=target_precision)
            for k, v in metrics.items():
                row[f"{name}.{k}"] = v
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["slice", "n", "pos"])
    return pd.DataFrame(rows).sort_values("slice")


def prepare_prediction_frame(test_df: pd.DataFrame, id_col: str, date_col: str, probabilities: Dict[str, pd.Series]):
    preds = test_df[[id_col, date_col, "y"]].copy()
    preds.rename(columns={"y": "y_true"}, inplace=True)
    for name, probs in probabilities.items():
        preds[f"p_{name}"] = probs.values
    return preds


def save_artifacts(preds: pd.DataFrame, metrics: Dict, feature_weights: pd.DataFrame, by_quarter: pd.DataFrame, by_industry: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    preds.to_csv(os.path.join(outdir, "predictions_test.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(outdir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    feature_weights.to_csv(os.path.join(outdir, "feature_weights.csv"), index=False, encoding="utf-8-sig")
    by_quarter.to_csv(os.path.join(outdir, "metrics_by_quarter.csv"), index=False, encoding="utf-8-sig")
    if not by_industry.empty:
        by_industry.to_csv(os.path.join(outdir, "metrics_by_industry.csv"), index=False, encoding="utf-8-sig")


__all__ = [
    "evaluate_predictions",
    "aggregate_slice_metrics",
    "prepare_prediction_frame",
    "save_artifacts",
]

