"""Metric utilities shared across models."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        w = mask.mean()
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
    if len(valid) == 0:
        return 0.5
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


__all__ = ["best_f1_threshold", "threshold_at_precision", "compute_metrics", "expected_calibration_error"]

