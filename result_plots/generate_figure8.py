"""Generate Figure 8 (ROC / PR / Reliability) for the report.

Reads each model's predictions_test.csv from outputs/<model>/ and produces
result_plots/figure8.png as well as figure8a/8b/8c separately.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
PLOTS_DIR = REPO / "result_plots"
PLOTS_DIR.mkdir(exist_ok=True)

MODELS = [
    ("Logistic (Baseline)",     "logistic_baseline",         "#1f77b4"),
    ("Logistic with no GICS",   "logistic_baseline_no_gics", "#ff7f0e"),
    ("Random Forest",           "rf_model",                  "#2ca02c"),
    ("XGBoost",                 "xgboost_model",             "#d62728"),
    ("SVM",                     "svm_model",                 "#9467bd"),
    ("Neural Network",          "nn_model",                  "#8c564b"),
    ("Gradient Boosting",       "gb_model",                  "#17becf"),
]


def expected_calibration_error(y_true, y_prob, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.any():
            prop = mask.mean()
            ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * prop
    return ece


def load_results():
    results = []
    for name, folder, color in MODELS:
        path = OUTPUTS / folder / "predictions_test.csv"
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        df = pd.read_csv(path)
        y_true = df["y_true"].to_numpy()
        y_prob = (df["p_platt"] if "p_platt" in df.columns else df["p_raw"]).to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

        results.append({
            "name": name,
            "color": color,
            "fpr": fpr, "tpr": tpr, "roc_auc": auc(fpr, tpr),
            "precision": precision, "recall": recall,
            "auprc": average_precision_score(y_true, y_prob),
            "frac": frac_pos, "mean": mean_pred,
            "brier": brier_score_loss(y_true, y_prob),
            "ece": expected_calibration_error(y_true, y_prob),
        })
    return results


def plot_roc(ax, results):
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    for r in results:
        ax.plot(r["fpr"], r["tpr"], lw=2, color=r["color"],
                label=f"{r['name']} (AUC={r['roc_auc']:.3f})")
    ax.set_title("Comparison: ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_pr(ax, results):
    for r in results:
        ax.plot(r["recall"], r["precision"], lw=2, color=r["color"],
                label=f"{r['name']} (AUPRC={r['auprc']:.3f})")
    ax.set_title("Comparison: Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_reliability(ax, results):
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for r in results:
        ax.plot(r["mean"], r["frac"], "s-", lw=2, color=r["color"],
                label=f"{r['name']} (ECE={r['ece']:.3f})")
    ax.set_title("Comparison: Reliability Curves")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    results = load_results()

    print(f"{'Model':<25} {'AUC':>6} {'AUPRC':>6} {'Brier':>6} {'ECE':>6}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<25} {r['roc_auc']:.4f} {r['auprc']:.4f} {r['brier']:.4f} {r['ece']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plot_roc(axes[0, 0], results)
    plot_pr(axes[0, 1], results)
    plot_reliability(axes[1, 0], results)
    axes[1, 1].axis("off")
    axes[0, 0].text(0.5, -0.13, "(a) ROC Curves", transform=axes[0, 0].transAxes,
                    ha="center", fontsize=11)
    axes[0, 1].text(0.5, -0.13, "(b) Precision-Recall Curves", transform=axes[0, 1].transAxes,
                    ha="center", fontsize=11)
    axes[1, 0].text(0.5, -0.13, "(c) Reliability Diagrams (Calibration)", transform=axes[1, 0].transAxes,
                    ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "figure8.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved: {PLOTS_DIR / 'figure8.png'}")

    for tag, plotter in [("8a_roc", plot_roc), ("8b_pr", plot_pr), ("8c_reliability", plot_reliability)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        plotter(ax, results)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"figure{tag}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {PLOTS_DIR / f'figure{tag}.png'}")


if __name__ == "__main__":
    main()
