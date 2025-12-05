"""SHAP explainability helpers."""
from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import shap


def _ensure_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def run_shap_for_model(
    fitted_model,
    X_train,
    X_test,
    feature_names: Sequence[str],
    model_name: str,
    max_background: int = 2000,
    check_additivity: bool = True,
):
    """
    Generate SHAP summary and bar plots for a trained classifier.
    """
    plots_dir = os.path.join("outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    X_train = _ensure_dense(X_train)
    X_test = _ensure_dense(X_test)

    rng = np.random.default_rng(0)
    background = X_train
    if background.shape[0] > max_background:
        idx = rng.choice(background.shape[0], size=max_background, replace=False)
        background = background[idx]

    explainer = None
    for constructor in (shap.TreeExplainer, shap.LinearExplainer, shap.Explainer):
        try:
            explainer = constructor(fitted_model, background)
            break
        except Exception:
            continue
    if explainer is None:
        # Fallback: use KernelExplainer for arbitrary models (e.g. MLP, SVM)
        predict_fn = getattr(fitted_model, "predict_proba", None) or getattr(fitted_model, "predict", None)
        if predict_fn is None:
            raise RuntimeError("Failed to build a SHAP explainer for the provided model.")
        try:
            explainer = shap.KernelExplainer(predict_fn, background)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("Failed to build a SHAP explainer for the provided model.") from exc

    try:
        shap_values = explainer(X_test, check_additivity=check_additivity)
    except TypeError:
        shap_values = explainer(X_test)

    # Support both modern shap.Explanation and older ndarray / list outputs
    values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]

    summary_path = os.path.join(plots_dir, f"shap_summary_{model_name}.png")
    shap.summary_plot(values, X_test, feature_names=feature_names, show=False)
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    bar_path = os.path.join(plots_dir, f"shap_bar_{model_name}.png")
    shap.summary_plot(values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()


__all__ = ["run_shap_for_model"]
