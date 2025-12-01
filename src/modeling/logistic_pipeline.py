"""Logistic regression training pipeline helpers."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import average_precision_score


def _build_preprocessor(num_cols, cat_cols):
    numerical = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformers = [("num", numerical, num_cols)]
    if len(cat_cols) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


def select_best_logistic_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    num_cols,
    cat_cols,
    Cs: Iterable[float],
    random_state: int,
) -> Tuple[Pipeline, float, float]:
    pre = _build_preprocessor(num_cols, cat_cols)
    best_model = None
    best_score = -np.inf
    best_C = None
    X_train = train[num_cols + cat_cols]
    y_train = train["y"]
    X_valid = valid[num_cols + cat_cols]
    y_valid = valid["y"]
    for C in Cs:
        logit = LogisticRegression(
            penalty="l2", C=C, class_weight="balanced", max_iter=2000, random_state=random_state, solver="lbfgs"
        )
        pipe = Pipeline([("pre", pre), ("clf", logit)])
        pipe.fit(X_train, y_train)
        p_valid = pipe.predict_proba(X_valid)[:, 1]
        pr_auc = average_precision_score(y_valid, p_valid)
        if pr_auc > best_score:
            best_model = pipe
            best_score = pr_auc
            best_C = C
    if best_model is None:
        raise RuntimeError("Failed to train logistic regression models")
    return best_model, float(best_C), float(best_score)


def predict_with_calibration(
    model: Pipeline,
    valid_features: pd.DataFrame,
    valid_target: pd.Series,
    test_features: pd.DataFrame,
) -> Dict[str, pd.Series]:
    raw = pd.Series(model.predict_proba(test_features)[:, 1], index=test_features.index, name="raw")
    platt = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    platt.fit(valid_features, valid_target)
    platt_probs = pd.Series(platt.predict_proba(test_features)[:, 1], index=test_features.index, name="platt")
    return {"raw": raw, "platt": platt_probs}


def extract_feature_weights(model: Pipeline) -> pd.DataFrame:
    feature_names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef, "abs_coefficient": np.abs(coef)})
    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


__all__ = ["select_best_logistic_model", "predict_with_calibration", "extract_feature_weights"]
