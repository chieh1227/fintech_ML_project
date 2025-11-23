"""Data ingestion and label creation utilities."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _map_tcri_value(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().upper()
    if s == "D":
        return 10.0
    try:
        return float(int(s))
    except Exception:
        return np.nan


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


def time_split(
    df: pd.DataFrame,
    date_col: str,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
):
    t0, t1 = pd.to_datetime(train_start), pd.to_datetime(train_end)
    v0, v1 = pd.to_datetime(valid_start), pd.to_datetime(valid_end)
    s0, s1 = pd.to_datetime(test_start), pd.to_datetime(test_end)
    train = df[(df[date_col] >= t0) & (df[date_col] <= t1)].copy()
    valid = df[(df[date_col] >= v0) & (df[date_col] <= v1)].copy()
    test = df[(df[date_col] >= s0) & (df[date_col] <= s1)].copy()
    return train, valid, test


def detect_feature_columns(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    tcri_col: str,
    categorical_cols: Optional[List[str]] = None,
):
    exclude = {id_col, date_col, tcri_col, "tcri_future", "y", "year", "quarter", "scr"}
    categorical_cols = categorical_cols or []
    for col in categorical_cols:
        exclude.add(col)
    present = set(df.columns)
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols: List[str] = []
    for col in categorical_cols:
        if col in present:
            cat_cols.append(col)
    return num_cols, cat_cols


__all__ = [
    "load_and_prepare",
    "create_label_next_period",
    "time_split",
    "detect_feature_columns",
]
