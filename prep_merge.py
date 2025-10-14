#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge TCRI ratings with semiannual financial features into a single modeling table.

Notebook usage example::

    python prep_merge.py --tcri tcri.csv --fin fin_semi.csv --anchor 12-01 --out outputs/merged.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments so the script can be reused in notebooks or batch jobs."""
    parser = argparse.ArgumentParser(description="Prepare merged dataset for TCRI baseline model.")
    parser.add_argument("--tcri", type=Path, required=True, help="Path to TCRI history CSV file.")
    parser.add_argument("--fin", type=Path, required=True, help="Path to financial semiannual CSV/Excel file.")
    parser.add_argument(
        "--fin-sheet",
        type=str,
        default=None,
        help="Optional sheet name when the financial file is an Excel workbook. Ignored for CSV input.",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="12-01",
        help="Month-day string appended to financial year/month (e.g. 12-01) to build a full date.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Where to write the merged CSV.")
    return parser.parse_args()


def load_tcri(path: Path) -> pd.DataFrame:
    """Load raw TCRI ratings and normalise date handling."""
    df = pd.read_csv(path)
    # TCRI mdate 帶時區；轉換為 naive datetime 方便後續 join。
    df["mdate"] = pd.to_datetime(df["mdate"], utc=True, errors="coerce").dt.tz_convert(None)
    # 針對同公司同日期若重複，以較新的一筆為主。
    df = df.sort_values(["coid", "mdate"]).drop_duplicates(subset=["coid", "mdate"], keep="last")
    return df


def _read_financials(path: Path, sheet: Optional[str]) -> pd.DataFrame:
    """Read financial input, supporting both CSV and Excel."""
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def load_financials(path: Path, sheet: Optional[str], anchor: str) -> pd.DataFrame:
    """Load financial semiannual data and add helper columns."""
    df = _read_financials(path, sheet)

    # 解析公司代碼。原始資料可能放在 company 欄位的字串裡。
    if "company" in df.columns:
        df["coid"] = df["company"].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
    elif "coid" in df.columns:
        df["coid"] = df["coid"].astype("Int64")
    else:
        raise ValueError("Financial file must contain a 'company' or 'coid' column to identify firms.")

    if df["coid"].isna().any():
        raise ValueError("Some financial rows are missing a valid company code.")

    if "mdate" not in df.columns:
        raise ValueError("Financial file must contain an 'mdate' column.")

    def _parse_fin_date(val: object) -> pd.Timestamp:
        """
        把財報日期字串轉為 pandas Timestamp。

        - 若原始格式為 YYYY/MM，補齊日為 01。
        - 若只有年份，使用 anchor（預設 12-01）補齊月份與日期。
        """
        text = str(val).strip()
        if not text:
            return pd.NaT
        if "/" in text:
            year, month = text.split("/", 1)
            return pd.to_datetime(f"{year}-{month}-01", format="%Y-%m-%d", errors="coerce")
        return pd.to_datetime(f"{text}-{anchor}", format="%Y-%m-%d", errors="coerce")

    df["mdate"] = df["mdate"].apply(_parse_fin_date)
    if df["mdate"].isna().any():
        raise ValueError("Unable to parse some financial 'mdate' values; check the input format.")

    # 留下產業資訊，若沒有 stock_prefix 則標示 unknown。
    if "stock_prefix" in df.columns:
        df["industry"] = df["stock_prefix"].astype(str)
    else:
        df["industry"] = "unknown"

    # 只保留數值特徵，coid 自身不視為特徵。
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "coid"]
    keep_cols = ["coid", "mdate", "industry"] + numeric_cols
    return df[keep_cols].copy()


def main() -> None:
    args = parse_args()

    tcri_df = load_tcri(args.tcri)
    fin_df = load_financials(args.fin, args.fin_sheet, args.anchor)

    # 以公司與日期為 key 內連結，保證模型資料兩邊都有觀測值。
    merged = pd.merge(tcri_df, fin_df, on=["coid", "mdate"], how="inner", validate="many_to_one")
    merged = merged.sort_values(["coid", "mdate"]).reset_index(drop=True)

    out_path = args.out
    if out_path.parent and out_path.parent != Path("."):
        os.makedirs(out_path.parent, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Wrote merged dataset with {len(merged):,} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
