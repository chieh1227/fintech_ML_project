#!/usr/bin/env python3
"""Utility script to merge TCRI labels with processed financial ratios."""
from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge tcri.csv with ratios csv for downstream modeling.")
    parser.add_argument("--tcri", type=str, required=True, help="Path to tcri.csv (contains coid/mdate/tcri/scr...).")
    parser.add_argument(
        "--ratios", type=str, required=True, help="Path to ratios csv (contains financial features + GICS info)."
    )
    parser.add_argument("--tcri-id-col", type=str, default="coid", help="ID column in tcri.csv")
    parser.add_argument("--tcri-date-col", type=str, default="mdate", help="Date column in tcri.csv")
    parser.add_argument("--ratios-id-col", type=str, default="Merge_Code", help="ID column in ratios file")
    parser.add_argument("--ratios-date-col", type=str, default="mdate", help="Date column in ratios file")
    parser.add_argument(
        "--ratios-date-format",
        type=str,
        default=None,
        help="Optional strftime format for ratios date parsing (e.g., %%Y/%%m).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/merged_with_gics.csv",
        help="Where to save the merged output CSV.",
    )
    parser.add_argument("--merge-how", type=str, choices=["inner", "left", "right"], default="inner")
    parser.add_argument(
        "--dedup-ratios",
        action="store_true",
        help="If set, drop duplicate rows in ratios file based on (id,date) before merging.",
    )
    return parser.parse_args()


def _parse_dates(series: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    dates = pd.to_datetime(series, format=fmt, errors="coerce")
    if hasattr(dates.dt, "tz_localize"):
        try:
            dates = dates.dt.tz_localize(None)
        except TypeError:
            dates = dates.dt.tz_convert(None)
    return dates


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tcri = pd.read_csv(args.tcri)
    ratios = pd.read_csv(args.ratios)

    tcri[args.tcri_date_col] = _parse_dates(tcri[args.tcri_date_col])
    ratios[args.ratios_date_col] = _parse_dates(ratios[args.ratios_date_col], fmt=args.ratios_date_format)

    ratios = ratios.rename(columns={args.ratios_id_col: args.tcri_id_col})

    if args.dedup_ratios:
        before = len(ratios)
        ratios = (
            ratios.sort_values([args.tcri_id_col, args.ratios_date_col])
            .drop_duplicates(subset=[args.tcri_id_col, args.ratios_date_col], keep="last")
        )
        dropped = before - len(ratios)
        if dropped > 0:
            print(f"[INFO] Dropped {dropped} duplicate rows from ratios data based on (id,date)")

    merged = pd.merge(
        tcri,
        ratios,
        on=[args.tcri_id_col, args.tcri_date_col],
        how=args.merge_how,
        suffixes=("", "_ratio"),
        validate="one_to_one",
    )
    merged = merged.sort_values([args.tcri_id_col, args.tcri_date_col]).reset_index(drop=True)
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[INFO] Merged shape: {merged.shape}. Saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
