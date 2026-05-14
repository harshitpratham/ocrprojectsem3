#!/usr/bin/env python3
"""Rebuild recognition_report.md and mismatches_only.csv from an existing word_predictions.csv (no model run)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from error_report import write_recognition_error_reports


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate recognition_report.md from word_predictions.csv")
    p.add_argument("--csv", type=Path, required=True, help="Path to word_predictions.csv")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs (default: same folder as CSV)",
    )
    args = p.parse_args()
    csv_path = args.csv.resolve()
    out_dir = (args.output_dir or csv_path.parent).resolve()
    df = pd.read_csv(csv_path)
    if "ground_truth" not in df.columns:
        raise SystemExit("CSV must contain a ground_truth column")
    write_recognition_error_reports(df, out_dir)
    print(f"Wrote recognition_report.md and mismatches_only.csv under {out_dir}")


if __name__ == "__main__":
    main()
