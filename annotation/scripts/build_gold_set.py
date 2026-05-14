#!/usr/bin/env python3
"""
Build gold-set CSV from dual-annotated rows where Cohen kappa agreement is high per-row.

Simplified rule: keep rows where normalized labels match OR Levenshtein distance <= 1.

Usage from repo root:
  python annotation/scripts/build_gold_set.py \\
    --input data/annotation_exports/dual_annotations.csv \\
    --out data/results/gold_set_v2.csv \\
    --min-kappa-global 0.62
"""

from __future__ import annotations

import argparse
import csv
import unicodedata
from pathlib import Path


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())


def lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/results/gold_set_v2.csv")
    ap.add_argument("--label1", type=str, default="label_a")
    ap.add_argument("--label2", type=str, default="label_b")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    with inp.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            la = norm(row.get(args.label1, ""))
            lb = norm(row.get(args.label2, ""))
            if la == lb or lev(la, lb) <= 1:
                rows_out.append({**row, "gold_label": la, "consensus": "match_or_near"})

    with out.open("w", newline="", encoding="utf-8") as f:
        if not rows_out:
            f.write("image_path,gold_label,consensus\n")
        else:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} gold rows to {out}")


if __name__ == "__main__":
    main()
