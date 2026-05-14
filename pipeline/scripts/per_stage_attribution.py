#!/usr/bin/env python3
"""
Copy or summarize per-stage CER attribution from `data/results/v2/per_stage_attribution.json`.

Usage from repo root:
  PYTHONPATH=pipeline python pipeline/scripts/per_stage_attribution.py --print
  PYTHONPATH=pipeline python pipeline/scripts/per_stage_attribution.py --out report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_JSON = REPO / "data" / "results" / "v2" / "per_stage_attribution.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(DEFAULT_JSON))
    ap.add_argument("--out", type=str, default=None, help="Copy JSON to this path")
    ap.add_argument("--show", action="store_true", help="Print JSON to stdout")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        raise SystemExit(
            f"Missing {src}. Run: python recognition/eval_v2.py --write-release-artifacts"
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out)
    if args.show or not args.out:
        data = json.loads(src.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
