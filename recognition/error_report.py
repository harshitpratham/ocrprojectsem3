"""Build human-readable recognition reports (model vs ground truth) from a predictions DataFrame."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _md_cell(text) -> str:
    """Escape pipe/newline so markdown table cells render safely."""
    if text is None:
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    return s.replace("\r", " ").replace("\n", " ").replace("|", "\\|") or "*(empty)*"


def write_recognition_error_reports(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Write recognition_report.md (every crop: recognized vs expected, Correct/Wrong)
    and mismatches_only.csv (wrong rows only).
    """
    if "ground_truth" not in df.columns or "predicted_text" not in df.columns:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    d["predicted_text"] = d["predicted_text"].fillna("").astype(str)
    d["ground_truth"] = d["ground_truth"].fillna("").astype(str)
    d["_match"] = d["predicted_text"].str.strip() == d["ground_truth"].str.strip()
    n_ok = int(d["_match"].sum())
    n_bad = int((~d["_match"]).sum())

    wrong = d.loc[~d["_match"]].drop(columns=["_match"], errors="ignore")
    wrong_path = output_dir / "mismatches_only.csv"
    wrong.to_csv(wrong_path, index=False, encoding="utf-8-sig")

    lines: list[str] = [
        "# Recognition report: model output vs ground truth",
        "",
        "Per-crop **Model recognized** (TrOCR) compared to **Ground truth (expected)**.",
        "Exact match = same string after stripping whitespace (no Unicode normalization).",
        "",
        f"- **Total crops:** {len(d)}",
        f"- **Correct (exact match):** {n_ok}",
        f"- **Wrong or misaligned:** {n_bad}",
        "",
        "CSV with errors only: `mismatches_only.csv` (same columns as `word_predictions.csv`).",
        "",
    ]

    for doc_key, group in d.sort_values(["doc_key", "crop_index"]).groupby("doc_key", sort=False):
        lines.append(f"## `{doc_key}`")
        lines.append("")
        lines.append("| Crop | Model recognized | Ground truth (expected) | Result |")
        lines.append("|:-----|:-----------------|:------------------------|:-------|")
        for _, row in group.iterrows():
            crop = _md_cell(row.get("crop_relpath", ""))
            pred = _md_cell(row["predicted_text"])
            ref = _md_cell(row["ground_truth"])
            if row["_match"]:
                result = "Correct"
            else:
                if not row["ground_truth"].strip() and row["predicted_text"].strip():
                    result = "**Wrong** (no GT line for this crop index)"
                elif not row["predicted_text"].strip() and row["ground_truth"].strip():
                    result = "**Wrong** (empty prediction)"
                else:
                    result = "**Wrong**"
            lines.append(f"| `{crop}` | {pred} | {ref} | {result} |")
        lines.append("")

    report_path = output_dir / "recognition_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s and %s (%d mismatches)", report_path, wrong_path, n_bad)
