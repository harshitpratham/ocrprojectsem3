#!/usr/bin/env python3
"""
TrOCR evaluation on annotation portal test set.

Reads data/trocr_annotation_test/labels.csv and evaluates model predictions
against the correct ground truth from the annotation portal.

Usage:
    python recognition/eval_annotation_test.py \
      --model-path models/trocr/final_model \
      --batch-size 8
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from jiwer import cer as compute_cer
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
_REC = Path(__file__).resolve().parent
if str(_REC) not in sys.path:
    sys.path.insert(0, str(_REC))

from trocr_hub import build_base_model, build_processor

TEST_DIR = REPO_ROOT / "data" / "trocr_annotation_test"
LABELS_CSV = TEST_DIR / "labels.csv"
DEFAULT_OUTPUT = TEST_DIR / "eval_results"


def load_model(model_path: Path, device: torch.device):
    processor = build_processor()
    try:
        model = VisionEncoderDecoderModel.from_pretrained(str(model_path), local_files_only=True)
    except Exception:
        model = build_base_model()
        import safetensors.torch
        state = safetensors.torch.load_file(str(model_path / "model.safetensors"))
        model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, processor


def infer_batch(model, processor, paths: list[Path], device: torch.device) -> list[str]:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    enc = processor(images=imgs, return_tensors="pt")
    pv = enc.pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pv)
    return [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]


def run(model_path: Path, output_dir: Path, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    df = pd.read_csv(LABELS_CSV)
    df["_abs"] = df["image_path"].apply(lambda p: str((TEST_DIR / p).resolve()))
    log.info("Loaded %d samples from %s", len(df), LABELS_CSV)

    missing = [p for p in df["_abs"] if not Path(p).exists()]
    if missing:
        log.warning("Missing %d images, filtering them out", len(missing))
        df = df[df["_abs"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        log.info("Remaining: %d samples", len(df))

    model, processor = load_model(model_path, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds: list[str] = []
    abs_paths = df["_abs"].tolist()
    for i in tqdm(range(0, len(abs_paths), batch_size), desc="TrOCR eval"):
        batch = [Path(p) for p in abs_paths[i : i + batch_size]]
        try:
            preds.extend(infer_batch(model, processor, batch, device))
        except Exception:
            log.exception("Batch %d failed, falling back per-image", i)
            for img in batch:
                try:
                    preds.append(infer_batch(model, processor, [img], device)[0])
                except Exception:
                    preds.append("")

    df["predicted"] = preds
    df["match"] = df.apply(
        lambda r: str(r["predicted"]).strip() == str(r["ground_truth"]).strip(), axis=1
    )

    results_csv = output_dir / "results.csv"
    df.drop(columns=["_abs"]).to_csv(results_csv, index=False, encoding="utf-8-sig")

    n = len(df)
    n_correct = int(df["match"].sum())
    n_wrong = n - n_correct
    word_acc = n_correct / n if n else 0

    gt_list = df["ground_truth"].fillna("").astype(str).tolist()
    pred_list = df["predicted"].fillna("").astype(str).tolist()
    cer_val = compute_cer(gt_list, pred_list)

    doc_rows = []
    for dk, g in df.groupby("doc_key"):
        gl = g["ground_truth"].fillna("").astype(str).tolist()
        pl = g["predicted"].fillna("").astype(str).tolist()
        dc = compute_cer(gl, pl)
        dex = int(g["match"].sum())
        doc_rows.append((dk, len(g), dc, dex, dex / len(g)))

    doc_df = pd.DataFrame(doc_rows, columns=["doc_key", "crops", "cer", "exact", "exact_pct"])
    doc_df = doc_df.sort_values("cer")

    summary = (
        f"{'='*60}\n"
        f"  TrOCR EVALUATION — Annotation Portal Test Set\n"
        f"{'='*60}\n"
        f"Model          : {model_path}\n"
        f"Labels CSV     : {LABELS_CSV}\n"
        f"Total crops    : {n}\n"
        f"Correct (exact): {n_correct} ({100*word_acc:.1f}%)\n"
        f"Wrong          : {n_wrong}\n"
        f"CER            : {cer_val:.6f} ({100*cer_val:.2f}%)\n"
        f"Word accuracy  : {100*word_acc:.2f}%\n"
        f"\nPer-document CER (sorted best → worst):\n"
    )
    summary += doc_df.to_string(index=False)
    print(summary)
    (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    log.info("Wrote %s", output_dir / "summary.txt")

    lines = [
        "# TrOCR Evaluation: Annotation Portal Test Set",
        "",
        f"- **Model**: `{model_path.name}`",
        f"- **Total crops**: {n}",
        f"- **Correct (exact match)**: {n_correct} ({100*word_acc:.1f}%)",
        f"- **Wrong**: {n_wrong}",
        f"- **CER**: {cer_val:.6f} ({100*cer_val:.2f}%)",
        f"- **Word accuracy**: {100*word_acc:.2f}%",
        "",
    ]

    for dk, group in df.sort_values(["doc_key", "crop_index"]).groupby("doc_key", sort=False):
        dc_info = doc_df.loc[doc_df["doc_key"] == dk]
        dc_cer = f"{float(dc_info['cer'].iloc[0]):.4f}" if len(dc_info) else "?"
        lines.append(f"## `{dk}` (CER {dc_cer})")
        lines.append("")
        lines.append("| Crop | Predicted | Ground Truth | Result |")
        lines.append("|:-----|:----------|:-------------|:-------|")
        for _, row in group.iterrows():
            pred = str(row["predicted"]).replace("|", "\\|")
            gt = str(row["ground_truth"]).replace("|", "\\|")
            tag = "Correct" if row["match"] else "**Wrong**"
            lines.append(f"| `{row['image_path']}` | {pred} | {gt} | {tag} |")
        lines.append("")

    (output_dir / "recognition_report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", output_dir / "recognition_report.md")

    wrong_df = df.loc[~df["match"]].drop(columns=["_abs"], errors="ignore")
    wrong_df.to_csv(output_dir / "mismatches_only.csv", index=False, encoding="utf-8-sig")
    log.info("Wrote %s (%d mismatches)", output_dir / "mismatches_only.csv", len(wrong_df))

    print(f"\nAll results saved to: {output_dir}/")


def main():
    p = argparse.ArgumentParser(description="TrOCR eval on annotation portal test set")
    p.add_argument(
        "--model-path", type=Path,
        default=REPO_ROOT / "models" / "trocr" / "final_model",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "8")))
    args = p.parse_args()
    run(args.model_path.resolve(), args.output_dir.resolve(), args.batch_size)


if __name__ == "__main__":
    main()
