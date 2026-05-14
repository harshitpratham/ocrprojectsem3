#!/usr/bin/env python3
"""
Run fine-tuned TrOCR on YOLO word crops (sorted_crops layout) with optional CER.

Adapted from pipeline/scripts/batch_predict_word.py: repo-relative paths, no S3,
CLI interface, optional alignment to per-document ground-truth .txt files.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import evaluate
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from error_report import write_recognition_error_reports
from transformers import VisionEncoderDecoderModel

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent

_REC = Path(__file__).resolve().parent
if str(_REC) not in sys.path:
    sys.path.insert(0, str(_REC))

from trocr_hub import ENCODER_HUB, build_base_model, build_processor

ENCODER_NAME = ENCODER_HUB
DECODER_NAME = "flax-community/roberta-hindi"

# Filename patterns for YOLO crops (same as batch_predict_word.py)
FNAME_RE = re.compile(
    r"(?P<orig>.+)__(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)__(?P<idx>\d+)\.jpg"
)


def parse_crop_filename(fname: str):
    m = FNAME_RE.match(fname)
    if m:
        return {
            "orig_image": m.group("orig"),
            "x1": int(m.group("x1")),
            "y1": int(m.group("y1")),
            "x2": int(m.group("x2")),
            "y2": int(m.group("y2")),
            "idx": int(m.group("idx")),
        }
    parts = fname.split("__")
    if len(parts) >= 3:
        orig = parts[0]
        coords = parts[1].split("_")
        if len(coords) >= 4:
            try:
                idx = int(Path(parts[2]).stem)
            except Exception:
                idx = 0
            return {
                "orig_image": orig,
                "x1": int(coords[0]),
                "y1": int(coords[1]),
                "x2": int(coords[2]),
                "y2": int(coords[3]),
                "idx": idx,
            }
    return None


def crop_sort_key(path: Path) -> tuple[str, int]:
    """Sort key: document folder name, then numeric crop index (000.jpg -> 0)."""
    doc = path.parent.name
    try:
        idx = int(path.stem)
    except ValueError:
        idx = 0
    return (doc, idx)


def discover_crops(crops_dir: Path) -> list[Path]:
    paths = sorted(crops_dir.rglob("*.jpg"), key=crop_sort_key)
    return paths


def load_ground_truth_map(ground_truth_dir: Path) -> dict[str, list[str]]:
    """Map document id -> list of reference strings (one per line)."""
    gt: dict[str, list[str]] = {}
    for f in sorted(ground_truth_dir.glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines()]
        gt[f.stem] = lines
    return gt


def load_model_and_processor(model_path: Path, device: torch.device):
    processor = build_processor()

    try:
        log.info("Loading VisionEncoderDecoderModel from %s", model_path)
        model = VisionEncoderDecoderModel.from_pretrained(
            str(model_path), local_files_only=True
        )
    except Exception as e:
        log.warning("from_pretrained failed (%s); building encoder-decoder and loading weights", e)
        model = build_base_model()
        safetensors_path = model_path / "model.safetensors"
        if safetensors_path.exists():
            import safetensors.torch

            state = safetensors.torch.load_file(str(safetensors_path))
            model.load_state_dict(state, strict=False)
            log.info("Loaded weights from %s", safetensors_path)
        else:
            raise FileNotFoundError(f"No weights at {safetensors_path}")

    model.to(device)
    model.eval()
    return model, processor


def load_pil(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def infer_batch(
    model, processor, image_paths: list[Path], device: torch.device
) -> list[str]:
    images = [load_pil(p) for p in image_paths]
    # TrOCRProcessor warns on `padding=`; default batching is sufficient here.
    encoding = processor(images=images, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [t.strip() for t in generated_texts]


def row_meta(path: Path, crops_dir: Path) -> dict:
    """Metadata for one crop: document key, indices, optional bbox fields."""
    crop_rel = str(path.relative_to(crops_dir))
    doc_key = path.parent.name
    try:
        crop_index = int(path.stem)
    except ValueError:
        crop_index = 0
    meta = parse_crop_filename(path.name)
    if meta is None:
        return {
            "doc_key": doc_key,
            "crop_index": crop_index,
            "crop_relpath": crop_rel,
            "original_image": doc_key,
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": 0,
            "idx": crop_index,
        }
    return {
        "doc_key": doc_key,
        "crop_index": crop_index,
        "crop_relpath": crop_rel,
        "original_image": meta["orig_image"],
        "x1": meta["x1"],
        "y1": meta["y1"],
        "x2": meta["x2"],
        "y2": meta["y2"],
        "idx": meta["idx"],
    }


def align_reference(doc_key: str, crop_index: int, gt_map: dict[str, list[str]] | None) -> str:
    if not gt_map or doc_key not in gt_map:
        return ""
    lines = gt_map[doc_key]
    if crop_index < len(lines):
        return lines[crop_index]
    return ""


def run(
    crops_dir: Path,
    output_dir: Path,
    model_path: Path,
    ground_truth_dir: Path | None,
    batch_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    crops_dir = crops_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_map = load_ground_truth_map(ground_truth_dir) if ground_truth_dir else None
    if ground_truth_dir:
        log.info("Loaded ground truth for %d documents from %s", len(gt_map), ground_truth_dir)

    crop_paths = discover_crops(crops_dir)
    log.info("Found %d crop images under %s", len(crop_paths), crops_dir)

    model, processor = load_model_and_processor(model_path, device)

    rows: list[dict] = []
    for i in tqdm(range(0, len(crop_paths), batch_size), desc="OCR batches"):
        batch_paths = crop_paths[i : i + batch_size]
        try:
            preds = infer_batch(model, processor, batch_paths, device)
        except Exception:
            log.exception("Batch failed at index %s; falling back per-image", i)
            preds = []
            for img_path in batch_paths:
                try:
                    preds.append(infer_batch(model, processor, [img_path], device)[0])
                except Exception as ex:
                    log.warning("Skipping %s: %s", img_path, ex)
                    preds.append("")

        for p, pred in zip(batch_paths, preds):
            m = row_meta(p, crops_dir)
            ref = align_reference(m["doc_key"], m["crop_index"], gt_map)
            rows.append(
                {
                    **m,
                    "predicted_text": pred,
                    "ground_truth": ref,
                }
            )

    df = pd.DataFrame(rows)

    csv_path = output_dir / "word_predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("Wrote %s", csv_path)

    if gt_map:
        write_recognition_error_reports(df, output_dir)

    # Transcripts: prefer reading order (y1, x1) when bbox metadata exists; else by crop_index
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    final_blocks: list[str] = []

    for doc_key, group in df.groupby("doc_key"):
        if (group["y1"] != 0).any() or (group["x1"] != 0).any():
            ordered = group.sort_values(by=["y1", "x1", "crop_index"])
        else:
            ordered = group.sort_values(by=["crop_index"])
        words = ordered["predicted_text"].astype(str).tolist()
        line_text = " ".join(words)
        safe_name = re.sub(r'[^\w\-., ()\u0900-\u097F]+', "_", doc_key)[:200]
        per_path = transcripts_dir / f"{safe_name}.txt"
        per_path.write_text(line_text + "\n", encoding="utf-8")
        final_blocks.append(f"=== {doc_key} ===\n{line_text}\n")

    combined_path = output_dir / "combined_transcription.txt"
    combined_path.write_text("\n".join(final_blocks), encoding="utf-8")
    log.info("Wrote %s and transcripts/", combined_path)

    if gt_map:
        cer_metric = evaluate.load("cer")
        preds = df["predicted_text"].astype(str).tolist()
        refs = df["ground_truth"].astype(str).tolist()

        missing_docs = sorted(set(df["doc_key"].unique()) - set(gt_map.keys()))
        if missing_docs:
            log.warning("No ground-truth file for %d document keys (showing up to 10): %s", len(missing_docs), missing_docs[:10])

        cer_score = cer_metric.compute(predictions=preds, references=refs)
        print(f"CER (aligned by crop index vs GT lines): {float(cer_score):.6f}")
        summary_path = output_dir / "cer_summary.txt"
        summary_path.write_text(
            f"CER={float(cer_score):.6f}\n"
            f"crops={len(df)}\n"
            f"crops_dir={crops_dir}\n"
            f"ground_truth_dir={ground_truth_dir}\n",
            encoding="utf-8",
        )
        log.info("Wrote %s", summary_path)


def main():
    parser = argparse.ArgumentParser(description="TrOCR inference on YOLO sorted_crops")
    parser.add_argument(
        "--crops-dir",
        type=Path,
        required=True,
        help="Root folder containing per-document subfolders of *.jpg crops",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write word_predictions.csv, transcripts/, combined_transcription.txt",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models" / "trocr" / "final_model",
        help="Directory with TrOCR config + model.safetensors",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Optional folder of <doc_key>.txt files (one word/ref per line, aligned to 000.jpg, ...)",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "8")))
    args = parser.parse_args()

    run(
        crops_dir=args.crops_dir,
        output_dir=args.output_dir,
        model_path=args.model_path.resolve(),
        ground_truth_dir=args.ground_truth_dir.resolve() if args.ground_truth_dir else None,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
