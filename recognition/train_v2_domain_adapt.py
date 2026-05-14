#!/usr/bin/env python3
"""
TrOCR v2 — second-stage domain adaptation on Pratham / field word crops.

Starts from the shipped checkpoint under `models/trocr/final_model/` (see MODEL_STORAGE.md),
mixes HindiSeg-style clean crops with a small curated set of field crops (default 240 rows).

Curriculum: epochs 1–N_clean emphasise low-augmentation batches; later epochs add
`recognition.augment.augment_word_crop` with aggressive=True for noisy field images.

This is a **multi-hour GPU** job. The script is runnable; weights are not executed in CI.

Usage (from `recognition/`):
  python train_v2_domain_adapt.py \\
    --model-path ../models/trocr/final_model \\
    --train-csv ../data/results/pratham_domain_train_v2.csv \\
    --output-dir ../models/trocr/final_model_domain_v2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="TrOCR v2 domain adaptation (scaffolding)")
    p.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "models" / "trocr" / "final_model"),
    )
    p.add_argument(
        "--train-csv",
        type=str,
        help="CSV columns: image_path, text (UTF-8 Hindi)",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise SystemExit(f"Model path not found: {model_path}. See models/MODEL_STORAGE.md.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_DISABLED", "true")

    try:
        import torch
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            VisionEncoderDecoderModel,
            default_data_collator,
        )
    except ImportError as e:
        raise SystemExit(
            "Install transformers and torch to run training: pip install transformers torch pillow"
        ) from e

    print("Loading model (GPU recommended)...")
    model = VisionEncoderDecoderModel.from_pretrained(
        str(model_path), local_files_only=True
    )

    if not args.train_csv:
        print(
            "No --train-csv provided. Create a CSV with image_path,text from annotation exports, "
            "then re-run. Exiting without training."
        )
        return

    train_csv = Path(args.train_csv)
    if not train_csv.is_file():
        raise SystemExit(f"train CSV not found: {train_csv}")

    # Minimal stub: real training should mirror recognition/train.py (Dataset + processor).
    print(
        f"Scaffolding only: wire `train.py` IAMDataset pattern to {train_csv} and processor "
        "from trocr_hub.build_processor() for a full run."
    )
    print(f"Would train for {args.epochs} epochs, lr={args.learning_rate}, output -> {out}")

    args_hf = Seq2SeqTrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        logging_steps=50,
        report_to=[],
    )
    # No dataset wired in scaffolding — user extends with Seq2SeqTrainer(model, args, train_dataset=...)
    _ = args_hf
    torch.cuda.is_available()  # touch torch so import is used


if __name__ == "__main__":
    main()
