#!/usr/bin/env python3
"""
YOLOv8 v2 training entrypoint — Pratham / Hindi handwritten word detection.

This script documents the v2 *recipe* (matra-aware emphasis + field-safe augmentation).
On stock Ultralytics, “matra-aware loss” is approximated by:
  - box loss weight elevation in YAML (`box`, `cls`)
  - smaller IoU for NMS in post-processing (see detection/postprocess.py)
  - no horizontal flip (Devanagari)

Full custom bbox loss would require a fork of ultralytics; this repo keeps a vanilla
train path that is reproducible on EC2.

Usage (from repo root, after preparing dataset per detection/configs/yolo_v2_pratham.yaml):
  cd detection
  python train_yolov8_v2.py --data configs/yolo_v2_pratham.yaml --epochs 80

GPU strongly recommended (multi-hour training).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 v2 for Hindi word detection")
    p.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "yolo_v2_pratham.yaml"),
        help="Path to dataset + hyperparameter YAML",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override epochs from YAML")
    p.add_argument("--imgsz", type=int, default=None, help="Override image size")
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Initial weights .pt (default: yolov8s.pt from YAML)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from e

    data_yaml = Path(args.data).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Data YAML not found: {data_yaml}")

    model_path = args.weights or "yolov8s.pt"
    model = YOLO(model_path)

    train_kw = dict(
        data=str(data_yaml),
        exist_ok=True,
    )
    if args.epochs is not None:
        train_kw["epochs"] = args.epochs
    if args.imgsz is not None:
        train_kw["imgsz"] = args.imgsz

    # Remaining hyperparameters are taken from the YAML (epochs, imgsz, augment, etc.).
    model.train(**train_kw)


if __name__ == "__main__":
    main()
