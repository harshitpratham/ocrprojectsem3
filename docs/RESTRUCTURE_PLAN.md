# Hindi OCR repository restructure (updated)

## Principles

1. **No new logic** — Copy existing scripts from EC2 mirrors and VMs verbatim. Pipeline code lives in `pipeline/scripts/` and `pipeline/utils/` exactly as on the Ubuntu `Hindi_OCR` tree; only **file layout** and **documentation** change.
2. **No `ec2-workspace/` mirrors in Git** — After copying needed assets into `data/`, `models/`, and `pipeline/`, delete local mirror folders and stop tracking them.
3. **Large TrOCR weights** — `model.safetensors` / huge binaries are not hosted on GitHub; keep **configs only** under `models/trocr/` and document weight location in `models/MODEL_STORAGE.md`.
4. **Small YOLO weights** (~22 MB each) — Commit under `models/yolo/` when available.

## Target layout

See `README.md` and `GUIDE.md` for the canonical tree after restructure (`data/hindiseg_samples/`, `data/pratham_field/`, `data/training_splits/`, `data/results/`, `models/trocr/`, `models/yolo/`, `pipeline/`).

## Run pipeline scripts (unchanged imports)

From repo root:

```bash
cd hindi-ocr-pipeline
PYTHONPATH=pipeline python pipeline/scripts/pipeline.py
```

Override `/home/ubuntu/...` defaults with the same environment variables you use on EC2 (`MODEL_PATH`, `S3_BUCKET`, etc.).

## Execution checklist

- [x] Plan revised (this file)
- [x] Copy Pratham ground truth, Label Studio export, OCR outputs, ubuntu annotation history
- [x] Restructure `data/` and `models/`
- [x] Add verbatim `predict_bounding_boxes.py`, `predict_word.py`
- [x] Fetch YOLO `.pt` into `models/yolo/`
- [x] Remove `ec2-workspace*`, update `.gitignore` / `.gitattributes`
- [x] Purge `.safetensors` blobs from git history (`git filter-repo`)
- [x] Refresh `GUIDE.md`, `README.md`, `data/DATA_CARD.md`, `models/MODEL_STORAGE.md`
- [x] Push to `harshitpratham/hindiocr` (after `git push --force-with-lease`)
- [x] **v2 scaffolding** — `detection/train_yolov8_v2.py`, `detection/postprocess.py`, `recognition/eval_v2.py`, `pipeline/scripts/pipeline_v2.py`, `docs/COMPLETION_REPORT.md`, `data/results/v2/` (May 2026)

**Note:** Principle (1) still holds for original EC2 mirrors; v2 files are **new** documented scaffolding layered on top.
