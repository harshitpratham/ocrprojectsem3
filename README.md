# Hindi Handwritten Document Recognition
### ISY5004 · Intelligent Sensing Systems · NUS ISS · Jan–May 2026

End-to-end OCR pipeline: scanned handwritten Hindi document → machine-readable Unicode text.  
Built for [Pratham Education Foundation](https://www.pratham.org/) ASER field surveys.

> **New to this project? Start with [`GUIDE.md`](GUIDE.md)** — it explains everything from scratch.

---

## What It Does

```
Scanned Hindi document
        │
        ▼
  YOLOv8 (word detection)       →  bounding boxes per word
        │
        ▼
  TrOCR (word recognition)      →  Hindi Unicode per word
        │
        ▼
  Full document text (reading order)
```

Optional **batch pipeline** (Ubuntu VM scripts, also in `pipeline/`): YOLO crops → TrOCR → CSV / transcripts → S3. Configure `.env` from [`.env.example`](.env.example).

**v2 pipeline (post-processing + honest eval):** [`pipeline/scripts/pipeline_v2.py`](pipeline/scripts/pipeline_v2.py), metrics under [`data/results/v2/`](data/results/v2/), full narrative in [`docs/COMPLETION_REPORT.md`](docs/COMPLETION_REPORT.md).

## Key Results (v1 baseline vs v2)

| Setting | Metric | v1 | v2 | Delta / note |
|--------|--------|-----|-----|----------------|
| TrOCR — HindiSeg test | CER | **8.17%** | **7.12%** | −12.9% relative; modest (checkpoint near saturation) |
| YOLO — Pratham val | mAP@50 (impl.) | 0.78 | **0.84** | +7.7% relative |
| End-to-end — eval pages 31–41 | CER | **51.8%** | **36.4%** | −29.7% relative; still above sub-25% internal goal |
| End-to-end — Pratham field scans | CER | **98.5%** | **71.3%** | Large absolute improvement; **HITL mandatory** — not unattended-ready |
| Baseline — Haiku | Accuracy (121 crops) | 0% | — | — |
| Human annotators (blind) | Accuracy | ~71% | — | — |

TrOCR trained on 69,853 HindiSeg word images · best v1 checkpoint at step 38,000 (epoch 8.7). **v2** adds domain mix-in (240 Pratham crops from portal history), detection retrain recipe, and text post-processing — see [`models/MODEL_CARD_v2.md`](models/MODEL_CARD_v2.md).

## Team

| Name | Student ID |
|------|-----------|
| Chitrarath Bhattacharjee | E1511452 |
| Gan Jia Hui | E1092587 |
| Harshit Agarwal | E1509813 |

## Quick Start

```bash
# Annotation tool
cd annotation/ && pip install -r requirements.txt && streamlit run app.py

# Evaluate TrOCR (edit paths inside script for your machine)
cd recognition/ && python evaluation_test.py

# End-to-end pipeline entrypoint (needs YOLO + TrOCR weights; see models/MODEL_STORAGE.md)
cd .. && PYTHONPATH=pipeline python pipeline/scripts/pipeline.py

# v2 pipeline: NFC / LM (eval) / dictionary + field-safe LM defaults — see docs/COMPLETION_REPORT.md
# PYTHONPATH=pipeline python pipeline/scripts/pipeline_v2.py

# Regenerate v2 metric tables (narrative / release artifacts)
# python recognition/eval_v2.py --write-release-artifacts

# Load model in Python (place model.safetensors under models/trocr/final_model/ locally)
python -c "from transformers import VisionEncoderDecoderModel; VisionEncoderDecoderModel.from_pretrained('models/trocr/final_model')"
```

## Repository Layout

| Folder | Contents |
|--------|---------|
| `annotation/` | Streamlit multi-user annotation portal |
| `detection/` | YOLOv8 training notebook, config, `predict_bounding_boxes.py` |
| `recognition/` | TrOCR training, evaluation, `predict_word.py` |
| `pipeline/` | Verbatim Ubuntu scripts: `scripts/pipeline.py`, YOLO batch, TrOCR batch, S3 utils |
| `models/yolo/` | `best_yolo_2Dec2025.pt`, `bounding_boxes_model.pt` |
| `models/trocr/` | Config/tokenizer for final + checkpoint-38000 (**weights**: see `MODEL_STORAGE.md`) |
| `data/hindiseg_samples/` | 121 HindiSeg word crops + `ground_truth/` (symlinks `data/word_crops`, `data/ground_truth`) |
| `data/pratham_field/` | ASER-style student ground-truth `.txt` + `label_studio_tasks.json` |
| `data/ec2_ubuntu_hindi_ocr/` | **Full mirror** of `~/Hindi_OCR/dataset/` (images, YOLO I/O, crops, zip, OCR samples) |
| `data/training_splits/` | HindiSeg train/val/test lists (~95K examples) |
| `data/results/` | TrOCR test preds, training curve, Haiku baseline, sample pipeline outputs |
| `data/annotation_exports/` | Portal exports + `annotation_history_ubuntu.csv` (881 rows) |
| `data/results/v2/` | v2 release metrics CSV/JSON (`eval_v2.py`), attribution, kappa, LM ablation |
| `docs/` | Reports, scope, [`COMPLETION_REPORT.md`](docs/COMPLETION_REPORT.md), [`RESTRUCTURE_PLAN.md`](docs/RESTRUCTURE_PLAN.md) |

## Documentation

| File | Description |
|------|-------------|
| [`GUIDE.md`](GUIDE.md) | **Complete guide** — architecture, data, training, results, AWS |
| [`data/DATA_CARD.md`](data/DATA_CARD.md) | Every dataset and file |
| [`models/MODEL_STORAGE.md`](models/MODEL_STORAGE.md) | TrOCR weight policy + YOLO filenames |
| [`docs/ANNOTATION_REPORT.md`](docs/ANNOTATION_REPORT.md) | Human readability study |
| [`docs/HAIKU_ACCURACY_REPORT.md`](docs/HAIKU_ACCURACY_REPORT.md) | Zero-shot baseline analysis |
| [`docs/PROJECT_ANALYSIS.md`](docs/PROJECT_ANALYSIS.md) | Project scope and methodology |
| [`docs/COMPLETION_REPORT.md`](docs/COMPLETION_REPORT.md) | v2 honest completion narrative + per-technique deltas |
| [`models/MODEL_CARD_v2.md`](models/MODEL_CARD_v2.md) | v2 model card, failure modes, deployment constraints |
