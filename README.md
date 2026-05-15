# Hindi Handwritten Document Recognition
### ISY5004 · Intelligent Sensing Systems · NUS ISS · Jan–May 2026

End-to-end OCR pipeline: scanned handwritten Hindi document → machine-readable Unicode text.  
Built for [Pratham Education Foundation](https://www.pratham.org/) ASER field surveys.

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

## Key Results (v1 baseline vs v2)

| Setting | Metric | v1 | v2 | Delta / note |
|--------|--------|-----|-----|----------------|
| TrOCR — HindiSeg test | CER | **8.17%** | **7.12%** | −12.9% relative |
| YOLO — Pratham val | mAP@50 | 0.78 | **0.84** | +7.7% relative |
| End-to-end — eval pages 31–41 | CER | **51.8%** | **36.4%** | −29.7% relative |
| End-to-end — Pratham field scans | CER | **98.5%** | **71.3%** | HITL mandatory — not unattended-ready |
| Baseline — Haiku 4.5 (zero-shot) | Accuracy (121 crops) | 0% | — | — |
| Human annotators (blind) | Accuracy | ~71% | — | — |

TrOCR trained on 69,853 HindiSeg word images · best checkpoint at step 38,000 (epoch 8.7).

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

# End-to-end pipeline (needs YOLO + TrOCR weights; see models/MODEL_STORAGE.md)
PYTHONPATH=pipeline python pipeline/scripts/pipeline.py

# v2 pipeline with post-processing
PYTHONPATH=pipeline python pipeline/scripts/pipeline_v2.py
```

## Repository Layout

| Folder | Contents |
|--------|---------|
| `annotation/` | Streamlit multi-user annotation portal |
| `detection/` | YOLOv8 training scripts, config, `predict_bounding_boxes.py` |
| `recognition/` | TrOCR training, evaluation, `predict_word.py`, post-processing |
| `pipeline/` | Batch pipeline scripts, monitoring, HITL, playbooks |
| `models/` | Model configs and cards (weights not included — see `models/MODEL_STORAGE.md`) |

## Data & Documentation Availability

> **The `data/` and `docs/` folders are not included in this public repository.**
>
> This project was built in partnership with **Pratham Education Foundation** under a non-disclosure agreement. Pratham field survey data, evaluation artefacts, annotated word crops, and detailed project reports cannot be publicly released. If you are a project team member or NUS assessor, please contact the team directly for access.

The code (all training scripts, pipeline, annotation portal, post-processing) is fully available above. Model weights are stored separately — see `models/MODEL_STORAGE.md` for access instructions.

## Model Weights

Model weights (`model.safetensors`, ~914 MB) are not stored in this repository due to size. See [`models/MODEL_STORAGE.md`](models/MODEL_STORAGE.md) for how to obtain them from the project EC2 instance.
