# Model card — Hindi OCR v2 (YOLO + TrOCR)

**Release:** v2 (student project narrative; weights policy unchanged — see [MODEL_STORAGE.md](MODEL_STORAGE.md)).

## Intended use

- **In scope:** Offline / batch digitisation of handwritten Hindi **worksheet-style** pages where word crops are reasonably consistent with training (IIIT HindiSeg + limited Pratham adaptation).
- **Out of scope (v2):** Fully unattended transcription of **diverse Pratham field** uploads; v2 **`requires HITL`** for field documents when word-level accuracy is below operational threshold.

## Models

| Component | v1 | v2 recipe |
|-----------|-----|-----------|
| Word detection | YOLOv8 (Pratham tuned) | Retrain emphasis: smaller IoU NMS, matra-aware **loss weighting** (YAML), scan augmentation |
| Word recognition | TrOCR (ViT + Hindi RoBERTa) | Second-stage **domain mixin** on **240** Pratham-labelled crops (curated from portal exports) |
| Post-processing | — | NFC Unicode; char 3-gram LM (**off** on field by default); ASER dictionary (~4.2k words) |

## Metrics (see `data/results/v2/`)

- HindiSeg test CER: **7.12%** (v1: **8.17%**)
- End-to-end CER (eval pages 31–41): **36.4%** (v1: **51.78%**)
- End-to-end CER (Pratham field): **71.3%** (v1: **98.54%**)
- YOLO mAP@50: **0.84** (v1 impl. **0.78**)

## Known failure modes

1. **Box merging** (`detection/postprocess.py`): adjacent words on dense rows can merge (~**3%** rows in internal audit).
2. **LM rescoring**: improves eval-style text; **hurts** field CER when LM is HindiSeg-skewed — **disabled for field** in `pipeline_v2`.
3. **Reading order**: fails when YOLO order does not match ground-truth line indexing (evaluation is intentionally strict).
4. **Gold-set quality**: Cohen’s κ **0.62** (target **0.8**); labels for adaptation are **noisy**.

## Ethical / deployment notes

- Human reviewers must handle low-confidence predictions; do not use v2 field outputs for high-stakes decisions without audit.
- Annotator data derived from Pratham-related exports — follow client data-handling agreements.

## How to reproduce metric tables

```bash
python recognition/eval_v2.py --write-release-artifacts
```

Training scripts (`detection/train_yolov8_v2.py`, `recognition/train_v2_domain_adapt.py`) require GPU time and curated CSVs — not run in-repo by default.
