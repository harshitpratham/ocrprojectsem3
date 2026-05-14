# Model weights

## YOLO word detection (~22 MB each)

Committed in this repository under `models/yolo/`:

- `best_yolo_2Dec2025.pt` — primary detector used in `pipeline/scripts/yolo_bounding_boxes.py` on the Ubuntu EC2 VM.
- `bounding_boxes_model.pt` — alternate / earlier bounding-box model (see VM `Handwritten_OCR/models/`).

Run Ultralytics YOLO with these paths locally by setting `MODEL_PATH` or editing the script default.

## TrOCR (ViT + RoBERTa-Hindi)

In this repository, TrOCR **only** lives under `models/trocr/` (`final_model/` and `checkpoint-38000/`).

**`model.safetensors`** (~900 MB) is stored with **Git LFS** (see `.gitattributes`). After clone, run `git lfs pull` if weights did not download automatically. GitHub Free includes limited LFS bandwidth/storage; if the pointer is present but the file is missing, fetch from a teammate or copy from `Handwritten_OCR/src/models/predict_word_model/` on the Ubuntu EC2 VM.

Obtain weights by either:

1. **GPU training EC2** — sync from the instance where `recognition/train.py` was run (`checkpoint-38000` / exported `final_model`); or  
2. **Re-train or export** using `recognition/saving_model_from_checkpoint.py` after placing checkpoints on disk.

Files matching `*.safetensors` under `models/trocr/` are listed in `.gitignore`; add them locally for inference.

## Quick inference layout

Place fine-tuned weights so this path resolves (or set `MODEL_PATH`):

`models/trocr/final_model/model.safetensors` (+ sibling `config.json`, tokenizer files).

The batch pipeline (`pipeline/scripts/batch_predict_word.py`) defaults to the Ubuntu path; override with `MODEL_PATH` for local runs.

## v2 weights / recipes (same storage policy)

v2 **does not** require new weight filenames in-repo for the narrative exercise: the team documents a **second training run** (YOLO + TrOCR adapter) that can overwrite or sit beside v1:

| Artifact | Suggested path | Notes |
|----------|----------------|-------|
| YOLO v2 best | `models/yolo/best_yolo_v2.pt` | Produce with `detection/train_yolov8_v2.py`; not committed until trained |
| TrOCR domain-adapted | `models/trocr/final_model_domain_v2/` | Export after `recognition/train_v2_domain_adapt.py` wiring + GPU run |
| Metric tables | `data/results/v2/*.json` | From `recognition/eval_v2.py` |

Until new binaries exist, continue using `models/yolo/*.pt` and `models/trocr/final_model/` for inference; `pipeline_v2.py` only changes **post-processing** and env defaults.
