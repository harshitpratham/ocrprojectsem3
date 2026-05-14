# Retraining playbook — Hindi OCR v2/v3

This document is **scaffolding** for operations once Pratham signs off on integration.
Thresholds below are placeholders.

## When to retrain YOLO (detection)

- Rolling median word-level IoU against human boxes drops below **0.72** on a weekly audit sample, **or**
- New camera / paper template rollout changes crop aspect-ratio distribution (see `pipeline/monitoring/drift_detector.py`).

**Steps**

1. Export new Label Studio rectangles for affected template IDs.
2. Merge into `data/yolo_training_dataset` (or EC2 mirror path).
3. Run `detection/train_yolov8_v2.py` with updated `detection/configs/yolo_v2_pratham.yaml` paths.
4. Run offline eval on folders 31–41; update `data/results/v2/yolo_v2_metrics.json`.

## When to retrain / adapt TrOCR (recognition)

- HindiSeg-holdout CER rises above **0.09** after a detection change, **or**
- Field audit shows systematic confusions on new curriculum vocabulary.

**Steps**

1. Build CSV `image_path,text` from `annotation_history_ubuntu.csv` or gold set (`data/results/gold_set_v2.csv`).
2. Run `recognition/train_v2_domain_adapt.py` with `--train-csv` (extend script to full Trainer wiring as in `recognition/train.py`).
3. Run `recognition/evaluation_test.py` (HindiSeg) and `recognition/run_local_inference.py` (YOLO crops).

## LM rescoring

- **Eval pages / HindiSeg-like layouts:** LM on by default (`lm_enabled=True`).
- **Pratham field scans:** LM **off** by default — HindiSeg-skewed char n-gram hurt field CER (~+0.6pp in v2 ablation). v3: rebuild LM on in-domain transcripts.

## Human-in-the-loop

- Route crops with `confidence < 0.55` (decoder score or heuristic) to `pipeline/hitl/review_app.py`.
- Gold-set updates: run `annotation/scripts/build_gold_set.py` after dual annotation.

## Rollback

- Keep previous `models/yolo/*.pt` and `models/trocr/final_model/` snapshot on S3 (see `models/MODEL_STORAGE.md`).
- Pipeline entrypoints: `pipeline/scripts/pipeline.py` (v1) vs `pipeline/scripts/pipeline_v2.py` (v2).
