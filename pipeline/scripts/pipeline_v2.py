#!/usr/bin/env python3
"""
End-to-end pipeline v2 — YOLO crops -> TrOCR -> Unicode / LM / dictionary -> optional HITL gate.

Integrates:
  - detection/postprocess.py for offline box refinement (when xyxy list available)
  - recognition/postprocess for NFC, LM rescoring (scenario-dependent), dictionary correction

Environment:
  LM_ENABLE_FIELD=false   (default) — LM rescoring disabled for Pratham field (v2: +0.6pp CER if enabled)
  LM_ENABLE_EVAL=true     (default) — LM on for HindiSeg-style eval pages
  ASER_VOCAB_PATH         — optional newline-separated Hindi words for dictionary correction
  V2_SCENARIO=eval_pages|field — passed to postprocess_predicted_text defaults

Batch OCR on sorted crops reuses `batch_predict_word.run_on_crops`; this module post-processes
the combined transcription file after batch inference.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO = Path(__file__).resolve().parents[2]
_RECOG = _REPO / "recognition"
_DET = _REPO / "detection"


def _load_detection_postprocess():
    path = _DET / "postprocess.py"
    name = "hindi_ocr_detection_postprocess_v2"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


det_pp = _load_detection_postprocess()
postprocess_detections = det_pp.postprocess_detections

if str(_RECOG) not in sys.path:
    sys.path.insert(0, str(_RECOG))

from postprocess import correct_with_vocab  # noqa: E402
from postprocess import normalize_unicode  # noqa: E402
from postprocess import rescore_with_char_lm  # noqa: E402


def postprocess_predicted_text(
    text: str,
    *,
    scenario: str = "eval_pages",
    vocab_path: Path | None = None,
) -> str:
    """
    scenario: 'eval_pages' | 'field' — controls LM rescoring default per v2 playbook.
    """
    t = normalize_unicode(text)
    lm_field = os.getenv("LM_ENABLE_FIELD", "false").lower() in ("1", "true", "yes")
    lm_eval = os.getenv("LM_ENABLE_EVAL", "true").lower() in ("1", "true", "yes")
    use_lm = lm_eval if scenario != "field" else lm_field
    t = rescore_with_char_lm(t, alternatives=None, enabled=use_lm)
    vp = vocab_path
    if vp is None and os.getenv("ASER_VOCAB_PATH"):
        vp = Path(os.environ["ASER_VOCAB_PATH"])
    t = correct_with_vocab(t, vocab_path=vp)
    return t


def run_full_pipeline_v2():
    """Same S3 orchestration as pipeline.py; apply v2 text post-process on combined transcript."""
    from scripts.batch_predict_word import run_on_crops
    from scripts.yolo_bounding_boxes import run_yolo_and_save_crops
    from utils.s3_utils import S3Client

    s3_bucket = os.getenv("S3_BUCKET")
    local_out = os.getenv("LOCAL_OCR_OUTPUT", "./dataset/ocr_outputs")

    print("Step 1: YOLO + crops")
    run_yolo_and_save_crops()

    print("Step 2: TrOCR batch")
    csv_path, combined_path, transcripts_dir = run_on_crops(
        str("./dataset/yolo-output/sorted_crops"), local_out
    )

    raw = Path(combined_path).read_text(encoding="utf-8", errors="replace")
    scenario = os.getenv("V2_SCENARIO", "eval_pages")
    processed_lines = [postprocess_predicted_text(line, scenario=scenario) for line in raw.splitlines()]
    Path(combined_path).write_text("\n".join(processed_lines), encoding="utf-8")

    if s3_bucket:
        s3 = S3Client(bucket_name=s3_bucket)
        prefix = os.getenv("S3_OUTPUT_PREFIX", "predictions/ocr-output")
        s3.upload_file(str(csv_path), f"{prefix}/word_predictions_v2.csv")
        s3.upload_file(str(combined_path), f"{prefix}/combined_transcription_v2.txt")
        s3.upload_folder(str(transcripts_dir), f"{prefix}/transcripts_v2")

    print("pipeline_v2 complete.")


def demo_offline_postprocess_boxes():
    sample = [(10, 10, 40, 28, 0.9), (42, 12, 70, 30, 0.85), (12, 35, 50, 55, 0.88)]
    for b in postprocess_detections(sample, iou_nms=0.42, enable_merge=True):
        print(b)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo-boxes":
        demo_offline_postprocess_boxes()
    else:
        run_full_pipeline_v2()
