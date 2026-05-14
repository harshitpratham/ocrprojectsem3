# scripts/batch_predict_word.py
import os
import re
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import logging

from transformers import VisionEncoderDecoderModel

_RECOG = Path(__file__).resolve().parents[2] / "recognition"
if str(_RECOG) not in sys.path:
    sys.path.insert(0, str(_RECOG))

from trocr_hub import build_base_model, build_processor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------
# Config (env-friendly + defaults)
# -----------------------------
CROPS_FOLDER = Path(os.getenv("CROPS_FOLDER", "./dataset/yolo-output/yolo_predictions/crops"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./dataset/ocr_outputs"))
# Use the path you actually have. Make sure it points to directory containing model.safetensors etc.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/home/ubuntu/Handwritten_OCR/src/models/predict_word_model"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))

# Hub IDs for logging / optional override of processor source (image side still MS handwritten)
ENCODER_NAME = os.getenv("ENCODER_NAME", "microsoft/trocr-base-handwritten")
DECODER_NAME = os.getenv("DECODER_NAME", "flax-community/roberta-hindi")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# -----------------------------
# Filename parser
# -----------------------------
# expects filenames like: <origstem>__x1_y1_x2_y2__idx.jpg
FNAME_RE = re.compile(r"(?P<orig>.+)__ (?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)__(?P<idx>\d+)\.jpg".replace(" ", ""))
# simpler fallback split uses '__' separator

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
                "idx": idx
            }
    # fallback
    return None

# -----------------------------
# Lazy model + processor loader
# -----------------------------
_model = None
_processor = None

def load_model_and_processor(model_path: Path):
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    _processor = build_processor()

    # Try to load composite VisionEncoderDecoderModel from the model_path first
    try:
        log.info(f"Attempting to load VisionEncoderDecoderModel from {model_path}")
        _model = VisionEncoderDecoderModel.from_pretrained(str(model_path), local_files_only=True)
        log.info("Loaded VisionEncoderDecoderModel from local directory.")
    except Exception as e:
        log.warning(f"Failed to load composite model from {model_path}: {e}")
        log.info("Falling back to building model from encoder+decoder and loading weights if available.")

        # Build model from encoder+decoder pretrained identifiers, then try to load weights
        _model = build_base_model()
        # Attempt to load saved weights (safetensors or pytorch bin)
        safetensors_path = model_path / "model.safetensors"
        pytorch_bin_path = model_path / "pytorch_model.bin"
        if safetensors_path.exists():
            try:
                import safetensors.torch
                state = safetensors.torch.load_file(str(safetensors_path))
                _model.load_state_dict(state, strict=False)
                log.info(f"Loaded weights from {safetensors_path}")
            except Exception as ex:
                log.warning(f"Could not load safetensors weights: {ex}")
        elif pytorch_bin_path.exists():
            try:
                state = torch.load(str(pytorch_bin_path), map_location="cpu")
                # if the checkpoint only contains .state_dict() under 'model', try to handle it
                if isinstance(state, dict) and "model_state_dict" in state:
                    _model.load_state_dict(state["model_state_dict"], strict=False)
                else:
                    _model.load_state_dict(state, strict=False)
                log.info(f"Loaded weights from {pytorch_bin_path}")
            except Exception as ex:
                log.warning(f"Could not load pytorch weights: {ex}")
        else:
            log.info("No local weight file found; using freshly initialized encoder-decoder model (no fine-tuned weights).")

    _model.to(device)
    _model.eval()
    return _model, _processor

# -----------------------------
# Inference helpers
# -----------------------------
def load_pil(image_path):
    return Image.open(image_path).convert("RGB")

def infer_batch(model, processor, image_paths):
    images = [load_pil(p) for p in image_paths]
    encoding = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = encoding.pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [t.strip() for t in generated_texts]

# -----------------------------
# Core runner
# -----------------------------
def run_on_crops(crops_folder: str | Path, out_dir: str | Path, batch_size: int = BATCH_SIZE):
    crops_folder = Path(crops_folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-load model & processor here (avoids import-time failures)
    model, processor = load_model_and_processor(MODEL_PATH)

    # crop_paths = sorted([p for p in crops_folder.glob("*.jpg")])
    crop_paths = sorted([p for p in crops_folder.rglob("*.jpg")])

    print("Crops path:", crop_paths)
    log.info(f"Found {len(crop_paths)} crop images in {crops_folder}")

    rows = []
    for i in tqdm(range(0, len(crop_paths), batch_size), desc="OCR Batches"):
        batch = crop_paths[i:i+batch_size]
        try:
            preds = infer_batch(model, processor, batch)
        except Exception as e:
            log.exception(f"Error during inference for batch starting at index {i}: {e}")
            # fallback: try single-image inference to isolate bad images
            preds = []
            for img_path in batch:
                try:
                    preds.append(infer_batch(model, processor, [img_path])[0])
                except Exception as ex:
                    log.warning(f"Skipping {img_path} due to inference error: {ex}")
                    preds.append("")
        for p, pred in zip(batch, preds):
            meta = parse_crop_filename(p.name)
            if meta is None:
                meta = {"orig_image": p.stem, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "idx": 0}
            rows.append({
                "original_image": meta["orig_image"],
                "crop_filename": p.name,
                "x1": meta["x1"],
                "y1": meta["y1"],
                "x2": meta["x2"],
                "y2": meta["y2"],
                "predicted_text": pred
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "word_predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved word-level CSV: {csv_path}")

    # Build transcripts
    transcripts_dir = out_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    final_lines = []
    for orig_name, group in df.groupby("original_image"):
        ordered = group.sort_values(by=["y1", "x1"])
        line_text = " ".join(ordered["predicted_text"].astype(str).tolist())
        per_img_path = transcripts_dir / f"{orig_name}.txt"
        per_img_path.write_text(line_text, encoding="utf-8")
        final_lines.append(f"=== {orig_name} ===\n{line_text}\n")

    combined_path = out_dir / "combined_transcription.txt"
    combined_path.write_text("\n".join(final_lines), encoding="utf-8")
    log.info(f"Saved combined transcription: {combined_path}")

    return csv_path, combined_path, transcripts_dir

# Run when executed directly
if __name__ == "__main__":
    run_on_crops(CROPS_FOLDER, OUTPUT_DIR)
