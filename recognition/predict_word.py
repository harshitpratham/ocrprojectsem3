import os
import sys
from pathlib import Path

from PIL import Image
from transformers import VisionEncoderDecoderModel

REC = Path(__file__).resolve().parent
if str(REC) not in sys.path:
    sys.path.insert(0, str(REC))

from trocr_hub import build_processor

REPO_ROOT = Path(__file__).resolve().parent.parent

processor = build_processor()

_default_model = REPO_ROOT / "models" / "trocr" / "final_model"
MODEL_DIR = Path(os.environ.get("PREDICT_WORD_MODEL", str(_default_model)))

model = VisionEncoderDecoderModel.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)


def preview(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_word.py <image_path>")
        raise SystemExit(1)
    preview(sys.argv[1])
