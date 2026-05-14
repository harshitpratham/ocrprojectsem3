import sys
from pathlib import Path

from transformers import VisionEncoderDecoderModel

_REC = Path(__file__).resolve().parent
if str(_REC) not in sys.path:
    sys.path.insert(0, str(_REC))

from trocr_hub import build_processor

# Path to your latest checkpoint (change as needed)
checkpoint_path = "/home/ec2-user/indic-trocr/checkpoints/checkpoint-38000"

model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)

processor = build_processor()
processor.save_pretrained("model")
