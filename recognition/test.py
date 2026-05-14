import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from transformers import VisionEncoderDecoderModel

_REC = Path(__file__).resolve().parent
if str(_REC) not in sys.path:
    sys.path.insert(0, str(_REC))

from trocr_hub import build_processor

processor = build_processor()

model = VisionEncoderDecoderModel.from_pretrained("checkpoints/checkpoint-26000")


def preview(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    plt.imshow(image)
    print(generated_text)


image_path = "/home/ec2-user/indic-trocr/dataset/HindiSeg/test/6/1/3.jpg"
preview(image_path=image_path)
