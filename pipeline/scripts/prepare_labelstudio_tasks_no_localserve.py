import os
import json
import unicodedata

IMAGE_ROOT = "/home/ubuntu/Hindi_OCR/dataset/yolo-output/sorted_crops"
TEXT_ROOT = "/home/ubuntu/Hindi_OCR/dataset/ground-truth"
OUTPUT_JSON = "tasks_all.json"
FOLDERS = [str(i) for i in range(31, 42)]

def normalize(text):
    return unicodedata.normalize("NFC", text.strip())

tasks = []

for folder in FOLDERS:
    with open(os.path.join(TEXT_ROOT, f"{folder}.txt"), encoding="utf-8") as f:
        texts = [normalize(l) for l in f if l.strip()]

    images = sorted(
        [f"{folder}/{img}" for img in os.listdir(os.path.join(IMAGE_ROOT, folder)) if img.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    for img, text in zip(images, texts):
        tasks.append({
            "data": {
                "image": img,
                "recognized_text": text
            }
        })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)
