# scripts/yolo_bounding_boxes.py

import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from utils.s3_utils import S3Client

# --- CONFIG ---
S3_INPUT_PREFIX = "dataset/handwritten-yolo/test"
S3_OUTPUT_PREFIX = "predictions/yolo-output"    
# MODEL_PATH = "/home/ubuntu/Handwritten_OCR/models/bounding_boxes_model.pt"
MODEL_PATH = "/home/ubuntu/Handwritten_OCR/models/best_yolo_2Dec2025.pt"
LOCAL_DATA_DIR = "/home/ubuntu/Hindi_OCR/dataset/plain-dataset/images"
LOCAL_OUTPUT_DIR = "/home/ubuntu/Hindi_OCR/dataset/plain-dataset/yolo-output"
SAVE_CROPS_DIR = "/home/ubuntu/Hindi_OCR/dataset/plain-dataset/yolo-output/sorted_crops"  # our custom sorted crops folder

s3 = S3Client(bucket_name=os.environ.get("S3_BUCKET"))

def ensure_dirs():
    Path(LOCAL_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOCAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(SAVE_CROPS_DIR).mkdir(parents=True, exist_ok=True)

def extract_sorted_word_crops(result, out_dir: Path):
    """
    Applies the robust line-clustering (vertical center + dynamic threshold)
    and saves crops in proper reading order: left→right, then top→bottom.
    """
    boxes = []
    image = result.orig_img

    # -----------------------------------------
    # Extract YOLO boxes
    # -----------------------------------------
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        h = y2 - y1
        cy = (y1 + y2) / 2  # vertical center
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        boxes.append([x1, y1, x2, y2, cy, h, crop])

    if len(boxes) == 0:
        print("⚠️ No boxes detected.")
        return

    # -----------------------------------------
    # STEP 1 – Cluster into lines using vertical center
    # -----------------------------------------
    lines = []
    for b in sorted(boxes, key=lambda x: x[4]):  # sorted by vertical center
        placed = False
        for line in lines:
            ref_cy = line[0][4]
            ref_h  = line[0][5]
            if abs(b[4] - ref_cy) < ref_h * 0.8:  # dynamic threshold
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    # -----------------------------------------
    # STEP 2 – Sort each line left → right
    # -----------------------------------------
    for line in lines:
        line.sort(key=lambda b: b[0])  # by x1

    # -----------------------------------------
    # STEP 3 – Merge lines top→bottom
    # -----------------------------------------
    ordered_boxes = [b for line in lines for b in line]

    # -----------------------------------------
    # STEP 4 – Save numbered crops
    # -----------------------------------------
    for idx, b in enumerate(ordered_boxes):
        crop = b[6]
        filename = f"{idx:03d}.jpg"
        cv2.imwrite(str(out_dir / filename), crop)

    print(f"Saved {len(ordered_boxes)} sorted crops → {out_dir}")


def run_yolo_and_save_crops():
    ensure_dirs()
    print("Downloading images from S3...")
    # s3.download_folder(S3_INPUT_PREFIX, LOCAL_DATA_DIR)
    # print("Done downloading images.")

    model = YOLO(MODEL_PATH)
    print("Running YOLO inference...")

    # IMPORTANT: We do NOT use YOLO save_crop=True because we want custom ordering
    results = model.predict(
        source=LOCAL_DATA_DIR,
        # conf=0.8,
        save=False,         # disable YOLO's save
        save_crop=False,    # we will handle cropping manually
        verbose=False
    )

    # Now process each image and generate sorted crops
    for result in results:
        image_name = Path(result.path).stem
        out_dir = Path(SAVE_CROPS_DIR) / image_name
        out_dir.mkdir(parents=True, exist_ok=True)

        extract_sorted_word_crops(result, out_dir)

    print("✅ YOLO prediction + sorted crop generation completed!")


if __name__ == "__main__":
    run_yolo_and_save_crops()
