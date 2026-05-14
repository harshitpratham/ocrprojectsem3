from ultralytics import YOLO
import os
import cv2
import numpy as np

model = YOLO("/home/ubuntu/Handwritten_OCR/models/bounding_boxes_model.pt")
results = model("/home/ubuntu/Handwritten_OCR/31.jpg")

save_dir = "sorted_crops"
os.makedirs(save_dir, exist_ok=True)

boxes = []

# collect boxes
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    h = y2 - y1
    cy = (y1 + y2) / 2   # vertical center
    crop = results[0].orig_img[int(y1):int(y2), int(x1):int(x2)]
    boxes.append([x1, y1, x2, y2, cy, h, crop])

# ----------------------------------------------------
# STEP 1: cluster into lines using vertical center
# ----------------------------------------------------
lines = []
for b in sorted(boxes, key=lambda x: x[4]):  # sort by center y
    placed = False
    for line in lines:
        # Compare with the first box in line
        ref_cy = line[0][4]
        ref_h = line[0][5]
        
        # Allow dynamic vertical tolerance based on height
        if abs(b[4] - ref_cy) < ref_h * 0.8:
            line.append(b)
            placed = True
            break
    
    if not placed:
        lines.append([b])

# ----------------------------------------------------
# STEP 2: sort each line left → right
# ----------------------------------------------------
for line in lines:
    line.sort(key=lambda b: b[0])  # x1

# ----------------------------------------------------
# STEP 3: flatten in reading order
# ----------------------------------------------------
ordered_boxes = [b for line in lines for b in line]

# ----------------------------------------------------
# Save crops
# ----------------------------------------------------
for idx, b in enumerate(ordered_boxes):
    crop = b[6]
    cv2.imwrite(os.path.join(save_dir, f"{idx:03d}.jpg"), crop)
