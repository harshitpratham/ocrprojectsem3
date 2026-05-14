import os
import json
import unicodedata

# ---------------- CONFIG ----------------
TEXT_ROOT = "/home/ubuntu/Hindi_OCR/dataset/ground-truth"
IMAGE_ROOT = "/home/ubuntu/Hindi_OCR/dataset/yolo-output/sorted_crops"
OUTPUT_JSON = "/home/ubuntu/Hindi_OCR/dataset/tasks_all.json"
FOLDERS = [str(i) for i in range(31, 42)]  # 31 to 41 inclusive
# ---------------------------------------


def normalize_hindi(text):
    """Unicode-safe normalization for Hindi"""
    return unicodedata.normalize("NFC", text.strip())


tasks = []
total_tasks = 0

for folder in FOLDERS:
    text_file = os.path.join(TEXT_ROOT, f"{folder}.txt")
    image_dir = os.path.join(IMAGE_ROOT, folder)

    if not os.path.exists(text_file):
        print(f"❌ Missing text file: {text_file} — skipping")
        continue

    if not os.path.isdir(image_dir):
        print(f"❌ Missing image folder: {image_dir} — skipping")
        continue

    # Read text lines
    with open(text_file, "r", encoding="utf-8") as f:
        texts = [normalize_hindi(line) for line in f if line.strip()]

    # Read and sort images numerically
    images = sorted(
        [img for img in os.listdir(image_dir) if img.lower().endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    num_images = len(images)
    num_texts = len(texts)

    # Safety checks
    if num_images == 0:
        print(f"⚠ Folder {folder}: No images found — skipping")
        continue

    if num_texts == 0:
        print(f"⚠ Folder {folder}: No text lines found — skipping")
        continue

    # Case 1: images > texts → reuse last text
    if num_images > num_texts:
        print(
            f"⚠ Folder {folder}: {num_images} images but {num_texts} texts. "
            f"Reusing last text for remaining images."
        )
        texts = texts + [texts[-1]] * (num_images - num_texts)

    # Case 2: texts > images → truncate extra texts
    elif num_texts > num_images:
        print(
            f"⚠ Folder {folder}: {num_texts} texts but {num_images} images. "
            f"Ignoring extra text lines."
        )
        texts = texts[:num_images]

    # Create tasks
    for idx, (img, text) in enumerate(zip(images, texts)):
        abs_path = os.path.join(image_dir, img)

        task = {
            "id": f"{folder}_{idx:03d}",
            "data": {
                "image": f"/data/local-files/?d={abs_path}",
                "recognized_text": text,
                "source_set": folder
            }
        }

        tasks.append(task)

    total_tasks += num_images
    print(f"✔ Folder {folder}: {num_images} tasks created")

# Write output
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"\n✅ Created {OUTPUT_JSON}")
print(f"📊 Total tasks: {total_tasks}")
