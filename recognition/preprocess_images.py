import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

_REC = Path(__file__).resolve().parent
if str(_REC) not in sys.path:
    sys.path.insert(0, str(_REC))

from trocr_hub import build_processor

# Paths
root_dir = "dataset/"
train_text_file = os.path.join(root_dir, "train.txt")
test_text_file = os.path.join(root_dir, "test.txt")
val_text_file = os.path.join(root_dir, "val.txt")

processor = build_processor()


def dataset_generator(data_path):
    with open(data_path) as f:
        dataset = f.readlines()
    dataset_list = []
    for i in range(len(dataset)):
        image_id = dataset[i].split("\n")[0].split(" ")[0].strip()
        text = dataset[i].split("\n")[0].split(" ")[1].strip()
        row = [image_id, text]
        dataset_list.append(row)
    dataset_df = pd.DataFrame(dataset_list, columns=["file_name", "text"])
    return dataset_df


def preprocess_and_save(df, split_name, processor, root_dir, out_dir, max_target_length=128):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for idx, row in df.iterrows():
        img_path = os.path.join(root_dir, row["file_name"])
        text = row["text"]
        out_path = os.path.join(out_dir, f"{split_name}_{idx}.pt")
        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
            torch.save(pixel_values, out_path)
            meta.append({"tensor_path": out_path, "text": text})
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            exit()
    pd.DataFrame(meta).to_csv(os.path.join(out_dir, f"{split_name}_meta.csv"), index=False)
    print(f"Saved {len(meta)} tensors for {split_name}")


train_df = dataset_generator(train_text_file)
test_df = dataset_generator(test_text_file)
val_df = dataset_generator(val_text_file)

preprocess_and_save(train_df, "train", processor, root_dir, "preprocessed/train")
preprocess_and_save(test_df, "test", processor, root_dir, "preprocessed/test")
preprocess_and_save(val_df, "val", processor, root_dir, "preprocessed/val")
