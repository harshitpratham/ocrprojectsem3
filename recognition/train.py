import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd

_REC = Path(__file__).resolve().parent
if str(_REC) not in sys.path:
    sys.path.insert(0, str(_REC))

from trocr_hub import build_base_model, build_processor
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import evaluate

# directory and file paths
root_dir = "dataset/"
train_text_file = os.path.join(root_dir, "train.txt")
test_text_file = os.path.join(root_dir, "test.txt")
val_text_file = os.path.join(root_dir, "val.txt")


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


train_df = dataset_generator(train_text_file)
test_df = dataset_generator(test_text_file)
val_df = dataset_generator(val_text_file)

print(f"Train, Test & Val shape: {train_df.shape, test_df.shape, val_df.shape}")

# Handwriting-oriented augmentation (train split only)
_TRAIN_AUGMENT = transforms.Compose(
    [
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    ]
)


class PreprocessedDataset(Dataset):
    """Offline .pt tensors (must match encoder image size — re-run preprocess_images if encoder changes)."""

    def __init__(self, meta_csv, processor, max_target_length=128):
        self.df = pd.read_csv(meta_csv)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tensor_path = self.df["tensor_path"][idx]
        pixel_values = torch.load(tensor_path)  # [3, H, W]
        text = self.df["text"][idx]
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
        ).input_ids
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels),
        }


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128, *, training: bool = False):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]
        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        if self.training:
            image = _TRAIN_AUGMENT(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
        ).input_ids
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]
        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }


processor = build_processor()

# Live images + augmentation (recommended). For EC2 SSD preprocessed tensors instead, use:
# train_dataset = PreprocessedDataset(meta_csv="/mnt/localssd/preprocessed/train/train_meta.csv", processor=processor)
train_dataset = IAMDataset(
    root_dir=root_dir,
    df=train_df,
    processor=processor,
    training=True,
)
eval_dataset = IAMDataset(
    root_dir=root_dir,
    df=test_df,
    processor=processor,
    training=False,
)

model = build_base_model()


def _pick_device() -> torch.device:
    force = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if force == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if force == "cuda" and not torch.cuda.is_available():
        print("WARNING: FORCE_DEVICE=cuda but CUDA not available; falling back.")
    if force == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("WARNING: FORCE_DEVICE=mps but MPS not available; using CPU.")
        return torch.device("cpu")
    if force == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _pick_device()
print("Device used:", device)
if device.type == "mps":
    print("Accelerator: Apple GPU (PyTorch MPS)")
elif device.type == "cuda":
    print("Accelerator:", torch.cuda.get_device_name(0))
model.to(device)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
print(f"processor.tokenizer.pad_token_id: {processor.tokenizer.pad_token_id}")
model.config.vocab_size = model.config.decoder.vocab_size

model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 0
model.config.length_penalty = 1.0
model.config.num_beams = 2

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

# FP16 AMP is reliable on CUDA; MPS uses float32 in Trainer (still fast on Apple GPU).
_use_fp16 = device.type == "cuda"
_is_darwin = platform.system() == "Darwin"
# MPS + forked DataLoader workers often hangs on macOS; default 0. Override: DATALOADER_NUM_WORKERS=2
_num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "0" if _is_darwin else min(8, (os.cpu_count() or 2))))
# Larger batches when an accelerator is available (384x384 images — reduce if OOM).
_default_train_bs = "8" if device.type != "cpu" else "4"
_default_eval_bs = "16" if device.type != "cpu" else "8"
_per_device_train = int(os.environ.get("TRAIN_BATCH_SIZE", _default_train_bs))
_per_device_eval = int(os.environ.get("EVAL_BATCH_SIZE", _default_eval_bs))

_use_mps = device.type == "mps"

training_args = Seq2SeqTrainingArguments(
    num_train_epochs=10,
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=_per_device_train,
    per_device_eval_batch_size=_per_device_eval,
    output_dir="./checkpoints/",
    logging_steps=2,
    save_steps=2000,
    eval_steps=2000,
    fp16=_use_fp16,
    use_mps_device=_use_mps,
    dataloader_num_workers=_num_workers,
    gradient_accumulation_steps=1,
)

cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    start = time.time()
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    print(f"compute_metrics time: {time.time() - start:.2f}s")
    return {"cer": cer}


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

print(
    f"Training: batch_size train={_per_device_train} eval={_per_device_eval}, "
    f"fp16={_use_fp16}, use_mps_device={_use_mps}, dataloader_num_workers={_num_workers}"
)

trainer.train()

os.makedirs("model", exist_ok=True)
model.save_pretrained("model")
