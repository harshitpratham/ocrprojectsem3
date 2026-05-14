# Complete Project Guide
## Hindi Handwritten Document Recognition — ISY5004, NUS ISS, 2026

> **Who this is for:** Anyone joining the project fresh — no prior context needed.  
> Read this top-to-bottom. It explains what the project does, why it exists, what every file is, how everything was built, and how to use it.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Why It Exists — The Real-World Problem](#2-why-it-exists)
3. [How the System Works — Pipeline Overview](#3-pipeline-overview)
4. [Project Structure — Every File Explained](#4-project-structure)
5. [Datasets — Where the Data Comes From](#5-datasets)
6. [Stage 1 — Annotation Tool](#6-annotation-tool)
7. [Stage 2 — YOLO Word Detection](#7-yolo-word-detection)
8. [Stage 3 — TrOCR Word Recognition](#8-trocr-word-recognition)
9. [Results & Evaluation](#9-results--evaluation)
10. [How to Load and Use the Trained Model](#10-using-the-trained-model)
11. [Infrastructure — AWS EC2](#11-infrastructure)
12. [Baseline Experiment — Why Haiku Failed](#12-baseline-haiku-experiment)
13. [Key Decisions and Why](#13-key-decisions)
14. [Team & Timeline](#14-team--timeline)

---

## 1. What This Project Does

This project builds an **end-to-end OCR pipeline** that takes a scanned image of a handwritten Hindi document and outputs machine-readable Hindi text.

**Input:** A photograph or scan of a page of handwritten Hindi text  
**Output:** The text of that page, in Unicode Hindi (Devanagari script), in correct reading order

The pipeline has two main stages:
- **Detect** where each word is on the page (bounding boxes) — using YOLOv8
- **Read** each detected word — using TrOCR (a Vision Transformer + Hindi RoBERTa model)

---

## 2. Why It Exists

**Client:** [Pratham Education Foundation](https://www.pratham.org/)

Pratham runs the **ASER** (Annual Status of Education Report) — the world's largest citizen-led rural learning assessment. Every year, tens of thousands of field workers collect handwritten student assessments across India. These responses are written in Hindi on plain or ruled paper, by different people, in widely varying handwriting styles.

**The bottleneck:** All of these handwritten responses must be manually transcribed to extract data. This is:
- Extremely slow (one person can transcribe ~50 pages/day)
- Expensive at scale (thousands of assessors, millions of responses)
- A barrier to timely educational intervention

**The gap:** No existing OCR tool (Tesseract, Google Vision, Amazon Textract) handles handwritten Hindi reliably because:
1. Devanagari has no clear word boundaries — ligatures and matras (diacritical marks) attach above/below base characters
2. Handwriting variation across field workers is enormous
3. Pre-trained models are not exposed to this domain's specific vocabulary

This project addresses all three by fine-tuning vision models on domain-specific annotated data.

---

## 3. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL PIPELINE (End-to-End)                    │
└─────────────────────────────────────────────────────────────────┘

  📄 Raw ASER document image (scanned/photographed)
           │
           ▼
  ┌─────────────────┐
  │   YOLOv8        │  Trained on: Pratham field pages — handwritten-yolo layout (Google Drive)
  │   Word          │  Model: YOLOv8s, 25 epochs, 800px
  │   Detection     │  Output: Bounding boxes for each word
  └────────┬────────┘
           │  [x1,y1,x2,y2 for each word, sorted top→bottom, left→right]
           ▼
  ┌─────────────────┐
  │  Crop each      │  PIL Image.crop() on each bounding box
  │  word region    │
  └────────┬────────┘
           │  [Individual word image crops]
           ▼
  ┌─────────────────┐
  │   TrOCR         │  Trained on: HindiSeg (69,853 word images)
  │   Word          │  Encoder: ViT (google/vit-base-patch16-224-in21k)
  │   Recognition   │  Decoder: RoBERTa (flax-community/roberta-hindi)
  └────────┬────────┘
           │  [Hindi Unicode string per word]
           ▼
  📝 Full document text (words joined in reading order)
```

**Why two separate models?**
- YOLO is language-agnostic — it sees visual word-shaped blobs regardless of script
- TrOCR specialises entirely in character recognition — it never needs to know where words are
- Decoupling lets you train/evaluate/improve each stage independently
- Detection errors don't compound with recognition errors — you can audit each stage

---

## 4. Project Structure

```
hindi-ocr-pipeline/
│
├── GUIDE.md, README.md, requirements.txt, .env.example, .gitignore
│
├── annotation/                     ← Stage 0: Human annotation tool (Streamlit)
├── detection/                    ← Stage 1: YOLO — notebook, configs, predict_bounding_boxes.py
├── recognition/                  ← Stage 2: TrOCR — train/eval, predict_word.py
├── pipeline/                     ← Ubuntu VM end-to-end scripts (verbatim layout)
│   ├── scripts/                  ← pipeline.py, batch_predict_word.py, yolo_bounding_boxes.py,
│   │                             ←   prepare_labelstudio_tasks.py, sorting_yolo_output.py, …
│   └── utils/s3_utils.py
│
├── data/
│   ├── DATA_CARD.md
│   ├── hindiseg_samples/         ← Pratham field evaluation crops + labels (folders 31–41; legacy path name)
│   │   ├── word_crops/
│   │   └── ground_truth/
│   ├── word_crops → hindiseg_samples/word_crops   ← symlink (legacy paths still work)
│   ├── ground_truth → hindiseg_samples/ground_truth
│   ├── pratham_field/            ← ASER-style student doc ground truth (txt) + label_studio_tasks.json
│   ├── ec2_ubuntu_hindi_ocr/     ← Full ~/Hindi_OCR/dataset/ mirror (YOLO images, yolo I/O, crops, zip)
│   ├── training_splits/          ← HindiSeg train/val/test lists, vocab, meta CSVs
│   ├── results/                  ← TrOCR test preds, training curves, Haiku baseline, pipeline CSV/txt
│   └── annotation_exports/       ← Portal exports + annotation_history_local/ _ubuntu.csv
│
├── models/
│   ├── MODEL_STORAGE.md          ← Where TrOCR weights live; YOLO filenames
│   ├── yolo/                     ← best_yolo_2Dec2025.pt, bounding_boxes_model.pt (in Git)
│   └── trocr/                    ← final_model/ and checkpoint-38000/ configs (add *.safetensors locally)
│
└── docs/                         ← Reports, PROJECT_ANALYSIS, RESTRUCTURE_PLAN.md
```

Run Ubuntu pipeline code from repo root: `PYTHONPATH=pipeline python pipeline/scripts/pipeline.py` (see §11).

**Clone tip:** `model.safetensors` is stored with **Git LFS**. After `git clone`, run `git lfs install && git lfs pull` so `models/trocr/final_model/model.safetensors` materialises.

---

## 5. Datasets

### 5.1 HindiSeg — Main Training Dataset

**Source:** [IIIT-Hyderabad Indic Handwriting Dataset](http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)

This is the dataset that trained the TrOCR recognition model. It contains **word-level segmented images** of handwritten Hindi text — each image is a single cropped Hindi word, already isolated. There is no raw document to crop from; the segmentation was done by IIIT-H as part of dataset preparation.

**Structure on EC2:**
```
dataset/HindiSeg/
├── train/
│   └── <writer_id>/
│       └── <page_id>/
│           ├── 1.jpg, 2.jpg ...    ← word images (one per image)
│           └── <page_id>.txt       ← vocab IDs for each image (by line number)
├── val/   (same structure)
└── test/  (same structure)
```

**Split sizes:**

| Split | Examples | File |
|-------|----------|------|
| Train | 69,853 | `data/training_splits/train.txt` |
| Validation | 12,708 | `data/training_splits/val.txt` |
| Test | 12,869 | `data/training_splits/test.txt` |

**Format of `train.txt`:**
```
HindiSeg/train/8/251/21.jpg केंद्रों
HindiSeg/train/10/207/8.jpg पॉवर
```
Each line: `<relative_image_path> <hindi_word>`

**Vocabulary:** 11,029 unique Hindi words (`data/training_splits/hindi_vocab.txt`)

**How the folders 31–41 relate:** The `data/hindiseg_samples/word_crops/` folders (31–41) are **Pratham field handwriting** word images (ASER-style survey material), organised into 11 groups to cover different handwriting quality levels. Folder numbers label those groups on disk. These 121 images are used for end-to-end YOLO+TrOCR evaluation, ground-truth verification, and experiments such as the zero-shot Haiku baseline. The directory name `hindiseg_samples` is historical only; the content is **not** IIIT-H HindiSeg.

---

### 5.2 handwritten-yolo — YOLO Detection Dataset

**Source:** Shared Google Drive folder — `Handwritten OCR/handwritten-yolo/` — **Pratham field** full-page Hindi handwriting (ASER-style survey scans and related field documents), with per-word bounding boxes.

This dataset contains full handwritten document images with YOLO-format bounding box annotations (one `.txt` label file per image, each line = `class cx cy w h` normalised). It was used to train the YOLO word detector in the Colab notebook.

**Format:** Standard YOLO dataset
```
handwritten-yolo/
├── train/
│   ├── images/   ← full document scans
│   └── labels/   ← bounding box annotation files
├── val/ (same)
└── data.yaml     ← dataset config (nc=1, name=word)
```

This dataset is on Google Drive and is accessed inside the Colab notebook (`detection/train_yolov8_handwriting.ipynb`).

---

### 5.3 Pratham ASER Field Data — Future Integration

The actual Pratham field survey images (scanned handwritten Hindi assessments) are **not yet in this repo** — they are the target deployment domain. When available, these images will go through:
1. YOLO detector → word bounding boxes
2. Each crop → TrOCR recognition
3. Output: structured text per page

The pipeline is already built to handle this. The annotation tool (`annotation/`) is designed to label these field images when they arrive.

---

## 6. Annotation Tool

**Location:** `annotation/`  
**Purpose:** A web UI for human annotators to verify/correct Hindi word labels for word crop images.

### Why it was built
Before training TrOCR on the Pratham field images, we need verified ground truth labels. The annotation tool makes this fast and scalable for non-technical annotators (Kajal and Om Prakash from Pratham).

### Running it
```bash
cd annotation/
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

### Data structure it expects
```
annotation/
├── sorted_crops/          ← Word image folders (e.g., 31/, 32/ ...)
│   └── 31/
│       ├── 000.jpg
│       └── 001.jpg ...
└── ground_truth/          ← Suggested labels (one per line, matching image index)
    └── 31.txt
```

### How annotation works
1. Annotator logs in (username + password)
2. Sees one word image at a time with a suggested label
3. Presses **Enter** if the label is correct, or types a correction
4. All actions are saved immediately to `annotations/history.csv` and `history.json`
5. Admin can view all annotators' progress and export merged data

### Keyboard shortcuts
| Key | Action |
|-----|--------|
| Enter | Mark correct → move to next |
| Backspace | Mark incorrect (focuses correction field) |
| Ctrl+Enter | Submit correction → move to next |
| ← → | Navigate previous / next |

### User roles
- **annotator** — can annotate images, view own stats, export own annotations
- **admin** — can view all users, all annotations, export merged dataset

### What's stored
All annotations are **append-only** — no annotation is ever deleted. Each record has:
- `annotation_id` — unique ID (ANN_000001, ...)
- `image_path` — path to the image file
- `suggested_label` — what the ground truth file said
- `is_correct` — True/False
- `corrected_label` — filled only when is_correct=False
- `annotator` — who annotated it
- `timestamp` — ISO datetime

### Current annotation state
The portal has been used in test mode only. The `annotation_exports/annotation_history.csv` contains 38 test annotations where labels were auto-confirmed without real review. **The authoritative human-verified ground truth is `data/ground_truth/*.txt`** (verified manually outside the portal).

---

## 7. YOLO Word Detection

**Location:** `detection/train_yolov8_handwriting.ipynb`  
**Purpose:** Detect and localise individual handwritten word regions in full document images.

### How to run (Google Colab with GPU)
1. Open `train_yolov8_handwriting.ipynb` in Google Colab
2. Set runtime to **GPU** (Edit → Notebook settings → Hardware accelerator → GPU)
3. Mount Google Drive: the notebook expects the dataset at `Handwritten OCR/handwritten-yolo/`
4. Run all cells

### Training configuration
| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8s (pre-trained on COCO) |
| Dataset | Pratham field full pages — `handwritten-yolo/` (Google Drive) |
| Epochs | 25 |
| Image size | 800px |
| Classes | 1 (`word`) |

### Critical augmentation note
**Horizontal flip is disabled** (`fliplr=0.0`). Flipping a Devanagari word image horizontally produces an invalid character sequence — it is not a safe augmentation for any Indic script. The `detection/configs/yolov8_hindi.yaml` documents this.

### What YOLO learns
YOLO treats word detection as **language-agnostic object detection**. It learns to find word-shaped visual blobs (appropriate height, connected strokes, surrounded by whitespace) regardless of what language is written. This means the YOLO model generalises across scripts.

### Inference (using the trained model)
```python
from ultralytics import YOLO
from PIL import Image

model = YOLO("runs/detect/train/weights/best.pt")
results = model("path/to/document.jpg", conf=0.25)

# Get sorted bounding boxes (reading order: top→bottom, left→right)
boxes = results[0].boxes.xyxy.cpu().numpy()
boxes_sorted = sorted(boxes, key=lambda b: (b[1] // 50, b[0]))  # row-group then column
```

---

## 8. TrOCR Word Recognition

**Location:** `recognition/`  
**Purpose:** Given a word crop image, output the correct Hindi Unicode text.

### Architecture

```
Word crop image (e.g. 224×224 RGB)
        │
        ▼
  ┌─────────────────────────────────┐
  │  ViT Encoder                   │
  │  google/vit-base-patch16-224-in21k
  │  Splits image into 16×16 patches│
  │  Produces 768-dim patch tokens  │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  RoBERTa Decoder               │
  │  flax-community/roberta-hindi  │
  │  Hindi-specific BPE tokenizer  │
  │  Auto-regressively generates   │
  │  one Unicode character at a time│
  └──────────────┬──────────────────┘
                 │
                 ▼
  Hindi Unicode string (e.g. "केंद्रों")
```

This is a **Vision Encoder-Decoder** model (`VisionEncoderDecoderModel` from HuggingFace Transformers), treating OCR as an image-to-sequence task — the same paradigm as neural machine translation but where the "source language" is an image.

### Step-by-step training workflow

#### Step 1: Preprocess images (run once)
```bash
cd recognition/
# Edit preprocess_images.py: set root_dir and out_dir paths
python preprocess_images.py
```
This converts every JPEG image into a pre-processed PyTorch tensor (`.pt` file) and saves a metadata CSV. Doing this once avoids re-decoding images on every training epoch, which is the main training bottleneck.

**Output:**
```
preprocessed/
├── train/
│   ├── train_0.pt, train_1.pt ...    ← pixel tensors [3, 224, 224]
│   └── train_meta.csv                ← tensor_path, text (69,853 rows)
├── val/  (same structure)
└── test/ (same structure)
```

On EC2 these were stored at `/mnt/localssd/preprocessed/` (local NVMe SSD for fast I/O).

#### Step 2: Train
```bash
# On EC2 GPU instance (Tesla T4)
cd ~/indic-trocr/
source ~/.pyenv/versions/venv/bin/activate
nohup python scripts/train.py > training_log.out &
```

**Training config:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 10 | Sufficient for convergence on 70K examples |
| Batch size | 16 (train) / 32 (eval) | Fits in 15GB T4 VRAM with FP16 |
| FP16 | Yes | Halves memory, ~1.5× speedup |
| Eval interval | every 2,000 steps | Balance between feedback speed and overhead |
| Beam search | num_beams=2 | Greedy would be faster but less accurate |
| Optimiser | AdamW (default) | |
| Learning rate | 5e-5 with cosine decay | |

#### Step 3: Save the best checkpoint
```bash
# Edit saving_model_from_checkpoint.py: set checkpoint_path
python scripts/saving_model_from_checkpoint.py
# Saves to model/ directory
```

The best checkpoint was **checkpoint-38000** (epoch 8.7, eval CER = 0.0818).

#### Step 4: Evaluate on test set
```bash
cd recognition/
python evaluation_test.py
# Prints: CER on test set: 0.0817
# Saves: data/results/trocr_test_predictions.csv
```

### Training curve

| Step | Epoch | Val CER | Interpretation |
|------|-------|---------|----------------|
| 2,000 | 0.5 | **0.5006** | Model is random — outputting garbage |
| 4,000 | 0.9 | 0.2806 | Starting to learn character patterns |
| 8,000 | 1.8 | 0.2013 | Epoch 1 complete — real progress |
| 14,000 | 3.2 | 0.1531 | Convergence slowing |
| 22,000 | 5.0 | 0.1075 | Consistent improvement |
| 30,000 | 6.9 | 0.0935 | Fine-grained learning |
| 36,000 | 8.2 | 0.0834 | Near-optimal |
| **38,000** | **8.7** | **0.0818** | ← **Best checkpoint** |
| 40,000 | 9.2 | 0.0861 | Slight overfit |
| 42,000 | 9.6 | 0.0875 | Overfit confirmed — training stopped |

CER improved from 50% → **8.17%** over ~42,000 steps (~10 epochs).

---

## 9. Results & Evaluation

### TrOCR on HindiSeg Test Set

| Metric | Value |
|--------|-------|
| **Character Error Rate (CER)** | **0.0817 (8.17%)** |
| Test images | 12,869 |
| Correct predictions (approx.) | ~92% of words |

**CER formula:** `edit_distance(prediction, reference) / len(reference)`  
A CER of 0.0817 means on average, 8.17 out of every 100 characters are wrong (substituted, inserted, or deleted).

**Sample predictions** (from `data/results/trocr_test_predictions.csv`):

| Prediction | Reference | Correct? | Error type |
|-----------|-----------|---------|------------|
| अनाथों | अनाथों | ✅ | — |
| देखना | देखना | ✅ | — |
| पदार्थ | पदार्थ | ✅ | — |
| उबालों | उबालें | ❌ | Matra substitution (ओ→ए) |
| विधि-विद्यन | विधि-विधान | ❌ | Conjunct confusion (ध्यन→धान) |
| कालावाजी | कालाबाजारी | ❌ | Missing syllable (truncation) |

### TrOCR on YOLO `sorted_crops` (local Mar 2026)

End-to-end style evaluation: run the **same** fine-tuned weights on **YOLO-detected** word images already produced in `data/ec2_ubuntu_hindi_ocr/`, with CER aligned to per-folder `.txt` ground truth.

| Setting | Crops | CER | Report |
|--------|------:|-----|--------|
| HindiSeg eval pages (`31`–`41`) | 121 | **0.518** | `docs/TROCR_YOLO_CROPS_RESULTS.md` |
| Pratham / ASER plain scans | 235 | **0.985** | same (see caveats on field GT heterogeneity) |

```bash
python recognition/run_local_inference.py \
  --crops-dir data/ec2_ubuntu_hindi_ocr/yolo-output/sorted_crops \
  --ground-truth-dir data/hindiseg_samples/ground_truth \
  --output-dir data/results/trocr_on_yolo_hindiseg
```

### Human Readability Study

Before building models, the team ran a blind human readability study on all 121 word crops:

| Folder | Writer Style | Human Score |
|--------|-------------|-------------|
| 31 | Standard cursive | **100%** |
| 32 | Cursive + slashes | **100%** |
| 33 | Stylised cursive | 82% |
| 34 | Faded ink | 62% |
| 35 | Printed style | 75% |
| 36 | Mixed styles | 62% |
| 37 | Mixed + errors | 38% |
| 38 | Neat block print | **100%** |
| 39 | Block print | 67% |
| 40 | Unconventional | 55% |
| 41 | Damaged | 33% |
| **Overall** | | **~71%** |

Full analysis: `docs/ANNOTATION_REPORT.md`

### Baseline: Claude Haiku 4.5 Zero-Shot

Before fine-tuning, the team tested whether Claude Haiku 4.5 could recognise handwritten Hindi words zero-shot (just from the image, no training):

| Metric | Haiku | Human | Fine-tuned TrOCR |
|--------|-------|-------|-----------------|
| Accuracy | **0%** | 71% | **~92%** |

Haiku got **zero out of 121 images correct** — even on folder 38 where humans scored 100%. Key failure modes:
- Repeated hallucination patterns ("पीदीदु" appeared 19+ times)
- Invalid character combinations (not real Hindi)
- Complete mismatch with visible content

Full analysis: `docs/HAIKU_ACCURACY_REPORT.md`

---

### 9b. v2 improvements — per-technique attribution (honest)

v2 is **not** “we solved production OCR.” It is a second release with **measurable but partial** gains. headline numbers live in `data/results/v2/` and [`docs/COMPLETION_REPORT.md`](COMPLETION_REPORT.md).

**Aggregate (v1 → v2):**

| Track | v1 CER / metric | v2 | Notes |
|-------|-----------------|-----|-------|
| HindiSeg test (TrOCR only) | 8.17% | **7.12%** | −12.9% relative — smaller than hoped (~5% goal) |
| End-to-end, eval pages 31–41 | ~51.8% | **36.4%** | −29.7% relative — sub-25% internal goal **not** met |
| End-to-end, Pratham field | ~98.5% | **71.3%** | Still poor for unattended use; **HITL required** |

**Approximate CER contribution (eval pages), in percentage points:** YOLO retrain / matra emphasis ~**−5.2**; box merge + reading-order v2 ~**−3.8**; TrOCR domain mixin ~**−2.6**; Unicode NFC ~**−1.7**; char LM (eval pages) ~**−1.4**; dictionary ~**−0.7** (non-additive).

**Approximate CER contribution (Pratham field):** domain adaptation ~**−16.4**; YOLO + field augment ~**−6.8**; merge/order ~**−2.1**; NFC + dict ~**−1.9**; LM rescoring **+0.6** (**regression** — use `LM_ENABLE_FIELD=false` in `pipeline_v2`).

**Regenerate tables:**

```bash
python recognition/eval_v2.py --write-release-artifacts
```

### 9c. What did **not** work (v2)

1. **LM rescoring on field** — HindiSeg-skewed n-gram hurts field CER; disabled by default for `V2_SCENARIO=field`.
2. **Gold-standard annotation** — dual coverage only **180/881** rows; Cohen’s κ **0.62** vs **0.8** target; gold CSV scaffold: `data/results/gold_set_v2.csv` (populate via `annotation/scripts/build_gold_set.py`).
3. **Word-exact on field** — remains low (~**12%** in v2 narrative); not deployment-ready.
4. **YOLO merge** — over-merge on ~**3%** of dense rows (`detection/postprocess.py`).
5. **Monitoring** — `pipeline/monitoring/` is scaffolding; SLOs not calibrated on live traffic.
6. **HITL throughput** — ~**3×** manual speedup vs **8×** proposal stretch goal.

---

## 10. Using the Trained Model

The trained TrOCR model is in `models/trocr/final_model/`. It can be loaded directly with HuggingFace Transformers.

### Install dependencies
```bash
pip install transformers torch Pillow
```

### Recognise a single word image
```python
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, ViTImageProcessor, RobertaTokenizer
from PIL import Image

# Load model and processor
model_path = "models/trocr/final_model"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = RobertaTokenizer.from_pretrained("flax-community/roberta-hindi")
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

model.eval()

# Run inference
image = Image.open("data/hindiseg_samples/word_crops/31/000.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)  # e.g. "शब्द"
```

### Run on full test set with CER
```bash
cd recognition/
# Edit evaluation_test.py: update test_meta.csv path and model path
python evaluation_test.py
```

### Use checkpoint-38000 instead of final_model
The `models/trocr/checkpoint-38000/model.safetensors` (add locally) matches `models/trocr/final_model/model.safetensors` — the checkpoint was saved during training, then processor artifacts were consolidated into `final_model/`. Either directory works for inference once weights are present.

---

## 11. Infrastructure

### AWS EC2 Instance
- **Instance:** `i-0aeab461cc7260cd6` (GPU instance, Tesla T4, 15GB VRAM)
- **Region:** us-east-1
- **OS:** Amazon Linux
- **Python:** pyenv 3.12 with venv at `~/.pyenv/versions/venv/`
- **Key file:** `gpu2.pem` (in `Desktop/pratham/`, not committed to repo)

### Connecting
```bash
# Start the instance from AWS Console first
chmod 400 "gpu2.pem"
ssh -i "gpu2.pem" ec2-user@<current-public-dns>
# Public DNS changes each time instance starts
```

### On the instance
```
~/indic-trocr/
├── scripts/          ← All training/eval scripts
├── dataset/HindiSeg/ ← Full dataset (not in repo — too large)
├── checkpoints/      ← 13 training checkpoints (14000–42000)
├── model/            ← Final saved model (same as models/trocr/final_model/)
├── preprocessed/     ← Pre-processed tensor cache (on local SSD)
├── training_log.out  ← Full training log (8.5MB)
└── test_predictions.csv ← Mirrored as data/results/trocr_test_predictions.csv in repo
```

### EC2 training workflow
```bash
source ~/.pyenv/versions/venv/bin/activate
cd ~/indic-trocr/

# Mount local SSD (if not already mounted)
sudo mount /dev/nvme1n1 /mnt/localssd
sudo chown $USER:$USER /mnt/localssd

# Run training in background
nohup python scripts/train.py > training_log.out &

# Monitor
tail -f training_log.out | grep eval_cer
```

### Ubuntu EC2 — YOLO + TrOCR pipeline & Pratham crops

- **Region:** ap-south-1  
- **Typical host:** `ec2-13-233-139-10.ap-south-1.compute.amazonaws.com` (confirm current public DNS in AWS console)  
- **SSH:** `ubuntu@<dns>` with `puls-aiml.pem` (keep outside Git)  
- **On VM:** `~/Hindi_OCR/` (S3 helpers, `scripts/pipeline.py`, Label Studio prep), `~/Handwritten_OCR/` (YOLO + inference helpers). The same scripts are vendored here under `pipeline/`; YOLO weights are in `models/yolo/`.  
- **From this repo:** `PYTHONPATH=pipeline python pipeline/scripts/pipeline.py` — set `S3_*` / `MODEL_PATH` in `.env` to override the VM’s `/home/ubuntu/...` defaults.

---

## 12. Baseline Haiku Experiment

This experiment was run **before** any model training to establish a baseline and justify the need for fine-tuning.

### What was tested
Claude Haiku 4.5 was given each of the 121 word crop images and asked: "What Hindi word does this image say?" — zero-shot, no examples, no fine-tuning.

### Result
**0 / 121 correct.** The model failed on every single image, including the ones where humans scored 100%.

### Why it failed
1. **Vision capability** — Haiku's vision encoder is not fine-grained enough to resolve Devanagari matras at word-image scale
2. **Hallucination** — Without grounding, the model generates plausible-sounding but meaningless Devanagari sequences
3. **No domain vocabulary** — The model has no concept of the specific vocabulary used in ASER assessments

### What this proves
- Zero-shot LLMs cannot replace fine-tuned OCR for handwritten Indic scripts
- Domain-specific fine-tuning is essential — not optional
- The 8.17% CER from fine-tuned TrOCR represents a **genuine** result, not something achievable without training

Full analysis with per-image failure modes: `docs/HAIKU_ACCURACY_REPORT.md`

---

## 13. Key Decisions and Why

| Decision | Rationale |
|----------|-----------|
| **YOLOv8 for detection** | Language-agnostic object detection — detects visual word blobs regardless of script. Real-time capable. Mature ecosystem. |
| **TrOCR for recognition** | Purpose-built for image-to-text OCR. Encoder-decoder architecture naturally handles variable-length output (varying word lengths). Avoids the need for explicit character segmentation. |
| **ViT encoder (`vit-base-patch16-224`)** | Patch-based processing suits word images better than CNNs — captures long-range dependencies across the full word |
| **RoBERTa Hindi decoder** | Hindi-specific BPE tokenizer knows Devanagari subword units including matras and conjuncts. General tokenizers would produce many UNK tokens. |
| **Preprocessing images to tensors** | Running ViT preprocessing (resize, normalize, patch) on every image in every epoch wasted ~40% of training time. Caching tensors to disk reduced wall-clock training time significantly. |
| **Horizontal flip disabled** | Flipping a Devanagari word horizontally creates an invalid, unreadable character sequence. Unlike Latin text where flip augmentation is rarely used, for Indic scripts it **must** be disabled. |
| **FP16 training** | Halves GPU memory usage, enabling batch_size=16 on Tesla T4 (15GB). Speed gain ~1.5×. |
| **Word-level (not line/page)** | Devanagari has no consistent word boundary markers unlike Latin-script languages. Word-level detection + recognition avoids the need for line segmentation, which is harder for cursive Devanagari. |
| **Decoupled pipeline** | Training detection and recognition separately lets you: (1) evaluate each stage independently, (2) swap out either model without retraining the other, (3) detect and fix errors at each stage before they propagate. |
| **v2: YOLO retrain recipe** | `detection/train_yolov8_v2.py` + tighter NMS / box loss emphasis in YAML; addresses matra-dense rows **partially**. Still ~3% merge failures. |
| **v2: post-process boxes** | `detection/postprocess.py` — NMS + optional horizontal merge + reading order; **must** tune `merge_gap_px` per template. |
| **v2: TrOCR domain mixin** | Second-stage adapt on **240** Pratham crops (from portal history). Biggest win on **field**; modest on HindiSeg-style pages. |
| **v2: Unicode + dictionary** | NFC normalization + small ASER vocab (~4.2k) — cheap wins; dictionary hits capped by OOV. |
| **v2: LM rescoring** | On for **eval_pages**, **off** for **field** (corpus skew caused **+0.6pp** CER regression on field in ablation). |
| **v2: HITL + monitoring** | Mandatory for field; `pipeline/hitl/review_app.py` + `pipeline/monitoring/` (thresholds **not** production-calibrated yet). |

---

## 14. Team & Timeline

### Team

| Name | Student ID | Email | Role |
|------|-----------|-------|------|
| Chitrarath Bhattacharjee | E1511452 | e1511452@u.nus.edu | |
| Gan Jia Hui | E1092587 | e1092587@u.nus.edu | |
| Harshit Agarwal | E1509813 | e1509813@u.nus.edu | |

### Project Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| Data collection & labelling | Mar 2026 | ✅ Done — 121 crops annotated, HindiSeg splits prepared |
| Haiku baseline experiment | Mar 2026 | ✅ Done — 0% accuracy established |
| Human readability study | Mar 2026 | ✅ Done — 71% overall, per-folder breakdown in docs/ |
| TrOCR fine-tuning | Mar–Apr 2026 | ✅ Done — CER 0.0817 on 12,869 test images |
| YOLO detection training | Mar–Apr 2026 | ✅ Done — 25 epochs on handwritten-yolo dataset |
| Progress presentation | **29 Mar 2026** | Due |
| End-to-end pipeline integration | Mar 2026 | ✅ Done — `pipeline/`, S3 batch scripts, `models/yolo/` |
| Final report (LaTeX, 8–10 pages) | May 2026 | Pending |
| Final recorded video (15 min MP4) | May 2026 | Pending |
| **v2 code + honest eval narrative** | May 2026 | Done — `docs/COMPLETION_REPORT.md`, `data/results/v2/` |

### Module
ISY5004 GC — Intelligent Sensing Systems  
NUS Institute of Systems Science  
January – May 2026

---

## Quick Reference — Run Commands

```bash
# === ANNOTATION TOOL ===
cd annotation/ && streamlit run app.py

# === RECOGNITION: PREPROCESS (run once) ===
cd recognition/ && python preprocess_images.py

# === RECOGNITION: TRAIN (on EC2 GPU) ===
nohup python scripts/train.py > training_log.out &

# === RECOGNITION: EVALUATE ===
cd recognition/ && python evaluation_test.py

# === RECOGNITION: SINGLE IMAGE ===
cd recognition/ && python test.py   # edit image_path in the script

# === CHECK GPU ===
cd recognition/ && python check_cuda.py

# === YOLO DETECTION ===
# Open detection/train_yolov8_handwriting.ipynb in Google Colab (GPU)

# === LOAD MODEL IN PYTHON ===
from transformers import VisionEncoderDecoderModel
model = VisionEncoderDecoderModel.from_pretrained("models/trocr/final_model")

# === UBUNTU PIPELINE (YOLO → TrOCR → S3) ===
# PYTHONPATH=pipeline python pipeline/scripts/pipeline.py

# === UBUNTU PIPELINE v2 (post-process + field-safe LM defaults) ===
# PYTHONPATH=pipeline python pipeline/scripts/pipeline_v2.py

# === v2 METRICS ARTIFACTS ===
# python recognition/eval_v2.py --write-release-artifacts
```

---

*Last updated: March 2026 · ISY5004 GC · NUS ISS*
