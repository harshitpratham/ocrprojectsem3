# Dataset Card — Hindi Handwritten OCR Pipeline
### ISY5004 · NUS ISS · 2026

---

## 0. Ubuntu EC2 dataset mirror (`data/ec2_ubuntu_hindi_ocr/`)

Complete copy of **`~/Hindi_OCR/dataset/`** from the Ubuntu pipeline VM (YOLO + TrOCR batch workflow). Tracked in Git so **GitHub and local clones match the server layout**.

| Path | Contents |
|------|----------|
| `plain-dataset/images/` | Full-page / scan **JPEG** inputs for YOLO (~129 files) |
| `plain-dataset/ground_truth/` | Per-document Hindi **.txt** labels (same material as `data/pratham_field/ground_truth/`) |
| `plain-dataset/yolo-output/` | YOLO-derived **sorted_crops** and related outputs |
| `yolo-input/` | Small auxiliary YOLO input set |
| `yolo-output/sorted_crops/` | **Pratham field** evaluation folders **31–41** word crops (121 images) |
| `ocr_outputs/` | Sample CSV / `transcripts/` from batched TrOCR |
| `plain_mapped.zip` | Packaged plain dataset from the VM |
| `tasks_all.json` | Label-studio style task list (duplicate of `data/pratham_field/label_studio_tasks.json` naming may differ) |
| `ground-truth/` | `31.txt`–`41.txt` lines for Pratham field evaluation crops |

`__MACOSX` junk from the VM is excluded from the mirror.

---

## 1. Pratham field evaluation word crops (folders 31–41)

### Source
**Pratham International** field handwriting from ASER-style surveys: **121 word-level crops** organised in folders **31–41** (11 handwriting groups) to span real-world quality — from neat print to degraded or unconventional styles. Paths under `data/hindiseg_samples/` are a legacy layout name; the images are **not** IIIT-H HindiSeg. Full-page scans, plain-document mirrors, and related ground-truth exports also appear under `data/pratham_field/` and `data/ec2_ubuntu_hindi_ocr/`.

### Word image crops (`data/hindiseg_samples/word_crops/`, symlink `data/word_crops/`)
| Folder | Images | Notes |
|--------|--------|-------|
| 31 | 13 | Mixed handwriting styles — standard to cursive |
| 32 | 11 | Cursive-heavy writing |
| 33 | 11 | Stylized text |
| 34 | 13 | Faded ink, degraded quality |
| 35 | 12 | Printed-style handwriting |
| 36 | 13 | Mixed styles |
| 37 | 15 | Mixed + annotation errors |
| 38 | 13 | Best quality (cleanest samples) |
| 39 | 3 | Block print style |
| 40 | 11 | Unconventional handwriting |
| 41 | 6 | Damaged / partially visible |
| **Total** | **121** | |

### Ground truth labels (`data/hindiseg_samples/ground_truth/`, symlink `data/ground_truth/`)
One `.txt` file per folder (`31.txt` → `41.txt`). Each line = the Hindi Unicode label for the corresponding image (by index order).

### Annotation Exports (`data/annotation_exports/`)
Produced by the Streamlit annotation portal (`annotation/`):

| File | Rows | Description |
|------|------|-------------|
| `annotation_history.csv` | 38 | Portal export (annotator: testannotate) |
| `annotation_history_local.csv` | 38 | Copy of the same local export |
| `annotation_history_ubuntu.csv` | 881 | Full history from Ubuntu deployment (annotator e.g. testAT) |
| `annotation_history.json` | 38 | JSON for local export |
| `users.json` | 2 | Registered users: `testannotate` (annotator), `admin` |
| `testuser_annotations.csv` | 18 | Earlier test-run annotations (annotator: testuser) |

**Note:** The `history.csv` annotations (rows 1–38) are a test run where labels were auto-confirmed without real review. The authoritative ground truth is `annotations_all_images.csv` (see below) which contains human-verified Hindi transcriptions for all 121 images.

### Full annotations (`data/results/annotations_all_images.csv`)
121 rows — complete annotation of all evaluation word crops:
- `image_path` — full path to the crop image
- `directory` — folder number (31–41)
- `filename` — image filename (e.g. `000.jpg`)
- `hindi_text` — actual Hindi Unicode text (human-annotated)
- `transliteration` — romanised phonetic transcription
- `notes` — quality notes (e.g. "Earth", "Poetic line", "Unclear")

---

## 2. HindiSeg Dataset (TrOCR Training)

### Source
[IIIT-H Indic Handwriting Dataset](http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)
- Collected from multiple writers (identified by writer-id folders)
- Word-level segmented images of handwritten Hindi text
- Images stored under `HindiSeg/train/`, `HindiSeg/val/`, `HindiSeg/test/`

### Split summary (`data/training_splits/`)
| Split | Examples | File |
|-------|----------|------|
| Train | 69,853 | `train.txt` |
| Validation | 12,708 | `val.txt` |
| Test | 12,869 | `test.txt` |
| **Total** | **95,430** | |

### Vocabulary
| File | Contents |
|------|----------|
| `hindi_vocab.txt` | 11,029 unique Hindi words in the dataset |
| `lexicon.txt` | Lexicon used during test-set evaluation |
| `Readme.txt` | Dataset structure documentation (from IIIT-H) |

### Format (`train.txt` / `val.txt` / `test.txt`)
```
<relative_image_path> <hindi_word>
```
Example:
```
HindiSeg/train/8/251/21.jpg केंद्रों
HindiSeg/train/10/207/8.jpg पॉवर
```

### Preprocessed Tensor Metadata (`*_meta.csv`)
Generated by `recognition/preprocess_images.py`. Each row maps a `.pt` tensor file to its text label:
- `train_meta.csv` — 69,853 rows (tensor → label mapping for training)
- `test_meta.csv` — 12,869 rows
- `val_meta.csv` — 12,708 rows

Tensors were stored at `/mnt/localssd/preprocessed/` on the EC2 instance (local SSD for fast I/O during training).

---

## 3. TrOCR Model

### Architecture
| Component | Model |
|-----------|-------|
| Encoder | `google/vit-base-patch16-224-in21k` (ViT) |
| Decoder | `flax-community/roberta-hindi` (RoBERTa) |
| Framework | HuggingFace `VisionEncoderDecoderModel` |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 10 (ran ~9.62 before stopping) |
| Batch size | 16 (train) / 32 (eval) |
| FP16 | Yes |
| Optimiser | AdamW (default) |
| Learning rate | 5e-5 with cosine decay |
| Warmup | 3 epochs |
| Eval & save interval | every 2,000 steps |
| Total training steps | 42,000 |
| GPU | Tesla T4 (AWS EC2 g4dn) |

### Training Curve (eval CER on validation set)

| Step | Epoch | Eval CER | Eval Loss |
|------|-------|----------|-----------|
| 2,000 | 0.46 | 0.5006 | 0.9921 |
| 4,000 | 0.92 | 0.2806 | 0.5144 |
| 6,000 | 1.37 | 0.2114 | 0.3746 |
| 8,000 | 1.83 | 0.2013 | 0.3493 |
| 10,000 | 2.29 | 0.1513 | 0.2698 |
| 14,000 | 3.20 | 0.1307 | 0.2463 |
| 18,000 | 4.12 | 0.1157 | 0.2313 |
| 22,000 | 5.03 | 0.1112 | 0.2264 |
| 26,000 | 5.95 | 0.1029 | 0.2188 |
| 30,000 | 6.87 | 0.0989 | 0.2127 |
| 34,000 | 7.79 | 0.0947 | 0.2189 |
| 36,000 | 8.25 | 0.0834 | 0.1969 |
| **38,000** | **8.70** | **0.0818** | **0.1919** ← best |
| 40,000 | 9.16 | 0.0861 | 0.2013 |
| 42,000 | 9.62 | 0.0875 | 0.2085 |

**Best checkpoint:** `checkpoint-38000` → eval CER = **0.0818**

### Final model files (`models/trocr/final_model/`)
Config + tokenizer files in Git; weights (`model.safetensors`, ~900MB+) stay on EC2 or are added locally (see `models/MODEL_STORAGE.md`):

| File | Purpose |
|------|---------|
| `config.json` | Model architecture config |
| `generation_config.json` | Beam search / generation params |
| `preprocessor_config.json` | ViT image processor config |
| `tokenizer_config.json` | RoBERTa tokenizer config |
| `vocab.json` | BPE vocabulary |
| `merges.txt` | BPE merge rules |
| `special_tokens_map.json` | Special tokens (CLS, SEP, PAD, etc.) |

> **Model weights** are excluded from Git (`.gitignore`). Copy `model.safetensors` next to these configs for inference.

---

## 3b. Pratham / ASER-style field transcripts (`data/pratham_field/`)

- **`ground_truth/`** — Per-student or per-sample Hindi reference `.txt` files from plain-document processing (128 files in the Ubuntu mirror; filenames match scans).
- **`label_studio_tasks.json`** — Label Studio task export used for review (from `Hindi_OCR/tasks_all.json` on the VM).

Document scans for these are large; keep on S3 / EC2 and sync as needed.

---

## 4. Test results (`data/results/trocr_test_predictions.csv`)

- **12,869 rows** of `prediction, reference` pairs
- Produced by `recognition/evaluation_test.py` using `checkpoint-38000`
- **Test CER: 0.0817** (8.17%)

---

## 5. Baseline comparison (`data/results/haiku_vs_ground_truth.csv`)

Zero-shot evaluation of Claude Haiku 4.5 on the 121 Pratham field evaluation word crops:
- **Accuracy: 0%** (0/121 correct)
- Compared against human annotation accuracy of ~71%
- Full analysis: `docs/HAIKU_ACCURACY_REPORT.md`

---

## 6. Training log (`data/results/trainer_state_final.json`)

Full HuggingFace `Seq2SeqTrainer` state from the final checkpoint (step 42,000). Contains:
- Complete `log_history` array (all training/eval steps with loss, CER, learning rate)
- Total training steps, epoch count
- Runtime metadata

---

## 7. v2 release metrics (`data/results/v2/`)

Machine-readable tables for the **honest v2 narrative** (regenerate with `python recognition/eval_v2.py --write-release-artifacts`):

| File | Purpose |
|------|---------|
| `recognition_hindiseg_v2.json` | HindiSeg test CER v1 vs v2 |
| `end_to_end_eval_pages.csv` | Per-doc CER (folders 31–41) v1 vs v2 |
| `end_to_end_pratham_field.csv` | Illustrative per-doc field CER v1 vs v2 |
| `per_stage_attribution.json` | Per-technique pp estimates |
| `yolo_v2_metrics.json` | mAP / failure modes |
| `annotation_kappa.json` | Dual-annotation κ + gold-set shortfall |
| `lm_rescore_ablation.csv` | LM on eval vs regression on field |

**Gold set (scaffold):** `data/results/gold_set_v2.csv` — populate from `annotation/scripts/build_gold_set.py` once dual labels exist.

**HITL queue (scaffold):** `data/results/v2/hitl_queue_sample.csv`

---

*Data card last updated: March 2026*
