# Detailed evaluation report: fine-tuned TrOCR on YOLO word crops

**Scope:** This document reports **pipeline-level** recognition performance when the project’s **fine-tuned TrOCR** checkpoint (`models/trocr/final_model/`) is applied to **YOLO-produced word crops** under `data/ec2_ubuntu_hindi_ocr/`. It explicitly contrasts these results with the well-known **~8.17% CER** figure from the **HindiSeg pre-segmented test set** (public-style evaluation), which is **not** the same experimental setup.

**Run date:** 2026-03-27 (CPU inference, batch size 8).  
**Tooling:** `recognition/run_local_inference.py`, CER via Hugging Face `evaluate` (`cer` metric, `jiwer` backend).

---

## 1. Executive summary

| Evaluation | Images | Ground truth | **Character Error Rate (CER)** | Word-exact match (string equality) |
|------------|-------:|--------------|-------------------------------:|-----------------------------------:|
| **YOLO crops — HindiSeg eval pages** (folders `31`–`41`) | 121 | `data/hindiseg_samples/ground_truth/*.txt` | **0.517838** (~51.8%) | **13.2%** (16 / 121) |
| **YOLO crops — Pratham / ASER plain scans** | 235 | `data/pratham_field/ground_truth/*.txt` | **0.985372** (~98.5%) | **1.3%** (3 / 235) |
| **Reference — HindiSeg test (pre-segmented tensors, not YOLO)** | 12,869+ | Official split / tensors | **~0.0817** (~8.2%) | — (see `GUIDE.md`, EC2 logs) |

**Takeaway:** The **~8% CER** number reflects **ideal word images** (dataset crops / preprocessed tensors) from the **HindiSeg recognition benchmark**. Running the **same weights** on **real YOLO boxes** on scanned evaluation worksheets yields roughly **52% CER** on the HindiSeg-style pages. On **field Pratham scans** with the current GT and detection coverage, aggregate CER approaches **99%**, dominated by domain shift, **label heterogeneity**, and **partial detection** relative to reference lists—not by a single trivial bug in the scorer.

Artifacts: `data/results/trocr_on_yolo_hindiseg/`, `data/results/trocr_on_pratham_crops/` — including **`recognition_report.md`** (model vs GT per crop, Correct/Wrong) and **`mismatches_only.csv`** for error-only review.

---

## 2. Problem setting and why the two CERs differ

### 2.1 HindiSeg “8%” setup

- TrOCR is trained on **word-level crops** from **IIIT HindiSeg (HindiSeg / IIIT-HW-Dev)**.
- The reported **8.17% CER** is computed on a **held-out test split** where each input is a **clean, isolated word image** (or equivalent preprocessed tensor), **without** a detector in the loop.
- Vocabulary and imaging statistics match (or closely follow) the training distribution.

### 2.2 YOLO crop setup (this report)

- **Detector:** YOLO places bounding boxes on **full-page scans** (`yolo-input` / plain-dataset images).
- **Crops:** `sorted_crops/<doc>/<000.jpg, …>` are **whatever the detector + sorting** produced: possible **trim errors**, **merged/split boxes**, **background bleed**, and **reading-order mistakes**.
- **Recognition model:** unchanged fine-tuned TrOCR; only the **input distribution** changes.

Therefore, a large gap between **8%** and **52%** is **expected** whenever detection quality is imperfect. The gap quantifies **end-to-end degradation** from “dataset word image → text” to “scan → YOLO crop → text”.

---

## 3. Methodology

### 3.1 Model and processor

- **Encoder weights id:** `google/vit-base-patch16-224-in21k` (ViT image processor).
- **Decoder tokenizer:** `flax-community/roberta-hindi`.
- **Composite weights:** `VisionEncoderDecoderModel.from_pretrained(models/trocr/final_model, local_files_only=True)` (`model.safetensors` ~914 MB).

### 3.2 Data sources

1. **Set A — HindiSeg evaluation worksheets (YOLO path)**  
   - Crops: `data/ec2_ubuntu_hindi_ocr/yolo-output/sorted_crops/`  
   - 121 `.jpg` files across document ids `31` … `41` (not all folders have the same count; YOLO may drop or split words).

2. **Set B — Pratham field plain scans (YOLO path)**  
   - Crops: `data/ec2_ubuntu_hindi_ocr/plain-dataset/yolo-output/sorted_crops/`  
   - 235 `.jpg` files in 128 student-level subfolders (many documents contribute only **one** detected word).

### 3.3 Reference alignment

- For each crop `…/<doc_key>/<NNN.jpg>`, **reference line** = line index `NNN` (zero-based) of `ground_truth_dir/<doc_key>.txt`.
- **Pratham:** `doc_key` equals the transcript folder name (e.g. `Aditi, Class 9th, Sample 9_0`).
- **No sequence re-ordering** is performed beyond the sort order of crop indices; if YOLO order ≠ line order in the sheet, CER pessimistically reflects that.

### 3.4 Metrics

- **CER:** standard character-level edit distance normalized by reference length (same family as in `recognition/evaluation_test.py`).
- **Word-exact rate:** fraction of rows where `predicted_text.strip() == ground_truth.strip()` (strict string match; no Unicode normalization).

---

## 4. Set A — HindiSeg YOLO crops (121 images)

### 4.1 Aggregate

| Metric | Value |
|--------|------:|
| Total word-crops | 121 |
| **CER** | **0.517838** |
| **Word-exact** | **16 / 121** (13.22%) |

### 4.2 Per-document CER and word accuracy

Sorted by CER (lower is better). These document ids correspond to **distinct student styles** on the same **general evaluation template** (mixed word lists across folders).

| doc | n_crops | CER | exact words | exact % |
|----:|--------:|----:|------------:|--------:|
| 38 | 13 | 0.212 | 3 | 23.1% |
| 35 | 12 | 0.303 | 3 | 25.0% |
| 32 | 11 | 0.386 | 2 | 18.2% |
| 33 | 11 | 0.411 | 1 | 9.1% |
| 31 | 13 | 0.495 | 2 | 15.4% |
| 34 | 13 | 0.525 | 1 | 7.7% |
| 39 | 3 | 0.542 | 0 | 0.0% |
| 36 | 13 | 0.556 | 1 | 7.7% |
| 40 | 11 | 0.696 | 2 | 18.2% |
| 41 | 6 | 0.782 | 1 | 16.7% |
| **37** | **15** | **1.000** | **0** | **0.0%** |

**Note on folder 37:** CER = 1.0 means **very large** character-level disagreement under the standard CER definition (including substitution/insertion cost). Contributing factors can include **severe handwriting**, **noisy crops**, and **possible reference or encoding issues** in the paired GT line (inspect `word_predictions.csv` for `doc_key=37`). Independent of the exact cause, the table shows **high variance by writer**: folder **38** is the easiest (**CER ~0.21**), **37** the worst in this run.

### 4.3 Qualitative error patterns (from `word_predictions.csv`)

Illustrative **exact** and **near-miss** behaviour:

- **Correct:** e.g. `32` → `न्यूट्रॉन` / `न्यूट्रॉन`; `35` → `विद्यार्थियों`, `योजना`, `शिक्षक` match references.
- **Matra / vowel drift:** `शब्द` → `शब्दः`; `दुर्गंध` → `दर्धि`, `दुर्ग`.
- **Science vocabulary fragility:** `क्लोरोफ्लोरो` / compounds often garbled (`लीतिफारी`, `कोर्फोलॉक्स`, …) — consistent with training/eval vocabulary stress.
- **Detector merge/split vs GT:** e.g. folder `32` ground truth may list `ब्रैल` and `पद्धति` on **separate** lines while a crop encodes **both** → model predicts a fused reading (`ब्रेलग्राह्त`) vs split references, inflating CER for that index.

These patterns align with **recognition limits** plus **box-level alignment** issues, not a single failing character class.

---

## 5. Set B — Pratham YOLO crops (235 images)

### 5.1 Aggregate

| Metric | Value |
|--------|------:|
| Total word-crops | 235 |
| **CER** | **0.985372** |
| **Word-exact** | **3 / 235** (1.28%) |

### 5.2 Why CER is so high (interpretation, not excuse)

1. **Domain shift:** Pratham ASER scans differ in **paper quality, pens, layout, and word inventory** from curated HindiSeg training data.
2. **Ground-truth heterogeneity:** Many `data/pratham_field/ground_truth/*.txt` files are **document-specific** (e.g. science terms for one student), while others still mirror the **12-word HindiSeg-style list**. Comparing **field handwriting** to a **template list** where the student wrote different words yields **near-random** character overlap.
3. **Sparse detection:** A large fraction of folders contain **only 1 crop** while the GT file lists **10+ words**; only the **first** line participates in scoring. The remaining reference lines are not evaluated as “missed detections” in the current CSV (they are simply absent from the crop list).
4. **CER above 1.0 (per document, with jiwer):** When predictions are **long noisy strings** and references are **short**, character-level edit cost can exceed reference length for that bucket; some per-doc summaries in exploratory stats exceeded 1.0 for that reason.

### 5.3 Best- and worst-case documents (by per-doc CER, illustrative)

**Among the better buckets** (still poor in absolute terms): e.g. `Vanshika_10_10_0` (single crop), `Dhani, Class 10th, Sample 9_0` (3 crops), etc.  

**Among the worst:** long sequences with high mismatch (e.g. `Anjali9_Sample9_1`, `Gunjan, Class 6th, Sample 4_1`) — consistent with **ordering**, **crop quality**, and **GT mismatch**.

Detailed inspection should always go through `data/results/trocr_on_pratham_crops/word_predictions.csv`.

---

## 6. Relation to full-page LLM / OCR baselines

The project also studied **full-page** reading of the same **yolo-input** images (e.g. Claude Opus 4.6 structured annotations, documented in `docs/OPUS_4_6_OCR_ANALYSIS.md`, and human/Sarvam-style baselines in `docs/SARVAM_AI_OCR_ANALYSIS.md`). Those numbers are **not directly comparable** to the tables here:

- Full-page models see **context, layout, and multiple words** jointly.
- This report evaluates **isolated word crops** after **YOLO**, one crop ↔ one GT line.

A fair system comparison would hold **detection + recognition** fixed or report **joint** metrics. The YOLO+TrOCR numbers here isolate the **recognition head’s behavior on detector output**.

---

## 7. Recommendations

1. **Treat 8% as HindiSeg-recognition-only.** Use **~52% CER** (Set A) as the headline **“with our current YOLO crops”** figure for HindiSeg-style evaluation sheets.
2. **Before trusting Set B CER:** audit `pratham_field/ground_truth` for **per-scan correctness**, normalize **Unicode**, and align **word order** with `sorted_crops` (or re-export GT in crop order).
3. **Improve detection** (recall/precision, sorting) — expected to **lower** recognition CER without retraining TrOCR.
4. **Optional fine-tuning** on Pratham crops (or distillation from full-page labels) if field deployment is the goal.

---

## 8. Reproduction

```bash
cd hindi-ocr-pipeline

# Set A
python recognition/run_local_inference.py \
  --crops-dir data/ec2_ubuntu_hindi_ocr/yolo-output/sorted_crops \
  --ground-truth-dir data/hindiseg_samples/ground_truth \
  --output-dir data/results/trocr_on_yolo_hindiseg

# Set B
python recognition/run_local_inference.py \
  --crops-dir data/ec2_ubuntu_hindi_ocr/plain-dataset/yolo-output/sorted_crops \
  --ground-truth-dir data/pratham_field/ground_truth \
  --output-dir data/results/trocr_on_pratham_crops
```

**Dependencies:** see `requirements.txt` (`transformers`, `torch`, `evaluate`, `jiwer`, `safetensors`, …).

---

## 9. References within the repository

| Document | Role |
|----------|------|
| `docs/TROCR_YOLO_CROPS_RESULTS.md` | Short summary + links |
| `GUIDE.md` | Training history, HindiSeg ~8% CER context |
| `docs/OPUS_4_6_OCR_ANALYSIS.md` | Full-page Opus vs GT word study |
| `docs/SARVAM_AI_OCR_ANALYSIS.md` | Human/commercial-style full-page comparison |
| `data/results/trocr_on_yolo_hindiseg/word_predictions.csv` | Per-crop predictions Set A |
| `data/results/trocr_on_pratham_crops/word_predictions.csv` | Per-crop predictions Set B |

---

## 10. v2 pipeline results (honest follow-up, May 2026)

The **v1** figures in Sections 4–5 are unchanged. **v2** reflects a second project pass: YOLO retrain recipe, box post-processing, TrOCR domain mix-in on **240** Pratham-labelled crops (from portal history), Unicode NFC, optional char LM, small ASER dictionary — documented in [`docs/COMPLETION_REPORT.md`](COMPLETION_REPORT.md) and machine-readable under `data/results/v2/`.

| Evaluation | Images | v1 CER | v2 CER | Met internal stretch goal? |
|------------|-------:|--------|--------|---------------------------|
| Set A — HindiSeg eval pages (YOLO crops 31–41) | 121 | **0.517838** | **0.364** | No (goal &lt; 0.25) |
| Set B — Pratham field (YOLO crops) | 235 (v1 table) | **0.985372** | **0.713** (v2 aggregate; see CSV) | **No** — HITL mandatory |

Per-document v2 table: `data/results/v2/end_to_end_eval_pages.csv`. Regenerate artifacts:

`python recognition/eval_v2.py --write-release-artifacts`

## 11. v2 per-technique ablation (approximate)

Non-additive percentage-point estimates — see `data/results/v2/per_stage_attribution.json`.

**Set A (eval pages):** YOLO / matra emphasis ~**−5.2** pp; merge + reading-order ~**−3.8**; TrOCR domain mixin ~**−2.6**; NFC ~**−1.7**; LM **−1.4**; dictionary ~**−0.7**.

**Set B (field):** domain adaptation ~**−16.4**; YOLO augment ~**−6.8**; merge/order ~**−2.1**; NFC + dict ~**−1.9**; LM **+0.6** (**regression** — disabled for field in `pipeline/scripts/pipeline_v2.py`).

**Failures:** Cohen’s κ **0.62** (target 0.8); word-exact on field ~**12%**; HITL throughput ~**3×** not **8×**; monitoring SLOs not live-calibrated (`pipeline/monitoring/`).

---

*Report generated from committed evaluation artifacts; numerical values match `cer_summary.txt` in each results directory. v2 tables are additionally defined in `data/results/v2/`.*
