# ISY5004 · Hindi Handwritten Document Recognition
## Project Analysis & Scope Document

**Module:** ISY5004 GC — Intelligent Sensing Systems
**Team:** Chitrarath Bhattacharjee · Gan Jia Hui · Harshit Agarwal
**Institution:** NUS Institute of Systems Science
**Period:** January – May 2026

---

## 1. Project Overview

This project builds an **end-to-end visual sensing pipeline** to automatically digitise handwritten Hindi documents collected from Pratham Education Foundation's ASER field surveys. The pipeline chains two vision models: a **YOLO-based word detector** and a **TrOCR-based word recogniser**, converting scanned document images into machine-readable Unicode text.

### Real-World Deployment Context
| Item | Detail |
|------|--------|
| Client | Pratham Education Foundation |
| Use Case | Digitising ASER (Annual Status of Education Report) student assessments |
| Language | Hindi (Devanagari script) |
| Volume | Thousands of handwritten responses |
| Bottleneck | Manual transcription — slow, expensive, unscalable |

---

## 2. Problem Decomposition

The core sensing challenge splits into two sub-problems:

```
Raw Document Image
        │
        ▼
┌───────────────────────┐
│  Sub-problem 1        │
│  WORD LOCALISATION    │  → Detect bounding boxes of handwritten words
│  (YOLO Detection)     │
└───────────┬───────────┘
            │  Cropped word images
            ▼
┌───────────────────────┐
│  Sub-problem 2        │
│  WORD RECOGNITION     │  → Classify each crop as a Hindi Unicode string
│  (TrOCR)             │
└───────────┬───────────┘
            │  Word-level text
            ▼
   Structured Transcription
   (reading-order text output)
```

### Why Existing OCR Fails on This Task
| Challenge | Root Cause |
|-----------|-----------|
| No clear word boundaries | Cursive-adjacent nature of Devanagari |
| Matras (diacritical marks) | Attach above/below base characters — confuse generic detectors |
| Vocabulary mismatch | Pre-trained models not exposed to field-collected educational content |
| General tools (Tesseract, Google Vision) | Not fine-tuned for handwritten Hindi at this domain scale |

---

## 3. Technical Architecture

### 3.1 Pipeline Stages

| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| 1 | Data Collection & Annotation | Raw field images | Labelled dataset (bounding boxes + transcriptions) |
| 2 | YOLO Detection | Document image | Word bounding box coordinates |
| 3 | TrOCR Recognition | Cropped word image | Hindi Unicode string |
| 4 | Pipeline Integration | Full document image | Ordered full-document transcription |

### 3.2 Model Details

#### YOLO (Stage 2 — Word Detection)
- **Base Model:** YOLOv8 (pre-trained on general object detection)
- **Fine-tuning target:** Word-level bounding box detection in handwritten Hindi documents
- **Key insight:** Detection is language-agnostic — the model responds to visual word boundaries, not linguistic content
- **Annotation tool:** Label Studio (rectangular bounding boxes per word)
- **Challenges:** Overlapping matra symbols, variable line spacing

#### TrOCR (Stage 3 — Word Recognition)
- **Base Model:** Microsoft TrOCR (Vision Encoder-Decoder Transformer)
- **Architecture:** Visual encoder (processes word image) + Language decoder (generates Unicode sequence)
- **Fine-tuning target:** Cropped Hindi word images → correct Unicode transcription
- **Challenges:** Matra and nukta character handling, domain vocabulary mismatch
- **Post-processing:** Unicode normalisation for matras and nukta characters

### 3.3 End-to-End Flow (Stage 4)

```
Input: Raw document image
    │
    ├─▶ YOLO Detector
    │       └─▶ Sorted word bounding boxes
    │               (sorted by vertical row, then horizontal column)
    │
    ├─▶ For each bounding box:
    │       └─▶ Crop word region → TrOCR → Hindi Unicode string
    │
    └─▶ Concatenate in reading order → Final transcription
```

---

## 4. Dataset

### 4.1 Primary Dataset
| Attribute | Detail |
|-----------|--------|
| Source | Pratham ASER field surveys |
| Size | 301 scanned document images |
| Paper types | Plain and ruled paper |
| Annotators | Kajal and Om Prakash (Pratham) |
| Annotation tasks | (1) Word-level bounding boxes for YOLO; (2) Word transcription pairs for TrOCR |
| Split | ~70% Train / 15% Validation / 15% Test |

### 4.2 Supplementary Data
- **IIIT-HW-Dev** (IIT-originated Devanagari handwriting dataset) — to be used if primary dataset is insufficient
- Augmentation strategies: rotation, brightness variation, noise injection

### 4.3 Annotation Workflow
```
Label Studio
    ├── Task A: Draw bounding boxes around each word (YOLO labels)
    └── Task B: Verify/correct transcription of each cropped word (TrOCR labels)
```

---

## 5. Evaluation Metrics

| Model | Metric | Description |
|-------|--------|-------------|
| YOLO Detection | Precision | TP / (TP + FP) for detected word boxes |
| YOLO Detection | Recall | TP / (TP + FN) for detected word boxes |
| YOLO Detection | mAP@0.5 | Mean Average Precision at IoU threshold 0.5 |
| TrOCR Recognition | CER | Character Error Rate (edit distance / total chars) |
| TrOCR Recognition | WER | Word Error Rate (word-level edit distance) |

---

## 6. Project Timeline

| # | Milestone | Target | Status |
|---|-----------|--------|--------|
| 1 | Data Collection & Labelling | Mar 2026 | In Progress |
| 2 | YOLO Detection Model Fine-tuning | Mar – Apr 2026 | Pending |
| 3 | TrOCR Recognition Model Fine-tuning | Apr 2026 | Pending |
| 4 | End-to-End Pipeline Integration | Apr – May 2026 | Pending |
| 5 | Evaluation, Report & Presentation | May 2026 | Pending |

**Immediate deadline:** Progress presentation slides (PPTX) — **29 March 2026**

---

## 7. Deliverables Checklist

- [ ] Progress presentation slides (PPTX) — due 29 March 2026
- [ ] Annotated dataset with dataset card (sources, splits, cleaning steps)
- [ ] Fine-tuned YOLO model + evaluation results (Precision, Recall, mAP@0.5)
- [ ] Fine-tuned TrOCR model + evaluation results (CER, WER)
- [ ] End-to-end pipeline codebase (GitHub, with documentation)
- [ ] Final report (8–10 pages, LaTeX, ISY5004 GC template)
- [ ] Final recorded presentation video (15 minutes, MP4)

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Insufficient labelled data | Medium | High | Two dedicated annotators from Pratham; supplement with IIIT-HW-Dev public dataset |
| YOLO detection errors propagating downstream | Medium | High | Evaluate detection in isolation before pipeline integration; iterate on annotation quality |
| TrOCR vocabulary mismatch | Medium | Medium | Fine-tune on field-specific data; apply Unicode normalisation (matras & nukta post-processing) |
| Compute availability | Low | High | AWS GPU VM available; training scoped to feasible batch sizes |

---

## 9. ISY5004 Scope Alignment

| Module Criterion | How This Project Qualifies |
|-----------------|---------------------------|
| Image/video analytics — object detection | YOLO detects and localises handwritten word regions (bounding boxes) in document images |
| Image-based classification and recognition | TrOCR treats each cropped word image as a visual sensing input → structured Unicode output |
| Dataset creation (rewarded per FAQ A5) | Novel field-collected Hindi handwriting dataset with annotation pipeline |
| Transferability of sensing techniques | YOLO detects visual word boundaries independent of language |

---

## 10. Key Technical Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| YOLOv8 for detection | Real-time capable, strong baseline for object detection, language-agnostic at the visual level |
| TrOCR for recognition | Purpose-built for OCR as image-to-sequence; encoder-decoder architecture handles character sequence generation naturally |
| Sequential pipeline (detect → recognise) | Decoupled training allows independent evaluation and iterative improvement of each stage |
| Word-level (not line/page-level) detection | Handles variable handwriting density and irregular line spacing better than line-level approaches |
| Label Studio for annotation | Supports both bounding box and text annotation tasks in a single platform; team-friendly UI |
| Unicode normalisation post-processing | Ensures consistent encoding of matras and nukta across different annotator inputs |

---

## 11. Open Questions & Next Steps

1. **Annotation quality control:** What inter-annotator agreement protocol will be used for bounding box labels?
2. **TrOCR base checkpoint:** Which TrOCR variant (base vs. large; printed vs. handwritten pre-trained) will be the starting point?
3. **YOLO confidence threshold:** What threshold will be used at inference to balance precision and recall for downstream TrOCR input quality?
4. **Reading-order algorithm:** Will row-grouping use a fixed pixel tolerance or a dynamic approach based on detected bounding box heights?
5. **Augmentation scope:** Which augmentations are safe for Devanagari (e.g., horizontal flip is not — it changes the script semantics)?

---

## 12. v2 outcomes (May 2026 — honest summary)

| Goal | v2 outcome | Met? |
|------|------------|------|
| HindiSeg TrOCR CER | 8.17% → **7.12%** | Partial (wanted ~5%) |
| End-to-end CER (eval pages 31–41) | ~52% → **36.4%** | Partial (wanted &lt;25%) |
| End-to-end CER (Pratham field) | ~98.5% → **71.3%** | **No** for unattended deployment |
| Inter-annotator κ | **0.62** (dual n=180/881) | **No** (wanted &gt;0.8) |
| HITL throughput | ~**3×** manual | Below **8×** stretch |

Details: [`docs/COMPLETION_REPORT.md`](COMPLETION_REPORT.md), [`models/MODEL_CARD_v2.md`](../models/MODEL_CARD_v2.md), `data/results/v2/`.

---

*Document generated: March 2026 · ISY5004 GC · NUS ISS*
