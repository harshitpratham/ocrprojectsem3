# Hindi OCR — Final Project Presentation Script
**Group 39 | ISY5004 Intelligent Sensing Systems | NUS ISS | Jan–May 2026**
**File:** `docs/presentation/final-presentation.html` | 23 slides | ~14 minutes

---

## Speaker Allocation

| Speaker | Slides | Topics | Approx time |
|---------|--------|--------|-------------|
| **Harshit AGARWAL** | 1–9 | Title, executive summary, client, business problem, script difficulty, baselines, goals, architecture, datasets | ~5 min |
| **Chitrarath BHATTACHARJEE** | 10–17 | YOLO deep-dive, YOLO post-processing, TrOCR architecture, training curve, domain adaptation, text post-processing, annotation portal, pipeline configuration & monitoring | ~5 min |
| **Jia Hui GAN** | 18–23 | Headline results, per-technique attribution (×2), per-doc table, failure modes, team + AI declaration | ~4 min |

**Narrative discipline for all speakers:**
- Lead with **deltas**, not absolutes. Say "−29.7% relative" before saying "36.4% CER."
- Name failures explicitly. "This didn't work because..." signals credibility, not weakness.
- Do not claim what was not achieved. HITL is mandatory, field is not production-ready — say that out loud.

---

## HARSHIT AGARWAL — Slides 1–9

### Slide 1: Title
**[20 seconds]**

"Good afternoon. I'm Harshit, and together with Chitrarath and Jia Hui we're presenting our final project: Hindi Handwritten Document Recognition via Visual Sensing, built for Pratham Education Foundation. This is ISY5004, Group 39."

---

### Slide 2: Executive Summary
**[50 seconds]**

"I'll lead with the honest summary before I explain anything else, so you have context for everything that follows.

We shipped a v2 release. The three headline improvements are all relative reductions in Character Error Rate. Recognition on clean pre-segmented images improved by **12.9% relative** — from 8.17% to 7.12%. The full pipeline on our evaluation pages improved by **29.7% relative** — from 51.8% to 36.4%. The hardest scenario, real Pratham field scans, improved by **27 percentage points absolute** — from about 98.5% to 71.3%.

What we did not achieve: the sub-25% pipeline goal, unattended field deployment — word-exact is still around 12% on field — and our annotation kappa target of 0.8. We landed at 0.62. I'll tell you exactly why each of these fell short during the failure modes slide. For now: meaningful progress, honest gaps."

---

### Slide 3: The Client
**[35 seconds]**

"Our client is Pratham International. They run ASER — the Annual Status of Education Report — the world's largest citizen-led learning assessment: 600,000+ households visited every year, all responses handwritten in Hindi on paper.

They also run the ATM app — Adaptive Teaching Module — on low-cost smartphones in rural India. Students write answers by hand, photograph the sheet, and upload it. The grading step is currently manual. Our pipeline replaces that bottleneck. The app is live with 6,000+ learners. OCR is the missing automation piece."

---

### Slide 4: Business Problem
**[40 seconds]**

"Look at the sequence diagram: student writes, photographs, uploads to ATM. Then the OCR step — currently a human transcribing 50 pages per day. That bottleneck delays grading by hours or days. We're replacing it.

Why don't existing tools work? Tesseract — designed for printed text, breaks on handwriting. Google Vision and Amazon Textract — English-centric, Devanagari handwriting is out-of-distribution. LLMs — Haiku 4.5 returned zero correct predictions on our 121-crop evaluation. Opus is partially capable but 15 times more expensive per image: not viable at millions of ASER responses per year."

---

### Slide 5: Why the Script is Hard
**[40 seconds]**

"Four compounding difficulties.

First, matra attachment: vowel diacritics attach above, below, and around the base character. Look at the Matra problem image — the same base consonant takes completely different visual forms with different matras. Generic detectors crop them wrong or merge adjacent words.

Second, ligature fusion: common consonant clusters merge into a single glyph that looks nothing like either constituent. Devanagari has hundreds of these.

Third, writer variability: our 11 eval folders range from CER 21% on neat print writers to 100% on erratic handwriting. No single model generalises trivially.

Fourth, no clear word boundaries: the shared shirorekha top bar runs across all letters; whitespace cues are weak and inconsistent."

---

### Slide 6: Baselines
**[35 seconds]**

"We tested everything accessible before building. Tesseract: near-random on handwritten Hindi. Google Vision and Textract: not designed for this domain. Sarvam AI: partial, layout mismatch. Haiku 4.5: **zero out of 121** on our eval crops — repeated hallucinations and invalid Devanagari characters. Opus: partial word matching but 15x cost.

Our v1 TrOCR achieved 13% exact word match on the pipeline evaluation. v2 moves the needle, but field exact match is still only 12%. The table shows why no drop-in solution existed: the domain specificity of handwritten Hindi requires domain-specific fine-tuning."

---

### Slide 7: Project Goals
**[30 seconds]**

"We set seven internal targets at the start. The table shows what we hit and what we missed. Two of seven are partial, three are not met. The recognition CER goal of 5% was too ambitious for the HindiSeg checkpoint near saturation. Field deployment is still far off. Kappa 0.62 tells us our annotation coverage was too thin. I want you to see this table because it grounds everything that follows — we're not claiming success across the board."

---

### Slide 8: System Architecture v2
**[35 seconds]**

"The architecture diagram shows the full pipeline. A raw scanned page enters YOLOv8 word detection — CSPDarknet backbone, PAN-FPN neck, anchor-free detect head. Each detected word crop is resized and passed through the ViT encoder — 196 patches, 768-dimensional embeddings, 12 attention heads — then cross-attended by the RoBERTa-Hindi decoder, which outputs Hindi Unicode token by token.

Why decoupled? Each stage can be trained, evaluated, and improved independently. Errors don't compound before you can measure them. The v2 addition: post-processing after TrOCR — NFC normalization, optional LM rescoring, dictionary correction — before the confidence gate routes to output or HITL."

---

### Slide 9: Datasets
**[35 seconds]**

"Three primary datasets. IIIT-H HindiSeg: 95,000 pre-segmented word images, 11,029 unique Hindi words, used exclusively for TrOCR training and recognition benchmarking. Pratham field pages: 155 full-page scans, YOLO-format annotations, used for word detection training. Evaluation crops: 121 YOLO-detected words from the 11 val pages, human-annotated via our portal, used for pipeline-level benchmarking.

For v2 domain adaptation we curated 240 Pratham crops from the portal history. Cohen's kappa on the dual-annotated subset was 0.62 — below target — and the gold set is only 140 rows, not the 400 we hoped for. These data quality limitations directly limit our field adaptation performance."

---

## CHITRARATH BHATTACHARJEE — Slides 10–17

### Slide 10: YOLO Modelling Deep-dive
**[45 seconds]**

"Thanks Harshit. I'll walk through the technical architecture, starting with YOLO detection.

Base model is YOLOv8s pre-trained on COCO, fine-tuned on 134 Pratham field pages. In v2 we made several targeted changes over v1: image size 640 to 800 pixels to capture more matra detail; training extended to 80 epochs with warmup over 5 epochs; NMS IoU threshold tightened to 0.42 to reduce over-suppression in matra-heavy rows; box loss weight elevated from 7.5 to 10.0 to force tighter box regression; confidence threshold lowered slightly to 0.22 for better recall.

The augmentation policy is Devanagari-safe: horizontal and vertical flip are both disabled at zero. Flipping Devanagari text creates semantically invalid glyphs. Brightness variation of 0.55 simulates scan quality differences.

Result: mAP@50 improved from 0.78 to 0.84."

---

### Slide 11: YOLO Post-processing
**[40 seconds]**

"The flowchart shows the post-processing steps: raw YOLO boxes go through class-agnostic NMS at IoU threshold 0.42, then a vertical-overlap grouping check — boxes sharing at least 35% of their smaller height are considered the same row — then a horizontal gap check. If boxes on the same row are separated by 8 pixels or less, they're merged. Then all boxes are sorted into reading order: bucket by y-center row, then left-to-right by x1 within each row.

Three clean Python functions in `detection/postprocess.py`: `nms`, `merge_adjacent_reading_order`, `sort_reading_order`. The over-merging rate of 3.1% is the documented residual failure at the current setting."

---

### Slide 12: TrOCR Architecture
**[45 seconds]**

"TrOCR is a VisionEncoderDecoderModel. The encoder is ViT-base-patch16-224, pre-trained on ImageNet. It divides each word-crop image into 196 patches of 16×16 pixels, produces 768-dimensional embeddings, and uses 12 attention heads to capture long-range matra context across the full word width.

The decoder is RoBERTa-Hindi from flax-community. Its BPE tokenizer knows Devanagari subword units, including matra-plus-base combinations, so the model starts with vastly fewer UNK tokens compared to a generic tokenizer.

We decode with beam search, beam=2, max length 64. Training used FP16 precision on Tesla T4 — halving GPU memory allows batch size 16 — AdamW with learning rate 5e-5 and cosine decay. We preprocessed all images to tensors offline, which reduced training wall-clock time by about 40%."

---

### Slide 13: Training Curve
**[35 seconds]**

"The training curve tells the learning story. CER starts at around 50%. Between steps 6k and 12k, it learns the dominant character shapes and common matras rapidly, dropping to about 22%. From 24k to 38k the improvement is slower — the model is learning minority matra combinations and conjunct disambiguation. At step 38,000 — epoch 8.7 — validation CER reaches 8.17% and begins to plateau. By step 42,000 CER rises to 8.75%: overfitting. We save the 38k checkpoint.

The green dashed extension shows the v2 domain adaptation second stage, bringing the effective test CER to 7.12% on HindiSeg — modest, as the base checkpoint was already near saturation."

---

### Slide 14: Domain Adaptation v2
**[40 seconds]**

"The curriculum diagram shows v2. Stage 1 is the base v1 model, trained on HindiSeg to checkpoint 38k. Stage 2 is a second fine-tune starting from that checkpoint.

Phase 1 mixes 80% HindiSeg and 20% Pratham crops with minimal augmentation — adapt without catastrophic forgetting. Phase 2 increases the Pratham ratio and applies the augmentation stack: ink-fade simulation, elastic warp, paper noise, box jitter. The learning rate is reduced to 5e-6 specifically to prevent overwriting the HindiSeg vocabulary.

The training set is 240 curated Pratham crops, augmented 3x to an effective size of ~720. This is the largest single improvement on the field scenario: approximately minus 16.4% CER."

---

### Slide 15: Text Post-processing
**[40 seconds]**

"Three stages, each with a distinct character.

Stage 1 is Unicode NFC normalization — a single Python call. Different annotators encode the same word with different Unicode representations; the TrOCR tokenizer uses one canonical form. Correcting this mismatch gives minus 1.7% CER at zero training cost. The cheapest win in the project.

Stage 2 is char 3-gram language model rescoring. It helps eval-page text — minus 1.4% — but regresses field text by plus 0.6% because the LM was trained on HindiSeg, not field vocabulary. This is why we ship it disabled for field by default.

Stage 3 is Levenshtein-1 ASER dictionary correction against about 4,200 words. Minus 0.7% on eval pages."

---

### Slide 16: Annotation Portal
**[35 seconds]**

"The annotation portal is a multi-user Streamlit app running on Ubuntu EC2. It collected 881 annotation rows. Annotators see the crop image, the TrOCR prediction, and can confirm correct, type a correction, or mark the crop as invalid.

v2 added confidence tags — high, medium, low — per crop, and multi-annotator agreement scoring using Cohen's kappa.

The honest numbers: kappa 0.62 against a 0.8 target. Only 180 of 881 entries had dual annotations — coverage was thin. The gold set is 140 rows, not the 400 we needed. This data quality ceiling directly limits domain adaptation."

---

### Slide 17: Pipeline — Configuration & Monitoring
**[30 seconds]**

"The environment variable table controls the key runtime behaviour. V2_SCENARIO and LM_ENABLE_FIELD determine whether LM rescoring is active. LM is off for field by default — turning it on regresses CER, as shown in slide 19.

Two monitoring modules exist: `cer_monitor.py` tracks rolling CER with an alert hook, and `drift_detector.py` flags input distribution shift from scan-quality changes. Both are scaffolding — thresholds are placeholders not yet calibrated on real Pratham traffic. SLOs must be agreed with Pratham ops before production use.

I'll hand over to Jia Hui for results."

---

## JIA HUI GAN — Slides 18–23

### Slide 18: Headline Results
**[40 seconds]**

"Thanks Chitrarath. Let me walk through results — leading with the deltas before the absolutes.

On recognition only, HindiSeg test CER went from 8.17% to 7.12% — a 12.9% relative improvement. Modest. The checkpoint was already near saturation.

On the full pipeline, evaluation pages CER went from 51.78% to 36.4% — a 29.7% relative improvement. We missed the sub-25% internal goal. The bar chart shows where we aimed and where we landed.

On Pratham field scans, CER dropped from about 98.5% to 71.3% — 27 percentage points absolute. The critical metric is word-exact accuracy: roughly 12% on field. That is why HITL is mandatory, not optional."

---

### Slide 19: Per-technique Attribution — Eval Pages
**[40 seconds]**

"This is the mid-level story — approximately what each technique was worth on the evaluation pages, in percentage points of CER reduction.

The biggest single lever was YOLO retraining with matra emphasis: approximately minus 5.2%. This validates the hypothesis that crop quality, not recognition quality, was the primary bottleneck.

Box merge and reading-order v2: minus 3.8%. TrOCR domain mixin: minus 2.6% — modest because eval pages are HindiSeg-style anyway.

Unicode NFC normalization: minus 1.7% for a single Python call. Highest return per engineering effort.

Char LM rescoring: minus 1.4% on eval pages. Dictionary correction: minus 0.7%."

---

### Slide 20: Per-technique Attribution — Pratham Field
**[40 seconds]**

"The field attribution tells a different story. The dominant lever was domain adaptation on 240 Pratham crops: approximately minus 16.4% — by far the largest improvement in the entire project on this scenario.

YOLO retrain with field augmentation: minus 6.8%. Box merge: minus 2.1%. NFC plus dictionary combined: minus 1.9%.

And then the red bar. LM rescoring on field: **plus 0.6%** — a regression. The HindiSeg-trained char LM prefers HindiSeg-style predictions over field-specific outputs, replacing correct field words with wrong ones. Evidence is in `data/results/v2/lm_rescore_ablation.csv`. We ship it disabled for field by default."

---

### Slide 21: Per-document Deep Dive
**[35 seconds]**

"The table drills into the 11 evaluation folders. Only two folders meet the sub-25% CER target after v2: folder 38 — neat block print, 13.6% CER — and folder 35 — printed style, 19.5%.

Folder 37 — the most erratic writer — went from 100% CER to 78%. Still a complete failure by any deployment standard.

The per-document view shows that 'average CER' hides a bimodal reality: clean print writers approaching usable accuracy; erratic writers needing targeted data collection, not more model training on the same distribution."

---

### Slide 22: Failure Modes
**[45 seconds]**

"This slide exists because honest projects describe what failed. We have six clear failures.

One: LM rescoring on field regressed CER by 0.6%. Corpus skew. Disabled for field by default.

Two: annotation kappa 0.62, not 0.8. We dual-annotated only 180 of 881 portal entries. Gold set is 140 rows, not 400. Limits domain adaptation data quality.

Three: field word-exact accuracy is 12% — far below the 70% threshold for unattended grading. HITL is a deployment constraint, not a feature.

Four: YOLO over-merge on about 3% of dense rows. Future fix: per-row dynamic gap from spacing histogram.

Five: HindiSeg CER gain was modest — 1%. The checkpoint was near saturation.

Six: monitoring is uncalibrated scaffolding, and HITL throughput is about 3× manual, not the 8× we proposed.

Folder 37 on the right shows complete recognition failure even after all v2 improvements."

---

### Slide 23: Thank You
**[15 seconds]**

"That's our final project presentation. Thank you.

Per NUS academic integrity policy: we used Claude from Anthropic to assist with drafting and structuring project documentation and presentation content, and Cursor for coding assistance and development workflow throughout the project. The team takes full responsibility for all methodology, results, and content.

We're happy to take questions."

---

## Timing Summary

| Speaker | Slides | Script total |
|---------|--------|-------------|
| Harshit | 1–9 | ~5 min 10 sec |
| Chitrarath | 10–17 | ~4 min 50 sec |
| Jia Hui | 18–23 | ~3 min 55 sec |
| **Total** | **23 slides** | **~13 min 55 sec** |

---

## Presenter Tips

### For Harshit
- Say "12.9% relative" and "27 percentage points absolute" — distinguish relative from absolute clearly.
- On slide 4, gesture to the Mermaid sequence diagram to trace the bottleneck step.
- On slide 7, pause on the red "Not met" rows — credibility comes from naming the misses directly.
- On slide 8, point to the architecture image — trace the path from scan to Unicode output.

### For Chitrarath
- Slide 10: walk only the changed parameters (image size, NMS IoU, box loss, warmup epochs). Skip what didn't change.
- Slide 11: the Mermaid flowchart does the work — point at stages in order.
- Slide 13: trace the curve with your cursor — the visual story is the point.
- Slide 15: the LM regression is the most important sentence. Pause and say "plus 0.6%" clearly. That candour is what makes the whole presentation believable.
- Slide 17: keep it brief — the env var table is self-explanatory. Focus on why LM is off for field.

### For Jia Hui
- Slide 19: name the biggest lever first. "YOLO retraining bought us 5.2% — that was the single biggest win."
- Slide 20: the red bar is the centrepiece. Let it land. "This is the one that regressed, and here's why."
- Slide 21: call out folder 37 by name. "Still a complete failure at 78% CER after all v2 improvements."
- Slide 22: own this slide. Six failures, clearly stated — that's what makes the rest of the results trustworthy.
- Slide 23: short and confident.

### General
- ~14 minutes total. Budget about 35–40 seconds per slide on average.
- Have the browser open to `final-presentation.html`. The sticky nav lets you jump to any slide.
- Keep the browser at ~80% zoom so all slide content is visible without scrolling.
- Pre-render Mermaid diagrams: navigate to slides 4, 11, 14 before starting.
- Know the four key numbers cold: **8.17% → 7.12%** (recognition), **51.8% → 36.4%** (pipeline), **98.5% → 71.3%** (field), **0.62** (kappa).

---

## Q&A Preparation

**1. "Is this production-ready for Pratham?"**

No — not for unattended field deployment. Word-exact accuracy on field uploads is approximately 12%, well below the 70% threshold for autonomous grading. HITL review is mandatory for every field upload. Production sign-off requires higher field word-exact accuracy, calibrated monitoring SLOs, and formal agreement from Pratham ops on the human review workflow.

**2. "Why did the LM regress on field text?"**

The char 3-gram LM was trained on HindiSeg transcripts. HindiSeg vocabulary and spelling conventions differ from Pratham field handwriting. When the decoder generates a field-specific output, the LM sometimes scores a HindiSeg-style alternative higher and substitutes it. The effect is small (+0.6% CER) but directionally wrong. Future work: build the LM from in-domain field transcripts, or remove the LM stage entirely for field.

**3. "Why is the HindiSeg improvement so modest — only 1%?"**

The step-38k checkpoint was already near saturation on the HindiSeg distribution. The learning curve flattened between 36k and 38k, and the model overfit slightly by 42k. More data — different writers, more word types — would be needed for significant further improvement.

**4. "Why did annotation kappa land at 0.62?"**

Only 180 of 881 portal entries were dual-annotated. With thin dual coverage, the kappa estimate reflects genuine annotator disagreement on ambiguous crops rather than systematic bias. The 0.8 target assumed 400+ dual-annotated rows.

**5. "Why not use a larger model — GPT-4o or similar?"**

Cost: at millions of ASER responses per year, frontier-model API cost is 15–50× that of a fine-tuned local model. Privacy: Pratham data involves children's educational records; cloud API calls create data-governance risks. Our Haiku baseline (0% accuracy) already demonstrated that zero-shot LLMs fail on this task.

**6. "Could this be extended to other Indic languages?"**

Yes — by design. YOLO is language-agnostic. The only language-specific component is the TrOCR decoder. Swapping RoBERTa-Hindi for a Tamil, Bengali, or Gujarati equivalent requires only retraining the decoder on the target script's handwriting data.
