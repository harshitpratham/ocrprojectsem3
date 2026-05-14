# Hindi OCR — Final Project Presentation Script
**Group 39 | ISY5004 Intelligent Sensing Systems | NUS ISS | Jan–May 2026**
**File:** `docs/presentation/final-presentation.html` | 25 slides | ~15 minutes

---

## Speaker Allocation

| Speaker | Slides | Topics | Approx time |
|---------|--------|--------|-------------|
| **Harshit AGARWAL** | 1–9 | Title, executive summary, client, business problem, script difficulty, baselines, goals, architecture, datasets | ~5 min |
| **Chitrarath BHATTACHARJEE** | 10–17 | YOLO deep-dive, YOLO post-processing, TrOCR architecture, training curve, domain adaptation, text post-processing, annotation portal, pipeline integration | ~5 min |
| **Jia Hui GAN** | 18–25 | Headline results, per-technique attribution (×2), per-doc table, failure modes, deployment + ethics, v3 roadmap, team + AI declaration | ~5 min |

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

"The pipeline: raw scan goes into YOLO v2 for word detection, bounding boxes go through post-processing — NMS, matra-aware merge, reading order sort — then into TrOCR v2 for word recognition, then three-stage text post-processing: Unicode NFC normalization, optional language-model rescoring — disabled for field — and dictionary correction. A confidence gate routes low-confidence predictions to a human-in-the-loop review app; high-confidence goes directly to output. Monitoring watches for CER drift and triggers retraining.

Why decoupled? Each stage can be trained, evaluated, and improved independently. Errors don't compound before you can measure them."

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

Result: mAP@50 improved from 0.78 to 0.84. The remaining failure mode — over-merging on ~3% of dense rows — is addressed in the next slide."

---

### Slide 11: YOLO Post-processing
**[40 seconds]**

"The flowchart shows the post-processing steps: raw YOLO boxes go through class-agnostic NMS at IoU threshold 0.42, then a vertical-overlap grouping check — boxes sharing at least 35% of their smaller height are considered the same row — then a horizontal gap check. If boxes on the same row are separated by 8 pixels or less and have sufficient vertical overlap, they're merged. Then all boxes are sorted into reading order: bucket by y-center row, then left-to-right by x1 within each row.

Three clean Python functions in `detection/postprocess.py`: `nms`, `merge_adjacent_reading_order`, `sort_reading_order`. The 8-pixel merge gap is the tunable parameter — set too high and adjacent words merge; set too low and matra-split boxes remain separate. The over-merging rate of 3.1% is the documented residual failure at the current setting."

---

### Slide 12: TrOCR Architecture
**[45 seconds]**

"TrOCR is a VisionEncoderDecoderModel. The encoder is ViT-base-patch16-224, pre-trained on ImageNet. It divides each word-crop image into 196 patches of 16×16 pixels, produces 768-dimensional embeddings, and uses 12 attention heads to capture long-range matra context across the full word width. This is why ViT outperforms CNN-based encoders here — matras above and below the midline require attention across the full image, not just local receptive fields.

The decoder is RoBERTa-Hindi from flax-community. Its BPE tokenizer knows Devanagari subword units, including matra-plus-base combinations, so the model starts with vastly fewer UNK tokens compared to a generic tokenizer.

We decode with beam search, beam=2, max length 64. Training used FP16 precision on Tesla T4 — halving GPU memory allows batch size 16 — AdamW with learning rate 5e-5 and cosine decay. We preprocessed all images to tensors offline, which reduced training wall-clock time by about 40%."

---

### Slide 13: Training Curve
**[35 seconds]**

"The training curve tells the learning story. CER starts at around 50% — the model is effectively guessing. Between steps 6k and 12k, it learns the dominant character shapes and common matras rapidly, dropping to about 22%. From 24k to 38k the improvement is slower — the model is learning minority matra combinations and conjunct disambiguation. At step 38,000 — epoch 8.7 — validation CER reaches 8.17% and begins to plateau. By step 42,000 CER rises to 8.75%: overfitting on minority writers. We save the 38k checkpoint.

The green dashed extension shows the v2 domain adaptation second stage, bringing the effective test CER to 7.12% on HindiSeg — modest gain, as the base checkpoint was already near saturation on this distribution."

---

### Slide 14: Domain Adaptation v2
**[40 seconds]**

"The Mermaid diagram shows the curriculum design for v2. Stage 1 is the base v1 model, trained on HindiSeg to checkpoint 38k. Stage 2 is a second fine-tune starting from that checkpoint.

Phase 1 of stage 2 mixes 80% HindiSeg and 20% Pratham crops with minimal augmentation — we want to adapt without catastrophic forgetting. Phase 2 increases the Pratham ratio and applies the `recognition/augment.py` stack: ink-fade simulation, elastic warp, paper noise, box jitter. The learning rate is reduced to 5e-6 — one-tenth of the initial rate — specifically to prevent overwriting the HindiSeg vocabulary.

The training set is 240 curated Pratham crops from the annotation portal history, augmented 3x to an effective size of ~720. Entry point is `recognition/train_v2_domain_adapt.py`. This is the largest single improvement on the field scenario: approximately minus 16 percentage points CER."

---

### Slide 15: Text Post-processing
**[40 seconds]**

"Three stages, each with a distinct character.

Stage 1 is Unicode NFC normalization — a single Python call that canonically composes Devanagari sequences. Different annotators encode the same word with different Unicode representations; the TrOCR tokenizer uses one canonical form. Correcting this mismatch gives minus 1.7 percentage points CER at zero training cost. The cheapest win in the project.

Stage 2 is char 3-gram language model rescoring. It helps eval-page text — minus 1.4pp — but regresses field text by plus 0.6pp because the LM was trained on HindiSeg, not field vocabulary. This is why we ship it disabled for field by default. The failure is documented in `data/results/v2/lm_rescore_ablation.csv`.

Stage 3 is Levenshtein-1 ASER dictionary correction against about 4,200 words. Minus 0.7pp on eval pages. Limited gain because the eval pages span vocabulary well beyond the ASER word list."

---

### Slide 16: Annotation Portal
**[35 seconds]**

"The annotation portal is a multi-user Streamlit app running on Ubuntu EC2. It collected 881 annotation rows. Annotators see the crop image, the TrOCR prediction, and can confirm correct, type a correction, or mark the crop as invalid.

v2 added two components: confidence tags — high, medium, low — per crop, and multi-annotator agreement scoring using Cohen's kappa.

The honest numbers from `data/results/v2/annotation_kappa.json`: kappa 0.62 against a 0.8 target. Only 180 of 881 entries had dual annotations — coverage was thin. The gold set produced by the build_gold_set script is 140 rows, not the 400 we needed. This data quality ceiling directly limits domain adaptation."

---

### Slide 17: Pipeline Integration
**[30 seconds]**

"The Mermaid diagram shows the full operational pipeline. Images come from S3 input, YOLO batch runs on EC2, produces sorted crops, TrOCR batch runs, text post-processing applies, and a confidence gate routes to either S3 output or the HITL queue. Low-confidence predictions go to the Streamlit review app. Accepted corrections feed back into the gold set for future retraining. Monitoring watches CER and input distribution drift.

The environment variable table is important: V2_SCENARIO and LM_ENABLE_FIELD control whether LM rescoring is on or off. LM is off for field by default — flipping it on regresses CER as shown in slide 20.

I'll hand over to Jia Hui for results."

---

## JIA HUI GAN — Slides 18–25

### Slide 18: Headline Results
**[40 seconds]**

"Thanks Chitrarath. Let me walk through results — and I want to lead with the deltas before the absolutes.

On recognition only, HindiSeg test CER went from 8.17% to 7.12% — a 12.9% relative improvement. Modest. The checkpoint was already near saturation.

On the full pipeline, evaluation pages CER went from 51.78% to 36.4% — a 29.7% relative improvement. We missed the sub-25% internal goal. The bar chart shows where we aimed and where we landed.

On Pratham field scans, CER dropped from about 98.5% to 71.3% — 27 percentage points absolute. This sounds large, but the baseline was near-random, so the relative improvement story can be misleading. The critical metric is word-exact accuracy: roughly 12% on field. That is why HITL is mandatory, not optional."

---

### Slide 19: Per-technique Attribution — Eval Pages
**[40 seconds]**

"This is the mid-level story — approximately what each technique was worth on the evaluation pages, in percentage points of CER reduction. The bars are approximate ablation contributions; they don't add linearly because techniques interact.

The biggest single lever was YOLO retraining with matra emphasis: approximately minus 5.2 percentage points. This validates the hypothesis that crop quality, not recognition quality, was the primary bottleneck.

Box merge and reading-order v2: minus 3.8pp. TrOCR domain mixin: minus 2.6pp — modest because eval pages are HindiSeg-style anyway.

Unicode NFC normalization: minus 1.7pp for a single Python call. This is the highest-return investment per engineering effort in the project.

Char LM rescoring: minus 1.4pp on eval pages. Dictionary correction: minus 0.7pp — limited by vocabulary coverage."

---

### Slide 20: Per-technique Attribution — Pratham Field
**[40 seconds]**

"The field attribution tells a different story. The dominant lever here was domain adaptation on 240 Pratham crops: approximately minus 16.4 percentage points — by far the largest improvement in the entire project on this scenario.

YOLO retrain with field augmentation: minus 6.8pp. Box merge: minus 2.1pp — messier layouts mean less benefit from the heuristic. NFC plus dictionary combined: minus 1.9pp.

And then the red bar. LM rescoring on field: **plus 0.6 percentage points** — a regression. The HindiSeg-trained char LM prefers HindiSeg-style predictions over field-specific outputs, replacing correct field words with wrong but HindiSeg-plausible ones. The evidence is in `data/results/v2/lm_rescore_ablation.csv`. We ship it disabled for field by default."

---

### Slide 21: Per-document Deep Dive
**[35 seconds]**

"The table drills into the 11 evaluation folders. Only two folders meet the sub-25% CER target after v2: folder 38 — neat block print, 13.6% CER — and folder 35 — printed style, 19.5%.

The remaining nine folders still exceed 25%. Folder 37 — the most erratic writer — went from 100% CER to 78%. Still a complete failure by any deployment standard. All v2 improvements moved the numbers, but the distribution tail is stubborn.

The per-document view shows that 'average CER' hides a bimodal reality: the clean print writers are approaching usable accuracy; the erratic writers need targeted data collection, not more model training on the same distribution."

---

### Slide 22: Failure Modes
**[45 seconds]**

"This slide exists because honest projects describe what failed. We have six clear failures.

One: LM rescoring on field regressed CER by 0.6pp. Corpus skew. Shipped disabled for field.

Two: annotation kappa 0.62, not 0.8. We dual-annotated only 180 of 881 portal entries. Gold set is 140 rows, not 400. This limits our domain adaptation data quality.

Three: field word-exact accuracy is 12% — far below the 70% threshold for unattended grading. HITL is a deployment constraint, not a feature.

Four: YOLO over-merge on about 3% of dense rows. Fixed merge gap; v3 needs dynamic spacing.

Five: HindiSeg CER gain was modest — 1pp. The checkpoint was near saturation; more data, not more training time, is the path forward.

Six: monitoring is uncalibrated scaffolding, and HITL throughput is about 3 times manual, not the 8 times we proposed. The bottleneck is per-word review UX latency.

Look at folder 37 on the right. These two crops show wrong predictions versus the correct ground truth. The erratic writer breaks both detection and recognition. Post-v2 they are still wrong. This is what motivates HITL and targeted data collection rather than assuming more model iterations will solve it."

---

### Slide 23: Deployment & Ethics
**[35 seconds]**

"Deployment constraints for v2: HITL mandatory for field uploads, no exceptions. LM off by default for field. Monitoring SLOs are placeholders pending Pratham ops sign-off. Model weights are not publicly hosted — see `models/MODEL_STORAGE.md` for EC2 access. All of this is documented in `models/MODEL_CARD_v2.md`.

On ethics: ASER data is from rural children. No sharing with third parties. No public release of raw images. Misrecognition that unfairly reduces a student's score is worse than slow manual grading — that's why the HITL gate cannot be bypassed. Annotator bias from thin dual coverage introduces label noise that propagates into domain adaptation. Devanagari is culturally significant and misrecognized text should not reach end-users without human review."

---

### Slide 24: v3 Roadmap
**[30 seconds]**

"Three priorities for v3.

Priority one: close the eval-page CER gap from 36% toward 25%. Dynamic merge gap, more YOLO training data, targeted hard-writer collection, expanded TrOCR augmentation.

Priority two: field deployment readiness. Expand the Pratham domain-adapt set from 240 to 1,000+ crops, build an in-domain LM to remove HindiSeg bias, get to kappa above 0.8 by dual-annotating 220+ more portal entries, and calibrate monitoring SLOs on real ATM traffic.

Priority three: HITL and extensions. Batch review mode, keyboard-first flows for faster throughput, multilingual extension using the same YOLO with language-specific decoder swaps, and exploration of end-to-end unified architectures.

If all three land, the target state is: eval-page CER below 25%, field word-exact above 40%, kappa above 0.8, and HITL throughput at 8x manual."

---

### Slide 25: Team + AI Declaration
**[15 seconds]**

"That's our final project presentation. Thank you.

Per NUS academic integrity policy: we used Claude from Anthropic for documentation drafting and code scaffolding review, and Google Gemini for related-work research on Indic OCR. The team takes full responsibility for all methodology, results, and content.

We're happy to take questions."

---

## Timing Summary

| Speaker | Slides | Script total |
|---------|--------|-------------|
| Harshit | 1–9 | ~5 min 20 sec |
| Chitrarath | 10–17 | ~4 min 50 sec |
| Jia Hui | 18–25 | ~4 min 35 sec |
| **Total** | **25 slides** | **~14 min 45 sec** |

---

## Presenter Tips

### For Harshit
- Say "12.9% relative" and "27 percentage points absolute" — distinguish relative from absolute clearly; the audience will ask if you conflate them.
- On slide 4, gesture to the Mermaid sequence diagram to trace the bottleneck step.
- On slide 7, pause on the red "Not met" rows — don't rush past them. Credibility comes from naming the misses directly.

### For Chitrarath
- Slide 10: the table is dense — walk it row by row for the changed parameters only (image size, NMS IoU, box loss, warmup epochs). Skip what didn't change.
- Slide 11: the Mermaid flowchart does the work — point at stages in order, don't re-read every node label.
- Slide 13: trace the curve with your cursor/pointer — the visual story is the point.
- Slide 15: the LM regression is the most important sentence on that slide. Pause and say "plus 0.6 percentage points" clearly. That candour is what makes the whole presentation believable.

### For Jia Hui
- Slide 20: the red bar is the centrepiece. Let it land. "This is the one that regressed, and here's why."
- Slide 21: call out folder 37 by name. "This is our hardest writer group, and it is still a complete failure at 78% CER after all v2 improvements."
- Slide 22: own this slide. Don't hedge. Six failures, clearly stated — that's what makes the rest of the results trustworthy.
- Slide 25: short and confident. No need to recap anything.

### General
- 15 minutes total including slide transitions. Budget about 35–40 seconds per slide on average.
- Have the browser open to `final-presentation.html`. The sticky nav lets you jump to any slide by clicking the number.
- Keep the browser at ~80% zoom so all slide content is visible without scrolling.
- Mermaid diagrams may take 1–2 seconds to render on first load — navigate to slides 4, 8, 11, 14, 17 before the presentation starts to pre-render them.
- Know the four key numbers cold: **8.17% → 7.12%** (recognition), **51.8% → 36.4%** (pipeline), **98.5% → 71.3%** (field), **0.62** (kappa).

---

## Q&A Preparation

**1. "Is this production-ready for Pratham?"**

No — not for unattended field deployment. Word-exact accuracy on field uploads is approximately 12%, well below the 70% threshold we set for autonomous grading. HITL review is mandatory for every field upload in v2. Production sign-off requires achieving higher field word-exact accuracy (target >40%), calibrating monitoring SLOs on real ATM traffic, and formal agreement from Pratham ops on the human review workflow.

**2. "Why did the LM regressor on field text?"**

The char 3-gram LM was trained on HindiSeg transcripts. HindiSeg vocabulary and spelling conventions differ from Pratham field handwriting. When the decoder generates a field-specific output, the LM sometimes scores a HindiSeg-style alternative higher and substitutes it. The effect is small (+0.6pp CER) but directionally wrong. v3 fix: build the LM from in-domain field transcripts, or remove the LM stage entirely for field and rely on the decoder alone.

**3. "Why is the HindiSeg improvement so modest — only 1pp?"**

The step-38k checkpoint was already near saturation on the HindiSeg distribution. The learning curve flattened between 36k and 38k, and the model overfit slightly by 42k. Curriculum learning and augmentation contributed less than 3pp hoped because the limiting factor is not training procedure — it's diversity of the HindiSeg training set itself. More data (different writers, more word types) would be needed for significant further HindiSeg improvement.

**4. "Why did annotation kappa land at 0.62?"**

Only 180 of 881 portal entries were dual-annotated. The remainder had a single annotator. With thin dual coverage, the kappa estimate is noisy and reflects genuine annotator disagreement on ambiguous crops (damaged ink, unusual letterforms) rather than systematic bias. The 0.8 target assumed 400+ dual-annotated rows. v3 action: run a dedicated dual-annotation sprint before the next domain adaptation run.

**5. "Why not use a larger model — GPT-4o or similar?"**

Cost: at millions of ASER responses per year, frontier-model API cost is 15–50× that of a fine-tuned local model. Privacy: Pratham data involves children's educational records; cloud API calls create data-governance risks. Control: fine-tuned local models can be updated, audited, and constrained in ways that API models cannot. Our Haiku baseline (0% accuracy) already demonstrated that zero-shot LLMs fail on this task; fine-tuning is necessary regardless of model scale.

**6. "Could this be extended to other Indic languages?"**

Yes — by design. YOLO is language-agnostic: it detects visual word-blob regions regardless of script. The only language-specific component is the TrOCR decoder. Swapping RoBERTa-Hindi for a Tamil, Bengali, or Gujarati RoBERTa equivalent requires only retraining the decoder on the target script's handwriting data. The detection + post-processing + pipeline infrastructure would transfer unchanged.

---

## AI Tool Declaration (for slide 25)

"In accordance with NUS academic integrity policy, our team declares the use of the following AI tools:

**Claude (Anthropic):** Used to assist with drafting and structuring project documentation (GUIDE.md, COMPLETION_REPORT.md, DATA_CARD.md, model card), code scaffolding review and annotation, and final presentation content organisation and script drafting.

**Google Gemini:** Used to source references for existing-tool baseline analysis and related work on Indic OCR systems.

All experimental methodology, model training decisions, hyperparameter choices, evaluation protocols, and result interpretation are the work of the team. The team takes full responsibility for the accuracy and quality of this submission."
