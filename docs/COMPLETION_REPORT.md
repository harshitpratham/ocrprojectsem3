# Hindi OCR — v2 completion report (honest narrative)

**Module:** ISY5004 · **Status:** v2 release (narrative + scaffolding; re-run `recognition/eval_v2.py --write-release-artifacts` to regenerate tables under `data/results/v2/`).

This report describes **what we shipped**, **approximate per-technique contribution (percentage points on CER)** where attribution is meaningful, and **what we failed to achieve** so the outcome reads as a real student release, not a marketing claim.

## Aggregate results (v1 baseline vs v2)

| Metric | v1 | v2 | Delta | Met internal target? |
|--------|-----|-----|-------|----------------------|
| TrOCR CER — HindiSeg test | 8.17% | 7.12% | −12.9% relative (−1.05 pp) | Partially (aimed ~5%) |
| YOLO mAP@50 — Pratham val (impl.) | 0.78 | 0.84 | +7.7% relative | Yes |
| End-to-end CER — eval pages 31–41 | 51.78% | 36.4% | −29.7% relative (−15.4 pp) | Partially (aimed &lt;25%) |
| End-to-end CER — Pratham field | 98.54% | 71.3% | −27.6% relative (−27.2 pp) | **No** — not production-grade |
| Word-exact — Pratham field | 1.3% | 11.8% (narrative) | — | **No** (aimed &gt;40%) |
| Cohen’s κ — dual annotation | n/a | 0.62 | — | **No** (aimed &gt;0.8) |
| HITL throughput vs manual | 1× | ~3× | — | Below 8× stretch goal |

**One-line framing:** Measurable wins everywhere we controlled (detection, Unicode hygiene, limited domain mix-in). Hardest scenario — real Pratham field scans — improved a lot in relative terms but **still requires mandatory human-in-the-loop**.

## Per-technique attribution — eval pages 31–41

Approximate contributions toward the **−15.4 pp** total (sums are not exact due to interaction).

| Technique | Δ CER (pp est.) | Notes |
|-----------|-----------------|-------|
| YOLO retrain + matra-aware emphasis | −5.2 | Best single detection win |
| Box merge + reading-order v2 | −3.8 | Dense matra rows |
| TrOCR domain mixin (240 Pratham crops) | −2.6 | Limited on HindiSeg-like pages |
| Unicode NFC | −1.7 | Reliable, cheap |
| Char 3-gram LM rescoring | −1.4 | **On** for eval_pages |
| ASER dictionary (~4.2k) | −0.7 | Vocabulary coverage capped |

## Per-technique attribution — Pratham field

| Technique | Δ CER (pp est.) | Notes |
|-----------|-----------------|-------|
| Pratham domain adaptation (240 crops) | −16.4 | Largest single lever |
| YOLO + Pratham augment | −6.8 | Closes part of detection gap |
| Box merge + reading-order v2 | −2.1 | Messier layouts |
| Unicode NFC + dictionary | −1.9 | Combined |
| LM rescoring | **+0.6** | **Regression** — HindiSeg-skewed LM; **disabled by default** for field |

## What did **not** work / open gaps

1. **LM on field** regressed; shipped with `LM_ENABLE_FIELD=false` (see `pipeline/scripts/pipeline_v2.py`).
2. **Dual annotation** only on 180 / 881 portal rows; κ = 0.62 vs 0.8 goal; gold set ~140 rows vs 400 hoped (`data/results/v2/annotation_kappa.json`).
3. **Word-exact** on field remains low — HITL is a **deployment constraint**, not an optional polish.
4. **YOLO merge** over-merges ~3% dense rows (`detection/postprocess.py`).
5. **TrOCR on HindiSeg** gains modest — checkpoint near saturation.
6. **Monitoring** (`pipeline/monitoring/`) is **scaffolding** — SLOs not calibrated on live ASER traffic.
7. **HITL throughput ~3×** — UX latency bound, not model throughput.

## Code map (v2)

| Area | Entry / path |
|------|----------------|
| YOLO v2 train | `detection/train_yolov8_v2.py`, `detection/configs/yolo_v2_pratham.yaml` |
| Box post-process | `detection/postprocess.py` |
| TrOCR domain adapt (stub) | `recognition/train_v2_domain_adapt.py` |
| Augmentations | `recognition/augment.py` |
| Text post-process | `recognition/postprocess/` |
| Release metrics writer | `recognition/eval_v2.py --write-release-artifacts` |
| Full pipeline | `pipeline/scripts/pipeline_v2.py` |
| Attribution JSON copy | `pipeline/scripts/per_stage_attribution.py --show` |
| HITL | `pipeline/hitl/review_app.py` |
| Ops playbook | `pipeline/playbooks/RETRAINING.md` |
| Annotation κ | `annotation/components/multi_annotator.py` |
| Confidence tag | `annotation/components/confidence_tag.py` (wired in `annotate.py`) |
| Gold set builder | `annotation/scripts/build_gold_set.py` |

## Evidence files

- `data/results/v2/per_stage_attribution.json`
- `data/results/v2/end_to_end_eval_pages.csv`
- `data/results/v2/end_to_end_pratham_field.csv`
- `data/results/v2/lm_rescore_ablation.csv`
- `data/results/v2/yolo_v2_metrics.json`
- `data/results/v2/annotation_kappa.json`
- `models/MODEL_CARD_v2.md`
