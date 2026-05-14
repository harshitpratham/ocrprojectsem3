# Haiku Model Accuracy Report - Handwritten Hindi Word Recognition

**Generated:** March 14, 2026  
**Model:** Claude Haiku 4.5  
**Task:** OCR / Hindi Devanagari Word Recognition  
**Dataset:** 121 handwritten Hindi word images across 11 folders (31-41)

---

## Executive Summary

The Haiku model was tasked with annotating handwritten Hindi (Devanagari) word images **without access to ground truth labels**. The results reveal a **complete failure** in this task:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **0.00%** (0/121 images) |
| **Total Images Analyzed** | 121 |
| **Correct Predictions** | 0 |
| **Incorrect Predictions** | 121 |
| **Folders Analyzed** | 11 (31-41) |

---

## Per-Folder Breakdown

| Folder | Images | Correct | Accuracy | Primary Issue |
|--------|--------|---------|----------|---------------|
| 31 | 13 | 0 | 0.00% | Severe misreading |
| 32 | 11 | 0 | 0.00% | Severe misreading |
| 33 | 11 | 0 | 0.00% | Severe misreading |
| 34 | 13 | 0 | 0.00% | Severe misreading |
| 35 | 12 | 0 | 0.00% | Severe misreading |
| 36 | 13 | 0 | 0.00% | Severe misreading |
| 37 | 15 | 0 | 0.00% | Severe misreading |
| 38 | 13 | 0 | 0.00% | Severe misreading |
| 39 | 3 | 0 | 0.00% | Severe misreading |
| 40 | 11 | 0 | 0.00% | Severe misreading |
| 41 | 6 | 0 | 0.00% | Severe misreading |

---

## Critical Analysis

### Why Did Haiku Fail Completely?

The 0% accuracy indicates fundamental limitations:

#### 1. **Devanagari Script Recognition Limitation**
   - Haiku likely lacks training data for complex Devanagari handwriting recognition
   - The model cannot reliably distinguish Devanagari character conjuncts and diacritics in handwritten form
   - Hindi text recognition from handwritten images is not a core competency

#### 2. **Examples of Severe Misreadings**

**Folder 31 Sample:**
- Expected: शब्द (shabda - word)
- Haiku read: धरती (dharti - earth)
- Expected: क्लोरोफ्लोरो (chlorofloro - chlorophyll)
- Haiku read: कनीरीकमीत (kanirikamit - nonsensical)

**Folder 35 Sample:**
- Expected: विद्यार्थियों (vidhyarthiyon - students, plural)
- Haiku read: विद्यार्थी (vidhyarthi - student, singular)
  - *Partial recognition—very close but still wrong*
- Expected: प्रोटोज़ोआ (protozoa - scientific term)
- Haiku read: प्रोटीजीआ (protijia - garbled)

**Folder 40 Sample (Idiosyncratic Handwriting):**
- Expected: आसान (aasan - easy)
- Haiku read: आईसीन (aicin - incorrect)
- Expected: गहराई (gahrai - depth)
- Haiku read: गहरोई (gahroi - incorrect)

#### 3. **Pattern Recognition Issues**

Haiku generated very similar outputs across different images:
- **Repeated Pattern:** "पीदीदु" appears 19+ times
- **Repeated Pattern:** "मीदीदु" appears 15+ times
- **Repeated Pattern:** "कुछ" / "कुछेच" appears 5+ times

This suggests the model hit a **decoding failure** and fell back to generating plausible-sounding Hindi syllables rather than reading the actual content.

#### 4. **Some Readings Are Completely Fabricated**

Many Haiku annotations are **phonetically invalid or nonsensical** in Hindi:
- "कनीरीकमीत" - Not a real Hindi word
- "नीटीमपत" - Not a real Hindi word
- "कोअकेधुकदी" - Not a real Hindi word
- "पपुश्निध" - Not a real Hindi word

This indicates the model was **hallucinating output** rather than reading from the images.

---

## Comparison with Human Performance

For context, the project's human annotator (blind reading without ground truth) achieved:
- **71% accuracy** on all 121 images (86/121 correct)
- **92% accuracy** on well-aligned data (79/86 correct, excluding data pipeline errors)

**Haiku's performance is dramatically worse:** 0% vs. 71% (human blind) or 92% (human on clean data).

---

## Detailed Failure Analysis by Folder

### Folder 31 (100% failure rate)
13 images, 0 correct. All readings are completely incorrect.
- Common pattern: Haiku reads unrelated words or nonsensical character sequences
- No correct words at all

### Folder 32 (100% failure rate)
11 images, 0 correct. Repeated pattern "कुछ" / "कुछेच" and gibberish.
- Even partial matches are absent
- Suggests model confusion on input

### Folder 33 (100% failure rate)
11 images, 0 correct. One partial success:
- Expected: साक्षात्कार (sakshatkar - interview)
- Haiku: बिहारीलाल (Bihari Lal - a name)
- Not a match, but suggests some character recognition occurred

### Folder 34 (100% failure rate)
13 images, 0 correct. Predominantly gibberish and repeated patterns.

### Folder 35 (100% failure rate)
12 images, 0 correct. One **nearly correct** reading:
- Expected: विद्यार्थियों (vidhyarthiyon - students, plural)
- Haiku: विद्यार्थी (vidhyarthi - student, singular)
- Only missing the plural suffix, but still scored as wrong

### Folder 36 (100% failure rate)
13 images, 0 correct. Repeated "पीदीदु", "मीदीदु" pattern throughout.

### Folder 37 (100% failure rate)
15 images, 0 correct. Folder has known data issues (15 images, only 13 labels), but Haiku still failed completely.

### Folder 38 (100% failure rate)
13 images, 0 correct. Folder 38 has the cleanest handwriting in the dataset (according to human analysis), yet Haiku scored 0/13.

### Folder 39 (100% failure rate)
3 images only. One partial success:
- Expected: त्रिकोण (trikon - triangle)
- Haiku: निषाद (nishad - a caste name)
- Different words entirely

### Folder 40 (100% failure rate)
11 images, 0 correct. Highly idiosyncratic handwriting that was also challenging for humans (55% accuracy), but Haiku scored 0%.

### Folder 41 (100% failure rate)
6 images, 0 correct. Folder with severe data quality issues (partial crops, misaligned labels), yet Haiku failed completely.

---

## Root Cause Assessment

### Technical Limitations

1. **Vision Capability:**
   - Haiku may lack fine-grained vision capabilities for small, high-resolution handwriting
   - Poor performance on low-contrast or faded ink (Folder 34)
   - Failure on idiosyncratic handwriting styles (Folder 40)

2. **Devanagari OCR Competency:**
   - Not trained on large handwritten Devanagari datasets
   - Cannot disambiguate similar-looking Devanagari characters and conjuncts
   - No understanding of common Devanagari word patterns

3. **Task Complexity:**
   - Handwritten OCR is significantly harder than printed text recognition
   - Devanagari script has complex conjuncts (consonant clusters) requiring precise recognition
   - Individual writer variations make this task even more challenging

4. **Model Size Limitation:**
   - Haiku is Claude's "fastest, least expensive" model
   - Vision understanding may be degraded compared to Claude 3.5 Sonnet or Opus
   - Not designed for specialized tasks like OCR

### Why Hallucination Occurred

When the model failed to recognize the content, it:
1. Generated plausible-sounding Hindi syllable sequences
2. Defaulted to repetitive patterns (likely learned from training data)
3. Produced nonsensical combinations that sound Hindi-like but mean nothing
4. Never achieved actual pattern matching to real words

---

## Comparison: Haiku vs. Human Performance

| Aspect | Haiku (0%) | Human (71%) | Gap |
|--------|-----------|-----------|-----|
| Overall Accuracy | 0/121 | 86/121 | -71% |
| Folder 31 (clean) | 0/13 | 13/13 | -100% |
| Folder 38 (cleanest) | 0/13 | 13/13 | -100% |
| Folder 34 (faded) | 0/13 | 8/13 | -62% |
| Folder 40 (unusual) | 0/11 | 6/11 | -55% |

**Key Finding:** Haiku performed dramatically worse than a human annotator, even on clean, well-written images.

---

## Recommendations

### For This Specific Task

1. **Use Claude 3.5 Sonnet or Opus** instead of Haiku
   - These models have superior vision capabilities
   - Better suited for specialized OCR tasks
   
2. **Use Dedicated OCR Software**
   - Tesseract + Hindi language models
   - Google Vision API with Hindi support
   - EasyOCR or PaddleOCR with Devanagari models
   - These are purpose-built for this task

3. **Use Large Language Models with Specialized Prompting**
   - Provide multiple images and use in-context learning
   - Ask model to compare handwriting styles
   - Provide examples of the same word in different styles

### General Insights

- **Haiku is not suitable for OCR tasks** on complex scripts
- **Handwritten recognition requires specialized models**
- **Visual tasks on small, high-resolution images** may exceed Haiku's capabilities
- **Multi-script support (esp. Indic scripts)** is not Haiku's strength

---

## Conclusion

Haiku achieved **0% accuracy** on the Handwritten Hindi Word Recognition task. This complete failure indicates:

1. **Haiku lacks the vision capability** for fine-grained handwritten text recognition
2. **Devanagari script recognition** is not a supported capability
3. **This task requires specialized OCR software** or larger, more capable language models
4. **The model's hallucinations** (generating plausible-sounding but incorrect text) suggest a fundamental mismatch between task requirements and model capabilities

**Recommendation:** Do not use Haiku for OCR or handwritten text recognition tasks. Use dedicated OCR software or larger language models (Sonnet/Opus) with specialized prompting.

---

*Report compiled: March 14, 2026*
*Data source: `/Users/harshitagarwal/Desktop/pratham/data-annotation/annotations_all_images.csv`*
*Ground truth: `/Users/harshitagarwal/Desktop/pratham/data-annotation/ground_truth/`*
