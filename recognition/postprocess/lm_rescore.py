"""
Character-level n-gram language model rescoring (stub + greedy implementation).

v2 note: On **Pratham field** text, LM trained on HindiSeg-skewed corpus can **regress**
CER (~+0.6pp in release eval). Use `enabled=False` for field inputs by default.

For production, plug in KenLM / custom ARPA or train a char-LM on in-domain text.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _default_char_ngrams(corpus_lines: Iterable[str], n: int = 3) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for line in corpus_lines:
        s = f"^{line.strip()}$"
        for i in range(len(s) - n + 1):
            counts[tuple(s[i : i + n])] += 1
    return counts


class CharNGramLM:
    """Tiny backoff-free char n-gram model for demonstration."""

    def __init__(self, n: int = 3):
        self.n = n
        self.counts: dict[tuple[str, ...], int] = defaultdict(int)
        self.V = 2000  # loose vocab cap for smoothing

    def fit(self, lines: Iterable[str]) -> None:
        self.counts = _default_char_ngrams(lines, self.n)

    def log_prob(self, text: str) -> float:
        s = f"^{text.strip()}$"
        if len(s) < self.n:
            return 0.0
        logp = 0.0
        for i in range(len(s) - self.n + 1):
            ng = tuple(s[i : i + self.n])
            c = self.counts.get(ng, 0)
            logp += math.log((c + 1) / (sum(self.counts.values()) + self.V))
        return logp


def load_corpus_stub() -> list[str]:
    """Placeholder corpus if no file — Hindi common function words only."""
    return list("अ आ इ ई उ ऊ ए ऐ ओ औ क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह ा ी ू े ो ् ".split())


def rescore_with_char_lm(
    hypothesis: str,
    alternatives: list[str] | None = None,
    *,
    corpus_path: Path | None = None,
    n: int = 3,
    beam_size: int = 5,
    enabled: bool = True,
) -> str:
    """
    Given primary hypothesis and optional local alternatives, pick highest LM score.

    If `alternatives` is None, returns `hypothesis` unchanged (or normalized path only).
    `beam_size` caps how many strings are scored when alternatives is long.
    """
    if not enabled:
        return hypothesis
    lm = CharNGramLM(n=n)
    lines = load_corpus_stub()
    if corpus_path and corpus_path.is_file():
        lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
    lm.fit(lines)

    candidates = [hypothesis]
    if alternatives:
        candidates.extend(alternatives[: max(0, beam_size - 1)])
    best = hypothesis
    best_lp = -1e18
    for c in candidates:
        lp = lm.log_prob(c)
        if lp > best_lp:
            best_lp = lp
            best = c
    return best


def suggest_alternatives_from_beam_placeholder(hypothesis: str) -> list[str]:
    """Placeholder: real integration would pass decoder beam list from TrOCR.generate."""
    # trivial local edits for demo
    alts = [hypothesis]
    if hypothesis:
        alts.append(hypothesis.rstrip("।"))
    return alts
