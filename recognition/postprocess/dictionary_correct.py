"""ASER-style vocabulary correction (Levenshtein-lite greedy match)."""

from __future__ import annotations

from pathlib import Path


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_vocab(path: Path | None, max_words: int = 4200) -> set[str]:
    """Load newline-separated Hindi words; if missing, return small stub set."""
    words: set[str] = set()
    if path and path.is_file():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if i >= max_words:
                break
            w = line.strip()
            if w:
                words.add(w)
    if not words:
        words = {"पाठ", "गणित", "घर", "स्कूल", "बच्चा", "पढ़ना"}
    return words


def correct_with_vocab(
    text: str,
    vocab_path: Path | None = None,
    max_dist: int = 1,
) -> str:
    """
    If `text` is not in vocabulary, replace with closest vocab word within max_dist.
    O(|vocab|) — fine for ~4k ASER words.
    """
    vocab = load_vocab(vocab_path)
    t = text.strip()
    if t in vocab:
        return t
    best = t
    best_d = max_dist + 1
    for w in vocab:
        d = _levenshtein(t, w)
        if d < best_d:
            best_d = d
            best = w
    return best if best_d <= max_dist else t
