"""
Multi-annotator agreement metrics (Cohen's kappa) for paired Hindi labels.

Expects a CSV with columns: item_id, annotator_a, annotator_b, label_a, label_b
or wide format: item_id, label_alice, label_bob
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable


def cohen_kappa(labels_a: list[Hashable], labels_b: list[Hashable]) -> float:
    """Cohen's kappa for two raters, same length aligned lists."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must align")
    n = len(labels_a)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n
    ct_a = Counter(labels_a)
    ct_b = Counter(labels_b)
    p_e = sum((ct_a[k] / n) * (ct_b[k] / n) for k in set(ct_a) | set(ct_b))
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def percent_agreement(labels_a: list[Hashable], labels_b: list[Hashable]) -> float:
    if not labels_a:
        return 0.0
    return sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a)
