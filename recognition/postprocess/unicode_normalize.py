"""NFC Unicode normalization for Devanagari OCR outputs."""

from __future__ import annotations

import unicodedata


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Apply Unicode normalization (default NFC — canonical composition)."""
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        form = "NFC"
    return unicodedata.normalize(form, text.strip())
