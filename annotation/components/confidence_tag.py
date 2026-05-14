"""Annotator confidence tag widget for Streamlit."""

from __future__ import annotations

import streamlit as st


CONFIDENCE_LEVELS = ("high", "medium", "low")


def confidence_tag_input(key_prefix: str = "conf") -> str:
    """Render radio for high/medium/low image-quality or label-confidence tag."""
    return st.radio(
        "Label confidence (handwriting legibility / certainty)",
        CONFIDENCE_LEVELS,
        horizontal=True,
        key=f"{key_prefix}_confidence",
    )
