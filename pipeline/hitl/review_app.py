#!/usr/bin/env python3
"""
Human-in-the-loop review for low-confidence TrOCR predictions.

Expects a CSV with columns: image_path, predicted_text, confidence (optional), ground_truth (optional).

  streamlit run pipeline/hitl/review_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hindi OCR — HITL review", layout="wide")

CSV_PATH = Path(os.getenv("HITL_CSV", "data/results/v2/hitl_queue_sample.csv"))


def main():
    st.title("HITL word review (v2)")
    st.caption(
        "v2 release: mandatory for Pratham field documents when word-exact accuracy is below deployment threshold."
    )

    if not CSV_PATH.is_file():
        st.warning(
            f"No CSV at {CSV_PATH}. Export low-confidence predictions from the batch job, "
            "or create a sample file with columns image_path,predicted_text,confidence."
        )
        return

    df = pd.read_csv(CSV_PATH)
    if "image_path" not in df.columns or "predicted_text" not in df.columns:
        st.error("CSV must include image_path and predicted_text")
        return

    idx = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
    row = df.iloc[int(idx)]
    imp = Path(row["image_path"])
    if imp.is_file():
        st.image(str(imp), caption=imp.name, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Model prediction", value=str(row["predicted_text"]), height=120, disabled=True)
    with col2:
        correction = st.text_input("Human correction (Unicode Hindi)", value="")
    if st.button("Submit correction (local demo only)"):
        st.success(f"Recorded locally: {correction!r} — wire to storage in production.")


if __name__ == "__main__":
    main()
