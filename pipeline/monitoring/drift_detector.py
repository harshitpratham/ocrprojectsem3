"""
Simple input drift detector: mean luminance + edge density vs baseline.

Baseline should be computed on a reference crop set. This is scaffolding only —
thresholds are not calibrated to production traffic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def image_stats(rgb: np.ndarray) -> dict[str, float]:
    """rgb: HxWx3 uint8."""
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    return {
        "mean_luma": float(gray.mean()),
        "edge_density": float((gx + gy) / 2.0),
    }


@dataclass
class DriftDetector:
    baseline: dict[str, float]
    luma_tol: float = 18.0
    edge_tol: float = 4.0

    @classmethod
    def from_json(cls, path: Path) -> DriftDetector:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(baseline=data)

    def check(self, stats: dict[str, float]) -> tuple[bool, dict[str, Any]]:
        dl = abs(stats["mean_luma"] - self.baseline["mean_luma"])
        de = abs(stats["edge_density"] - self.baseline["edge_density"])
        flags = []
        if dl > self.luma_tol:
            flags.append("luma_drift")
        if de > self.edge_tol:
            flags.append("edge_drift")
        return len(flags) > 0, {
            "flags": flags,
            "delta_luma": dl,
            "delta_edge": de,
        }


def write_stub_baseline(path: Path, rgb_arrays: list[np.ndarray]) -> None:
    """Aggregate mean stats across reference images."""
    agg_l, agg_e = [], []
    for a in rgb_arrays:
        s = image_stats(a)
        agg_l.append(s["mean_luma"])
        agg_e.append(s["edge_density"])
    baseline = {
        "mean_luma": float(np.mean(agg_l)) if agg_l else 128.0,
        "edge_density": float(np.mean(agg_e)) if agg_e else 5.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
