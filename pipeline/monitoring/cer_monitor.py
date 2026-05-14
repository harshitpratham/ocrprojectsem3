"""
Rolling CER monitor — aggregates batch scores and triggers alert callback when above SLO.

SLO thresholds are placeholders until calibrated against live Pratham traffic.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CERMonitor:
    """In-memory rolling window of recent CER samples."""

    window_size: int = 200
    samples: list[float] = field(default_factory=list)
    slo_cer: float = 0.45
    alert_fn: Callable[[str], None] | None = None

    def record(self, cer: float, meta: dict[str, Any] | None = None) -> None:
        self.samples.append(float(cer))
        if len(self.samples) > self.window_size:
            self.samples.pop(0)
        if len(self.samples) >= 10:
            med = statistics.median(self.samples)
            if med > self.slo_cer and self.alert_fn:
                self.alert_fn(
                    f"CER median {med:.4f} exceeds SLO {self.slo_cer:.4f} "
                    f"(n={len(self.samples)}, meta={meta})"
                )

    def snapshot(self) -> dict[str, Any]:
        if not self.samples:
            return {"n": 0, "median": None, "mean": None}
        return {
            "n": len(self.samples),
            "median": statistics.median(self.samples),
            "mean": statistics.mean(self.samples),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
