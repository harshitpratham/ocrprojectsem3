"""
YOLO detection post-processing v2: NMS tuning, optional box merging, reading order.

Known failure mode (documented in v2 eval): matra-aware merge can **over-merge**
adjacent words on ~3% of dense rows when horizontal gap is smaller than merge_gap_px.
Tune `merge_gap_px` per layout or disable merge with enable_merge=False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class Box:
    """Axis-aligned word box in pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def iou(self, other: Box) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        a1 = self.width * self.height
        a2 = other.width * other.height
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0


def nms(boxes: Sequence[Box], iou_threshold: float = 0.42) -> list[Box]:
    """Class-agnostic NMS sorted by confidence."""
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b.conf, reverse=True)
    keep: list[Box] = []
    while sorted_boxes:
        cur = sorted_boxes.pop(0)
        keep.append(cur)
        remain = []
        for b in sorted_boxes:
            if cur.iou(b) < iou_threshold:
                remain.append(b)
        sorted_boxes = remain
    return keep


def merge_adjacent_reading_order(
    boxes: Sequence[Box],
    merge_gap_px: float = 8.0,
    vertical_overlap_min: float = 0.35,
) -> list[Box]:
    """
    Merge boxes on the same line that are separated by <= merge_gap_px horizontally.
    vertical_overlap_min: min IoU of vertical spans (as 1D segments) to consider same row.
    """
    if len(boxes) < 2:
        return list(boxes)

    def vertical_overlap_ratio(a: Box, b: Box) -> float:
        ay1, ay2 = a.y1, a.y2
        by1, by2 = b.y1, b.y2
        inter = max(0.0, min(ay2, by2) - max(ay1, by1))
        h = max(1e-6, min(ay2 - ay1, by2 - by1))
        return inter / h

    sorted_ltr = sorted(boxes, key=lambda b: (b.cy, b.cx))
    merged: list[Box] = []
    i = 0
    n = len(sorted_ltr)
    while i < n:
        cur = sorted_ltr[i]
        j = i + 1
        while j < n:
            nxt = sorted_ltr[j]
            if vertical_overlap_ratio(cur, nxt) < vertical_overlap_min:
                break
            gap = nxt.x1 - cur.x2
            if gap <= merge_gap_px and nxt.cy <= cur.y2 and nxt.cy >= cur.y1:
                cur = Box(
                    min(cur.x1, nxt.x1),
                    min(cur.y1, nxt.y1),
                    max(cur.x2, nxt.x2),
                    max(cur.y2, nxt.y2),
                    conf=min(cur.conf, nxt.conf),
                )
                j += 1
            else:
                break
        merged.append(cur)
        i = j
    return merged


def sort_reading_order(boxes: Sequence[Box], line_tol_px: float = 0.35) -> list[Box]:
    """
    Sort boxes: primary key = text row (bucket by y-center), secondary = x.
    line_tol_px is interpreted as fraction of median box height when line_tol_px < 1,
    else as absolute pixels if >= 1 (callers typically pass median_height * 0.35).
    """
    if not boxes:
        return []
    heights = [b.height for b in boxes if b.height > 0]
    med_h = sorted(heights)[len(heights) // 2] if heights else 10.0
    tol = line_tol_px * med_h if line_tol_px < 1.0 else line_tol_px

    def line_key(b: Box) -> int:
        return int(round(b.cy / tol))

    return sorted(boxes, key=lambda b: (line_key(b), b.x1))


def postprocess_detections(
    xyxy_conf: Iterable[tuple[float, float, float, float, float]],
    iou_nms: float = 0.42,
    enable_merge: bool = True,
    merge_gap_px: float = 8.0,
) -> list[Box]:
    """Full v2 post-process: boxes (x1,y1,x2,y2,conf) -> sorted merged boxes."""
    raw = [Box(x1, y1, x2, y2, c) for x1, y1, x2, y2, c in xyxy_conf]
    kept = nms(raw, iou_threshold=iou_nms)
    if enable_merge:
        kept = merge_adjacent_reading_order(kept, merge_gap_px=merge_gap_px)
    return sort_reading_order(kept, line_tol_px=0.35)
