"""
backend/tracking.py

Prototype Cross-Scan Tracking.

HONESTY NOTE: this is a simple nearest-neighbour matcher on bounding-box
centroid distance + class-name agreement across the *most recent* previous
scans in the database. It is not a learned re-identification / feature-
matching model, and is always labelled "Prototype Cross-Scan Tracking" in
the UI.
"""

from __future__ import annotations

import math
import uuid
from typing import Optional

MATCH_DISTANCE_PX = 60.0


def _centroid(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def assign_tracking_id(new_bbox: list[float], new_class: str,
                        previous_detections: list[dict]) -> tuple[str, str]:
    """
    previous_detections: list of dicts with keys 'tracking_id', 'bbox_json' (list), 'class_name'
    Returns (tracking_id, track_status) where track_status in
    {'new', 'existing', 'possibly_moved'}.
    """
    cx, cy = _centroid(new_bbox)
    best_match = None
    best_dist = float("inf")

    for prev in previous_detections:
        if prev["class_name"] != new_class:
            continue
        pcx, pcy = _centroid(prev["bbox"])
        dist = math.hypot(cx - pcx, cy - pcy)
        if dist < best_dist:
            best_dist = dist
            best_match = prev

    if best_match is not None and best_dist <= MATCH_DISTANCE_PX:
        return best_match["tracking_id"], "existing"
    if best_match is not None and best_dist <= MATCH_DISTANCE_PX * 3:
        return best_match["tracking_id"], "possibly_moved"

    return "TRK-" + uuid.uuid4().hex[:8].upper(), "new"
