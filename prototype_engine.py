"""
ai/prototype_engine.py

PrototypeInferenceEngine
-------------------------
A heuristic, classical computer-vision detector used ONLY when no trained
YOLO-Seg checkpoint is available (see ai/inference.py). It is NOT a neural
network and must never be presented as one.

Method (blob analysis on the CLAHE-enhanced grayscale sonar image):
    1. Local-contrast anomaly map = |enhanced - blurred(enhanced)|
    2. Threshold -> binary mask
    3. Connected-component labelling (scipy.ndimage.label) -> candidate blobs
    4. Filter by area, compute per-blob geometry + contrast features
    5. Each blob's pixel mask IS the "segmentation" for that object
       (pixel-level, not just a bounding box)

Every detection returned by this engine carries "inference_mode": "prototype"
which the UI must surface next to every result.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

from backend.classification import BlobFeatures

MIN_AREA_PX = 120
MAX_AREA_FRACTION = 0.25
CONTRAST_THRESHOLD = 14
MAX_DETECTIONS = 12


def _box_blur(gray: np.ndarray, radius: int = 6) -> np.ndarray:
    if _HAS_SCIPY:
        return ndimage.uniform_filter(gray.astype(np.float32), size=2 * radius + 1)
    # naive fallback (small radius only, adequate for demo-sized images)
    pad = np.pad(gray.astype(np.float32), radius, mode="edge")
    out = np.zeros_like(gray, dtype=np.float32)
    k = (2 * radius + 1) ** 2
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            out += pad[i:i + gray.shape[0], j:j + gray.shape[1]]
    return out / k


def _label(binary: np.ndarray):
    if _HAS_SCIPY:
        return ndimage.label(binary)
    # extremely small fallback flood-fill labeller (only used if scipy missing)
    labels = np.zeros(binary.shape, dtype=np.int32)
    current = 0
    visited = np.zeros(binary.shape, dtype=bool)
    h, w = binary.shape
    for y in range(h):
        for x in range(w):
            if binary[y, x] and not visited[y, x]:
                current += 1
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    labels[cy, cx] = current
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
    return labels, current


def _edge_straightness(mask: np.ndarray) -> float:
    """Prototype proxy for rectilinearity: ratio of mask area to its bbox area
    compared against a filled ellipse of the same bbox (ellipses -> low score,
    rectangles -> high score)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    fill = mask.sum() / (h * w)
    # a filled ellipse has fill ratio ~0.785; rectangles approach 1.0
    return float(np.clip((fill - 0.785) / (1.0 - 0.785), 0.0, 1.0)) if fill > 0.785 else 0.0


def run_prototype_detection(enhanced_gray: np.ndarray) -> list[dict]:
    """
    Returns a list of raw detections:
      { bbox: [x,y,w,h], mask: np.bool_ array (full-image sized), features: BlobFeatures,
        blob_mean, surround_mean }
    """
    blurred = _box_blur(enhanced_gray, radius=6)
    diff = np.abs(enhanced_gray.astype(np.float32) - blurred)
    binary = diff > CONTRAST_THRESHOLD

    labels, num = _label(binary)
    total_px = enhanced_gray.shape[0] * enhanced_gray.shape[1]

    candidates = []
    for lbl in range(1, num + 1):
        mask = labels == lbl
        area = int(mask.sum())
        if area < MIN_AREA_PX or area > MAX_AREA_FRACTION * total_px:
            continue

        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        bw, bh = int(x1 - x0 + 1), int(y1 - y0 + 1)

        blob_mean = float(enhanced_gray[mask].mean())
        pad = 8
        sy0, sy1 = max(0, y0 - pad), min(enhanced_gray.shape[0], y1 + pad + 1)
        sx0, sx1 = max(0, x0 - pad), min(enhanced_gray.shape[1], x1 + pad + 1)
        surround_region = enhanced_gray[sy0:sy1, sx0:sx1]
        surround_mean = float(surround_region.mean()) if surround_region.size else blob_mean

        contrast = abs(blob_mean - surround_mean) / 255.0
        fill_ratio = area / (bw * bh)
        elongation = max(bw, bh) / max(1, min(bw, bh))

        features = BlobFeatures(
            area_px=area, bbox_w=bw, bbox_h=bh, fill_ratio=fill_ratio,
            contrast=contrast, elongation=elongation,
            edge_straightness=_edge_straightness(mask[y0:y1 + 1, x0:x1 + 1]),
        )

        detection_confidence = float(np.clip(0.35 + contrast * 2.2 + (fill_ratio * 0.1), 0.2, 0.98))

        candidates.append({
            "bbox": [int(x0), int(y0), bw, bh],
            "mask": mask,
            "features": features,
            "detection_confidence": round(detection_confidence, 4),
        })

    candidates.sort(key=lambda c: c["detection_confidence"], reverse=True)
    return candidates[:MAX_DETECTIONS]


def mask_to_rle(mask: np.ndarray) -> list[int]:
    """Compact run-length encoding so masks are cheap to store/export as JSON."""
    flat = mask.flatten(order="C")
    if flat.size == 0:
        return []
    changes = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate(([0], changes, [flat.size])))
    starts_with_one = bool(flat[0])
    return [int(flat.size)] + ([0] if not starts_with_one else []) + [int(r) for r in runs]
