"""
backend/classification.py

Object-category classification and Natural-vs-Artificial estimation.

IMPORTANT HONESTY NOTE
-----------------------
This module implements a *prototype heuristic classifier*, not a trained
neural network. It derives a category and a natural/artificial probability
from simple blob-geometry and intensity-contrast features (area, aspect
ratio, edge regularity, sonar-return contrast). It is deliberately labelled
as such everywhere it is surfaced in the UI ("Prototype Classification").

It exists so the full SONARSHIELD pipeline (detection -> classification ->
natural/artificial -> anomaly -> risk -> report) is runnable end-to-end
before a trained YOLO-Seg / classification head is available. When a real
trained model is plugged in (see ai/yolo_seg_engine.py), that model's class
head should replace the `classify_object` call below.

Permitted classes come from the datasets referenced in the SIH PPT:
  - AI4Shipwrecks   -> "Shipwreck"
  - GhostVision      -> "Fishing Gear"
  - SubPipe            -> "Pipeline/Cable"
Plus general categories used by the prototype heuristic pipeline itself:
  - "Marine Debris", "Man-made Object", "Natural Formation", "Unknown Anomaly"
"""

from __future__ import annotations

from dataclasses import dataclass

CLASSES = [
    "Shipwreck",
    "Pipeline/Cable",
    "Fishing Gear",
    "Marine Debris",
    "Man-made Object",
    "Natural Formation",
    "Unknown Anomaly",
]


@dataclass
class BlobFeatures:
    area_px: int
    bbox_w: int
    bbox_h: int
    fill_ratio: float          # blob area / bbox area -> regularity of shape
    contrast: float            # 0..1, |blob mean - surround mean| / 255
    elongation: float          # max(w,h)/min(w,h)
    edge_straightness: float   # 0..1, how rectilinear the boundary is (prototype proxy)


def classify_object(f: BlobFeatures) -> dict:
    """
    Prototype heuristic classification.

    Returns a dict with:
      class_name, natural_prob, artificial_prob, final_type, class_confidence
    """
    artificial_score = 0.0

    # Long, straight, high-contrast, elongated -> pipeline/cable-like
    if f.elongation >= 4.0 and f.edge_straightness >= 0.6:
        artificial_score += 0.35

    # High fill ratio + straight edges -> man-made rectilinear structure
    if f.fill_ratio >= 0.65 and f.edge_straightness >= 0.5:
        artificial_score += 0.30

    # Strong sonar return contrast is typical of metal/hard artificial surfaces
    artificial_score += min(f.contrast, 1.0) * 0.35

    # Very irregular, low-contrast, low fill ratio -> natural seabed feature
    natural_score = (1.0 - f.fill_ratio) * 0.4 + (1.0 - f.edge_straightness) * 0.3 \
        + max(0.0, 0.4 - f.contrast) * 0.3

    total = artificial_score + natural_score
    if total <= 0:
        artificial_prob, natural_prob = 0.5, 0.5
    else:
        artificial_prob = artificial_score / total
        natural_prob = natural_score / total

    # normalize to sum to 1
    s = artificial_prob + natural_prob
    artificial_prob, natural_prob = artificial_prob / s, natural_prob / s

    if abs(artificial_prob - natural_prob) < 0.12:
        final_type = "Uncertain"
    elif artificial_prob > natural_prob:
        final_type = "Artificial"
    else:
        final_type = "Natural"

    # category assignment (only meaningful when final_type suggests artificial;
    # natural/uncertain objects default to Natural Formation / Unknown Anomaly)
    class_confidence = max(artificial_prob, natural_prob)

    if final_type == "Natural":
        class_name = "Natural Formation"
    elif final_type == "Uncertain":
        class_name = "Unknown Anomaly"
    else:
        if f.elongation >= 5.0 and f.edge_straightness >= 0.65:
            class_name = "Pipeline/Cable"
        elif f.area_px >= 4000 and f.fill_ratio >= 0.55:
            class_name = "Shipwreck"
        elif 0.25 <= f.fill_ratio <= 0.6 and f.elongation < 3.0:
            class_name = "Fishing Gear"
        elif f.fill_ratio < 0.35:
            class_name = "Marine Debris"
        else:
            class_name = "Man-made Object"

        # low classification confidence on an artificial-looking blob that
        # doesn't cleanly match a known pattern -> flag as unknown rather
        # than forcing it into a category (per anomaly-flagging requirement)
        if class_confidence < 0.55:
            class_name = "Unknown Anomaly"

    return {
        "class_name": class_name,
        "natural_prob": round(float(natural_prob), 4),
        "artificial_prob": round(float(artificial_prob), 4),
        "final_type": final_type,
        "class_confidence": round(float(class_confidence), 4),
    }
