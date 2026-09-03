"""
ai/inference.py

InferenceEngine abstraction:

    InferenceEngine
        |
        |---- YOLOSegInferenceEngine   (used only if models/sonarshield_yoloseg.pt
        |                                exists AND loads successfully)
        |
        |---- PrototypeInferenceEngine (heuristic OpenCV/NumPy fallback, always available)

get_active_engine() is the single entry point the rest of the app should call.
It NEVER silently mislabels which engine actually produced a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ai.prototype_engine import run_prototype_detection, mask_to_rle
from ai.yolo_seg_engine import try_load_yolo_seg, MODEL_PATH


@dataclass
class EngineStatus:
    mode: str            # "yolo_seg" | "prototype"
    label: str            # human-readable, safe to show in UI
    detail: str


def get_engine_status() -> EngineStatus:
    engine = try_load_yolo_seg()
    if engine is not None:
        return EngineStatus(
            mode="yolo_seg",
            label="Inference Mode: YOLO-Seg",
            detail=f"Trained checkpoint loaded from {MODEL_PATH.name}.",
        )
    if MODEL_PATH.exists():
        return EngineStatus(
            mode="prototype",
            label="Inference Mode: Prototype (checkpoint found but failed to load)",
            detail="A file exists at models/sonarshield_yoloseg.pt but could not be "
                   "loaded (missing PyTorch or incompatible checkpoint). Falling back "
                   "to the prototype heuristic engine.",
        )
    return EngineStatus(
        mode="prototype",
        label="Inference Mode: Prototype",
        detail="No trained YOLO-Seg checkpoint installed at models/sonarshield_yoloseg.pt. "
               "Using the OpenCV/NumPy heuristic prototype engine.",
    )


def run_inference(enhanced_gray: np.ndarray) -> tuple[list[dict], EngineStatus]:
    """
    Runs whichever engine is currently active. Returns (raw_detections, status).
    raw_detections: list of {bbox, mask, features, detection_confidence}
    """
    status = get_engine_status()
    if status.mode == "yolo_seg":
        engine = try_load_yolo_seg()
        try:
            detections = engine.run(enhanced_gray)
            return detections, status
        except NotImplementedError:
            # Checkpoint loads but forward-pass isn't wired up yet -> fall back
            # honestly rather than pretending inference happened.
            fallback_status = EngineStatus(
                mode="prototype",
                label="Inference Mode: Prototype (YOLO-Seg forward pass not implemented)",
                detail="A checkpoint was found and loaded, but ai/yolo_seg_engine.py's "
                       "run() method still needs the actual forward-pass/postprocessing "
                       "implemented for your specific training export.",
            )
            return run_prototype_detection(enhanced_gray), fallback_status

    return run_prototype_detection(enhanced_gray), status
