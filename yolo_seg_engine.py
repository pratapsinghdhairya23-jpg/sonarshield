"""
ai/yolo_seg_engine.py

YOLOSegInferenceEngine
------------------------
Thin wrapper around a trained YOLO-Seg-compatible PyTorch checkpoint.

This module is imported lazily and guarded with try/except everywhere it is
used, so the application starts and runs fully in Prototype Mode on machines
that don't have PyTorch / a trained checkpoint installed (per project spec:
"offline / no GPU is not a reason to abandon the UI").

To activate real inference:
    1. pip install torch ultralytics   (or your training framework of choice)
    2. Place a trained checkpoint at models/sonarshield_yoloseg.pt
    3. Restart the app - ai/inference.py will auto-detect the file and switch
       "Inference Mode" from "Prototype" to "YOLO-Seg" in the UI.

Nothing in this file fabricates accuracy numbers or claims training has
happened; get_status() reports only what is verifiably true in the current
environment (file present + library importable + model loads without error).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sonarshield_yoloseg.pt"


class YOLOSegUnavailable(RuntimeError):
    pass


class YOLOSegInferenceEngine:
    """Loaded only when MODEL_PATH exists AND torch/ultralytics import successfully."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise YOLOSegUnavailable(f"No trained checkpoint found at {self.model_path}")
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise YOLOSegUnavailable(
                "PyTorch is not installed in this environment. "
                "Install `torch` (and `ultralytics` if using a YOLO-Seg checkpoint) to enable real inference."
            ) from e
        try:
            # Generic loading path - adapt to your actual training framework/export format.
            # Example for an Ultralytics YOLO-Seg export:
            #   from ultralytics import YOLO
            #   self._model = YOLO(str(self.model_path))
            import torch
            self._model = torch.load(self.model_path, map_location="cpu")
        except Exception as e:
            raise YOLOSegUnavailable(f"Failed to load checkpoint: {e}") from e

    def run(self, rgb_image) -> list[dict]:
        """
        Must return the same detection dict shape as
        ai.prototype_engine.run_prototype_detection(), i.e. a list of:
            { bbox: [x,y,w,h], mask: np.bool_ array, features: BlobFeatures,
              detection_confidence: float }
        Implement the actual forward pass + postprocessing here once a real
        checkpoint/training pipeline exists. Left unimplemented intentionally
        so this module never silently returns fabricated results.
        """
        raise NotImplementedError(
            "YOLOSegInferenceEngine.run() must be implemented once a trained "
            "SONARSHIELD YOLO-Seg checkpoint and its pre/post-processing are available."
        )


def try_load_yolo_seg() -> Optional[YOLOSegInferenceEngine]:
    try:
        return YOLOSegInferenceEngine()
    except YOLOSegUnavailable:
        return None
