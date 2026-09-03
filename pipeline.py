"""
backend/pipeline.py

Orchestrates the full SONARSHIELD pipeline for a single uploaded scan:

    SSS Image
      -> preprocessing.sonar_preprocessor.preprocess_pipeline
      -> ai.inference.run_inference               (YOLO-Seg or Prototype)
      -> backend.classification.classify_object
      -> backend.anomaly_engine.compute_anomaly
      -> backend.risk_engine.compute_risk
      -> backend.tracking.assign_tracking_id       (vs. previous scans in DB)
      -> backend.database.insert_scan / insert_detection

This is the single function app.py's "Analyze Scan" page calls, so the
Streamlit layer stays purely presentational.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ai.inference import run_inference
from ai.prototype_engine import mask_to_rle
from backend import database as db
from backend.anomaly_engine import compute_anomaly
from backend.classification import classify_object
from backend.risk_engine import compute_risk
from backend.tracking import assign_tracking_id
from preprocessing.sonar_preprocessor import preprocess_pipeline


def run_full_pipeline(
    rgb_image: np.ndarray,
    filename: str,
    latitude: Optional[float],
    longitude: Optional[float],
    depth_m: Optional[float],
    timestamp_label: str,
    dataset_source: str,
    image_path: str,
    preview_path: str,
) -> dict[str, Any]:
    scan_id = db.new_scan_id()

    # 1. Preprocessing
    stages = preprocess_pipeline(rgb_image)

    # 2. Detection + segmentation (YOLO-Seg if available, else prototype)
    raw_detections, engine_status = run_inference(stages["enhanced"])

    # gather previous detections (across all prior scans) for cross-scan tracking
    previous = [
        {
            "tracking_id": d["tracking_id"],
            "bbox": __import__("json").loads(d["bbox_json"]),
            "class_name": d["class_name"],
        }
        for d in db.get_all_detections()
    ]

    location_source = "user_provided" if (latitude is not None and longitude is not None) else "unavailable"

    detections_out = []
    confidences = []
    high_risk_count = 0
    unknown_count = 0

    for i, raw in enumerate(raw_detections):
        object_id = db.new_object_id(scan_id, i)
        cls = classify_object(raw["features"])
        anomaly = compute_anomaly(
            class_name=cls["class_name"],
            class_confidence=cls["class_confidence"],
            contrast=raw["features"].contrast,
            fill_ratio=raw["features"].fill_ratio,
        )
        risk = compute_risk(
            detection_confidence=raw["detection_confidence"],
            artificial_prob=cls["artificial_prob"],
            anomaly_score=anomaly.anomaly_score,
            class_name=cls["class_name"],
            is_ood=anomaly.is_ood,
        )
        tracking_id, track_status = assign_tracking_id(raw["bbox"], cls["class_name"], previous)

        det = {
            "object_id": object_id,
            "scan_id": scan_id,
            "tracking_id": tracking_id,
            "track_status": track_status,
            "class_name": cls["class_name"],
            "detection_confidence": raw["detection_confidence"],
            "natural_prob": cls["natural_prob"],
            "artificial_prob": cls["artificial_prob"],
            "final_type": cls["final_type"],
            "anomaly_score": anomaly.anomaly_score,
            "is_ood": anomaly.is_ood,
            "anomaly_status": anomaly.status,
            "risk_level": risk.risk_level,
            "risk_score": risk.risk_score,
            "risk_reason": f"{risk.reason} Recommended action: {risk.recommended_action}",
            "bbox": raw["bbox"],
            "mask_rle": mask_to_rle(raw["mask"]),
            "dataset_source": dataset_source,
            "inference_mode": engine_status.mode,
        }
        db.insert_detection(det)
        detections_out.append(det)

        confidences.append(raw["detection_confidence"])
        if risk.risk_level in ("HIGH", "CRITICAL"):
            high_risk_count += 1
        if anomaly.is_ood:
            unknown_count += 1

    scan_record = db.ScanRecord(
        scan_id=scan_id,
        filename=filename,
        latitude=latitude,
        longitude=longitude,
        depth_m=depth_m,
        location_source=location_source,
        timestamp_label=timestamp_label,
        dataset_source=dataset_source,
        inference_mode=engine_status.mode,
        image_path=image_path,
        preview_path=preview_path,
        num_detections=len(detections_out),
        num_high_risk=high_risk_count,
        num_unknown=unknown_count,
        avg_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
    )
    db.insert_scan(scan_record)

    return {
        "scan_id": scan_id,
        "scan_record": scan_record,
        "stages": stages,
        "detections": detections_out,
        "engine_status": engine_status,
    }
