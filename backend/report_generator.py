"""
backend/report_generator.py

Builds CSV / JSON reports directly from the SQLite-backed scan + detection
records (never from hardcoded/frontend values).
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

CSV_COLUMNS = [
    "scan_id", "object_id", "dataset_source", "class_name", "detection_confidence",
    "artificial_prob", "natural_prob", "anomaly_score", "risk_score", "risk_level",
    "latitude", "longitude", "depth_m", "timestamp", "inference_mode", "recommended_action",
]


def _row_from_detection(det: dict, scan: dict) -> dict[str, Any]:
    return {
        "scan_id": det["scan_id"],
        "object_id": det["object_id"],
        "dataset_source": det.get("dataset_source") or scan.get("dataset_source"),
        "class_name": det["class_name"],
        "detection_confidence": det["detection_confidence"],
        "artificial_prob": det["artificial_prob"],
        "natural_prob": det["natural_prob"],
        "anomaly_score": det["anomaly_score"],
        "risk_score": det["risk_score"],
        "risk_level": det["risk_level"],
        "latitude": scan.get("latitude"),
        "longitude": scan.get("longitude"),
        "depth_m": scan.get("depth_m"),
        "timestamp": scan.get("timestamp_label") or scan.get("created_at"),
        "inference_mode": det.get("inference_mode") or scan.get("inference_mode"),
        "recommended_action": det.get("risk_reason", ""),
    }


def build_csv_bytes(detections: list[dict], scan: dict) -> bytes:
    rows = [_row_from_detection(d, scan) for d in detections]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def build_json_bytes(detections: list[dict], scan: dict) -> bytes:
    payload = {
        "scan_metadata": {
            "scan_id": scan.get("scan_id"),
            "filename": scan.get("filename"),
            "created_at": scan.get("created_at"),
            "latitude": scan.get("latitude"),
            "longitude": scan.get("longitude"),
            "depth_m": scan.get("depth_m"),
            "location_source": scan.get("location_source"),
            "timestamp_label": scan.get("timestamp_label"),
            "dataset_source": scan.get("dataset_source"),
            "inference_mode": scan.get("inference_mode"),
        },
        "detections": [
            {
                "object_id": d["object_id"],
                "tracking_id": d.get("tracking_id"),
                "track_status": d.get("track_status"),
                "classification": {
                    "class_name": d["class_name"],
                    "natural_prob": d["natural_prob"],
                    "artificial_prob": d["artificial_prob"],
                    "final_type": d["final_type"],
                },
                "anomaly": {
                    "anomaly_score": d["anomaly_score"],
                    "is_ood": bool(d["is_ood"]),
                    "status": d["anomaly_status"],
                },
                "confidence": {
                    "detection_confidence": d["detection_confidence"],
                },
                "risk": {
                    "risk_level": d["risk_level"],
                    "risk_score": d["risk_score"],
                    "reason": d["risk_reason"],
                },
                "geometry": {
                    "bbox_xywh": d.get("bbox"),
                },
                "dataset_source": d.get("dataset_source"),
                "inference_mode": d.get("inference_mode"),
            }
            for d in detections
        ],
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")
