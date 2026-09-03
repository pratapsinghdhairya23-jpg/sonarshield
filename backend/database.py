"""
backend/database.py

Persistent SQLite storage for SONARSHIELD.

This module is intentionally dependency-light (stdlib `sqlite3` only) so it can be
unit-tested without Streamlit, OpenCV, or a GPU environment installed.

Two tables:
    scans       - one row per uploaded/analyzed sonar scan
    detections  - one row per detected object within a scan (FK -> scans.scan_id)

Data survives Streamlit reruns and app restarts because it lives in a real file
on disk (default: sonarshield.db in the project root), NOT in st.session_state
or browser storage.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "sonarshield.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id           TEXT PRIMARY KEY,
    filename          TEXT,
    created_at        REAL,
    latitude          REAL,
    longitude         REAL,
    depth_m           REAL,
    location_source   TEXT,        -- 'user_provided' | 'unavailable'
    timestamp_label   TEXT,        -- free-text sonar-log timestamp, if provided
    dataset_source    TEXT,        -- which permitted research dataset this came from, if any
    inference_mode    TEXT,        -- 'yolo_seg' | 'prototype'
    image_path        TEXT,
    preview_path       TEXT,
    num_detections    INTEGER,
    num_high_risk     INTEGER,
    num_unknown       INTEGER,
    avg_confidence    REAL
);

CREATE TABLE IF NOT EXISTS detections (
    object_id           TEXT PRIMARY KEY,
    scan_id              TEXT,
    tracking_id          TEXT,
    track_status          TEXT,     -- 'new' | 'existing' | 'possibly_moved'
    class_name            TEXT,
    detection_confidence  REAL,
    natural_prob           REAL,
    artificial_prob        REAL,
    final_type              TEXT,   -- 'Natural' | 'Artificial' | 'Uncertain'
    anomaly_score           REAL,
    is_ood                   INTEGER,
    anomaly_status            TEXT, -- 'Known' | 'UNKNOWN / OOD ANOMALY'
    risk_level                 TEXT,
    risk_score                  REAL,
    risk_reason                  TEXT,
    bbox_json                     TEXT,  -- [x, y, w, h] in pixels
    mask_rle_json                  TEXT, -- run-length-encoded segmentation mask
    dataset_source                   TEXT,
    inference_mode                    TEXT,
    FOREIGN KEY (scan_id) REFERENCES scans (scan_id)
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(_SCHEMA)


@dataclass
class ScanRecord:
    scan_id: str
    filename: str
    created_at: float = field(default_factory=time.time)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth_m: Optional[float] = None
    location_source: str = "unavailable"
    timestamp_label: str = ""
    dataset_source: str = "Prototype / User Upload"
    inference_mode: str = "prototype"
    image_path: str = ""
    preview_path: str = ""
    num_detections: int = 0
    num_high_risk: int = 0
    num_unknown: int = 0
    avg_confidence: float = 0.0


def new_scan_id() -> str:
    return "SCAN-" + uuid.uuid4().hex[:8].upper()


def new_object_id(scan_id: str, index: int) -> str:
    return f"{scan_id}-OBJ-{index + 1:03d}"


def insert_scan(record: ScanRecord) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO scans
               (scan_id, filename, created_at, latitude, longitude, depth_m,
                location_source, timestamp_label, dataset_source, inference_mode,
                image_path, preview_path, num_detections, num_high_risk, num_unknown, avg_confidence)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.scan_id, record.filename, record.created_at, record.latitude,
                record.longitude, record.depth_m, record.location_source, record.timestamp_label,
                record.dataset_source, record.inference_mode, record.image_path, record.preview_path,
                record.num_detections, record.num_high_risk, record.num_unknown, record.avg_confidence,
            ),
        )


def insert_detection(det: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO detections
               (object_id, scan_id, tracking_id, track_status, class_name, detection_confidence,
                natural_prob, artificial_prob, final_type, anomaly_score, is_ood, anomaly_status,
                risk_level, risk_score, risk_reason, bbox_json, mask_rle_json, dataset_source, inference_mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                det["object_id"], det["scan_id"], det["tracking_id"], det["track_status"],
                det["class_name"], det["detection_confidence"], det["natural_prob"], det["artificial_prob"],
                det["final_type"], det["anomaly_score"], int(det["is_ood"]), det["anomaly_status"],
                det["risk_level"], det["risk_score"], det["risk_reason"],
                json.dumps(det["bbox"]), json.dumps(det.get("mask_rle", [])),
                det.get("dataset_source", ""), det.get("inference_mode", "prototype"),
            ),
        )


def delete_scan(scan_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM detections WHERE scan_id = ?", (scan_id,))
        conn.execute("DELETE FROM scans WHERE scan_id = ?", (scan_id,))


def get_scans(limit: Optional[int] = None) -> list[sqlite3.Row]:
    with get_conn() as conn:
        q = "SELECT * FROM scans ORDER BY created_at DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return conn.execute(q).fetchall()


def get_scan(scan_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()


def get_detections_for_scan(scan_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM detections WHERE scan_id = ? ORDER BY object_id", (scan_id,)
        ).fetchall()


def get_all_detections(limit: Optional[int] = None) -> list[sqlite3.Row]:
    with get_conn() as conn:
        q = "SELECT * FROM detections ORDER BY scan_id DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return conn.execute(q).fetchall()


def get_dashboard_stats() -> dict[str, Any]:
    with get_conn() as conn:
        scans = conn.execute("SELECT COUNT(*) c FROM scans").fetchone()["c"]
        dets = conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"]
        high = conn.execute(
            "SELECT COUNT(*) c FROM detections WHERE risk_level IN ('HIGH','CRITICAL')"
        ).fetchone()["c"]
        unknown = conn.execute("SELECT COUNT(*) c FROM detections WHERE is_ood = 1").fetchone()["c"]
        avg_conf_row = conn.execute("SELECT AVG(detection_confidence) a FROM detections").fetchone()
        avg_conf = avg_conf_row["a"] or 0.0
        latest = conn.execute("SELECT * FROM scans ORDER BY created_at DESC LIMIT 1").fetchone()
        return {
            "total_scans": scans,
            "total_detections": dets,
            "high_risk": high,
            "unknown_anomalies": unknown,
            "avg_confidence": avg_conf,
            "latest_scan": dict(latest) if latest else None,
        }
