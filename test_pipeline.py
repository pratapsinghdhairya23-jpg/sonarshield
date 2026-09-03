"""
tests/test_pipeline.py

Covers everything testable WITHOUT Streamlit, PyTorch, or Folium installed:
  - database persistence (SQLite, real file on disk, survives reopening)
  - preprocessing pipeline (OpenCV/NumPy)
  - prototype detection engine (finds synthetic blobs)
  - classification / anomaly / risk engines
  - tracking (new vs existing object across two "scans")
  - dataset manager (detects the empty placeholder folders honestly)
  - report generation (CSV / JSON)
  - full end-to-end pipeline via backend.pipeline.run_full_pipeline

Run with:
    python -m pytest tests/ -v
or plain:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_synthetic_sonar_image(size=256) -> np.ndarray:
    """Grey noisy background with two bright rectangular 'objects' burned in,
    so the prototype detector has something deterministic to find."""
    rng = np.random.default_rng(42)
    base = rng.normal(loc=90, scale=6, size=(size, size)).clip(0, 255).astype(np.uint8)
    img = np.stack([base, base, base], axis=-1)
    # bright rectangular artificial-looking object
    img[40:80, 40:110] = 210
    # a second, smaller, elongated object (pipeline-like)
    img[150:158, 30:190] = 205
    return img


class TestDatabase(unittest.TestCase):
    def setUp(self):
        import backend.database as db
        self.db = db
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        db.DB_PATH = Path(self.tmp.name)
        db.init_db()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_insert_and_fetch_scan_persists(self):
        rec = self.db.ScanRecord(scan_id="SCAN-TEST01", filename="x.png",
                                  num_detections=2, num_high_risk=1, num_unknown=0, avg_confidence=0.7)
        self.db.insert_scan(rec)
        fetched = self.db.get_scan("SCAN-TEST01")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["filename"], "x.png")

    def test_data_survives_reconnect(self):
        rec = self.db.ScanRecord(scan_id="SCAN-TEST02", filename="y.png")
        self.db.insert_scan(rec)
        # simulate "app restart" by just opening a brand new connection
        with self.db.get_conn() as conn:
            row = conn.execute("SELECT * FROM scans WHERE scan_id=?", ("SCAN-TEST02",)).fetchone()
        self.assertIsNotNone(row)

    def test_dashboard_stats_not_hardcoded(self):
        stats_before = self.db.get_dashboard_stats()
        self.db.insert_scan(self.db.ScanRecord(scan_id="SCAN-TEST03", filename="z.png"))
        stats_after = self.db.get_dashboard_stats()
        self.assertEqual(stats_after["total_scans"], stats_before["total_scans"] + 1)


class TestPreprocessing(unittest.TestCase):
    def test_pipeline_stages_present(self):
        from preprocessing.sonar_preprocessor import preprocess_pipeline
        img = make_synthetic_sonar_image()
        stages = preprocess_pipeline(img)
        for key in ("original", "grayscale", "denoised", "enhanced"):
            self.assertIn(key, stages)
            self.assertGreater(stages[key].size, 0)
        self.assertIn(stages["backend"], ("opencv", "numpy-fallback"))


class TestPrototypeEngine(unittest.TestCase):
    def test_detects_synthetic_blobs(self):
        from preprocessing.sonar_preprocessor import preprocess_pipeline
        from ai.prototype_engine import run_prototype_detection
        img = make_synthetic_sonar_image()
        stages = preprocess_pipeline(img)
        dets = run_prototype_detection(stages["enhanced"])
        self.assertGreaterEqual(len(dets), 1, "Expected at least one synthetic object to be detected")
        for d in dets:
            self.assertIn("bbox", d)
            self.assertIn("mask", d)
            self.assertTrue(0.0 <= d["detection_confidence"] <= 1.0)


class TestClassificationAnomalyRisk(unittest.TestCase):
    def test_elongated_high_contrast_classified_artificial_like(self):
        from backend.classification import classify_object, BlobFeatures
        f = BlobFeatures(area_px=1200, bbox_w=160, bbox_h=8, fill_ratio=0.9,
                          contrast=0.6, elongation=20.0, edge_straightness=0.8)
        result = classify_object(f)
        self.assertGreaterEqual(result["artificial_prob"], result["natural_prob"])
        self.assertIn(result["final_type"], ("Artificial", "Uncertain"))

    def test_ambiguous_blob_flagged_unknown(self):
        """A blob whose artificial/natural evidence is nearly balanced should be
        classified as Uncertain/Unknown Anomaly, which the anomaly engine must
        flag as OOD - it should NOT be silently forced into a known category."""
        from backend.classification import classify_object, BlobFeatures
        from backend.anomaly_engine import compute_anomaly
        f = BlobFeatures(area_px=500, bbox_w=25, bbox_h=20, fill_ratio=0.6,
                          contrast=0.97, elongation=1.5, edge_straightness=0.4)
        cls = classify_object(f)
        self.assertEqual(cls["final_type"], "Uncertain")
        self.assertEqual(cls["class_name"], "Unknown Anomaly")
        anomaly = compute_anomaly(cls["class_name"], cls["class_confidence"], f.contrast, f.fill_ratio)
        self.assertTrue(anomaly.is_ood)
        self.assertEqual(anomaly.status, "UNKNOWN / OOD ANOMALY")

    def test_strongly_natural_blob_not_forced_into_wrong_category(self):
        """A very low-contrast, irregular, low-fill blob should be confidently
        'Natural Formation' rather than flagged as an anomaly - this is
        correct heuristic behavior, not a bug."""
        from backend.classification import classify_object, BlobFeatures
        f = BlobFeatures(area_px=200, bbox_w=15, bbox_h=13, fill_ratio=0.05,
                          contrast=0.03, elongation=1.2, edge_straightness=0.0)
        cls = classify_object(f)
        self.assertEqual(cls["class_name"], "Natural Formation")
        self.assertGreater(cls["natural_prob"], cls["artificial_prob"])

    def test_risk_engine_never_trivializes_ood(self):
        from backend.risk_engine import compute_risk
        risk = compute_risk(detection_confidence=0.3, artificial_prob=0.2, anomaly_score=0.9,
                             class_name="Unknown Anomaly", is_ood=True)
        self.assertIn(risk.risk_level, ("HIGH", "CRITICAL"))

    def test_risk_score_is_computed_not_constant(self):
        from backend.risk_engine import compute_risk
        low = compute_risk(0.3, 0.1, 0.1, "Natural Formation", False)
        high = compute_risk(0.95, 0.9, 0.8, "Fishing Gear", False)
        self.assertLess(low.risk_score, high.risk_score)


class TestTracking(unittest.TestCase):
    def test_new_object_gets_new_id(self):
        from backend.tracking import assign_tracking_id
        tid, status = assign_tracking_id([10, 10, 20, 20], "Fishing Gear", [])
        self.assertEqual(status, "new")
        self.assertTrue(tid.startswith("TRK-"))

    def test_nearby_same_class_matches_existing(self):
        from backend.tracking import assign_tracking_id
        prev = [{"tracking_id": "TRK-ABC123", "bbox": [10, 10, 20, 20], "class_name": "Fishing Gear"}]
        tid, status = assign_tracking_id([12, 11, 20, 20], "Fishing Gear", prev)
        self.assertEqual(tid, "TRK-ABC123")
        self.assertEqual(status, "existing")

    def test_far_away_same_class_is_new(self):
        from backend.tracking import assign_tracking_id
        prev = [{"tracking_id": "TRK-ABC123", "bbox": [10, 10, 20, 20], "class_name": "Fishing Gear"}]
        tid, status = assign_tracking_id([900, 900, 20, 20], "Fishing Gear", prev)
        self.assertNotEqual(tid, "TRK-ABC123")
        self.assertEqual(status, "new")


class TestDatasetManager(unittest.TestCase):
    def test_reports_not_installed_honestly_for_empty_folders(self):
        from backend.dataset_manager import get_dataset_status
        statuses = get_dataset_status()
        self.assertEqual(len(statuses), 3)
        names = {s["name"] for s in statuses}
        self.assertEqual(names, {"AI4Shipwrecks", "Ghost Pot / GhostVision", "SubPipe"})
        for s in statuses:
            # placeholder dirs only contain a .gitkeep (not an image) -> installed must be False
            self.assertFalse(s["installed"])
            self.assertEqual(s["status_label"], "Research dataset not installed.")


class TestReportGenerator(unittest.TestCase):
    def test_csv_and_json_contain_required_fields(self):
        from backend.report_generator import build_csv_bytes, build_json_bytes
        scan = {"scan_id": "SCAN-1", "filename": "f.png", "created_at": 0, "latitude": 12.9,
                "longitude": 77.6, "depth_m": 30, "location_source": "user_provided",
                "timestamp_label": "t", "dataset_source": "Prototype / User Upload",
                "inference_mode": "prototype"}
        det = {"object_id": "SCAN-1-OBJ-001", "scan_id": "SCAN-1", "tracking_id": "TRK-1",
               "track_status": "new", "class_name": "Fishing Gear", "detection_confidence": 0.8,
               "natural_prob": 0.2, "artificial_prob": 0.8, "final_type": "Artificial",
               "anomaly_score": 0.3, "is_ood": False, "anomaly_status": "Known",
               "risk_level": "HIGH", "risk_score": 0.7, "risk_reason": "reason text",
               "bbox": [1, 2, 3, 4], "dataset_source": "Prototype / User Upload",
               "inference_mode": "prototype"}
        csv_bytes = build_csv_bytes([det], scan)
        json_bytes = build_json_bytes([det], scan)
        csv_text = csv_bytes.decode()
        self.assertIn("scan_id", csv_text)
        self.assertIn("risk_level", csv_text)
        self.assertIn("latitude", csv_text)
        payload = json.loads(json_bytes)
        self.assertIn("scan_metadata", payload)
        self.assertIn("detections", payload)
        self.assertEqual(payload["detections"][0]["risk"]["risk_level"], "HIGH")


class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        import backend.database as db
        self.db = db
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        db.DB_PATH = Path(self.tmp.name)
        db.init_db()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_end_to_end_prototype_pipeline(self):
        from backend.pipeline import run_full_pipeline
        img = make_synthetic_sonar_image()
        result = run_full_pipeline(
            rgb_image=img, filename="synthetic.png", latitude=19.07, longitude=72.87,
            depth_m=25.0, timestamp_label="unit-test", dataset_source="Prototype / User Upload",
            image_path="", preview_path="",
        )
        self.assertTrue(result["scan_id"].startswith("SCAN-"))
        self.assertEqual(result["engine_status"].mode, "prototype")
        scan = self.db.get_scan(result["scan_id"])
        self.assertIsNotNone(scan)
        self.assertEqual(scan["num_detections"], len(result["detections"]))
        for d in result["detections"]:
            self.assertIn(d["risk_level"], ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
            self.assertIn(d["final_type"], ("Natural", "Artificial", "Uncertain"))

    def test_same_scan_detections_never_share_a_tracking_id(self):
        """
        Regression test: two distinct objects detected within the SAME scan
        must always get distinct tracking_ids, even when they are the same
        class and close enough in pixel space that cross-scan matching would
        normally treat them as "possibly_moved". Tracking is meant to link
        an object across DIFFERENT scans, not merge two different objects
        within one scan just because they're moderately close together.
        """
        from backend.pipeline import run_full_pipeline
        rng = __import__("numpy").random.default_rng(11)
        img = rng.normal(90, 6, (256, 256, 3)).clip(0, 255).astype("uint8")
        img[40:80, 40:110] = 210     # object A
        img[150:190, 150:220] = 208  # object B - same likely class, ~155px from A

        result = run_full_pipeline(
            rgb_image=img, filename="multi.png", latitude=13.08, longitude=80.27,
            depth_m=30.0, timestamp_label="", dataset_source="Prototype / User Upload",
            image_path="", preview_path="",
        )
        tracking_ids = [d["tracking_id"] for d in result["detections"]]
        self.assertEqual(
            len(tracking_ids), len(set(tracking_ids)),
            f"Detections within a single scan must not share a tracking_id: {tracking_ids}",
        )

    def test_second_scan_still_matches_a_real_prior_object(self):
        """
        Companion to the regression test above: confirms the fix didn't
        accidentally disable genuine cross-scan matching. A second scan with
        an object at nearly the same position/class as scan 1's object
        should still be recognized as 'existing' or 'possibly_moved'.
        """
        from backend.pipeline import run_full_pipeline
        rng = __import__("numpy").random.default_rng(11)

        img1 = rng.normal(90, 6, (256, 256, 3)).clip(0, 255).astype("uint8")
        img1[40:80, 40:110] = 210
        r1 = run_full_pipeline(
            rgb_image=img1, filename="s1.png", latitude=13.08, longitude=80.27,
            depth_m=30.0, timestamp_label="", dataset_source="Prototype / User Upload",
            image_path="", preview_path="",
        )
        prior_ids = {d["tracking_id"] for d in r1["detections"]}

        img2 = rng.normal(90, 6, (256, 256, 3)).clip(0, 255).astype("uint8")
        img2[42:82, 42:112] = 210  # nearly same position as scan 1's object
        r2 = run_full_pipeline(
            rgb_image=img2, filename="s2.png", latitude=13.08, longitude=80.27,
            depth_m=30.0, timestamp_label="", dataset_source="Prototype / User Upload",
            image_path="", preview_path="",
        )
        matched = any(
            d["track_status"] in ("existing", "possibly_moved") and d["tracking_id"] in prior_ids
            for d in r2["detections"]
        )
        self.assertTrue(matched, "Expected scan 2 to re-identify scan 1's object via cross-scan tracking")


if __name__ == "__main__":
    unittest.main(verbosity=2)
