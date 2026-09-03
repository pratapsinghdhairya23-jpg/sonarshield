"""
app.py

SONARSHIELD - AI-Powered Underwater Marine Debris and Anomaly Detection
Team SonarSentinels | SIH26057

Run with:
    streamlit run app.py

This file is presentation/routing ONLY. All real logic lives in backend/, ai/,
preprocessing/, geospatial/ so it can be unit-tested independently of Streamlit.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from backend import database as db
from backend import report_generator as reports
from backend.dataset_manager import get_dataset_status
from backend.pipeline import run_full_pipeline
from ai.inference import get_engine_status
from frontend.styles import CUSTOM_CSS
from frontend import components as ui
from geospatial.map_engine import build_map, detections_to_dataframe, RISK_COLORS

ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = ROOT / "uploads"
OUTPUTS_DIR = ROOT / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 15
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

st.set_page_config(page_title="SONARSHIELD", page_icon="🛰️", layout="wide")
db.init_db()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "route" not in st.session_state:
    st.session_state.route = "Command Center"
if "active_scan_id" not in st.session_state:
    st.session_state.active_scan_id = None

NAV_PAGES = [
    "Command Center", "Analyze Scan", "Results", "Risk Intelligence",
    "Geospatial Intelligence", "Scan History", "Analytics",
    "Research & Datasets", "System Status",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def goto(route: str, scan_id: str | None = None):
    st.session_state.route = route
    if scan_id:
        st.session_state.active_scan_id = scan_id
    st.rerun()


def draw_overlay(base_rgb: np.ndarray, detections: list[dict]) -> Image.Image:
    img = Image.fromarray(base_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    color_map = {"LOW": (53, 183, 121), "MEDIUM": (242, 184, 75),
                 "HIGH": (231, 111, 81), "CRITICAL": (231, 111, 81)}
    for d in detections:
        x, y, w, h = d["bbox"]
        color = color_map.get(d["risk_level"], (139, 111, 216))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        label = f'{d["object_id"]} {d["class_name"]} {d["detection_confidence"]:.0%}'
        draw.rectangle([x, max(0, y - 16), x + 8 * len(label), y], fill=color)
        draw.text((x + 2, max(0, y - 15)), label, fill=(16, 20, 24))
    return img


def validate_upload(uploaded_file) -> str | None:
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return f"Unsupported file type '{ext}'. Allowed: PNG, JPG, JPEG, TIFF."
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return f"File too large ({size_mb:.1f} MB). Maximum is {MAX_UPLOAD_MB} MB."
    return None


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #

with st.sidebar:
    ui.render_header()
    for page in NAV_PAGES:
        if st.button(page, use_container_width=True,
                      type="primary" if st.session_state.route == page else "secondary"):
            goto(page)
    st.markdown("---")
    status = get_engine_status()
    st.markdown(ui.mode_badge_html(status.mode), unsafe_allow_html=True)
    st.caption(status.label)


# --------------------------------------------------------------------------- #
# Page: Command Center
# --------------------------------------------------------------------------- #

def page_command_center():
    st.title("Command Center")
    st.caption("SONARSHIELD system overview")
    ui.render_pipeline_diagram()
    st.write("")

    stats = db.get_dashboard_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1: ui.render_stat_card("Total Scans", str(stats["total_scans"]))
    with c2: ui.render_stat_card("Objects Detected", str(stats["total_detections"]))
    with c3: ui.render_stat_card("High-Risk Objects", str(stats["high_risk"]))
    with c4: ui.render_stat_card("Unknown Anomalies", str(stats["unknown_anomalies"]))

    c5, c6 = st.columns(2)
    with c5:
        ui.render_stat_card("Average Detection Confidence",
                             f"{stats['avg_confidence']:.0%}" if stats["total_detections"] else "N/A")
    with c6:
        latest = stats["latest_scan"]
        ui.render_stat_card("Latest Scan", latest["scan_id"] if latest else "No scans yet")

    st.write("")
    if st.button("➕ Analyze New Sonar Scan", type="primary"):
        goto("Analyze Scan")

    st.subheader("Recent Activity")
    scans = db.get_scans(limit=8)
    if not scans:
        st.info("No scans recorded yet. Upload a sonar image on the Analyze Scan page to begin.")
        return
    df = pd.DataFrame([dict(s) for s in scans])[
        ["scan_id", "filename", "num_detections", "num_high_risk", "num_unknown", "inference_mode"]
    ]
    st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Page: Analyze Scan
# --------------------------------------------------------------------------- #

def page_analyze_scan():
    st.title("Analyze Sonar Scan")
    status = get_engine_status()
    ui.render_model_status(status.label, status.detail)

    st.subheader("1. Upload Side-Scan Sonar Image")
    uploaded = st.file_uploader("Drop Side-Scan Sonar Image Here", type=["png", "jpg", "jpeg", "tif", "tiff"])

    st.subheader("2. Scan Metadata")
    col1, col2 = st.columns(2)
    with col1:
        scan_label = st.text_input("Sonar log timestamp / label (optional)", "")
        lat = st.text_input("Latitude (optional)", "")
        lon = st.text_input("Longitude (optional)", "")
    with col2:
        depth = st.text_input("Depth (m, optional)", "")
        dataset_status = get_dataset_status()
        installed = [d["name"] for d in dataset_status if d["installed"]]
        source_options = installed + ["Prototype / User Upload (not a research dataset)"]
        dataset_source = st.selectbox("Dataset / source label", source_options)

    run = st.button("Run Analysis", type="primary", disabled=uploaded is None)

    if not uploaded:
        st.info("Upload an image to enable analysis.")
        return

    err = validate_upload(uploaded)
    if err:
        st.error(err)
        return

    if run:
        try:
            image = Image.open(uploaded).convert("RGB")
            rgb = np.array(image)
        except Exception as e:
            st.error(f"Could not read this image file: {e}")
            return

        def parse_float(v):
            try:
                return float(v) if v.strip() != "" else None
            except ValueError:
                return None

        lat_v, lon_v, depth_v = parse_float(lat), parse_float(lon), parse_float(depth)

        save_path = UPLOADS_DIR / f"{int(time.time())}_{uploaded.name}"
        try:
            image.save(save_path)
        except Exception:
            save_path = Path("")

        with st.spinner("Running preprocessing, detection, classification, risk analysis..."):
            try:
                result = run_full_pipeline(
                    rgb_image=rgb, filename=uploaded.name, latitude=lat_v, longitude=lon_v,
                    depth_m=depth_v, timestamp_label=scan_label, dataset_source=dataset_source,
                    image_path=str(save_path), preview_path=str(save_path),
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        st.success(f"Scan {result['scan_id']} analyzed - {len(result['detections'])} object(s) found.")
        goto("Results", scan_id=result["scan_id"])


# --------------------------------------------------------------------------- #
# Page: Results
# --------------------------------------------------------------------------- #

def page_results():
    st.title("Scan Results")
    scan_id = st.session_state.active_scan_id
    if not scan_id:
        st.info("No scan selected. Analyze a new scan or open one from Scan History.")
        return

    scan = db.get_scan(scan_id)
    if not scan:
        st.error("Scan not found.")
        return
    scan = dict(scan)
    detections = [dict(d) for d in db.get_detections_for_scan(scan_id)]
    for d in detections:
        d["bbox"] = json.loads(d["bbox_json"])

    st.caption(f"{scan_id} &middot; {scan['filename']} &middot; {ui.mode_badge_html(scan['inference_mode'])}")
    st.markdown(ui.mode_badge_html(scan["inference_mode"]), unsafe_allow_html=True)

    if scan["location_source"] == "user_provided":
        st.markdown(
            f'<div class="ss-status-note">Location: {scan["latitude"]:.5f}, {scan["longitude"]:.5f} '
            f'&mdash; User Provided / Demo Location</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ss-status-note">Location unavailable &mdash; no metadata provided.</div>',
                     unsafe_allow_html=True)

    if not detections:
        st.warning("No objects detected in this scan above the confidence threshold.")
        return

    try:
        base_img = Image.open(scan["image_path"]).convert("RGB")
        base_arr = np.array(base_img)
    except Exception:
        base_arr = None

    tabs = st.tabs(["Original", "Detection Overlay", "Segmentation"])
    if base_arr is not None:
        with tabs[0]:
            st.image(base_arr, use_container_width=True, caption="Original sonar image")
        with tabs[1]:
            st.image(draw_overlay(base_arr, detections), use_container_width=True,
                      caption="Bounding boxes + object ID / class / confidence")
        with tabs[2]:
            st.caption("Pixel-level segmentation masks are computed per-object during analysis "
                       "(stored as run-length-encoded data) and available via the JSON report. "
                       "A full mask-overlay renderer is a natural next enhancement once a trained "
                       "YOLO-Seg model produces higher-resolution masks.")
    else:
        st.warning("Original image file could not be reloaded for display (it may not have been saved to disk).")

    st.subheader("Detections")
    rows = []
    for d in detections:
        rows.append({
            "Object ID": d["object_id"],
            "Class": d["class_name"],
            "Confidence": f'{d["detection_confidence"]:.0%}',
            "Type": d["final_type"],
            "Anomaly": d["anomaly_status"],
            "Risk": d["risk_level"],
            "Tracking": f'{d["tracking_id"]} ({d["track_status"]})',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Object Detail")
    obj_ids = [d["object_id"] for d in detections]
    selected = st.selectbox("Select an object", obj_ids)
    d = next(x for x in detections if x["object_id"] == selected)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Class:** {d['class_name']}")
        st.markdown(f"**Detection confidence:** {d['detection_confidence']:.0%}")
        st.markdown(ui.type_badge_html(d["final_type"]), unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Natural probability:** {d['natural_prob']:.0%}")
        st.markdown(f"**Artificial probability:** {d['artificial_prob']:.0%}")
        st.markdown(ui.anomaly_badge_html(d["anomaly_status"]), unsafe_allow_html=True)
    with c3:
        st.markdown(ui.risk_badge_html(d["risk_level"]), unsafe_allow_html=True)
        st.markdown(f"**Risk score:** {d['risk_score']:.2f}")
        st.caption(d["risk_reason"])

    st.subheader("Export")
    csv_bytes = reports.build_csv_bytes(detections, scan)
    json_bytes = reports.build_json_bytes(detections, scan)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ Download CSV", csv_bytes, file_name=f"{scan_id}_report.csv", mime="text/csv")
    with c2:
        st.download_button("⬇ Download JSON", json_bytes, file_name=f"{scan_id}_report.json", mime="application/json")


# --------------------------------------------------------------------------- #
# Page: Risk Intelligence
# --------------------------------------------------------------------------- #

def page_risk_intelligence():
    st.title("Risk Intelligence")
    st.caption("Prototype decision-support priorities - not an official regulatory hazard classification.")
    all_dets = [dict(d) for d in db.get_all_detections()]
    if not all_dets:
        st.info("No detections yet.")
        return

    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for d in all_dets:
        counts[d["risk_level"]] = counts.get(d["risk_level"], 0) + 1
    unknown = sum(1 for d in all_dets if d["is_ood"])

    cols = st.columns(5)
    labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    for c, lbl in zip(cols[:4], labels):
        with c:
            ui.render_stat_card(lbl, str(counts.get(lbl, 0)))
    with cols[4]:
        ui.render_stat_card("Unknown / OOD", str(unknown))

    st.subheader("Priority Ranking")
    ranked = sorted(all_dets, key=lambda d: d["risk_score"], reverse=True)[:25]
    rows = [{
        "Object ID": d["object_id"], "Scan": d["scan_id"], "Class": d["class_name"],
        "Risk": d["risk_level"], "Score": round(d["risk_score"], 2),
        "Suggested Action": d["risk_reason"].split("Recommended action:")[-1].strip(),
    } for d in ranked]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Page: Geospatial Intelligence
# --------------------------------------------------------------------------- #

def page_geospatial():
    st.title("Geospatial Intelligence")
    scans = {s["scan_id"]: dict(s) for s in db.get_scans()}
    all_dets = [dict(d) for d in db.get_all_detections()]
    rows = []
    for d in all_dets:
        s = scans.get(d["scan_id"], {})
        if s.get("latitude") is None:
            continue
        rows.append({**d, "latitude": s["latitude"], "longitude": s["longitude"],
                      "location_source": s["location_source"]})

    if not rows:
        st.info("Location unavailable — no scans with latitude/longitude metadata yet. "
                 "Enter coordinates on the Analyze Scan page to plot detections here.")
        return

    all_risk = sorted({r["risk_level"] for r in rows})
    all_classes = sorted({r["class_name"] for r in rows})
    c1, c2 = st.columns(2)
    with c1:
        risk_filter = st.multiselect("Filter by risk", all_risk, default=all_risk)
    with c2:
        class_filter = st.multiselect("Filter by class", all_classes, default=all_classes)

    fmap = build_map(rows, risk_filter=risk_filter, class_filter=class_filter)
    if fmap is not None:
        try:
            from streamlit_folium import st_folium
            st_folium(fmap, width=None, height=520)
        except ImportError:
            st.warning("`streamlit-folium` is not installed, so the rich Folium map can't render inline. "
                       "Install it with `pip install streamlit-folium` for popups/markers. "
                       "Showing a basic point map instead.")
            df = detections_to_dataframe(rows)
            if not df.empty:
                st.map(df[["lat", "lon"]])
    else:
        st.caption("Folium is not installed - showing a basic point map (no popups/legend). "
                   "Install `folium` for the full interactive map.")
        df = detections_to_dataframe(rows)
        if not df.empty:
            st.map(df[["lat", "lon"]])

    st.dataframe(
        pd.DataFrame(rows)[["object_id", "scan_id", "class_name", "risk_level",
                             "latitude", "longitude", "location_source"]],
        use_container_width=True, hide_index=True,
    )


# --------------------------------------------------------------------------- #
# Page: Scan History
# --------------------------------------------------------------------------- #

def page_scan_history():
    st.title("Scan History")
    scans = db.get_scans()
    if not scans:
        st.info("No scans yet.")
        return

    for s in scans:
        s = dict(s)
        with st.container():
            st.markdown('<div class="ss-card">', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            with c1:
                st.markdown(f"**{s['scan_id']}**  \n{s['filename']}")
            with c2:
                st.markdown(f"Detections: {s['num_detections']}  \nHigh-risk: {s['num_high_risk']}")
            with c3:
                st.markdown(f"Unknown: {s['num_unknown']}  \nAvg conf: {s['avg_confidence']:.0%}")
            with c4:
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("View", key=f"view_{s['scan_id']}"):
                        goto("Results", scan_id=s["scan_id"])
                with b2:
                    if st.button("Delete", key=f"del_{s['scan_id']}"):
                        db.delete_scan(s["scan_id"])
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page: Analytics
# --------------------------------------------------------------------------- #

def page_analytics():
    st.title("Analytics")
    all_dets = [dict(d) for d in db.get_all_detections()]
    if not all_dets:
        st.info("No detections yet.")
        return
    df = pd.DataFrame(all_dets)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Objects by Category")
        st.bar_chart(df["class_name"].value_counts())
    with c2:
        st.subheader("Risk Distribution")
        st.bar_chart(df["risk_level"].value_counts())

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Natural vs Artificial")
        st.bar_chart(df["final_type"].value_counts())
    with c4:
        st.subheader("Confidence Distribution")
        st.bar_chart(pd.cut(df["detection_confidence"], bins=5).astype(str).value_counts().sort_index())

    st.subheader("Scan Activity Over Time")
    scans = pd.DataFrame([dict(s) for s in db.get_scans()])
    if not scans.empty:
        scans["date"] = pd.to_datetime(scans["created_at"], unit="s").dt.date
        st.bar_chart(scans.groupby("date").size())


# --------------------------------------------------------------------------- #
# Page: Research & Datasets
# --------------------------------------------------------------------------- #

def page_datasets():
    st.title("Research & Dataset Sources")
    st.caption("Only datasets referenced in the SIH presentation are used by this application.")
    statuses = get_dataset_status()
    rows = [{
        "Dataset": d["name"], "Purpose": d["purpose"], "Type": d["type"],
        "Role": d["role"], "Local Path": d["local_path"], "Status": d["status_label"],
    } for d in statuses]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    any_installed = any(d["installed"] for d in statuses)
    if not any_installed:
        st.warning(
            "Dataset integration architecture is implemented, but the research dataset files "
            "need to be supplied locally. Place imagery under `data/ai4shipwrecks/`, "
            "`data/ghostvision/`, or `data/subpipe/` and reload this page."
        )
    st.markdown(
        "**Strategy (per SIH proposal):** AI4Shipwrecks as the primary baseline, expanding "
        "validation to Ghost Pot/GhostVision and SubPipe. No Kaggle, COCO, ImageNet, or other "
        "unrelated dataset is used anywhere in this application."
    )


# --------------------------------------------------------------------------- #
# Page: System Status
# --------------------------------------------------------------------------- #

def page_system_status():
    st.title("System Status")
    status = get_engine_status()
    ui.render_model_status(status.label, status.detail)

    st.subheader("Current Prototype")
    for item in [
        "SSS image upload & validation", "OpenCV preprocessing (grayscale, denoise, CLAHE, normalize)",
        "Prototype detection + per-object segmentation mask", "Heuristic classification",
        "Natural vs artificial estimation", "Prototype OOD / unknown flagging",
        "Backend-computed risk prioritization", "Geo-tagging (user-provided coordinates only)",
        "Interactive map (Folium, with fallback)", "SQLite-backed scan history",
        "Prototype cross-scan tracking", "CSV / JSON reporting",
    ]:
        st.markdown(f"- {item}")

    st.subheader("Advanced Prototype / Model Integration")
    for item in [
        "Multi-class trained sonar model (YOLO-Seg)", "Higher-resolution segmentation masks",
        "Dataset-based validation against AI4Shipwrecks / GhostVision / SubPipe",
        "Learned cross-scan re-identification",
    ]:
        st.markdown(f"- {item} — *requires a trained checkpoint at `models/sonarshield_yoloseg.pt`*")

    st.subheader("Future Deployment")
    for item in ["ONNX / TensorRT edge optimization", "AUV / ROV integration",
                  "Near-real-time underwater sonar analysis"]:
        st.markdown(f"- {item} — *not implemented, roadmap only*")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

PAGES = {
    "Command Center": page_command_center,
    "Analyze Scan": page_analyze_scan,
    "Results": page_results,
    "Risk Intelligence": page_risk_intelligence,
    "Geospatial Intelligence": page_geospatial,
    "Scan History": page_scan_history,
    "Analytics": page_analytics,
    "Research & Datasets": page_datasets,
    "System Status": page_system_status,
}

PAGES[st.session_state.route]()
