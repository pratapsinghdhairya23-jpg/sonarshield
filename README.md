# SONARSHIELD

**AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery**

Team: **SonarSentinels** &nbsp;|&nbsp; Problem Statement ID: **SIH26057**

A Streamlit web application implementing the full SONARSHIELD pipeline:

```
SSS Image / Sonar Log → Pre-processing → Detection + Segmentation → Object
Classification → Natural vs Artificial → Unknown/OOD → Confidence →
Risk Priority → Geo-tagging → Map → CSV/JSON Report
```

## 1. Project overview

SONARSHIELD analyzes uploaded side-scan sonar images, detects candidate
objects, classifies them, estimates whether each is natural or artificial,
flags unknown/out-of-distribution objects, scores risk, geo-tags results
where coordinates are available, and lets you export CSV/JSON reports —
all backed by a real SQLite database that persists across restarts.

## 2. Features

- **Command Center** dashboard with live stats from the database
- **Analyze Scan**: drag/drop upload, metadata entry, one-click analysis
- **Results**: original image, bounding-box overlay, per-object detail,
  natural/artificial + anomaly + risk badges, CSV/JSON export
- **Risk Intelligence**: backend-computed LOW/MEDIUM/HIGH/CRITICAL priority ranking
- **Geospatial Intelligence**: interactive map (Folium if installed, else
  automatic `st.map` fallback), never fabricates GPS
- **Scan History**: view / reopen / delete past scans (SQLite-backed)
- **Analytics**: category, risk, natural/artificial, and confidence distributions
- **Research & Datasets**: honest status of AI4Shipwrecks / GhostVision / SubPipe
- **System Status**: current vs. model-dependent vs. future functionality

## 3. Architecture

```
app.py                      <- Streamlit UI/routing only
backend/
    database.py              SQLite persistence
    pipeline.py               orchestrates the full pipeline
    classification.py          prototype natural/artificial + category heuristic
    anomaly_engine.py           prototype OOD/unknown flagging
    risk_engine.py                backend-computed risk scoring
    tracking.py                    prototype cross-scan object tracking
    dataset_manager.py              detects installed research datasets
    report_generator.py              CSV / JSON export
ai/
    inference.py               InferenceEngine factory (YOLO-Seg vs Prototype)
    prototype_engine.py         OpenCV/NumPy/SciPy heuristic detector+segmenter
    yolo_seg_engine.py           loads models/sonarshield_yoloseg.pt if present
preprocessing/
    sonar_preprocessor.py       OpenCV grayscale/denoise/CLAHE/normalize/resize
geospatial/
    map_engine.py                Folium map builder (+ fallback)
frontend/
    styles.py                    charcoal/slate/teal/emerald/amber/coral/purple CSS
    components.py                  reusable Streamlit render helpers
data/{ai4shipwrecks,ghostvision,subpipe}/   <- put dataset files here
models/sonarshield_yoloseg.pt (not included) <- put a trained checkpoint here
tests/test_pipeline.py          automated tests (see below)
```

## 4. Folder structure

See the tree above — it matches what's on disk in this delivered project.

## 5. Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`folium` + `streamlit-folium` are optional (rich interactive map); without
them the Geospatial Intelligence page automatically falls back to a plain
`st.map()`. `torch`/`ultralytics` are optional and only needed if you plug
in a real trained YOLO-Seg checkpoint.

## 6. Dependencies

See `requirements.txt`. Core: `streamlit`, `numpy`, `pandas`, `Pillow`,
`opencv-python-headless`, `scipy`. Optional: `folium`, `streamlit-folium`,
`torch`, `ultralytics`.

## 7. How to run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## 8. How to add a trained YOLO-Seg model

See `models/README.md`. In short: place a checkpoint at
`models/sonarshield_yoloseg.pt`, install `torch` (and your training
framework's inference library), and implement the forward pass in
`ai/yolo_seg_engine.py`'s `run()` method. The app will then report
**Inference Mode: YOLO-Seg** instead of **Inference Mode: Prototype**
automatically — nothing else needs to change.

## 9. Dataset setup

Place imagery (and annotations, if any) into:

```
data/ai4shipwrecks/
data/ghostvision/
data/subpipe/
```

The **Research & Datasets** page detects and counts files automatically —
it never fabricates dataset statistics. **No dataset files are bundled
with this project.**

## 10. Dataset restrictions

This application references and displays **only** the three datasets named
in the SIH proposal: AI4Shipwrecks (primary), Ghost Pot/GhostVision, and
SubPipe (both secondary/validation). No Kaggle, COCO, ImageNet, or other
dataset is used or mentioned anywhere in the codebase.

## 11. Prototype inference limitations

`ai/prototype_engine.py` is a classical computer-vision heuristic (local
contrast + connected-component blob analysis with SciPy), **not a trained
neural network**. `backend/classification.py` and `backend/anomaly_engine.py`
are rule-based heuristics tuned on synthetic test blobs, not on the
AI4Shipwrecks/GhostVision/SubPipe datasets (which are not installed in
this delivered project). Every page that shows prototype-derived results
also shows an "Inference Mode: Prototype" badge.

## 12. Testing

```bash
python -m unittest discover -s tests -v
```

16 tests cover: SQLite persistence (incl. simulated restart), preprocessing
stages, prototype blob detection, classification/anomaly/risk logic (including
edge cases), cross-scan tracking, dataset-manager honesty, CSV/JSON report
content, and a full end-to-end pipeline run. All 16 pass in the build
environment (Python 3.12, numpy/pandas/opencv/scipy available).
**`app.py` itself could not be runtime-tested in the build sandbox because
Streamlit/Folium/PyTorch could not be installed there (no network access) —
it was syntax-checked (`py_compile`) but not executed. Please run
`streamlit run app.py` locally as the final verification step.**

## 13. Future improvements

- Wire up a real trained YOLO-Seg checkpoint (`ai/yolo_seg_engine.py::run()`)
- Full pixel-mask overlay rendering on the Results page (currently masks are
  computed and exported but not painted onto the image)
- ONNX/TensorRT edge optimization, AUV/ROV integration, near-real-time
  deployment — see the System Status page; none of this is implemented,
  by design, until a trained model and deployment target exist.
