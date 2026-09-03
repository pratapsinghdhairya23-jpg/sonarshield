"""
frontend/components.py

Small reusable Streamlit rendering helpers so app.py's page functions stay
readable. Pure presentation - no business logic lives here.
"""

from __future__ import annotations

import streamlit as st

PIPELINE_STEPS = [
    "SSS Image / Sonar Log", "Pre-processing", "Detection + Segmentation",
    "Classification", "Natural vs Artificial", "Unknown / OOD",
    "Confidence", "Risk Priority", "Geo-tagging", "Map", "CSV / JSON Report",
]


def render_header():
    st.markdown(
        """
        <div class="ss-header">
            <div class="ss-logo">SS</div>
            <div>
                <div class="ss-title">SONARSHIELD</div>
                <div class="ss-subtitle">AI-Powered Underwater Anomaly Intelligence &middot; Team SonarSentinels</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_diagram():
    steps_html = ""
    for i, step in enumerate(PIPELINE_STEPS):
        steps_html += f'<span class="step">{step}</span>'
        if i < len(PIPELINE_STEPS) - 1:
            steps_html += '<span class="arrow">&rarr;</span>'
    st.markdown(f'<div class="ss-pipeline">{steps_html}</div>', unsafe_allow_html=True)


def render_stat_card(label: str, value: str):
    st.markdown(
        f"""<div class="ss-card">
                <div class="ss-stat-label">{label}</div>
                <div class="ss-stat-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def risk_badge_html(level: str) -> str:
    cls = f"ss-badge-{level.lower()}" if level.lower() in (
        "low", "medium", "high", "critical", "unknown"
    ) else "ss-badge-unknown"
    return f'<span class="ss-badge {cls}">{level}</span>'


def type_badge_html(final_type: str) -> str:
    cls = f"ss-badge-{final_type.lower()}" if final_type.lower() in (
        "natural", "artificial", "uncertain"
    ) else "ss-badge-uncertain"
    return f'<span class="ss-badge {cls}">{final_type.upper()}</span>'


def anomaly_badge_html(status: str) -> str:
    if "UNKNOWN" in status.upper() or "OOD" in status.upper():
        return '<span class="ss-badge ss-badge-unknown">UNKNOWN / OOD</span>'
    return '<span class="ss-badge ss-badge-low">KNOWN</span>'


def mode_badge_html(mode: str) -> str:
    label = "YOLO-Seg" if mode == "yolo_seg" else "Prototype"
    return f'<span class="ss-badge ss-badge-mode">{label}</span>'


def render_model_status(status_label: str, detail: str):
    st.markdown(
        f"""<div class="ss-card">
                <b>{status_label}</b>
                <div class="ss-status-note">{detail}</div>
            </div>""",
        unsafe_allow_html=True,
    )
