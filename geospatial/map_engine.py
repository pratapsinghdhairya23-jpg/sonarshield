"""
geospatial/map_engine.py

Builds an interactive map of detected objects.

Primary path: Folium (rich popups, filtering-friendly).
Fallback: if folium isn't installed, returns a plain pandas DataFrame the
caller can render with Streamlit's built-in st.map() so the Geospatial
Intelligence page never breaks the whole app over a missing optional dep.

Locations that were manually typed in by the user (not extracted from real
sonar-log metadata) are always labelled "User Provided / Demo Location" -
never presented as verified GPS.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

try:
    import folium
    _HAS_FOLIUM = True
except ImportError:  # pragma: no cover
    _HAS_FOLIUM = False

RISK_COLORS = {
    "LOW": "#35B779",
    "MEDIUM": "#F2B84B",
    "HIGH": "#E76F51",
    "CRITICAL": "#E76F51",
    "UNKNOWN": "#8B6FD8",
}


def detections_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """rows: joined scan+detection dicts with latitude/longitude/class_name/risk_level/etc."""
    geo_rows = [r for r in rows if r.get("latitude") is not None and r.get("longitude") is not None]
    if not geo_rows:
        return pd.DataFrame(columns=["lat", "lon", "object_id", "class_name", "risk_level"])
    df = pd.DataFrame(geo_rows)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    return df


def build_map(rows: list[dict], risk_filter: Optional[list[str]] = None,
              class_filter: Optional[list[str]] = None):
    """
    Returns a folium.Map if folium is available, else None
    (caller should fall back to st.map on the dataframe from detections_to_dataframe).
    """
    geo_rows = [r for r in rows if r.get("latitude") is not None and r.get("longitude") is not None]
    if risk_filter:
        geo_rows = [r for r in geo_rows if r.get("risk_level") in risk_filter]
    if class_filter:
        geo_rows = [r for r in geo_rows if r.get("class_name") in class_filter]

    if not geo_rows or not _HAS_FOLIUM:
        return None

    avg_lat = sum(r["latitude"] for r in geo_rows) / len(geo_rows)
    avg_lon = sum(r["longitude"] for r in geo_rows) / len(geo_rows)
    fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=6, tiles="CartoDB positron")

    for r in geo_rows:
        color = RISK_COLORS.get(r.get("risk_level", ""), "#16A6A0")
        loc_label = "User Provided / Demo Location" if r.get("location_source") == "user_provided" else "Location metadata"
        popup_html = (
            f"<b>{r.get('object_id')}</b><br>"
            f"Scan: {r.get('scan_id')}<br>"
            f"Class: {r.get('class_name')}<br>"
            f"Risk: {r.get('risk_level')}<br>"
            f"Confidence: {r.get('detection_confidence', 0):.0%}<br>"
            f"<i>{loc_label}</i>"
        )
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fmap)

    return fmap
