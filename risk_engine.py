"""
backend/risk_engine.py

Backend risk-prioritization logic. All scoring happens here in Python -
the frontend only displays the returned values, it never invents them.

Risk is a decision-support priority for a prototype, not an official
regulatory hazard classification.
"""

from __future__ import annotations

from dataclasses import dataclass

RISK_ACTIONS = {
    "LOW": "Monitor / no immediate action.",
    "MEDIUM": "Review / schedule inspection.",
    "HIGH": "Prioritize investigation or removal.",
    "CRITICAL": "Immediate expert review recommended.",
    "UNKNOWN": "Manual expert review (unknown/OOD object).",
}


@dataclass
class RiskResult:
    risk_level: str
    risk_score: float   # 0..1
    reason: str
    recommended_action: str


def compute_risk(detection_confidence: float, artificial_prob: float,
                  anomaly_score: float, class_name: str, is_ood: bool) -> RiskResult:
    """
    Weighted combination of:
      - detection confidence   (35%)
      - artificial probability (30%)  - artificial objects are the operational concern
      - anomaly/OOD score       (35%) - unfamiliar objects warrant more caution
    """
    score = (
        0.35 * detection_confidence
        + 0.30 * artificial_prob
        + 0.35 * anomaly_score
    )
    score = max(0.0, min(1.0, score))

    reasons = []
    if is_ood:
        reasons.append("flagged as unknown/OOD")
        score = max(score, 0.6)  # unknown objects are never treated as trivially low-risk
    if artificial_prob >= 0.7:
        reasons.append(f"high artificial probability ({artificial_prob:.0%})")
    if detection_confidence >= 0.85:
        reasons.append(f"high detection confidence ({detection_confidence:.0%})")
    if class_name in ("Fishing Gear", "Pipeline/Cable", "Shipwreck"):
        reasons.append(f"classified as {class_name.lower()}")

    if score >= 0.85:
        level = "CRITICAL"
    elif score >= 0.65:
        level = "HIGH"
    elif score >= 0.40:
        level = "MEDIUM"
    else:
        level = "LOW"

    if is_ood:
        level = "HIGH" if level in ("LOW", "MEDIUM") else level

    if reasons:
        reason_text = "; ".join(reasons).capitalize() + "."
    else:
        reason_text = "No strong risk indicators; combined score within normal range."

    return RiskResult(
        risk_level=level,
        risk_score=round(score, 4),
        reason=reason_text,
        recommended_action=RISK_ACTIONS.get(level, RISK_ACTIONS["MEDIUM"]),
    )
