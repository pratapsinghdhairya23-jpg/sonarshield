"""
backend/anomaly_engine.py

Prototype Out-Of-Distribution (OOD) / anomaly flagging layer.

HONESTY NOTE: this is NOT a trained OOD neural network (e.g. a Mahalanobis /
energy-based OOD head trained on AI4Shipwrecks embeddings). It is a rule-based
prototype that flags an object as UNKNOWN / OOD when the classifier's
confidence is low, the class was already routed to "Unknown Anomaly", or the
blob's features fall outside the ranges the heuristic classifier was tuned
for. It is always surfaced in the UI as "Prototype OOD Analysis".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnomalyResult:
    anomaly_score: float          # 0..1, higher = more anomalous / less familiar
    is_ood: bool
    status: str                    # "Known" | "UNKNOWN / OOD ANOMALY"
    recommendation: str


def compute_anomaly(class_name: str, class_confidence: float, contrast: float,
                     fill_ratio: float) -> AnomalyResult:
    # Base anomaly score rises as classification confidence falls.
    score = 1.0 - class_confidence

    # Extreme geometry (very sparse or near-solid blobs) is under-represented
    # in the heuristic's tuning range, so nudge the anomaly score up.
    if fill_ratio < 0.15 or fill_ratio > 0.92:
        score += 0.15

    # Very low sonar-return contrast makes classification unreliable.
    if contrast < 0.08:
        score += 0.10

    score = max(0.0, min(1.0, score))

    is_ood = class_name == "Unknown Anomaly" or score >= 0.55
    status = "UNKNOWN / OOD ANOMALY" if is_ood else "Known"

    if is_ood:
        recommendation = "Manual expert review recommended (prototype OOD flag)."
    else:
        recommendation = "Matches expected prototype feature range for its class."

    return AnomalyResult(
        anomaly_score=round(score, 4),
        is_ood=is_ood,
        status=status,
        recommendation=recommendation,
    )
