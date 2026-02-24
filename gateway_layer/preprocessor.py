"""
Gateway Layer — Data Preprocessor

Implements Step 3 of the paper's pseudocode:
  - Normalization of vital sign readings
  - Detection and handling of missing/anomalous values
  - Feature engineering for the ML classifier
"""

import math
from typing import Optional

# Min-max bounds used for normalization (expanded from config VITAL_RANGES)
FEATURE_BOUNDS = {
    "heart_rate":       (30,  200),
    "systolic_bp":      (60,  250),
    "diastolic_bp":     (40,  150),
    "spo2":             (70,  100),
    "body_temperature": (34.0, 42.0),
    "respiratory_rate": (5,   60),
}

REQUIRED_FIELDS = list(FEATURE_BOUNDS.keys())


def _min_max_normalize(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return round((value - lo) / (hi - lo), 6)


def validate(payload: dict) -> tuple[bool, str]:
    """Return (is_valid, error_message)."""
    for field in ["patient_id"] + REQUIRED_FIELDS:
        if field not in payload:
            return False, f"Missing required field: {field}"
    for field in REQUIRED_FIELDS:
        val = payload[field]
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            return False, f"Invalid value for {field}: {val}"
        lo, hi = FEATURE_BOUNDS[field]
        if not (lo <= val <= hi):
            return False, f"Out-of-range value for {field}: {val} (expected {lo}-{hi})"
    return True, ""


def preprocess(payload: dict) -> Optional[dict]:
    """
    Clean and normalise the raw sensor payload.
    Returns enriched dict ready for the Medical Center Server, or None on failure.
    """
    ok, err = validate(payload)
    if not ok:
        return None

    normalized = {
        f"norm_{field}": _min_max_normalize(payload[field], *FEATURE_BOUNDS[field])
        for field in REQUIRED_FIELDS
    }

    # Mean Arterial Pressure (MAP) — useful derived feature
    map_val = round(payload["diastolic_bp"] + (payload["systolic_bp"] - payload["diastolic_bp"]) / 3, 2)

    # Pulse pressure
    pp = round(payload["systolic_bp"] - payload["diastolic_bp"], 2)

    enriched = {
        **payload,
        **normalized,
        "map": map_val,
        "pulse_pressure": pp,
        "preprocessed": True,
    }
    return enriched
