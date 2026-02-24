"""
Gateway Layer (Tier 2) — Flask Server

Responsibilities (per the paper):
  1. Data Aggregation   — receives raw readings from IoT devices
  2. Pre-Processing     — normalisation, cleaning, feature engineering
  3. Real-Time Comm     — forwards clean data to the Medical Center Server
  4. mHealth Interface  — lightweight REST API for mobile app queries

Run:  python gateway_layer/gateway_server.py
"""

import sys
import os
import json
import logging
import requests
from flask import Flask, request, jsonify
from datetime import datetime, timezone
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GATEWAY_HOST, GATEWAY_PORT, MEDICAL_CENTER_HOST, MEDICAL_CENTER_PORT
from gateway_layer.preprocessor import preprocess, validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GATEWAY] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MEDICAL_CENTER_URL = f"http://{MEDICAL_CENTER_HOST}:{MEDICAL_CENTER_PORT}/api/vitals"

# In-memory ring buffer for the last 500 readings (for mHealth queries)
_recent_readings: deque = deque(maxlen=500)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "gateway", "timestamp": _now()}), 200


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Entry point for IoT sensor data (Tier 1 → Tier 2).
    Validates, pre-processes, caches, then forwards to the Medical Center.
    """
    raw = request.get_json(silent=True)
    if not raw:
        return jsonify({"error": "Empty or non-JSON payload"}), 400

    ok, err = validate(raw)
    if not ok:
        logger.warning("Rejected payload from %s: %s", raw.get("patient_id", "?"), err)
        return jsonify({"error": err}), 422

    enriched = preprocess(raw)
    if enriched is None:
        return jsonify({"error": "Pre-processing failed"}), 500

    enriched["gateway_received_at"] = _now()
    _recent_readings.appendleft(enriched)

    _forward_to_medical_center(enriched)

    return jsonify({"status": "accepted", "patient_id": enriched["patient_id"]}), 202


@app.route("/api/recent", methods=["GET"])
def recent_readings():
    """mHealth endpoint — last N readings for a patient or all patients."""
    patient_id = request.args.get("patient_id")
    limit = min(int(request.args.get("limit", 20)), 100)

    data = list(_recent_readings)
    if patient_id:
        data = [r for r in data if r.get("patient_id") == patient_id]
    return jsonify(data[:limit]), 200


@app.route("/api/patients", methods=["GET"])
def list_patients():
    """Return distinct patient IDs seen by this gateway."""
    seen = {}
    for r in _recent_readings:
        pid = r.get("patient_id")
        if pid and pid not in seen:
            seen[pid] = {"patient_id": pid, "name": r.get("patient_name", ""),
                         "last_seen": r.get("gateway_received_at", "")}
    return jsonify(list(seen.values())), 200


def _forward_to_medical_center(payload: dict) -> None:
    """Push pre-processed data to the Medical Center Server (Tier 2 → Tier 3)."""
    try:
        resp = requests.post(MEDICAL_CENTER_URL, json=payload, timeout=8)
        resp.raise_for_status()
        logger.info("Forwarded %s → Medical Center", payload["patient_id"])
    except requests.exceptions.ConnectionError:
        logger.warning("Medical Center unreachable — data buffered locally.")
    except Exception as exc:
        logger.error("Error forwarding to Medical Center: %s", exc)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    logger.info("Gateway server starting on %s:%d", GATEWAY_HOST, GATEWAY_PORT)
    app.run(host=GATEWAY_HOST, port=GATEWAY_PORT, debug=False)
