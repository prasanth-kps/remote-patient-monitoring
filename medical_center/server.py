"""
Medical Center Server (Tier 3) — Main Flask Application

Responsibilities:
  - Receive pre-processed vitals from the Gateway Layer
  - Classify health status with the K-star model
  - Persist all readings and alerts to SQLite
  - Generate and dispatch real-time alerts
  - Serve the web dashboard and REST API

Run:  python medical_center/server.py
"""

import sys
import os
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MEDICAL_CENTER_HOST, MEDICAL_CENTER_PORT, SECRET_KEY, HEALTH_STATUS
from medical_center import database as db
from medical_center import classifier
from medical_center import alerts as alert_engine
from medical_center.ahp_vikor import default_scenario, run_hospital_selection
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MEDICAL] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))
app.secret_key = SECRET_KEY
CORS(app)


# ────────────────────────────────────────────────────────────────────────────
# Startup
# ────────────────────────────────────────────────────────────────────────────

@app.before_request
def _startup():
    app.before_request_funcs[None].remove(_startup)
    db.init_db()
    classifier._get_model()   # warm-up: train/load the classifier once
    logger.info("Medical Center Server ready.")


# ────────────────────────────────────────────────────────────────────────────
# Dashboard
# ────────────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ────────────────────────────────────────────────────────────────────────────
# Ingest endpoint  (Gateway → Medical Center)
# ────────────────────────────────────────────────────────────────────────────

@app.route("/api/vitals", methods=["POST"])
def ingest_vitals():
    """
    Receive pre-processed vitals from the Gateway Layer.
    Pipeline: classify → persist → alert-check → respond.
    """
    data = request.get_json(silent=True)
    if not data or "patient_id" not in data:
        return jsonify({"error": "Invalid payload"}), 400

    # Ensure patient record exists
    db.upsert_patient(
        data["patient_id"],
        data.get("patient_name", data["patient_id"]),
        data.get("age", 0),
        data.get("gender", "U"),
        data.get("condition", "Unknown"),
    )

    # Step 4: Classify health status
    status_code, status_label, probabilities = classifier.predict(data)

    # Persist
    vital_id = db.insert_vital(data, status_code, status_label)

    # Step 5: Evaluate alert rules
    triggered = alert_engine.evaluate_vitals(data)
    alert_ids = []
    for a in triggered:
        aid = db.insert_alert(
            data["patient_id"], vital_id,
            a["vital_key"], a["severity"], a["message"],
        )
        alert_ids.append(aid)
        logger.warning(
            "[%s] %s — %s: %s",
            a["severity"].upper(), data["patient_id"], a["vital_key"], a["message"],
        )

    # Email dispatch for critical alerts
    if any(a["severity"] == "critical" for a in triggered):
        alert_engine.send_email_alert(
            data["patient_id"],
            data.get("patient_name", data["patient_id"]),
            triggered, data,
        )

    return jsonify({
        "status":       "ok",
        "vital_id":     vital_id,
        "health_status": status_code,
        "health_label":  status_label,
        "probabilities": probabilities,
        "alerts_raised": len(triggered),
        "alert_ids":     alert_ids,
    }), 201


# ────────────────────────────────────────────────────────────────────────────
# Patient REST API
# ────────────────────────────────────────────────────────────────────────────

@app.route("/api/patients", methods=["GET"])
def list_patients():
    return jsonify(db.get_all_patients()), 200


@app.route("/api/patients/<patient_id>", methods=["GET"])
def get_patient(patient_id):
    patient = db.get_patient(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    stats = db.get_vitals_stats(patient_id)
    return jsonify({**patient, "stats": stats}), 200


@app.route("/api/patients/<patient_id>/vitals", methods=["GET"])
def get_patient_vitals(patient_id):
    limit = min(int(request.args.get("limit", 50)), 200)
    return jsonify(db.get_vitals(patient_id, limit)), 200


# ────────────────────────────────────────────────────────────────────────────
# Dashboard API  (latest readings + overview)
# ────────────────────────────────────────────────────────────────────────────

@app.route("/api/dashboard", methods=["GET"])
def dashboard_data():
    """One-shot payload for the dashboard UI."""
    latest_vitals = db.get_latest_vitals_all(limit=20)
    unacked_alerts = db.get_alerts(acknowledged=False, limit=50)
    patients       = db.get_all_patients()

    # Summary statistics
    normal_count   = sum(1 for v in latest_vitals if v.get("health_status") == 0)
    critical_count = sum(1 for a in unacked_alerts if a.get("severity") == "critical")
    warning_count  = sum(1 for a in unacked_alerts if a.get("severity") == "warning")

    return jsonify({
        "patients":          patients,
        "latest_vitals":     latest_vitals,
        "unacked_alerts":    unacked_alerts,
        "summary": {
            "total_patients":  len(patients),
            "normal_patients": normal_count,
            "critical_alerts": critical_count,
            "warning_alerts":  warning_count,
        },
        "health_status_map": HEALTH_STATUS,
        "timestamp": _now(),
    }), 200


# ────────────────────────────────────────────────────────────────────────────
# Alerts API
# ────────────────────────────────────────────────────────────────────────────

@app.route("/api/alerts", methods=["GET"])
def list_alerts():
    acked_param = request.args.get("acknowledged")
    if acked_param is None:
        alerts = db.get_alerts(acknowledged=None)
    else:
        alerts = db.get_alerts(acknowledged=acked_param.lower() in ("1", "true", "yes"))
    return jsonify(alerts), 200


@app.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id):
    db.acknowledge_alert(alert_id)
    return jsonify({"status": "acknowledged", "alert_id": alert_id}), 200


# ────────────────────────────────────────────────────────────────────────────
# AHP-VIKOR Hospital Selection API
# ────────────────────────────────────────────────────────────────────────────

@app.route("/api/hospital-selection", methods=["GET"])
def get_hospital_selection():
    """Return the latest saved ranking or run the default scenario."""
    cached = db.get_latest_hospital_ranking()
    if cached:
        return jsonify(cached), 200
    result = default_scenario()
    db.save_hospital_ranking(result)
    return jsonify(result), 200


@app.route("/api/hospital-selection", methods=["POST"])
def run_hospital_selection_api():
    """
    Run custom AHP-VIKOR.
    Body: { hospitals, criteria, pairwise_matrix, benefit_criteria }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "JSON body required"}), 400
    try:
        result = run_hospital_selection(
            body["hospitals"],
            body["criteria"],
            np.array(body["pairwise_matrix"]),
            body["benefit_criteria"],
        )
        db.save_hospital_ranking(result)
        return jsonify(result), 200
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422


# ────────────────────────────────────────────────────────────────────────────
# Classifier API
# ────────────────────────────────────────────────────────────────────────────

@app.route("/api/classify", methods=["POST"])
def classify_vitals():
    """Classify a set of vitals without persisting them (for testing)."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    code, label, proba = classifier.predict(data)
    return jsonify({"health_status": code, "health_label": label, "probabilities": proba}), 200


@app.route("/api/classifier/retrain", methods=["POST"])
def retrain_classifier():
    result = classifier.retrain()
    return jsonify(result), 200


# ────────────────────────────────────────────────────────────────────────────
# Health check
# ────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "medical-center", "timestamp": _now()}), 200


# ────────────────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    logger.info("Medical Center Server starting on %s:%d", MEDICAL_CENTER_HOST, MEDICAL_CENTER_PORT)
    app.run(host=MEDICAL_CENTER_HOST, port=MEDICAL_CENTER_PORT, debug=False)
