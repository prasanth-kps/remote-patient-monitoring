"""
Remote Patient Monitoring — All-in-One Demo Server

Runs the complete three-tier RPM system in a single process:
  - Built-in IoT simulator (background thread, no external devices needed)
  - Gateway pre-processing (in-process, no HTTP hop)
  - Medical Center Flask server on 0.0.0.0:$PORT

Deploy to Render.com, Railway, or any PaaS with:
    Start command:  gunicorn demo_server:app
"""

import os
import sys
import time
import random
import threading
import math
import json
import pickle
import logging
import sqlite3
import smtplib
from datetime import datetime, timezone
from contextlib import contextmanager
from collections import deque
from typing import Optional

import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RPM] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

PORT                   = int(os.getenv("PORT", 5002))
NUM_SIMULATED_PATIENTS = int(os.getenv("NUM_SIMULATED_PATIENTS", 6))
SENSOR_SEND_INTERVAL   = int(os.getenv("SENSOR_SEND_INTERVAL", 15))  # faster for demo
DATABASE_PATH          = os.getenv("DATABASE_PATH", "/tmp/rpm_demo.db")

VITAL_RANGES = {
    "heart_rate":       {"min": 60,  "max": 100,  "unit": "bpm"},
    "systolic_bp":      {"min": 90,  "max": 120,  "unit": "mmHg"},
    "diastolic_bp":     {"min": 60,  "max": 80,   "unit": "mmHg"},
    "spo2":             {"min": 95,  "max": 100,  "unit": "%"},
    "body_temperature": {"min": 36.1,"max": 37.2, "unit": "°C"},
    "respiratory_rate": {"min": 12,  "max": 20,   "unit": "breaths/min"},
}

HEALTH_STATUS = {
    0: "Normal",
    1: "Hypercholesterolemia (HCLS)",
    2: "Hypertension (HTN)",
    3: "Heart Disease (HD)",
    4: "Blood Pressure Issue (BP)",
    5: "Oxygen Saturation Issue (SpO₂)",
}

DEMO_PATIENTS = [
    {"patient_id": "P001", "name": "Arjun Mehta",    "age": 54, "gender": "M", "condition": "Hypertension"},
    {"patient_id": "P002", "name": "Priya Sharma",   "age": 67, "gender": "F", "condition": "Heart Disease"},
    {"patient_id": "P003", "name": "Ravi Kumar",     "age": 45, "gender": "M", "condition": "Diabetes"},
    {"patient_id": "P004", "name": "Lakshmi Nair",   "age": 72, "gender": "F", "condition": "COPD"},
    {"patient_id": "P005", "name": "Suresh Iyer",    "age": 38, "gender": "M", "condition": "Healthy"},
    {"patient_id": "P006", "name": "Ananya Reddy",   "age": 61, "gender": "F", "condition": "Hypertension"},
][:NUM_SIMULATED_PATIENTS]

# ─── Database ─────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY, name TEXT, age INTEGER,
    gender TEXT, condition TEXT,
    registered_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE IF NOT EXISTS vital_signs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL, timestamp TEXT NOT NULL,
    heart_rate REAL, systolic_bp REAL, diastolic_bp REAL,
    spo2 REAL, body_temperature REAL, respiratory_rate REAL,
    map REAL, pulse_pressure REAL,
    health_status INTEGER DEFAULT 0, health_label TEXT DEFAULT 'Normal',
    device_id TEXT, location TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL, vital_sign_id INTEGER,
    alert_type TEXT, severity TEXT, message TEXT,
    acknowledged INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE IF NOT EXISTS hospital_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    results_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_vp ON vital_signs(patient_id);
CREATE INDEX IF NOT EXISTS idx_vt ON vital_signs(timestamp);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(DDL)


def upsert_patient(p):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO patients (patient_id,name,age,gender,condition)
               VALUES (?,?,?,?,?)
               ON CONFLICT(patient_id) DO UPDATE SET
               name=excluded.name,age=excluded.age,
               gender=excluded.gender,condition=excluded.condition""",
            (p["patient_id"], p["name"], p["age"], p["gender"], p["condition"]),
        )


def insert_vital(data, health_status, health_label):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO vital_signs
               (patient_id,timestamp,heart_rate,systolic_bp,diastolic_bp,
                spo2,body_temperature,respiratory_rate,map,pulse_pressure,
                health_status,health_label,device_id,location)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (data["patient_id"], data.get("timestamp", _now()),
             data.get("heart_rate"), data.get("systolic_bp"), data.get("diastolic_bp"),
             data.get("spo2"), data.get("body_temperature"), data.get("respiratory_rate"),
             data.get("map"), data.get("pulse_pressure"),
             health_status, health_label, data.get("device_id", ""), data.get("location", "")),
        )
        return cur.lastrowid


def insert_alert(patient_id, vital_id, alert_type, severity, message):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO alerts (patient_id,vital_sign_id,alert_type,severity,message) VALUES (?,?,?,?,?)",
            (patient_id, vital_id, alert_type, severity, message),
        )
        return cur.lastrowid


def db_get_all_patients():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM patients ORDER BY patient_id").fetchall()]


def db_get_vitals(patient_id, limit=50):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id,patient_id,timestamp,heart_rate,systolic_bp,diastolic_bp,
                      spo2,body_temperature,respiratory_rate,map,pulse_pressure,
                      health_status,health_label,location
               FROM vital_signs WHERE patient_id=?
               ORDER BY timestamp DESC LIMIT ?""",
            (patient_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def db_get_latest_vitals_all():
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT v.* FROM vital_signs v
               INNER JOIN (SELECT patient_id, MAX(timestamp) AS mt
                           FROM vital_signs GROUP BY patient_id) l
               ON v.patient_id=l.patient_id AND v.timestamp=l.mt
               ORDER BY v.timestamp DESC""",
        ).fetchall()
    return [dict(r) for r in rows]


def db_get_alerts(acknowledged=None, limit=50):
    with get_conn() as conn:
        if acknowledged is None:
            rows = conn.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE acknowledged=? ORDER BY created_at DESC LIMIT ?",
                (int(acknowledged), limit),
            ).fetchall()
    return [dict(r) for r in rows]


def db_acknowledge_alert(alert_id):
    with get_conn() as conn:
        conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))


def db_save_hospital_ranking(results):
    with get_conn() as conn:
        conn.execute("INSERT INTO hospital_rankings (results_json) VALUES (?)", (json.dumps(results),))


def db_get_latest_hospital_ranking():
    with get_conn() as conn:
        row = conn.execute(
            "SELECT results_json FROM hospital_rankings ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
    return json.loads(row["results_json"]) if row else None


# ─── K-star Classifier ────────────────────────────────────────────────────────

_model: Optional[Pipeline] = None
MODEL_PATH = "/tmp/rpm_kstar_model.pkl"


def _generate_training_data(n=500):
    rng = np.random.default_rng(42)
    def s(hr, sbp, dbp, spo2, temp, rr, label):
        X = np.column_stack([rng.normal(*hr, n), rng.normal(*sbp, n), rng.normal(*dbp, n),
                             rng.normal(*spo2, n), rng.normal(*temp, n), rng.normal(*rr, n)])
        return X, np.full(n, label)
    datasets = [
        s((78,8),(112,8),(72,6),(98,1),(36.6,.3),(15,2),  0),
        s((82,9),(128,10),(84,7),(97,1),(36.7,.4),(16,2), 1),
        s((85,10),(155,12),(98,8),(96,1.5),(36.8,.4),(17,2),2),
        s((92,15),(140,15),(90,10),(93,3),(36.9,.5),(20,4),3),
        s((75,12),(175,15),(108,10),(96,2),(36.7,.4),(16,3),4),
        s((95,12),(118,10),(76,8),(88,4),(37.1,.6),(22,5), 5),
    ]
    X = np.vstack([d[0] for d in datasets])
    y = np.concatenate([d[1] for d in datasets])
    X[:, 0] = np.clip(X[:, 0], 30, 200)
    X[:, 1] = np.clip(X[:, 1], 60, 250)
    X[:, 2] = np.clip(X[:, 2], 40, 150)
    X[:, 3] = np.clip(X[:, 3], 70, 100)
    X[:, 4] = np.clip(X[:, 4], 34, 42)
    X[:, 5] = np.clip(X[:, 5], 5, 60)
    return X, y


def _train_model():
    logger.info("Training K-star classifier…")
    X, y = _generate_training_data()
    pipe = Pipeline([("sc", MinMaxScaler()), ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))])
    pipe.fit(X, y)
    scores = cross_val_score(pipe, X, y, cv=10, scoring="accuracy")
    logger.info("10-fold CV accuracy: %.1f%% ± %.1f%%", scores.mean()*100, scores.std()*100)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    _model = pickle.load(f)
            except Exception:
                _model = _train_model()
        else:
            _model = _train_model()
    return _model


def classify(vitals):
    try:
        X = np.array([[vitals["heart_rate"], vitals["systolic_bp"], vitals["diastolic_bp"],
                       vitals["spo2"], vitals["body_temperature"], vitals["respiratory_rate"]]])
        m = get_model()
        code = int(m.predict(X)[0])
        proba = m.predict_proba(X)[0]
        proba_dict = {HEALTH_STATUS[int(c)]: round(float(p), 4) for c, p in zip(m.classes_, proba)}
        return code, HEALTH_STATUS.get(code, "Normal"), proba_dict
    except Exception as exc:
        logger.error("Classifier error: %s", exc)
        return 0, "Normal", {}


# ─── Alert Engine ─────────────────────────────────────────────────────────────

ALERT_RULES = [
    ("heart_rate",        ">", 130,  "critical", "Tachycardia — HR={value:.0f} bpm"),
    ("heart_rate",        "<", 45,   "critical", "Severe Bradycardia — HR={value:.0f} bpm"),
    ("heart_rate",        ">", 100,  "warning",  "Elevated heart rate — HR={value:.0f} bpm"),
    ("heart_rate",        "<", 60,   "warning",  "Low heart rate — HR={value:.0f} bpm"),
    ("systolic_bp",       ">", 180,  "critical", "Hypertensive crisis — SBP={value:.0f} mmHg"),
    ("systolic_bp",       "<", 80,   "critical", "Hypotension — SBP={value:.0f} mmHg"),
    ("systolic_bp",       ">", 140,  "warning",  "Elevated systolic BP — SBP={value:.0f} mmHg"),
    ("systolic_bp",       "<", 90,   "warning",  "Low systolic BP — SBP={value:.0f} mmHg"),
    ("diastolic_bp",      ">", 120,  "critical", "Hypertensive crisis — DBP={value:.0f} mmHg"),
    ("diastolic_bp",      ">", 90,   "warning",  "Elevated diastolic BP — DBP={value:.0f} mmHg"),
    ("spo2",              "<", 90,   "critical", "Severe hypoxemia — SpO₂={value:.1f}%"),
    ("spo2",              "<", 94,   "warning",  "Low oxygen saturation — SpO₂={value:.1f}%"),
    ("body_temperature",  ">", 39.5, "critical", "High fever — Temp={value:.1f}°C"),
    ("body_temperature",  "<", 35.0, "critical", "Hypothermia — Temp={value:.1f}°C"),
    ("body_temperature",  ">", 38.0, "warning",  "Fever — Temp={value:.1f}°C"),
    ("respiratory_rate",  ">", 30,   "critical", "Severe tachypnea — RR={value:.0f} br/min"),
    ("respiratory_rate",  "<", 8,    "critical", "Severe bradypnea — RR={value:.0f} br/min"),
    ("respiratory_rate",  ">", 20,   "warning",  "Elevated respiratory rate — RR={value:.0f} br/min"),
    ("respiratory_rate",  "<", 12,   "warning",  "Low respiratory rate — RR={value:.0f} br/min"),
]


def evaluate_alerts(vitals):
    triggered, seen = [], set()
    for key, op, thr, sev, tmpl in ALERT_RULES:
        val = vitals.get(key)
        if val is None or key in seen:
            continue
        if (op == ">" and val > thr) or (op == "<" and val < thr):
            triggered.append({"vital_key": key, "severity": sev,
                               "message": tmpl.format(value=val), "value": val})
            if sev == "critical":
                seen.add(key)
    return triggered


# ─── AHP-VIKOR ────────────────────────────────────────────────────────────────

_RI = {1:0,2:0,3:.58,4:.9,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}


def run_ahp_vikor():
    hospitals = [
        {"name": "City General Hospital",     "scores": [8.5, 7.0, 9.0, 6.5, 8.0]},
        {"name": "Metro Medical Center",      "scores": [7.0, 9.0, 7.5, 8.0, 7.5]},
        {"name": "Regional Health Institute", "scores": [6.5, 7.5, 8.0, 9.0, 6.0]},
        {"name": "St. Luke's Hospital",       "scores": [9.0, 6.5, 7.0, 7.5, 9.0]},
        {"name": "Community Care Hospital",   "scores": [7.5, 8.0, 6.5, 8.5, 7.0]},
    ]
    criteria     = ["ICU Capacity", "Telemedicine", "Response Time", "EHR Integration", "Specialists"]
    benefit      = [True, True, False, True, True]
    pairwise     = np.array([[1,2,3,2,2],[.5,1,2,1,1],[1/3,.5,1,.5,.5],[.5,1,2,1,1],[.5,1,2,1,1]])

    col_sums     = pairwise.sum(axis=0)
    norm         = pairwise / col_sums
    weights      = norm.mean(axis=1)
    Aw           = pairwise @ weights
    lmax         = float(np.mean(Aw / weights))
    n            = len(criteria)
    CI           = (lmax - n) / (n - 1)
    CR           = CI / _RI[n]

    X = np.array([h["scores"] for h in hospitals], dtype=float)
    names = [h["name"] for h in hospitals]

    f_best  = np.array([X[:,j].max() if benefit[j] else X[:,j].min() for j in range(len(criteria))])
    f_worst = np.array([X[:,j].min() if benefit[j] else X[:,j].max() for j in range(len(criteria))])
    denom   = np.where(np.abs(f_best - f_worst) < 1e-12, 1e-12, f_best - f_worst)
    nd      = weights * (f_best - X) / denom
    S, R    = nd.sum(axis=1), nd.max(axis=1)
    sd, rd  = (S.max()-S.min()) or 1e-12, (R.max()-R.min()) or 1e-12
    Q       = .5*(S-S.min())/sd + .5*(R-R.min())/rd

    rankings = [{"rank": int(r+1), "hospital": names[i], "Q": round(float(Q[i]),4),
                  "S": round(float(S[i]),4), "R": round(float(R[i]),4)}
                for r, i in enumerate(np.argsort(Q))]

    return {
        "ahp": {"criteria": criteria, "weights": {c: round(float(w),6) for c,w in zip(criteria, weights)},
                "consistency_ratio": round(CR, 6), "consistent": CR <= 0.10},
        "vikor": {"rankings": rankings, "hospitals": names, "criteria": criteria},
    }


# ─── IoT Simulator (background thread) ───────────────────────────────────────

def _noise(v, pct=0.03):
    return round(v + v * pct * random.gauss(0, 1), 2)


def _gen_vitals(patient):
    c = patient["condition"]
    if c == "Hypertension":
        hr,sbp,dbp,spo2,temp,rr = _noise(random.uniform(70,110)),_noise(random.uniform(130,175)),\
            _noise(random.uniform(85,108)),_noise(random.uniform(94,99)),\
            _noise(random.uniform(36.0,37.5)),_noise(random.uniform(14,22))
    elif c == "Heart Disease":
        hr,sbp,dbp,spo2,temp,rr = _noise(random.uniform(55,115)),_noise(random.uniform(100,160)),\
            _noise(random.uniform(65,100)),_noise(random.uniform(90,98)),\
            _noise(random.uniform(36.0,37.8)),_noise(random.uniform(14,24))
    elif c == "COPD":
        hr,sbp,dbp,spo2,temp,rr = _noise(random.uniform(70,105)),_noise(random.uniform(100,140)),\
            _noise(random.uniform(65,90)),_noise(random.uniform(85,95)),\
            _noise(random.uniform(36.0,38.0)),_noise(random.uniform(18,28))
    elif c == "Diabetes":
        hr,sbp,dbp,spo2,temp,rr = _noise(random.uniform(65,100)),_noise(random.uniform(110,145)),\
            _noise(random.uniform(70,95)),_noise(random.uniform(93,99)),\
            _noise(random.uniform(36.0,37.5)),_noise(random.uniform(13,20))
    else:
        hr,sbp,dbp,spo2,temp,rr = _noise(random.uniform(60,100)),_noise(random.uniform(90,120)),\
            _noise(random.uniform(60,80)),_noise(random.uniform(96,100)),\
            _noise(random.uniform(36.1,37.2)),_noise(random.uniform(12,18))

    map_val = round(dbp + (sbp - dbp) / 3, 2)
    return {
        "patient_id":       patient["patient_id"],
        "patient_name":     patient["name"],
        "age":              patient["age"],
        "gender":           patient["gender"],
        "condition":        c,
        "heart_rate":       round(max(30, hr), 1),
        "systolic_bp":      round(max(60, sbp), 1),
        "diastolic_bp":     round(max(40, dbp), 1),
        "spo2":             round(min(100, max(70, spo2)), 1),
        "body_temperature": round(temp, 2),
        "respiratory_rate": round(max(8, rr), 1),
        "map":              map_val,
        "pulse_pressure":   round(sbp - dbp, 2),
        "timestamp":        _now(),
        "device_id":        f"DEV-{patient['patient_id']}",
        "location":         random.choice(["Home", "Ward A", "Ward B", "ICU"]),
    }


def _ingest(vitals):
    """Process one vitals reading in-process (no HTTP)."""
    upsert_patient({
        "patient_id": vitals["patient_id"], "name": vitals["patient_name"],
        "age": vitals["age"], "gender": vitals["gender"], "condition": vitals["condition"],
    })
    code, label, _ = classify(vitals)
    vid = insert_vital(vitals, code, label)
    for a in evaluate_alerts(vitals):
        insert_alert(vitals["patient_id"], vid, a["vital_key"], a["severity"], a["message"])


def _simulator_loop():
    time.sleep(3)  # wait for DB init
    logger.info("IoT simulator started (%d patients, %ds interval)", len(DEMO_PATIENTS), SENSOR_SEND_INTERVAL)
    while True:
        for patient in DEMO_PATIENTS:
            try:
                _ingest(_gen_vitals(patient))
            except Exception as exc:
                logger.error("Simulator error for %s: %s", patient["patient_id"], exc)
            time.sleep(0.3)
        time.sleep(SENSOR_SEND_INTERVAL)


# ─── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Remote Patient Monitoring — Live Demo</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root{--bg:#0f1117;--surface:#1a1d2e;--card:#1e2235;--border:#2a2f4a;--text:#e2e8f0;--muted:#8892a4;--accent:#4f8ef7;--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--radius:12px}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
    nav{background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
    .nav-brand{display:flex;align-items:center;gap:10px;font-size:1.05rem;font-weight:700;color:var(--accent)}
    .nav-right{display:flex;align-items:center;gap:16px;font-size:.8rem;color:var(--muted)}
    .pulse-dot{width:8px;height:8px;background:var(--green);border-radius:50%;animation:pulse 2s infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
    main{padding:24px;max-width:1400px;margin:0 auto}
    .tabs{display:flex;gap:4px;background:var(--surface);border-radius:8px;padding:4px;margin-bottom:20px}
    .tab{flex:1;padding:9px 12px;border:none;background:transparent;color:var(--muted);font-size:.82rem;font-weight:600;border-radius:6px;cursor:pointer;transition:all .2s}
    .tab.active{background:var(--card);color:var(--text)}
    .tab-section{display:none}.tab-section.active{display:block}
    .summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin-bottom:28px}
    .summary-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:20px}
    .summary-card .label{font-size:.72rem;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em}
    .summary-card .value{font-size:2rem;font-weight:700;line-height:1}
    .summary-card .sub{font-size:.68rem;color:var(--muted);margin-top:4px}
    .card-blue{border-top:3px solid var(--accent)}.card-green{border-top:3px solid var(--green)}
    .card-red{border-top:3px solid var(--red)}.card-yellow{border-top:3px solid var(--yellow)}
    .two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}
    @media(max-width:900px){.two-col{grid-template-columns:1fr}}
    .panel{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:20px}
    .panel-header{font-size:.85rem;font-weight:600;color:var(--text);margin-bottom:16px;display:flex;align-items:center;justify-content:space-between}
    .badge{font-size:.65rem;padding:2px 8px;border-radius:999px;font-weight:700}
    .badge-red{background:rgba(239,68,68,.2);color:var(--red)}.badge-blue{background:rgba(79,142,247,.2);color:var(--accent)}
    table{width:100%;border-collapse:collapse;font-size:.82rem}
    th{text-align:left;padding:8px 12px;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);border-bottom:1px solid var(--border)}
    td{padding:10px 12px;border-bottom:1px solid rgba(42,47,74,.5);color:var(--text)}
    tr:hover td{background:rgba(79,142,247,.05)}
    tr:last-child td{border-bottom:none}
    .status-pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.7rem;font-weight:700}
    .status-0{background:rgba(34,197,94,.15);color:var(--green)}
    .status-1,.status-2{background:rgba(245,158,11,.15);color:var(--yellow)}
    .status-3,.status-4,.status-5{background:rgba(239,68,68,.15);color:var(--red)}
    .alert-item{padding:12px 14px;border-radius:8px;margin-bottom:8px;font-size:.82rem;display:flex;align-items:flex-start;gap:10px}
    .alert-critical{background:rgba(255,56,96,.12);border:1px solid rgba(255,56,96,.3)}
    .alert-warning{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3)}
    .alert-icon{font-size:1rem;flex-shrink:0;margin-top:1px}
    .alert-body .meta{font-size:.68rem;color:var(--muted);margin-top:3px}
    .ack-btn{margin-left:auto;flex-shrink:0;background:transparent;border:1px solid var(--border);color:var(--muted);padding:3px 10px;border-radius:6px;font-size:.7rem;cursor:pointer;transition:all .2s}
    .ack-btn:hover{border-color:var(--accent);color:var(--accent)}
    .chart-wrap{position:relative;height:240px}
    .modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:200;align-items:center;justify-content:center}
    .modal-overlay.open{display:flex}
    .modal{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);width:90%;max-width:820px;max-height:90vh;overflow-y:auto;padding:24px}
    .modal-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
    .modal-header h2{font-size:1rem;font-weight:700}
    .close-btn{background:transparent;border:none;color:var(--muted);font-size:1.2rem;cursor:pointer}
    .vitals-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:12px;margin-bottom:20px}
    .vital-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;text-align:center}
    .vital-card .vname{font-size:.68rem;color:var(--muted);margin-bottom:4px}
    .vital-card .vval{font-size:1.4rem;font-weight:700}
    .vital-card .vunit{font-size:.65rem;color:var(--muted)}
    .vital-ok{color:var(--green)}.vital-warn{color:var(--yellow)}.vital-bad{color:var(--red)}
    .rank-num{width:28px;height:28px;border-radius:50%;background:var(--surface);border:1px solid var(--border);display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem}
    .rank-1{background:rgba(255,215,0,.2);border-color:gold;color:gold}
    .rank-2{background:rgba(192,192,192,.2);border-color:silver;color:silver}
    .rank-3{background:rgba(205,127,50,.2);border-color:#cd7f32;color:#cd7f32}
    .scroll-list{max-height:400px;overflow-y:auto}
    .scroll-list::-webkit-scrollbar{width:4px}
    .scroll-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
    .empty-state{text-align:center;color:var(--muted);font-size:.82rem;padding:32px 0}
    .demo-banner{background:linear-gradient(135deg,rgba(79,142,247,.15),rgba(79,142,247,.05));border:1px solid rgba(79,142,247,.3);border-radius:var(--radius);padding:14px 20px;margin-bottom:20px;font-size:.82rem;display:flex;align-items:center;gap:10px}
    .loader{display:flex;align-items:center;justify-content:center;height:80px;color:var(--muted);font-size:.82rem;gap:8px}
    .spinner{width:16px;height:16px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
  </style>
</head>
<body>
<nav>
  <div class="nav-brand">🏥 Remote Patient Monitoring — Live Demo</div>
  <div class="nav-right">
    <div style="display:flex;align-items:center;gap:6px"><div class="pulse-dot"></div><span id="last-update">Connecting…</span></div>
  </div>
</nav>
<main>
  <div class="demo-banner">
    ℹ️ <strong>Live Demo</strong> — Simulating {{ patient_count }} patients in real time.
    Vitals update every {{ interval }}s. Critical events trigger automatic alerts.
    Data resets on server restart.
  </div>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('overview')">Overview</button>
    <button class="tab" onclick="switchTab('patients')">Patients</button>
    <button class="tab" onclick="switchTab('alerts')">Alerts <span id="alert-badge" class="badge badge-red" style="display:none"></span></button>
    <button class="tab" onclick="switchTab('hospitals')">Hospital Ranking</button>
  </div>

  <!-- Overview -->
  <div id="tab-overview" class="tab-section active">
    <div class="summary-grid">
      <div class="summary-card card-blue"><div class="label">Total Patients</div><div class="value" id="s-total">—</div><div class="sub">being monitored</div></div>
      <div class="summary-card card-green"><div class="label">Normal Status</div><div class="value" id="s-normal">—</div><div class="sub">patients</div></div>
      <div class="summary-card card-red"><div class="label">Critical Alerts</div><div class="value" id="s-critical">—</div><div class="sub">unacknowledged</div></div>
      <div class="summary-card card-yellow"><div class="label">Warnings</div><div class="value" id="s-warning">—</div><div class="sub">unacknowledged</div></div>
    </div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-header">Live Patient Vitals <span class="badge badge-blue" id="vitals-ts"></span></div>
        <div style="overflow-x:auto">
          <table><thead><tr><th>Patient</th><th>HR</th><th>BP</th><th>SpO₂</th><th>Temp</th><th>Status</th></tr></thead>
          <tbody id="vitals-tbody"><tr><td colspan="6"><div class="loader"><div class="spinner"></div> Loading…</div></td></tr></tbody></table>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">Health Status Distribution</div>
        <div class="chart-wrap"><canvas id="status-chart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- Patients -->
  <div id="tab-patients" class="tab-section">
    <div class="panel">
      <div class="panel-header">Patient Registry</div>
      <div style="overflow-x:auto">
        <table><thead><tr><th>ID</th><th>Name</th><th>Age</th><th>Gender</th><th>Condition</th><th>Action</th></tr></thead>
        <tbody id="patients-tbody"><tr><td colspan="6"><div class="loader"><div class="spinner"></div> Loading…</div></td></tr></tbody></table>
      </div>
    </div>
  </div>

  <!-- Alerts -->
  <div id="tab-alerts" class="tab-section">
    <div class="panel">
      <div class="panel-header">Active Alerts
        <button onclick="loadAlerts()" style="background:transparent;border:1px solid var(--border);color:var(--muted);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:.75rem">Refresh</button>
      </div>
      <div id="alerts-list" class="scroll-list"><div class="loader"><div class="spinner"></div> Loading…</div></div>
    </div>
  </div>

  <!-- Hospital Ranking -->
  <div id="tab-hospitals" class="tab-section">
    <div class="two-col">
      <div class="panel">
        <div class="panel-header">VIKOR Hospital Rankings</div>
        <div id="hospital-list"><div class="loader"><div class="spinner"></div> Loading…</div></div>
      </div>
      <div class="panel">
        <div class="panel-header">AHP Criterion Weights</div>
        <div class="chart-wrap" style="height:200px"><canvas id="ahp-chart"></canvas></div>
      </div>
    </div>
  </div>
</main>

<!-- Patient Modal -->
<div class="modal-overlay" id="patient-modal">
  <div class="modal">
    <div class="modal-header">
      <h2 id="modal-title">Patient Details</h2>
      <button class="close-btn" onclick="closeModal()">✕</button>
    </div>
    <div id="modal-vitals-grid" class="vitals-grid"></div>
    <div class="panel-header" style="margin-top:8px">HR &amp; SpO₂ Trend</div>
    <div class="chart-wrap" style="height:200px;margin-bottom:16px"><canvas id="modal-chart"></canvas></div>
    <div style="overflow-x:auto">
      <table><thead><tr><th>Time</th><th>HR</th><th>SBP</th><th>DBP</th><th>SpO₂</th><th>Temp</th><th>RR</th><th>Status</th></tr></thead>
      <tbody id="modal-tbody"></tbody></table>
    </div>
  </div>
</div>

<script>
let statusChart=null,ahpChart=null,modalChart=null;
const HEALTH_STATUS={0:'Normal',1:'Hypercholesterolemia (HCLS)',2:'Hypertension (HTN)',3:'Heart Disease (HD)',4:'Blood Pressure Issue (BP)',5:'Oxygen Saturation Issue (SpO₂)'};

async function fetchDashboard(){
  try{
    const d=await(await fetch('/api/dashboard')).json();
    const s=d.summary;
    document.getElementById('s-total').textContent=s.total_patients;
    document.getElementById('s-normal').textContent=s.normal_patients;
    document.getElementById('s-critical').textContent=s.critical_alerts;
    document.getElementById('s-warning').textContent=s.warning_alerts;
    const tot=s.critical_alerts+s.warning_alerts;
    const b=document.getElementById('alert-badge');
    b.textContent=tot; b.style.display=tot>0?'inline-block':'none';
    document.getElementById('last-update').textContent='Updated '+new Date(d.timestamp).toLocaleTimeString();
    document.getElementById('vitals-ts').textContent=new Date(d.timestamp).toLocaleTimeString();
    renderVitalsTable(d.latest_vitals);
    renderStatusChart(d.latest_vitals);
    renderPatientsTable(d.patients);
  }catch(e){document.getElementById('last-update').textContent='Reconnecting…';}
}

function renderVitalsTable(vitals){
  const tb=document.getElementById('vitals-tbody');
  if(!vitals.length){tb.innerHTML='<tr><td colspan="6"><div class="empty-state">Waiting for first readings…</div></td></tr>';return;}
  tb.innerHTML=vitals.map(v=>`<tr>
    <td><strong>${v.patient_id}</strong></td>
    <td>${v.heart_rate?.toFixed(0)??'—'} <small style="color:var(--muted)">bpm</small></td>
    <td>${v.systolic_bp?.toFixed(0)??'—'}/${v.diastolic_bp?.toFixed(0)??'—'} <small style="color:var(--muted)">mmHg</small></td>
    <td>${v.spo2?.toFixed(1)??'—'}<small style="color:var(--muted)">%</small></td>
    <td>${v.body_temperature?.toFixed(1)??'—'}<small style="color:var(--muted)">°C</small></td>
    <td><span class="status-pill status-${v.health_status}">${HEALTH_STATUS[v.health_status]??'Normal'}</span></td>
  </tr>`).join('');
}

function renderPatientsTable(patients){
  const tb=document.getElementById('patients-tbody');
  if(!patients.length){tb.innerHTML='<tr><td colspan="6"><div class="empty-state">No patients yet.</div></td></tr>';return;}
  tb.innerHTML=patients.map(p=>`<tr>
    <td><code style="color:var(--accent);font-size:.8rem">${p.patient_id}</code></td>
    <td>${p.name}</td><td>${p.age}</td><td>${p.gender}</td><td>${p.condition}</td>
    <td><button onclick="openModal('${p.patient_id}','${p.name}')"
      style="background:transparent;border:1px solid var(--border);color:var(--accent);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:.75rem">View</button></td>
  </tr>`).join('');
}

function renderStatusChart(vitals){
  const counts={};
  vitals.forEach(v=>{const l=HEALTH_STATUS[v.health_status]??'Normal';counts[l]=(counts[l]||0)+1;});
  const labels=Object.keys(counts),values=Object.values(counts);
  const colors=['#22c55e','#f59e0b','#f59e0b','#ef4444','#ef4444','#ef4444'];
  const ctx=document.getElementById('status-chart').getContext('2d');
  if(statusChart)statusChart.destroy();
  statusChart=new Chart(ctx,{type:'doughnut',data:{labels,datasets:[{data:values,backgroundColor:labels.map((_,i)=>colors[i]||'#4f8ef7'),borderWidth:0}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'right',labels:{color:'#8892a4',font:{size:11},padding:14}}}}});
}

async function loadAlerts(){
  const el=document.getElementById('alerts-list');
  const data=await(await fetch('/api/alerts?acknowledged=false')).json();
  if(!data.length){el.innerHTML='<div class="empty-state">No active alerts 🎉</div>';return;}
  el.innerHTML=data.map(a=>`<div class="alert-item alert-${a.severity}" id="alert-${a.id}">
    <span class="alert-icon">${a.severity==='critical'?'🚨':'⚠️'}</span>
    <div class="alert-body"><strong>${a.message}</strong><div class="meta">${a.patient_id} · ${a.created_at?.replace('T',' ').replace('Z','')}</div></div>
    <button class="ack-btn" onclick="ackAlert(${a.id})">Acknowledge</button>
  </div>`).join('');
}

async function loadHospitals(){
  const data=await(await fetch('/api/hospital-selection')).json();
  const rankings=data.vikor?.rankings??[];
  document.getElementById('hospital-list').innerHTML=`<table style="width:100%;border-collapse:collapse;font-size:.82rem">
    <thead><tr>
      <th style="padding:8px 12px;color:var(--muted);font-size:.7rem;text-transform:uppercase;border-bottom:1px solid var(--border)">Rank</th>
      <th style="padding:8px 12px;color:var(--muted);font-size:.7rem;text-transform:uppercase;border-bottom:1px solid var(--border)">Hospital</th>
      <th style="padding:8px 12px;color:var(--muted);font-size:.7rem;text-transform:uppercase;border-bottom:1px solid var(--border)">Q Score</th>
    </tr></thead>
    <tbody>${rankings.map(r=>`<tr>
      <td style="padding:10px 12px;border-bottom:1px solid rgba(42,47,74,.5)"><span class="rank-num rank-${r.rank}">${r.rank}</span></td>
      <td style="padding:10px 12px;border-bottom:1px solid rgba(42,47,74,.5);font-weight:600">${r.hospital}</td>
      <td style="padding:10px 12px;border-bottom:1px solid rgba(42,47,74,.5);color:var(--accent)">${r.Q.toFixed(4)}</td>
    </tr>`).join('')}</tbody></table>`;
  renderAHPChart(data.ahp);
}

function renderAHPChart(ahp){
  if(!ahp)return;
  const labels=ahp.criteria,values=labels.map(c=>ahp.weights[c]*100);
  const ctx=document.getElementById('ahp-chart').getContext('2d');
  if(ahpChart)ahpChart.destroy();
  ahpChart=new Chart(ctx,{type:'bar',data:{labels,datasets:[{label:'Weight (%)',data:values,backgroundColor:'#4f8ef7',borderRadius:4}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#8892a4',font:{size:10}},grid:{color:'#2a2f4a'}},y:{ticks:{color:'#8892a4',font:{size:10}},grid:{color:'#2a2f4a'}}}}});
}

async function openModal(pid,name){
  document.getElementById('modal-title').textContent=`${name} (${pid})`;
  document.getElementById('patient-modal').classList.add('open');
  document.getElementById('modal-vitals-grid').innerHTML='<div class="loader"><div class="spinner"></div> Loading…</div>';
  const vitals=await(await fetch(`/api/patients/${pid}/vitals?limit=30`)).json();
  if(!vitals.length){document.getElementById('modal-vitals-grid').innerHTML='<div class="empty-state">No readings yet.</div>';return;}
  const latest=vitals[0];
  const defs=[{k:'heart_rate',l:'Heart Rate',u:'bpm',lo:60,hi:100},{k:'systolic_bp',l:'Systolic BP',u:'mmHg',lo:90,hi:120},
    {k:'diastolic_bp',l:'Diastolic BP',u:'mmHg',lo:60,hi:80},{k:'spo2',l:'SpO₂',u:'%',lo:95,hi:100},
    {k:'body_temperature',l:'Temperature',u:'°C',lo:36.1,hi:37.2},{k:'respiratory_rate',l:'Resp Rate',u:'br/min',lo:12,hi:20}];
  document.getElementById('modal-vitals-grid').innerHTML=defs.map(d=>{
    const v=latest[d.k];
    const cls=v==null?'':v<d.lo||v>d.hi?(Math.abs(v-d.lo)>5||Math.abs(v-d.hi)>5?'vital-bad':'vital-warn'):'vital-ok';
    return `<div class="vital-card"><div class="vname">${d.l}</div><div class="vval ${cls}">${v?.toFixed(1)??'—'}</div><div class="vunit">${d.u}</div></div>`;
  }).join('');
  const pts=vitals.slice(0,20).reverse();
  const labels=pts.map(v=>v.timestamp?.slice(11,19));
  const ctx=document.getElementById('modal-chart').getContext('2d');
  if(modalChart)modalChart.destroy();
  modalChart=new Chart(ctx,{type:'line',data:{labels,datasets:[
    {label:'HR (bpm)',data:pts.map(v=>v.heart_rate),borderColor:'#ef4444',tension:.4,fill:false,pointRadius:2},
    {label:'SpO₂ (%)',data:pts.map(v=>v.spo2),borderColor:'#4f8ef7',tension:.4,fill:false,pointRadius:2,yAxisID:'y2'}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8892a4',font:{size:11}}}},
      scales:{x:{ticks:{color:'#8892a4',font:{size:9},maxTicksLimit:8},grid:{color:'#2a2f4a'}},
        y:{ticks:{color:'#8892a4',font:{size:10}},grid:{color:'#2a2f4a'}},
        y2:{position:'right',min:80,max:102,ticks:{color:'#8892a4',font:{size:10}},grid:{drawOnChartArea:false}}}}});
  document.getElementById('modal-tbody').innerHTML=vitals.map(v=>`<tr>
    <td style="font-size:.75rem;color:var(--muted)">${v.timestamp?.replace('T',' ').replace('Z','')||'—'}</td>
    <td>${v.heart_rate?.toFixed(0)??'—'}</td><td>${v.systolic_bp?.toFixed(0)??'—'}</td>
    <td>${v.diastolic_bp?.toFixed(0)??'—'}</td><td>${v.spo2?.toFixed(1)??'—'}</td>
    <td>${v.body_temperature?.toFixed(1)??'—'}</td><td>${v.respiratory_rate?.toFixed(0)??'—'}</td>
    <td><span class="status-pill status-${v.health_status}">${HEALTH_STATUS[v.health_status]??'Normal'}</span></td>
  </tr>`).join('');
}

function closeModal(){document.getElementById('patient-modal').classList.remove('open');}
document.getElementById('patient-modal').addEventListener('click',function(e){if(e.target===this)closeModal();});

async function ackAlert(id){
  await fetch(`/api/alerts/${id}/acknowledge`,{method:'POST'});
  document.getElementById(`alert-${id}`)?.remove();
}

function switchTab(name){
  const names=['overview','patients','alerts','hospitals'];
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',names[i]===name));
  document.querySelectorAll('.tab-section').forEach(s=>s.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
  if(name==='alerts')loadAlerts();
  if(name==='hospitals')loadHospitals();
}

fetchDashboard();
setInterval(fetchDashboard,15000);
</script>
</body>
</html>"""


@app.route("/")
def dashboard():
    return render_template_string(
        DASHBOARD_HTML,
        patient_count=len(DEMO_PATIENTS),
        interval=SENSOR_SEND_INTERVAL,
    )


@app.route("/api/dashboard")
def api_dashboard():
    latest  = db_get_latest_vitals_all()
    alerts  = db_get_alerts(acknowledged=False, limit=50)
    patients = db_get_all_patients()
    normal  = sum(1 for v in latest if v.get("health_status") == 0)
    return jsonify({
        "patients":       patients,
        "latest_vitals":  latest,
        "unacked_alerts": alerts,
        "summary": {
            "total_patients":  len(patients),
            "normal_patients": normal,
            "critical_alerts": sum(1 for a in alerts if a.get("severity") == "critical"),
            "warning_alerts":  sum(1 for a in alerts if a.get("severity") == "warning"),
        },
        "timestamp": _now(),
    })


@app.route("/api/patients")
def api_patients():
    return jsonify(db_get_all_patients())


@app.route("/api/patients/<patient_id>/vitals")
def api_patient_vitals(patient_id):
    limit = min(int(request.args.get("limit", 50)), 200)
    return jsonify(db_get_vitals(patient_id, limit))


@app.route("/api/alerts")
def api_alerts():
    ack = request.args.get("acknowledged")
    if ack is None:
        return jsonify(db_get_alerts())
    return jsonify(db_get_alerts(acknowledged=ack.lower() in ("1","true","yes")))


@app.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def api_ack_alert(alert_id):
    db_acknowledge_alert(alert_id)
    return jsonify({"status": "acknowledged"})


@app.route("/api/hospital-selection")
def api_hospitals():
    cached = db_get_latest_hospital_ranking()
    if not cached:
        cached = run_ahp_vikor()
        db_save_hospital_ranking(cached)
    return jsonify(cached)


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json(silent=True) or {}
    code, label, proba = classify(data)
    return jsonify({"health_status": code, "health_label": label, "probabilities": proba})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": _now()})


# ─── Utility ──────────────────────────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Startup ──────────────────────────────────────────────────────────────────

def startup():
    init_db()
    get_model()   # train/load classifier
    t = threading.Thread(target=_simulator_loop, daemon=True)
    t.start()
    logger.info("Demo server ready — dashboard at http://0.0.0.0:%d", PORT)


startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
