"""
Medical Center Server — Database Layer

SQLite-backed persistence for:
  - patients              : demographic records
  - vital_signs           : time-series health readings
  - alerts                : generated alert records
  - hospital_rankings     : AHP-VIKOR results
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATABASE_PATH

logger = logging.getLogger(__name__)

# ── DDL ─────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    name         TEXT,
    age          INTEGER,
    gender       TEXT,
    condition    TEXT,
    registered_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS vital_signs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id       TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    heart_rate       REAL,
    systolic_bp      REAL,
    diastolic_bp     REAL,
    spo2             REAL,
    body_temperature REAL,
    respiratory_rate REAL,
    map              REAL,
    pulse_pressure   REAL,
    health_status    INTEGER DEFAULT 0,
    health_label     TEXT DEFAULT 'Normal',
    device_id        TEXT,
    location         TEXT,
    raw_json         TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    TEXT NOT NULL,
    vital_sign_id INTEGER,
    alert_type    TEXT,
    severity      TEXT,
    message       TEXT,
    acknowledged  INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS hospital_rankings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    results_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_vitals_patient ON vital_signs(patient_id);
CREATE INDEX IF NOT EXISTS idx_vitals_ts      ON vital_signs(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_patient ON alerts(patient_id);
"""


# ── Connection helper ────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(DDL)
    logger.info("Database initialised at %s", DATABASE_PATH)


# ── Patient operations ───────────────────────────────────────────────────────

def upsert_patient(patient_id: str, name: str, age: int, gender: str, condition: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO patients (patient_id, name, age, gender, condition)
               VALUES (?,?,?,?,?)
               ON CONFLICT(patient_id) DO UPDATE SET
                   name=excluded.name, age=excluded.age,
                   gender=excluded.gender, condition=excluded.condition""",
            (patient_id, name, age, gender, condition),
        )


def get_all_patients() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM patients ORDER BY patient_id").fetchall()
    return [dict(r) for r in rows]


def get_patient(patient_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
    return dict(row) if row else None


# ── Vital signs operations ───────────────────────────────────────────────────

def insert_vital(data: dict, health_status: int, health_label: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO vital_signs
               (patient_id, timestamp, heart_rate, systolic_bp, diastolic_bp,
                spo2, body_temperature, respiratory_rate, map, pulse_pressure,
                health_status, health_label, device_id, location, raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data["patient_id"],
                data.get("timestamp", _now()),
                data.get("heart_rate"),
                data.get("systolic_bp"),
                data.get("diastolic_bp"),
                data.get("spo2"),
                data.get("body_temperature"),
                data.get("respiratory_rate"),
                data.get("map"),
                data.get("pulse_pressure"),
                health_status,
                health_label,
                data.get("device_id", ""),
                data.get("location", ""),
                json.dumps(data),
            ),
        )
        return cur.lastrowid


def get_vitals(patient_id: str, limit: int = 50) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, patient_id, timestamp, heart_rate, systolic_bp, diastolic_bp,
                      spo2, body_temperature, respiratory_rate, map, pulse_pressure,
                      health_status, health_label, location
               FROM vital_signs WHERE patient_id=?
               ORDER BY timestamp DESC LIMIT ?""",
            (patient_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_latest_vitals_all(limit: int = 100) -> list:
    """Latest reading per patient, plus recent history up to `limit` total rows."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT v.* FROM vital_signs v
               INNER JOIN (
                   SELECT patient_id, MAX(timestamp) AS max_ts
                   FROM vital_signs GROUP BY patient_id
               ) latest ON v.patient_id=latest.patient_id AND v.timestamp=latest.max_ts
               ORDER BY v.timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_vitals_stats(patient_id: str) -> dict:
    """Mean + std-dev for each vital — used in Phase 3 validation."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT
                 AVG(heart_rate)         AS hr_mean,
                 AVG(systolic_bp)        AS sbp_mean,
                 AVG(diastolic_bp)       AS dbp_mean,
                 AVG(spo2)               AS spo2_mean,
                 AVG(body_temperature)   AS temp_mean,
                 AVG(respiratory_rate)   AS rr_mean,
                 COUNT(*)                AS n
               FROM vital_signs WHERE patient_id=?""",
            (patient_id,),
        ).fetchone()
    return dict(row) if row else {}


# ── Alert operations ─────────────────────────────────────────────────────────

def insert_alert(patient_id: str, vital_sign_id: int, alert_type: str,
                 severity: str, message: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO alerts (patient_id, vital_sign_id, alert_type, severity, message)
               VALUES (?,?,?,?,?)""",
            (patient_id, vital_sign_id, alert_type, severity, message),
        )
        return cur.lastrowid


def get_alerts(acknowledged: Optional[bool] = None, limit: int = 50) -> list:
    with get_conn() as conn:
        if acknowledged is None:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE acknowledged=? ORDER BY created_at DESC LIMIT ?",
                (int(acknowledged), limit),
            ).fetchall()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))


# ── Hospital ranking operations ──────────────────────────────────────────────

def save_hospital_ranking(results: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO hospital_rankings (results_json) VALUES (?)",
            (json.dumps(results),),
        )


def get_latest_hospital_ranking() -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT results_json FROM hospital_rankings ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
    return json.loads(row["results_json"]) if row else None


# ── Utility ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
