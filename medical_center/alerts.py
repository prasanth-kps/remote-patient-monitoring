"""
Medical Center Server — Alert Generation System

Implements Step 5 of the paper's pseudocode:
  "If critical condition is detected → Notify healthcare providers and
   emergency services in real time."

Alert severity levels:
  critical  — immediate intervention required
  warning   — elevated risk, requires close monitoring
  info      — minor deviation, informational only
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    VITAL_RANGES, ALERT_EMAIL,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS,
)

logger = logging.getLogger(__name__)


# ── Threshold-based rule engine ──────────────────────────────────────────────

ALERT_RULES = [
    # (vital_key, operator, threshold, severity, message_template)
    ("heart_rate",        ">",  130,  "critical", "Tachycardia detected — HR={value:.0f} bpm"),
    ("heart_rate",        "<",  45,   "critical", "Severe Bradycardia — HR={value:.0f} bpm"),
    ("heart_rate",        ">",  100,  "warning",  "Elevated heart rate — HR={value:.0f} bpm"),
    ("heart_rate",        "<",  60,   "warning",  "Low heart rate — HR={value:.0f} bpm"),

    ("systolic_bp",       ">",  180,  "critical", "Hypertensive crisis — SBP={value:.0f} mmHg"),
    ("systolic_bp",       "<",  80,   "critical", "Hypotension — SBP={value:.0f} mmHg"),
    ("systolic_bp",       ">",  140,  "warning",  "Elevated systolic BP — SBP={value:.0f} mmHg"),
    ("systolic_bp",       "<",  90,   "warning",  "Low systolic BP — SBP={value:.0f} mmHg"),

    ("diastolic_bp",      ">",  120,  "critical", "Hypertensive crisis — DBP={value:.0f} mmHg"),
    ("diastolic_bp",      ">",  90,   "warning",  "Elevated diastolic BP — DBP={value:.0f} mmHg"),
    ("diastolic_bp",      "<",  60,   "warning",  "Low diastolic BP — DBP={value:.0f} mmHg"),

    ("spo2",              "<",  90,   "critical", "Severe hypoxemia — SpO2={value:.1f}%"),
    ("spo2",              "<",  94,   "warning",  "Low oxygen saturation — SpO2={value:.1f}%"),

    ("body_temperature",  ">",  39.5, "critical", "High fever — Temp={value:.1f}°C"),
    ("body_temperature",  "<",  35.0, "critical", "Hypothermia — Temp={value:.1f}°C"),
    ("body_temperature",  ">",  38.0, "warning",  "Fever — Temp={value:.1f}°C"),

    ("respiratory_rate",  ">",  30,   "critical", "Severe tachypnea — RR={value:.0f} breaths/min"),
    ("respiratory_rate",  "<",  8,    "critical", "Severe bradypnea — RR={value:.0f} breaths/min"),
    ("respiratory_rate",  ">",  20,   "warning",  "Elevated respiratory rate — RR={value:.0f} breaths/min"),
    ("respiratory_rate",  "<",  12,   "warning",  "Low respiratory rate — RR={value:.0f} breaths/min"),
]


def evaluate_vitals(vitals: dict) -> list[dict]:
    """
    Run all alert rules against a vitals reading.
    Returns a list of triggered alert dicts (may be empty).
    """
    triggered = []
    seen_keys: set = set()  # emit only the highest-severity alert per vital

    for vital_key, op, threshold, severity, template in ALERT_RULES:
        value = vitals.get(vital_key)
        if value is None:
            continue
        # Only fire the most severe alert per vital
        if vital_key in seen_keys:
            continue

        if (op == ">" and value > threshold) or (op == "<" and value < threshold):
            message = template.format(value=value)
            triggered.append({
                "vital_key": vital_key,
                "severity":  severity,
                "message":   message,
                "value":     value,
                "threshold": threshold,
            })
            if severity == "critical":
                seen_keys.add(vital_key)  # suppress lower-severity alerts for same vital

    return triggered


# ── Notification dispatch ────────────────────────────────────────────────────

def send_email_alert(patient_id: str, patient_name: str,
                     alerts: list[dict], vitals: dict) -> bool:
    """Send an email notification to the configured alert recipient."""
    if not ALERT_EMAIL or not SMTP_USER:
        logger.debug("Email alerting not configured — skipping.")
        return False

    critical = [a for a in alerts if a["severity"] == "critical"]
    subject = (
        f"[CRITICAL] RPM Alert — {patient_name} ({patient_id})"
        if critical
        else f"[WARNING] RPM Alert — {patient_name} ({patient_id})"
    )

    body_lines = [
        f"Remote Patient Monitoring — Alert Notification",
        f"",
        f"Patient: {patient_name} ({patient_id})",
        f"Time:    {vitals.get('timestamp', 'N/A')}",
        f"",
        f"Triggered Alerts:",
    ]
    for a in alerts:
        body_lines.append(f"  [{a['severity'].upper()}] {a['message']}")

    body_lines += [
        "",
        "Current Vitals:",
        f"  Heart Rate:    {vitals.get('heart_rate', 'N/A')} bpm",
        f"  BP:            {vitals.get('systolic_bp', 'N/A')}/{vitals.get('diastolic_bp', 'N/A')} mmHg",
        f"  SpO2:          {vitals.get('spo2', 'N/A')}%",
        f"  Temperature:   {vitals.get('body_temperature', 'N/A')}°C",
        f"  Resp Rate:     {vitals.get('respiratory_rate', 'N/A')} breaths/min",
        "",
        "Please review the patient dashboard immediately.",
    ]

    msg = MIMEMultipart()
    msg["From"]    = SMTP_USER
    msg["To"]      = ALERT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText("\n".join(body_lines), "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, ALERT_EMAIL, msg.as_string())
        logger.info("Email alert sent to %s for patient %s", ALERT_EMAIL, patient_id)
        return True
    except Exception as exc:
        logger.error("Failed to send email alert: %s", exc)
        return False
