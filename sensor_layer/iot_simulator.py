"""
Sensor Layer (Tier 1) - IoT Device Simulator

Simulates wearable/implantable IoT sensors for multiple patients.
Measures: Heart Rate, Blood Pressure, SpO2, Body Temperature, Respiratory Rate.
Transmits data to the Gateway Layer every SENSOR_SEND_INTERVAL seconds.
"""

import time
import random
import json
import threading
import requests
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GATEWAY_HOST, GATEWAY_PORT, SENSOR_SEND_INTERVAL,
    NUM_SIMULATED_PATIENTS, VITAL_RANGES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SENSOR] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GATEWAY_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}/api/ingest"

PATIENTS = [
    {"patient_id": f"P{i+1:03d}", "name": f"Patient {i+1}", "age": random.randint(30, 80),
     "gender": random.choice(["M", "F"]), "condition": random.choice(
         ["Hypertension", "Diabetes", "Healthy", "Heart Disease", "COPD"]
     )}
    for i in range(NUM_SIMULATED_PATIENTS)
]


def _add_noise(value: float, noise_pct: float = 0.03) -> float:
    """Add small Gaussian noise to simulate sensor variability."""
    noise = value * noise_pct * random.gauss(0, 1)
    return round(value + noise, 2)


def generate_vitals(patient: dict) -> dict:
    """
    Generate realistic vital signs for a patient.
    Patients with chronic conditions may show occasional abnormal readings.
    """
    condition = patient.get("condition", "Healthy")

    # Base values — slightly adjusted per condition
    if condition == "Hypertension":
        hr      = _add_noise(random.uniform(70, 110))
        sys_bp  = _add_noise(random.uniform(130, 180))
        dia_bp  = _add_noise(random.uniform(85, 110))
        spo2    = _add_noise(random.uniform(94, 99))
        temp    = _add_noise(random.uniform(36.0, 37.5))
        rr      = _add_noise(random.uniform(14, 22))
    elif condition == "Heart Disease":
        hr      = _add_noise(random.uniform(55, 115))
        sys_bp  = _add_noise(random.uniform(100, 160))
        dia_bp  = _add_noise(random.uniform(65, 100))
        spo2    = _add_noise(random.uniform(90, 98))
        temp    = _add_noise(random.uniform(36.0, 37.8))
        rr      = _add_noise(random.uniform(14, 24))
    elif condition == "COPD":
        hr      = _add_noise(random.uniform(70, 105))
        sys_bp  = _add_noise(random.uniform(100, 140))
        dia_bp  = _add_noise(random.uniform(65, 90))
        spo2    = _add_noise(random.uniform(88, 95))
        temp    = _add_noise(random.uniform(36.0, 38.0))
        rr      = _add_noise(random.uniform(18, 28))
    elif condition == "Diabetes":
        hr      = _add_noise(random.uniform(65, 100))
        sys_bp  = _add_noise(random.uniform(110, 145))
        dia_bp  = _add_noise(random.uniform(70, 95))
        spo2    = _add_noise(random.uniform(93, 99))
        temp    = _add_noise(random.uniform(36.0, 37.5))
        rr      = _add_noise(random.uniform(13, 20))
    else:  # Healthy
        hr      = _add_noise(random.uniform(60, 100))
        sys_bp  = _add_noise(random.uniform(90, 120))
        dia_bp  = _add_noise(random.uniform(60, 80))
        spo2    = _add_noise(random.uniform(96, 100))
        temp    = _add_noise(random.uniform(36.1, 37.2))
        rr      = _add_noise(random.uniform(12, 18))

    return {
        "patient_id":       patient["patient_id"],
        "patient_name":     patient["name"],
        "age":              patient["age"],
        "gender":           patient["gender"],
        "condition":        condition,
        "heart_rate":       round(max(30, hr), 1),
        "systolic_bp":      round(max(60, sys_bp), 1),
        "diastolic_bp":     round(max(40, dia_bp), 1),
        "spo2":             round(min(100, max(70, spo2)), 1),
        "body_temperature": round(temp, 2),
        "respiratory_rate": round(max(8, rr), 1),
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_id":        f"DEV-{patient['patient_id']}",
        "location":         random.choice(["Home", "Ward A", "Ward B", "ICU"]),
    }


def send_to_gateway(vitals: dict) -> bool:
    """Transmit vital signs payload to the Gateway Layer."""
    try:
        resp = requests.post(
            GATEWAY_URL,
            json=vitals,
            timeout=10,
            headers={"Content-Type": "application/json", "X-Device-Id": vitals["device_id"]},
        )
        resp.raise_for_status()
        logger.info(
            "Sent vitals for %s (HR=%.0f bpm, SpO2=%.1f%%, BP=%s/%s mmHg)",
            vitals["patient_id"],
            vitals["heart_rate"],
            vitals["spo2"],
            vitals["systolic_bp"],
            vitals["diastolic_bp"],
        )
        return True
    except requests.exceptions.ConnectionError:
        logger.warning("Gateway unreachable — will retry next cycle.")
    except requests.exceptions.HTTPError as exc:
        logger.error("Gateway HTTP error: %s", exc)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
    return False


def patient_sensor_loop(patient: dict) -> None:
    """Continuous sensing loop for a single patient (runs in its own thread)."""
    # Stagger start times to avoid thundering-herd on the gateway
    time.sleep(random.uniform(0, 5))
    logger.info("Started sensor thread for %s (%s)", patient["patient_id"], patient["condition"])
    while True:
        vitals = generate_vitals(patient)
        send_to_gateway(vitals)
        time.sleep(SENSOR_SEND_INTERVAL)


def run_simulator() -> None:
    """Launch one sensor thread per simulated patient."""
    logger.info("IoT Simulator starting — %d patients, interval=%ds", len(PATIENTS), SENSOR_SEND_INTERVAL)
    threads = []
    for patient in PATIENTS:
        t = threading.Thread(target=patient_sensor_loop, args=(patient,), daemon=True)
        t.start()
        threads.append(t)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Simulator stopped.")


if __name__ == "__main__":
    run_simulator()
