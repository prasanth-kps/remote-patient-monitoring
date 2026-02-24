import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "rpm_database.db")

GATEWAY_HOST = os.getenv("GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", 5001))

MEDICAL_CENTER_HOST = os.getenv("MEDICAL_CENTER_HOST", "127.0.0.1")
MEDICAL_CENTER_PORT = int(os.getenv("MEDICAL_CENTER_PORT", 5002))

SENSOR_SEND_INTERVAL = int(os.getenv("SENSOR_SEND_INTERVAL", 30))  # seconds
NUM_SIMULATED_PATIENTS = int(os.getenv("NUM_SIMULATED_PATIENTS", 5))

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

# Normal ranges for vital signs
VITAL_RANGES = {
    "heart_rate":       {"min": 60,  "max": 100,  "unit": "bpm"},
    "systolic_bp":      {"min": 90,  "max": 120,  "unit": "mmHg"},
    "diastolic_bp":     {"min": 60,  "max": 80,   "unit": "mmHg"},
    "spo2":             {"min": 95,  "max": 100,  "unit": "%"},
    "body_temperature": {"min": 36.1,"max": 37.2, "unit": "°C"},
    "respiratory_rate": {"min": 12,  "max": 20,   "unit": "breaths/min"},
}

# Health status labels (as per paper)
HEALTH_STATUS = {
    0: "Normal",
    1: "Hypercholesterolemia (HCLS)",
    2: "Hypertension (HTN)",
    3: "Heart Disease (HD)",
    4: "Blood Pressure Issue (BP)",
    5: "Oxygen Saturation Issue (SpO2)",
}

SECRET_KEY = os.getenv("SECRET_KEY", "rpm-secret-key-change-in-production")
