# Distributed and Scalable Remote Patient Monitoring System

A cloud-based Remote Patient Monitoring (RPM) system built around the three-tier architecture described in:

> *A Distributed and Scalable System for Remote Patient Monitoring using Cloud Based Architecture*
> IJIRCCE, Volume 12, Issue 8, August 2024. DOI: 10.15680/IJIRCCE.2024.1208067

---

## Live Demo

'https://remote-patient-montioring.onrender.com/'

The demo runs a built-in IoT simulator — **no hardware, no extra config needed.** Six patients with different conditions (Hypertension, Heart Disease, COPD, Diabetes, Healthy) stream live vitals every 15 seconds. Critical events trigger automatic alerts visible on the dashboard.

> **Note:** Render's free tier spins down after 15 minutes of inactivity. The first request after a spin-down may take ~30 seconds to respond.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Tier 1 — Sensor Layer                                  │
│  IoT devices / wearables → iot_simulator.py             │
│  Measures: HR, BP (Sys/Dia), SpO₂, Temperature, RR      │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP POST every 30 s
┌──────────────────────▼──────────────────────────────────┐
│  Tier 2 — Gateway Layer         (port 5001)             │
│  Data aggregation · Normalisation · Pre-processing      │
│  mHealth REST API · Forwards to Medical Center          │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP POST (pre-processed payload)
┌──────────────────────▼──────────────────────────────────┐
│  Tier 3 — Medical Center Server (port 5002)             │
│  K-star classifier · AHP-VIKOR hospital ranking         │
│  Alert engine · SQLite persistence · Web dashboard      │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option A — All-in-one demo server (simplest)

Runs the gateway, medical center, and IoT simulator in **one process**:

```bash
pip install -r requirements.txt
python demo_server.py
```

Dashboard → http://localhost:5002 — patients start streaming in ~15 seconds.

### Option B — Full three-service setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Medical Center Server

```bash
python medical_center/server.py
```

Dashboard → http://localhost:5002

### 3. Start the Gateway

```bash
python gateway_layer/gateway_server.py
```

### 4. Start the IoT Simulator

```bash
python sensor_layer/iot_simulator.py
```

Readings will appear in the dashboard within 30 seconds.

---

### Option C — Docker Compose (all three services)

```bash
docker-compose up --build
```

| Service         | URL                    |
|-----------------|------------------------|
| Dashboard       | http://localhost:5002  |
| Gateway API     | http://localhost:5001  |

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

Key settings:

| Variable                 | Default       | Description                      |
|--------------------------|---------------|----------------------------------|
| `SENSOR_SEND_INTERVAL`   | `30`          | Seconds between sensor readings  |
| `NUM_SIMULATED_PATIENTS` | `5`           | Number of simulated patients     |
| `ALERT_EMAIL`            | —             | Email to send critical alerts to |

---

## Project Structure

```
remote-patient-monitoring/
├── sensor_layer/
│   └── iot_simulator.py          # Tier 1: IoT device simulation
├── gateway_layer/
│   ├── gateway_server.py         # Tier 2: data aggregation + forwarding
│   └── preprocessor.py           # Normalisation & validation
├── medical_center/
│   ├── server.py                 # Tier 3: main Flask application
│   ├── database.py               # SQLite persistence layer
│   ├── classifier.py             # K-star (KNN) health classifier
│   ├── ahp_vikor.py              # AHP-VIKOR hospital selection
│   ├── alerts.py                 # Real-time alert engine
│   └── templates/
│       └── dashboard.html        # Web dashboard
├── config.py                     # Centralised configuration
├── requirements.txt
├── docker-compose.yml
└── .env.example
```

---

## REST API Reference

### Medical Center Server (port 5002)

| Method | Endpoint                                   | Description                       |
|--------|--------------------------------------------|-----------------------------------|
| GET    | `/`                                        | Web dashboard                     |
| POST   | `/api/vitals`                              | Ingest readings from gateway      |
| GET    | `/api/patients`                            | List all patients                 |
| GET    | `/api/patients/<id>/vitals`               | Patient vital-sign history        |
| GET    | `/api/dashboard`                           | Dashboard summary data (JSON)     |
| GET    | `/api/alerts`                              | List alerts                       |
| POST   | `/api/alerts/<id>/acknowledge`             | Acknowledge an alert              |
| GET    | `/api/hospital-selection`                  | Run/retrieve AHP-VIKOR ranking    |
| POST   | `/api/classify`                            | Classify vitals (no persistence)  |

### Gateway Server (port 5001)

| Method | Endpoint           | Description                        |
|--------|--------------------|------------------------------------|
| POST   | `/api/ingest`      | Receive raw IoT readings           |
| GET    | `/api/recent`      | Recent readings buffer             |
| GET    | `/api/patients`    | Patients seen by this gateway      |

---

## Methodology

### Phase 1 — Service Identification (Decision Matrix)

Each healthcare service is scored on **Feasibility (F)**, **Relevance (R)**, and **Impact (I)**:

```
P = wF·F + wR·R + wI·I
```

### Phase 2 — Hospital Selection (AHP-VIKOR)

**AHP** derives criterion weights via pairwise comparison matrices (CR < 0.10 check included).  
**VIKOR** ranks hospitals by computing utility (S), regret (R), and compromise score (Q).

### Phase 3 — Validation

Statistical validation uses mean, standard deviation, and the final validation metric:

```
Fv = (Sum(x) × 100) / 500
```

### Health Classification (K-star)

Six health status classes (per the paper):

| Code | Label                          |
|------|--------------------------------|
| 0    | Normal                         |
| 1    | Hypercholesterolemia (HCLS)    |
| 2    | Hypertension (HTN)             |
| 3    | Heart Disease (HD)             |
| 4    | Blood Pressure Issue (BP)      |
| 5    | Oxygen Saturation Issue (SpO₂) |

10-fold cross-validation is performed automatically on first run.
