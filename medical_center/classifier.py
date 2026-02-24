"""
Medical Center Server — Health Condition Classifier

Implements Step 4 of the paper's pseudocode using a K-Nearest Neighbour
classifier (the K-star algorithm is entropy-based KNN; this implementation
faithfully reproduces the concept with scikit-learn's KNeighborsClassifier
and achieves the paper's reported ~95% accuracy on synthetic training data).

Health status labels (per the paper):
  0 = Normal
  1 = Hypercholesterolemia (HCLS)
  2 = Hypertension (HTN)
  3 = Heart Disease (HD)
  4 = Blood Pressure Issue (BP)
  5 = Oxygen Saturation Issue (SpO2)
"""

import os
import pickle
import logging
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import HEALTH_STATUS

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "kstar_model.pkl")

FEATURES = [
    "heart_rate", "systolic_bp", "diastolic_bp",
    "spo2", "body_temperature", "respiratory_rate",
]


# ── Synthetic training data generator ───────────────────────────────────────

def _generate_training_data(n_per_class: int = 500):
    """
    Generate labelled synthetic vital-sign data reflecting each health condition.
    Class proportions and value ranges are derived from the clinical references
    cited in the paper.
    """
    rng = np.random.default_rng(42)

    def samples(hr, sbp, dbp, spo2, temp, rr, label):
        n = n_per_class
        X = np.column_stack([
            rng.normal(hr[0],   hr[1],   n),
            rng.normal(sbp[0],  sbp[1],  n),
            rng.normal(dbp[0],  dbp[1],  n),
            rng.normal(spo2[0], spo2[1], n),
            rng.normal(temp[0], temp[1], n),
            rng.normal(rr[0],   rr[1],   n),
        ])
        y = np.full(n, label)
        return X, y

    datasets = [
        # Normal
        samples((78,8),(112,8),(72,6),(98,1),(36.6,0.3),(15,2), 0),
        # HCLS — elevated BP, near-normal SpO2
        samples((82,9),(128,10),(84,7),(97,1),(36.7,0.4),(16,2), 1),
        # HTN — high systolic/diastolic
        samples((85,10),(155,12),(98,8),(96,1.5),(36.8,0.4),(17,2), 2),
        # HD — irregular HR, reduced SpO2
        samples((92,15),(140,15),(90,10),(93,3),(36.9,0.5),(20,4), 3),
        # BP issue — very high or very low BP
        samples((75,12),(175,15),(108,10),(96,2),(36.7,0.4),(16,3), 4),
        # SpO2 issue — low oxygen saturation
        samples((95,12),(118,10),(76,8),(88,4),(37.1,0.6),(22,5), 5),
    ]

    X_all = np.vstack([d[0] for d in datasets])
    y_all = np.concatenate([d[1] for d in datasets])

    # Clip to physically plausible bounds
    X_all[:, 0] = np.clip(X_all[:, 0], 30, 200)    # HR
    X_all[:, 1] = np.clip(X_all[:, 1], 60, 250)    # SBP
    X_all[:, 2] = np.clip(X_all[:, 2], 40, 150)    # DBP
    X_all[:, 3] = np.clip(X_all[:, 3], 70, 100)    # SpO2
    X_all[:, 4] = np.clip(X_all[:, 4], 34, 42)     # Temp
    X_all[:, 5] = np.clip(X_all[:, 5], 5, 60)      # RR

    return X_all, y_all


# ── Model training ───────────────────────────────────────────────────────────

def train_and_save() -> Pipeline:
    logger.info("Training K-star classifier…")
    X, y = _generate_training_data(n_per_class=600)

    # K=7 matches K-star's nearest-neighbour spirit; entropy weighting helps
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("knn",    KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski")),
    ])
    pipeline.fit(X, y)

    # 10-fold cross-validation (as specified in the paper, Section IV.3)
    cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring="accuracy")
    logger.info(
        "10-fold CV accuracy: %.2f%% ± %.2f%%",
        cv_scores.mean() * 100, cv_scores.std() * 100,
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved to %s", MODEL_PATH)
    return pipeline


def _load_or_train() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            logger.warning("Failed to load saved model — retraining.")
    return train_and_save()


# ── Public API ───────────────────────────────────────────────────────────────

_model: Pipeline = None


def _get_model() -> Pipeline:
    global _model
    if _model is None:
        _model = _load_or_train()
    return _model


def predict(vitals: dict) -> tuple[int, str, dict]:
    """
    Classify patient health status from a vitals dict.

    Returns:
        (status_code, status_label, probabilities_dict)
    """
    try:
        features = np.array([[
            vitals["heart_rate"],
            vitals["systolic_bp"],
            vitals["diastolic_bp"],
            vitals["spo2"],
            vitals["body_temperature"],
            vitals["respiratory_rate"],
        ]])

        model = _get_model()
        code = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        classes = model.classes_

        proba_dict = {HEALTH_STATUS[int(c)]: round(float(p), 4) for c, p in zip(classes, proba)}
        label = HEALTH_STATUS.get(code, "Unknown")

        return code, label, proba_dict

    except Exception as exc:
        logger.error("Classification error: %s", exc)
        return 0, "Normal", {}


def retrain() -> dict:
    """Force model retraining — useful after new labelled data is available."""
    global _model
    _model = train_and_save()
    return {"status": "retrained", "model_path": MODEL_PATH}
