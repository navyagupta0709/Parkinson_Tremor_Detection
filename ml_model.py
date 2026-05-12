"""
ml_model.py
===========
Machine Learning prediction for Parkinson tremor severity.
- Classes: 0 = Normal, 1 = Mild Tremor, 2 = Severe Tremor
- Features match notebook pipeline (7 features)
- Trains a RandomForest + StandardScaler pipeline
- Saves model.pkl and scaler.pkl
- Loads from disk for inference
"""

import os
import numpy as np
import joblib
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SCALER_PATH= BASE_DIR / "models" / "scaler.pkl"

# ── Class Labels ──────────────────────────────────────────────────────────────
CLASS_NAMES  = {0: "Normal", 1: "Mild Tremor", 2: "Severe Tremor"}
CLASS_COLORS = {0: "#00e676", 1: "#ffab40", 2: "#ff1744"}

# ── Feature Names (must match extract_features keys used below) ───────────────
FEATURE_NAMES = [
    "mean", "std", "rms", "amplitude",
    "peak_count", "dom_freq_hz", "band_power",
]


# ── Synthetic Training Data Generator ────────────────────────────────────────
def _generate_training_data(n_per_class: int = 500, seed: int = 42):
    """
    Generate realistic synthetic training data grounded in known
    Parkinson tremor characteristics:
      Normal      : low amplitude, low band power, freq 0.5–2 Hz
      Mild Tremor : moderate amplitude, moderate band power, 3–5 Hz
      Severe      : high amplitude, high band power, 5–7 Hz
    """
    rng = np.random.default_rng(seed)

    def sample_class(cls, n):
        if cls == 0:   # Normal
            mean      = rng.uniform(1.8, 2.5, n)
            std       = rng.uniform(0.02, 0.08, n)
            rms       = std * 1.05 + mean
            amplitude = rng.uniform(0.05, 0.25, n)
            peaks     = rng.integers(0, 4, n).astype(float)
            freq      = rng.uniform(0.5, 2.0, n)
            band      = rng.uniform(0.02, 0.15, n)
        elif cls == 1: # Mild Tremor
            mean      = rng.uniform(1.5, 3.0, n)
            std       = rng.uniform(0.10, 0.30, n)
            rms       = std * 1.1  + mean
            amplitude = rng.uniform(0.25, 0.80, n)
            peaks     = rng.integers(4, 10, n).astype(float)
            freq      = rng.uniform(3.0, 5.0, n)
            band      = rng.uniform(0.25, 0.55, n)
        else:          # Severe Tremor
            mean      = rng.uniform(1.0, 3.5, n)
            std       = rng.uniform(0.35, 0.80, n)
            rms       = std * 1.15 + mean
            amplitude = rng.uniform(0.80, 2.50, n)
            peaks     = rng.integers(10, 20, n).astype(float)
            freq      = rng.uniform(5.0, 7.0, n)
            band      = rng.uniform(0.55, 0.95, n)

        X = np.column_stack([mean, std, rms, amplitude, peaks, freq, band])
        y = np.full(n, cls, dtype=int)
        return X, y

    parts = [sample_class(c, n_per_class) for c in [0, 1, 2]]
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── Train & Save ──────────────────────────────────────────────────────────────
def train_and_save(n_per_class: int = 500):
    """Train model on synthetic data and persist to disk."""
    logger.info("Training Parkinson tremor ML model…")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    X, y = _generate_training_data(n_per_class)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model,  MODEL_PATH)
    logger.info(f"Model saved → {MODEL_PATH}")
    logger.info(f"Scaler saved → {SCALER_PATH}")
    return model, scaler


# ── Load ──────────────────────────────────────────────────────────────────────
def load_model():
    """Load model and scaler from disk. Train first if missing."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        logger.warning("model.pkl / scaler.pkl not found — training now…")
        return train_and_save()
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded from disk.")
    return model, scaler


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(features: dict, model, scaler) -> dict:
    """
    Run inference on a feature dict (from signal_processing.extract_features).
    Returns: { class_id, label, confidence, severity_pct, probabilities }
    """
    x = np.array([[features.get(k, 0.0) for k in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)

    proba   = model.predict_proba(x_scaled)[0]
    cls_id  = int(np.argmax(proba))
    conf    = float(proba[cls_id])

    # Severity percentage: weighted sum (mild=50, severe=100)
    severity_pct = float(proba[1] * 50.0 + proba[2] * 100.0)

    return {
        "class_id":     cls_id,
        "label":        CLASS_NAMES[cls_id],
        "color":        CLASS_COLORS[cls_id],
        "confidence":   conf,
        "severity_pct": severity_pct,
        "probabilities": {
            "Normal":        float(proba[0]),
            "Mild Tremor":   float(proba[1]),
            "Severe Tremor": float(proba[2]),
        },
    }
