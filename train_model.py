"""
train_model.py
==============
Trains a Binary Random Forest classifier:
    0 = Non-Tremor  (dominant frequency < 3 Hz)
    1 = Tremor      (dominant frequency 3 – 7 Hz, Parkinson's band)

Synthetic training data reproduces the READINGS.xlsx distribution:
    7 frequency classes  ×  3 signals  ×  windowed windows

Saves:
    model_binary_rf.pkl   ← used by app.py (primary live model)

Run standalone:
    python train_model.py
"""

import os
import numpy as np
import joblib

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, f1_score,
                                     precision_score, recall_score,
                                     roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split

from signal_processing import (
    FS, WINDOW_SIZE, FEAT_COLS, TREMOR_LO,
    process_signal, extract_features, features_to_vec,
)

MODEL_PATH = "models/model_binary_rf.pkl"


# ──────────────────────────────────────────────────────────────
# Synthetic training data
# ──────────────────────────────────────────────────────────────
def generate_data(n_per_class: int = 500) -> tuple:
    """
    Synthetic windows matching READINGS.xlsx distribution.
    7 frequency classes → binary labels:
        ≥ 3 Hz → Tremor (1)
        < 3 Hz → Non-Tremor (0)
    """
    t = np.linspace(0, WINDOW_SIZE / FS, WINDOW_SIZE)

    # (freq_hz, amplitude_V, binary_label)
    configs = [
        (1.0, 0.08, 0),   # 1 Hz  — normal voluntary
        (2.0, 0.15, 0),   # 2 Hz  — normal
        (3.0, 0.45, 1),   # 3 Hz  — mild Parkinson tremor
        (4.0, 0.75, 1),   # 4 Hz  — Parkinson
        (5.0, 1.10, 1),   # 5 Hz  — Parkinson
        (6.0, 1.50, 1),   # 6 Hz  — severe
        (7.0, 1.80, 1),   # 7 Hz  — severe
    ]

    X, y = [], []
    for freq, amp, label in configs:
        for _ in range(n_per_class):
            noise = np.random.normal(0, amp * 0.15, WINDOW_SIZE)
            sig   = amp * np.sin(2 * np.pi * freq * t) + noise
            proc  = process_signal(sig)
            feat  = extract_features(proc)
            X.append(features_to_vec(feat))
            y.append(label)

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
def train_and_save(force: bool = False) -> Pipeline:
    """
    Train RF binary classifier. Skip if model already exists.
    Set force=True to retrain.
    """
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH) and not force:
        print(f"Model already exists at {MODEL_PATH}. Skipping training.")
        print("Use train_and_save(force=True) to retrain.")
        return joblib.load(MODEL_PATH)

    print("Generating synthetic training data…")
    X, y = generate_data(500)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Class balance — Non-Tremor: {(y==0).sum()}  Tremor: {(y==1).sum()}")

    # ── Random Forest (primary model) ─────────────────────────
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred) * 100
    f1   = f1_score(y_test, y_pred) * 100
    auc  = roc_auc_score(y_test, y_prob) * 100

    print()
    print("=" * 50)
    print("  BINARY RF  —  Test Results")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.1f}%")
    print(f"  F1 Score  : {f1:.1f}%")
    print(f"  ROC-AUC   : {auc:.1f}%")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Non-Tremor (0)", "Tremor (1)"],
        digits=4,
    ))
    print("=" * 50)

    joblib.dump(rf, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    return rf


def load_model() -> Pipeline:
    """Load or auto-train the primary binary RF model."""
    if not os.path.exists(MODEL_PATH):
        print("model_binary_rf.pkl not found — training now…")
        return train_and_save()
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train_and_save(force=True)
