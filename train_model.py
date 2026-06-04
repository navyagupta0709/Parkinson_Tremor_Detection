"""
train_model.py
Trains exactly the models from notebook Cell 17:
  - KNN (k=5, euclidean)
  - SVM (RBF, C=1.0, gamma=scale)
  - RF  (100 trees, max_depth=6)

Binary task  (label2): Non-Tremor vs Tremor  ← used in live dashboard
7-class task (label7): 1Hz … 7Hz             ← used for offline research

Saves:
  model_binary_rf.pkl   ← primary live model (most reliable)
  model_binary_svm.pkl
  model_binary_knn.pkl
  model_7class_rf.pkl
"""

import os
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from signal_processing import (
    FS, WINDOW_SIZE, FEAT_COLS,
    process_signal, extract_features, features_to_vec
)

TREMOR_LO = 3.0
TREMOR_HI = 7.0


def generate_training_data(n_per_class: int = 400):
    """
    Synthetic data matching READINGS.xlsx distribution.
    7 frequency classes × 3 signals × windowed = ~420 windows per class.

    Binary labels:
      0 = Non-Tremor  (1–2 Hz, low amplitude)
      1 = Tremor      (3–7 Hz, higher amplitude)

    7-class labels: 0..6 → 1Hz..7Hz (same as notebook label7)
    """
    t = np.linspace(0, WINDOW_SIZE / FS, WINDOW_SIZE)

    # (label7, freq_hz, amplitude_V, n_samples)
    configs = [
        (0, 1.0, 0.08, n_per_class),  # 1 Hz — Non-Tremor
        (1, 2.0, 0.15, n_per_class),  # 2 Hz — Non-Tremor
        (2, 3.0, 0.45, n_per_class),  # 3 Hz — Tremor (Mild)
        (3, 4.0, 0.75, n_per_class),  # 4 Hz — Tremor
        (4, 5.0, 1.10, n_per_class),  # 5 Hz — Tremor
        (5, 6.0, 1.50, n_per_class),  # 6 Hz — Tremor (Severe)
        (6, 7.0, 1.80, n_per_class),  # 7 Hz — Tremor (Severe)
    ]

    X7, y7, y2 = [], [], []

    for lbl7, freq, amp, n in configs:
        for _ in range(n):
            noise = np.random.normal(0, amp * 0.15, WINDOW_SIZE)
            sig   = amp * np.sin(2 * np.pi * freq * t) + noise
            proc  = process_signal(sig)
            feat  = extract_features(proc)
            vec   = features_to_vec(feat)

            X7.append(vec)
            y7.append(lbl7)
            y2.append(1 if freq >= TREMOR_LO else 0)

    return np.array(X7), np.array(y7), np.array(y2)


def build_pipelines():
    return {
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, metric='euclidean')),
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                        probability=True, random_state=42)),
        ]),
        'RF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=6,
                min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ]),
    }


def train_and_save(force: bool = False):
    """Train all models. Skip if .pkl files already exist unless force=True."""
    models_needed = {
        'model_binary_rf.pkl':  None,
        'model_binary_svm.pkl': None,
        'model_binary_knn.pkl': None,
        'model_7class_rf.pkl':  None,
    }

    all_exist = all(os.path.exists(p) for p in models_needed)
    if all_exist and not force:
        print("All model files found — skipping training.")
        return

    print("Generating synthetic training data (matches READINGS.xlsx distribution)…")
    X, y7, y2 = generate_training_data(400)
    print(f"  {len(X)} windows — Binary: {np.bincount(y2)} | 7-class: {np.bincount(y7)}")

    pipes = build_pipelines()

    # ── Binary models ─────────────────────────────────────────
    for name, pipe in pipes.items():
        pipe.fit(X, y2)
        fname = f'model_binary_{name.lower()}.pkl'
        joblib.dump(pipe, fname)
        preds = pipe.predict(X)
        acc   = (preds == y2).mean() * 100
        print(f"  Binary {name:4s}: train acc {acc:.1f}%  → saved {fname}")

    # ── 7-class RF ────────────────────────────────────────────
    rf7 = build_pipelines()['RF']
    rf7.fit(X, y7)
    joblib.dump(rf7, 'model_7class_rf.pkl')
    acc7 = (rf7.predict(X) == y7).mean() * 100
    print(f"  7-class RF  : train acc {acc7:.1f}%  → saved model_7class_rf.pkl")
    print("Training complete.")


def load_primary_model():
    """Load best model for live dashboard (RF binary classifier)."""
    path = 'model_binary_rf.pkl'
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    # auto-train if missing
    train_and_save()
    return joblib.load(path)


if __name__ == '__main__':
    train_and_save(force=True)
