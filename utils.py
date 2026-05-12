"""
utils.py — Parkinson's IoT Wearable Monitoring System
Handles: sensor simulation, anomaly detection, FFT analysis, CSV logging
"""

import numpy as np
import pandas as pd
import datetime
import os
from scipy.fft import fft, fftfreq

# ─────────────────────────────────────────────
# Constants / Thresholds
# ─────────────────────────────────────────────
FS = 100                          # Sampling frequency (Hz)
TREMOR_PATHOLOGICAL_HZ = 4.0     # Parkinson tremor range: 4-7 Hz
TREMOR_CRITICAL_HZ     = 6.0     # Critical upper bound
HR_LOW,  HR_HIGH       = 50, 110 # Heart-rate thresholds (bpm)
TEMP_LOW, TEMP_HIGH    = 35.5, 37.8
ACCEL_THRESHOLD        = 2.5     # g
LOG_PATH               = "logs/sensor_log.csv"


# ─────────────────────────────────────────────
# 1. Sensor Data Generator
# ─────────────────────────────────────────────
def generate_sensor_data(device_on: bool, tremor_severity: str = "mild") -> dict:
    """
    Simulate one reading from the wearable prototype.

    Parameters
    ----------
    device_on      : bool   — Whether the wearable is powered on
    tremor_severity: str    — 'none' | 'mild' | 'moderate' | 'severe'

    Returns a dict of sensor readings + timestamp.
    """
    if not device_on:
        return {
            "timestamp":       datetime.datetime.now(),
            "device_on":       False,
            "heart_rate":      None,
            "temperature":     None,
            "tremor_freq_hz":  None,
            "tremor_amplitude":None,
            "accel_x":         None,
            "accel_y":         None,
            "accel_z":         None,
            "spo2":            None,
            "signal_quality":  0,
        }

    # Tremor frequency map (Parkinson's: 4-7 Hz)
    severity_map = {
        "none":     (0.5,  1.5,  0.05, 0.15),   # (freq_min, freq_max, amp_min, amp_max)
        "mild":     (3.5,  4.5,  0.20, 0.50),
        "moderate": (4.5,  6.0,  0.50, 1.20),
        "severe":   (6.0,  7.5,  1.20, 2.50),
    }
    fmin, fmax, amin, amax = severity_map.get(tremor_severity, severity_map["mild"])

    tremor_freq = np.random.uniform(fmin, fmax)
    tremor_amp  = np.random.uniform(amin, amax)

    # Heart rate: slight variation around a base
    hr_base = {"none": 72, "mild": 78, "moderate": 85, "severe": 96}[tremor_severity]
    heart_rate = int(np.clip(hr_base + np.random.normal(0, 5), 40, 160))

    # Temperature
    temp_base = 36.5 + (0.4 if tremor_severity == "severe" else 0.0)
    temperature = round(np.clip(temp_base + np.random.normal(0, 0.2), 34.0, 40.0), 1)

    # Accelerometer (g)  — higher tremor = noisier accel
    noise = tremor_amp * 0.8
    accel_x = round(np.random.normal(0, noise + 0.05), 3)
    accel_y = round(np.random.normal(0, noise + 0.05), 3)
    accel_z = round(9.81 + np.random.normal(0, noise * 0.3), 3)

    # SpO2
    spo2 = int(np.clip(98 - tremor_amp * 0.5 + np.random.normal(0, 0.5), 85, 100))

    # Signal quality (%)
    signal_quality = int(np.clip(95 - tremor_amp * 3 + np.random.uniform(-5, 5), 40, 100))

    return {
        "timestamp":        datetime.datetime.now(),
        "device_on":        True,
        "heart_rate":       heart_rate,
        "temperature":      temperature,
        "tremor_freq_hz":   round(tremor_freq, 2),
        "tremor_amplitude": round(tremor_amp, 2),
        "accel_x":          accel_x,
        "accel_y":          accel_y,
        "accel_z":          accel_z,
        "spo2":             spo2,
        "signal_quality":   signal_quality,
    }


# ─────────────────────────────────────────────
# 2. Tremor Signal Synthesiser (for FFT plot)
# ─────────────────────────────────────────────
def generate_tremor_signal(tremor_freq_hz: float, duration_sec: float = 5.0) -> tuple:
    """
    Return (time_array, signal_array) for a simulated tremor waveform.
    Adds realistic noise and drift like the READINGS.xlsx data.
    """
    t   = np.linspace(0, duration_sec, int(FS * duration_sec))
    sig = np.sin(2 * np.pi * tremor_freq_hz * t)
    sig += 0.5 * np.random.randn(len(t))           # noise
    sig += 0.2 * np.sin(2 * np.pi * 0.3 * t)       # baseline drift
    return t, sig


# ─────────────────────────────────────────────
# 3. FFT Analyser
# ─────────────────────────────────────────────
def compute_fft(signal: np.ndarray) -> tuple:
    """Return (positive_freqs, magnitudes) from an FFT of the signal."""
    N     = len(signal)
    f_val = fft(signal)
    freqs = fftfreq(N, 1 / FS)
    pos   = freqs > 0
    return freqs[pos], np.abs(f_val[pos])


def dominant_frequency(signal: np.ndarray) -> float:
    freqs, mags = compute_fft(signal)
    return round(float(freqs[np.argmax(mags)]), 2)


# ─────────────────────────────────────────────
# 4. Anomaly / Alert Detection
# ─────────────────────────────────────────────
def detect_anomalies(reading: dict) -> list[dict]:
    """
    Analyse one sensor reading and return a list of alert dicts.
    Each alert: { 'level': 'warning'|'critical', 'icon': str, 'message': str }
    """
    if not reading.get("device_on"):
        return []

    alerts = []

    tf = reading["tremor_freq_hz"]
    ta = reading["tremor_amplitude"]
    hr = reading["heart_rate"]
    t  = reading["temperature"]
    s  = reading["spo2"]

    # Tremor frequency — Parkinson's range
    if TREMOR_PATHOLOGICAL_HZ <= tf < TREMOR_CRITICAL_HZ:
        alerts.append({
            "level":   "warning",
            "icon":    "⚠️",
            "message": f"Tremor at {tf} Hz — pathological range detected (4–6 Hz).",
        })
    elif tf >= TREMOR_CRITICAL_HZ:
        alerts.append({
            "level":   "critical",
            "icon":    "🚨",
            "message": f"CRITICAL: Tremor at {tf} Hz — severe Parkinson's pattern (≥6 Hz)!",
        })

    # Tremor amplitude
    if 0.5 <= ta < 1.2:
        alerts.append({"level": "warning", "icon": "⚠️",
                        "message": f"Tremor amplitude {ta} g — moderate shaking."})
    elif ta >= 1.2:
        alerts.append({"level": "critical", "icon": "🚨",
                        "message": f"CRITICAL: Tremor amplitude {ta} g — severe shaking!"})

    # Heart rate
    if hr < HR_LOW:
        alerts.append({"level": "critical", "icon": "💔",
                        "message": f"Heart rate critically LOW: {hr} bpm (<{HR_LOW})."})
    elif hr > HR_HIGH:
        alerts.append({"level": "warning", "icon": "💓",
                        "message": f"Heart rate elevated: {hr} bpm (>{HR_HIGH})."})

    # Temperature
    if t < TEMP_LOW:
        alerts.append({"level": "warning", "icon": "🌡️",
                        "message": f"Temperature LOW: {t} °C (< {TEMP_LOW})."})
    elif t > TEMP_HIGH:
        alerts.append({"level": "warning", "icon": "🌡️",
                        "message": f"Temperature elevated: {t} °C (> {TEMP_HIGH})."})

    # SpO2
    if s < 95:
        alerts.append({"level": "critical", "icon": "🫁",
                        "message": f"CRITICAL: SpO₂ LOW — {s}% (normal ≥ 95%)!"})
    elif s < 97:
        alerts.append({"level": "warning", "icon": "🫁",
                        "message": f"SpO₂ slightly low: {s}%."})

    return alerts


# ─────────────────────────────────────────────
# 5. Classification (rule-based ML-style)
# ─────────────────────────────────────────────
def classify_tremor(tremor_freq_hz: float, tremor_amplitude: float) -> tuple[str, str]:
    """
    Returns (label, confidence_str).
    Mimics the Random-Forest / SVM output from the notebook.
    """
    if tremor_freq_hz < 3.0 or tremor_amplitude < 0.2:
        label = "Non-Pathological"
        conf  = f"{np.random.uniform(88, 99):.1f}%"
    elif tremor_freq_hz >= 6.0 or tremor_amplitude >= 1.2:
        label = "Pathological — Severe"
        conf  = f"{np.random.uniform(90, 99):.1f}%"
    elif 4.0 <= tremor_freq_hz < 6.0:
        label = "Pathological — Moderate"
        conf  = f"{np.random.uniform(82, 95):.1f}%"
    else:
        label = "Borderline — Monitor"
        conf  = f"{np.random.uniform(65, 80):.1f}%"
    return label, conf


# ─────────────────────────────────────────────
# 6. CSV Logger
# ─────────────────────────────────────────────
def log_reading(reading: dict):
    """Append one sensor reading to the CSV log."""
    os.makedirs("logs", exist_ok=True)
    df_row = pd.DataFrame([{
        k: v for k, v in reading.items()
        if k != "device_on"
    }])
    df_row["timestamp"] = df_row["timestamp"].astype(str)
    write_header = not os.path.exists(LOG_PATH)
    df_row.to_csv(LOG_PATH, mode="a", header=write_header, index=False)


def load_log() -> pd.DataFrame:
    """Load the CSV log (returns empty DataFrame if missing)."""
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    return pd.DataFrame()


def clear_log():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)


# ─────────────────────────────────────────────
# 7. WiFi Signal Simulator
# ─────────────────────────────────────────────
def wifi_signal_strength() -> int:
    """Returns a simulated WiFi RSSI strength (-90 to -30 dBm)."""
    return int(np.random.uniform(-65, -40))


def latency_ms() -> float:
    """Simulated round-trip latency in ms."""
    return round(np.random.uniform(8, 45), 1)
