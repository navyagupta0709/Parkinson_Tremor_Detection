
import csv
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from collections import deque

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
LOG_PATH  = BASE_DIR / "data" / "tremor_logs.csv"

CSV_FIELDS = [
    "timestamp", "elapsed_s", "voltage_V",
    "amplitude_V", "dom_freq_hz", "band_power",
    "peak_count", "signal_quality",
    "prediction", "confidence", "severity_pct",
]


# ── CSV Logger ────────────────────────────────────────────────────────────────
class DataLogger:
    def __init__(self, path: Path = LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_header()

    def _write_header(self):
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()

    def log(self, row: dict):
        row.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writerow(row)

    def read_recent(self, n: int = 200) -> list:
        """Return the last n rows as list of dicts."""
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-n:]

    def clear(self):
        if self.path.exists():
            self.path.unlink()
        self._write_header()


# ── Alert Manager ─────────────────────────────────────────────────────────────
class AlertManager:
    """
    Stores alert events with deduplication.
    An alert is raised when severity_pct > threshold.
    """

    THRESHOLDS = {"Normal": 0, "Mild Tremor": 30, "Severe Tremor": 70}

    def __init__(self, max_history: int = 50):
        self.history: deque = deque(maxlen=max_history)
        self._last_label: Optional[str] = None
        self._last_alert_time: float = 0.0
        self.MIN_INTERVAL = 5.0   # minimum seconds between duplicate alerts

    def evaluate(self, prediction: dict) -> Optional[dict]:
        """
        Check if an alert should be raised.
        Returns alert dict or None.
        """
        label       = prediction["label"]
        severity    = prediction["severity_pct"]
        confidence  = prediction["confidence"]
        now         = time.time()

        if label == "Normal":
            self._last_label = label
            return None

        # Dedup: don't repeat same alert within MIN_INTERVAL
        if (label == self._last_label and
                now - self._last_alert_time < self.MIN_INTERVAL):
            return None

        alert = {
            "time":       datetime.now().strftime("%H:%M:%S"),
            "label":      label,
            "severity":   round(severity, 1),
            "confidence": round(confidence * 100, 1),
            "is_severe":  label == "Severe Tremor",
        }
        self.history.appendleft(alert)
        self._last_label      = label
        self._last_alert_time = now
        logger.warning(f"ALERT: {label} | Severity {severity:.1f}% | Conf {confidence*100:.1f}%")
        return alert

    def get_history(self) -> list:
        return list(self.history)

    def clear(self):
        self.history.clear()
        self._last_label      = None
        self._last_alert_time = 0.0


# ── Latency Tracker ───────────────────────────────────────────────────────────
class LatencyTracker:
    def __init__(self, window: int = 20):
        self._times: deque = deque(maxlen=window)

    def ping(self):
        self._times.append(time.time())

    def latency_ms(self) -> float:
        if len(self._times) < 2:
            return 0.0
        diffs = [
            (self._times[i] - self._times[i - 1]) * 1000
            for i in range(1, len(self._times))
        ]
        return float(sum(diffs) / len(diffs))


# ── Display Helpers ───────────────────────────────────────────────────────────
def severity_color(severity_pct: float) -> str:
    if severity_pct < 25:
        return "#00e676"
    if severity_pct < 60:
        return "#ffab40"
    return "#ff1744"


def format_uptime(start_ts: float) -> str:
    elapsed = int(time.time() - start_ts)
    h, rem  = divmod(elapsed, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
