"""
serial_reader.py
================
Reads serial data from Arduino running sketch_feb18a.ino.

Arduino output format (one line per sample):
    "<time_seconds> <voltage_volts>\\n"
    e.g.  "1.230 2.4812"

Specs from sketch_feb18a.ino:
    - Serial.begin(9600)          → BAUD = 9600
    - samplingInterval = 10 ms   → 100 Hz
    - analogPin = A0
    - referenceVoltage = 5.0 V
    - ADC formula: voltage = (rawADC * 5.0) / 1023.0
"""

import threading
import time
from collections import deque

# pyserial is optional — app works on cloud without it
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# must match sketch_feb18a.ino  Serial.begin(9600)
BAUD = 9600


# ─────────────────────────────────────────────────────────────
# Port discovery
# ─────────────────────────────────────────────────────────────
def list_ports() -> list[str]:
    """Return all available COM / tty port names."""
    if not HAS_SERIAL:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def auto_detect_port() -> str | None:
    """
    Try to find Arduino UNO by USB descriptor keywords.
    Falls back to first available port.
    """
    if not HAS_SERIAL:
        return None
    keywords = ["Arduino", "CH340", "USB-SERIAL",
                "ttyUSB", "ttyACM", "usbserial"]
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "") + (p.manufacturer or "")
        if any(k.lower() in desc.lower() for k in keywords):
            return p.device
    # fallback
    ports = serial.tools.list_ports.comports()
    return ports[0].device if ports else None


# ─────────────────────────────────────────────────────────────
# Arduino reader
# ─────────────────────────────────────────────────────────────
class ArduinoReader:
    """
    Background thread that reads serial lines from Arduino.

    Parses lines produced by sketch_feb18a.ino:
        Serial.print(timeString);   // e.g. "1.230"
        Serial.print(" ");
        Serial.println(voltage, 4); // e.g. "2.4812"

    Public API:
        reader.start()
        reader.stop()
        reader.latest(n)      → (timestamps, voltages)
        reader.sample_count() → int
        reader.connected      → bool
        reader.error          → str (last exception)
        reader.bytes_rx       → int
    """

    def __init__(self, port: str, baud: int = BAUD, maxlen: int = 1000):
        self.port      = port
        self.baud      = baud
        self._buf      = deque(maxlen=maxlen)  # (time_s, voltage_V)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None
        self.connected = False
        self.error     = ""
        self.bytes_rx  = 0

    # ── control ───────────────────────────────────────────────
    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running  = False
        self.connected = False

    # ── data access ───────────────────────────────────────────
    def latest(self, n: int = 200):
        """Return (timestamps, voltages) for the last n samples."""
        with self._lock:
            data = list(self._buf)[-n:]
        if not data:
            return [], []
        return [d[0] for d in data], [d[1] for d in data]

    def sample_count(self) -> int:
        with self._lock:
            return len(self._buf)

    # ── background loop ───────────────────────────────────────
    def _loop(self):
        while self._running:
            try:
                ser = serial.Serial(self.port, self.baud, timeout=1)
                self.connected = True
                self.error     = ""

                while self._running:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw:
                        continue

                    self.bytes_rx += len(raw)

                    # Parse: "time_s voltage_V"
                    # Matches: Serial.print(timeString); Serial.print(" "); Serial.println(voltage,4)
                    parts = raw.split()
                    if len(parts) >= 2:
                        try:
                            ts  = float(parts[0])   # time in seconds
                            val = float(parts[1])   # voltage 0–5 V
                            with self._lock:
                                self._buf.append((ts, val))
                        except ValueError:
                            pass   # skip malformed lines silently

                ser.close()

            except Exception as exc:
                self.connected = False
                self.error     = str(exc)
                time.sleep(2)   # wait 2 s before retry
