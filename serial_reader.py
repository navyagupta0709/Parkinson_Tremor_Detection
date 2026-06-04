"""
serial_reader.py
Thread-safe Arduino UNO serial reader.
Parses format: "TIME_SECONDS VOLTAGE\n"  e.g. "1.230 2.4812"
Sent by tremor_sensor.ino at 9600 baud, 100 Hz.

NO simulation — if Arduino is not connected, returns empty data.
"""

import threading
import time
from collections import deque

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

BAUD = 9600


def list_ports() -> list:
    if not HAS_SERIAL:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def auto_detect_port() -> str | None:
    """Auto-detect Arduino UNO by USB descriptor."""
    if not HAS_SERIAL:
        return None
    keywords = ["Arduino", "CH340", "USB-SERIAL", "ttyUSB", "ttyACM", "usbserial"]
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "") + (p.manufacturer or "")
        if any(k.lower() in desc.lower() for k in keywords):
            return p.device
    # fallback: return first port
    ports = serial.tools.list_ports.comports()
    return ports[0].device if ports else None


class ArduinoReader:
    """
    Reads serial data from Arduino in a background thread.
    - connected  : True when port is open and data is flowing
    - error      : last exception string
    - bytes_rx   : total bytes received this session
    """

    def __init__(self, port: str, baud: int = BAUD, maxlen: int = 800):
        self.port      = port
        self.baud      = baud
        self._buf      = deque(maxlen=maxlen)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None
        self.connected = False
        self.error     = ""
        self.bytes_rx  = 0

    # ─── public API ─────────────────────────────────────────────
    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running  = False
        self.connected = False

    def latest(self, n: int = 200):
        """Return (timestamps, voltages) for last n readings."""
        with self._lock:
            data = list(self._buf)[-n:]
        if not data:
            return [], []
        return [d[0] for d in data], [d[1] for d in data]

    def sample_count(self) -> int:
        with self._lock:
            return len(self._buf)

    # ─── background thread ──────────────────────────────────────
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
                    parts = raw.split()
                    if len(parts) >= 2:
                        try:
                            ts  = float(parts[0])
                            val = float(parts[1])
                            with self._lock:
                                self._buf.append((ts, val))
                        except ValueError:
                            pass
                ser.close()
            except Exception as exc:
                self.connected = False
                self.error     = str(exc)
                time.sleep(2)   # retry after 2 s
