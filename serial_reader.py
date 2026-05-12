
import serial
import serial.tools.list_ports
import threading
import time
import logging
from collections import deque
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
BAUD_RATE      = 9600
BUFFER_SIZE    = 1000          # rolling sample buffer (~10 s @ 100 Hz)
CONNECT_RETRY  = 3.0           # seconds between reconnect attempts
READ_TIMEOUT   = 2.0           # serial read timeout
HANDSHAKE_STR  = "TENG_READY"  # first line Arduino sends


# ── Port Detection ────────────────────────────────────────────────────────────
def list_arduino_ports() -> list:
    """Return list of likely Arduino COM ports."""
    candidates = []
    for port in serial.tools.list_ports.comports():
        desc  = (port.description or "").lower()
        manuf = (port.manufacturer or "").lower()
        if any(k in desc  for k in ["arduino", "ch340", "ch341", "cp210", "ftdi"]):
            candidates.append(port.device)
        elif any(k in manuf for k in ["arduino", "wch", "silicon"]):
            candidates.append(port.device)
        elif port.vid in (0x2341, 0x2A03, 0x1A86, 0x10C4, 0x0403):
            candidates.append(port.device)
    return candidates


def detect_port() -> Optional[str]:
    """Auto-detect the first available Arduino port."""
    ports = list_arduino_ports()
    if ports:
        logger.info(f"Auto-detected Arduino on: {ports[0]}")
        return ports[0]

    # Fallback: try common names
    fallbacks = [
        "/dev/ttyUSB0", "/dev/ttyACM0",
        "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    ]
    for p in fallbacks:
        try:
            s = serial.Serial(p, BAUD_RATE, timeout=0.5)
            s.close()
            logger.info(f"Fallback port found: {p}")
            return p
        except serial.SerialException:
            continue
    return None


# ── Serial Reader Thread ──────────────────────────────────────────────────────
class SerialReader:
    """
    Background thread that reads from Arduino serial and fills a deque buffer.
    Thread-safe: all public attributes use locks.
    """

    def __init__(self, port: Optional[str] = None, baud: int = BAUD_RATE):
        self._port        = port
        self._baud        = baud
        self._ser: Optional[serial.Serial] = None
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()

        # Public state (read with lock)
        self.buffer: deque[Tuple[float, float]] = deque(maxlen=BUFFER_SIZE)
        self.connected    = False
        self.active_port  = None
        self.error_msg    = ""
        self.bytes_read   = 0
        self.samples_read = 0
        self.last_value   = 0.0
        self.last_time    = 0.0

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SerialReader"
        )

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self):
        self._stop_event.clear()
        self._thread.start()
        logger.info("SerialReader thread started")

    def stop(self):
        self._stop_event.set()
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        self._thread.join(timeout=5)
        logger.info("SerialReader thread stopped")

    def get_snapshot(self) -> list:
        """Return a copy of the current buffer as list of (time, voltage)."""
        with self._lock:
            return list(self.buffer)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "connected":    self.connected,
                "port":         self.active_port,
                "error":        self.error_msg,
                "samples":      self.samples_read,
                "last_voltage": self.last_value,
                "last_time":    self.last_time,
            }

    # ── Internal ──────────────────────────────────────────────────────────────
    def _run(self):
        while not self._stop_event.is_set():
            port = self._port or detect_port()
            if port is None:
                with self._lock:
                    self.connected = False
                    self.error_msg = "No Arduino detected. Check USB connection."
                time.sleep(CONNECT_RETRY)
                continue

            try:
                self._connect(port)
                self._read_loop()
            except serial.SerialException as e:
                with self._lock:
                    self.connected = False
                    self.error_msg = f"Serial error: {e}"
                logger.warning(f"Serial error: {e}. Retrying in {CONNECT_RETRY}s…")
                time.sleep(CONNECT_RETRY)
            except Exception as e:
                with self._lock:
                    self.connected = False
                    self.error_msg = f"Unexpected error: {e}"
                logger.error(f"Unexpected error: {e}")
                time.sleep(CONNECT_RETRY)

    def _connect(self, port: str):
        if self._ser and self._ser.is_open:
            self._ser.close()

        self._ser = serial.Serial(
            port,
            self._baud,
            timeout=READ_TIMEOUT,
            write_timeout=1,
        )
        time.sleep(2.0)          # wait for Arduino reset
        self._ser.reset_input_buffer()

        # Wait for handshake (up to 5 s)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            line = self._ser.readline().decode("ascii", errors="ignore").strip()
            if HANDSHAKE_STR in line or line:   # accept any valid line
                break

        with self._lock:
            self.connected   = True
            self.active_port = port
            self.error_msg   = ""
        logger.info(f"Connected to Arduino on {port}")

    def _read_loop(self):
        while not self._stop_event.is_set():
            raw = self._ser.readline()
            if not raw:
                continue

            line = raw.decode("ascii", errors="ignore").strip()
            if not line or HANDSHAKE_STR in line:
                continue

            parsed = self._parse_line(line)
            if parsed is not None:
                t, v = parsed
                with self._lock:
                    self.buffer.append((t, v))
                    self.last_value   = v
                    self.last_time    = t
                    self.samples_read += 1
                    self.bytes_read   += len(raw)

    @staticmethod
    def _parse_line(line: str) -> Optional[Tuple[float, float]]:
        """Parse 'time voltage' line. Returns (time, voltage) or None."""
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            t = float(parts[0])
            v = float(parts[1])
            # Sanity check
            if 0.0 <= v <= 5.1 and t >= 0.0:
                return t, v
        except ValueError:
            pass
        return None
