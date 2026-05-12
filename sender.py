"""
TremorWatch — sender.py
Run this on your PC with Arduino plugged in via USB.
It reads serial data, detects tremors via FFT,
and publishes to MQTT so the Streamlit Cloud dashboard receives it live.

Usage:
    python sender.py               # auto-detect Arduino port
    python sender.py COM3          # specify port (Windows)
    python sender.py /dev/ttyUSB0  # specify port (Linux/Mac)
"""

import sys
import time
import json
import serial
import serial.tools.list_ports
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque
from datetime import datetime

# ── Config (must match app.py) ────────────────────────────────────
MQTT_BROKER  = "broker.hivemq.com"
MQTT_PORT    = 1883
MQTT_TOPIC   = "tremorwatch/data/v1"

BAUD_RATE    = 9600
SAMPLE_RATE  = 100
BUFFER_SIZE  = SAMPLE_RATE * 5   # 5-second FFT window

TREMOR_LOW   = 3.0
TREMOR_HIGH  = 12.0
POWER_THRESH = 0.01
AMP_THRESH   = 0.15

# ── MQTT setup ────────────────────────────────────────────────────
mq = mqtt.Client(client_id=f"tremorwatch_sender_{int(time.time())}")
mq.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
mq.loop_start()
print(f"[MQTT] Connected to {MQTT_BROKER}")

# ── Serial port ───────────────────────────────────────────────────
def find_port():
    override = sys.argv[1] if len(sys.argv) > 1 else None
    if override:
        return override
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if any(k in desc for k in ["arduino","ch340","cp210","usb serial"]):
            return p.device
    ports = serial.tools.list_ports.comports()
    return ports[0].device if ports else None

port = find_port()
if not port:
    print("[ERROR] No serial port found. Plug in Arduino and try again.")
    sys.exit(1)

print(f"[Serial] Opening {port} @ {BAUD_RATE} baud ...")
ser = serial.Serial(port, BAUD_RATE, timeout=1)
time.sleep(2)
ser.flushInput()
print(f"[Serial] Connected. Streaming to: {MQTT_TOPIC}")
print("         Open your Streamlit Cloud URL — data appears live.\n")

# ── Buffers ───────────────────────────────────────────────────────
voltage_buf   = deque(maxlen=BUFFER_SIZE)
tremor_active = False
tremor_start  = None
analysis_tick = 0

def classify(power):
    if power > 0.4:    return "Severe"
    elif power > 0.2:  return "Moderate"
    elif power > 0.05: return "Mild"
    return "Trace"

def detect_tremor(buf):
    if len(buf) < SAMPLE_RATE:
        return False, 0.0, 0.0
    arr = np.array(buf, dtype=float)
    arr -= arr.mean()
    if arr.max() - arr.min() < AMP_THRESH:
        return False, 0.0, 0.0
    fft_v = np.abs(np.fft.rfft(arr))
    fft_f = np.fft.rfftfreq(len(arr), d=1.0 / SAMPLE_RATE)
    total = np.sum(fft_v ** 2)
    if total == 0:
        return False, 0.0, 0.0
    mask  = (fft_f >= TREMOR_LOW) & (fft_f <= TREMOR_HIGH)
    bpow  = float(np.sum(fft_v[mask] ** 2) / total)
    dc    = fft_f > 0.5
    dom   = float(fft_f[np.argmax(fft_v * dc)])
    hit   = bpow >= POWER_THRESH and TREMOR_LOW <= dom <= TREMOR_HIGH
    return hit, dom, bpow

# ── Main loop ─────────────────────────────────────────────────────
print("[Loop] Reading Arduino ...  (Ctrl+C to stop)\n")
try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t_sec   = float(parts[0])
            voltage = float(parts[1])
        except ValueError:
            continue

        voltage_buf.append(voltage)
        analysis_tick += 1

        payload = {
            "voltage": round(voltage, 4),
            "time":    round(t_sec,   3),
            "freq":    0.0,
            "power":   0.0,
            "tremor":  tremor_active,
            "event":   None,
        }

        # FFT every 10 samples = 100 ms
        if analysis_tick >= 10:
            analysis_tick = 0
            detected, dom_freq, band_power = detect_tremor(list(voltage_buf))

            payload["freq"]   = round(dom_freq,    2)
            payload["power"]  = round(band_power,  4)
            payload["tremor"] = detected

            if detected and not tremor_active:
                tremor_active = True
                tremor_start  = datetime.now()
                print(f"[ALERT] Tremor START  freq={dom_freq:.1f}Hz  power={band_power:.3f}")

            elif not detected and tremor_active:
                dur = (datetime.now() - tremor_start).total_seconds()
                ev  = {
                    "start":    tremor_start.strftime("%H:%M:%S"),
                    "duration": round(dur,        1),
                    "freq":     round(dom_freq,   2),
                    "power":    round(band_power, 4),
                    "severity": classify(band_power),
                }
                payload["event"] = ev
                tremor_active = False
                tremor_start  = None
                print(f"[INFO]  Tremor END  dur={dur:.1f}s  sev={ev['severity']}")

        mq.publish(MQTT_TOPIC, json.dumps(payload), qos=0)

except KeyboardInterrupt:
    print("\n[Stopped] sender.py shut down cleanly.")
    ser.close()
    mq.loop_stop()
    mq.disconnect()
