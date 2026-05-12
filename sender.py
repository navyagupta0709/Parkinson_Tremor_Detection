# =========================
# sender.py
# =========================

import serial
import json
import numpy as np
import scipy.signal as scipy_signal
from scipy.fft import fft, fftfreq
import paho.mqtt.client as mqtt
import time

# ---------------------------------------------------
# SERIAL CONFIG
# ---------------------------------------------------

SERIAL_PORT = "COM3"
BAUD_RATE = 9600

# ---------------------------------------------------
# MQTT CONFIG
# ---------------------------------------------------

BROKER = "broker.hivemq.com"
PORT = 1883
MQTT_TOPIC = "tremorwatch/data/navya"

# ---------------------------------------------------
# CONNECT SERIAL
# ---------------------------------------------------

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

# ---------------------------------------------------
# MQTT
# ---------------------------------------------------

client = mqtt.Client()

client.connect(
    BROKER,
    PORT,
    60
)

client.loop_start()

# ---------------------------------------------------
# PARAMETERS
# ---------------------------------------------------

FS = 100

buffer = []

print("Reading Arduino Data...")

# ---------------------------------------------------
# LOOP
# ---------------------------------------------------

while True:

    try:

        line = ser.readline().decode().strip()

        if not line:
            continue

        parts = line.split(",")

        if len(parts) != 2:
            continue

        t_sec = float(parts[0])

        voltage = float(parts[1])

        buffer.append(voltage)

        if len(buffer) > 500:
            buffer.pop(0)

        freq = 0.0
        power = 0.0
        tremor = False

        # ---------------------------------------------------
        # FFT
        # ---------------------------------------------------

        if len(buffer) >= 256:

            sig = np.array(buffer)

            sig = sig - np.mean(sig)

            sig = scipy_signal.detrend(sig)

            b, a = scipy_signal.butter(
                4,
                [0.5, 10],
                btype="bandpass",
                fs=FS
            )

            sig = scipy_signal.filtfilt(
                b,
                a,
                sig
            )

            n = len(sig)

            fft_vals = fft(sig)

            freqs = fftfreq(
                n,
                d=1/FS
            )

            mask = freqs > 0

            freqs = freqs[mask]

            amps = (2/n) * np.abs(
                fft_vals[mask]
            )

            idx = np.argmax(amps)

            freq = float(freqs[idx])

            power = float(np.max(amps))

            if 3 <= freq <= 7:
                tremor = True

        # ---------------------------------------------------
        # MQTT PAYLOAD
        # ---------------------------------------------------

        payload = {

            "time": t_sec,
            "voltage": voltage,
            "freq": freq,
            "power": power,
            "tremor": tremor
        }

        client.publish(
            MQTT_TOPIC,
            json.dumps(payload)
        )

        print(payload)

    except Exception as e:

        print(e)

        time.sleep(1)
