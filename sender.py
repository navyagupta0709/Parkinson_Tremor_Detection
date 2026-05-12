# =========================================
# sender.py
# REAL LIVE ARDUINO STREAM
# =========================================

import serial
import json
import numpy as np
from scipy.fft import fft, fftfreq
import scipy.signal as scipy_signal
import paho.mqtt.client as mqtt

# =========================================
# SERIAL
# =========================================

SERIAL_PORT = "COM3"
BAUD_RATE = 9600

# =========================================
# MQTT
# =========================================

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "teng/live/navya"

# =========================================
# CONNECT SERIAL
# =========================================

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

# =========================================
# MQTT CLIENT
# =========================================

client = mqtt.Client()

client.connect(
    BROKER,
    PORT,
    60
)

client.loop_start()

# =========================================
# PARAMETERS
# =========================================

FS = 100

buffer = []

print("Reading Arduino Data...")

# =========================================
# LOOP
# =========================================

while True:

    try:

        line = ser.readline().decode().strip()

        print(line)

        if not line:
            continue

        parts = line.split(",")

        if len(parts) != 2:
            continue

        t_sec = float(parts[0])

        voltage = float(parts[1])

        buffer.append(voltage)

        if len(buffer) > 256:
            buffer.pop(0)

        freq = 0.0
        power = 0.0
        tremor = False

        # =========================================
        # FFT
        # =========================================

        if len(buffer) >= 128:

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

            amps = np.abs(
                fft_vals[mask]
            )

            idx = np.argmax(
                amps
            )

            freq = float(
                freqs[idx]
            )

            power = float(
                amps[idx]
            )

            if 3 <= freq <= 7:

                tremor = True

        payload = {

            "time": t_sec,
            "voltage": voltage,
            "freq": freq,
            "power": power,
            "tremor": tremor
        }

        client.publish(
            TOPIC,
            json.dumps(payload)
        )

    except Exception as e:

        print(e)
