import numpy as np
import pandas as pd
import datetime
import os
from scipy.fft import fft, fftfreq
import scipy.signal as scipy_signal

FS = 100

TREMOR_LOW = 3.0
TREMOR_HIGH = 7.0

LOG_PATH = "logs/teng_log.csv"

# =========================================
# REAL SIGNAL PROCESSING
# =========================================

def process_signal(signal):

    signal = np.nan_to_num(signal)

    signal = signal - np.mean(signal)

    signal = scipy_signal.detrend(signal)

    b, a = scipy_signal.butter(
        4,
        [0.5, 10],
        btype='bandpass',
        fs=FS
    )

    signal = scipy_signal.filtfilt(
        b,
        a,
        signal
    )

    signal = scipy_signal.savgol_filter(
        signal,
        11,
        2
    )

    return signal

# =========================================
# FFT
# =========================================

def compute_fft(signal):

    n = len(signal)

    fft_vals = fft(signal)

    freqs = fftfreq(
        n,
        d=1/FS
    )

    mask = freqs > 0

    freqs = freqs[mask]

    amps = (2/n) * np.abs(
        fft_vals[mask]
    )

    return freqs, amps

# =========================================
# DOMINANT FREQUENCY
# =========================================

def dominant_frequency(signal):

    freqs, amps = compute_fft(signal)

    idx = np.argmax(amps)

    return round(
        float(freqs[idx]),
        2
    )

# =========================================
# BAND POWER
# =========================================

def tremor_band_power(signal):

    freqs, amps = compute_fft(signal)

    mask = (
        (freqs >= 3) &
        (freqs <= 7)
    )

    power = np.trapz(
        amps[mask],
        freqs[mask]
    )

    return round(
        float(power),
        4
    )

# =========================================
# DETECT TREMOR
# =========================================

def detect_tremor(freq):

    if TREMOR_LOW <= freq <= TREMOR_HIGH:

        return True

    return False

# =========================================
# CLASSIFICATION
# =========================================

def classify_tremor(freq):

    if freq < 3:

        return "Normal"

    elif 3 <= freq < 4:

        return "Mild Tremor"

    elif 4 <= freq < 5.5:

        return "Moderate Tremor"

    elif 5.5 <= freq <= 7:

        return "Severe Tremor"

    else:

        return "Abnormal"

# =========================================
# CREATE READING
# =========================================

def create_reading(freq, power, status):

    return {

        "timestamp": str(
            datetime.datetime.now()
        ),

        "dominant_frequency": freq,

        "band_power": power,

        "status": status
    }

# =========================================
# LOGGING
# =========================================

def log_reading(reading):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    df = pd.DataFrame([reading])

    write_header = not os.path.exists(
        LOG_PATH
    )

    df.to_csv(
        LOG_PATH,
        mode='a',
        header=write_header,
        index=False
    )

# =========================================
# LOAD LOG
# =========================================

def load_log():

    if os.path.exists(LOG_PATH):

        return pd.read_csv(LOG_PATH)

    return pd.DataFrame()

# =========================================
# CLEAR LOG
# =========================================

def clear_log():

    if os.path.exists(LOG_PATH):

        os.remove(LOG_PATH)
