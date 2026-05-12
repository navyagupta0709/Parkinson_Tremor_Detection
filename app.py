# =========================================
# app.py
# REAL TENG FFT STREAMLIT DASHBOARD
# =========================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scipy_signal
from scipy.fft import fft, fftfreq
import serial
import threading
import time

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="TENG Tremor Detection",
    page_icon="⚡",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

body {
    background-color:#020617;
}

.big-alert {
    background:#7f1d1d;
    border:4px solid red;
    padding:25px;
    border-radius:15px;
    text-align:center;
    margin-top:20px;
}

.ok-box {
    background:#052e16;
    border:3px solid #22c55e;
    padding:20px;
    border-radius:15px;
    text-align:center;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# PARAMETERS
# =========================================

SERIAL_PORT = "COM3"
BAUD_RATE = 9600
FS = 100

# =========================================
# SESSION STATE
# =========================================

if "signal" not in st.session_state:
    st.session_state.signal = []

if "times" not in st.session_state:
    st.session_state.times = []

if "connected" not in st.session_state:
    st.session_state.connected = False

if "thread_started" not in st.session_state:
    st.session_state.thread_started = False

# =========================================
# SERIAL READER
# =========================================

def serial_worker():

    try:

        ser = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1
        )

        st.session_state.connected = True

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

                st.session_state.times.append(
                    t_sec
                )

                st.session_state.signal.append(
                    voltage
                )

                if len(st.session_state.signal) > 500:

                    st.session_state.signal.pop(0)
                    st.session_state.times.pop(0)

            except:
                pass

    except:

        st.session_state.connected = False

# =========================================
# START THREAD
# =========================================

if not st.session_state.thread_started:

    thread = threading.Thread(
        target=serial_worker,
        daemon=True
    )

    thread.start()

    st.session_state.thread_started = True

# =========================================
# HEADER
# =========================================

st.title(
    "⚡ TENG Parkinson Tremor Detection"
)

st.caption(
    "Real-Time FFT Spectrum Analysis"
)

# =========================================
# CONNECTION STATUS
# =========================================

if st.session_state.connected:

    st.success(
        "🟢 Arduino Connected — Live Data Streaming"
    )

else:

    st.warning(
        "🟡 Waiting for Arduino Data..."
    )

# =========================================
# GET SIGNAL
# =========================================

sig = np.array(
    st.session_state.signal
)

t = np.array(
    st.session_state.times
)

# =========================================
# SIGNAL PROCESSING
# =========================================

freq = 0.0
power = 0.0

if len(sig) >= 128:

    sig = np.nan_to_num(sig)

    sig = sig - np.mean(sig)

    sig = scipy_signal.detrend(sig)

    b, a = scipy_signal.butter(
        4,
        [0.5, 10],
        btype='bandpass',
        fs=FS
    )

    sig = scipy_signal.filtfilt(
        b,
        a,
        sig
    )

    sig = scipy_signal.savgol_filter(
        sig,
        11,
        2
    )

    # =========================================
    # FFT
    # =========================================

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

    idx = np.argmax(
        amps
    )

    freq = float(
        freqs[idx]
    )

    power = float(
        amps[idx]
    )

# =========================================
# ALERT SYSTEM
# =========================================

if 3 <= freq <= 7:

    if freq < 4:

        severity = "MILD"

    elif freq < 5.5:

        severity = "MODERATE"

    else:

        severity = "SEVERE"

    st.markdown(f"""
    <div class="big-alert">

    <h1 style="color:white;">
    🚨 TREMOR DETECTED
    </h1>

    <h2 style="color:#fecaca;">
    {freq:.2f} Hz
    </h2>

    <h3 style="color:#fca5a5;">
    Severity : {severity}
    </h3>

    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class="ok-box">

    <h2 style="color:#bbf7d0;">
    ✅ Normal Signal
    </h2>

    </div>
    """, unsafe_allow_html=True)

# =========================================
# METRICS
# =========================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Dominant Frequency",
    f"{freq:.2f} Hz"
)

c2.metric(
    "Band Power",
    f"{power:.4f}"
)

c3.metric(
    "Status",
    "Tremor" if 3 <= freq <= 7 else "Normal"
)

# =========================================
# TABS
# =========================================

tabs = st.tabs([
    "📈 Live Signal",
    "⚙️ Signal Processing",
    "🔬 FFT Spectrum"
])

# =========================================
# LIVE SIGNAL
# =========================================

with tabs[0]:

    fig1, ax1 = plt.subplots(
        figsize=(14,4)
    )

    fig1.patch.set_facecolor(
        '#020617'
    )

    ax1.set_facecolor(
        '#0f172a'
    )

    ax1.plot(
        t,
        sig,
        color='cyan',
        linewidth=1.2
    )

    ax1.set_title(
        'Live TENG Signal',
        color='white'
    )

    ax1.set_xlabel(
        'Time (s)',
        color='white'
    )

    ax1.set_ylabel(
        'Voltage (V)',
        color='white'
    )

    ax1.tick_params(
        colors='white'
    )

    ax1.grid(alpha=0.2)

    st.pyplot(fig1)

# =========================================
# SIGNAL PROCESSING
# =========================================

with tabs[1]:

    fig2, ax2 = plt.subplots(
        figsize=(14,4)
    )

    fig2.patch.set_facecolor(
        '#020617'
    )

    ax2.set_facecolor(
        '#0f172a'
    )

    ax2.plot(
        sig,
        color='lime'
    )

    ax2.set_title(
        'Processed Signal',
        color='white'
    )

    ax2.tick_params(
        colors='white'
    )

    ax2.grid(alpha=0.2)

    st.pyplot(fig2)

# =========================================
# FFT
# =========================================

with tabs[2]:

    if len(sig) >= 128:

        fig3, ax3 = plt.subplots(
            figsize=(14,4)
        )

        fig3.patch.set_facecolor(
            '#020617'
        )

        ax3.set_facecolor(
            '#0f172a'
        )

        ax3.plot(
            freqs,
            amps,
            color='deepskyblue',
            linewidth=1.5
        )

        ax3.axvspan(
            3,
            7,
            color='red',
            alpha=0.15,
            label='Tremor Band'
        )

        ax3.plot(
            freq,
            power,
            'ro'
        )

        ax3.set_xlim(0,12)

        ax3.set_title(
            'FFT Spectrum',
            color='white'
        )

        ax3.set_xlabel(
            'Frequency (Hz)',
            color='white'
        )

        ax3.set_ylabel(
            'Amplitude',
            color='white'
        )

        ax3.tick_params(
            colors='white'
        )

        ax3.legend()

        ax3.grid(alpha=0.2)

        st.pyplot(fig3)

# =========================================
# AUTO REFRESH
# =========================================

time.sleep(1)

st.rerun()
