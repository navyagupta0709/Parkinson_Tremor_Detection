"""
app.py
IEEE-Level TENG Parkinson Tremor Detection Dashboard
REALTIME Arduino + FFT + AI Monitoring System

RUN:
streamlit run app.py
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import datetime
from collections import deque
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import threading
import queue
import random

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="TENG Parkinson Tremor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background-color: #07111f;
    color: white;
}

/* HEADER */
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #00d4ff;
    text-align: center;
}

.sub-title {
    text-align: center;
    color: #9fb3c8;
    font-size: 15px;
    margin-bottom: 20px;
}

/* CARDS */
.metric-card {
    background: #0d1b2a;
    border: 1px solid #1b3350;
    border-radius: 12px;
    padding: 20px;
}

/* ALERTS */
.alert-normal {
    background: rgba(0,255,127,0.12);
    border-left: 6px solid #00ff7f;
    padding: 18px;
    border-radius: 10px;
    color: #00ff7f;
    font-size: 22px;
    font-weight: bold;
}

.alert-mild {
    background: rgba(255,165,0,0.15);
    border-left: 6px solid orange;
    padding: 18px;
    border-radius: 10px;
    color: orange;
    font-size: 22px;
    font-weight: bold;
}

.alert-severe {
    background: rgba(255,0,0,0.15);
    border-left: 6px solid red;
    padding: 20px;
    border-radius: 10px;
    color: #ff4d4d;
    font-size: 24px;
    font-weight: bold;
    animation: blink 1s infinite;
}

@keyframes blink {
    50% { opacity: 0.4; }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "serial_running" not in st.session_state:
    st.session_state.serial_running = False

if "data_buffer" not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=1000)

if "fft_buffer" not in st.session_state:
    st.session_state.fft_buffer = deque(maxlen=512)

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

if "connected" not in st.session_state:
    st.session_state.connected = False

# =========================================================
# SERIAL DATA QUEUE
# =========================================================
data_queue = queue.Queue()

# =========================================================
# ARDUINO PORT DETECTION
# =========================================================
def detect_arduino():

    ports = serial.tools.list_ports.comports()

    available_ports = []

    for port in ports:
        available_ports.append(port.device)

    return available_ports

# =========================================================
# SERIAL READER THREAD
# =========================================================
def serial_reader(port, baudrate=9600):

    try:
        ser = serial.Serial(port, baudrate, timeout=1)

        st.session_state.connected = True

        while st.session_state.serial_running:

            try:
                line = ser.readline().decode().strip()

                if line:

                    value = float(line)

                    timestamp = time.time()

                    data_queue.put((timestamp, value))

            except:
                pass

        ser.close()

    except:
        st.session_state.connected = False

# =========================================================
# SIGNAL FILTER
# =========================================================
def butter_lowpass_filter(data, cutoff=10, fs=100, order=3):

    nyq = 0.5 * fs

    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low')

    y = filtfilt(b, a, data)

    return y

# =========================================================
# FFT ANALYSIS
# =========================================================
def compute_fft(signal, fs=100):

    N = len(signal)

    yf = fft(signal)

    xf = fftfreq(N, 1/fs)

    positive = xf >= 0

    xf = xf[positive]

    yf = np.abs(yf[positive])

    dominant_freq = xf[np.argmax(yf)]

    return xf, yf, dominant_freq

# =========================================================
# AI PREDICTION
# =========================================================
def predict_tremor(freq, amplitude):

    if freq < 3:

        return "Normal", 15

    elif 3 <= freq < 5:

        return "Mild Tremor", 58

    else:

        return "Severe Tremor", 92

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="main-title">
    🧠 TENG BASED PARKINSON TREMOR DETECTION SYSTEM
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    IEEE-Level Realtime TENG Signal Analysis · FFT Spectrum · AI Tremor Detection
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header("⚙️ System Control")

    ports = detect_arduino()

    selected_port = st.selectbox(
        "COM Port",
        ports if ports else ["No Device"]
    )

    baudrate = st.selectbox(
        "Baud Rate",
        [9600, 115200],
        index=0
    )

    st.markdown("---")

    if not st.session_state.serial_running:

        if st.button("▶ START MONITORING"):

            if selected_port != "No Device":

                st.session_state.serial_running = True

                thread = threading.Thread(
                    target=serial_reader,
                    args=(selected_port, baudrate),
                    daemon=True
                )

                thread.start()

    else:

        if st.button("⏹ STOP MONITORING"):

            st.session_state.serial_running = False

   
