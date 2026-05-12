"""
IEEE-Level TENG Based Parkinson Tremor Detection Dashboard
Realtime Streamlit IoT + FFT + AI Monitoring System

Run:
streamlit run app.py
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import datetime
from collections import deque
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import random

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="TENG Parkinson Tremor Monitor",
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

.main-title {
    font-size: 34px;
    font-weight: 700;
    color: #00d4ff;
    margin-bottom: 0px;
}

.sub-title {
    font-size: 14px;
    color: #9db4d1;
    margin-top: 0px;
}

.metric-card {
    background: #0f1c2e;
    border: 1px solid #1c3557;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.alert-normal {
    background: rgba(0,255,127,0.15);
    border-left: 5px solid #00ff7f;
    padding: 16px;
    border-radius: 10px;
    color: #00ff7f;
    font-weight: bold;
}

.alert-mild {
    background: rgba(255,165,0,0.15);
    border-left: 5px solid orange;
    padding: 16px;
    border-radius: 10px;
    color: orange;
    font-weight: bold;
}

.alert-severe {
    background: rgba(255,0,0,0.18);
    border-left: 5px solid red;
    padding: 18px;
    border-radius: 10px;
    color: #ff4d4d;
    font-weight: bold;
    animation: blink 1s infinite;
}

@keyframes blink {
    50% {
        opacity: 0.5;
    }
}

.sidebar .sidebar-content {
    background-color: #081421;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=500)

if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🧠 TENG Tremor Monitor")

    st.markdown("""
    IEEE-Level Parkinson Tremor Detection System
    
    - TENG Sensor
    - FFT Signal Processing
    - AI Tremor Detection
    - Realtime IoT Monitoring
    """)

    st.markdown("---")

    monitoring = st.toggle(
        "▶ Enable Monitoring",
        value=False
    )

    severity_mode = st.selectbox(
        "Simulation Mode",
        [
            "Normal",
            "Mild Tremor",
            "Severe Tremor"
        ]
    )

    refresh_rate = st.slider(
        "Refresh Rate (sec)",
        1,
        5,
        1
    )

    st.markdown("---")

    st.markdown("### 🚨 Alert Threshold")

    tremor_threshold = st.slider(
        "Tremor Frequency Threshold",
        3.0,
        8.0,
        4.0
    )

    st.markdown("---")

    st.markdown("### 📡 System Status")

    if monitoring:
        st.success("🟢 TENG Device Connected")
    else:
        st.error("🔴 Device Offline")

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="main-title">
    🧠 TENG-Based Parkinson Tremor Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Realtime TENG Signal Processing · FFT Spectrum Analysis · AI Tremor Detection · IEEE Healthcare IoT Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# SIGNAL GENERATION
# =========================================================
def generate_teng_signal(mode):

    fs = 100
    t = np.linspace(0, 5, fs * 5)

    if mode == "Normal":
        freq = 1.5
        amp = 0.15

    elif mode == "Mild Tremor":
        freq = 4.5
        amp = 0.55

    else:
        freq = 6.2
        amp = 1.1

    signal = (
        amp * np.sin(2 * np.pi * freq * t)
        + 0.1 * np.random.randn(len(t))
    )

    voltage = np.clip(
        2.5 + signal,
        0,
        5
    )

    return t, voltage, freq, amp

# =========================================================
# FFT ANALYSIS
# =========================================================
def compute_fft(signal, fs=100):

    N = len(signal)

    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)

    pos_mask = xf >= 0

    xf = xf[pos_mask]
    yf = np.abs(yf[pos_mask])

    dominant_freq = xf[np.argmax(yf)]

    return xf, yf, dominant_freq

# =========================================================
# AI CLASSIFICATION
# =========================================================
def classify_tremor(freq):

    if freq < 3:
        return "Normal", 15

    elif freq < 5:
        return "Mild Tremor", 55

    else:
        return "Severe Tremor", 92

# =========================================================
# GENERATE DATA
# =========================================================
if monitoring:

    t, voltage, freq, amp = generate_teng_signal(
        severity_mode
    )

    xf, yf, dom_freq = compute_fft(voltage)

    label, severity = classify_tremor(dom_freq)

    signal_quality = random.randint(88, 99)

    current_data = {
        "timestamp": datetime.datetime.now(),
        "frequency": dom_freq,
        "amplitude": amp,
        "severity": severity,
        "signal_quality": signal_quality,
        "label": label
    }

    st.session_state.history.append(current_data)

else:

    voltage = np.zeros(500)
    t = np.linspace(0, 5, 500)

    xf = np.zeros(500)
    yf = np.zeros(500)

    dom_freq = 0
    amp = 0
    severity = 0
    signal_quality = 0
    label = "Offline"

# =========================================================
# ALERT BANNER
# =========================================================
if label == "Normal":

    st.markdown(
        f"""
        <div class="alert-normal">
        ✅ NO PATHOLOGICAL TREMOR DETECTED
        <br><br>
        Tremor Frequency: {dom_freq:.2f} Hz
        </div>
        """,
        unsafe_allow_html=True
    )

elif label == "Mild Tremor":

    st.markdown(
        f"""
        <div class="alert-mild">
        ⚠ MILD PARKINSON TREMOR DETECTED
        <br><br>
        Tremor Frequency: {dom_freq:.2f} Hz
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        f"""
        <div class="alert-severe">
        🚨 SEVERE PARKINSON TREMOR DETECTED
        <br><br>
        Tremor Frequency: {dom_freq:.2f} Hz
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("")

# =========================================================
# METRICS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "⚡ TENG Voltage",
        f"{np.mean(voltage):.2f} V"
    )

with c2:
    st.metric(
        "〰 Tremor Frequency",
        f"{dom_freq:.2f} Hz"
    )

with c3:
    st.metric(
        "📳 Tremor Amplitude",
        f"{amp:.2f} g"
    )

with c4:
    st.metric(
        "🧠 AI Severity",
        f"{severity}%"
    )

st.markdown("---")

# =========================================================
# CHARTS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📈 TENG Waveform",
    "🔬 FFT Spectrum",
    "📊 Tremor Analytics"
])

# =========================================================
# WAVEFORM
# =========================================================
with tab1:

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=t,
            y=voltage,
            mode='lines',
            name='TENG Voltage',
            line=dict(
                color='#00d4ff',
                width=2
            )
        )
    )

    fig1.update_layout(
        title="Realtime TENG Voltage Waveform",
        template="plotly_dark",
        height=420,
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# =========================================================
# FFT
# =========================================================
with tab2:

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=xf,
            y=yf,
            mode='lines',
            name='FFT Spectrum',
            line=dict(
                color='orange',
                width=2
            )
        )
    )

    fig2.add_vrect(
        x0=4,
        x1=7,
        fillcolor="red",
        opacity=0.15,
        annotation_text="Pathological Tremor Band"
    )

    fig2.add_vline(
        x=dom_freq,
        line_dash="dash",
        line_color="red"
    )

    fig2.update_layout(
        title="TENG FFT Spectrum — Parkinson Tremor Frequency Analysis",
        template="plotly_dark",
        height=420,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =========================================================
# ANALYTICS
# =========================================================
with tab3:

    if len(st.session_state.history) > 0:

        df = pd.DataFrame(
            list(st.session_state.history)
        )

        fig3 = px.line(
            df,
            y="severity",
            title="Realtime Tremor Severity Trend"
        )

        fig3.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.dataframe(
            df.tail(20),
            use_container_width=True
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    """
    <center>

    IEEE-Level Parkinson Tremor Detection Platform

    TENG Wearable Sensor · FFT Analysis · AI Classification · Realtime Monitoring

    </center>
    """,
    unsafe_allow_html=True
)

# =========================================================
# AUTO REFRESH
# =========================================================
if monitoring:

    time.sleep(refresh_rate)

    st.rerun()
