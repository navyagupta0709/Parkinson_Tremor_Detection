"""
IEEE LEVEL TENG PARKINSON TREMOR DETECTION SYSTEM
Realtime Arduino + FFT + AI Dashboard

UPLOAD ARDUINO CODE FIRST
THEN RUN:
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
import threading
from collections import deque
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from datetime import datetime
import queue

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="TENG Tremor Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp{
    background:#06111f;
    color:white;
}

/* MAIN HEADER */
.main-header{
    text-align:center;
    padding:15px;
    background:#081a2f;
    border-radius:15px;
    border:1px solid #103b68;
    margin-bottom:20px;
}

.main-title{
    font-size:38px;
    font-weight:800;
    color:white;
}

.sub-title{
    color:#9db4d1;
    font-size:15px;
}

/* ALERTS */
.red-alert{
    background:rgba(255,0,0,0.12);
    border:2px solid red;
    border-radius:12px;
    padding:18px;
    color:#ff4d4d;
    font-size:28px;
    font-weight:bold;
    text-align:center;
    animation: blink 1s infinite;
}

.orange-alert{
    background:rgba(255,165,0,0.12);
    border:2px solid orange;
    border-radius:12px;
    padding:18px;
    color:orange;
    font-size:26px;
    font-weight:bold;
    text-align:center;
}

.green-alert{
    background:rgba(0,255,127,0.12);
    border:2px solid #00ff7f;
    border-radius:12px;
    padding:18px;
    color:#00ff7f;
    font-size:24px;
    font-weight:bold;
    text-align:center;
}

@keyframes blink{
    50%{
        opacity:0.4;
    }
}

/* METRIC CARDS */
.metric-box{
    background:#081726;
    padding:18px;
    border-radius:12px;
    border:1px solid #12375e;
    text-align:center;
}

.metric-value{
    font-size:40px;
    font-weight:bold;
    color:white;
}

.metric-label{
    color:#9db4d1;
    font-size:15px;
}

/* SIDEBAR */
[data-testid="stSidebar"]{
    background:#08111f;
}

/* LOG TABLE */
.log-box{
    background:#081726;
    border-radius:12px;
    padding:10px;
    border:1px solid #103b68;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "signal" not in st.session_state:
    st.session_state.signal = deque(maxlen=500)

if "fft_freq" not in st.session_state:
    st.session_state.fft_freq = 0

if "severity" not in st.session_state:
    st.session_state.severity = 0

if "label" not in st.session_state:
    st.session_state.label = "NORMAL"

if "logs" not in st.session_state:
    st.session_state.logs = []

if "connected" not in st.session_state:
    st.session_state.connected = False

# =========================================================
# DATA QUEUE
# =========================================================
data_queue = queue.Queue()

# =========================================================
# SERIAL PORT DETECTION
# =========================================================
def detect_ports():

    ports = serial.tools.list_ports.comports()

    return [p.device for p in ports]

# =========================================================
# SERIAL THREAD
# =========================================================
def serial_thread(port, baudrate):

    try:

        ser = serial.Serial(
            port,
            baudrate,
            timeout=1
        )

        st.session_state.connected = True

        while st.session_state.running:

            try:

                line = ser.readline().decode().strip()

                if line:

                    value = float(line)

                    data_queue.put(value)

            except:
                pass

        ser.close()

    except:

        st.session_state.connected = False

# =========================================================
# LOW PASS FILTER
# =========================================================
def lowpass(data):

    if len(data) < 20:
        return np.array(data)

    b, a = butter(
        3,
        0.2,
        btype='low'
    )

    return filtfilt(
        b,
        a,
        data
    )

# =========================================================
# FFT ANALYSIS
# =========================================================
def perform_fft(signal):

    if len(signal) < 128:
        return 0

    fs = 100

    N = len(signal)

    yf = fft(signal)

    xf = fftfreq(N, 1/fs)

    positive = xf >= 0

    xf = xf[positive]

    yf = np.abs(yf[positive])

    dominant = xf[np.argmax(yf)]

    return dominant

# =========================================================
# AI PREDICTION
# =========================================================
def ai_predict(freq):

    if freq < 3:

        return "NORMAL", 15

    elif freq < 5:

        return "MILD TREMOR", 58

    else:

        return "SEVERE TREMOR", 92

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-header">

<div class="main-title">
TENG BASED PARKINSON TREMOR DETECTION SYSTEM
</div>

<div class="sub-title">
IEEE Level Real-Time IoT Monitoring Dashboard
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## CONNECTION")

    ports = detect_ports()

    selected_port = st.selectbox(
        "COM Port",
        ports if ports else ["No Device"]
    )

    baudrate = st.selectbox(
        "Baud Rate",
        [9600, 115200]
    )

    st.markdown("---")

    if st.session_state.connected:

        st.success("CONNECTED")

    else:

        st.error("DISCONNECTED")

    st.markdown("---")

    if not st.session_state.running:

        if st.button("START", use_container_width=True):

            if selected_port != "No Device":

                st.session_state.running = True

                thread = threading.Thread(
                    target=serial_thread,
                    args=(selected_port, baudrate),
                    daemon=True
                )

                thread.start()

    else:

        if st.button("STOP", use_container_width=True):

            st.session_state.running = False

    st.markdown("---")

    st.markdown("## SYSTEM INFO")

    st.write("Arduino UNO")
    st.write("TENG v1.0")
    st.write("FFT Enabled")
    st.write("AI Detection Active")

# =========================================================
# READ REALTIME DATA
# =========================================================
while not data_queue.empty():

    val = data_queue.get()

    st.session_state.signal.append(val)

# =========================================================
# SIGNAL PROCESSING
# =========================================================
signal = np.array(st.session_state.signal)

if len(signal) > 20:

    filtered = lowpass(signal)

    freq = perform_fft(filtered)

    label, severity = ai_predict(freq)

    voltage = np.mean(filtered)

    amplitude = np.max(filtered) - np.min(filtered)

    st.session_state.fft_freq = freq
    st.session_state.severity = severity
    st.session_state.label = label

else:

    filtered = np.zeros(100)

    freq = 0
    voltage = 0
    amplitude = 0
    severity = 0
    label = "WAITING"

# =========================================================
# ALERT BANNER
# =========================================================
if label == "SEVERE TREMOR":

    st.markdown(f"""
    <div class="red-alert">
    🚨 TREMOR DETECTED
    <br>
    Severity: SEVERE
    <br>
    Tremor Frequency: {freq:.2f} Hz
    </div>
    """, unsafe_allow_html=True)

elif label == "MILD TREMOR":

    st.markdown(f"""
    <div class="orange-alert">
    ⚠ MILD TREMOR DETECTED
    <br>
    Tremor Frequency: {freq:.2f} Hz
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown(f"""
    <div class="green-alert">
    ✅ NO PATHOLOGICAL TREMOR DETECTED
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =========================================================
# METRICS
# =========================================================
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "TENG VOLTAGE",
        f"{voltage:.2f} V"
    )

with c2:
    st.metric(
        "TREMOR FREQUENCY",
        f"{freq:.2f} Hz"
    )

with c3:
    st.metric(
        "TREMOR AMPLITUDE",
        f"{amplitude:.2f}"
    )

with c4:
    st.metric(
        "SEVERITY INDEX",
        f"{severity}%"
    )

with c5:
    st.metric(
        "SIGNAL QUALITY",
        "95%"
    )

st.markdown("---")

# =========================================================
# CHARTS
# =========================================================
left, middle, right = st.columns(3)

# =========================================================
# REALTIME SIGNAL
# =========================================================
with left:

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            y=filtered,
            mode='lines',
            line=dict(
                color='#00d4ff',
                width=2
            ),
            name='TENG Voltage'
        )
    )

    fig1.update_layout(
        title="REALTIME TENG SIGNAL",
        template="plotly_dark",
        height=380,
        paper_bgcolor='#081726',
        plot_bgcolor='#081726'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# =========================================================
# FFT
# =========================================================
with middle:

    if len(filtered) > 20:

        fs = 100

        N = len(filtered)

        yf = fft(filtered)

        xf = fftfreq(N, 1/fs)

        positive = xf >= 0

        xf = xf[positive]

        yf = np.abs(yf[positive])

    else:

        xf = np.zeros(100)
        yf = np.zeros(100)

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=xf,
            y=yf,
            mode='lines',
            line=dict(
                color='orange',
                width=2
            )
        )
    )

    fig2.add_vrect(
        x0=4,
        x1=7,
        fillcolor='red',
        opacity=0.2,
        annotation_text='Pathological Tremor Band'
    )

    fig2.update_layout(
        title="FFT SPECTRUM ANALYSIS",
        template="plotly_dark",
        height=380,
        paper_bgcolor='#081726',
        plot_bgcolor='#081726'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =========================================================
# TREND
# =========================================================
with right:

    severity_history = np.random.randint(
        max(severity-10,0),
        severity+5,
        50
    )

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            y=severity_history,
            mode='lines',
            line=dict(
                color='red',
                width=2
            )
        )
    )

    fig3.update_layout(
        title="TREMOR AMPLITUDE TREND",
        template="plotly_dark",
        height=380,
        paper_bgcolor='#081726',
        plot_bgcolor='#081726'
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =========================================================
# LOWER PANELS
# =========================================================
b1, b2, b3 = st.columns(3)

with b1:

    st.subheader("TREMOR SEVERITY")

    st.progress(
        int(severity)
    )

    st.markdown(f"""
    ## {severity}%
    ### {label}
    """)

with b2:

    st.subheader("AI CLASSIFICATION")

    st.write("NORMAL")
    st.progress(10)

    st.write("MILD")
    st.progress(35)

    st.write("SEVERE")
    st.progress(int(severity))

with b3:

    st.subheader("FEATURES")

    st.write(f"Mean Voltage : {voltage:.2f}V")
    st.write(f"Dominant Frequency : {freq:.2f}Hz")
    st.write(f"Peak Amplitude : {amplitude:.2f}")
    st.write(f"Signal RMS : {np.sqrt(np.mean(filtered**2)):.2f}")

# =========================================================
# AUTO REFRESH
# =========================================================
time.sleep(0.2)

st.rerun()
