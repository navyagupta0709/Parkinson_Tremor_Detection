"""
REALTIME IEEE LEVEL TENG PARKINSON TREMOR DETECTION SYSTEM
LIVE Arduino + FFT + AI Dashboard

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
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from collections import deque
import threading
import queue
import time
from datetime import datetime
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="TENG Tremor Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

.stApp{
    background:#07111f;
    color:white;
}

.main-title{
    text-align:center;
    font-size:38px;
    font-weight:800;
    color:white;
}

.sub-title{
    text-align:center;
    color:#8ea5c0;
    font-size:15px;
    margin-bottom:20px;
}

.red-alert{
    background:rgba(255,0,0,0.12);
    border:2px solid red;
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:#ff4d4d;
    font-size:28px;
    font-weight:bold;
    animation:blink 1s infinite;
}

.orange-alert{
    background:rgba(255,165,0,0.12);
    border:2px solid orange;
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:orange;
    font-size:25px;
    font-weight:bold;
}

.green-alert{
    background:rgba(0,255,127,0.12);
    border:2px solid #00ff7f;
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:#00ff7f;
    font-size:22px;
    font-weight:bold;
}

@keyframes blink{
    50%{
        opacity:0.4;
    }
}

[data-testid="stSidebar"]{
    background:#081726;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "connected" not in st.session_state:
    st.session_state.connected = False

if "signal_buffer" not in st.session_state:
    st.session_state.signal_buffer = deque(maxlen=512)

if "freq_history" not in st.session_state:
    st.session_state.freq_history = deque(maxlen=100)

if "severity_history" not in st.session_state:
    st.session_state.severity_history = deque(maxlen=100)

# =========================================================
# SERIAL QUEUE
# =========================================================
serial_queue = queue.Queue()

# =========================================================
# DETECT PORTS
# =========================================================
def detect_ports():

    ports = serial.tools.list_ports.comports()

    return [p.device for p in ports]

# =========================================================
# SERIAL THREAD
# =========================================================
def serial_reader(port, baudrate):

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

                    serial_queue.put(value)

            except:
                pass

        ser.close()

    except:

        st.session_state.connected = False

# =========================================================
# FILTER
# =========================================================
def lowpass_filter(data):

    if len(data) < 20:
        return np.array(data)

    b, a = butter(
        3,
        0.15,
        btype='low'
    )

    filtered = filtfilt(
        b,
        a,
        data
    )

    return filtered

# =========================================================
# FFT
# =========================================================
def calculate_fft(signal):

    if len(signal) < 128:
        return 0, [], []

    fs = 100

    N = len(signal)

    yf = fft(signal)

    xf = fftfreq(N, 1/fs)

    positive = xf >= 0

    xf = xf[positive]

    yf = np.abs(yf[positive])

    dominant = xf[np.argmax(yf)]

    return dominant, xf, yf

# =========================================================
# AI DETECTION
# =========================================================
def detect_tremor(freq):

    if freq < 3:

        return "NORMAL", 12

    elif freq < 5:

        return "MILD TREMOR", 55

    else:

        return "SEVERE TREMOR", 91

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-title">
🧠 TENG BASED PARKINSON TREMOR DETECTION SYSTEM
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
IEEE Level Realtime IoT Monitoring Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## CONNECTION")

    ports = detect_ports()

    selected_port = st.selectbox(
        "COM Port",
        ports if ports else ["NO DEVICE"]
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

        if st.button(
            "START",
            use_container_width=True
        ):

            if selected_port != "NO DEVICE":

                st.session_state.running = True

                thread = threading.Thread(
                    target=serial_reader,
                    args=(selected_port, baudrate),
                    daemon=True
                )

                thread.start()

    else:

        if st.button(
            "STOP",
            use_container_width=True
        ):

            st.session_state.running = False

# =========================================================
# REALTIME SERIAL DATA
# =========================================================
while not serial_queue.empty():

    val = serial_queue.get()

    st.session_state.signal_buffer.append(val)

# =========================================================
# SIGNAL PROCESSING
# =========================================================
signal = np.array(
    st.session_state.signal_buffer
)

if len(signal) > 128:

    filtered = lowpass_filter(signal)

    freq, xf, yf = calculate_fft(filtered)

    label, severity = detect_tremor(freq)

    voltage = np.mean(filtered)

    amplitude = np.max(filtered) - np.min(filtered)

    rms = np.sqrt(np.mean(filtered**2))

    st.session_state.freq_history.append(freq)

    st.session_state.severity_history.append(severity)

else:

    filtered = np.zeros(512)

    xf = np.zeros(512)

    yf = np.zeros(512)

    freq = 0
    severity = 0
    voltage = 0
    amplitude = 0
    rms = 0
    label = "WAITING"

# =========================================================
# ALERTS
# =========================================================
if label == "SEVERE TREMOR":

    st.markdown(f"""
    <div class="red-alert">
    🚨 TREMOR DETECTED
    <br>
    SEVERITY: SEVERE
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

    st.markdown("""
    <div class="green-alert">
    ✅ NO PATHOLOGICAL TREMOR DETECTED
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =========================================================
# METRICS
# =========================================================
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(
        "TENG VOLTAGE",
        f"{voltage:.2f} V"
    )

with m2:
    st.metric(
        "TREMOR FREQUENCY",
        f"{freq:.2f} Hz"
    )

with m3:
    st.metric(
        "AMPLITUDE",
        f"{amplitude:.2f}"
    )

with m4:
    st.metric(
        "SEVERITY INDEX",
        f"{severity}%"
    )

with m5:
    st.metric(
        "RMS SIGNAL",
        f"{rms:.2f}"
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
            name='TENG Signal'
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
# FFT SPECTRUM
# =========================================================
with middle:

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=xf,
            y=yf,
            mode='lines',
            line=dict(
                color='orange',
                width=2
            ),
            name='FFT'
        )
    )

    fig2.add_vrect(
        x0=4,
        x1=7,
        fillcolor='red',
        opacity=0.2,
        annotation_text='Pathological Tremor Band'
    )

    fig2.add_vline(
        x=freq,
        line_dash="dash",
        line_color="red"
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
# TREND GRAPH
# =========================================================
with right:

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            y=list(st.session_state.severity_history),
            mode='lines',
            line=dict(
                color='red',
                width=2
            ),
            name='Severity'
        )
    )

    fig3.update_layout(
        title="TREMOR SEVERITY TREND",
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

# =========================================================
# SEVERITY
# =========================================================
with b1:

    st.subheader("TREMOR SEVERITY")

    st.progress(int(severity))

    st.markdown(f"""
    ## {severity}%
    ### {label}
    """)

# =========================================================
# AI CLASSIFICATION
# =========================================================
with b2:

    st.subheader("AI CLASSIFICATION")

    st.write("NORMAL")
    st.progress(10)

    st.write("MILD TREMOR")
    st.progress(40)

    st.write("SEVERE TREMOR")
    st.progress(int(severity))

# =========================================================
# FEATURES
# =========================================================
with b3:

    st.subheader("FEATURES")

    st.write(f"Mean Voltage : {voltage:.2f}V")
    st.write(f"Dominant Frequency : {freq:.2f}Hz")
    st.write(f"Peak Amplitude : {amplitude:.2f}")
    st.write(f"RMS : {rms:.2f}")

# =========================================================
# LOG TABLE
# =========================================================
st.markdown("---")

st.subheader("ALERT LOG")

log_df = pd.DataFrame({
    "Time": [datetime.now().strftime("%H:%M:%S")],
    "Severity": [label],
    "Frequency": [round(freq,2)],
    "Amplitude": [round(amplitude,2)]
})

st.dataframe(
    log_df,
    use_container_width=True
)

# =========================================================
# AUTO REFRESH
# =========================================================
time.sleep(0.2)

st.rerun()
