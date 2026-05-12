# ================================
# app.py
# REAL LIVE TENG FFT DASHBOARD
# ================================

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import paho.mqtt.client as mqtt
import json
from collections import deque
import time

# =========================================
# PAGE
# =========================================

st.set_page_config(
    page_title="TENG Tremor Detection",
    page_icon="⚡",
    layout="wide"
)

# =========================================
# SESSION STATE
# =========================================

if "times" not in st.session_state:
    st.session_state.times = deque(maxlen=500)

if "voltages" not in st.session_state:
    st.session_state.voltages = deque(maxlen=500)

if "freq" not in st.session_state:
    st.session_state.freq = 0.0

if "power" not in st.session_state:
    st.session_state.power = 0.0

if "tremor" not in st.session_state:
    st.session_state.tremor = False

if "connected" not in st.session_state:
    st.session_state.connected = False

# =========================================
# MQTT
# =========================================

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "teng/live/navya"

# =========================================
# CALLBACK
# =========================================

def on_message(client, userdata, msg):

    try:

        d = json.loads(
            msg.payload.decode()
        )

        st.session_state.times.append(
            d["time"]
        )

        st.session_state.voltages.append(
            d["voltage"]
        )

        st.session_state.freq = d["freq"]

        st.session_state.power = d["power"]

        st.session_state.tremor = d["tremor"]

        st.session_state.connected = True

    except:
        pass

# =========================================
# MQTT CLIENT
# =========================================

client = mqtt.Client()

client.on_message = on_message

try:

    client.connect(
        BROKER,
        PORT,
        60
    )

    client.subscribe(
        TOPIC
    )

    client.loop_start()

except:

    st.error(
        "MQTT connection failed"
    )

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
# ALERT SYSTEM
# =========================================

freq = st.session_state.freq

if 3 <= freq <= 7:

    st.markdown(f"""
    <div style="
        background:#7f1d1d;
        padding:30px;
        border-radius:15px;
        border:4px solid red;
        text-align:center;
        animation: blinker 1s linear infinite;
    ">

    <h1 style="color:white;">
    🚨 TREMOR DETECTED
    </h1>

    <h2 style="color:#fecaca;">
    Dominant Frequency : {freq:.2f} Hz
    </h2>

    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div style="
        background:#052e16;
        padding:25px;
        border-radius:15px;
        border:3px solid #22c55e;
        text-align:center;
    ">

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
    f"{st.session_state.power:.4f}"
)

c3.metric(
    "Status",
    "Tremor" if st.session_state.tremor else "Normal"
)

# =========================================
# LIVE SIGNAL GRAPH
# =========================================

st.subheader(
    "📈 Live TENG Signal"
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=list(st.session_state.times),
        y=list(st.session_state.voltages),
        mode="lines",
        line=dict(color="cyan"),
        name="Voltage"
    )
)

fig.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="Time (s)",
    yaxis_title="Voltage (V)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================
# AUTO REFRESH
# =========================================

time.sleep(1)

st.rerun()
