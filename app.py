# =========================
# app.py
# =========================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import paho.mqtt.client as mqtt
import json
import time
from collections import deque

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="TENG Tremor Detection",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "voltages" not in st.session_state:
    st.session_state.voltages = deque(maxlen=500)

if "times" not in st.session_state:
    st.session_state.times = deque(maxlen=500)

if "dominant_freq" not in st.session_state:
    st.session_state.dominant_freq = 0.0

if "band_power" not in st.session_state:
    st.session_state.band_power = 0.0

if "tremor_active" not in st.session_state:
    st.session_state.tremor_active = False

if "severity" not in st.session_state:
    st.session_state.severity = "Normal"

# ---------------------------------------------------
# MQTT CONFIG
# ---------------------------------------------------

BROKER = "broker.hivemq.com"
PORT = 1883
MQTT_TOPIC = "tremorwatch/data/navya"

# ---------------------------------------------------
# MQTT CALLBACK
# ---------------------------------------------------

def on_message(client, userdata, msg):

    try:
        d = json.loads(msg.payload.decode())

    except:
        return

    st.session_state.voltages.append(
        d.get("voltage", 0.0)
    )

    st.session_state.times.append(
        d.get("time", 0.0)
    )

    st.session_state.dominant_freq = d.get(
        "freq",
        0.0
    )

    st.session_state.band_power = d.get(
        "power",
        0.0
    )

    st.session_state.tremor_active = d.get(
        "tremor",
        False
    )

    freq = st.session_state.dominant_freq

    if 3 <= freq < 4:
        st.session_state.severity = "Mild"

    elif 4 <= freq < 5.5:
        st.session_state.severity = "Moderate"

    elif 5.5 <= freq <= 7:
        st.session_state.severity = "Severe"

    else:
        st.session_state.severity = "Normal"

# ---------------------------------------------------
# MQTT CLIENT
# ---------------------------------------------------

client = mqtt.Client()

client.on_message = on_message

try:

    client.connect(
        BROKER,
        PORT,
        60
    )

    client.subscribe(MQTT_TOPIC)

    client.loop_start()

except:

    st.error("MQTT Connection Failed")

# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title(
    "⚡ TENG Parkinson Tremor Detection"
)

st.caption(
    "Real-Time FFT Spectrum Analysis"
)

# ---------------------------------------------------
# ALERT
# ---------------------------------------------------

freq = st.session_state.dominant_freq

if 3 <= freq <= 7:

    st.markdown(f"""
    <div style="
        background:#7f1d1d;
        padding:25px;
        border-radius:15px;
        border:3px solid red;
        text-align:center;
    ">

    <h1 style="color:white;">
    🚨 TREMOR DETECTED
    </h1>

    <h2 style="color:#fecaca;">
    {freq:.2f} Hz
    </h2>

    <h3 style="color:#fca5a5;">
    Severity : {st.session_state.severity}
    </h3>

    </div>
    """, unsafe_allow_html=True)

else:

    st.success(
        "✅ Signal Normal — No Tremor Detected"
    )

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

c1, c2, c3 = st.columns(3)

c1.metric(
    "Dominant Frequency",
    f"{freq:.2f} Hz"
)

c2.metric(
    "Band Power",
    f"{st.session_state.band_power:.4f}"
)

c3.metric(
    "Status",
    st.session_state.severity
)

# ---------------------------------------------------
# LIVE SIGNAL GRAPH
# ---------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=list(st.session_state.times),
        y=list(st.session_state.voltages),
        mode="lines",
        name="TENG Signal"
    )
)

fig.update_layout(
    template="plotly_dark",
    title="Live Voltage Waveform",
    xaxis_title="Time (s)",
    yaxis_title="Voltage (V)",
    height=400
)

st.plotly_chart(
    fig,
    use_container_width=True
)

time.sleep(1)

st.rerun()
