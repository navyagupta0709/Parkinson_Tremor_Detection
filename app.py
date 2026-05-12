"""
app.py
======
Parkinson Tremor Detection & Monitoring System
IEEE-Level Real-Time Healthcare IoT Dashboard

Run:
    streamlit run app.py
"""

import time
import threading
import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# OPTIONAL IMPORTS
# =========================================================
try:
    from streamlit_autorefresh import st_autorefresh
except:
    st_autorefresh = None

# =========================================================
# PROJECT IMPORTS
# =========================================================
from serial_reader import SerialReader, list_arduino_ports
from signal_processing import extract_features
from ml_model import load_model, predict
from utils import (
    DataLogger,
    AlertManager,
    severity_color,
    format_uptime,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Parkinson Tremor Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# LOAD CSS
# =========================================================
css_path = Path("assets/custom.css")

if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
def init_state():

    defaults = {
        "reader": None,
        "monitoring": False,
        "start_ts": time.time(),
        "waveform": deque(maxlen=500),
        "trend": deque(maxlen=120),
        "features": {},
        "prediction": {},
        "logs": deque(maxlen=50),
        "alerts": [],
        "samples": 0,
        "model": None,
        "scaler": None,
        "logger_obj": None,
        "alert_mgr": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def get_model():
    return load_model()

if st.session_state.model is None:
    model, scaler = get_model()

    st.session_state.model = model
    st.session_state.scaler = scaler

if st.session_state.logger_obj is None:
    st.session_state.logger_obj = DataLogger()

if st.session_state.alert_mgr is None:
    st.session_state.alert_mgr = AlertManager()

# =========================================================
# LOG FUNCTION
# =========================================================
def add_log(msg):

    ts = datetime.now().strftime("%H:%M:%S")

    st.session_state.logs.appendleft(
        f"[{ts}] {msg}"
    )

# =========================================================
# MONITORING LOOP
# =========================================================
def monitoring_loop():

    reader = st.session_state.reader

    while st.session_state.monitoring:

        snapshot = reader.get_snapshot()

        if len(snapshot) < 5:
            time.sleep(0.05)
            continue

        voltages = np.array([v[1] for v in snapshot])

        for t, v in snapshot[-20:]:
            st.session_state.waveform.append((t, v))

        st.session_state.samples = reader.samples_read

        # =================================================
        # FEATURE EXTRACTION
        # =================================================
        feats = extract_features(voltages)

        st.session_state.features = feats

        # =================================================
        # ML PREDICTION
        # =================================================
        pred = predict(
            feats,
            st.session_state.model,
            st.session_state.scaler
        )

        st.session_state.prediction = pred

        # =================================================
        # TREND
        # =================================================
        now = time.time() - st.session_state.start_ts

        st.session_state.trend.append(
            (now, pred["severity_pct"])
        )

        # =================================================
        # ALERTS
        # =================================================
        alert = st.session_state.alert_mgr.evaluate(pred)

        if alert:

            st.session_state.alerts = (
                st.session_state.alert_mgr.get_history()
            )

            add_log(
                f"ALERT: {alert['label']} "
                f"({alert['severity']}%)"
            )

        # =================================================
        # CSV LOGGING
        # =================================================
        st.session_state.logger_obj.log({
            "timestamp": datetime.now(),
            "voltage": float(voltages[-1]),
            "frequency": feats["dom_freq_hz"],
            "severity": pred["severity_pct"],
            "prediction": pred["label"]
        })

        time.sleep(0.1)

# =========================================================
# CHARTS
# =========================================================
def waveform_chart():

    buf = list(st.session_state.waveform)

    fig = go.Figure()

    if not buf:
        fig.update_layout(
            title="Waiting for Sensor Data..."
        )
        return fig

    x = [b[0] for b in buf]
    y = [b[1] for b in buf]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=2)
        )
    )

    fig.update_layout(
        title="Live TENG Waveform",
        template="plotly_dark",
        height=350
    )

    return fig


def trend_chart():

    buf = list(st.session_state.trend)

    fig = go.Figure()

    if not buf:
        return fig

    x = [b[0] for b in buf]
    y = [b[1] for b in buf]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers"
        )
    )

    fig.update_layout(
        title="Tremor Severity Trend",
        template="plotly_dark",
        yaxis_range=[0, 100],
        height=350
    )

    return fig

# =========================================================
# ALERT BANNER
# =========================================================
def render_alert():

    pred = st.session_state.prediction

    if not pred:
        st.info("Waiting for monitoring...")
        return

    label = pred.get("label", "Normal")

    sev = pred.get("severity_pct", 0)

    if label == "Severe Tremor":

        st.markdown(
            f"""
            <div style="
                background:#ff1744;
                padding:18px;
                border-radius:10px;
                color:white;
                font-size:24px;
                text-align:center;
                font-weight:bold;
            ">
            🚨 SEVERE PARKINSON TREMOR DETECTED
            <br>
            Severity: {sev:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    elif label == "Mild Tremor":

        st.warning(
            f"⚠ Mild Tremor Detected | Severity: {sev:.1f}%"
        )

    else:

        st.success(
            f"✅ Normal | Severity: {sev:.1f}%"
        )

# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar():

    with st.sidebar:

        st.markdown(
            """
            ## 🧠 Tremor Monitor

            IEEE Healthcare IoT System
            """
        )

        st.markdown("---")

        # =============================================
        # PORT DETECTION
        # =============================================
        st.subheader("⚙️ Serial Connection")

        auto_ports = list_arduino_ports()

        port_opts = (
            list(auto_ports)
            if auto_ports else []
        )

        port_opts.extend([
            "Manual…",
            "Demo Mode"
        ])

        default_index = 0

        if not auto_ports:
            default_index = len(port_opts) - 1

        selected_port = st.selectbox(
            "COM Port",
            port_opts,
            index=default_index
        )

        # =============================================
        # START / STOP
        # =============================================
        st.markdown("")

        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "▶ START",
                use_container_width=True,
                disabled=st.session_state.monitoring
            ):

                reader = SerialReader(
                    port=selected_port
                )

                reader.start()

                st.session_state.reader = reader

                st.session_state.monitoring = True

                thread = threading.Thread(
                    target=monitoring_loop,
                    daemon=True
                )

                thread.start()

                add_log("Monitoring Started")

        with c2:

            if st.button(
                "⏹ STOP",
                use_container_width=True,
                disabled=not st.session_state.monitoring
            ):

                st.session_state.monitoring = False

                if st.session_state.reader:
                    st.session_state.reader.stop()

                add_log("Monitoring Stopped")

        st.markdown("---")

        st.metric(
            "Samples",
            st.session_state.samples
        )

        st.metric(
            "Uptime",
            format_uptime(
                st.session_state.start_ts
            )
        )

# =========================================================
# MAIN
# =========================================================
def main():

    render_sidebar()

    # =====================================================
    # TITLE
    # =====================================================
    st.markdown(
        """
        # 🧠 Parkinson Tremor Detection System

        ### IEEE-Level Realtime Healthcare IoT Dashboard
        """
    )

    st.markdown("---")

    # =====================================================
    # ALERT
    # =====================================================
    render_alert()

    # =====================================================
    # METRICS
    # =====================================================
    pred = st.session_state.prediction
    feats = st.session_state.features

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Voltage",
        f"{feats.get('mean', 0):.3f} V"
    )

    c2.metric(
        "Frequency",
        f"{feats.get('dom_freq_hz', 0):.2f} Hz"
    )

    c3.metric(
        "Severity",
        f"{pred.get('severity_pct', 0):.1f}%"
    )

    c4.metric(
        "Prediction",
        pred.get("label", "—")
    )

    st.markdown("")

    # =====================================================
    # CHARTS
    # =====================================================
    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            waveform_chart(),
            use_container_width=True
        )

    with right:
        st.plotly_chart(
            trend_chart(),
            use_container_width=True
        )

    # =====================================================
    # ALERT HISTORY
    # =====================================================
    st.subheader("🚨 Alert History")

    if st.session_state.alerts:

        df = pd.DataFrame(
            st.session_state.alerts
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    else:
        st.info("No alerts yet")

    # =====================================================
    # LIVE LOGS
    # =====================================================
    st.subheader("📋 Live Logs")

    for line in st.session_state.logs:
        st.code(line)

    # =====================================================
    # AUTO REFRESH
    # =====================================================
    if st.session_state.monitoring:

        if st_autorefresh:
            st_autorefresh(
                interval=1000,
                key="refresh"
            )

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()
