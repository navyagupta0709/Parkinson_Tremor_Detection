"""
app.py
======
Parkinson Tremor Detection & Monitoring System
IEEE-Level Real-Time Healthcare IoT Dashboard

Run:  streamlit run app.py
"""

import time
import logging
import threading
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
from collections import deque

# ── Project imports ───────────────────────────────────────────────────────────
from serial_reader     import SerialReader, detect_port, list_arduino_ports
from signal_processing import (
    FS, WINDOW_SIZE,
    extract_features, bandpass_filter, smooth_signal,
)
from ml_model  import load_model, predict, CLASS_COLORS, CLASS_NAMES
from utils     import DataLogger, AlertManager, LatencyTracker, \
                      severity_color, format_uptime

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson Tremor Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ───────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent / "assets" / "custom.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "reader":           None,
        "model":            None,
        "scaler":           None,
        "logger":           None,
        "alert_mgr":        None,
        "latency":          None,
        "start_ts":         time.time(),
        "monitoring":       False,
        "waveform_buf":     deque(maxlen=500),  # (time, voltage)
        "trend_buf":        deque(maxlen=120),  # (time, severity_pct)
        "freq_buf":         deque(maxlen=120),  # (time, freq_hz)
        "last_features":    {},
        "last_prediction":  {},
        "alert_history":    [],
        "log_lines":        deque(maxlen=80),
        "total_samples":    0,
        "com_port":         None,
        "manual_port":      "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: LOG LINE
# ═════════════════════════════════════════════════════════════════════════════
def _log(msg: str, level: str = "info"):
    ts  = datetime.now().strftime("%H:%M:%S")
    tag = {"info": "", "warn": "⚠ ", "error": "✖ ", "ok": "✔ "}.get(level, "")
    st.session_state.log_lines.append((level, f"[{ts}] {tag}{msg}"))
    getattr(logger, level if level != "ok" else "info")(msg)


# ═════════════════════════════════════════════════════════════════════════════
# INITIALISE RESOURCES (once per session)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ML model…")
def _get_model():
    return load_model()


def _ensure_resources():
    if st.session_state.model is None:
        m, s = _get_model()
        st.session_state.model  = m
        st.session_state.scaler = s
        _log("ML model loaded (RandomForest + StandardScaler)", "ok")

    if st.session_state.logger is None:
        st.session_state.logger    = DataLogger()
        st.session_state.alert_mgr = AlertManager()
        st.session_state.latency   = LatencyTracker()
        _log("Data logger & alert manager ready", "ok")


_ensure_resources()


# ═════════════════════════════════════════════════════════════════════════════
# MONITORING LOOP (runs in background thread)
# ═════════════════════════════════════════════════════════════════════════════
def _monitoring_loop():
    """
    Background thread: reads serial buffer → processes signal →
    runs ML → updates session_state buffers.
    """
    reader  = st.session_state.reader
    model   = st.session_state.model
    scaler  = st.session_state.scaler
    dlogger = st.session_state.logger
    almgr   = st.session_state.alert_mgr
    lat     = st.session_state.latency

    while st.session_state.monitoring:
        lat.ping()
        snapshot = reader.get_snapshot()   # list of (t, v)

        if len(snapshot) < 10:
            time.sleep(0.05)
            continue

        times    = np.array([s[0] for s in snapshot])
        voltages = np.array([s[1] for s in snapshot])

        # Update waveform buffer
        for t, v in snapshot[-50:]:        # push latest 50 pts
            st.session_state.waveform_buf.append((t, v))
        st.session_state.total_samples = reader.samples_read

        # Feature extraction on latest window
        window = voltages[-WINDOW_SIZE:] if len(voltages) >= WINDOW_SIZE \
                 else voltages
        feats  = extract_features(window)
        st.session_state.last_features = feats

        # ML prediction
        pred = predict(feats, model, scaler)
        st.session_state.last_prediction = pred

        # Trend buffers
        now = time.time() - st.session_state.start_ts
        st.session_state.trend_buf.append((now, pred["severity_pct"]))
        st.session_state.freq_buf.append((now, feats["dom_freq_hz"]))

        # Alert
        alert = almgr.evaluate(pred)
        if alert:
            st.session_state.alert_history = almgr.get_history()
            _log(f"ALERT — {alert['label']} | {alert['severity']}% severity",
                 "error" if alert["is_severe"] else "warn")

        # CSV logging (every 10th cycle to avoid IO bottleneck)
        if st.session_state.total_samples % 10 == 0:
            dlogger.log({
                "elapsed_s":      round(now, 2),
                "voltage_V":      round(float(voltages[-1]), 4),
                "amplitude_V":    round(feats["amplitude"],  4),
                "dom_freq_hz":    round(feats["dom_freq_hz"],3),
                "band_power":     round(feats["band_power"],  4),
                "peak_count":     int(feats["peak_count"]),
                "signal_quality": round(feats["signal_quality"], 1),
                "prediction":     pred["label"],
                "confidence":     round(pred["confidence"],  4),
                "severity_pct":   round(pred["severity_pct"],2),
            })

        time.sleep(0.1)   # 10 Hz update cycle


# ═════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,22,50,0.9)",
    font=dict(family="Inter", color="#90a4ae", size=11),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)


def _waveform_chart():
    buf = list(st.session_state.waveform_buf)
    if not buf:
        fig = go.Figure()
        fig.update_layout(title="Waiting for sensor data…", **CHART_LAYOUT)
        return fig

    times = [b[0] for b in buf]
    volts = [b[1] for b in buf]

    # Smooth for display
    v_arr = np.array(volts)
    if len(v_arr) > 11:
        from scipy.signal import savgol_filter
        v_smooth = savgol_filter(v_arr, min(11, len(v_arr) | 1), 3)
    else:
        v_smooth = v_arr

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=volts,
        mode="lines",
        line=dict(color="#1565c0", width=1, dash="dot"),
        name="Raw",
        opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=times, y=v_smooth.tolist(),
        mode="lines",
        line=dict(color="#00b4d8", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,180,216,0.06)",
        name="Smoothed",
    ))
    fig.update_layout(
        title="🔬 Live TENG Voltage Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
        yaxis_range=[0, 5.2],
        **CHART_LAYOUT,
    )
    return fig


def _trend_chart():
    buf = list(st.session_state.trend_buf)
    if not buf:
        fig = go.Figure()
        fig.update_layout(title="Severity Trend — waiting…", **CHART_LAYOUT)
        return fig

    times  = [b[0] for b in buf]
    sevs   = [b[1] for b in buf]
    colors = [severity_color(s) for s in sevs]

    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,230,118,0.05)",  line_width=0)
    fig.add_hrect(y0=30, y1=70,  fillcolor="rgba(255,171,64,0.05)", line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,23,68,0.05)",  line_width=0)
    fig.add_trace(go.Scatter(
        x=times, y=sevs,
        mode="lines+markers",
        line=dict(color="#ffab40", width=2),
        marker=dict(color=colors, size=5),
        name="Severity %",
    ))
    fig.update_layout(
        title="📈 Tremor Severity Trend",
        xaxis_title="Elapsed (s)",
        yaxis_title="Severity (%)",
        yaxis_range=[0, 100],
        **CHART_LAYOUT,
    )
    return fig


def _freq_chart():
    buf = list(st.session_state.freq_buf)
    if not buf:
        fig = go.Figure()
        fig.update_layout(title="Frequency — waiting…", **CHART_LAYOUT)
        return fig

    times = [b[0] for b in buf]
    freqs = [b[1] for b in buf]

    fig = go.Figure()
    fig.add_hrect(y0=3, y1=7, fillcolor="rgba(255,23,68,0.08)",
                  annotation_text="Tremor Band (3–7 Hz)",
                  annotation_font=dict(color="#ff1744", size=10),
                  line_width=0)
    fig.add_trace(go.Scatter(
        x=times, y=freqs,
        mode="lines",
        line=dict(color="#e040fb", width=2),
        fill="tozeroy",
        fillcolor="rgba(224,64,251,0.06)",
        name="Frequency",
    ))
    fig.update_layout(
        title="🎵 Dominant Tremor Frequency",
        xaxis_title="Elapsed (s)",
        yaxis_title="Frequency (Hz)",
        yaxis_range=[0, 12],
        **CHART_LAYOUT,
    )
    return fig


def _confidence_gauge(confidence: float, label: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number=dict(suffix="%", font=dict(color=color, size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#90a4ae",
                      tickfont=dict(size=9)),
            bar=dict(color=color),
            bgcolor="rgba(13,22,50,0.8)",
            bordercolor="#1e2d4a",
            steps=[
                dict(range=[0,   40], color="rgba(0,0,0,0)"),
                dict(range=[40,  70], color="rgba(255,171,64,0.1)"),
                dict(range=[70, 100], color="rgba(255,23,68,0.1)"),
            ],
            threshold=dict(line=dict(color=color, width=3),
                           thickness=0.8, value=confidence * 100),
        ),
        title=dict(text=f"AI Confidence<br><b>{label}</b>",
                   font=dict(color="#90a4ae", size=12)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ALERT BANNER HTML
# ═════════════════════════════════════════════════════════════════════════════
def _render_alert_banner():
    pred = st.session_state.last_prediction
    if not pred:
        st.markdown(
            '<div class="alert-normal">⬤ &nbsp;SYSTEM INITIALISING — Connect Arduino to begin monitoring</div>',
            unsafe_allow_html=True,
        )
        return

    label    = pred.get("label",       "Normal")
    severity = pred.get("severity_pct", 0)
    conf     = pred.get("confidence",   0)

    if label == "Normal":
        html = (
            f'<div class="alert-normal">'
            f'✅ &nbsp;<b>NORMAL</b> &nbsp;|&nbsp; '
            f'Severity: {severity:.1f}% &nbsp;|&nbsp; '
            f'Confidence: {conf*100:.1f}%'
            f'</div>'
        )
    elif label == "Mild Tremor":
        html = (
            f'<div class="alert-mild">'
            f'⚠️ &nbsp;<b>MILD TREMOR DETECTED</b> &nbsp;|&nbsp; '
            f'Severity: {severity:.1f}% &nbsp;|&nbsp; '
            f'Confidence: {conf*100:.1f}%'
            f'</div>'
        )
    else:
        html = (
            f'<div class="alert-severe">'
            f'🚨 &nbsp;<b>SEVERE PARKINSON TREMOR DETECTED</b> &nbsp;|&nbsp; '
            f'Severity: {severity:.1f}% &nbsp;|&nbsp; '
            f'Confidence: {conf*100:.1f}%'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
def _render_sidebar():
    with st.sidebar:
        st.markdown(
            "## 🧠 Tremor Monitor\n"
            "**IEEE IoT Healthcare System**\n\n"
            "TENG Wearable · Arduino UNO · ML"
        )
        st.markdown("---")

        # ── Port detection ────────────────────────────────────────
        # ── Port detection ────────────────────────────────────────
st.subheader("⚙️ Serial Connection")

auto_ports = list_arduino_ports()
port_opts  = (auto_ports if auto_ports else []) + ["Manual…", "Demo Mode"]

default_index = 0

if not auto_ports and len(port_opts) > 1:
    default_index = len(port_opts) - 1

sel = st.selectbox(
    "COM Port",
    port_opts,
    index=default_index
)

        # ── Start / Stop ──────────────────────────────────────────

        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ START", use_container_width=True,
                         disabled=st.session_state.monitoring):
                _start_monitoring(sel)
        with col_b:
            if st.button("⏹ STOP", use_container_width=True,
                         disabled=not st.session_state.monitoring):
                _stop_monitoring()

        # ── Connection status ─────────────────────────────────────
        st.markdown("---")
        st.subheader("📡 Connection")
        if st.session_state.reader:
            status = st.session_state.reader.get_status()
            if status["connected"]:
                st.markdown(f'<span class="badge badge-green">● CONNECTED</span> &nbsp; `{status["port"]}`',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge badge-red">● DISCONNECTED</span>',
                            unsafe_allow_html=True)
                if status["error"]:
                    st.caption(status["error"])
        else:
            st.markdown('<span class="badge badge-gray">● IDLE</span>',
                        unsafe_allow_html=True)

        # ── Stats ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Session Stats")
        uptime = format_uptime(st.session_state.start_ts)
        lat_ms = st.session_state.latency.latency_ms() \
                 if st.session_state.latency else 0
        st.metric("Uptime",   uptime)
        st.metric("Samples",  f"{st.session_state.total_samples:,}")
        st.metric("Latency",  f"{lat_ms:.0f} ms")

        # ── Clear / Reset ─────────────────────────────────────────
        st.markdown("---")
        if st.button("🗑 Clear Logs & Alerts", use_container_width=True):
            _clear_session()

        # ── Patient info ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏥 Patient")
        st.text_input("Patient ID", value="PT-001", key="patient_id")
        st.text_input("Session",    value=datetime.now().strftime("%Y-%m-%d"),
                      key="session_date", disabled=True)

        # ── Thresholds ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🎛 Thresholds")
        st.slider("Mild Alert (%)",   10, 50, 30, key="thresh_mild")
        st.slider("Severe Alert (%)", 50, 95, 70, key="thresh_severe")


# ═════════════════════════════════════════════════════════════════════════════
# START / STOP MONITORING
# ═════════════════════════════════════════════════════════════════════════════
def _start_monitoring(port: str):
    demo = (port == "Demo Mode")

    if not demo:
        reader = SerialReader(port=port if (port and port != "Manual…") else None)
        reader.start()
        st.session_state.reader = reader
        _log(f"Serial reader started on {port or 'auto-detect'}", "ok")
    else:
        # Demo mode: inject synthetic data via DemoReader
        st.session_state.reader = DemoReader()
        st.session_state.reader.start()
        _log("DEMO MODE active — simulated TENG data", "warn")

    st.session_state.monitoring = True
    st.session_state.start_ts   = time.time()

    t = threading.Thread(target=_monitoring_loop, daemon=True, name="MonitorLoop")
    t.start()
    _log("Monitoring loop started", "ok")


def _stop_monitoring():
    st.session_state.monitoring = False
    if st.session_state.reader:
        st.session_state.reader.stop()
    _log("Monitoring stopped", "warn")


def _clear_session():
    st.session_state.waveform_buf.clear()
    st.session_state.trend_buf.clear()
    st.session_state.freq_buf.clear()
    st.session_state.log_lines.clear()
    if st.session_state.logger:
        st.session_state.logger.clear()
    if st.session_state.alert_mgr:
        st.session_state.alert_mgr.clear()
    st.session_state.alert_history = []
    st.session_state.total_samples = 0
    _log("Session cleared", "ok")


# ═════════════════════════════════════════════════════════════════════════════
# DEMO READER (synthetic TENG signal for testing without Arduino)
# ═════════════════════════════════════════════════════════════════════════════
class DemoReader:
    """Generates synthetic TENG-like signal for demo / testing."""

    def __init__(self):
        self._stop = threading.Event()
        self.buffer  = deque(maxlen=1000)
        self.connected    = True
        self.active_port  = "DEMO"
        self.error_msg    = ""
        self.samples_read = 0
        self.last_value   = 2.5
        self.last_time    = 0.0
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self._t.start()

    def stop(self):
        self._stop.set()

    def get_snapshot(self):
        return list(self.buffer)

    def get_status(self):
        return {
            "connected": True,
            "port":      "DEMO",
            "error":     "",
            "samples":   self.samples_read,
            "last_voltage": self.last_value,
            "last_time":    self.last_time,
        }

    def _run(self):
        start = time.time()
        phase = 0.0
        severity_phase = 0.0
        while not self._stop.is_set():
            t = time.time() - start
            severity_phase += 0.002

            # Cycle through Normal → Mild → Severe every ~60 s
            cycle = (severity_phase % 1.0)
            if cycle < 0.45:
                # Normal
                freq  = 1.2
                amp   = 0.08
                noise = 0.02
                dc    = 2.5
            elif cycle < 0.75:
                # Mild
                freq  = 4.0
                amp   = 0.35
                noise = 0.05
                dc    = 2.3
            else:
                # Severe
                freq  = 6.0
                amp   = 0.90
                noise = 0.12
                dc    = 2.1

            v = (dc
                 + amp * np.sin(2 * np.pi * freq * t + phase)
                 + amp * 0.3 * np.sin(2 * np.pi * freq * 2 * t)
                 + noise * np.random.randn())
            v = float(np.clip(v, 0.1, 4.9))

            self.buffer.append((t, v))
            self.last_value   = v
            self.last_time    = t
            self.samples_read += 1
            time.sleep(0.01)   # 100 Hz


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD RENDER
# ═════════════════════════════════════════════════════════════════════════════
def main():
    _render_sidebar()

    # ── Header bar ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <span style="font-size:2rem;">🧠</span>
          <div>
            <div style="font-size:1.4rem;font-weight:700;color:#00b4d8;letter-spacing:0.03em;">
              PARKINSON TREMOR DETECTION & MONITORING SYSTEM
            </div>
            <div style="font-size:0.8rem;color:#546e7a;letter-spacing:0.1em;">
              IEEE-LEVEL REAL-TIME HEALTHCARE IOT DASHBOARD &nbsp;·&nbsp;
              TENG WEARABLE SENSOR &nbsp;·&nbsp; ARDUINO UNO &nbsp;·&nbsp;
              MACHINE LEARNING
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── ALERT BANNER ──────────────────────────────────────────────────────────
    _render_alert_banner()

    # ── TOP METRICS ROW ───────────────────────────────────────────────────────
    pred  = st.session_state.last_prediction
    feats = st.session_state.last_features

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    voltage   = feats.get("mean",         0.0)
    amplitude = feats.get("amplitude",    0.0)
    dom_freq  = feats.get("dom_freq_hz",  0.0)
    sq        = feats.get("signal_quality", 0.0)
    severity  = pred.get("severity_pct",  0.0)
    label     = pred.get("label",         "—")

    m1.metric("⚡ Voltage (V)",    f"{voltage:.3f}")
    m2.metric("📶 Amplitude (V)", f"{amplitude:.3f}")
    m3.metric("🎵 Frequency (Hz)",f"{dom_freq:.2f}")
    m4.metric("📊 Signal Quality",f"{sq:.0f}%")
    m5.metric("🔥 Severity",      f"{severity:.1f}%")
    m6.metric("🤖 AI Label",      label)

    st.markdown("")

    # ── WAVEFORM + CONFIDENCE ─────────────────────────────────────────────────
    col_wave, col_gauge = st.columns([3, 1])
    with col_wave:
        st.plotly_chart(_waveform_chart(), use_container_width=True,
                        config=dict(displayModeBar=False))
    with col_gauge:
        conf  = pred.get("confidence", 0.0)
        color = pred.get("color", "#90a4ae")
        st.plotly_chart(_confidence_gauge(conf, label, color),
                        use_container_width=True,
                        config=dict(displayModeBar=False))

        # Probability bars
        if pred.get("probabilities"):
            st.markdown("**Probability Breakdown**")
            for cls_label, prob in pred["probabilities"].items():
                col = CLASS_COLORS.get(
                    [k for k, v in CLASS_NAMES.items() if v == cls_label][0],
                    "#90a4ae"
                )
                bar_w = int(prob * 100)
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#90a4ae;margin-bottom:2px;">'
                    f'{cls_label}</div>'
                    f'<div style="background:#1e2d4a;border-radius:4px;height:10px;margin-bottom:8px;">'
                    f'<div style="background:{col};width:{bar_w}%;height:100%;border-radius:4px;'
                    f'transition:width 0.5s;"></div></div>'
                    f'<div style="font-size:0.7rem;color:{col};margin-top:-6px;margin-bottom:8px;">'
                    f'{prob*100:.1f}%</div>',
                    unsafe_allow_html=True,
                )

    # ── TREND + FREQUENCY CHARTS ─────────────────────────────────────────────
    col_trend, col_freq = st.columns(2)
    with col_trend:
        st.plotly_chart(_trend_chart(), use_container_width=True,
                        config=dict(displayModeBar=False))
    with col_freq:
        st.plotly_chart(_freq_chart(), use_container_width=True,
                        config=dict(displayModeBar=False))

    # ── ALERT HISTORY + LIVE LOG ──────────────────────────────────────────────
    col_alerts, col_log = st.columns(2)

    with col_alerts:
        st.subheader("🚨 Alert History")
        alerts = st.session_state.alert_history
        if not alerts:
            st.markdown('<div class="info-panel">No alerts recorded.</div>',
                        unsafe_allow_html=True)
        else:
            rows = []
            for a in alerts[:15]:
                badge = ("🔴" if a["is_severe"] else "🟡")
                rows.append({
                    "Time":       a["time"],
                    "Status":     badge + " " + a["label"],
                    "Severity":   f"{a['severity']}%",
                    "Confidence": f"{a['confidence']}%",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with col_log:
        st.subheader("📋 Live Monitor Log")
        log_html = ""
        for level, msg in reversed(list(st.session_state.log_lines)):
            css = {"warn": "warn", "error": "error", "ok": "ok"}.get(level, "")
            log_html += f'<div class="log-entry {css}">{msg}</div>'
        st.markdown(
            f'<div style="height:200px;overflow-y:auto;'
            f'background:#0f1629;border:1px solid #1e2d4a;'
            f'border-radius:8px;padding:8px;">{log_html}</div>',
            unsafe_allow_html=True,
        )

    # ── FEATURE DETAILS TABLE ─────────────────────────────────────────────────
    with st.expander("🔬 Feature Details (Signal Analysis)", expanded=False):
        if feats:
            fd = {k: v for k, v in feats.items() if not k.startswith("_")}
            df_f = pd.DataFrame(
                [{"Feature": k, "Value": round(float(v), 5)}
                 for k, v in fd.items()]
            )
            st.dataframe(df_f, use_container_width=True, hide_index=True)
        else:
            st.info("Features appear here once monitoring starts.")

    # ── CSV LOG PREVIEW ───────────────────────────────────────────────────────
    with st.expander("💾 Recent Data Log (CSV Preview)", expanded=False):
        if st.session_state.logger:
            rows = st.session_state.logger.read_recent(20)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
            else:
                st.info("No log data yet.")
        col_dl1, col_dl2 = st.columns([1, 4])
        with col_dl1:
            log_path = Path(__file__).parent / "data" / "tremor_logs.csv"
            if log_path.exists():
                with open(log_path, "rb") as f:
                    st.download_button(
                        "⬇ Download CSV",
                        data=f,
                        file_name="tremor_logs.csv",
                        mime="text/csv",
                    )

    # ── AUTO-REFRESH ──────────────────────────────────────────────────────────
    if st.session_state.monitoring:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=800, limit=None, key="autorefresh")
        except ImportError:
            st.button("🔄 Refresh", use_container_width=False)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;font-size:0.75rem;color:#37474f;">'
        'Parkinson Tremor Detection System &nbsp;·&nbsp; '
        'TENG Wearable Sensor + Arduino UNO + Python + Streamlit + ML &nbsp;·&nbsp; '
        'IEEE Healthcare IoT Platform'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
