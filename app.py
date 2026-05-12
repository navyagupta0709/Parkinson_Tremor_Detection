"""
app.py — Parkinson's IoT Wearable Monitoring System
Streamlit dashboard replicating the IoT architecture:
  Wearable → WiFi Gateway → Internet → Web Server → User Dashboard

Fixed: matplotlib axvline None-safety, Python 3.14 compatibility,
       all float() coercions guarded against None values.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os

from utils import (
    generate_sensor_data,
    generate_tremor_signal,
    compute_fft,
    detect_anomalies,
    classify_tremor,
    log_reading,
    load_log,
    clear_log,
    wifi_signal_strength,
    latency_ms,
    TREMOR_PATHOLOGICAL_HZ,
    TREMOR_CRITICAL_HZ,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's IoT Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        padding: 18px 28px; border-radius: 12px; margin-bottom: 24px;
    }
    .main-header h1 { color:#e94560; margin:0; font-size:1.8rem; }
    .main-header p  { color:#a8b2d8; margin:0; font-size:0.9rem; }

    .card {
        background:#f8f9fc; border:1px solid #e2e8f0;
        border-radius:10px; padding:16px 20px; margin-bottom:16px;
    }
    .card-title {
        font-weight:700; font-size:0.85rem; color:#2d3748;
        margin-bottom:8px; text-transform:uppercase; letter-spacing:.04em;
    }

    .alert-warning {
        background:#fffbeb; border-left:4px solid #f59e0b;
        border-radius:6px; padding:10px 14px; margin:5px 0;
        font-size:0.88rem; color:#92400e;
    }
    .alert-critical {
        background:#fef2f2; border-left:4px solid #ef4444;
        border-radius:6px; padding:10px 14px; margin:5px 0;
        font-size:0.88rem; color:#7f1d1d; font-weight:600;
    }
    .alert-ok {
        background:#f0fdf4; border-left:4px solid #22c55e;
        border-radius:6px; padding:10px 14px; margin:5px 0;
        font-size:0.88rem; color:#14532d;
    }

    .arch-box {
        display:inline-block; background:#1a1a2e; color:#e2e8f0;
        border-radius:8px; padding:10px 16px; font-size:0.82rem;
        font-weight:600; text-align:center; min-width:100px;
    }
    .arch-arrow { color:#e94560; font-size:1.3rem; margin:0 4px; vertical-align:middle; }

    .pill-green {
        background:#dcfce7; color:#166534; border-radius:999px;
        padding:3px 12px; font-size:0.8rem; font-weight:700; display:inline-block;
    }
    .pill-red {
        background:#fee2e2; color:#7f1d1d; border-radius:999px;
        padding:3px 12px; font-size:0.8rem; font-weight:700; display:inline-block;
    }
    #MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)
@keyframes blink {
    50% {
        opacity: 0.4;
    }
}


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "device_on":     False,
        "live_mode":     False,
        "history":       pd.DataFrame(),
        "latest":        {},
        "alerts":        [],
        "tick":          0,
        "total_alerts":  0,
        "session_start": datetime.datetime.now(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")

    dev_on = st.toggle("🔌 Wearable Device Power", value=st.session_state.device_on)
    st.session_state.device_on = dev_on

    if dev_on:
        st.markdown('<span class="pill-green">● DEVICE ONLINE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-red">● DEVICE OFFLINE</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Simulation Settings")
    severity = st.selectbox(
        "Tremor Severity",
        ["none", "mild", "moderate", "severe"],
        index=1,
        help="Simulates different Parkinson's tremor profiles (1–7 Hz range)"
    )
    refresh_rate = st.slider("Refresh interval (sec)", 1, 10, 2)

    st.markdown("---")
    st.markdown("### 🚦 Alert Thresholds")
    custom_tremor = st.number_input("Tremor alert ≥ (Hz)", 2.0, 7.0, float(TREMOR_PATHOLOGICAL_HZ), 0.5)
    custom_hr_hi  = st.number_input("Heart rate high (bpm)", 80, 160, 110, 5)
    custom_hr_lo  = st.number_input("Heart rate low  (bpm)", 30, 70,  50,  5)

    st.markdown("---")
    st.markdown("### 📋 Logging")
    log_enabled = st.checkbox("Enable data logging (CSV)", value=True)
    if st.button("🗑️ Clear Log"):
        clear_log()
        st.session_state.history = pd.DataFrame()
        st.success("Log cleared.")

    st.markdown("---")
    live = st.toggle("▶️ Live Streaming Mode", value=st.session_state.live_mode)
    st.session_state.live_mode = live

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "Parkinson's Tremor Detection IoT Dashboard\n"
        "Pathological range: 4–7 Hz\n"
        "Models: KNN / SVM / Random Forest"
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧠 Parkinson's IoT Wearable Monitor</h1>
  <p>Real-time tremor detection · WiFi Gateway · Cloud Processing · Clinical Dashboard</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Architecture flow
# ─────────────────────────────────────────────
with st.expander("📡 System Architecture — IoT Data Flow", expanded=False):
    st.markdown("""
    <div style="text-align:center;padding:20px 0;background:#f8f9fc;border-radius:10px;">
      <span class="arch-box">⌚ Wearable<br><small>Glove / Watch</small></span>
      <span class="arch-arrow">──►</span>
      <span class="arch-box">📶 WiFi<br><small>Gateway</small></span>
      <span class="arch-arrow">──►</span>
      <span class="arch-box">☁️ Internet<br><small>Cloud</small></span>
      <span class="arch-arrow">──►</span>
      <span class="arch-box">🖥️ Web Server<br><small>ML Processing</small></span>
      <span class="arch-arrow">──►</span>
      <span class="arch-box">👤 User<br><small>Dashboard</small></span>
      <span class="arch-arrow">──►</span>
      <span class="arch-box">🚨 Alerts &amp;<br><small>Monitoring</small></span>
    </div>
    <p style="text-align:center;color:#64748b;font-size:0.84rem;margin-top:10px;">
      Sensors → FFT Feature Extraction → KNN / SVM / Random Forest → Real-time alerts
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def safe_float(val, default=0.0):
    """Convert val to float; return default on None/error."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def display_val(key, fmt, fallback="—"):
    v = st.session_state.latest.get(key)
    if v is None:
        return fallback
    return fmt.format(v)


def take_reading():
    reading = generate_sensor_data(st.session_state.device_on, severity)
    st.session_state.latest       = reading
    st.session_state.alerts       = detect_anomalies(reading)
    st.session_state.total_alerts += len(st.session_state.alerts)
    st.session_state.tick        += 1

    if reading.get("device_on"):
        row = {k: v for k, v in reading.items() if k != "device_on"}
        row["timestamp"] = row["timestamp"].strftime("%H:%M:%S")
        new_row = pd.DataFrame([row])
        st.session_state.history = pd.concat(
            [st.session_state.history, new_row], ignore_index=True
        ).tail(120)
        if log_enabled:
            log_reading(reading)


# Initial reading on first load
if st.session_state.tick == 0:
    take_reading()

latest = st.session_state.latest


# ─────────────────────────────────────────────
# Row 1 — Status + Metrics
# ─────────────────────────────────────────────
status_col, m1, m2, m3, m4, m5 = st.columns([1.4, 1, 1, 1, 1, 1])

with status_col:
    st.markdown('<div class="card-title">📡 Connection Status</div>', unsafe_allow_html=True)
    if st.session_state.device_on:
        wifi = wifi_signal_strength()
        lat  = latency_ms()
        bars = "▂▄▆█" if wifi > -50 else ("▂▄▆_" if wifi > -60 else "▂▄__")
        st.markdown(f"""
        <div class="card">
          <div class="pill-green">● CONNECTED</div><br>
          <small>📶 WiFi {bars}&nbsp;{wifi} dBm</small><br>
          <small>⏱️ Latency: {lat} ms</small><br>
          <small>🕒 {datetime.datetime.now().strftime('%H:%M:%S')}</small><br>
          <small>🔁 Tick #{st.session_state.tick}</small>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card">
          <div class="pill-red">● DISCONNECTED</div><br>
          <small>Device is powered off.</small><br>
          <small>Enable wearable in sidebar ↑</small>
        </div>""", unsafe_allow_html=True)

hr_val = latest.get("heart_rate")
with m1:
    st.metric("❤️ Heart Rate",
              display_val("heart_rate", "{} bpm"),
              delta=f"{int(hr_val) - 72:+d}" if hr_val is not None else None)
with m2:
    st.metric("🌡️ Temperature",  display_val("temperature",      "{} °C"))
with m3:
    st.metric("〰️ Tremor Freq",  display_val("tremor_freq_hz",   "{} Hz"))
with m4:
    st.metric("📳 Tremor Amp",   display_val("tremor_amplitude",  "{} g"))
with m5:
    st.metric("🫁 SpO₂",         display_val("spo2",             "{}%"))

st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Row 2 — Alerts + Classification
# ─────────────────────────────────────────────
alert_col, class_col = st.columns([1.6, 1])

with alert_col:
    st.markdown("### 🚨 Alerts")
    if not st.session_state.device_on:
        st.markdown('<div class="alert-ok">Device offline — no active monitoring.</div>',
                    unsafe_allow_html=True)
    elif not st.session_state.alerts:
        st.markdown('<div class="alert-ok">✅ All vitals normal — no anomalies detected.</div>',
                    unsafe_allow_html=True)
    else:
        for a in st.session_state.alerts:
            css = "alert-critical" if a["level"] == "critical" else "alert-warning"
            st.markdown(f'<div class="{css}">{a["icon"]} {a["message"]}</div>',
                        unsafe_allow_html=True)
            # ─────────────────────────────────────────────
# REALTIME TREMOR ALERT
# ─────────────────────────────────────────────
tf_alert = latest.get("tremor_freq_hz")
ta_alert = latest.get("tremor_amplitude")

if tf_alert is not None and ta_alert is not None:

    tf_alert = safe_float(tf_alert, 0.0)
    ta_alert = safe_float(ta_alert, 0.0)

    # Severe Parkinson Tremor
    if tf_alert >= 5.0 and ta_alert >= 1.2:

        st.markdown(f"""
        <div style="
            background:rgba(255,0,0,0.14);
            border:2px solid red;
            border-radius:10px;
            padding:18px;
            margin-top:10px;
            color:#ef4444;
            font-size:1rem;
            font-weight:700;
            animation: blink 1s infinite;
        ">
        🚨 SEVERE PARKINSON TREMOR DETECTED<br>
        Tremor Frequency: {tf_alert:.2f} Hz<br>
        Tremor Amplitude: {ta_alert:.2f} g
        </div>
        """, unsafe_allow_html=True)

    # Mild Tremor
    elif tf_alert >= 4.0:

        st.markdown(f"""
        <div style="
            background:#fffbeb;
            border-left:4px solid orange;
            border-radius:8px;
            padding:12px;
            margin-top:10px;
            color:#92400e;
            font-weight:600;
        ">
        ⚠ MILD TREMOR DETECTED<br>
        Tremor Frequency: {tf_alert:.2f} Hz
        </div>
        """, unsafe_allow_html=True)

    # Normal
    else:

        st.markdown("""
        <div style="
            background:#f0fdf4;
            border-left:4px solid #22c55e;
            border-radius:8px;
            padding:12px;
            margin-top:10px;
            color:#14532d;
            font-weight:600;
        ">
        ✅ NO PATHOLOGICAL TREMOR DETECTED
        </div>
        """, unsafe_allow_html=True)

with class_col:
    st.markdown("### 🤖 ML Classification")
    tf_raw = latest.get("tremor_freq_hz")
    ta_raw = latest.get("tremor_amplitude")
    if tf_raw is not None and ta_raw is not None:
        tf_c = safe_float(tf_raw, 0.0)
        ta_c = safe_float(ta_raw, 0.0)
        label, conf = classify_tremor(tf_c, ta_c)
        colour = ("#ef4444" if "Pathological" in label
                  else "#f59e0b" if "Borderline" in label
                  else "#22c55e")
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {colour};">
          <div class="card-title">Diagnosis</div>
          <div style="font-size:1.1rem;font-weight:700;color:{colour};">{label}</div>
          <div style="font-size:0.84rem;color:#64748b;margin-top:6px;">
            Confidence: <b>{conf}</b><br>
            Tremor: {tf_c} Hz · {ta_c} g<br>
            Model: RF / SVM / KNN ensemble
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Power on device to see classification.")

st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Row 3 — Charts
# ─────────────────────────────────────────────
st.markdown("### 📈 Live Monitoring Charts")
chart_tabs = st.tabs(["📊 Vitals History", "〰️ Tremor Waveform",
                       "🔬 FFT Spectrum",   "📋 Accelerometer"])

# ── Tab 0: Vitals history ──────────────────
with chart_tabs[0]:
    hist = st.session_state.history
    if hist.empty:
        st.info("No data yet — enable device and press Refresh.")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#f8f9fc")
        plots = [
            ("heart_rate",       "Heart Rate (bpm)", "#e94560", (40, 160)),
            ("temperature",      "Temperature (°C)", "#f59e0b", (35, 39)),
            ("tremor_freq_hz",   "Tremor Freq (Hz)", "#7c3aed", (0,  8)),
            ("tremor_amplitude", "Tremor Amp (g)",   "#0f3460", (0,  3)),
        ]
        for ax, (col, title, color, ylim) in zip(axes.flat, plots):
            if col in hist.columns:
                values = pd.to_numeric(hist[col], errors="coerce").dropna().values
                if len(values):
                    ax.plot(values, color=color, linewidth=1.8, alpha=0.9)
                    ax.axhline(float(np.mean(values)), color=color,
                               linestyle="--", alpha=0.4, linewidth=1)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
            ax.set_facecolor("white")
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 1: Tremor waveform ─────────────────
with chart_tabs[1]:
    tf_wave = safe_float(latest.get("tremor_freq_hz"), 4.5)
    t_arr, sig = generate_tremor_signal(tf_wave)
    fig2, ax2 = plt.subplots(figsize=(14, 3.5))
    ax2.plot(t_arr, sig, color="#e94560", linewidth=1.2, alpha=0.85)
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Amplitude (V)", fontsize=10)
    ax2.set_title(
        f"Simulated Tremor Waveform — {tf_wave:.2f} Hz  |  Severity: {severity}",
        fontsize=11, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    fig2.patch.set_facecolor("#f8f9fc")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption("Waveform includes additive noise and baseline drift matching real sensor recordings.")

# ── Tab 2: FFT Spectrum ────────────────────
with chart_tabs[2]:
    # safe_float ensures never None; np.clip ensures within xlim=[0,15]
    tf_fft = float(np.clip(safe_float(latest.get("tremor_freq_hz"), 4.5), 0.01, 14.99))

    _, sig_fft = generate_tremor_signal(tf_fft, duration_sec=10.0)
    freqs, mags = compute_fft(sig_fft)

    fig3, ax3 = plt.subplots(figsize=(14, 3.5))
    ax3.plot(freqs, mags, color="#7c3aed", linewidth=1.5)
    ax3.axvspan(4.0, 7.0, alpha=0.15, color="#ef4444",
                label="Pathological range (4–7 Hz)")
    # x= keyword + plain Python float avoids the Python 3.14 comparison bug
    ax3.axvline(x=tf_fft, color="#ef4444", linestyle="--",
                linewidth=1.5, label=f"Dominant: {tf_fft:.2f} Hz")
    ax3.set_xlim(0.0, 15.0)
    ax3.set_xlabel("Frequency (Hz)", fontsize=10)
    ax3.set_ylabel("Magnitude",      fontsize=10)
    ax3.set_title("FFT Spectrum — Tremor Frequency Analysis",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.patch.set_facecolor("#f8f9fc")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# ── Tab 3: Accelerometer ──────────────────
with chart_tabs[3]:
    hist = st.session_state.history
    if hist.empty or "accel_x" not in hist.columns:
        st.info("No accelerometer data yet.")
    else:
        fig4, axes4 = plt.subplots(1, 3, figsize=(14, 3.5))
        for ax4, col, color in zip(axes4,
                                   ["accel_x", "accel_y", "accel_z"],
                                   ["#e94560",  "#22c55e", "#0f3460"]):
            values4 = pd.to_numeric(hist[col], errors="coerce").dropna().values
            if len(values4):
                ax4.plot(values4, color=color, linewidth=1.4)
            ax4.set_title(col.upper(), fontsize=10, fontweight="bold")
            ax4.set_ylabel("Acceleration (g)" if col == "accel_x" else "")
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor("white")
        fig4.suptitle("3-Axis Accelerometer", fontsize=11, fontweight="bold")
        fig4.patch.set_facecolor("#f8f9fc")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Row 4 — Log table + Stats + Download
# ─────────────────────────────────────────────
log_col, stat_col = st.columns([2, 1])

with log_col:
    st.markdown("### 🗃️ Sensor Log (last 20 readings)")
    hist = st.session_state.history
    if not hist.empty:
        display_cols = ["timestamp", "heart_rate", "temperature",
                        "tremor_freq_hz", "tremor_amplitude", "spo2", "signal_quality"]
        available = [c for c in display_cols if c in hist.columns]
        st.dataframe(hist[available].tail(20).reset_index(drop=True),
                     use_container_width=True, height=260)
        csv_bytes = hist.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download Full Report (CSV)",
            data=csv_bytes,
            file_name=f"parkinson_monitor_{datetime.date.today()}.csv",
            mime="text/csv",
        )
    else:
        st.info("No logged data yet.")

with stat_col:
    st.markdown("### 📊 Session Stats")
    dur = (datetime.datetime.now() - st.session_state.session_start).seconds
    mins, secs = divmod(dur, 60)
    n  = len(st.session_state.history)
    sq = latest.get("signal_quality")
    gw = "🟢 Online" if st.session_state.device_on else "🔴 Offline"

    st.metric("⏱️ Session Duration",   f"{mins}m {secs}s")
    st.metric("📦 Readings Collected", n)
    st.metric("🚨 Total Alerts Fired", st.session_state.total_alerts)
    st.metric("📶 Gateway Status",     gw)
    st.metric("⚡ Signal Quality",     f"{sq}%" if sq is not None else "—")


# ─────────────────────────────────────────────
# Refresh controls
# ─────────────────────────────────────────────
st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)
btn_col, info_col = st.columns([1, 4])

with btn_col:
    if st.button("🔄 Refresh Reading", use_container_width=True):
        take_reading()
        st.rerun()

with info_col:
    if st.session_state.live_mode and st.session_state.device_on:
        st.info(f"▶️ **Live mode ON** — auto-refreshing every {refresh_rate}s.")
    elif st.session_state.live_mode and not st.session_state.device_on:
        st.warning("Live mode is ON but device is OFF. Enable device in sidebar.")
    else:
        st.caption("Manual mode — click Refresh or enable Live Streaming in sidebar.")

# Auto-refresh loop (live mode)
if st.session_state.live_mode and st.session_state.device_on:
    take_reading()
    time.sleep(refresh_rate)
    st.rerun()
