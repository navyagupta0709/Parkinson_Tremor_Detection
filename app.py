"""
app.py — Parkinson's IoT Wearable Monitoring System
Streamlit dashboard replicating the IoT architecture:
  Wearable → WiFi Gateway → Internet → Web Server → User Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import datetime
import io

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
# Custom CSS — minimal dark-accent theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Top header bar */
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 18px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #a8b2d8; margin: 0; font-size: 0.9rem; }

    /* Section cards */
    .card {
        background: #f8f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .card-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #2d3748;
        margin-bottom: 10px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Alert boxes */
    .alert-warning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
        color: #92400e;
    }
    .alert-critical {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
        color: #7f1d1d;
        font-weight: 600;
    }
    .alert-ok {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
        color: #14532d;
    }

    /* Architecture flow */
    .arch-box {
        display: inline-block;
        background: #1a1a2e;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 10px 18px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        min-width: 110px;
    }
    .arch-arrow {
        display: inline-block;
        color: #e94560;
        font-size: 1.4rem;
        vertical-align: middle;
        margin: 0 6px;
    }

    /* Status pill */
    .pill-green {
        background: #dcfce7; color: #166534;
        border-radius: 999px; padding: 3px 12px;
        font-size: 0.8rem; font-weight: 700;
        display: inline-block;
    }
    .pill-red {
        background: #fee2e2; color: #7f1d1d;
        border-radius: 999px; padding: 3px 12px;
        font-size: 0.8rem; font-weight: 700;
        display: inline-block;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "device_on":       False,
        "live_mode":       False,
        "history":         pd.DataFrame(),
        "latest":          {},
        "alerts":          [],
        "tick":            0,
        "total_alerts":    0,
        "session_start":   datetime.datetime.now(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# Sidebar — Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")

    # Device toggle
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
    custom_tremor = st.number_input("Tremor alert ≥ (Hz)", 2.0, 7.0, TREMOR_PATHOLOGICAL_HZ, 0.5)
    custom_hr_hi  = st.number_input("Heart rate high (bpm)", 80, 160, 110, 5)
    custom_hr_lo  = st.number_input("Heart rate low  (bpm)", 30, 70,  50,  5)

    st.markdown("---")
    st.markdown("### 📋 Logging")
    log_enabled = st.checkbox("Enable data logging (CSV)", value=True)

    if st.button("🗑️ Clear Log"):
        clear_log()
        st.session_state.history = pd.DataFrame()
        st.success("Log cleared.")

    # Live mode toggle
    st.markdown("---")
    live = st.toggle("▶️ Live Streaming Mode", value=st.session_state.live_mode)
    st.session_state.live_mode = live

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "Parkinson's Tremor Detection IoT Dashboard\n"
        "Freq range: 4–7 Hz (pathological)\n"
        "Based on KNN / SVM / Random Forest models."
    )


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div>
    <h1>🧠 Parkinson's IoT Wearable Monitor</h1>
    <p>Real-time tremor detection · WiFi Gateway · Cloud Processing · Clinical Dashboard</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# IoT Architecture Flow
# ─────────────────────────────────────────────
with st.expander("📡 System Architecture — IoT Data Flow", expanded=False):
    st.markdown("""
    <div style="text-align:center; padding: 20px 0; background:#f8f9fc; border-radius:10px;">
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
    <br>
    <p style="text-align:center; color:#64748b; font-size:0.85rem;">
      Sensors → FFT Feature Extraction → KNN / SVM / Random Forest classification → Real-time alerts
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: generate + store a reading
# ─────────────────────────────────────────────
def take_reading():
    reading = generate_sensor_data(st.session_state.device_on, severity)
    st.session_state.latest  = reading
    st.session_state.alerts  = detect_anomalies(reading)
    st.session_state.total_alerts += len(st.session_state.alerts)
    st.session_state.tick   += 1

    if reading.get("device_on"):
        row = {k: v for k, v in reading.items() if k != "device_on"}
        row["timestamp"] = row["timestamp"].strftime("%H:%M:%S")
        new_row = pd.DataFrame([row])
        st.session_state.history = pd.concat(
            [st.session_state.history, new_row], ignore_index=True
        ).tail(120)          # keep last 2 min

        if log_enabled:
            log_reading(reading)


# Initial reading on first load
if st.session_state.tick == 0:
    take_reading()


# ─────────────────────────────────────────────
# Row 1 — Connection Status + Key Metrics
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
          <small>📶 WiFi {bars}  {wifi} dBm</small><br>
          <small>⏱️ Latency: {lat} ms</small><br>
          <small>🕒 {datetime.datetime.now().strftime('%H:%M:%S')}</small><br>
          <small>🔁 Tick #{st.session_state.tick}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card">
          <div class="pill-red">● DISCONNECTED</div><br>
          <small>Device is powered off.</small><br>
          <small>Enable wearable in sidebar.</small>
        </div>
        """, unsafe_allow_html=True)

latest = st.session_state.latest

def _val(key, fmt=None, fallback="—"):
    v = latest.get(key)
    if v is None: return fallback
    return fmt.format(v) if fmt else v

with m1:
    st.metric("❤️ Heart Rate",
              _val("heart_rate", "{} bpm"),
              delta=None if not latest.get("heart_rate") else
              f"{latest['heart_rate']-72:+d}" )

with m2:
    st.metric("🌡️ Temperature",
              _val("temperature", "{} °C"))

with m3:
    st.metric("〰️ Tremor Freq",
              _val("tremor_freq_hz", "{} Hz"))

with m4:
    st.metric("📳 Tremor Amp",
              _val("tremor_amplitude", "{} g"))

with m5:
    st.metric("🫁 SpO₂",
              _val("spo2", "{}%"))


st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Row 2 — Alerts + Classification
# ─────────────────────────────────────────────
alert_col, class_col = st.columns([1.6, 1])

with alert_col:
    st.markdown("### 🚨 Alerts")
    alerts = st.session_state.alerts
    if not st.session_state.device_on:
        st.markdown('<div class="alert-ok">Device offline — no active monitoring.</div>',
                    unsafe_allow_html=True)
    elif not alerts:
        st.markdown('<div class="alert-ok">✅ All vitals normal — no anomalies detected.</div>',
                    unsafe_allow_html=True)
    else:
        for a in alerts:
            css_cls = "alert-critical" if a["level"] == "critical" else "alert-warning"
            st.markdown(f'<div class="{css_cls}">{a["icon"]} {a["message"]}</div>',
                        unsafe_allow_html=True)

with class_col:
    st.markdown("### 🤖 ML Classification")
    tf = latest.get("tremor_freq_hz")
    ta = latest.get("tremor_amplitude")
    if tf and ta:
        label, conf = classify_tremor(tf, ta)
        colour = "#ef4444" if "Pathological" in label else (
                 "#f59e0b" if "Borderline" in label else "#22c55e")
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid {colour};">
          <div class="card-title">Diagnosis</div>
          <div style="font-size:1.1rem; font-weight:700; color:{colour};">{label}</div>
          <div style="font-size:0.85rem; color:#64748b; margin-top:6px;">
            Confidence: <b>{conf}</b><br>
            Tremor: {tf} Hz · {ta} g<br>
            Model: Random Forest (KNN/SVM ensemble)
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Power on device to see classification.")


st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Row 3 — Live Charts
# ─────────────────────────────────────────────
st.markdown("### 📈 Live Monitoring Charts")

chart_tabs = st.tabs(["📊 Vitals History", "〰️ Tremor Waveform", "🔬 FFT Spectrum", "📋 Accelerometer"])

with chart_tabs[0]:
    hist = st.session_state.history
    if hist.empty:
        st.info("No data yet — enable device and press Refresh.")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#f8f9fc")

        plots = [
            ("heart_rate",       "Heart Rate (bpm)", "#e94560", [40, 160]),
            ("temperature",      "Temperature (°C)", "#f59e0b", [35, 39]),
            ("tremor_freq_hz",   "Tremor Freq (Hz)", "#7c3aed", [0, 8]),
            ("tremor_amplitude", "Tremor Amp (g)",   "#0f3460", [0, 3]),
        ]
        for ax, (col, title, color, ylim) in zip(axes.flat, plots):
            if col in hist.columns:
                ax.plot(hist[col].values, color=color, linewidth=1.8, alpha=0.9)
                ax.axhline(hist[col].mean(), color=color, linestyle="--", alpha=0.4, linewidth=1)
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.set_ylim(ylim)
                ax.set_facecolor("white")
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with chart_tabs[1]:
    tf_now = latest.get("tremor_freq_hz", 4.5)
    t_arr, sig = generate_tremor_signal(tf_now if tf_now else 4.5)
    fig2, ax2 = plt.subplots(figsize=(14, 3.5))
    ax2.plot(t_arr, sig, color="#e94560", linewidth=1.2, alpha=0.85)
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Amplitude (V)", fontsize=10)
    ax2.set_title(f"Simulated Tremor Waveform — {tf_now} Hz  |  Severity: {severity}",
                  fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    fig2.patch.set_facecolor("#f8f9fc")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.caption("Waveform includes additive noise and baseline drift as in real sensor recordings.")

with chart_tabs[2]:
    tf_now = latest.get("tremor_freq_hz", 4.5)
    _, sig_fft = generate_tremor_signal(tf_now if tf_now else 4.5, duration_sec=10.0)
    freqs, mags = compute_fft(sig_fft)

    fig3, ax3 = plt.subplots(figsize=(14, 3.5))
    ax3.plot(freqs, mags, color="#7c3aed", linewidth=1.5)
    ax3.axvspan(4, 7, alpha=0.15, color="#ef4444", label="Pathological range (4–7 Hz)")
    ax3.axvline(tf_now, color="#ef4444", linestyle="--", linewidth=1.5,
                label=f"Dominant: {tf_now} Hz")
    ax3.set_xlim(0, 15)
    ax3.set_xlabel("Frequency (Hz)", fontsize=10)
    ax3.set_ylabel("Magnitude", fontsize=10)
    ax3.set_title("FFT Spectrum — Tremor Frequency Analysis", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.patch.set_facecolor("#f8f9fc")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with chart_tabs[3]:
    hist = st.session_state.history
    if hist.empty or "accel_x" not in hist.columns:
        st.info("No accelerometer data yet.")
    else:
        fig4, axes4 = plt.subplots(1, 3, figsize=(14, 3.5))
        colors = ["#e94560", "#22c55e", "#0f3460"]
        for ax4, col, color in zip(axes4, ["accel_x", "accel_y", "accel_z"], colors):
            ax4.plot(hist[col].values, color=color, linewidth=1.4)
            ax4.set_title(col.upper(), fontsize=10, fontweight="bold")
            ax4.set_ylabel("Acceleration (g)" if col == "accel_x" else "")
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor("white")
        fig4.suptitle("3-Axis Accelerometer", fontsize=11, fontweight="bold")
        fig4.patch.set_facecolor("#f8f9fc")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()


st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Row 4 — Data Table + Download
# ─────────────────────────────────────────────
log_col, stat_col = st.columns([2, 1])

with log_col:
    st.markdown("### 🗃️ Sensor Log (last 20 readings)")
    hist = st.session_state.history
    if not hist.empty:
        display_cols = ["timestamp", "heart_rate", "temperature",
                        "tremor_freq_hz", "tremor_amplitude", "spo2", "signal_quality"]
        available   = [c for c in display_cols if c in hist.columns]
        st.dataframe(hist[available].tail(20).reset_index(drop=True),
                     use_container_width=True, height=260)

        # Download button
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
    n = len(st.session_state.history)
    st.markdown(f"""
    <div class="card">
      <b>⏱️ Session duration</b><br>{mins}m {secs}s<br><br>
      <b>📦 Readings collected</b><br>{n}<br><br>
      <b>🚨 Total alerts fired</b><br>{st.session_state.total_alerts}<br><br>
      <b>📶 Gateway status</b><br>{'🟢 Online' if st.session_state.device_on else '🔴 Offline'}<br><br>
      <b>⚡ Signal quality</b><br>{latest.get('signal_quality', '—')}%
    </div>
    """, unsafe_allow_html=True)


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
        st.info(f"▶️ **Live mode ON** — auto-refreshing every {refresh_rate}s. Toggle off in sidebar to pause.")
    elif st.session_state.live_mode and not st.session_state.device_on:
        st.warning("Live mode is ON but device is OFF. Enable device in sidebar.")
    else:
        st.caption("Manual mode — click Refresh or enable Live Streaming in sidebar.")

# Auto-refresh loop
if st.session_state.live_mode and st.session_state.device_on:
    take_reading()
    time.sleep(refresh_rate)
    st.rerun()
