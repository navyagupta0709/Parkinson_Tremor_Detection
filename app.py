# ============================================================
# PURE TENG FFT RESEARCH DASHBOARD MODIFICATIONS
# Replace only the required sections in your existing app.py
# ============================================================


# ============================================================
# 1. HEADER SECTION
# Replace old header section with this
# ============================================================

st.markdown("""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

from utils import (
    generate_sensor_data,
    generate_tremor_signal,
    compute_fft,
    detect_anomalies,
    classify_tremor,
    wifi_signal_strength,
    latency_ms,
)

st.set_page_config(
    page_title="TENG Tremor Detection",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main-header {
    background: linear-gradient(
        90deg,
        #111827 0%,
        #1e293b 100%
    );

    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.main-header h1 {
    color: #38bdf8;
}

.main-header p {
    color: white;
}

</style>
""", unsafe_allow_html=True)
<div class="main-header">
  <h1>⚡ TENG Parkinson Tremor Detection System</h1>
  <p>
      Real-Time Signal Processing · FFT Spectrum Analysis ·
      Spectral Classification · IEEE Research Dashboard
  </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 2. SIDEBAR ABOUT SECTION
# Replace only About section
# ============================================================

st.markdown("---")
st.markdown("### ℹ️ About")

st.caption(
    "IEEE Research Prototype\n"
    "TENG-Based Parkinson Tremor Detection\n"
    "Real-Time FFT + PSD + ML Analysis\n"
    "Sampling Frequency = 100 Hz"
)


# ============================================================
# 3. REMOVE THESE OLD METRICS
# DELETE:
# Heart Rate
# Temperature
# SpO2
# ============================================================


# ============================================================
# 4. ADD NEW PURE TENG METRICS
# Replace old metric cards section
# ============================================================

status_col, m1, m2, m3, m4, m5 = st.columns([1.4,1,1,1,1,1])

with status_col:

    st.markdown(
        '<div class="card-title">📡 TENG Sensor Status</div>',
        unsafe_allow_html=True
    )

    if st.session_state.device_on:

        wifi = wifi_signal_strength()
        lat  = latency_ms()

        st.markdown(f"""
        <div class="card">
            <div class="pill-green">
                ● SENSOR ONLINE
            </div>

            <br>

            <small>📶 WiFi : {wifi} dBm</small><br>
            <small>⏱️ Latency : {lat} ms</small><br>
            <small>🔁 Frame : {st.session_state.tick}</small>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="card">
            <div class="pill-red">
                ● SENSOR OFFLINE
            </div>
        </div>
        """, unsafe_allow_html=True)


with m1:

    st.metric(
        "Dominant Frequency",
        display_val(
            "tremor_freq_hz",
            "{:.2f} Hz"
        )
    )


with m2:

    st.metric(
        "Tremor Amplitude",
        display_val(
            "tremor_amplitude",
            "{:.3f} V"
        )
    )


with m3:

    st.metric(
        "Signal Quality",
        display_val(
            "signal_quality",
            "{} %"
        )
    )


with m4:

    tremor_power = latest.get(
        "tremor_amplitude",
        0
    )

    st.metric(
        "Tremor Power",
        f"{float(tremor_power):.3f}"
    )


with m5:

    detected_bpm = (
        safe_float(
            latest.get(
                "tremor_freq_hz",
                0
            )
        ) * 60
    )

    st.metric(
        "Detected BPM",
        f"{detected_bpm:.0f}"
    )


# ============================================================
# 5. REAL-TIME RED TREMOR ALERT
# Replace old alert section completely
# ============================================================

alert_col, class_col = st.columns([1.7,1])

with alert_col:

    st.markdown(
        "## 🚨 Real-Time Tremor Detection"
    )

    tremor_freq = safe_float(
        latest.get(
            "tremor_freq_hz"
        ),
        0
    )

    if 3 <= tremor_freq <= 7:

        if tremor_freq < 4:

            severity_label = "MILD"

        elif tremor_freq < 5.5:

            severity_label = "MODERATE"

        else:

            severity_label = "SEVERE"

        st.markdown(f"""
        <div style="
            background:#7f1d1d;
            border:3px solid red;
            padding:25px;
            border-radius:15px;
            text-align:center;
            animation:pulse 1s infinite;
        ">

            <h1 style="color:white;">
                🚨 TREMOR DETECTED
            </h1>

            <h2 style="color:#fecaca;">
                Dominant Frequency :
                {tremor_freq:.2f} Hz
            </h2>

            <h3 style="color:#fca5a5;">
                Severity :
                {severity_label}
            </h3>

            <p style="color:white;">
                Parkinsonian Tremor Band
                Detected (3–7 Hz)
            </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
            background:#052e16;
            border:3px solid #22c55e;
            padding:25px;
            border-radius:15px;
            text-align:center;
        ">

            <h1 style="color:#bbf7d0;">
                ✅ NORMAL SIGNAL
            </h1>

            <p style="color:white;">
                No Parkinsonian Tremor Detected
            </p>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 6. ML CLASSIFICATION PANEL
# ============================================================

with class_col:

    st.markdown(
        "## 🤖 ML Prediction"
    )

    tf_c = safe_float(
        latest.get(
            "tremor_freq_hz"
        ),
        0
    )

    ta_c = safe_float(
        latest.get(
            "tremor_amplitude"
        ),
        0
    )

    label, conf = classify_tremor(
        tf_c,
        ta_c
    )

    st.markdown(f"""
    <div class="card">

        <h3>
            Prediction :
        </h3>

        <h2 style="color:#ef4444;">
            {label}
        </h2>

        <p>
            Confidence :
            <b>{conf}</b>
        </p>

        <p>
            Model :
            Random Forest / SVM / KNN
        </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 7. CHANGE TABS
# Replace old chart_tabs section
# ============================================================

chart_tabs = st.tabs([
    "⚡ Raw TENG Signal",
    "🧹 Signal Processing",
    "🔬 FFT Spectrum",
    "📈 PSD Analysis"
])


# ============================================================
# 8. RAW TENG SIGNAL
# ============================================================

with chart_tabs[0]:

    tf_wave = safe_float(
        latest.get(
            "tremor_freq_hz"
        ),
        4.5
    )

    t_arr, sig = generate_tremor_signal(
        tf_wave
    )

    fig1, ax1 = plt.subplots(
        figsize=(14,4)
    )

    fig1.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#111827")

    ax1.plot(
        t_arr,
        sig,
        color="#38bdf8",
        linewidth=1.2
    )

    ax1.set_title(
        "Live Raw TENG Signal",
        fontsize=13,
        color="white",
        fontweight="bold"
    )

    ax1.set_xlabel(
        "Time (s)",
        color="white"
    )

    ax1.set_ylabel(
        "Voltage (V)",
        color="white"
    )

    ax1.tick_params(colors="white")

    ax1.grid(alpha=0.2)

    st.pyplot(fig1)

    plt.close(fig1)


# ============================================================
# 9. SIGNAL PROCESSING PIPELINE
# ============================================================

with chart_tabs[1]:

    st.markdown("""
    ## 🧹 Signal Processing Pipeline

    Raw Signal

    ↓

    Remove Invalid Samples

    ↓

    DC Offset Removal

    ↓

    Detrending

    ↓

    4th Order Butterworth Bandpass Filter

    (0.5–10 Hz)

    ↓

    Savitzky–Golay Smoothing

    ↓

    FFT Analysis

    ↓

    Tremor Classification
    """)


# ============================================================
# 10. FFT ANALYSIS
# Replace old FFT section completely
# ============================================================

with chart_tabs[2]:

    tf_fft = float(
        np.clip(
            safe_float(
                latest.get(
                    "tremor_freq_hz"
                ),
                4.5
            ),
            0.1,
            14
        )
    )

    _, sig_fft = generate_tremor_signal(
        tf_fft,
        duration_sec=10
    )

    freqs, mags = compute_fft(
        sig_fft
    )

    peak_idx = np.argmax(mags)

    peak_freq = freqs[peak_idx]

    peak_amp = mags[peak_idx]

    fig3, ax3 = plt.subplots(
        figsize=(15,5)
    )

    fig3.patch.set_facecolor("#0f172a")

    ax3.set_facecolor("#111827")

    ax3.plot(
        freqs,
        mags,
        color="#38bdf8",
        linewidth=1.5
    )

    ax3.axvspan(
        3,
        7,
        color="red",
        alpha=0.15,
        label="Tremor Band (3–7 Hz)"
    )

    ax3.plot(
        peak_freq,
        peak_amp,
        "ro",
        markersize=8
    )

    ax3.annotate(
        f"Peak : {peak_freq:.2f} Hz",
        xy=(peak_freq, peak_amp),
        xytext=(peak_freq + 0.5, peak_amp),
        color="white",
        arrowprops=dict(
            arrowstyle="->",
            color="white"
        )
    )

    ax3.set_xlim(0,12)

    ax3.set_xlabel(
        "Frequency (Hz)",
        color="white"
    )

    ax3.set_ylabel(
        "Amplitude (V)",
        color="white"
    )

    ax3.set_title(
        "Real-Time FFT Spectrum Analysis",
        fontsize=14,
        color="white",
        fontweight="bold"
    )

    ax3.tick_params(colors="white")

    ax3.grid(alpha=0.2)

    ax3.legend()

    st.pyplot(fig3)

    plt.close(fig3)


# ============================================================
# 11. PSD ANALYSIS
# ============================================================

with chart_tabs[3]:

    fig4, ax4 = plt.subplots(
        figsize=(14,4)
    )

    fig4.patch.set_facecolor("#0f172a")

    ax4.set_facecolor("#111827")

    ax4.psd(
        sig_fft,
        Fs=100,
        color="#22c55e"
    )

    ax4.set_title(
        "Welch Power Spectral Density",
        color="white",
        fontsize=13,
        fontweight="bold"
    )

    ax4.tick_params(colors="white")

    ax4.grid(alpha=0.2)

    st.pyplot(fig4)

    plt.close(fig4)
