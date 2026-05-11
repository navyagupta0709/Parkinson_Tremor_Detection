"""
IEEE-Style TENG Parkinson Tremor Alert System
Real-time TENG sensor signal acquisition, FFT analysis, and tremor classification.
Supports: Live Arduino serial stream OR dataset replay from READINGS.xlsx
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfiltfilt, detrend, savgol_filter, welch
import time
import datetime
import csv
import os
import io
import threading
import queue

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TENG Tremor Alert System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — IEEE Dark Theme ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #070d1a !important;
    color: #c8d8f0 !important;
    font-family: 'Exo 2', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1628 0%, #0d1f3c 100%) !important;
    border-right: 1px solid #1e3a5f;
}

.stMetric {
    background: linear-gradient(135deg, #0d1f3c 0%, #112244 100%);
    border: 1px solid #1e4080;
    border-radius: 6px;
    padding: 12px;
}

[data-testid="stMetricValue"] {
    color: #4fc3f7 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.6rem !important;
}

[data-testid="stMetricLabel"] {
    color: #7a9cc5 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.alert-tremor {
    background: linear-gradient(135deg, #3d0000 0%, #6b0000 100%);
    border: 2px solid #ff2222;
    border-radius: 8px;
    padding: 20px 28px;
    text-align: center;
    animation: pulse-border 1.2s ease-in-out infinite;
    box-shadow: 0 0 40px rgba(255,0,0,0.35), inset 0 0 20px rgba(255,0,0,0.08);
    margin-bottom: 16px;
}

.alert-tremor h1 {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: #ff4444;
    margin: 0 0 4px 0;
    letter-spacing: 0.12em;
    text-shadow: 0 0 20px rgba(255,50,50,0.8);
}

.alert-tremor .freq-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    color: #ffaaaa;
    margin: 6px 0;
}

.alert-tremor .severity {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 8px;
}

.alert-tremor .ts {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #cc8888;
    margin-top: 8px;
}

.alert-normal {
    background: linear-gradient(135deg, #001a0d 0%, #002b14 100%);
    border: 2px solid #00c853;
    border-radius: 8px;
    padding: 20px 28px;
    text-align: center;
    box-shadow: 0 0 24px rgba(0,200,83,0.2);
    margin-bottom: 16px;
}

.alert-normal h1 {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: #00c853;
    margin: 0 0 4px 0;
    letter-spacing: 0.1em;
}

.alert-normal .freq-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.05rem;
    color: #80cfa0;
    margin: 4px 0;
}

@keyframes pulse-border {
    0%   { border-color: #ff2222; box-shadow: 0 0 30px rgba(255,0,0,0.3); }
    50%  { border-color: #ff6666; box-shadow: 0 0 60px rgba(255,0,0,0.6); }
    100% { border-color: #ff2222; box-shadow: 0 0 30px rgba(255,0,0,0.3); }
}

.pipeline-box {
    background: #0a1628;
    border: 1px solid #1a3660;
    border-radius: 6px;
    padding: 14px 18px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #7ab3e0;
    line-height: 2.0;
}

.pipeline-box .step {
    color: #4fc3f7;
}

.pipeline-box .arrow {
    color: #1e4a80;
    margin-left: 8px;
}

.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #4fc3f7;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-bottom: 1px solid #1e3a60;
    padding-bottom: 4px;
    margin-bottom: 10px;
}

.stat-card {
    background: #0d1f3c;
    border: 1px solid #1a3a6e;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: 'Share Tech Mono', monospace;
}

.stat-card .label { color: #5a80a8; font-size: 0.7rem; text-transform: uppercase; }
.stat-card .value { color: #e0f0ff; font-size: 1.15rem; margin-top: 2px; }

.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #00ff88;
    border-radius: 50%;
    animation: blink 1s ease-in-out infinite;
    margin-right: 6px;
    box-shadow: 0 0 8px #00ff88;
}

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stRadio"] label {
    color: #7ab3e0 !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stButton > button {
    background: linear-gradient(135deg, #0d2a5c, #1a4080) !important;
    border: 1px solid #2a60b0 !important;
    color: #90c0f0 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600;
    letter-spacing: 0.06em;
    border-radius: 4px;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1a4080, #2a60c0) !important;
    border-color: #4a90f0 !important;
    color: #c8e8ff !important;
    box-shadow: 0 0 12px rgba(80,160,255,0.3);
}

.ieee-header {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: #4fc3f7;
    letter-spacing: 0.08em;
}

.ieee-sub {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.8rem;
    color: #4a7090;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FS            = 100.0          # sampling frequency Hz
WINDOW_SEC    = 4.0            # analysis window (seconds)
WINDOW_SAMP   = int(FS * WINDOW_SEC)
TREMOR_LO     = 3.0            # Hz
TREMOR_HI     = 7.0            # Hz
STEP_SEC      = 0.2            # update interval (seconds)
LOG_FILE      = "tremor_log.csv"
EXCEL_PATH    = "READINGS.xlsx"

# ── Session state init ─────────────────────────────────────────────────────────
defaults = dict(
    running        = False,
    buffer         = np.zeros(WINDOW_SAMP),
    alert_count    = 0,
    total_windows  = 0,
    session_start  = None,
    log_rows       = [],
    last_result    = None,
    dataset_idx    = 0,
    dataset_signal = None,
    dataset_bpm    = 60,
    serial_ok      = False,
    source         = "Dataset",
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Signal processing helpers ─────────────────────────────────────────────────

def process_signal(sig: np.ndarray, fs: float = FS) -> np.ndarray:
    """DC removal → detrend → Butterworth 0.5-10 Hz → Savitzky-Golay."""
    s = sig.copy().astype(np.float64)
    s -= np.mean(s)
    s  = detrend(s)
    sos = butter(4, [0.5, 10.0], btype='bandpass', fs=fs, output='sos')
    s  = sosfiltfilt(sos, s)
    s  = savgol_filter(s, window_length=11, polyorder=3)
    return s


def compute_fft(sig: np.ndarray, fs: float = FS):
    n      = len(sig)
    freqs  = fftfreq(n, d=1.0 / fs)
    mag    = np.abs(fft(sig)) / n
    pos    = freqs > 0
    return freqs[pos], mag[pos]


def dominant_frequency(freqs: np.ndarray, mag: np.ndarray,
                        lo: float = 0.5, hi: float = 15.0) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(freqs[mask][np.argmax(mag[mask])])


def tremor_power(freqs: np.ndarray, mag: np.ndarray) -> float:
    mask = (freqs >= TREMOR_LO) & (freqs <= TREMOR_HI)
    if not np.any(mask):
        return 0.0
    return float(np.sum(mag[mask] ** 2))


def signal_quality(sig: np.ndarray) -> float:
    """Simple SNR proxy: ratio of signal variance to noise floor."""
    rms = np.sqrt(np.mean(sig ** 2))
    noise = np.std(np.diff(sig))
    if noise < 1e-10:
        return 100.0
    snr = 20 * np.log10(rms / (noise + 1e-10))
    return float(np.clip((snr + 20) / 60 * 100, 0, 100))


def classify(dom_freq: float) -> dict:
    is_tremor = TREMOR_LO <= dom_freq <= TREMOR_HI
    if is_tremor:
        if dom_freq < 4.0:
            severity = "Mild Tremor"
            sev_color = "#ff9800"
        elif dom_freq < 5.5:
            severity = "Moderate Tremor"
            sev_color = "#ff5722"
        else:
            severity = "Severe Tremor"
            sev_color = "#f44336"
        confidence = min(100, 60 + 40 * (dom_freq - TREMOR_LO) / (TREMOR_HI - TREMOR_LO + 0.01))
    else:
        severity   = "No Tremor"
        sev_color  = "#00c853"
        confidence = min(100, 50 + 50 * abs(dom_freq - 5.0) / 5.0)

    return dict(
        is_tremor  = is_tremor,
        dom_freq   = dom_freq,
        severity   = severity,
        sev_color  = sev_color,
        confidence = confidence,
        timestamp  = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
    )

# ── Dataset loader ─────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset(path: str = EXCEL_PATH):
    try:
        xl = pd.read_excel(path, sheet_name=None, header=None)
    except Exception:
        return None, []

    records = []
    for sheet_name, df_raw in xl.items():
        bpm = int(sheet_name.split()[0])
        df_data = df_raw.iloc[2:].reset_index(drop=True)
        df_data.columns = ['t1', 'v1', 't2', 'v2', 't3', 'v3']
        for vcol in ['v1', 'v2', 'v3']:
            sig = pd.to_numeric(df_data[vcol], errors='coerce').dropna().values.astype(np.float64)
            if len(sig) > 50:
                records.append({'bpm': bpm, 'signal': sig, 'label': f"{bpm} BPM"})
    return xl, records

# ── Serial helper (best-effort) ───────────────────────────────────────────────

def try_open_serial(port: str, baud: int):
    try:
        import serial
        s = serial.Serial(port, baud, timeout=0.05)
        return s
    except Exception:
        return None

# ── Plotly chart builders ─────────────────────────────────────────────────────

PLOT_BG   = "#070d1a"
GRID_CLR  = "#111d35"
FONT_CLR  = "#7a9cc5"
ACCENT    = "#4fc3f7"
RED_TRACE = "#ff4444"
GREEN_TR  = "#00c853"

def base_layout(title: str, xlab: str, ylab: str) -> dict:
    return dict(
        title       = dict(text=title, font=dict(family="Rajdhani", size=13, color=ACCENT)),
        paper_bgcolor = PLOT_BG,
        plot_bgcolor  = PLOT_BG,
        font          = dict(family="Share Tech Mono", size=10, color=FONT_CLR),
        xaxis = dict(title=xlab, gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=9)),
        yaxis = dict(title=ylab, gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=9)),
        margin  = dict(l=48, r=18, t=38, b=38),
        showlegend = True,
        legend  = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
    )


def make_time_plot(sig: np.ndarray, fs: float = FS):
    t = np.arange(len(sig)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=sig, mode='lines',
        line=dict(color=ACCENT, width=1.2),
        name="TENG Signal",
    ))
    fig.update_layout(**base_layout(
        "● LIVE TIME-DOMAIN SIGNAL", "Time (s)", "Voltage (V)"
    ))
    return fig


def make_fft_plot(freqs: np.ndarray, mag: np.ndarray, dom_freq: float, is_tremor: bool):
    fig = go.Figure()

    # tremor band shading
    fig.add_vrect(
        x0=TREMOR_LO, x1=TREMOR_HI,
        fillcolor="rgba(255,50,50,0.12)",
        layer="below", line_width=0,
        annotation_text="Tremor Band",
        annotation_position="top left",
        annotation_font=dict(color="#ff7070", size=9),
    )

    # spectrum
    fig.add_trace(go.Scatter(
        x=freqs, y=mag, mode='lines',
        line=dict(color="#1e88e5", width=1.4),
        name="FFT Spectrum",
        fill='tozeroy', fillcolor='rgba(30,136,229,0.08)',
    ))

    # dominant peak
    peak_mag = float(np.interp(dom_freq, freqs, mag)) if len(freqs) > 0 else 0
    peak_color = RED_TRACE if is_tremor else GREEN_TR
    fig.add_trace(go.Scatter(
        x=[dom_freq], y=[peak_mag],
        mode='markers+text',
        marker=dict(color=peak_color, size=10, symbol='diamond',
                    line=dict(color='white', width=1)),
        text=[f"  {dom_freq:.2f} Hz"],
        textfont=dict(color=peak_color, size=10),
        textposition='middle right',
        name=f"Peak {dom_freq:.2f} Hz",
    ))

    fig.update_layout(**base_layout("◆ FFT SPECTRUM — TREMOR BAND ANALYSIS", "Frequency (Hz)", "Magnitude"))
    fig.update_xaxes(range=[0, 15])
    return fig


def make_psd_plot(sig: np.ndarray, fs: float = FS):
    fw, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)//2 or 64))
    fig = go.Figure()
    fig.add_vrect(x0=TREMOR_LO, x1=TREMOR_HI,
                  fillcolor="rgba(255,80,80,0.10)", layer="below", line_width=0)
    fig.add_trace(go.Scatter(
        x=fw, y=10*np.log10(psd + 1e-20), mode='lines',
        line=dict(color="#ab47bc", width=1.3),
        fill='tozeroy', fillcolor='rgba(171,71,188,0.07)',
        name="Welch PSD",
    ))
    fig.update_layout(**base_layout("▲ WELCH PSD (dB/Hz)", "Frequency (Hz)", "PSD (dBV²/Hz)"))
    fig.update_xaxes(range=[0, 15])
    return fig

# ── CSV logging ───────────────────────────────────────────────────────────────

def log_result(result: dict, sq: float, tp: float, bpm_approx: float):
    row = dict(
        timestamp    = result['timestamp'],
        dom_freq_hz  = round(result['dom_freq'], 4),
        is_tremor    = result['is_tremor'],
        severity     = result['severity'],
        confidence   = round(result['confidence'], 2),
        signal_quality = round(sq, 2),
        tremor_power = round(tp, 6),
        detected_bpm = round(bpm_approx, 1),
    )
    st.session_state.log_rows.append(row)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0 or os.path.getsize(LOG_FILE) == 0:
            w.writeheader()
        w.writerow(row)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ieee-header">⚡ TENG TREMOR<br>ALERT SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="ieee-sub">IEEE Biomedical Signal Processing Interface<br>Parkinson Disease Tremor Detection v2.0</div>', unsafe_allow_html=True)
    st.markdown("---")

    source = st.radio("Signal Source", ["Dataset (READINGS.xlsx)", "Arduino Serial"],
                      index=0, key="src_radio")
    st.session_state.source = source

    if "Arduino" in source:
        st.markdown('<div class="section-header">Serial Config</div>', unsafe_allow_html=True)
        serial_port = st.text_input("Port", value="COM3")
        baud_rate   = st.selectbox("Baud Rate", [9600, 115200, 57600], index=1)
        if st.button("Connect"):
            ser = try_open_serial(serial_port, baud_rate)
            if ser:
                st.session_state['serial_obj'] = ser
                st.session_state.serial_ok = True
                st.success("Connected!")
            else:
                st.error("Could not open port.")
    else:
        st.markdown('<div class="section-header">Dataset Config</div>', unsafe_allow_html=True)
        _, ds_records = load_dataset()
        if ds_records:
            labels = [r['label'] for r in ds_records]
            unique_labels = list(dict.fromkeys(labels))
            chosen = st.selectbox("BPM Class", unique_labels, index=0)
            chosen_recs = [r for r in ds_records if r['label'] == chosen]
            trial_idx = st.selectbox("Trial", [f"Trial {i+1}" for i in range(len(chosen_recs))], index=0)
            trial_i   = int(trial_idx.split()[-1]) - 1
            st.session_state.dataset_signal = chosen_recs[trial_i]['signal']
            st.session_state.dataset_bpm    = chosen_recs[trial_i]['bpm']
            st.session_state.dataset_idx    = 0

            replay_speed = st.slider("Replay Speed", 0.5, 4.0, 1.0, 0.5)
        else:
            st.warning("READINGS.xlsx not found — place it in the same folder.")
            replay_speed = 1.0

    st.markdown("---")
    st.markdown('<div class="section-header">Detection Params</div>', unsafe_allow_html=True)
    win_sec   = st.slider("Window (s)", 1.0, 8.0, 4.0, 0.5)
    update_ms = st.slider("Update Rate (ms)", 100, 1000, 200, 50)

    st.markdown("---")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        start_btn = st.button("▶ START", use_container_width=True)
    with col_s2:
        stop_btn  = st.button("■ STOP",  use_container_width=True)

    if start_btn:
        st.session_state.running       = True
        st.session_state.session_start = datetime.datetime.now()
        st.session_state.alert_count   = 0
        st.session_state.total_windows = 0
        st.session_state.log_rows      = []
        st.session_state.dataset_idx   = 0
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

    if stop_btn:
        st.session_state.running = False

    st.markdown("---")
    if st.session_state.log_rows:
        df_log = pd.DataFrame(st.session_state.log_rows)
        csv_bytes = df_log.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV Log", csv_bytes, "tremor_log.csv", "text/csv",
                           use_container_width=True)

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
  <span style="font-family:'Rajdhani',sans-serif; font-size:1.9rem; font-weight:700;
               color:#4fc3f7; letter-spacing:0.08em;">
    🧠 TENG PARKINSON TREMOR DETECTION SYSTEM
  </span>
</div>
<div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#3a6090;
            margin-bottom:18px; letter-spacing:0.05em;">
  Real-Time Biomedical Signal Processing · FFT Tremor Classification · IEEE Research Prototype
</div>
""", unsafe_allow_html=True)

# Alert placeholder
alert_ph   = st.empty()

# Metrics row
m1, m2, m3, m4, m5, m6 = st.columns(6)
mph = [m1.empty(), m2.empty(), m3.empty(), m4.empty(), m5.empty(), m6.empty()]

# Charts row
gc1, gc2 = st.columns([3, 2])
time_ph = gc1.empty()
fft_ph  = gc2.empty()

gc3, gc4 = st.columns([2, 3])
pipe_ph  = gc3.empty()
psd_ph   = gc4.empty()

# Stats row
st.markdown("---")
sc1, sc2, sc3 = st.columns(3)
sess_ph  = sc1.empty()
hist_ph  = sc2.empty()
sev_ph   = sc3.empty()

# ── Pipeline display ──────────────────────────────────────────────────────────
def render_pipeline(active_step: int = -1):
    steps = [
        "Raw TENG Signal Input",
        "DC Offset Removal",
        "Linear Detrending",
        "Butterworth Bandpass  [0.5–10 Hz]",
        "Savitzky–Golay Smoothing",
        "FFT Spectral Analysis",
        "Tremor Band Extraction  [3–7 Hz]",
        "Dominant Peak Detection",
        "Severity Classification",
    ]
    html = '<div class="pipeline-box">'
    html += '<div style="color:#4fc3f7;font-size:0.8rem;margin-bottom:8px;letter-spacing:0.1em;">▶ PROCESSING PIPELINE</div>'
    for i, s in enumerate(steps):
        if i == active_step:
            color = "#00ff88"
            prefix = "→ "
        elif i < active_step:
            color = "#2a6030"
            prefix = "✓ "
        else:
            color = "#2a4060"
            prefix = "  "
        arrow = "<br>" if i < len(steps) - 1 else ""
        html += f'<span style="color:{color};">{prefix}{s}</span>{arrow}'
    html += '</div>'
    return html


# ── Main loop ─────────────────────────────────────────────────────────────────
if not st.session_state.running:
    alert_ph.markdown("""
    <div class="alert-normal">
      <h1>● SYSTEM IDLE</h1>
      <div class="freq-badge">Press ▶ START to begin real-time monitoring</div>
    </div>
    """, unsafe_allow_html=True)

    pipe_ph.markdown(render_pipeline(-1), unsafe_allow_html=True)

    for i, (label, val) in enumerate([
        ("Dom. Frequency", "— Hz"),
        ("Tremor Power",   "— V²"),
        ("Signal Quality", "— %"),
        ("Detected BPM",   "—"),
        ("Alert Count",    "0"),
        ("Confidence",     "— %"),
    ]):
        mph[i].metric(label, val)

    sess_ph.info("Session not started. Press ▶ START.")
else:
    # --- Determine window of samples ---
    WSAMPLES = int(FS * win_sec)

    # Pull samples from dataset or serial
    if "Dataset" in st.session_state.source:
        raw_signal = st.session_state.dataset_signal
        if raw_signal is None or len(raw_signal) < WSAMPLES:
            st.warning("No dataset signal loaded or signal too short.")
            st.stop()

        idx = st.session_state.dataset_idx
        end = idx + WSAMPLES
        if end >= len(raw_signal):
            idx = 0
            end = WSAMPLES
        window_raw = raw_signal[idx:end]
        st.session_state.dataset_idx = idx + int(FS * (update_ms / 1000) * replay_speed)
    else:
        # Arduino serial
        ser = st.session_state.get('serial_obj', None)
        if ser is None or not ser.is_open:
            st.error("Serial port not connected. Use sidebar to connect.")
            st.stop()
        # Read available bytes
        new_vals = []
        while ser.in_waiting and len(new_vals) < int(FS * 0.5):
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                val  = float(line)
                new_vals.append(val)
            except Exception:
                pass
        buf = np.roll(st.session_state.buffer, -len(new_vals))
        if new_vals:
            buf[-len(new_vals):] = np.array(new_vals)
        st.session_state.buffer = buf
        window_raw = buf[-WSAMPLES:]

    # --- Signal processing ---
    if len(window_raw) < 20:
        st.warning("Not enough samples yet…")
        st.stop()

    proc   = process_signal(window_raw, FS)
    freqs, mag = compute_fft(proc, FS)
    dom_f  = dominant_frequency(freqs, mag)
    tp     = tremor_power(freqs, mag)
    sq     = signal_quality(proc)
    result = classify(dom_f)
    bpm_approx = dom_f * 60.0

    st.session_state.last_result   = result
    st.session_state.total_windows += 1
    if result['is_tremor']:
        st.session_state.alert_count += 1

    log_result(result, sq, tp, bpm_approx)

    # ── Alert box ──────────────────────────────────────────────────────────
    if result['is_tremor']:
        alert_ph.markdown(f"""
        <div class="alert-tremor">
          <h1>⚠ TREMOR DETECTED</h1>
          <div class="freq-badge">Dominant Frequency: <b>{dom_f:.3f} Hz</b></div>
          <div class="severity" style="color:{result['sev_color']};">
            ▶ {result['severity']}
          </div>
          <div class="freq-badge">Signal Quality: {sq:.1f}% &nbsp;|&nbsp; Confidence: {result['confidence']:.1f}%</div>
          <div class="ts">⏱ {result['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        alert_ph.markdown(f"""
        <div class="alert-normal">
          <h1>✔ Normal Signal</h1>
          <div class="freq-badge">Dominant Frequency: {dom_f:.3f} Hz &nbsp;|&nbsp;
          Signal Quality: {sq:.1f}%</div>
          <div class="freq-badge" style="color:#4caf50; margin-top:4px;">No Parkinsonian Tremor Detected</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────────────────────────────
    labels_vals = [
        ("Dom. Frequency",  f"{dom_f:.3f} Hz"),
        ("Tremor Power",    f"{tp:.4f} V²"),
        ("Signal Quality",  f"{sq:.1f} %"),
        ("Detected BPM",    f"{bpm_approx:.1f}"),
        ("Alerts This Run", str(st.session_state.alert_count)),
        ("ML Confidence",   f"{result['confidence']:.1f} %"),
    ]
    for i, (label, val) in enumerate(labels_vals):
        mph[i].metric(label, val)

    # ── Time-domain chart ──────────────────────────────────────────────────
    time_ph.plotly_chart(make_time_plot(proc, FS), use_container_width=True, key="tp")

    # ── FFT chart ──────────────────────────────────────────────────────────
    fft_ph.plotly_chart(make_fft_plot(freqs, mag, dom_f, result['is_tremor']),
                        use_container_width=True, key="fp")

    # ── Pipeline ───────────────────────────────────────────────────────────
    step = (st.session_state.total_windows % 9)
    pipe_ph.markdown(render_pipeline(step), unsafe_allow_html=True)

    # ── PSD chart ──────────────────────────────────────────────────────────
    psd_ph.plotly_chart(make_psd_plot(proc, FS), use_container_width=True, key="pp")

    # ── Session stats ──────────────────────────────────────────────────────
    elapsed = (datetime.datetime.now() - st.session_state.session_start).total_seconds()
    alert_rate = (st.session_state.alert_count / max(1, st.session_state.total_windows)) * 100

    sess_ph.markdown(f"""
    <div class="section-header">SESSION STATISTICS</div>
    <div class="stat-card"><div class="label">Elapsed Time</div>
      <div class="value">{int(elapsed//60):02d}:{int(elapsed%60):02d}</div></div>
    <div class="stat-card"><div class="label">Windows Processed</div>
      <div class="value">{st.session_state.total_windows}</div></div>
    <div class="stat-card"><div class="label">Tremor Alert Rate</div>
      <div class="value">{alert_rate:.1f} %</div></div>
    <div class="stat-card"><div class="label">Total Alerts</div>
      <div class="value">{st.session_state.alert_count}</div></div>
    <div class="stat-card"><div class="label">Sampling Rate</div>
      <div class="value">{int(FS)} Hz</div></div>
    """, unsafe_allow_html=True)

    # ── History chart ──────────────────────────────────────────────────────
    if len(st.session_state.log_rows) >= 2:
        df_h = pd.DataFrame(st.session_state.log_rows[-60:])
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=list(range(len(df_h))), y=df_h['dom_freq_hz'],
            mode='lines+markers', line=dict(color=ACCENT, width=1.2),
            marker=dict(color=[RED_TRACE if t else GREEN_TR for t in df_h['is_tremor']],
                        size=5), name="Dominant Freq",
        ))
        fig_h.add_hrect(y0=TREMOR_LO, y1=TREMOR_HI,
                        fillcolor="rgba(255,50,50,0.10)", line_width=0)
        fig_h.update_layout(
            **base_layout("◈ FREQUENCY HISTORY (last 60 windows)", "Window #", "Hz"),
            height=220,
        )
        hist_ph.plotly_chart(fig_h, use_container_width=True, key="hp")
    else:
        hist_ph.info("Frequency history will appear after a few windows…")

    # ── Severity distribution ──────────────────────────────────────────────
    if st.session_state.log_rows:
        df_sv = pd.DataFrame(st.session_state.log_rows)
        sev_counts = df_sv['severity'].value_counts().reset_index()
        sev_counts.columns = ['Severity', 'Count']
        clr_map = {
            "No Tremor":       "#00c853",
            "Mild Tremor":     "#ff9800",
            "Moderate Tremor": "#ff5722",
            "Severe Tremor":   "#f44336",
        }
        colors = [clr_map.get(s, ACCENT) for s in sev_counts['Severity']]
        fig_sv = go.Figure(go.Bar(
            x=sev_counts['Severity'], y=sev_counts['Count'],
            marker_color=colors, marker_line_width=0,
        ))
        fig_sv.update_layout(
            **base_layout("▬ SEVERITY DISTRIBUTION", "Class", "Count"),
            height=220,
        )
        sev_ph.plotly_chart(fig_sv, use_container_width=True, key="sp")

    # ── Auto-refresh ────────────────────────────────────────────────────────
    time.sleep(update_ms / 1000)
    st.rerun()
