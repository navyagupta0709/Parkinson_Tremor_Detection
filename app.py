"""
⚡ TENG Parkinson Tremor Detection System
Real-Time Signal Processing · FFT Spectrum Analysis · IEEE Research Dashboard

Hardware : Arduino UNO + TENG sensor
Protocol : Serial @ 9600 baud  →  "time,voltage\n"
Fs       : 100 Hz
Author   : Research Dashboard v1.0
"""

import time
import threading
import collections
import warnings

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import rfft, rfftfreq
from scipy.signal import (
    butter, sosfiltfilt, detrend,
    savgol_filter, welch
)

warnings.filterwarnings("ignore")

# ── try importing pyserial; graceful fallback ──────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

FS           = 100          # Hz  – must match Arduino sketch
BUFFER_SEC   = 6            # seconds of rolling data kept in memory
BUFFER_LEN   = FS * BUFFER_SEC
MIN_SAMPLES  = FS * 2       # need at least 2 s before first FFT

TREMOR_LO    = 3.0          # Hz  Parkinsonian tremor band
TREMOR_HI    = 7.0

SEVERITY_MAP = [
    (3.0, 4.0,   "Mild",     "#FFA500"),
    (4.0, 5.5,   "Moderate", "#FF6600"),
    (5.5, 7.0,   "Severe",   "#FF0000"),
]

DARK_BG      = "#0E1117"
PANEL_BG     = "#1A1D27"
ACCENT       = "#00BFFF"
GRID_COLOR   = "#2A2D3A"
TEXT_COLOR   = "#E0E0E0"


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TENG Tremor Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] {
        background-color: #12151F;
        border-right: 1px solid #2A2D3A;
    }
    .block-container { padding-top: 1rem; }

    /* metric cards */
    .metric-card {
        background: #1A1D27;
        border: 1px solid #2A2D3A;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00BFFF;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #8A8D9A;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* alert boxes */
    .alert-tremor {
        background: linear-gradient(135deg, #3a0000 0%, #1a0000 100%);
        border: 2px solid #FF0000;
        border-radius: 10px;
        padding: 18px 24px;
        text-align: center;
        animation: pulse 1.2s infinite;
    }
    .alert-normal {
        background: linear-gradient(135deg, #003a00 0%, #001a00 100%);
        border: 2px solid #00CC44;
        border-radius: 10px;
        padding: 18px 24px;
        text-align: center;
    }
    .alert-idle {
        background: #1A1D27;
        border: 1px solid #2A2D3A;
        border-radius: 10px;
        padding: 18px 24px;
        text-align: center;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0   0   rgba(255,0,0,0.5); }
        70%  { box-shadow: 0 0 0  12px rgba(255,0,0,0);   }
        100% { box-shadow: 0 0 0   0   rgba(255,0,0,0);   }
    }

    div[data-testid="stHorizontalBlock"] > div { gap: 0.6rem; }
    .stTabs [data-baseweb="tab"] { color: #8A8D9A; }
    .stTabs [aria-selected="true"] { color: #00BFFF !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE  –  shared buffer between serial thread and UI
# ═══════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "running":        False,
        "ser":            None,
        "thread":         None,
        "raw_buf":        collections.deque(maxlen=BUFFER_LEN),
        "time_buf":       collections.deque(maxlen=BUFFER_LEN),
        "port_status":    "Disconnected",
        "pkt_count":      0,
        "err_count":      0,
        "dom_freq":       0.0,
        "amplitude":      0.0,
        "snr":            0.0,
        "severity":       "—",
        "sev_color":      "#888888",
        "confidence":     0.0,
        "tremor_flag":    False,
        "threshold":      0.05,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
ss = st.session_state   # shorthand


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def butterworth_bandpass(lowcut=0.5, highcut=10.0, fs=FS, order=4):
    nyq  = fs / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    sos  = butter(order, [low, high], btype="band", output="sos")
    return sos


_BPF = butterworth_bandpass()   # pre-computed once


def process_signal(raw: np.ndarray, use_savgol: bool = True):
    """
    Full pipeline:
      raw → remove invalid → DC removal → detrend
          → 4th-order Butterworth BPF (0.5–10 Hz)
          → optional Savitzky-Golay smoothing
    Returns processed signal (same length as input).
    """
    sig = raw.copy()

    # 1. remove NaN / Inf
    sig = sig[np.isfinite(sig)]
    if sig.size < 10:
        return np.zeros_like(raw)

    # 2. DC offset removal
    sig = sig - np.mean(sig)

    # 3. linear detrend
    sig = detrend(sig, type="linear")

    # 4. Butterworth bandpass
    sig = sosfiltfilt(_BPF, sig)

    # 5. optional Savitzky-Golay
    if use_savgol and len(sig) >= 11:
        wl = min(11, len(sig) if len(sig) % 2 == 1 else len(sig) - 1)
        sig = savgol_filter(sig, window_length=wl, polyorder=3)

    return sig


def compute_fft(sig: np.ndarray, fs: float = FS):
    """Returns (freqs, magnitudes) for positive half of spectrum."""
    n    = len(sig)
    yf   = np.abs(rfft(sig * np.hanning(n)))
    xf   = rfftfreq(n, d=1.0 / fs)
    # normalise
    yf   = yf / (n / 2)
    return xf, yf


def compute_psd(sig: np.ndarray, fs: float = FS):
    """Welch PSD estimate."""
    nperseg = min(256, len(sig))
    freqs, pxx = welch(sig, fs=fs, nperseg=nperseg)
    return freqs, pxx


def find_dominant(xf, yf, flo=0.5, fhi=15.0):
    """Dominant frequency peak within [flo, fhi] Hz."""
    mask = (xf >= flo) & (xf <= fhi)
    if not mask.any():
        return 0.0, 0.0
    idx  = np.argmax(yf[mask])
    freqs_m = xf[mask]
    mags_m  = yf[mask]
    return float(freqs_m[idx]), float(mags_m[idx])


def classify_tremor(dom_freq: float, amplitude: float, threshold: float):
    """
    Returns (tremor_flag, severity_label, severity_color, confidence).
    Confidence is based on how well amplitude clears the threshold.
    """
    in_band = TREMOR_LO <= dom_freq <= TREMOR_HI

    if not in_band or amplitude < threshold:
        return False, "Normal", "#00CC44", 0.0

    # severity
    label = "Mild"
    color = "#FFA500"
    for lo, hi, lbl, clr in SEVERITY_MAP:
        if lo <= dom_freq < hi:
            label = lbl
            color = clr
            break

    # confidence: sigmoid-like based on amplitude vs threshold
    ratio      = min(amplitude / (threshold * 1.5), 1.0)
    confidence = round(ratio * 100, 1)

    return True, label, color, confidence


# ═══════════════════════════════════════════════════════════════════════════
#  SERIAL READER THREAD
# ═══════════════════════════════════════════════════════════════════════════

def serial_reader(port: str, baud: int = 9600):
    """
    Runs in a daemon thread. Reads 'time,voltage' lines from Arduino
    and pushes voltage values into the shared rolling buffer.
    """
    try:
        ser = serial.Serial(port, baud, timeout=1)
        ss["ser"]         = ser
        ss["port_status"] = f"Connected · {port} @ {baud}"
        ser.reset_input_buffer()
        time.sleep(0.5)

        while ss["running"]:
            try:
                raw_line = ser.readline()
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 2:
                    ss["err_count"] += 1
                    continue

                t_val = float(parts[0])
                v_val = float(parts[1])

                ss["raw_buf"].append(v_val)
                ss["time_buf"].append(t_val)
                ss["pkt_count"] += 1

            except (ValueError, UnicodeDecodeError):
                ss["err_count"] += 1
            except serial.SerialException:
                break

    except serial.SerialException as e:
        ss["port_status"] = f"Error: {e}"
    finally:
        if ss.get("ser") and ss["ser"].is_open:
            ss["ser"].close()
        ss["port_status"] = "Disconnected"
        ss["running"]     = False


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS  (dark IEEE-style)
# ═══════════════════════════════════════════════════════════════════════════

LAYOUT_BASE = dict(
    paper_bgcolor=PANEL_BG,
    plot_bgcolor =PANEL_BG,
    font         =dict(color=TEXT_COLOR, size=11),
    margin       =dict(l=50, r=20, t=40, b=40),
    xaxis        =dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis        =dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
)


def fig_raw(t_arr, v_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_arr, y=v_arr,
        mode="lines",
        line=dict(color=ACCENT, width=1.2),
        name="Raw TENG"
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Live Raw TENG Signal", font=dict(size=13)),
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
        height=300,
    )
    return fig


def fig_processed(t_arr, raw_arr, proc_arr):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=t_arr, y=raw_arr,
        mode="lines", line=dict(color="#888888", width=1),
        name="Raw"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_arr, y=proc_arr,
        mode="lines", line=dict(color=ACCENT, width=1.4),
        name="Filtered"
    ), row=2, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Signal Processing Pipeline", font=dict(size=13)),
        height=380,
    )
    fig.update_yaxes(title_text="Raw (V)",      row=1, col=1,
                     gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Filtered (V)", row=2, col=1,
                     gridcolor=GRID_COLOR)
    fig.update_xaxes(title_text="Time (s)",     row=2, col=1,
                     gridcolor=GRID_COLOR)
    fig.update_xaxes(gridcolor=GRID_COLOR, row=1, col=1)
    return fig


def fig_fft(xf, yf, dom_freq):
    # find harmonic peaks (multiples of dominant)
    harm_freqs, harm_mags = [], []
    if dom_freq > 0:
        for h in range(1, 5):
            hf = dom_freq * h
            mask = np.abs(xf - hf) < 0.3
            if mask.any():
                idx = np.argmax(yf[mask])
                harm_freqs.append(xf[mask][idx])
                harm_mags.append(yf[mask][idx])

    fig = go.Figure()

    # tremor band shading
    fig.add_vrect(
        x0=TREMOR_LO, x1=TREMOR_HI,
        fillcolor="rgba(255,50,50,0.12)",
        layer="below", line_width=0,
        annotation_text="Parkinsonian 3–7 Hz",
        annotation_position="top left",
        annotation_font=dict(color="#FF6666", size=10),
    )

    # main spectrum
    fig.add_trace(go.Scatter(
        x=xf, y=yf,
        mode="lines",
        line=dict(color=ACCENT, width=1.5),
        name="FFT Magnitude",
        fill="tozeroy",
        fillcolor="rgba(0,191,255,0.08)",
    ))

    # dominant peak marker
    if dom_freq > 0:
        dom_mag = float(yf[np.argmin(np.abs(xf - dom_freq))])
        fig.add_trace(go.Scatter(
            x=[dom_freq], y=[dom_mag],
            mode="markers+text",
            marker=dict(color="#FFD700", size=10, symbol="diamond"),
            text=[f"  {dom_freq:.2f} Hz"],
            textposition="top right",
            textfont=dict(color="#FFD700", size=11),
            name="Dominant Peak",
        ))

    # harmonic markers
    if harm_freqs:
        fig.add_trace(go.Scatter(
            x=harm_freqs, y=harm_mags,
            mode="markers",
            marker=dict(color="#FF8C00", size=7, symbol="triangle-up"),
            name="Harmonics",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="FFT Spectrum Analysis", font=dict(size=13)),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (V)",
        xaxis_range=[0, 15],
        height=340,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


def fig_psd(freqs, pxx, dom_freq):
    fig = go.Figure()

    fig.add_vrect(
        x0=TREMOR_LO, x1=TREMOR_HI,
        fillcolor="rgba(255,50,50,0.12)",
        layer="below", line_width=0,
    )

    fig.add_trace(go.Scatter(
        x=freqs, y=10 * np.log10(pxx + 1e-12),
        mode="lines",
        line=dict(color="#9B59B6", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(155,89,182,0.08)",
        name="PSD (Welch)",
    ))

    if dom_freq > 0:
        mask  = np.abs(freqs - dom_freq) < 0.5
        if mask.any():
            pxx_db = 10 * np.log10(pxx[mask][0] + 1e-12)
            fig.add_trace(go.Scatter(
                x=[dom_freq], y=[pxx_db],
                mode="markers",
                marker=dict(color="#FFD700", size=9, symbol="diamond"),
                name=f"Peak {dom_freq:.2f} Hz",
            ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Power Spectral Density (Welch)", font=dict(size=13)),
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD (dB/Hz)",
        xaxis_range=[0, 15],
        height=340,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px'>
      <div style='font-size:2rem'>⚡</div>
      <div style='font-size:1rem; font-weight:700; color:#00BFFF'>TENG Tremor System</div>
      <div style='font-size:0.7rem; color:#888; margin-top:4px'>IEEE Research Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔌 Serial Port")

    if SERIAL_AVAILABLE:
        ports     = [p.device for p in serial.tools.list_ports.comports()]
        port_opts = ports if ports else ["COM3", "COM4", "/dev/ttyUSB0"]
    else:
        port_opts = ["COM3", "COM4", "/dev/ttyUSB0", "/dev/ttyACM0"]

    sel_port = st.selectbox("COM Port", port_opts)
    sel_baud = st.selectbox("Baud Rate", [9600, 115200], index=0)

    st.markdown("---")

    # Start / Stop button
    if not ss["running"]:
        if st.button("▶  Start Live Detection", use_container_width=True, type="primary"):
            if not SERIAL_AVAILABLE:
                st.error("pyserial not installed. Run: pip install pyserial")
            else:
                ss["running"]    = True
                ss["pkt_count"]  = 0
                ss["err_count"]  = 0
                ss["raw_buf"].clear()
                ss["time_buf"].clear()
                t = threading.Thread(
                    target=serial_reader,
                    args=(sel_port, sel_baud),
                    daemon=True,
                )
                t.start()
                ss["thread"] = t
                st.rerun()
    else:
        if st.button("⏹  Stop Detection", use_container_width=True):
            ss["running"] = False
            st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ Signal Settings")
    st.metric("Sampling Frequency", f"{FS} Hz")

    threshold = st.slider(
        "Tremor Amplitude Threshold (V)",
        min_value=0.001, max_value=0.5,
        value=ss["threshold"], step=0.001, format="%.3f"
    )
    ss["threshold"] = threshold

    use_savgol = st.toggle("Savitzky-Golay Smoothing", value=True)

    st.markdown("---")
    st.markdown("#### 📡 Port Status")
    status_color = "#00CC44" if ss["running"] else "#FF4444"
    st.markdown(f"""
    <div style='padding:8px; background:#12151F; border-radius:6px;
                border-left:3px solid {status_color};
                font-size:0.78rem; color:#CCC;'>
      {ss['port_status']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:8px; font-size:0.75rem; color:#888;'>
      Packets received : <b style='color:#CCC'>{ss['pkt_count']}</b><br>
      Parse errors     : <b style='color:#FF8888'>{ss['err_count']}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    #### ℹ️ About
    <div style='font-size:0.75rem; color:#888; line-height:1.6'>
    TENG-based tremor detection prototype.<br>
    Parkinsonian tremor band: <b style='color:#FF6666'>3–7 Hz</b><br>
    Filter: 4th-order Butterworth BPF<br>
    FFT: scipy.fft · PSD: Welch method<br><br>
    Arduino → Serial → Streamlit → FFT
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='text-align:center; padding:10px 0 4px'>
  <h1 style='margin:0; font-size:1.9rem; color:#00BFFF; letter-spacing:1px'>
    ⚡ TENG Parkinson Tremor Detection System
  </h1>
  <p style='color:#8A8D9A; font-size:0.85rem; margin:4px 0 0'>
    Real-Time Signal Processing &nbsp;·&nbsp; FFT Spectrum Analysis
    &nbsp;·&nbsp; IEEE Research Dashboard
  </p>
</div>
<hr style='border:none; border-top:1px solid #2A2D3A; margin:12px 0'>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA SNAPSHOT  (atomic read from the deque)
# ═══════════════════════════════════════════════════════════════════════════

raw_snap  = np.array(list(ss["raw_buf"]),  dtype=np.float64)
time_snap = np.array(list(ss["time_buf"]), dtype=np.float64)

have_data   = raw_snap.size >= MIN_SAMPLES
proc_snap   = np.array([])
xf = yf     = freqs = pxx = np.array([])
dom_freq    = 0.0
dom_mag     = 0.0
amplitude   = 0.0
tremor_flag = False
severity    = "—"
sev_color   = "#888888"
confidence  = 0.0
snr_db      = 0.0

if have_data:
    proc_snap   = process_signal(raw_snap, use_savgol=use_savgol)
    xf, yf      = compute_fft(proc_snap)
    freqs, pxx  = compute_psd(proc_snap)
    dom_freq, dom_mag = find_dominant(xf, yf)
    amplitude   = float(np.std(proc_snap))
    tremor_flag, severity, sev_color, confidence = classify_tremor(
        dom_freq, amplitude, ss["threshold"]
    )
    # SNR estimate
    noise_mask  = (xf < 0.4) | (xf > 12.0)
    if noise_mask.any() and yf[noise_mask].max() > 0:
        snr_db = float(20 * np.log10(dom_mag / (yf[noise_mask].mean() + 1e-9)))
    snr_db = max(0.0, min(snr_db, 60.0))

    # cache for sidebar stats
    ss["dom_freq"]    = dom_freq
    ss["amplitude"]   = amplitude
    ss["snr"]         = snr_db
    ss["severity"]    = severity
    ss["sev_color"]   = sev_color
    ss["confidence"]  = confidence
    ss["tremor_flag"] = tremor_flag


# ═══════════════════════════════════════════════════════════════════════════
#  TOP METRIC ROW
# ═══════════════════════════════════════════════════════════════════════════

m1, m2, m3, m4, m5 = st.columns(5)

def metric_html(value, label):
    return f"""
    <div class='metric-card'>
      <div class='metric-value'>{value}</div>
      <div class='metric-label'>{label}</div>
    </div>"""

with m1:
    st.markdown(metric_html(
        f"{dom_freq:.2f} Hz" if have_data else "— Hz",
        "Dominant Frequency"
    ), unsafe_allow_html=True)

with m2:
    st.markdown(metric_html(
        f"{amplitude*1000:.1f} mV" if have_data else "— mV",
        "Tremor Amplitude"
    ), unsafe_allow_html=True)

with m3:
    sq_color = "#00CC44" if snr_db > 20 else "#FFA500" if snr_db > 10 else "#FF4444"
    sq_label = "Good" if snr_db > 20 else "Fair" if snr_db > 10 else "Poor"
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value' style='color:{sq_color}'>{sq_label}</div>
      <div class='metric-label'>Signal Quality ({snr_db:.1f} dB)</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(metric_html(
        f"{ss['pkt_count']}",
        "Samples Received"
    ), unsafe_allow_html=True)

with m5:
    run_color = "#00CC44" if ss["running"] else "#FF4444"
    run_label = "● LIVE" if ss["running"] else "○ IDLE"
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value' style='color:{run_color}'>{run_label}</div>
      <div class='metric-label'>Acquisition Status</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  DETECTION ALERT  (full width)
# ═══════════════════════════════════════════════════════════════════════════

if not have_data:
    st.markdown("""
    <div class='alert-idle'>
      <div style='font-size:1.5rem'>⏳</div>
      <div style='font-size:1.1rem; font-weight:600; color:#888; margin-top:6px'>
        Waiting for Serial Data…
      </div>
      <div style='font-size:0.8rem; color:#666; margin-top:4px'>
        Connect Arduino and press ▶ Start Live Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

elif tremor_flag:
    st.markdown(f"""
    <div class='alert-tremor'>
      <div style='font-size:2rem'>🚨</div>
      <div style='font-size:1.6rem; font-weight:800; color:#FF4444; letter-spacing:2px;
                  margin-top:4px'>
        TREMOR DETECTED
      </div>
      <div style='display:flex; justify-content:center; gap:40px; margin-top:10px'>
        <div>
          <div style='font-size:1.3rem; font-weight:700; color:#FFD700'>
            {dom_freq:.2f} Hz
          </div>
          <div style='font-size:0.7rem; color:#FF8888; text-transform:uppercase'>
            Frequency
          </div>
        </div>
        <div>
          <div style='font-size:1.3rem; font-weight:700; color:{sev_color}'>
            {severity}
          </div>
          <div style='font-size:0.7rem; color:#FF8888; text-transform:uppercase'>
            Severity
          </div>
        </div>
        <div>
          <div style='font-size:1.3rem; font-weight:700; color:#FF8C00'>
            {confidence:.1f}%
          </div>
          <div style='font-size:0.7rem; color:#FF8888; text-transform:uppercase'>
            Confidence
          </div>
        </div>
        <div>
          <div style='font-size:1.3rem; font-weight:700; color:#FFD700'>
            Parkinsonian
          </div>
          <div style='font-size:0.7rem; color:#FF8888; text-transform:uppercase'>
            Classification
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class='alert-normal'>
      <div style='font-size:1.6rem; font-weight:700; color:#00CC44; letter-spacing:2px'>
        ✅ NORMAL — No Tremor Detected
      </div>
      <div style='font-size:0.85rem; color:#88CC88; margin-top:6px'>
        Dominant frequency {dom_freq:.2f} Hz is outside the 3–7 Hz Parkinsonian band
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TABBED GRAPHS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Raw Signal",
    "🔧 Signal Processing",
    "📊 FFT Spectrum",
    "🌊 PSD Analysis",
])

# ── TAB 1 · Raw Signal ────────────────────────────────────────────────────
with tab1:
    if have_data:
        # build time axis from sample count if Arduino time resets
        t_axis = time_snap if time_snap.size == raw_snap.size else \
                 np.arange(len(raw_snap)) / FS
        st.plotly_chart(fig_raw(t_axis, raw_snap),
                        use_container_width=True)
        st.markdown(f"""
        <div style='font-size:0.78rem; color:#888; text-align:right'>
          Buffer: {len(raw_snap)} samples &nbsp;|&nbsp;
          {len(raw_snap)/FS:.1f} s &nbsp;|&nbsp;
          Min {raw_snap.min():.4f} V &nbsp;|&nbsp;
          Max {raw_snap.max():.4f} V &nbsp;|&nbsp;
          RMS {float(np.sqrt(np.mean(raw_snap**2))):.4f} V
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Collecting data… (need ≥ 2 seconds)")

# ── TAB 2 · Signal Processing ─────────────────────────────────────────────
with tab2:
    if have_data and proc_snap.size > 0:
        t_axis = time_snap if time_snap.size == raw_snap.size else \
                 np.arange(len(raw_snap)) / FS
        st.plotly_chart(fig_processed(t_axis, raw_snap, proc_snap),
                        use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-value' style='font-size:1.2rem'>
                0.5 – 10 Hz
              </div>
              <div class='metric-label'>Butterworth BPF (4th order)</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            dc = float(np.mean(raw_snap))
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-value' style='font-size:1.2rem'>
                {dc:.4f} V
              </div>
              <div class='metric-label'>DC Offset Removed</div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            sg_txt = "Enabled" if use_savgol else "Disabled"
            sg_col = "#00CC44" if use_savgol else "#888"
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-value' style='font-size:1.2rem; color:{sg_col}'>
                {sg_txt}
              </div>
              <div class='metric-label'>Savitzky-Golay (w=11, p=3)</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Collecting data…")

# ── TAB 3 · FFT Spectrum ──────────────────────────────────────────────────
with tab3:
    if have_data and xf.size > 0:
        st.plotly_chart(fig_fft(xf, yf, dom_freq),
                        use_container_width=True)

        st.markdown("**Frequency Band Energy Distribution**")
        bands = [
            ("0.5 – 3 Hz",  0.5, 3.0,  "#4488FF"),
            ("3 – 7 Hz",    3.0, 7.0,  "#FF4444"),
            ("7 – 10 Hz",   7.0, 10.0, "#FFA500"),
            ("10 – 15 Hz", 10.0, 15.0, "#888888"),
        ]
        total_energy = float(np.trapz(yf**2, xf)) + 1e-12
        b_cols = st.columns(len(bands))
        for col, (lbl, flo, fhi, clr) in zip(b_cols, bands):
            mask  = (xf >= flo) & (xf <= fhi)
            e_pct = float(np.trapz(yf[mask]**2, xf[mask])) / total_energy * 100
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-value' style='font-size:1.1rem; color:{clr}'>
                    {e_pct:.1f}%
                  </div>
                  <div class='metric-label'>{lbl}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Collecting data…")

# ── TAB 4 · PSD Analysis ──────────────────────────────────────────────────
with tab4:
    if have_data and freqs.size > 0:
        st.plotly_chart(fig_psd(freqs, pxx, dom_freq),
                        use_container_width=True)

        mask_tremor = (freqs >= TREMOR_LO) & (freqs <= TREMOR_HI)
        mask_all    = (freqs >= 0.5) & (freqs <= 15.0)
        p_tremor    = float(np.trapz(pxx[mask_tremor], freqs[mask_tremor]))
        p_total     = float(np.trapz(pxx[mask_all],    freqs[mask_all])) + 1e-12
        band_pct    = p_tremor / p_total * 100

        st.markdown(f"""
        <div style='display:flex; gap:12px; margin-top:8px'>
          <div class='metric-card' style='flex:1'>
            <div class='metric-value' style='font-size:1.1rem'>{p_tremor:.2e} V²/Hz</div>
            <div class='metric-label'>Tremor Band Power (3–7 Hz)</div>
          </div>
          <div class='metric-card' style='flex:1'>
            <div class='metric-value' style='font-size:1.1rem; color:#FF6666'>
              {band_pct:.1f}%
            </div>
            <div class='metric-label'>% Total Power in Tremor Band</div>
          </div>
          <div class='metric-card' style='flex:1'>
            <div class='metric-value' style='font-size:1.1rem'>{snr_db:.1f} dB</div>
            <div class='metric-label'>Estimated SNR</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Collecting data…")


# ═══════════════════════════════════════════════════════════════════════════
#  AUTO-REFRESH
# ═══════════════════════════════════════════════════════════════════════════

if ss["running"]:
    time.sleep(0.15)      # ~6–7 UI refreshes per second
    st.rerun()
