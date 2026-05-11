"""
⚡ TENG Parkinson Tremor Detection System
──────────────────────────────────────────
Architecture  : Wearable TENG → Arduino UNO → Serial/WiFi → Dashboard
Serial format : "time_s,voltage_V"  @  9600 baud, 100 Hz
Demo mode     : Replays real READINGS.xlsx data when no Arduino is connected
"""

import time, queue, threading, warnings, gzip, base64, json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.fft        import fft, fftfreq
from scipy.signal     import butter, sosfiltfilt, detrend, savgol_filter, welch

warnings.filterwarnings("ignore")

# ── Optional pyserial ────────────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

# ── Load embedded real signal data (READINGS.xlsx, compressed) ───────────────
from signal_data import SIGNAL_DATA_B64
_raw_signals: dict = json.loads(gzip.decompress(base64.b64decode(SIGNAL_DATA_B64)))
# keys: '60','120','180','240','300','360','420'  → list of 3 sets × 2001 samples

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
FS           = 100          # Hz
BUFFER_MAX   = 1000         # rolling window (10 s @ 100 Hz)
REFRESH_MS   = 160          # UI rerun interval
FFT_WIN      = 512
TREMOR_LO    = 3.0          # Hz
TREMOR_HI    = 7.0          # Hz
DEMO_BPM_SEQ = [60, 60, 120, 180, 240, 300, 360, 420, 420, 360, 300, 240]

# ── Colour palette ───────────────────────────────────────────────────────────
C_BG     = "#0a0e1a"
C_PANEL  = "#0f1623"
C_BORDER = "#1e2736"
C_ACCENT = "#4f9cf9"
C_GREEN  = "#22c55e"
C_RED    = "#ef4444"
C_YELLOW = "#f59e0b"
C_PURPLE = "#a78bfa"
C_TEXT   = "#e2e8f0"
C_MUTED  = "#64748b"

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TENG Tremor Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  html, body, .stApp          {{ background:{C_BG} !important; color:{C_TEXT}; }}
  section[data-testid="stSidebar"] {{ background:{C_PANEL} !important;
                                      border-right:1px solid {C_BORDER}; }}
  .block-container             {{ padding-top:.8rem; padding-bottom:.8rem; }}
  div[data-testid="stTabs"]    {{ background:transparent; }}

  /* metric cards */
  div[data-testid="metric-container"] {{
      background:{C_PANEL}; border:1px solid {C_BORDER};
      border-radius:10px; padding:.55rem .9rem;
  }}
  div[data-testid="metric-container"] label {{
      color:{C_MUTED} !important; font-size:.68rem !important;
      letter-spacing:.07em; text-transform:uppercase;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
      color:{C_ACCENT} !important; font-size:1.45rem !important; font-weight:700;
  }}

  /* alert boxes */
  .alert-tremor {{
      background:rgba(239,68,68,.13); border:2px solid {C_RED};
      border-radius:12px; padding:1.1rem 1.4rem; text-align:center;
      animation:pulse-red 1.1s ease-in-out infinite;
  }}
  .alert-normal {{
      background:rgba(34,197,94,.10); border:2px solid {C_GREEN};
      border-radius:12px; padding:1.1rem 1.4rem; text-align:center;
  }}
  .alert-idle {{
      background:rgba(79,156,249,.07); border:2px solid {C_BORDER};
      border-radius:12px; padding:1.1rem 1.4rem; text-align:center;
  }}
  @keyframes pulse-red {{
      0%   {{ box-shadow:0 0 0 0 rgba(239,68,68,.6); }}
      70%  {{ box-shadow:0 0 0 16px rgba(239,68,68,0); }}
      100% {{ box-shadow:0 0 0 0 rgba(239,68,68,0); }}
  }}

  /* status dot */
  .dot-live   {{ display:inline-block;width:9px;height:9px;border-radius:50%;
                 background:{C_RED};animation:blink .8s ease-in-out infinite;
                 vertical-align:middle;margin-right:5px; }}
  .dot-idle   {{ display:inline-block;width:9px;height:9px;border-radius:50%;
                 background:{C_MUTED};vertical-align:middle;margin-right:5px; }}
  @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.25}} }}

  /* status bar pills */
  .pill {{
      display:inline-block; border-radius:6px; padding:3px 10px;
      font-size:.72rem; font-weight:600; letter-spacing:.05em; margin:1px;
  }}
  .pill-green {{ background:rgba(34,197,94,.15); color:{C_GREEN};
                 border:1px solid rgba(34,197,94,.35); }}
  .pill-red   {{ background:rgba(239,68,68,.15);  color:{C_RED};
                 border:1px solid rgba(239,68,68,.35); }}
  .pill-blue  {{ background:rgba(79,156,249,.15); color:{C_ACCENT};
                 border:1px solid rgba(79,156,249,.35); }}
  .pill-grey  {{ background:rgba(100,116,139,.15);color:{C_MUTED};
                 border:1px solid rgba(100,116,139,.35); }}
  .pill-yellow{{ background:rgba(245,158,11,.15); color:{C_YELLOW};
                 border:1px solid rgba(245,158,11,.35); }}

  hr {{ border-color:{C_BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
_defaults = dict(
    running=False, mode="idle",
    raw_buf=[], time_buf=[],
    serial_obj=None,
    data_queue=queue.Queue(maxsize=5000),
    port_status="Disconnected",
    sample_count=0, drop_count=0,
    dom_freq=0.0, amplitude=0.0,
    sig_quality=0.0, band_ratio=0.0,
    tremor_flag=False, severity="—",
    confidence=0.0,
    # demo playback state
    _demo_bpm_idx=0, _demo_sample_ptr=0, _demo_elapsed=0.0,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
S = st.session_state

# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING  (mirrors notebook pipeline exactly)
# ════════════════════════════════════════════════════════════════════════════

def _remove_invalid(sig: np.ndarray) -> np.ndarray:
    sig = sig[np.isfinite(sig)]
    if len(sig) < 4:
        return sig
    q1, q3 = np.percentile(sig, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    return sig[(sig >= lo) & (sig <= hi)]

def process_signal(sig: np.ndarray, fs: float = FS) -> np.ndarray:
    s = _remove_invalid(sig.copy())
    if len(s) < 20:
        return s
    s -= np.mean(s)                                                 # 1. DC removal
    s  = detrend(s)                                                 # 2. detrend
    sos = butter(4, [0.5, 10.0], btype="bandpass", fs=fs, output="sos")
    s  = sosfiltfilt(sos, s)                                        # 3. bandpass
    wl = 11 if len(s) >= 11 else (len(s) | 1)
    if wl >= 5:
        s = savgol_filter(s, window_length=wl, polyorder=3)        # 4. SG smooth
    return s

def compute_fft(sig: np.ndarray, fs: float = FS):
    n   = len(sig)
    mag = np.abs(fft(sig))
    f   = fftfreq(n, d=1.0 / fs)
    pos = f > 0
    return f[pos], mag[pos]

def compute_psd(sig: np.ndarray, fs: float = FS):
    nperseg = min(len(sig), 256)
    f, psd  = welch(sig, fs=fs, nperseg=nperseg)
    return f, psd

def classify(dom_freq: float, amplitude: float, thresh: float):
    """Returns (tremor_bool, severity_str, confidence_pct)."""
    if not (TREMOR_LO <= dom_freq <= TREMOR_HI) or amplitude < thresh:
        return False, "Normal", 0.0
    centre     = (TREMOR_LO + TREMOR_HI) / 2.0
    half_width = (TREMOR_HI - TREMOR_LO) / 2.0
    conf = max(0.0, (1.0 - abs(dom_freq - centre) / half_width) * 100.0)
    for lo, hi, lbl in [(3.0, 4.0, "Mild"), (4.0, 5.5, "Moderate"), (5.5, 7.1, "Severe")]:
        if lo <= dom_freq < hi:
            return True, lbl, round(conf, 1)
    return True, "Severe", round(conf, 1)

def sig_quality_score(proc: np.ndarray, fs: float = FS) -> float:
    if len(proc) < 10:
        return 0.0
    f, psd  = compute_psd(proc, fs)
    total   = np.trapezoid(psd, f) + 1e-12
    inband  = np.trapezoid(psd[(f >= 0.5) & (f <= 10.0)],
                           f[(f >= 0.5) & (f <= 10.0)])
    return round(min(inband / total * 100.0, 100.0), 1)

# ════════════════════════════════════════════════════════════════════════════
#  SERIAL READER THREAD
# ════════════════════════════════════════════════════════════════════════════

def _serial_reader(port, baud, q: queue.Queue, stop: threading.Event):
    try:
        ser = serial.Serial(port, baud, timeout=1.0)
        S["port_status"] = f"Connected · {port}"
        S["serial_obj"]  = ser
        while not stop.is_set():
            raw = ser.readline()
            if not raw:
                continue
            try:
                line  = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                t_v, v_v = float(parts[0]), float(parts[1])
                if not (np.isfinite(t_v) and np.isfinite(v_v)):
                    S["drop_count"] += 1
                    continue
                if not q.full():
                    q.put_nowait((t_v, v_v))
            except Exception:
                S["drop_count"] += 1
        ser.close()
        S["port_status"] = "Disconnected"
        S["serial_obj"]  = None
    except Exception as exc:
        S["port_status"] = f"Error: {exc}"

# ════════════════════════════════════════════════════════════════════════════
#  DEMO DATA FEEDER  (replays real READINGS.xlsx signals)
# ════════════════════════════════════════════════════════════════════════════

def _feed_demo_chunk(q: queue.Queue, n_samples: int = 16):
    """Push ~16 samples from the current real-signal segment into q."""
    bpm_list = DEMO_BPM_SEQ
    bpm_key  = str(bpm_list[S["_demo_bpm_idx"] % len(bpm_list)])
    sig_set  = _raw_signals[bpm_key][0]          # use SET 1 always
    ptr      = S["_demo_sample_ptr"]
    pushed   = 0
    for _ in range(n_samples):
        idx = ptr % len(sig_set)
        voltage = sig_set[idx]
        t_val   = S["_demo_elapsed"]
        if not q.full():
            q.put_nowait((t_val, voltage))
        ptr    += 1
        S["_demo_elapsed"] += 1.0 / FS
        pushed += 1
        # advance BPM segment every 300 samples (~3 s)
        if ptr % 300 == 0:
            S["_demo_bpm_idx"] = (S["_demo_bpm_idx"] + 1) % len(bpm_list)
    S["_demo_sample_ptr"] = ptr

# ════════════════════════════════════════════════════════════════════════════
#  PLOTLY DARK THEME HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _base_layout(title="", xl="", yl="", h=290) -> dict:
    return dict(
        title        = dict(text=title, font=dict(color=C_TEXT, size=12, family="monospace")),
        xaxis        = dict(title=xl, color=C_TEXT, gridcolor=C_BORDER,
                            zerolinecolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10)),
        yaxis        = dict(title=yl, color=C_TEXT, gridcolor=C_BORDER,
                            zerolinecolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10)),
        plot_bgcolor  = C_PANEL,
        paper_bgcolor = C_PANEL,
        font          = dict(color=C_TEXT, family="monospace"),
        margin        = dict(l=52, r=16, t=36, b=38),
        height        = h,
        legend        = dict(bgcolor="rgba(0,0,0,0)",
                             font=dict(color=C_TEXT, size=9), x=0.01, y=0.99),
    )

def fig_raw(t_arr, raw_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_arr, y=raw_arr, mode="lines", name="Raw TENG",
        line=dict(color=C_ACCENT, width=1.1),
    ))
    lo = _base_layout("Raw TENG Signal", "Time (s)", "Voltage (V)", h=270)
    fig.update_layout(**lo)
    return fig

def fig_processed(t_arr, raw_arr, proc_arr):
    fig = go.Figure()
    if len(raw_arr):
        fig.add_trace(go.Scatter(
            x=t_arr, y=raw_arr, mode="lines", name="Raw",
            line=dict(color=C_MUTED, width=.9), opacity=.55,
        ))
    if len(proc_arr):
        fig.add_trace(go.Scatter(
            x=t_arr[-len(proc_arr):], y=proc_arr, mode="lines", name="Filtered",
            line=dict(color=C_GREEN, width=1.5),
        ))
    lo = _base_layout("Signal Processing Pipeline — Raw vs Filtered", "Time (s)", "Voltage (V)", h=270)
    fig.update_layout(**lo)
    return fig

def fig_fft(freqs, amps, dom_freq, tremor_flag):
    peak_color = C_RED if tremor_flag else C_YELLOW
    fig = go.Figure()
    fig.add_vrect(x0=TREMOR_LO, x1=TREMOR_HI,
                  fillcolor="rgba(239,68,68,.10)", line_width=0,
                  annotation_text="Tremor Band 3–7 Hz",
                  annotation_font=dict(color=C_RED, size=9),
                  annotation_position="top left")
    if len(freqs):
        fig.add_trace(go.Scatter(
            x=freqs, y=amps, mode="lines", name="FFT",
            line=dict(color=C_ACCENT, width=1.4),
            fill="tozeroy", fillcolor="rgba(79,156,249,.07)",
        ))
        # dominant peak
        if 0 < dom_freq <= 20:
            pk = float(np.interp(dom_freq, freqs, amps))
            fig.add_trace(go.Scatter(
                x=[dom_freq], y=[pk], mode="markers+text",
                marker=dict(color=peak_color, size=11, symbol="circle",
                            line=dict(color="white", width=1.5)),
                text=[f"  {dom_freq:.2f} Hz"],
                textfont=dict(color=peak_color, size=10, family="monospace"),
                textposition="middle right", name="Peak", showlegend=False,
            ))
            # vertical line at peak
            fig.add_vline(x=dom_freq, line_dash="dot",
                          line_color=peak_color, line_width=1, opacity=.6)
    lo = _base_layout("FFT Spectrum — Real-Time", "Frequency (Hz)", "Amplitude", h=290)
    lo["xaxis"]["range"] = [0, 14]
    fig.update_layout(**lo)
    return fig

def fig_psd(psd_f, psd_v):
    fig = go.Figure()
    fig.add_vrect(x0=TREMOR_LO, x1=TREMOR_HI,
                  fillcolor="rgba(239,68,68,.09)", line_width=0)
    if len(psd_f):
        fig.add_trace(go.Scatter(
            x=psd_f, y=psd_v, mode="lines", name="PSD (Welch)",
            line=dict(color=C_PURPLE, width=1.5),
            fill="tozeroy", fillcolor="rgba(167,139,250,.07)",
        ))
    lo = _base_layout("Power Spectral Density — Welch Method", "Frequency (Hz)", "PSD (V²/Hz)", h=290)
    lo["xaxis"]["range"] = [0, 14]
    fig.update_layout(**lo)
    return fig

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
<div style='text-align:center;padding:.4rem 0 .6rem'>
  <div style='font-size:1.6rem'>⚡</div>
  <div style='color:{C_ACCENT};font-weight:800;font-size:1.05rem;letter-spacing:.04em'>
    TENG TREMOR SYSTEM
  </div>
  <div style='color:{C_MUTED};font-size:.68rem;letter-spacing:.06em;margin-top:2px'>
    IEEE RESEARCH DASHBOARD
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Connection mode ──────────────────────────────────────────────────────
    st.markdown(f"<div style='color:{C_MUTED};font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;margin-bottom:4px'>Connection Mode</div>", unsafe_allow_html=True)
    conn_mode = st.radio("", ["🔌  Arduino Serial", "📂  Demo (Real Data)"],
                         label_visibility="collapsed")
    use_serial = conn_mode.startswith("🔌")

    if use_serial:
        st.markdown(f"<div style='color:{C_MUTED};font-size:.7rem;text-transform:uppercase;margin-top:.6rem;margin-bottom:4px'>Serial Port</div>", unsafe_allow_html=True)

        # ── Scan ports ──────────────────────────────────────────────────────
        if "scanned_ports" not in S:
            S["scanned_ports"] = []

        scan_col, _ = st.columns([1, 0.01])
        with scan_col:
            if st.button("🔍  Scan Ports", use_container_width=True):
                if SERIAL_OK:
                    found = serial.tools.list_ports.comports()
                    S["scanned_ports"] = [(p.device, p.description) for p in found]
                else:
                    S["scanned_ports"] = []

        # Show scan results as a mini table
        if SERIAL_OK:
            live_ports = serial.tools.list_ports.comports()
            port_map   = {p.device: p.description for p in live_ports}
        else:
            port_map = {}

        all_found = list(port_map.keys())

        if all_found:
            rows = ""
            for dev, desc in port_map.items():
                is_arduino = any(k in desc.lower() for k in
                                 ["arduino", "ch340", "ch341", "cp210", "ftdi", "usb serial"])
                badge = (f"<span style='color:{C_GREEN};font-size:.65rem'>● Arduino</span>"
                         if is_arduino
                         else f"<span style='color:{C_MUTED};font-size:.65rem'>○ Serial</span>")
                rows += (f"<tr>"
                         f"<td style='color:{C_ACCENT};padding:2px 6px'>{dev}</td>"
                         f"<td style='color:{C_MUTED};padding:2px 4px;font-size:.68rem'>{desc[:28]}</td>"
                         f"<td style='padding:2px 4px'>{badge}</td>"
                         f"</tr>")
            st.markdown(f"""
<div style='background:{C_PANEL};border:1px solid {C_BORDER};border-radius:8px;
     padding:.5rem .7rem;margin-bottom:.4rem'>
<table style='width:100%;border-collapse:collapse;font-size:.72rem'>{rows}</table>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style='background:{C_PANEL};border:1px solid {C_BORDER};border-radius:8px;
     padding:.45rem .7rem;margin-bottom:.4rem;font-size:.72rem;color:{C_MUTED}'>
  No ports detected. Press <b style='color:{C_ACCENT}'>Scan Ports</b> or type manually below.
</div>""", unsafe_allow_html=True)

        # Manual override + dropdown
        manual = st.text_input("Type port manually",
                               placeholder="e.g. COM4  or  /dev/ttyUSB0",
                               label_visibility="visible")
        dropdown_options = sorted(set(all_found + ([manual] if manual.strip() else [])))
        if not dropdown_options:
            dropdown_options = ["COM3"]
        sel_port = st.selectbox("Select Port", dropdown_options,
                                label_visibility="visible")
        baud = st.selectbox("Baud Rate", [9600, 115200], index=0)

        # Tip box
        st.markdown(f"""
<div style='background:rgba(79,156,249,.06);border:1px solid {C_BORDER};
     border-radius:8px;padding:.45rem .7rem;font-size:.68rem;color:{C_MUTED};margin-top:.3rem'>
  💡 <b style='color:{C_TEXT}'>How to find your Arduino port:</b><br>
  <b>Windows:</b> Device Manager → Ports (COM &amp; LPT)<br>
  <b>Linux/Mac:</b> <code style='color:{C_ACCENT}'>ls /dev/tty*</code>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style='background:rgba(79,156,249,.07);border:1px solid {C_BORDER};
     border-radius:8px;padding:.55rem .75rem;font-size:.72rem;color:{C_MUTED};margin-top:.5rem'>
  📊 Replaying <b style='color:{C_ACCENT}'>real READINGS.xlsx</b> signal segments.<br>
  Cycles through 1–7 Hz to demonstrate live tremor detection.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Start / Stop ─────────────────────────────────────────────────────────
    btn_label = "⏹  Stop Detection" if S["running"] else "▶  Start Live Detection"
    if st.button(btn_label, type="secondary" if S["running"] else "primary",
                 use_container_width=True):
        if not S["running"]:
            # clear buffers
            S.update(raw_buf=[], time_buf=[], sample_count=0, drop_count=0,
                     dom_freq=0.0, amplitude=0.0, sig_quality=0.0, band_ratio=0.0,
                     tremor_flag=False, severity="—", confidence=0.0,
                     _demo_bpm_idx=0, _demo_sample_ptr=0, _demo_elapsed=0.0)
            while not S["data_queue"].empty():
                try: S["data_queue"].get_nowait()
                except: break

            if use_serial:
                if not SERIAL_OK:
                    st.error("pyserial not installed: pip install pyserial")
                else:
                    se = threading.Event()
                    S["_stop_event"] = se
                    th = threading.Thread(
                        target=_serial_reader,
                        args=(sel_port, baud, S["data_queue"], se),
                        daemon=True)
                    th.start()
                    S["mode"] = "serial"
            else:
                S["mode"] = "demo"

            S["running"] = True
        else:
            if "_stop_event" in S:
                S["_stop_event"].set()
            S["running"]     = False
            S["mode"]        = "idle"
            S["port_status"] = "Disconnected"

    st.markdown("---")

    # ── Parameters ───────────────────────────────────────────────────────────
    st.markdown(f"<div style='color:{C_MUTED};font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;margin-bottom:6px'>Parameters</div>", unsafe_allow_html=True)
    st.markdown(f"**Sampling Freq** &nbsp; `{FS} Hz`")
    tremor_thresh = st.slider("Amplitude Threshold (V)",
                              0.0, 0.5, 0.03, 0.005, format="%.3f")
    fft_win = st.select_slider("FFT Window (samples)", [128, 256, 512, 1024], value=512)

    st.markdown("---")
    st.markdown(f"""
<div style='font-size:.7rem;color:{C_MUTED};line-height:1.9em'>
<b style='color:{C_ACCENT}'>System Architecture</b><br>
Wearable TENG → Arduino UNO<br>
→ Serial / WiFi → Dashboard<br><br>
<b>Tremor Band:</b> 3 – 7 Hz<br>
<b>Filter:</b> 4th-order Butterworth<br>
<b>PSD:</b> Welch · <b>FFT:</b> scipy.fft<br>
<b>Smooth:</b> Savitzky-Golay (11, 3)<br>
<b>Data:</b> Real READINGS.xlsx
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align:center;padding:.3rem 0 .1rem'>
  <span style='font-size:1.9rem;font-weight:900;color:{C_ACCENT};letter-spacing:.04em'>
    ⚡ TENG Parkinson Tremor Detection System
  </span><br>
  <span style='font-size:.75rem;color:{C_MUTED};letter-spacing:.1em'>
    REAL-TIME SIGNAL PROCESSING &nbsp;·&nbsp; FFT SPECTRUM ANALYSIS &nbsp;·&nbsp; IEEE RESEARCH DASHBOARD
  </span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  DRAIN QUEUE → ROLLING BUFFER
# ════════════════════════════════════════════════════════════════════════════
if S["running"]:
    if S["mode"] == "demo":
        _feed_demo_chunk(S["data_queue"], n_samples=18)

    drained = 0
    while not S["data_queue"].empty() and drained < 600:
        t_v, v_v = S["data_queue"].get_nowait()
        S["raw_buf"].append(v_v)
        S["time_buf"].append(t_v)
        S["sample_count"] += 1
        drained += 1

    if len(S["raw_buf"]) > BUFFER_MAX:
        excess = len(S["raw_buf"]) - BUFFER_MAX
        S["raw_buf"]  = S["raw_buf"][excess:]
        S["time_buf"] = S["time_buf"][excess:]

# ════════════════════════════════════════════════════════════════════════════
#  STATUS BAR
# ════════════════════════════════════════════════════════════════════════════
live = S["running"]
mode_lbl = "SERIAL" if S["mode"] == "serial" else ("DEMO" if S["mode"] == "demo" else "IDLE")
port_pill = (f"<span class='pill pill-green'>● {S['port_status']}</span>"
             if "Connected" in S["port_status"]
             else f"<span class='pill pill-grey'>○ {S['port_status']}</span>")
acq_pill  = (f"<span class='pill pill-red'><span class='dot-live'></span>LIVE · {mode_lbl}</span>"
             if live
             else f"<span class='pill pill-grey'><span class='dot-idle'></span>IDLE</span>")
fs_pill   = f"<span class='pill pill-blue'>Fs = {FS} Hz</span>"
samp_pill = f"<span class='pill pill-blue'>{S['sample_count']:,} samples</span>"
drop_pill = (f"<span class='pill pill-yellow'>⚠ {S['drop_count']} dropped</span>"
             if S['drop_count'] else f"<span class='pill pill-green'>0 dropped</span>")

st.markdown(
    f"<div style='display:flex;gap:6px;flex-wrap:wrap;align-items:center;padding:4px 0'>"
    f"{acq_pill}{port_pill}{fs_pill}{samp_pill}{drop_pill}"
    f"</div>",
    unsafe_allow_html=True
)
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  PROCESS LIVE BUFFER
# ════════════════════════════════════════════════════════════════════════════
raw_arr  = np.array(S["raw_buf"],  dtype=np.float64)
time_arr = np.array(S["time_buf"], dtype=np.float64)
proc_arr = np.array([])
fft_f, fft_a = np.array([]), np.array([])
psd_f, psd_v = np.array([]), np.array([])

has_data = len(raw_arr) >= 24

if has_data:
    proc_arr = process_signal(raw_arr)
    seg = proc_arr[-fft_win:] if len(proc_arr) >= fft_win else proc_arr
    fft_f, fft_a = compute_fft(seg)
    psd_f, psd_v = compute_psd(seg)

    if len(fft_f) > 0:
        S["dom_freq"]   = float(fft_f[np.argmax(fft_a)])
    S["amplitude"]      = float(np.sqrt(np.mean(seg ** 2)))
    S["sig_quality"]    = sig_quality_score(proc_arr)

    # band ratio
    if len(psd_f) > 0:
        tot = np.trapezoid(psd_v, psd_f) + 1e-12
        bm  = (psd_f >= TREMOR_LO) & (psd_f <= TREMOR_HI)
        S["band_ratio"] = round(np.trapezoid(psd_v[bm], psd_f[bm]) / tot * 100.0
                                if bm.any() else 0.0, 1)

    trem, sev, conf = classify(S["dom_freq"], S["amplitude"], tremor_thresh)
    S["tremor_flag"] = trem
    S["severity"]    = sev
    S["confidence"]  = conf

# ════════════════════════════════════════════════════════════════════════════
#  ALERT + METRICS ROW
# ════════════════════════════════════════════════════════════════════════════
col_alert, col_metrics = st.columns([1, 2.2], gap="large")

with col_alert:
    if not has_data:
        st.markdown(f"""
<div class='alert-idle'>
  <div style='font-size:1.9rem'>⏳</div>
  <div style='font-size:1rem;font-weight:700;color:{C_MUTED};margin-top:4px'>
    Waiting for Signal
  </div>
  <div style='font-size:.73rem;color:{C_MUTED};margin-top:5px'>
    Connect Arduino or start Demo mode
  </div>
</div>""", unsafe_allow_html=True)
    elif S["tremor_flag"]:
        sev_color = {
            "Mild": C_YELLOW, "Moderate": "#fb923c", "Severe": C_RED
        }.get(S["severity"], C_RED)
        st.markdown(f"""
<div class='alert-tremor'>
  <div style='font-size:2.4rem'>🚨</div>
  <div style='font-size:1.55rem;font-weight:900;color:{C_RED};
              letter-spacing:.08em;margin-top:2px'>TREMOR DETECTED</div>
  <div style='font-size:1rem;font-weight:700;color:{sev_color};margin-top:8px'>
    {S['dom_freq']:.2f} Hz &nbsp;·&nbsp; {S['severity']}
  </div>
  <div style='font-size:.78rem;color:{C_TEXT};margin-top:5px'>
    Confidence: <b>{S['confidence']:.1f}%</b>
  </div>
  <div style='font-size:.68rem;color:{C_MUTED};margin-top:6px;letter-spacing:.04em'>
    PARKINSONIAN TREMOR BAND DETECTED
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class='alert-normal'>
  <div style='font-size:2.2rem'>✅</div>
  <div style='font-size:1.4rem;font-weight:800;color:{C_GREEN};margin-top:4px'>
    NORMAL
  </div>
  <div style='font-size:.9rem;color:{C_TEXT};margin-top:7px'>
    {S['dom_freq']:.2f} Hz — Outside tremor band
  </div>
  <div style='font-size:.7rem;color:{C_MUTED};margin-top:5px'>
    No Parkinsonian tremor signature detected
  </div>
</div>""", unsafe_allow_html=True)

with col_metrics:
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Dominant Freq",    f"{S['dom_freq']:.2f} Hz")
    r1c2.metric("Tremor Amplitude", f"{S['amplitude']:.4f} V")
    r1c3.metric("Signal Quality",   f"{S['sig_quality']:.1f}%")
    r1c4.metric("Band Power Ratio", f"{S['band_ratio']:.1f}%")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Classification",  S["severity"])
    r2c2.metric("Confidence",      f"{S['confidence']:.1f}%")
    r2c3.metric("Buffer Samples",  str(len(S["raw_buf"])))
    r2c4.metric("Mode",
                "🔌 Serial" if S["mode"] == "serial"
                else ("📂 Demo" if S["mode"] == "demo" else "⏸ Idle"))

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4 = st.tabs([
    "📈  Raw Signal",
    "🔧  Signal Processing",
    "📡  FFT Spectrum",
    "🌊  PSD Analysis",
])

# ── Tab 1 : Raw ──────────────────────────────────────────────────────────────
with t1:
    if has_data:
        st.plotly_chart(fig_raw(time_arr, raw_arr),
                        use_container_width=True, config={"displayModeBar": False})
        sc = st.columns(4)
        sc[0].metric("Mean",    f"{np.mean(raw_arr):.4f} V")
        sc[1].metric("Std Dev", f"{np.std(raw_arr):.4f} V")
        sc[2].metric("Min",     f"{np.min(raw_arr):.4f} V")
        sc[3].metric("Max",     f"{np.max(raw_arr):.4f} V")
    else:
        st.info("▶ Start detection to stream the live TENG signal.")

# ── Tab 2 : Signal Processing ────────────────────────────────────────────────
with t2:
    if has_data and len(proc_arr) > 0:
        t_trim = time_arr[-len(proc_arr):]
        r_trim = raw_arr[-len(proc_arr):]
        st.plotly_chart(fig_processed(t_trim, r_trim, proc_arr),
                        use_container_width=True, config={"displayModeBar": False})
        left, right = st.columns(2)
        with left:
            st.markdown(f"""
<div style='background:{C_PANEL};border:1px solid {C_BORDER};
     border-radius:10px;padding:.9rem 1.1rem'>
<b style='color:{C_ACCENT};font-size:.8rem;letter-spacing:.05em'>PROCESSING CHAIN</b>
<ol style='color:{C_TEXT};font-size:.78rem;margin-top:.6rem;line-height:2.1em;padding-left:1.2rem'>
  <li>Remove Invalid Samples &nbsp;<span style='color:{C_MUTED}'>(NaN · IQR 3×)</span></li>
  <li>DC Offset Removal &nbsp;<span style='color:{C_MUTED}'>(subtract mean)</span></li>
  <li>Detrending &nbsp;<span style='color:{C_MUTED}'>(scipy.signal.detrend)</span></li>
  <li>4th-order Butterworth Bandpass &nbsp;<span style='color:{C_MUTED}'>(0.5–10 Hz)</span></li>
  <li>Savitzky-Golay Smoothing &nbsp;<span style='color:{C_MUTED}'>(w=11, poly=3)</span></li>
</ol>
</div>""", unsafe_allow_html=True)
        with right:
            rms = float(np.sqrt(np.mean(proc_arr**2)))
            ptp = float(np.ptp(proc_arr))
            zc  = int(np.sum(np.diff(np.sign(proc_arr)) != 0) / (len(proc_arr) / FS))
            st.markdown(f"""
<div style='background:{C_PANEL};border:1px solid {C_BORDER};
     border-radius:10px;padding:.9rem 1.1rem'>
<b style='color:{C_ACCENT};font-size:.8rem;letter-spacing:.05em'>PROCESSED SIGNAL STATS</b>
<table style='color:{C_TEXT};font-size:.78rem;margin-top:.6rem;width:100%;line-height:2.1em'>
  <tr><td style='color:{C_MUTED}'>Samples in buffer</td>
      <td style='color:{C_ACCENT};text-align:right'>{len(proc_arr)}</td></tr>
  <tr><td style='color:{C_MUTED}'>RMS amplitude</td>
      <td style='color:{C_ACCENT};text-align:right'>{rms:.5f} V</td></tr>
  <tr><td style='color:{C_MUTED}'>Peak-to-peak</td>
      <td style='color:{C_ACCENT};text-align:right'>{ptp:.5f} V</td></tr>
  <tr><td style='color:{C_MUTED}'>Zero crossings / s</td>
      <td style='color:{C_ACCENT};text-align:right'>{zc}</td></tr>
  <tr><td style='color:{C_MUTED}'>Signal quality</td>
      <td style='color:{C_GREEN};text-align:right'>{S["sig_quality"]:.1f}%</td></tr>
</table>
</div>""", unsafe_allow_html=True)
    else:
        st.info("▶ Start detection to view the signal processing pipeline.")

# ── Tab 3 : FFT ───────────────────────────────────────────────────────────────
with t3:
    if has_data and len(fft_f) > 0:
        st.plotly_chart(fig_fft(fft_f, fft_a, S["dom_freq"], S["tremor_flag"]),
                        use_container_width=True, config={"displayModeBar": False})
        fc = st.columns(3)
        fc[0].metric("Dominant Frequency",  f"{S['dom_freq']:.3f} Hz")
        fc[1].metric("Peak FFT Amplitude",  f"{float(np.max(fft_a)):.4f}")
        in_band = TREMOR_LO <= S["dom_freq"] <= TREMOR_HI
        fc[2].metric("Band Status",
                     "⚡ IN TREMOR BAND" if in_band else "✅ Normal")
    else:
        st.info("▶ Start detection to view the live FFT spectrum.")

# ── Tab 4 : PSD ───────────────────────────────────────────────────────────────
with t4:
    if has_data and len(psd_f) > 0:
        st.plotly_chart(fig_psd(psd_f, psd_v),
                        use_container_width=True, config={"displayModeBar": False})
        pc = st.columns(3)
        tot = float(np.trapezoid(psd_v, psd_f))
        bm  = (psd_f >= TREMOR_LO) & (psd_f <= TREMOR_HI)
        bp  = float(np.trapezoid(psd_v[bm], psd_f[bm]) if bm.any() else 0.0)
        pc[0].metric("Total PSD Power",     f"{tot:.5f} V²")
        pc[1].metric("3–7 Hz Band Power",   f"{bp:.5f} V²")
        pc[2].metric("Band / Total Ratio",  f"{S['band_ratio']:.1f}%")
    else:
        st.info("▶ Start detection to view the PSD analysis.")

# ════════════════════════════════════════════════════════════════════════════
#  AUTO RERUN
# ════════════════════════════════════════════════════════════════════════════
if S["running"]:
    time.sleep(REFRESH_MS / 1000.0)
    st.rerun()
