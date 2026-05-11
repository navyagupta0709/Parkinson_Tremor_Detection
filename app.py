"""
=============================================================================
  IEEE TENG Tremor Detection Dashboard
  Parkinson's Disease — Real-Time FFT Signal Processing Interface
  Data Source : READINGS.xlsx  (or live Arduino via pyserial)
=============================================================================
  Run:  streamlit run teng_dashboard.py
=============================================================================
"""

# ─────────────────────────── Imports ────────────────────────────────────────
import time
import io
import threading
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.signal import welch, butter, sosfiltfilt, detrend, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# ─────────────────────────── Constants ──────────────────────────────────────
FS            = 100.0          # Sampling frequency (Hz)
WINDOW_SIZE   = 200            # Samples per analysis window
STEP_SIZE     = 100            # Overlap step
TREMOR_LO     = 3.0            # Tremor band lower bound (Hz)
TREMOR_HI     = 7.0            # Tremor band upper bound (Hz)
STREAM_WINDOW = 500            # Live plot rolling window (samples)
EXCEL_PATH    = "READINGS.xlsx"

# ─────────────────────────── Page Config ────────────────────────────────────
st.set_page_config(
    page_title="TENG Tremor Detection — IEEE Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Dark IEEE Theme ────────────────────────────────
st.markdown("""
<style>
/* ---------- base -------------------------------------------------------- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a;
    color: #e0e6f0;
    font-family: 'Courier New', Courier, monospace;
}
[data-testid="stSidebar"] {
    background-color: #0d1322;
    border-right: 1px solid #1e2d4a;
}
/* ---------- header banner ---------------------------------------------- */
.ieee-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #102040 50%, #0d1b2a 100%);
    border: 1px solid #1e4070;
    border-radius: 6px;
    padding: 18px 24px;
    margin-bottom: 16px;
    text-align: center;
}
.ieee-header h1 {
    color: #00c8ff;
    font-size: 1.6rem;
    letter-spacing: 2px;
    margin: 0;
    text-shadow: 0 0 12px #00c8ff55;
}
.ieee-header p {
    color: #7898b8;
    font-size: 0.75rem;
    margin: 4px 0 0;
    letter-spacing: 1px;
}
/* ---------- section cards ---------------------------------------------- */
.section-card {
    background: #0d1425;
    border: 1px solid #1a2e4a;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.section-title {
    color: #00c8ff;
    font-size: 0.72rem;
    font-weight: bold;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
    border-bottom: 1px solid #1a2e4a;
    padding-bottom: 6px;
}
/* ---------- metric tiles ------------------------------------------------ */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; }
.metric-tile {
    flex: 1 1 130px;
    background: #0f1e35;
    border: 1px solid #1a3a5c;
    border-radius: 6px;
    padding: 12px 14px;
    text-align: center;
    min-width: 120px;
}
.metric-tile .label {
    font-size: 0.65rem;
    color: #607898;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.metric-tile .value {
    font-size: 1.4rem;
    font-weight: bold;
    color: #00c8ff;
    margin-top: 4px;
    font-family: 'Courier New', monospace;
}
.metric-tile .unit {
    font-size: 0.65rem;
    color: #607898;
}
/* ---------- alert boxes ------------------------------------------------- */
.alert-tremor {
    background: #2a0a0a;
    border: 2px solid #ff2222;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    animation: pulse-red 1.2s ease-in-out infinite;
}
.alert-tremor h2 { color: #ff4444; font-size: 1.8rem; margin: 0; letter-spacing: 4px; }
.alert-tremor p  { color: #ff8888; font-size: 0.85rem; margin: 6px 0 0; }
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 8px #ff000055;  border-color: #ff2222; }
    50%       { box-shadow: 0 0 24px #ff0000bb; border-color: #ff5555; }
}
.alert-normal {
    background: #0a1f12;
    border: 2px solid #22cc55;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.alert-normal h2 { color: #33ee66; font-size: 1.8rem; margin: 0; letter-spacing: 3px; }
.alert-normal p  { color: #66cc88; font-size: 0.85rem; margin: 6px 0 0; }
/* ---------- severity badge --------------------------------------------- */
.sev-mild     { color: #ffdd55; font-weight: bold; }
.sev-moderate { color: #ff9933; font-weight: bold; }
.sev-severe   { color: #ff3333; font-weight: bold; }
/* ---------- model prediction card -------------------------------------- */
.pred-card {
    background: #0f1e35;
    border: 1px solid #1a3a5c;
    border-left: 4px solid #00c8ff;
    border-radius: 6px;
    padding: 14px;
}
.pred-card .class-name { font-size: 1.3rem; color: #00ffcc; font-weight: bold; }
.pred-card .conf       { font-size: 0.8rem;  color: #7898b8; margin-top: 4px; }
/* ---------- footer ----------------------------------------------------- */
.footer {
    text-align: center;
    color: #2a3a5a;
    font-size: 0.65rem;
    padding: 10px 0;
    letter-spacing: 1px;
    border-top: 1px solid #1a2e4a;
    margin-top: 20px;
}
/* ---------- plotly override -------------------------------------------- */
.js-plotly-plot .plotly .modebar { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def remove_outliers(sig: np.ndarray) -> np.ndarray:
    """IQR-based outlier removal (3× IQR fence)."""
    q1, q3 = np.percentile(sig, [25, 75])
    iqr     = q3 - q1
    mask    = (sig >= q1 - 3*iqr) & (sig <= q3 + 3*iqr)
    return sig[mask]


def process_signal(sig: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Full preprocessing chain:
      1. Remove DC offset
      2. Detrend
      3. 4th-order Butterworth bandpass  0.5–10 Hz
      4. Savitzky–Golay smoothing  (window=11, poly=3)
    """
    s = sig.copy().astype(np.float64)
    s -= np.mean(s)
    s  = detrend(s)
    sos = butter(4, [0.5, 10.0], btype='bandpass', fs=fs, output='sos')
    s   = sosfiltfilt(sos, s)
    if len(s) >= 11:
        s = savgol_filter(s, window_length=11, polyorder=3)
    return s


def compute_fft(sig: np.ndarray, fs: float = FS):
    """Return positive-frequency FFT amplitude spectrum."""
    n      = len(sig)
    freqs  = fftfreq(n, d=1.0/fs)
    amps   = np.abs(fft(sig)) / n
    mask   = freqs > 0
    return freqs[mask], amps[mask]


def compute_psd(sig: np.ndarray, fs: float = FS):
    """Welch PSD with nperseg=256."""
    nperseg = min(256, len(sig))
    f, psd  = welch(sig, fs=fs, nperseg=nperseg)
    return f, psd


def dominant_frequency(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Frequency of the highest FFT amplitude peak."""
    return float(freqs[np.argmax(amps)])


def tremor_power(f_psd: np.ndarray, psd: np.ndarray) -> float:
    """Integrated PSD power in the 3–7 Hz tremor band."""
    mask = (f_psd >= TREMOR_LO) & (f_psd <= TREMOR_HI)
    if mask.any():
        return float(np.trapezoid(psd[mask], f_psd[mask]))
    return 0.0


def signal_quality(sig: np.ndarray) -> str:
    """Heuristic SNR-based quality label."""
    rms = np.sqrt(np.mean(sig**2))
    if rms > 0.05:
        return "GOOD"
    elif rms > 0.01:
        return "FAIR"
    return "POOR"


def severity_label(dom_freq: float, power: float) -> str:
    """Classify tremor severity inside the 3–7 Hz band."""
    if not (TREMOR_LO <= dom_freq <= TREMOR_HI):
        return "None"
    if power > 0.5:
        return "Severe"
    elif power > 0.1:
        return "Moderate"
    return "Mild"


# ══════════════════════════════════════════════════════════════════════════════
#  ML FEATURE EXTRACTION & TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(window: np.ndarray, fs: float = FS) -> dict:
    n       = len(window)
    freqs_w, amps_w = compute_fft(window, fs)
    f_p, psd_w      = compute_psd(window, fs)
    dom_f   = dominant_frequency(freqs_w, amps_w)
    band_p  = tremor_power(f_p, psd_w)
    psd_norm = amps_w / (amps_w.sum() + 1e-12)
    sp_ent  = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return {
        'mean'       : np.mean(window),
        'std'        : np.std(window, ddof=1),
        'rms'        : np.sqrt(np.mean(window**2)),
        'energy'     : np.sum(window**2) / n,
        'dom_freq'   : dom_f,
        'sp_entropy' : sp_ent,
        'psd_peak'   : np.max(psd_w),
        'band_power' : band_p,
    }


@st.cache_resource(show_spinner=False)
def train_models(records: list):
    """Train RF / SVM / KNN on the READINGS dataset (cached)."""
    rows = []
    for r in records:
        sig = r['signal_proc']
        for i in range(0, len(sig) - WINDOW_SIZE + 1, STEP_SIZE):
            w    = sig[i : i + WINDOW_SIZE]
            feat = extract_features(w)
            feat['bpm'] = r['bpm']
            rows.append(feat)

    df      = pd.DataFrame(rows)
    FEAT_COLS = ['mean','std','rms','energy','dom_freq','sp_entropy','psd_peak','band_power']
    X, y    = df[FEAT_COLS].values, df['bpm'].values

    models = {
        'Random Forest': Pipeline([('sc', StandardScaler()),
                                   ('clf', RandomForestClassifier(n_estimators=100, random_state=42))]),
        'SVM'          : Pipeline([('sc', StandardScaler()),
                                   ('clf', SVC(probability=True, kernel='rbf', random_state=42))]),
        'KNN'          : Pipeline([('sc', StandardScaler()),
                                   ('clf', KNeighborsClassifier(n_neighbors=5))]),
    }
    for m in models.values():
        m.fit(X, y)
    return models, FEAT_COLS


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_excel_data(path: str):
    """Load all sheets from READINGS.xlsx and build signal records."""
    try:
        xl = pd.read_excel(path, sheet_name=None, header=None)
    except FileNotFoundError:
        return None, f"File '{path}' not found. Upload it beside this script."

    records = []
    sid     = 0
    for sheet_name, df_raw in xl.items():
        try:
            bpm      = int(sheet_name.split()[0])
            freq_hz  = bpm / 60.0
            df_data  = df_raw.iloc[2:].reset_index(drop=True)
            df_data.columns = ['t1','v1','t2','v2','t3','v3']
            for set_idx, vcol in enumerate(['v1','v2','v3'], start=1):
                raw = pd.to_numeric(df_data[vcol], errors='coerce').dropna().values.astype(np.float64)
                if len(raw) < WINDOW_SIZE:
                    continue
                cleaned = remove_outliers(raw)
                proc    = process_signal(cleaned)
                records.append({
                    'bpm'         : bpm,
                    'freq_hz'     : freq_hz,
                    'set_idx'     : set_idx,
                    'signal_id'   : sid,
                    'signal_raw'  : raw,
                    'signal_clean': cleaned,
                    'signal_proc' : proc,
                })
                sid += 1
        except Exception:
            continue

    return records, None


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS  (dark IEEE theme)
# ══════════════════════════════════════════════════════════════════════════════

DARK_LAYOUT = dict(
    paper_bgcolor='#0a0e1a',
    plot_bgcolor ='#0d1222',
    font         =dict(color='#a0b4cc', family='Courier New', size=11),
    margin       =dict(l=50, r=20, t=40, b=40),
    xaxis        =dict(gridcolor='#1a2840', zerolinecolor='#1a2840', linecolor='#1a2840'),
    yaxis        =dict(gridcolor='#1a2840', zerolinecolor='#1a2840', linecolor='#1a2840'),
)


def apply_dark(fig):
    fig.update_layout(**DARK_LAYOUT)
    return fig


def plot_raw_signal(sig: np.ndarray, fs: float = FS) -> go.Figure:
    t   = np.arange(len(sig)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=sig,
        mode='lines',
        line=dict(color='#00c8ff', width=1.2),
        name='TENG Voltage',
    ))
    fig.update_layout(
        title='Live Raw TENG Signal  ·  Time Domain',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (V)',
        **DARK_LAYOUT,
    )
    return fig


def plot_preprocessing_stages(sig_raw: np.ndarray, fs: float = FS) -> go.Figure:
    """Show 4 preprocessing stages as stacked subplots."""
    s1 = sig_raw - np.mean(sig_raw)
    s2 = detrend(s1)
    sos = butter(4, [0.5, 10.0], btype='bandpass', fs=fs, output='sos')
    s3  = sosfiltfilt(sos, s2)
    s4  = savgol_filter(s3, window_length=11, polyorder=3) if len(s3) >= 11 else s3

    labels  = ['DC Removal', 'Detrend', 'Butterworth BPF', 'Savitzky–Golay']
    signals = [s1, s2, s3, s4]
    colors  = ['#00c8ff', '#00ffaa', '#ff9933', '#ee44ff']

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=labels, vertical_spacing=0.07)

    for i, (s, c, lbl) in enumerate(zip(signals, colors, labels), start=1):
        t = np.arange(len(s)) / fs
        fig.add_trace(go.Scatter(x=t, y=s, mode='lines',
                                 line=dict(color=c, width=1.0), name=lbl), row=i, col=1)

    fig.update_layout(
        title='Signal Preprocessing Pipeline',
        height=520,
        showlegend=False,
        **DARK_LAYOUT,
    )
    fig.update_xaxes(title_text='Time (s)', row=4, col=1)
    return fig


def plot_fft(freqs: np.ndarray, amps: np.ndarray) -> go.Figure:
    dom_f = dominant_frequency(freqs, amps)
    fig   = go.Figure()

    # Tremor band shading
    fig.add_vrect(
        x0=TREMOR_LO, x1=TREMOR_HI,
        fillcolor='rgba(255,40,40,0.12)',
        line=dict(color='rgba(255,80,80,0.5)', width=1.5, dash='dot'),
        annotation_text='Tremor Band 3–7 Hz',
        annotation_position='top left',
        annotation=dict(font_color='#ff7070', font_size=10),
    )

    # Full spectrum
    fig.add_trace(go.Scatter(
        x=freqs, y=amps,
        mode='lines', fill='tozeroy',
        line=dict(color='#00c8ff', width=1.4),
        fillcolor='rgba(0,200,255,0.06)',
        name='FFT Amplitude',
    ))

    # Dominant frequency marker
    dom_amp = amps[np.argmin(np.abs(freqs - dom_f))]
    fig.add_trace(go.Scatter(
        x=[dom_f], y=[dom_amp],
        mode='markers+text',
        marker=dict(color='#ffdd22', size=10, symbol='diamond'),
        text=[f'{dom_f:.2f} Hz'],
        textposition='top center',
        textfont=dict(color='#ffdd22', size=10),
        name='Dominant Freq',
    ))

    # Harmonics (2×, 3× fundamental if within range)
    for k in [2, 3]:
        hf = dom_f * k
        if hf < 25:
            ha = amps[np.argmin(np.abs(freqs - hf))]
            fig.add_trace(go.Scatter(
                x=[hf], y=[ha],
                mode='markers',
                marker=dict(color='#ff9933', size=7, symbol='circle'),
                name=f'{k}× harmonic',
            ))

    fig.update_layout(
        title='FFT Amplitude Spectrum',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Amplitude',
        xaxis=dict(range=[0, 25], gridcolor='#1a2840', zerolinecolor='#1a2840', linecolor='#1a2840'),
        yaxis=dict(gridcolor='#1a2840', zerolinecolor='#1a2840', linecolor='#1a2840'),
        **DARK_LAYOUT,
    )
    return fig


def plot_psd(f_psd: np.ndarray, psd: np.ndarray) -> go.Figure:
    fig = go.Figure()

    fig.add_vrect(
        x0=TREMOR_LO, x1=TREMOR_HI,
        fillcolor='rgba(255,40,40,0.12)',
        line=dict(color='rgba(255,80,80,0.5)', width=1.5, dash='dot'),
    )

    fig.add_trace(go.Scatter(
        x=f_psd, y=psd,
        mode='lines', fill='tozeroy',
        line=dict(color='#ff6644', width=1.4),
        fillcolor='rgba(255,100,60,0.08)',
        name='PSD (Welch)',
    ))

    fig.update_layout(
        title='Welch Power Spectral Density',
        xaxis_title='Frequency (Hz)',
        yaxis_title='PSD  (V²/Hz)',
        xaxis=dict(range=[0, 20], gridcolor='#1a2840', zerolinecolor='#1a2840', linecolor='#1a2840'),
        yaxis=dict(type='log', gridcolor='#1a2840', zerolinecolor='#1a2840', linecolor='#1a2840'),
        **DARK_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  ARDUINO SERIAL THREAD  (optional — used when serial port is selected)
# ══════════════════════════════════════════════════════════════════════════════

class SerialReader(threading.Thread):
    def __init__(self, port: str, baud: int = 115200, maxlen: int = STREAM_WINDOW):
        super().__init__(daemon=True)
        self.port   = port
        self.baud   = baud
        self.buffer = deque(maxlen=maxlen)
        self.running = False
        self._error  = None

    def run(self):
        try:
            import serial
            self.running = True
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                while self.running:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    try:
                        self.buffer.append(float(line))
                    except ValueError:
                        pass
        except Exception as e:
            self._error = str(e)

    def stop(self):
        self.running = False

    def get_data(self) -> np.ndarray:
        return np.array(list(self.buffer))

    @property
    def error(self):
        return self._error


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

for k, v in {
    'serial_reader' : None,
    'log_rows'      : [],
    'session_start' : datetime.now(),
    'analysis_count': 0,
    'tremor_count'  : 0,
    'selected_rec'  : 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙ CONTROL PANEL")
    st.markdown("---")

    data_source = st.radio(
        "DATA SOURCE",
        ["📁 READINGS.xlsx", "🔌 Arduino Serial"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**SIGNAL SELECTOR** *(file mode)*")
    excel_path = st.text_input("Excel file path", value=EXCEL_PATH, label_visibility="collapsed")

    st.markdown("---")
    ml_model_name = st.selectbox(
        "ML CLASSIFIER",
        ["Random Forest", "SVM", "KNN"],
    )

    st.markdown("---")
    st.markdown("**SERIAL PORT** *(Arduino mode)*")
    serial_port = st.text_input("Port", value="/dev/ttyUSB0", label_visibility="collapsed")
    serial_baud = st.selectbox("Baud rate", [9600, 57600, 115200], index=2)

    if data_source == "🔌 Arduino Serial":
        col1, col2 = st.columns(2)
        if col1.button("▶ Connect", use_container_width=True):
            if st.session_state.serial_reader is None:
                reader = SerialReader(serial_port, serial_baud)
                reader.start()
                st.session_state.serial_reader = reader
                st.success("Serial connected.")
        if col2.button("⏹ Disconnect", use_container_width=True):
            if st.session_state.serial_reader:
                st.session_state.serial_reader.stop()
                st.session_state.serial_reader = None
                st.info("Disconnected.")

    st.markdown("---")
    auto_refresh = st.checkbox("⟳ Auto-refresh (2 s)", value=False)
    if auto_refresh:
        time.sleep(2)
        st.rerun()

    st.markdown("---")
    st.markdown("**SESSION LOG**")
    elapsed = (datetime.now() - st.session_state.session_start).seconds
    st.metric("Elapsed", f"{elapsed//60:02d}:{elapsed%60:02d}")
    st.metric("Analyses",  st.session_state.analysis_count)
    st.metric("Tremor Events", st.session_state.tremor_count)

    if st.session_state.log_rows:
        log_df  = pd.DataFrame(st.session_state.log_rows)
        csv_buf = io.BytesIO()
        log_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Download Log CSV",
            data=csv_buf.getvalue(),
            file_name=f"teng_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class='ieee-header'>
  <h1>🧠 TENG TREMOR DETECTION SYSTEM</h1>
  <p>REAL-TIME FFT SIGNAL PROCESSING  ·  IEEE RESEARCH PROTOTYPE  ·  PARKINSON'S DISEASE DETECTION</p>
</div>
""", unsafe_allow_html=True)

ts_col, _, _ = st.columns([2,5,2])
ts_col.markdown(
    f"<span style='color:#3a5a7a;font-size:0.72rem;'>⏱ {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}</span>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  DETERMINE ACTIVE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

active_signal = None
records       = None
error_msg     = None

if data_source == "📁 READINGS.xlsx":
    records, error_msg = load_excel_data(excel_path)
    if records:
        bpm_options = sorted(set(r['bpm'] for r in records))
        rec_labels  = [f"{r['bpm']} BPM · Set {r['set_idx']} · {r['bpm']/60:.1f} Hz"
                       for r in records]

        with st.sidebar:
            sel_idx = st.selectbox(
                "SELECT SIGNAL",
                range(len(records)),
                format_func=lambda i: rec_labels[i],
                index=st.session_state.selected_rec,
            )
            st.session_state.selected_rec = sel_idx

        chosen_rec   = records[sel_idx]
        active_signal = chosen_rec['signal_proc']
        raw_signal    = chosen_rec['signal_raw']

else:  # Arduino
    reader = st.session_state.serial_reader
    if reader and reader.error:
        st.error(f"Serial error: {reader.error}")
    elif reader:
        buf = reader.get_data()
        if len(buf) >= WINDOW_SIZE:
            raw_signal    = buf
            cleaned       = remove_outliers(buf)
            active_signal = process_signal(cleaned)
        else:
            st.info(f"Buffering… {len(buf)}/{WINDOW_SIZE} samples")
    else:
        st.warning("Click **▶ Connect** in the sidebar to start Arduino acquisition.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD — only when we have a signal
# ══════════════════════════════════════════════════════════════════════════════

if error_msg:
    st.error(f"⚠ {error_msg}")

elif active_signal is not None and len(active_signal) >= WINDOW_SIZE:

    # ── analysis window (last WINDOW_SIZE samples) ─────────────────────────
    win_proc  = active_signal[-WINDOW_SIZE:]
    win_raw   = raw_signal[-WINDOW_SIZE:]

    freqs_f, amps_f = compute_fft(win_proc)
    f_psd, psd_vals = compute_psd(win_proc)
    dom_f           = dominant_frequency(freqs_f, amps_f)
    t_power         = tremor_power(f_psd, psd_vals)
    sq              = signal_quality(win_proc)
    sev             = severity_label(dom_f, t_power)
    detected_bpm    = int(round(dom_f * 60))
    in_tremor_band  = TREMOR_LO <= dom_f <= TREMOR_HI

    # ── ML prediction ───────────────────────────────────────────────────────
    if records:
        models, feat_cols = train_models(records)
        clf    = models[ml_model_name]
        feat   = extract_features(win_proc)
        X_pred = np.array([[feat[c] for c in feat_cols]])
        pred_bpm   = int(clf.predict(X_pred)[0])
        pred_proba = clf.predict_proba(X_pred).max()
        pred_label = f"{pred_bpm} BPM ({pred_bpm/60:.1f} Hz)"
    else:
        pred_label, pred_proba = "N/A", 0.0

    # ── session logging ─────────────────────────────────────────────────────
    st.session_state.analysis_count += 1
    if in_tremor_band:
        st.session_state.tremor_count += 1

    st.session_state.log_rows.append({
        'timestamp'   : datetime.now().isoformat(timespec='seconds'),
        'dom_freq_hz' : round(dom_f, 4),
        'detected_bpm': detected_bpm,
        'tremor_power': round(t_power, 6),
        'severity'    : sev,
        'signal_quality': sq,
        'in_tremor_band': in_tremor_band,
        'ml_prediction': pred_label,
        'ml_confidence': round(pred_proba, 4),
    })

    # ════════════════════════════════════════════════════════════════════════
    #  ROW 1  –  Raw signal  |  Preprocessing
    # ════════════════════════════════════════════════════════════════════════
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("<div class='section-title'>① LIVE RAW TENG SIGNAL</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_raw_signal(raw_signal), use_container_width=True, config={'displayModeBar': False})

    with r1c2:
        st.markdown("<div class='section-title'>② SIGNAL PREPROCESSING PIPELINE</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_preprocessing_stages(win_raw), use_container_width=True, config={'displayModeBar': False})

    # ════════════════════════════════════════════════════════════════════════
    #  ROW 2  –  FFT  |  Welch PSD
    # ════════════════════════════════════════════════════════════════════════
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("<div class='section-title'>③ FFT AMPLITUDE SPECTRUM</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_fft(freqs_f, amps_f), use_container_width=True, config={'displayModeBar': False})

    with r2c2:
        st.markdown("<div class='section-title'>④ WELCH POWER SPECTRAL DENSITY</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_psd(f_psd, psd_vals), use_container_width=True, config={'displayModeBar': False})

    # ════════════════════════════════════════════════════════════════════════
    #  ROW 3  –  Detection metrics  |  Alert  |  ML
    # ════════════════════════════════════════════════════════════════════════
    r3c1, r3c2, r3c3 = st.columns([2, 1.6, 1.4])

    # ── (5) Detection panel ─────────────────────────────────────────────────
    with r3c1:
        st.markdown("<div class='section-title'>⑤ REAL-TIME DETECTION METRICS</div>", unsafe_allow_html=True)

        sev_css = {'Mild':'sev-mild','Moderate':'sev-moderate','Severe':'sev-severe'}.get(sev, 'sev-mild')
        sq_color = {'GOOD':'#33ee66','FAIR':'#ffcc33','POOR':'#ff4444'}.get(sq, '#ffffff')

        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-tile'>
            <div class='label'>Dominant Freq</div>
            <div class='value'>{dom_f:.2f}</div>
            <div class='unit'>Hz</div>
          </div>
          <div class='metric-tile'>
            <div class='label'>Detected BPM</div>
            <div class='value'>{detected_bpm}</div>
            <div class='unit'>bpm</div>
          </div>
          <div class='metric-tile'>
            <div class='label'>Tremor Power</div>
            <div class='value'>{t_power:.4f}</div>
            <div class='unit'>V²/Hz</div>
          </div>
          <div class='metric-tile'>
            <div class='label'>Tremor Severity</div>
            <div class='value {sev_css}' style='font-size:1.1rem;'>{sev}</div>
          </div>
          <div class='metric-tile'>
            <div class='label'>Signal Quality</div>
            <div class='value' style='color:{sq_color};font-size:1.1rem;'>{sq}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── (6) Alert box ────────────────────────────────────────────────────────
    with r3c2:
        st.markdown("<div class='section-title'>⑥ TREMOR ALERT SYSTEM</div>", unsafe_allow_html=True)
        if in_tremor_band:
            st.markdown(f"""
            <div class='alert-tremor'>
              <h2>⚠ TREMOR<br>DETECTED</h2>
              <p>{dom_f:.2f} Hz — {detected_bpm} BPM<br>
                 Severity: <span class='{sev_css}'>{sev}</span><br>
                 Band: {TREMOR_LO}–{TREMOR_HI} Hz (Parkinsonian)
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='alert-normal'>
              <h2>✔ NORMAL<br>SIGNAL</h2>
              <p>{dom_f:.2f} Hz — {detected_bpm} BPM<br>
                 Outside tremor band<br>
                 ({TREMOR_LO}–{TREMOR_HI} Hz)
              </p>
            </div>
            """, unsafe_allow_html=True)

    # ── (7) ML prediction ────────────────────────────────────────────────────
    with r3c3:
        st.markdown("<div class='section-title'>⑦ ML CLASSIFICATION</div>", unsafe_allow_html=True)
        bar_w = int(pred_proba * 100)
        bar_c = '#00ffcc' if pred_proba > 0.7 else '#ffcc33' if pred_proba > 0.4 else '#ff4444'
        st.markdown(f"""
        <div class='pred-card'>
          <div class='label'>MODEL</div>
          <div style='color:#7898b8;font-size:0.8rem;margin-bottom:8px;'>{ml_model_name}</div>
          <div class='label'>PREDICTION</div>
          <div class='class-name'>{pred_label}</div>
          <div class='conf'>Confidence: {pred_proba*100:.1f}%</div>
          <div style='background:#0a1422;border-radius:4px;height:8px;margin-top:10px;'>
            <div style='width:{bar_w}%;height:100%;background:{bar_c};border-radius:4px;
                        transition:width 0.4s ease;'></div>
          </div>
          <div class='conf' style='margin-top:4px;'>
            Freq detect: <span style='color:#00c8ff;'>{dom_f:.3f} Hz</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    #  ROW 4  –  Session statistics bar
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⑧ SESSION STATISTICS</div>", unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Analyses",    st.session_state.analysis_count)
    s2.metric("Tremor Events",     st.session_state.tremor_count)
    s3.metric("Dominant Freq",     f"{dom_f:.3f} Hz")
    s4.metric("Tremor Band Power", f"{t_power:.4f} V²/Hz")
    s5.metric("Model Confidence",  f"{pred_proba*100:.1f}%")

    # ────────────────────────────────────────────────────────────────────────
    #  Signal metadata (expandable)
    # ────────────────────────────────────────────────────────────────────────
    if records and data_source == "📁 READINGS.xlsx":
        with st.expander("📋 Signal Record Details", expanded=False):
            r = records[st.session_state.selected_rec]
            st.json({
                "BPM"          : r['bpm'],
                "Frequency (Hz)": round(r['freq_hz'], 4),
                "Set Index"    : r['set_idx'],
                "Signal ID"    : r['signal_id'],
                "Raw Samples"  : len(r['signal_raw']),
                "Proc Samples" : len(r['signal_proc']),
                "RMS (proc)"   : round(float(np.sqrt(np.mean(r['signal_proc']**2))), 6),
            })

else:
    st.info("⏳ Waiting for signal data — check that READINGS.xlsx is in the working directory.")


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
  IEEE RESEARCH PROTOTYPE  ·  TENG-BASED PARKINSON TREMOR DETECTION  ·  FFT / WELCH PSD ANALYSIS
  <br>Sensor: Triboelectric Nanogenerator (TENG)  ·  Fs = 100 Hz  ·  Tremor Band: 3–7 Hz  ·  Butterworth BPF 0.5–10 Hz
</div>
""", unsafe_allow_html=True)
