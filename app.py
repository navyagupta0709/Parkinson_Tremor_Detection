"""
app.py  —  Parkinson's Tremor Real-Time Detection
Arduino UNO  →  Serial  →  FFT  →  ML  →  Live Dashboard

Cloud-safe: all optional imports guarded, simulation mode always works.
Arduino serial only activates when pyserial is available (local use).

Serial format from tremor_sensor.ino:
  "TIME_SECONDS VOLTAGE\n"
  e.g.  "1.230 2.4812"
"""

# ── Standard library ───────────────────────────────────────────
import os
import time
import threading
import datetime
from collections import deque, Counter

# ── Core scientific (always present on Streamlit Cloud) ────────
import numpy as np
import pandas as pd
import streamlit as st
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfiltfilt, detrend, savgol_filter, welch

# ── Plotly (listed in requirements.txt — should always load) ───
import plotly.graph_objects as go

# ── Optional: pyserial  (not available on Streamlit Cloud) ─────
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── Optional: ML (should be available, but guard anyway) ───────
try:
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    HAS_ML = True
except ImportError:
    HAS_ML = False

# ═══════════════════════════════════════════════════════════════
# CONSTANTS  (from notebook: FS=100 Hz, WINDOW=200 samples)
# ═══════════════════════════════════════════════════════════════
FS   = 100      # Arduino sends at 10 ms intervals → 100 Hz
WIN  = 200      # 2-second analysis window
BAUD = 9600

PATHOLOGICAL_LO = 3.0   # Parkinson's tremor: 3–7 Hz resting
PATHOLOGICAL_HI = 7.0

THR_MILD   = 0.30        # Voltage amplitude thresholds (V)
THR_MOD    = 0.80
THR_SEVERE = 1.50

LOG_FILE   = "tremor_log.csv"
MODEL_FILE = "model.pkl"

# ═══════════════════════════════════════════════════════════════
# SIGNAL PROCESSING  (matches notebook Cell 7 exactly)
# ═══════════════════════════════════════════════════════════════
def process_signal(sig: np.ndarray) -> np.ndarray:
    """DC removal → detrend → bandpass 0.5–10 Hz → Savitzky-Golay."""
    s = sig.astype(float).copy()
    if len(s) < 20:
        return s
    s -= np.mean(s)
    s  = detrend(s)
    try:
        sos = butter(4, [0.5, 10.0], btype="bandpass", fs=FS, output="sos")
        s   = sosfiltfilt(sos, s)
        s   = savgol_filter(s, window_length=11, polyorder=3)
    except Exception:
        pass
    return s


def dominant_freq(sig: np.ndarray) -> float:
    n = len(sig)
    mags  = np.abs(fft(sig))
    freqs = fftfreq(n, 1.0 / FS)
    pos   = freqs > 0
    fp, fv = freqs[pos], mags[pos]
    return float(fp[np.argmax(fv)]) if len(fp) else 0.0


def fft_spectrum(sig: np.ndarray):
    n     = len(sig)
    mags  = np.abs(fft(sig))
    freqs = fftfreq(n, 1.0 / FS)
    pos   = freqs > 0
    return freqs[pos], mags[pos]


def extract_features(sig: np.ndarray) -> np.ndarray:
    """10 features from notebook Cell 9."""
    n    = len(sig)
    mean = float(np.mean(sig))
    std  = float(np.std(sig, ddof=1))
    rms  = float(np.sqrt(np.mean(sig ** 2)))
    energy = float(np.sum(sig ** 2) / n)
    dom  = dominant_freq(sig)

    mags  = np.abs(fft(sig))
    pn    = mags / (mags.sum() + 1e-12)
    ent   = float(-np.sum(pn * np.log2(pn + 1e-12)))

    try:
        fw, psd = welch(sig, fs=FS, nperseg=min(n, 64))
        band    = (fw >= 3.0) & (fw <= 7.0)
        # numpy 2.x uses trapezoid; fall back to trapz for older versions
        _trapz  = getattr(np, "trapezoid", np.trapz)
        bpwr    = float(_trapz(psd[band], fw[band])) if band.any() else 0.0
        tpwr    = float(_trapz(psd, fw)) + 1e-12
        bratio  = bpwr / tpwr
    except Exception:
        bpwr, bratio = 0.0, 0.0

    p2p = float(np.ptp(sig))
    zc  = int(np.sum(np.diff(np.sign(sig)) != 0))

    return np.array([mean, std, rms, energy, dom, ent, bpwr, bratio, p2p, zc])


# ═══════════════════════════════════════════════════════════════
# ML MODEL  —  train synthetic if model.pkl missing
# ═══════════════════════════════════════════════════════════════
LABEL = {0: "Normal", 1: "Mild Tremor", 2: "Moderate Tremor", 3: "Severe Tremor"}
LABEL_COLOR = {
    "Normal":           "#22c55e",
    "Mild Tremor":      "#eab308",
    "Moderate Tremor":  "#f97316",
    "Severe Tremor":    "#ef4444",
    "—":                "#4a6fa5",
}


def _train():
    if not HAS_ML:
        return None
    t   = np.linspace(0, WIN / FS, WIN)
    rows, labels = [], []
    for lbl, freq, amp, n in [(0, 0.8, 0.08, 200), (1, 3.2, 0.45, 200),
                               (2, 5.0, 0.95, 200), (3, 6.8, 1.80, 200)]:
        for _ in range(n):
            sig = amp * np.sin(2 * np.pi * freq * t) + np.random.normal(0, amp * 0.15, WIN)
            rows.append(extract_features(process_signal(sig)))
            labels.append(lbl)
    X, y = np.array(rows), np.array(labels)
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("rf",  RandomForestClassifier(150, random_state=42, class_weight="balanced")),
    ])
    pipe.fit(X, y)
    try:
        joblib.dump(pipe, MODEL_FILE)
    except Exception:
        pass
    return pipe


@st.cache_resource(show_spinner="🧠 Training AI model…")
def get_model():
    if not HAS_ML:
        return None
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass
    return _train()


def ml_predict(model, window: np.ndarray) -> tuple:
    """Returns (label_str, confidence_float 0–100)."""
    proc = process_signal(window)
    if model is not None and HAS_ML:
        try:
            feat  = extract_features(proc).reshape(1, -1)
            pred  = model.predict(feat)[0]
            proba = model.predict_proba(feat)[0]
            return LABEL[int(pred)], float(np.max(proba)) * 100
        except Exception:
            pass
    # Rule-based fallback
    freq = dominant_freq(proc)
    amp  = float(np.ptp(proc))
    if PATHOLOGICAL_LO <= freq <= PATHOLOGICAL_HI and amp >= THR_SEVERE:
        return "Severe Tremor",   88.0
    if PATHOLOGICAL_LO <= freq <= PATHOLOGICAL_HI and amp >= THR_MOD:
        return "Moderate Tremor", 80.0
    if freq >= PATHOLOGICAL_LO and amp >= THR_MILD:
        return "Mild Tremor",     72.0
    return "Normal", 91.0


# ═══════════════════════════════════════════════════════════════
# ARDUINO SERIAL READER  (only used when HAS_SERIAL=True locally)
# ═══════════════════════════════════════════════════════════════
class ArduinoReader:
    def __init__(self, port, baud=BAUD, maxlen=600):
        self.port      = port
        self.baud      = baud
        self._buf      = deque(maxlen=maxlen)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None
        self.connected = False
        self.error     = ""

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running  = False
        self.connected = False

    def _loop(self):
        while self._running:
            try:
                ser = serial.Serial(self.port, self.baud, timeout=1)
                self.connected = True
                self.error     = ""
                while self._running:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw:
                        continue
                    parts = raw.split()
                    if len(parts) >= 2:
                        try:
                            ts  = float(parts[0])
                            val = float(parts[1])
                            with self._lock:
                                self._buf.append((ts, val))
                        except ValueError:
                            pass
                ser.close()
            except Exception as e:
                self.connected = False
                self.error     = str(e)
                time.sleep(2)

    def latest(self, n=WIN):
        with self._lock:
            data = list(self._buf)[-n:]
        if not data:
            return [], []
        return [d[0] for d in data], [d[1] for d in data]


def auto_port():
    if not HAS_SERIAL:
        return None
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "") + (p.manufacturer or "")
        if any(k in desc for k in ["Arduino", "CH340", "USB-SERIAL", "ttyUSB", "ttyACM"]):
            return p.device
    ports = serial.tools.list_ports.comports()
    return ports[0].device if ports else None


# ═══════════════════════════════════════════════════════════════
# SIMULATION READER  (always works — cloud + local)
# ═══════════════════════════════════════════════════════════════
class SimReader:
    _CFG = {
        "None (Healthy)":   (0.8,  0.08),
        "Mild Tremor":      (3.2,  0.45),
        "Moderate Tremor":  (5.0,  0.95),
        "Severe Tremor":    (6.8,  1.80),
    }

    def __init__(self, severity="Mild Tremor", maxlen=600):
        self._buf      = deque(maxlen=maxlen)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None
        self.connected = True
        self.error     = ""
        self._sev      = severity
        self._t        = 0.0

    def set_severity(self, s):
        self._sev = s

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            freq, amp = self._CFG.get(self._sev, (3.2, 0.45))
            val = amp * np.sin(2 * np.pi * freq * self._t)
            val += np.random.normal(0, amp * 0.12)
            val  = float(np.clip(val + 2.5, 0.0, 5.0))
            with self._lock:
                self._buf.append((round(self._t, 3), round(val, 4)))
            self._t += 0.01
            time.sleep(0.01)

    def latest(self, n=WIN):
        with self._lock:
            data = list(self._buf)[-n:]
        if not data:
            return [], []
        return [d[0] for d in data], [d[1] for d in data]


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════
def log_row(ts, raw, freq, label, conf):
    try:
        row = pd.DataFrame([{
            "time":       ts,
            "voltage":    round(raw,  4),
            "freq_hz":    round(freq, 2),
            "prediction": label,
            "confidence": round(conf, 1),
        }])
        row.to_csv(LOG_FILE, mode="a",
                   header=not os.path.exists(LOG_FILE), index=False)
    except Exception:
        pass   # cloud may have read-only filesystem — silently skip


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Parkinson's Tremor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, [class*="css"]          { font-family:'Inter',sans-serif!important; }
.stApp, .main              { background:#060d1a!important; color:#e2e8f0!important; }
.block-container           { padding:1.2rem 1.8rem!important; max-width:100%!important; }
section[data-testid="stSidebar"] { background:#0a1628!important; border-right:1px solid #1a2d4a!important; }
[data-testid="stSidebar"] * { color:#e2e8f0!important; }

.hdr {
  background:linear-gradient(135deg,#0d1f38 0%,#0a1628 60%,#071220 100%);
  border:1px solid #1a3050; border-radius:16px;
  padding:20px 28px; margin-bottom:18px;
  display:flex; align-items:center; justify-content:space-between;
}
.hdr-title { font-size:1.55rem; font-weight:800; color:#fff; letter-spacing:-.03em; }
.hdr-title em { color:#00e5b4; font-style:normal; }
.hdr-sub   { font-size:.76rem; color:#4a6fa5; margin-top:3px; }

.status-card {
  border-radius:16px; padding:28px 24px; text-align:center;
  transition:all .4s; position:relative; overflow:hidden;
}
.s-green  { background:linear-gradient(145deg,#042f1e,#063d26); border:1.5px solid #16a34a; box-shadow:0 0 40px rgba(34,197,94,.18); }
.s-yellow { background:linear-gradient(145deg,#2d1f04,#3d2a06); border:1.5px solid #ca8a04; box-shadow:0 0 40px rgba(234,179,8,.18); }
.s-orange { background:linear-gradient(145deg,#2d1104,#3d1806); border:1.5px solid #ea580c; box-shadow:0 0 40px rgba(249,115,22,.18); }
.s-red    { background:linear-gradient(145deg,#2d0404,#3d0808); border:1.5px solid #dc2626; box-shadow:0 0 50px rgba(239,68,68,.30); }
.s-idle   { background:linear-gradient(145deg,#0d1a2e,#0a1422); border:1.5px solid #1a3050; }

.status-icon  { font-size:3.2rem; line-height:1; margin-bottom:10px; }
.status-label { font-size:1.7rem; font-weight:800; letter-spacing:-.02em; }
.status-freq  { font-size:.95rem; margin-top:8px; opacity:.75; }
.status-conf  { font-size:.78rem; margin-top:4px; opacity:.5; }
.status-msg   { margin-top:14px; font-size:.78rem; color:#6b8ab0; line-height:1.5; }

.tile {
  background:#0d1a2e; border:1px solid #1a3050;
  border-radius:12px; padding:16px 18px;
}
.tile-lbl  { font-size:.67rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#4a6fa5; margin-bottom:6px; }
.tile-val  { font-size:1.85rem; font-weight:700; color:#fff; line-height:1; }
.tile-unit { font-size:.85rem; color:#4a6fa5; font-weight:400; }
.tile-sub  { font-size:.7rem; color:#2d4a6a; margin-top:4px; }

.badge { display:inline-flex; align-items:center; gap:6px; font-size:.73rem; font-weight:600; padding:5px 12px; border-radius:20px; }
.b-live  { background:rgba(0,229,180,.12); color:#00e5b4; border:1px solid rgba(0,229,180,.3); }
.b-sim   { background:rgba(59,130,246,.12); color:#60a5fa; border:1px solid rgba(59,130,246,.3); }
.b-off   { background:rgba(100,116,139,.1); color:#64748b; border:1px solid rgba(100,116,139,.2); }
.b-err   { background:rgba(239,68,68,.12);  color:#f87171; border:1px solid rgba(239,68,68,.3); }
.dot     { width:7px; height:7px; border-radius:50%; background:currentColor; display:inline-block; }
.pulse   { animation:pulse 1.3s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

.fbar-wrap { background:#0d1a2e; border-radius:8px; height:10px; width:100%; overflow:hidden; margin:8px 0; }
.fbar-fill { height:100%; border-radius:8px; transition:width .5s; }

.stButton>button {
  background:#00e5b4!important; color:#000!important; border:none!important;
  font-weight:700!important; border-radius:10px!important;
  font-size:.85rem!important; width:100%!important; transition:all .2s!important;
}
.stButton>button:hover { opacity:.85!important; transform:translateY(-1px)!important; }
.stSelectbox label, .stSlider label, .stToggle label, .stTextInput label {
  color:#4a6fa5!important; font-size:.73rem!important;
  font-weight:600!important; text-transform:uppercase!important; letter-spacing:.06em!important;
}
#MainMenu, footer, header { visibility:hidden!important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
def _init():
    defaults = dict(
        reader=None, monitoring=False, sim_mode=False,
        vals=[], freqs=[], labels=[], confs=[],
        cur_label="—", cur_freq=0.0, cur_conf=0.0, cur_raw=0.0,
        tick=0, alert_count=0, t0=datetime.datetime.now(),
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

MAX_HIST = 500
model    = get_model()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Controls")
    st.markdown("---")

    # Force simulation on cloud (no USB)
    cloud_mode = not HAS_SERIAL
    if cloud_mode:
        st.info("🌐 Running on Streamlit Cloud — Simulation mode only (no USB serial on cloud).")

    sim_toggle = True if cloud_mode else st.toggle(
        "📵 Simulation (no Arduino)", value=st.session_state.sim_mode)
    st.session_state.sim_mode = sim_toggle

    sim_sev = st.selectbox(
        "Patient / Severity",
        ["None (Healthy)", "Mild Tremor", "Moderate Tremor", "Severe Tremor"],
        index=1,
    )
    if st.session_state.reader and isinstance(st.session_state.reader, SimReader):
        st.session_state.reader.set_severity(sim_sev)

    if not cloud_mode and not sim_toggle:
        detected = auto_port() or ""
        port = st.text_input("Arduino COM Port", value=detected,
                             placeholder="COM3 or /dev/ttyUSB0")
        st.caption("Auto-detected: " + (detected or "none found"))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶ START")
    with c2:
        stop_btn  = st.button("⏹ STOP")

    if start_btn:
        if st.session_state.reader:
            st.session_state.reader.stop()

        use_sim = cloud_mode or sim_toggle
        if use_sim:
            r = SimReader(sim_sev)
        else:
            r = ArduinoReader(port)

        r.start()
        st.session_state.reader    = r
        st.session_state.monitoring = True
        st.session_state.t0        = datetime.datetime.now()
        st.success("✅ Monitoring started!")

    if stop_btn:
        if st.session_state.reader:
            st.session_state.reader.stop()
        st.session_state.monitoring = False
        st.session_state.reader     = None
        st.info("⏹ Stopped.")

    st.markdown("---")
    st.markdown("### ⚙️ Display")
    refresh_ms = st.slider("Refresh rate (ms)", 300, 2000, 600, 100)
    log_on     = st.checkbox("Save to CSV log", True)

    st.markdown("---")
    if st.button("🗑️ Clear history"):
        for k in ["vals", "freqs", "labels", "confs"]:
            st.session_state[k] = []
        st.session_state.alert_count = 0
        if os.path.exists(LOG_FILE):
            try:
                os.remove(LOG_FILE)
            except Exception:
                pass
        st.success("Cleared.")

    if os.path.exists(LOG_FILE):
        try:
            df_log = pd.read_csv(LOG_FILE)
            st.download_button(
                "⬇️ Download CSV", df_log.to_csv(index=False).encode(),
                f"tremor_{datetime.date.today()}.csv", "text/csv",
                use_container_width=True,
            )
        except Exception:
            pass

    st.markdown("---")
    st.caption(
        "Arduino → Serial → FFT → Random Forest\n"
        "Parkinson's Tremor · 100 Hz · 200-sample window\n"
        "Pathological band: 3–7 Hz"
    )


# ═══════════════════════════════════════════════════════════════
# INGEST  — pull one tick, run ML
# ═══════════════════════════════════════════════════════════════
r   = st.session_state.reader
mon = st.session_state.monitoring

if r and mon:
    _, val_list = r.latest(WIN)
    if len(val_list) >= 20:
        arr    = np.array(val_list, dtype=float)
        raw    = float(arr[-1])
        window = arr if len(arr) >= WIN else np.pad(arr, (WIN - len(arr), 0))

        label, conf = ml_predict(model, window)
        proc  = process_signal(window)
        freq  = dominant_freq(proc)

        st.session_state.cur_label = label
        st.session_state.cur_freq  = freq
        st.session_state.cur_conf  = conf
        st.session_state.cur_raw   = raw
        st.session_state.tick     += 1

        st.session_state.vals.append(raw)
        st.session_state.freqs.append(freq)
        st.session_state.labels.append(label)
        st.session_state.confs.append(conf)

        if label in ("Moderate Tremor", "Severe Tremor"):
            st.session_state.alert_count += 1

        for k in ["vals", "freqs", "labels", "confs"]:
            if len(st.session_state[k]) > MAX_HIST:
                st.session_state[k] = st.session_state[k][-MAX_HIST:]

        if log_on:
            now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_row(now_str, raw, freq, label, conf)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
is_live = r is not None and getattr(r, "connected", False) and mon
err_msg = getattr(r, "error", "") if r else ""
dur     = (datetime.datetime.now() - st.session_state.t0).seconds
mm, ss  = divmod(dur, 60)

if is_live and st.session_state.sim_mode:
    badge = '<span class="badge b-sim"><span class="dot pulse"></span>SIMULATION</span>'
elif is_live:
    badge = '<span class="badge b-live"><span class="dot pulse"></span>ARDUINO LIVE</span>'
elif err_msg:
    badge = f'<span class="badge b-err">⚠ {err_msg[:35]}</span>'
else:
    badge = '<span class="badge b-off">⏸ IDLE</span>'

st.markdown(f"""
<div class="hdr">
  <div>
    <div class="hdr-title">🧠 Parkinson's <em>Tremor Detector</em></div>
    <div class="hdr-sub">Real-time · Arduino UNO → 100 Hz → FFT → Random Forest AI</div>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    {badge}
    <span class="badge b-off">🕒 {mm:02d}:{ss:02d}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# BIG STATUS CARD
# ═══════════════════════════════════════════════════════════════
label = st.session_state.cur_label
freq  = st.session_state.cur_freq
conf  = st.session_state.cur_conf
raw   = st.session_state.cur_raw

STATUS_CFG = {
    "Normal":           ("s-green",  "✅", "#22c55e", "No tremor detected — Hand movement is normal"),
    "Mild Tremor":      ("s-yellow", "⚠️",  "#eab308", "Mild tremor detected (3–4 Hz) — Please monitor"),
    "Moderate Tremor":  ("s-orange", "🔶", "#f97316", "Moderate tremor (4–6 Hz) — Consult neurologist"),
    "Severe Tremor":    ("s-red",    "🚨", "#ef4444", "SEVERE TREMOR DETECTED — Immediate attention!"),
    "—":                ("s-idle",   "⏸",  "#4a6fa5", "Press START then shake hand to detect tremor"),
}
css, icon, color, msg = STATUS_CFG.get(label, STATUS_CFG["—"])
in_path  = PATHOLOGICAL_LO <= freq <= PATHOLOGICAL_HI and freq > 0
conf_str = f"AI confidence: {conf:.0f}%" if conf > 0 else ""
freq_note = f"⚡ {freq:.2f} Hz — Parkinson band: 3–7 Hz" if freq > 0 else "Awaiting signal…"

left, right = st.columns([1.05, 1.95])

with left:
    st.markdown(f"""
    <div class="status-card {css}">
      <div class="status-icon">{icon}</div>
      <div class="status-label" style="color:{color}">{label}</div>
      <div class="status-freq">{freq_note}</div>
      <div class="status-conf">{conf_str}</div>
      <div class="status-msg">{msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Frequency band bar
    band_pct = min(100.0, (freq / 12.0) * 100.0) if freq > 0 else 0.0
    band_col = (
        "#22c55e" if freq < PATHOLOGICAL_LO else
        "#eab308" if freq < 4.0 else
        "#f97316" if freq < 6.0 else
        "#ef4444"
    )
    path_note = "⚠ Pathological" if in_path and freq > 0 else ("✓ Normal range" if freq > 0 else "")
    st.markdown(f"""
    <div class="tile" style="margin-top:12px">
      <div class="tile-lbl">Frequency Position</div>
      <div style="font-size:.72rem;color:#4a6fa5;margin-bottom:4px;
                  display:flex;justify-content:space-between">
        <span>0 Hz</span>
        <span style="color:#ef4444">3–7 Hz (Parkinson)</span>
        <span>12 Hz</span>
      </div>
      <div class="fbar-wrap">
        <div class="fbar-fill" style="width:{band_pct:.1f}%;background:{band_col}"></div>
      </div>
      <div style="font-size:1.05rem;font-weight:700;color:{band_col};margin-top:6px">
        {freq:.2f} Hz &nbsp; {path_note}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT: Live charts ──────────────────────────────────────────
with right:
    hv = st.session_state.vals
    hf = st.session_state.freqs
    hl = st.session_state.labels

    if len(hv) < 5:
        st.info("⏸  Press **▶ START** then shake your hand to see the live signal.")
    else:
        n_show = min(len(hv), 200)
        yv = hv[-n_show:]
        yf = hf[-n_show:]
        x  = list(range(n_show))

        clr_map = {
            "Normal":          "#22c55e",
            "Mild Tremor":     "#eab308",
            "Moderate Tremor": "#f97316",
            "Severe Tremor":   "#ef4444",
            "—":               "#4a6fa5",
        }
        pt_colors = [clr_map.get(l, "#4a6fa5") for l in hl[-n_show:]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=yv, mode="lines", name="Voltage (V)",
            line=dict(color="rgba(0,229,180,0.55)", width=1.2), yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=yf, mode="lines", name="Tremor Freq (Hz)",
            line=dict(color="#60a5fa", width=2.2), yaxis="y2",
        ))
        # Parkinson zone shading (freq axis)
        fig.add_hrect(
            yref="y2", y0=PATHOLOGICAL_LO, y1=PATHOLOGICAL_HI,
            fillcolor="rgba(239,68,68,0.07)", layer="below", line_width=0,
        )
        # Amplitude threshold lines
        for thr, col, lbl in [
            (THR_MILD,   "#eab308", "Mild"),
            (THR_MOD,    "#f97316", "Moderate"),
            (THR_SEVERE, "#ef4444", "Severe"),
        ]:
            fig.add_hline(
                y=thr, yref="y1", line_dash="dot",
                line_color=col, line_width=1,
                annotation_text=lbl,
                annotation_font_size=9, annotation_font_color=col,
                annotation_position="top right",
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a1628",
            font=dict(color="#4a6fa5", size=10),
            margin=dict(l=8, r=8, t=36, b=8),
            height=310,
            legend=dict(orientation="h", y=1.14, x=0,
                        bgcolor="rgba(0,0,0,0)", font_size=10),
            title=dict(text="Live Signal + Tremor Frequency",
                       font_color="#4a6fa5", font_size=11),
            xaxis=dict(showgrid=True, gridcolor="#0d1f38", title="Samples"),
            yaxis=dict(showgrid=True, gridcolor="#0d1f38",
                       title="V", side="left", range=[0, 5]),
            yaxis2=dict(showgrid=False, title="Hz",
                        overlaying="y", side="right", range=[0, 12]),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════
# METRIC TILES ROW
# ═══════════════════════════════════════════════════════════════
st.markdown("")
m1, m2, m3, m4, m5 = st.columns(5)

def tile_html(lbl, val, unit="", sub="", color="#fff"):
    return (f'<div class="tile">'
            f'<div class="tile-lbl">{lbl}</div>'
            f'<div class="tile-val" style="color:{color}">'
            f'{val}<span class="tile-unit"> {unit}</span></div>'
            f'<div class="tile-sub">{sub}</div></div>')

freq_c = "#ef4444" if in_path and freq > 0 else ("#22c55e" if freq > 0 else "#4a6fa5")
lc     = LABEL_COLOR.get(label, "#4a6fa5")
ac     = st.session_state.alert_count

with m1:
    st.markdown(tile_html("Live Voltage", f"{raw:.3f}", "V", "From A0 pin"),
                unsafe_allow_html=True)
with m2:
    st.markdown(tile_html("Dominant Freq", f"{freq:.2f}", "Hz",
                          "⚠ Pathological" if in_path and freq > 0 else "Normal range",
                          freq_c), unsafe_allow_html=True)
with m3:
    st.markdown(tile_html("AI Prediction", label.split()[0], "",
                          f"{conf:.0f}% confidence", lc), unsafe_allow_html=True)
with m4:
    st.markdown(tile_html("Alerts", str(ac), "",
                          "Moderate / Severe",
                          "#ef4444" if ac > 0 else "#22c55e"),
                unsafe_allow_html=True)
with m5:
    st.markdown(tile_html("Readings", str(st.session_state.tick), "",
                          f"Session {mm:02d}:{ss:02d}"), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS — FFT / History / Log
# ═══════════════════════════════════════════════════════════════
st.markdown("")
tab_fft, tab_hist, tab_log = st.tabs([
    "🔬 FFT Spectrum", "📈 Frequency History", "📋 Data Log"
])

# ── FFT tab ──────────────────────────────────────────────────
with tab_fft:
    hv = st.session_state.vals
    if len(hv) < WIN:
        st.info(f"Need {WIN} samples for FFT — have {len(hv)} so far. Keep monitoring…")
    else:
        window  = np.array(hv[-WIN:])
        proc    = process_signal(window)
        fx, mag = fft_spectrum(proc)

        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(
            x=fx.tolist(), y=mag.tolist(),
            fill="tozeroy", mode="lines",
            line=dict(color="#00e5b4", width=1.5),
            fillcolor="rgba(0,229,180,0.08)",
        ))
        fig_fft.add_vrect(
            x0=PATHOLOGICAL_LO, x1=PATHOLOGICAL_HI,
            fillcolor="rgba(239,68,68,0.12)", layer="below", line_width=0,
            annotation_text="Parkinson's band 3–7 Hz",
            annotation_font_color="#ef4444", annotation_font_size=10,
            annotation_position="top left",
        )
        if freq > 0:
            vline_x = float(np.clip(freq, 0.01, 49.9))
            fig_fft.add_vline(
                x=vline_x, line_color="#f59e0b",
                line_dash="dash", line_width=2,
                annotation_text=f"Peak: {freq:.2f} Hz",
                annotation_font_color="#f59e0b", annotation_font_size=11,
            )
        fig_fft.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a1628",
            font=dict(color="#4a6fa5", size=10),
            margin=dict(l=8, r=8, t=20, b=8), height=260,
            xaxis=dict(showgrid=True, gridcolor="#0d1f38",
                       title="Frequency (Hz)", range=[0, 15]),
            yaxis=dict(showgrid=True, gridcolor="#0d1f38", title="Magnitude"),
        )
        st.plotly_chart(fig_fft, use_container_width=True,
                        config={"displayModeBar": False})

        ca, cb = st.columns(2)
        with ca:
            st.markdown(tile_html("Dominant Frequency", f"{freq:.2f}", "Hz",
                                  "⚠ In Parkinson band" if in_path else "✓ Outside pathological",
                                  freq_c), unsafe_allow_html=True)
        with cb:
            feat  = extract_features(proc)
            brat  = feat[7]
            bc    = "#ef4444" if brat > 0.4 else "#22c55e"
            st.markdown(tile_html("Band Power Ratio", f"{brat*100:.1f}", "%",
                                  "Power in 3–7 Hz / total", bc),
                        unsafe_allow_html=True)

# ── History tab ───────────────────────────────────────────────
with tab_hist:
    hf = st.session_state.freqs
    hl = st.session_state.labels
    if len(hf) < 5:
        st.info("Start monitoring to build history.")
    else:
        clr_map = {"Normal":"#22c55e","Mild Tremor":"#eab308",
                   "Moderate Tremor":"#f97316","Severe Tremor":"#ef4444","—":"#4a6fa5"}
        pt_c = [clr_map.get(l, "#4a6fa5") for l in hl]

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            y=hf, mode="lines+markers",
            marker=dict(color=pt_c, size=5),
            line=dict(color="rgba(96,165,250,.35)", width=1),
        ))
        fig_h.add_hrect(
            y0=PATHOLOGICAL_LO, y1=PATHOLOGICAL_HI,
            fillcolor="rgba(239,68,68,0.07)", layer="below", line_width=0,
            annotation_text="Pathological zone",
            annotation_font_color="#ef4444", annotation_font_size=9,
        )
        fig_h.add_hline(y=PATHOLOGICAL_LO, line_dash="dot",
                        line_color="rgba(239,68,68,.4)", line_width=1)
        fig_h.add_hline(y=PATHOLOGICAL_HI, line_dash="dot",
                        line_color="rgba(239,68,68,.4)", line_width=1)
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a1628",
            font=dict(color="#4a6fa5", size=10),
            margin=dict(l=8, r=8, t=20, b=8), height=270,
            xaxis=dict(showgrid=True, gridcolor="#0d1f38", title="Reading #"),
            yaxis=dict(showgrid=True, gridcolor="#0d1f38",
                       title="Hz", range=[0, 12]),
        )
        st.plotly_chart(fig_h, use_container_width=True,
                        config={"displayModeBar": False})

        counts = Counter(hl)
        total  = len(hl) or 1
        cols_b = st.columns(len(counts))
        for col, (lbl, cnt) in zip(cols_b, counts.items()):
            pct = cnt / total * 100
            c   = clr_map.get(lbl, "#4a6fa5")
            col.markdown(tile_html(lbl, f"{pct:.0f}", "%", f"{cnt} readings", c),
                         unsafe_allow_html=True)

# ── Log tab ───────────────────────────────────────────────────
with tab_log:
    if os.path.exists(LOG_FILE):
        try:
            df_l = (pd.read_csv(LOG_FILE)
                    .tail(60)
                    .sort_index(ascending=False)
                    .reset_index(drop=True))
            st.dataframe(df_l, use_container_width=True, height=340)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total readings",  len(df_l))
            s2.metric("Avg freq",        f"{df_l['freq_hz'].mean():.2f} Hz")
            s3.metric("Avg confidence",  f"{df_l['confidence'].mean():.0f}%")
            s4.metric("Severe events",   len(df_l[df_l["prediction"] == "Severe Tremor"]))
        except Exception as e:
            st.warning(f"Could not read log: {e}")
    else:
        st.info("No log yet. Enable 'Save to CSV log' in sidebar and start monitoring.")


# ═══════════════════════════════════════════════════════════════
# AUTO REFRESH
# ═══════════════════════════════════════════════════════════════
if mon and r:
    time.sleep(refresh_ms / 1000.0)
    st.rerun()
