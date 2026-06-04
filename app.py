"""
app.py  —  Parkinson's Tremor Live Detection Dashboard
IEEE / Biomedical Grade

Pipeline:  Arduino UNO  →  Serial 9600  →  Signal Processing  →  FFT  →  RF Classifier  →  Display

- NO simulation.  NO fake data.  Arduino must be physically connected.
- Binary classification: Tremor / Non-Tremor
- Live frequency display with Parkinson's band (3–7 Hz) highlighted
- Red alert on tremor, Green on normal
"""

# ── stdlib ─────────────────────────────────────────────────────
import os, time, threading, datetime
from collections import deque, Counter

# ── scientific ─────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── project modules ────────────────────────────────────────────
from signal_processing import (
    FS, WINDOW_SIZE, FEAT_COLS, TREMOR_LO, TREMOR_HI,
    process_signal, extract_features, features_to_vec,
    compute_fft, dominant_freq, freq_to_severity,
)
from serial_reader import ArduinoReader, list_ports, auto_detect_port, HAS_SERIAL
from train_model   import load_primary_model

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Parkinson's Tremor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS — clean biomedical dark theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, [class*="css"]                    { font-family:'Inter',sans-serif!important; }
.stApp, .main                        { background:#060d1a!important; color:#e2e8f0!important; }
.block-container                     { padding:1.1rem 1.6rem!important; max-width:100%!important; }
section[data-testid="stSidebar"]     { background:#09152a!important; border-right:1px solid #162340!important; }
[data-testid="stSidebar"] *          { color:#e2e8f0!important; }

/* ── top bar ─────────────────── */
.topbar {
  background:linear-gradient(135deg,#0d1e38,#091628);
  border:1px solid #162340;
  border-radius:12px;
  padding:14px 22px;
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:14px;
}
.topbar-title    { font-size:1.35rem; font-weight:800; color:#fff; letter-spacing:-.03em; }
.topbar-title em { color:#00e5b4; font-style:normal; }
.topbar-sub      { font-size:.72rem; color:#3d5a80; margin-top:2px; }

/* ── ALERT CARD — the main thing ─ */
.alert-card {
  border-radius:14px;
  padding:30px 24px;
  text-align:center;
  position:relative;
  overflow:hidden;
  transition:all .35s ease;
}
/* GREEN — no tremor */
.alert-green {
  background:radial-gradient(ellipse at 50% 0%, #052a18 0%, #041e12 100%);
  border:2px solid #16a34a;
  box-shadow:0 0 50px rgba(22,163,74,.22), inset 0 0 30px rgba(34,197,94,.04);
}
/* RED — tremor detected */
.alert-red {
  background:radial-gradient(ellipse at 50% 0%, #2a0505 0%, #1e0404 100%);
  border:2px solid #dc2626;
  box-shadow:0 0 60px rgba(220,38,38,.30), inset 0 0 30px rgba(239,68,68,.06);
  animation:redpulse 1.5s ease-in-out infinite;
}
@keyframes redpulse {
  0%,100% { box-shadow:0 0 60px rgba(220,38,38,.30); }
  50%     { box-shadow:0 0 90px rgba(220,38,38,.55); }
}
/* IDLE */
.alert-idle {
  background:linear-gradient(145deg,#0d1a2e,#091220);
  border:1.5px dashed #1a3050;
}

.alert-icon     { font-size:3.5rem; line-height:1; margin-bottom:10px; }
.alert-status   { font-size:2rem; font-weight:800; letter-spacing:-.02em; margin-bottom:8px; }
.alert-freq     { font-size:1.05rem; font-weight:600; margin-bottom:6px; }
.alert-severity { font-size:.85rem; opacity:.75; margin-bottom:12px; }
.alert-conf     { font-size:.78rem; opacity:.5; }

/* ── freq bar ─────────────────── */
.fbar-outer {
  background:#06101e;
  border:1px solid #162340;
  border-radius:8px;
  padding:14px 18px;
  margin-top:12px;
}
.fbar-label { font-size:.65rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#3d5a80; margin-bottom:8px; }
.fbar-track { background:#060d1a; border-radius:6px; height:12px; overflow:hidden; position:relative; }
.fbar-fill  { height:100%; border-radius:6px; transition:width .4s ease; }
.fbar-zones {
  display:flex; justify-content:space-between;
  font-size:.65rem; color:#3d5a80; margin-top:5px;
}
.fbar-readout { font-size:1.3rem; font-weight:700; margin-top:8px; }

/* ── metric tiles ─────────────── */
.tile {
  background:#0a1628;
  border:1px solid #162340;
  border-radius:11px;
  padding:14px 16px;
}
.tile-lbl  { font-size:.63rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#3d5a80; margin-bottom:5px; }
.tile-val  { font-size:1.75rem; font-weight:700; color:#fff; line-height:1; }
.tile-unit { font-size:.8rem; font-weight:400; color:#3d5a80; }
.tile-sub  { font-size:.67rem; color:#1e3a5f; margin-top:4px; }

/* ── badges ───────────────────── */
.badge { display:inline-flex; align-items:center; gap:6px; font-size:.7rem; font-weight:600; padding:4px 10px; border-radius:20px; }
.b-live  { background:rgba(0,229,180,.1); color:#00e5b4; border:1px solid rgba(0,229,180,.25); }
.b-off   { background:rgba(61,90,128,.12); color:#3d5a80; border:1px solid rgba(61,90,128,.25); }
.b-err   { background:rgba(239,68,68,.1);  color:#f87171; border:1px solid rgba(239,68,68,.25); }
.dot     { width:7px; height:7px; border-radius:50%; background:currentColor; display:inline-block; }
.pulse   { animation:dp 1.3s ease-in-out infinite; }
@keyframes dp { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── waiting screen ───────────── */
.wait-wrap {
  text-align:center; padding:56px 40px;
  background:#09152a; border:1.5px dashed #162340;
  border-radius:14px; margin:16px 0;
}
.wait-icon  { font-size:3.5rem; margin-bottom:14px; }
.wait-title { font-size:1.3rem; font-weight:700; color:#fff; margin-bottom:10px; }
.wait-sub   { font-size:.82rem; color:#3d5a80; line-height:1.8; }
.step-pill  {
  display:inline-block; background:#0d1a2e;
  border:1px solid #162340; border-radius:7px;
  padding:7px 16px; margin:4px; font-size:.78rem; color:#7a9cc0;
}
.step-pill b { color:#00e5b4; }

/* ── Streamlit overrides ──────── */
.stButton>button {
  background:#00e5b4!important; color:#000!important;
  border:none!important; font-weight:700!important;
  border-radius:9px!important; width:100%!important;
  font-size:.83rem!important; transition:all .2s!important;
}
.stButton>button:hover { opacity:.83!important; transform:translateY(-1px)!important; }
.stButton>button:disabled { background:#1a3050!important; color:#3d5a80!important; }
.stSelectbox label,.stSlider label,.stToggle label,.stTextInput label,.stCheckbox label {
  color:#3d5a80!important; font-size:.7rem!important;
  font-weight:700!important; text-transform:uppercase!important; letter-spacing:.06em!important;
}
div[data-testid="stExpander"] {
  background:#09152a!important; border:1px solid #162340!important; border-radius:9px!important;
}
#MainMenu, footer, header { visibility:hidden!important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
def _init():
    d = dict(
        reader=None, monitoring=False,
        vals=[], freqs=[], labels=[], confs=[],
        cur_label="—", cur_freq=0.0, cur_conf=0.0, cur_raw=0.0,
        tick=0, alert_count=0,
        t0=datetime.datetime.now(),
    )
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

MAX_HIST = 500
LOG_FILE = "tremor_log.csv"


# ══════════════════════════════════════════════════════════════
# LOAD MODEL — cached, trains automatically if pkl missing
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🧠 Loading RF classifier…")
def get_model():
    return load_primary_model()

model = get_model()


# ══════════════════════════════════════════════════════════════
# PREDICT on one 200-sample window
# ══════════════════════════════════════════════════════════════
def predict_window(window: np.ndarray):
    """
    Returns (binary_label, confidence, dominant_freq_hz, severity_str)
    binary_label: 0=Non-Tremor, 1=Tremor
    """
    proc = process_signal(window)
    freq = dominant_freq(proc)
    feat = features_to_vec(extract_features(proc))

    try:
        pred  = int(model.predict(feat.reshape(1,-1))[0])
        proba = model.predict_proba(feat.reshape(1,-1))[0]
        conf  = float(np.max(proba)) * 100
    except Exception:
        # Rule-based fallback
        pred = 1 if (TREMOR_LO <= freq <= TREMOR_HI) else 0
        conf = 80.0

    sev = freq_to_severity(freq)
    return pred, conf, freq, sev


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════
def log_row(ts, raw, freq, label, conf):
    try:
        pd.DataFrame([{
            "time": ts, "voltage": round(raw, 4),
            "freq_hz": round(freq, 2),
            "label": "Tremor" if label == 1 else "Non-Tremor",
            "severity": freq_to_severity(freq),
            "confidence": round(conf, 1),
        }]).to_csv(LOG_FILE, mode="a",
                   header=not os.path.exists(LOG_FILE), index=False)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Parkinson's Monitor")
    st.markdown("**IEEE Biomedical Grade**")
    st.markdown("---")

    # Cloud / no-serial warning
    if not HAS_SERIAL:
        st.error(
            "**pyserial not installed.**\n\n"
            "Run locally:\n"
            "```\npip install pyserial\nstreamlit run app.py\n```"
        )

    # Port selection
    st.markdown("### 📡 Arduino Port")
    ports     = list_ports()
    detected  = auto_detect_port() or ""

    if ports:
        idx         = ports.index(detected) if detected in ports else 0
        port_choice = st.selectbox("COM Port", ports, index=idx)
    else:
        port_choice = st.text_input("COM Port (manual)", value="COM3",
                                    placeholder="COM3  or  /dev/ttyUSB0")
        if HAS_SERIAL:
            st.warning("No ports found. Connect Arduino via USB.")

    baud_sel = st.selectbox("Baud Rate", [9600, 115200], index=0)

    st.markdown("---")

    # Start / Stop
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶ START", disabled=(not HAS_SERIAL))
    with c2:
        stop_btn  = st.button("⏹ STOP")

    if start_btn and HAS_SERIAL and port_choice:
        if st.session_state.reader:
            st.session_state.reader.stop()
        r = ArduinoReader(port_choice, baud_sel)
        r.start()
        st.session_state.reader    = r
        st.session_state.monitoring = True
        st.session_state.t0        = datetime.datetime.now()
        st.success(f"Connecting to {port_choice}…")

    if stop_btn:
        if st.session_state.reader:
            st.session_state.reader.stop()
        st.session_state.monitoring = False
        st.session_state.reader     = None
        st.info("Stopped.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    refresh_ms = st.slider("Refresh (ms)", 300, 2000, 500, 100)
    log_on     = st.checkbox("Log to CSV", True)

    st.markdown("---")
    if st.button("🗑 Clear Session"):
        for k in ["vals","freqs","labels","confs"]:
            st.session_state[k] = []
        st.session_state.alert_count = 0
        st.session_state.tick        = 0
        if os.path.exists(LOG_FILE):
            try: os.remove(LOG_FILE)
            except Exception: pass
        st.success("Cleared.")

    if os.path.exists(LOG_FILE):
        try:
            df_log = pd.read_csv(LOG_FILE)
            st.download_button(
                "⬇️ Download CSV",
                df_log.to_csv(index=False).encode(),
                f"tremor_{datetime.date.today()}.csv",
                "text/csv", use_container_width=True,
            )
        except Exception:
            pass

    st.markdown("---")
    st.caption(
        "tremor_sensor.ino → Arduino UNO A0\n"
        "100 Hz · 200-sample window · 50% overlap\n"
        "RF Binary Classifier: Tremor / Non-Tremor\n"
        "Pathological band: 3–7 Hz (Parkinson's)\n"
        "IEEE Biomedical Signal Processing"
    )


# ══════════════════════════════════════════════════════════════
# INGEST — pull one tick of real data from Arduino
# ══════════════════════════════════════════════════════════════
r            = st.session_state.reader
mon          = st.session_state.monitoring
is_connected = r is not None and getattr(r, "connected", False)
err_msg      = getattr(r, "error", "") if r else ""

if r and mon and is_connected:
    _, val_list = r.latest(WINDOW_SIZE)
    n = len(val_list)

    if n >= 20:
        arr    = np.array(val_list, dtype=float)
        raw    = float(arr[-1])
        window = arr if n >= WINDOW_SIZE else np.pad(arr, (WINDOW_SIZE - n, 0))

        label, conf, freq, sev = predict_window(window)

        st.session_state.cur_label = label
        st.session_state.cur_freq  = freq
        st.session_state.cur_conf  = conf
        st.session_state.cur_raw   = raw
        st.session_state.tick     += 1

        st.session_state.vals.append(raw)
        st.session_state.freqs.append(freq)
        st.session_state.labels.append(label)
        st.session_state.confs.append(conf)

        if label == 1:
            st.session_state.alert_count += 1

        for k in ["vals","freqs","labels","confs"]:
            if len(st.session_state[k]) > MAX_HIST:
                st.session_state[k] = st.session_state[k][-MAX_HIST:]

        if log_on:
            ts_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_row(ts_str, raw, freq, label, conf)


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
dur    = (datetime.datetime.now() - st.session_state.t0).seconds
mm, ss = divmod(dur, 60)

if is_connected and mon:
    badge = '<span class="badge b-live"><span class="dot pulse"></span>ARDUINO LIVE</span>'
elif err_msg:
    badge = f'<span class="badge b-err">⚠ {err_msg[:45]}</span>'
else:
    badge = '<span class="badge b-off"><span class="dot"></span>WAITING FOR ARDUINO</span>'

st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-title">🧠 Parkinson's <em>Tremor Detector</em></div>
    <div class="topbar-sub">
      Arduino UNO → Serial 9600 → 100 Hz → FFT → RF Binary Classifier · IEEE Biomedical
    </div>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    {badge}
    <span class="badge b-off">🕒 {mm:02d}:{ss:02d}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN  —  3 states
# ══════════════════════════════════════════════════════════════

# ── STATE 1: No pyserial (Streamlit Cloud) ────────────────────
if not HAS_SERIAL:
    st.markdown("""
    <div class="wait-wrap">
      <div class="wait-icon">☁️</div>
      <div class="wait-title">pyserial not available</div>
      <div class="wait-sub">
        This app reads <b>real voltage data from Arduino UNO</b>.<br>
        It does <b>not simulate</b> data — that would defeat the purpose of a research project.<br><br>
        Run this app locally on your laptop with Arduino connected:
      </div><br>
      <span class="step-pill"><b>1</b> pip install -r requirements.txt</span>
      <span class="step-pill"><b>2</b> Upload tremor_sensor.ino to Arduino UNO</span>
      <span class="step-pill"><b>3</b> Plug USB cable into laptop</span>
      <span class="step-pill"><b>4</b> streamlit run app.py</span>
      <span class="step-pill"><b>5</b> Select COM port → ▶ START</span>
      <br><br>
      <div class="wait-sub">
        Sensor on A0 → 100 Hz signal → FFT → RF model → Tremor / Non-Tremor + frequency
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── STATE 2: pyserial present but Arduino not connected ────────
elif not mon or not is_connected:
    ports = list_ports()

    connecting_html = ""
    if mon and not is_connected and not err_msg:
        connecting_html = """
        <div style="background:#0d1a2e;border:1px solid #162340;border-radius:8px;
                    padding:10px 16px;margin:10px 0;font-size:.8rem;color:#3d5a80;
                    display:flex;align-items:center;gap:8px">
          <span class="dot pulse" style="color:#eab308">●</span>
          Connecting to Arduino… please wait
        </div>"""
    elif err_msg:
        connecting_html = f"""
        <div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);
                    border-radius:8px;padding:10px 16px;margin:10px 0;
                    font-size:.8rem;color:#f87171">
          ⚠ {err_msg}
        </div>"""

    st.markdown(f"""
    <div class="wait-wrap">
      <div class="wait-icon">🔌</div>
      <div class="wait-title">Connect Arduino UNO</div>
      <div class="wait-sub">
        No artificial data will be shown.<br>
        This dashboard only displays <b>real sensor readings</b> from your TENG/voltage sensor.<br>
        Shake your hand near the sensor — the system detects tremor frequency in real time.
      </div><br>
      <span class="step-pill"><b>1</b> Upload tremor_sensor.ino</span>
      <span class="step-pill"><b>2</b> Connect USB cable</span>
      <span class="step-pill"><b>3</b> Select COM port in sidebar</span>
      <span class="step-pill"><b>4</b> Click ▶ START</span>
      {connecting_html}
    </div>
    """, unsafe_allow_html=True)

    if ports:
        st.info(f"📡 **Ports detected:** {', '.join(ports)}  — select in sidebar and press ▶ START")
    else:
        st.warning("No COM ports found. Connect Arduino via USB, then refresh.")

# ── STATE 3: ARDUINO LIVE ─────────────────────────────────────
else:
    label = st.session_state.cur_label
    freq  = st.session_state.cur_freq
    conf  = st.session_state.cur_conf
    raw   = st.session_state.cur_raw
    hv    = st.session_state.vals
    hf    = st.session_state.freqs
    hl    = st.session_state.labels

    in_path = TREMOR_LO <= freq <= TREMOR_HI and freq > 0
    n_samp  = r.sample_count() if r else 0

    # ── Sample collection progress bar ────────────────────────
    if n_samp < WINDOW_SIZE:
        pct = int(n_samp / WINDOW_SIZE * 100)
        st.markdown(f"""
        <div style="background:#0a1628;border:1px solid #162340;border-radius:8px;
                    padding:10px 18px;margin-bottom:12px;font-size:.8rem;color:#3d5a80;
                    display:flex;align-items:center;gap:10px">
          <span class="dot pulse" style="color:#00e5b4">●</span>
          Collecting samples: <b style="color:#e2e8f0">{n_samp} / {WINDOW_SIZE}</b>
          ({pct}%) — shake your hand near the sensor
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # LEFT: Alert card  +  Frequency bar
    # RIGHT: Live chart
    # ─────────────────────────────────────────────────────────
    left, right = st.columns([1.0, 2.0])

    with left:
        # ── BIG ALERT CARD ────────────────────────────────────
        if label == "—":
            card_cls = "alert-idle"
            icon     = "⏳"
            status   = "Collecting Data…"
            freq_str = "Analysing signal…"
            sev_str  = ""
            conf_str = ""
            color    = "#3d5a80"
        elif label == 1:                        # TREMOR
            card_cls = "alert-red"
            icon     = "🚨"
            status   = "TREMOR DETECTED"
            sev      = freq_to_severity(freq)
            freq_str = f"⚡ {freq:.2f} Hz  (Parkinson band: 3–7 Hz)"
            sev_str  = sev
            conf_str = f"RF confidence: {conf:.0f}%"
            color    = "#ef4444"
        else:                                   # NON-TREMOR
            card_cls = "alert-green"
            icon     = "✅"
            status   = "NO TREMOR"
            freq_str = f"⚡ {freq:.2f} Hz  (Normal range)"
            sev_str  = "Normal hand movement"
            conf_str = f"RF confidence: {conf:.0f}%"
            color    = "#22c55e"

        st.markdown(f"""
        <div class="alert-card {card_cls}">
          <div class="alert-icon">{icon}</div>
          <div class="alert-status" style="color:{color}">{status}</div>
          <div class="alert-freq">{freq_str}</div>
          <div class="alert-severity">{sev_str}</div>
          <div class="alert-conf">{conf_str}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Frequency bar ────────────────────────────────────
        band_pct  = min(100.0, freq / 12.0 * 100) if freq > 0 else 0.0
        if freq < TREMOR_LO:
            bar_col = "#22c55e"
        elif freq < 4.5:
            bar_col = "#eab308"
        elif freq < 6.0:
            bar_col = "#f97316"
        else:
            bar_col = "#ef4444"

        path_note = (
            f'<span style="color:#ef4444">⚠ In Parkinson band</span>'
            if in_path and freq > 0
            else ('<span style="color:#22c55e">✓ Normal range</span>' if freq > 0 else "")
        )

        st.markdown(f"""
        <div class="fbar-outer">
          <div class="fbar-label">Frequency Position</div>
          <div class="fbar-zones" style="margin-bottom:5px">
            <span>0 Hz</span>
            <span style="color:#ef4444">3–7 Hz (Parkinson)</span>
            <span>12 Hz</span>
          </div>
          <div class="fbar-track">
            <div class="fbar-fill" style="width:{band_pct:.1f}%;background:{bar_col}"></div>
          </div>
          <div class="fbar-readout" style="color:{bar_col}">
            {freq:.2f} Hz &nbsp; {path_note}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        # ── LIVE CHART ────────────────────────────────────────
        if len(hv) < 5:
            st.markdown("""
            <div style="background:#09152a;border:1px solid #162340;border-radius:10px;
                        height:200px;display:flex;align-items:center;justify-content:center;
                        flex-direction:column;gap:8px;color:#3d5a80;font-size:.85rem">
              <span style="font-size:2rem">📡</span>
              Receiving sensor data… shake your hand
            </div>""", unsafe_allow_html=True)
        else:
            n_show = min(len(hv), 200)
            yv = hv[-n_show:]
            yf = hf[-n_show:]
            x  = list(range(n_show))
            pt_c = ["#ef4444" if l == 1 else "#22c55e" for l in hl[-n_show:]]

            fig = go.Figure()
            # voltage trace
            fig.add_trace(go.Scatter(
                x=x, y=yv, mode="lines", name="Voltage (V)",
                line=dict(color="rgba(0,229,180,.45)", width=1.2),
                yaxis="y1",
            ))
            # frequency trace
            fig.add_trace(go.Scatter(
                x=x, y=yf, mode="lines", name="Freq (Hz)",
                line=dict(color="#60a5fa", width=2.2),
                yaxis="y2",
            ))
            # Parkinson band shading on freq axis
            fig.add_hrect(
                yref="y2", y0=TREMOR_LO, y1=TREMOR_HI,
                fillcolor="rgba(239,68,68,.08)", layer="below", line_width=0,
            )
            # 3 Hz reference line
            fig.add_hline(y=TREMOR_LO, yref="y2", line_dash="dot",
                          line_color="rgba(239,68,68,.5)", line_width=1,
                          annotation_text="3 Hz", annotation_font_size=9,
                          annotation_font_color="#ef4444",
                          annotation_position="top left")
            fig.add_hline(y=TREMOR_HI, yref="y2", line_dash="dot",
                          line_color="rgba(239,68,68,.5)", line_width=1,
                          annotation_text="7 Hz", annotation_font_size=9,
                          annotation_font_color="#ef4444",
                          annotation_position="top left")

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#09152a",
                font=dict(color="#3d5a80", size=10),
                margin=dict(l=8, r=8, t=36, b=8),
                height=310,
                title=dict(text="Real-time Sensor Signal + Detected Tremor Frequency",
                           font_color="#3d5a80", font_size=11),
                legend=dict(orientation="h", y=1.13, x=0,
                            bgcolor="rgba(0,0,0,0)", font_size=10),
                xaxis=dict(showgrid=True, gridcolor="#0d1a2e", title="Samples"),
                yaxis=dict(showgrid=True, gridcolor="#0d1a2e",
                           title="Voltage (V)", side="left", range=[0, 5]),
                yaxis2=dict(showgrid=False, title="Freq (Hz)",
                            overlaying="y", side="right", range=[0, 12]),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    # ── METRIC TILES ──────────────────────────────────────────
    st.markdown("")

    def tile(lbl, val, unit="", sub="", color="#fff"):
        return (f'<div class="tile"><div class="tile-lbl">{lbl}</div>'
                f'<div class="tile-val" style="color:{color}">{val}'
                f'<span class="tile-unit"> {unit}</span></div>'
                f'<div class="tile-sub">{sub}</div></div>')

    freq_c  = "#ef4444" if in_path and freq > 0 else ("#22c55e" if freq > 0 else "#3d5a80")
    lbl_col = "#ef4444" if label == 1 else ("#22c55e" if label == 0 else "#3d5a80")
    lbl_txt = "TREMOR" if label == 1 else ("NORMAL" if label == 0 else "—")
    ac      = st.session_state.alert_count

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(tile("Voltage", f"{raw:.3f}", "V", "Sensor A0"), unsafe_allow_html=True)
    with c2:
        st.markdown(tile("Tremor Freq", f"{freq:.2f}", "Hz",
                         "⚠ Pathological" if in_path and freq > 0 else "Normal",
                         freq_c), unsafe_allow_html=True)
    with c3:
        st.markdown(tile("Diagnosis", lbl_txt, "",
                         f"{conf:.0f}% RF confidence", lbl_col), unsafe_allow_html=True)
    with c4:
        st.markdown(tile("Tremor Events", str(ac), "",
                         "Detections this session",
                         "#ef4444" if ac > 0 else "#22c55e"), unsafe_allow_html=True)
    with c5:
        st.markdown(tile("Samples", str(st.session_state.tick), "",
                         f"Session {mm:02d}:{ss:02d}"), unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────
    st.markdown("")
    t1, t2, t3 = st.tabs(["🔬 FFT Spectrum", "📈 History", "📋 Log"])

    with t1:
        if len(hv) < WINDOW_SIZE:
            st.info(f"Need {WINDOW_SIZE} samples for FFT — have {len(hv)} so far…")
        else:
            window = np.array(hv[-WINDOW_SIZE:])
            proc   = process_signal(window)
            fx, mg = compute_fft(proc)
            vx     = float(np.clip(freq, 0.01, 49.9)) if freq > 0 else 0.5

            f2 = go.Figure()
            f2.add_trace(go.Scatter(
                x=fx.tolist(), y=mg.tolist(), fill="tozeroy", mode="lines",
                line=dict(color="#00e5b4", width=1.5),
                fillcolor="rgba(0,229,180,.07)",
            ))
            f2.add_vrect(x0=TREMOR_LO, x1=TREMOR_HI,
                         fillcolor="rgba(239,68,68,.10)", layer="below", line_width=0,
                         annotation_text="Parkinson band 3–7 Hz",
                         annotation_font_color="#ef4444", annotation_font_size=10,
                         annotation_position="top left")
            if freq > 0:
                f2.add_vline(x=vx, line_color="#f59e0b", line_dash="dash", line_width=2,
                             annotation_text=f"Peak: {freq:.2f} Hz",
                             annotation_font_color="#f59e0b", annotation_font_size=11)
            f2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#09152a",
                font=dict(color="#3d5a80", size=10),
                margin=dict(l=8,r=8,t=20,b=8), height=250,
                xaxis=dict(showgrid=True, gridcolor="#0d1a2e",
                           title="Frequency (Hz)", range=[0,15]),
                yaxis=dict(showgrid=True, gridcolor="#0d1a2e", title="Magnitude"),
            )
            st.plotly_chart(f2, use_container_width=True,
                            config={"displayModeBar": False})

            feat_d = extract_features(proc)
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown(tile("Dominant Freq", f"{freq:.2f}", "Hz",
                                 "⚠ Pathological" if in_path else "Normal", freq_c),
                            unsafe_allow_html=True)
            with cb:
                br = feat_d['band_power']
                bc = "#ef4444" if br > 0.5 else "#22c55e"
                st.markdown(tile("Band Power (3–7 Hz)", f"{br:.4f}", "V²/Hz",
                                 "Tremor band energy", bc), unsafe_allow_html=True)
            with cc:
                bpr_total = feat_d['psd_peak']
                st.markdown(tile("PSD Peak", f"{bpr_total:.4f}", "V²/Hz",
                                 "Max power spectral density", "#60a5fa"),
                            unsafe_allow_html=True)

    with t2:
        if len(hf) < 5:
            st.info("More data needed…")
        else:
            fh = go.Figure()
            pt_c2 = ["#ef4444" if l==1 else "#22c55e" for l in hl]
            fh.add_trace(go.Scatter(
                y=hf, mode="lines+markers",
                marker=dict(color=pt_c2, size=5),
                line=dict(color="rgba(96,165,250,.3)", width=1),
                name="Tremor Freq"
            ))
            fh.add_hrect(y0=TREMOR_LO, y1=TREMOR_HI,
                         fillcolor="rgba(239,68,68,.07)", layer="below", line_width=0)
            fh.add_hline(y=TREMOR_LO, line_dash="dot",
                         line_color="rgba(239,68,68,.4)", line_width=1)
            fh.add_hline(y=TREMOR_HI, line_dash="dot",
                         line_color="rgba(239,68,68,.4)", line_width=1)
            fh.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#09152a",
                font=dict(color="#3d5a80", size=10),
                margin=dict(l=8,r=8,t=20,b=8), height=260,
                xaxis=dict(showgrid=True, gridcolor="#0d1a2e", title="Reading #"),
                yaxis=dict(showgrid=True, gridcolor="#0d1a2e",
                           title="Hz", range=[0,12]),
            )
            st.plotly_chart(fh, use_container_width=True,
                            config={"displayModeBar": False})

            total  = len(hl) or 1
            n_trem = sum(1 for l in hl if l == 1)
            n_norm = total - n_trem
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.markdown(tile("Tremor Readings", str(n_trem), "",
                                 f"{n_trem/total*100:.0f}% of total", "#ef4444"),
                            unsafe_allow_html=True)
            with bc2:
                st.markdown(tile("Normal Readings", str(n_norm), "",
                                 f"{n_norm/total*100:.0f}% of total", "#22c55e"),
                            unsafe_allow_html=True)
            with bc3:
                avg_f = np.mean(hf) if hf else 0.0
                st.markdown(tile("Mean Frequency", f"{avg_f:.2f}", "Hz",
                                 "Session average", "#60a5fa"), unsafe_allow_html=True)

    with t3:
        if os.path.exists(LOG_FILE):
            try:
                df_l = (pd.read_csv(LOG_FILE)
                        .tail(60).sort_index(ascending=False)
                        .reset_index(drop=True))
                st.dataframe(df_l, use_container_width=True, height=310)
                s1,s2,s3,s4 = st.columns(4)
                s1.metric("Total",      len(df_l))
                s2.metric("Avg Freq",   f"{df_l['freq_hz'].mean():.2f} Hz")
                s3.metric("Avg Conf",   f"{df_l['confidence'].mean():.0f}%")
                s4.metric("Tremor",     len(df_l[df_l['label']=='Tremor']))
            except Exception as e:
                st.warning(f"Log error: {e}")
        else:
            st.info("No log yet. Enable logging in sidebar.")


# ══════════════════════════════════════════════════════════════
# AUTO REFRESH — only when live Arduino data is flowing
# ══════════════════════════════════════════════════════════════
if mon and r and is_connected:
    time.sleep(refresh_ms / 1000.0)
    st.rerun()
