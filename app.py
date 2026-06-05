"""
app.py
======
Parkinson's Tremor Real-Time Detection Dashboard


Hardware pipeline:
   

Serial format from sketch_feb18a.ino:
    "<time_seconds> <voltage_V>\\n"
    e.g.  "1.230 2.4812"
    Rate: 100 Hz (samplingInterval = 10ms)
    Baud: 9600

Rules:
    - NO simulation. NO fake data. Arduino must be physically connected.
    - Arduino NOT connected → shows waiting screen, zero data displayed.
    - Arduino connected     → real signal, FFT, RF prediction, alerts.
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import os, time, datetime
from collections import Counter

# ── scientific ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── project modules ─────────────────────────────────────────────────────────
from signal_processing import (
    FS, WINDOW_SIZE, FEAT_COLS,
    TREMOR_LO, TREMOR_HI,
    process_signal, extract_features, features_to_vec,
    compute_fft, dominant_freq, freq_to_severity,
)
from serial_reader import (
    ArduinoReader, list_ports, auto_detect_port, HAS_SERIAL,
)
from train_model import load_model

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Parkinson's Tremor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# THEME  —  clean clinical dark UI
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, [class*="css"]                      { font-family:'Inter',sans-serif!important; }
.stApp, .main                          { background:#060d1a!important; color:#e2e8f0!important; }
.block-container                       { padding:1rem 1.5rem!important; max-width:100%!important; }
section[data-testid="stSidebar"]       { background:#080f1e!important; border-right:1px solid #112036!important; }
[data-testid="stSidebar"] *            { color:#e2e8f0!important; }

/* ── header ─────────────────────────────────── */
.hdr {
  background:linear-gradient(135deg,#0c1b34,#081426);
  border:1px solid #112036; border-radius:12px;
  padding:14px 22px; margin-bottom:14px;
  display:flex; align-items:center; justify-content:space-between;
}
.hdr-title     { font-size:1.4rem; font-weight:800; color:#fff; letter-spacing:-.03em; }
.hdr-title em  { color:#00e5b4; font-style:normal; }
.hdr-sub       { font-size:.7rem; color:#2c4a6e; margin-top:3px; letter-spacing:.02em; }

/* ── main alert card ────────────────────────── */
.acard {
  border-radius:14px; padding:28px 22px;
  text-align:center; transition:all .35s;
}
.a-green {
  background:radial-gradient(ellipse at 50% 0%,#052a18,#031610);
  border:2px solid #16a34a;
  box-shadow:0 0 48px rgba(22,163,74,.22);
}
.a-red {
  background:radial-gradient(ellipse at 50% 0%,#2a0505,#180303);
  border:2px solid #dc2626;
  box-shadow:0 0 60px rgba(220,38,38,.32);
  animation:rpulse 1.4s ease-in-out infinite;
}
@keyframes rpulse {
  0%,100% { box-shadow:0 0 60px rgba(220,38,38,.32); }
  50%     { box-shadow:0 0 95px rgba(220,38,38,.58); }
}
.a-idle {
  background:#09152a; border:1.5px dashed #112036;
}
.ac-icon  { font-size:3.2rem; line-height:1; margin-bottom:10px; }
.ac-title { font-size:1.8rem; font-weight:800; letter-spacing:-.02em; margin-bottom:6px; }
.ac-freq  { font-size:1rem; font-weight:600; margin-bottom:5px; }
.ac-sev   { font-size:.82rem; opacity:.72; margin-bottom:10px; }
.ac-conf  { font-size:.74rem; opacity:.48; }

/* ── frequency bar ──────────────────────────── */
.fbar-card {
  background:#08131f; border:1px solid #112036;
  border-radius:10px; padding:14px 18px; margin-top:12px;
}
.fbar-lbl   { font-size:.63rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#2c4a6e; margin-bottom:8px; }
.fbar-track { background:#040b14; border-radius:6px; height:12px; overflow:hidden; }
.fbar-fill  { height:100%; border-radius:6px; transition:width .4s ease; }
.fbar-ticks { display:flex; justify-content:space-between; font-size:.63rem; color:#2c4a6e; margin-top:4px; }
.fbar-val   { font-size:1.2rem; font-weight:700; margin-top:8px; }

/* ── metric tiles ───────────────────────────── */
.tile       { background:#08131f; border:1px solid #112036; border-radius:11px; padding:14px 16px; }
.tile-lbl   { font-size:.62rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#2c4a6e; margin-bottom:5px; }
.tile-val   { font-size:1.75rem; font-weight:700; color:#fff; line-height:1; }
.tile-unit  { font-size:.78rem; font-weight:400; color:#2c4a6e; }
.tile-sub   { font-size:.66rem; color:#122030; margin-top:4px; }

/* ── badges ─────────────────────────────────── */
.badge { display:inline-flex; align-items:center; gap:6px; font-size:.7rem; font-weight:600; padding:4px 10px; border-radius:20px; }
.b-live { background:rgba(0,229,180,.1);  color:#00e5b4; border:1px solid rgba(0,229,180,.25); }
.b-off  { background:rgba(44,74,110,.12); color:#2c4a6e; border:1px solid rgba(44,74,110,.25); }
.b-err  { background:rgba(239,68,68,.1);  color:#f87171; border:1px solid rgba(239,68,68,.25); }
.dot    { width:7px; height:7px; border-radius:50%; background:currentColor; display:inline-block; }
.pulse  { animation:dp 1.3s ease-in-out infinite; }
@keyframes dp { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── waiting screen ─────────────────────────── */
.wait {
  text-align:center; padding:52px 36px;
  background:#08131f; border:1.5px dashed #112036;
  border-radius:14px; margin:14px 0;
}
.wait-icon  { font-size:3.2rem; margin-bottom:14px; }
.wait-title { font-size:1.25rem; font-weight:700; color:#e2e8f0; margin-bottom:10px; }
.wait-sub   { font-size:.8rem; color:#2c4a6e; line-height:1.8; }
.step { display:inline-block; background:#0c1b34; border:1px solid #112036;
        border-radius:7px; padding:7px 15px; margin:4px;
        font-size:.77rem; color:#5a7fa8; }
.step b { color:#00e5b4; }

/* ── Streamlit overrides ────────────────────── */
.stButton>button {
  background:#00e5b4!important; color:#000!important; border:none!important;
  font-weight:700!important; border-radius:9px!important; width:100%!important;
  font-size:.82rem!important; transition:all .2s!important; padding:9px!important;
}
.stButton>button:hover    { opacity:.83!important; transform:translateY(-1px)!important; }
.stButton>button:disabled { background:#0d1f38!important; color:#2c4a6e!important; }
.stSelectbox label, .stSlider label, .stToggle label,
.stTextInput label, .stCheckbox label {
  color:#2c4a6e!important; font-size:.7rem!important;
  font-weight:700!important; text-transform:uppercase!important; letter-spacing:.06em!important;
}
div[data-testid="stExpander"] {
  background:#08131f!important; border:1px solid #112036!important; border-radius:9px!important;
}
#MainMenu, footer, header { visibility:hidden!important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
def _init():
    defaults = dict(
        reader=None, monitoring=False,
        vals=[], freqs=[], labels=[], confs=[],
        cur_label=-1, cur_freq=0.0,
        cur_conf=0.0,  cur_raw=0.0,
        tick=0, alert_count=0,
        t0=datetime.datetime.now(),
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

LOG_FILE = "logs/tremor_log.csv"
MAX_HIST = 600
os.makedirs("logs", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL  (auto-trains if missing)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🧠 Loading RF model…")
def get_model():
    return load_model()

model = get_model()


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION  on one 200-sample window
# ═══════════════════════════════════════════════════════════════════════════
def predict_window(window: np.ndarray):
    """
    Process raw voltage window → RF binary prediction.
    Returns (label_int, confidence_pct, dom_freq_hz, severity_str)
        label_int: 0=Non-Tremor  1=Tremor
    """
    proc = process_signal(window)
    freq = dominant_freq(proc)
    vec  = features_to_vec(extract_features(proc)).reshape(1, -1)

    try:
        label = int(model.predict(vec)[0])
        proba = model.predict_proba(vec)[0]
        conf  = float(np.max(proba)) * 100
    except Exception:
        # Rule-based fallback if model fails
        label = 1 if (TREMOR_LO <= freq <= TREMOR_HI) else 0
        conf  = 80.0

    return label, conf, freq, freq_to_severity(freq)


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════
def log_row(ts, raw, freq, label, conf):
    try:
        pd.DataFrame([{
            "time":       ts,
            "voltage_V":  round(raw,  4),
            "freq_hz":    round(freq, 2),
            "result":     "Tremor" if label == 1 else "Non-Tremor",
            "severity":   freq_to_severity(freq),
            "confidence": round(conf, 1),
        }]).to_csv(LOG_FILE, mode="a",
                   header=not os.path.exists(LOG_FILE), index=False)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Parkinson's Monitor")
    st.markdown("**IEEE Biomedical**")
    st.markdown("---")

    # ── Cloud / no-pyserial notice ────────────────────────────
    if not HAS_SERIAL:
        st.error(
            "**pyserial not installed.**\n\n"
            "Run locally with Arduino:\n"
            "```\npip install -r requirements.txt\n"
            "streamlit run app.py\n```"
        )
    else:
        st.success("✅ pyserial ready")

    # ── Port selection ────────────────────────────────────────
    st.markdown("### 📡 Arduino Connection")
    ports    = list_ports()
    detected = auto_detect_port() or ""

    if ports:
        idx  = ports.index(detected) if detected in ports else 0
        port = st.selectbox("COM Port", ports, index=idx)
    else:
        port = st.text_input("COM Port (manual)",
                             value="COM3",
                             placeholder="COM3 or /dev/ttyUSB0")
        if HAS_SERIAL:
            st.warning("No ports found. Connect Arduino.")

    baud_opt = st.selectbox("Baud Rate", [9600, 115200], index=0,
                            help="sketch_feb18a.ino uses Serial.begin(9600)")

    st.markdown("---")

    # ── Start / Stop ──────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶ START", disabled=not HAS_SERIAL)
    with c2:
        stop_btn  = st.button("⏹ STOP")

    if start_btn and HAS_SERIAL and port:
        if st.session_state.reader:
            st.session_state.reader.stop()
        r = ArduinoReader(port, baud_opt)
        r.start()
        st.session_state.reader    = r
        st.session_state.monitoring = True
        st.session_state.t0        = datetime.datetime.now()
        st.success(f"Connecting to {port}…")

    if stop_btn:
        if st.session_state.reader:
            st.session_state.reader.stop()
        st.session_state.monitoring = False
        st.session_state.reader     = None
        st.info("Stopped.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    refresh_ms = st.slider("Refresh (ms)", 300, 2000, 500, 100)
    log_on     = st.checkbox("Log to CSV", value=True)

    st.markdown("---")
    if st.button("🗑 Clear Session"):
        for k in ["vals","freqs","labels","confs"]:
            st.session_state[k] = []
        st.session_state.alert_count = 0
        st.session_state.tick        = 0
        try:
            os.remove(LOG_FILE)
        except Exception:
            pass
        st.success("Cleared.")

    if os.path.exists(LOG_FILE):
        try:
            dl = pd.read_csv(LOG_FILE)
            st.download_button(
                "⬇️ Download CSV",
                dl.to_csv(index=False).encode(),
                f"tremor_{datetime.date.today()}.csv",
                "text/csv",
                use_container_width=True,
            )
        except Exception:
            pass

    st.markdown("---")
    st.caption(
        "sketch_feb18a.ino → Arduino UNO A0\n"
        "100 Hz · 9600 baud · WINDOW=200 samples\n"
        "Binary RF: Tremor / Non-Tremor\n"
        "Parkinson's band: 3–7 Hz\n"
        "Features: mean, std, rms, energy,\n"
        "dom_freq, entropy, psd_peak, band_power"
    )


# ═══════════════════════════════════════════════════════════════════════════
# INGEST  —  read one tick of real Arduino data
# ═══════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# HEADER BAR
# ═══════════════════════════════════════════════════════════════════════════
dur    = (datetime.datetime.now() - st.session_state.t0).seconds
mm, ss = divmod(dur, 60)

if is_connected and mon:
    badge = '<span class="badge b-live"><span class="dot pulse"></span>ARDUINO LIVE</span>'
elif err_msg:
    badge = f'<span class="badge b-err">⚠ {err_msg[:45]}</span>'
else:
    badge = '<span class="badge b-off"><span class="dot"></span>WAITING FOR ARDUINO</span>'

st.markdown(f"""
<div class="hdr">
  <div>
    <div class="hdr-title">🧠 Parkinson's <em>Tremor Detector</em></div>
    <div class="hdr-sub">
      sketch_feb18a.ino → Arduino UNO A0 → 100 Hz → Serial 9600 →
      FFT + Binary RF → Live Detection
    </div>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    {badge}
    <span class="badge b-off">🕒 {mm:02d}:{ss:02d}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# THREE STATES
# ═══════════════════════════════════════════════════════════════════════════

# ── STATE 1: pyserial missing  (Streamlit Cloud / no install) ─────────────
if not HAS_SERIAL:
    st.markdown("""
    <div class="wait">
      <div class="wait-icon">☁️</div>
      <div class="wait-title">pyserial not installed</div>
      <div class="wait-sub">
        This app reads <b>real voltage data</b> from your TENG sensor
        connected to <b>Arduino A0</b> via <code>sketch_feb18a.ino</code>.<br>
        It never shows artificial/simulated data.<br><br>
        To run locally with Arduino:
      </div><br>
      <span class="step"><b>1</b> Clone / download this repo</span>
      <span class="step"><b>2</b> pip install -r requirements.txt</span>
      <span class="step"><b>3</b> Upload sketch_feb18a.ino to Arduino UNO</span>
      <span class="step"><b>4</b> Connect USB cable to laptop</span>
      <span class="step"><b>5</b> streamlit run app.py</span>
      <span class="step"><b>6</b> Select COM port → ▶ START</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── STATE 2: pyserial ready but Arduino not yet connected ─────────────────
elif not mon or not is_connected:
    ports = list_ports()

    status_html = ""
    if mon and not is_connected and not err_msg:
        status_html = """
        <div style="background:#0c1b34;border:1px solid #112036;border-radius:8px;
                    padding:10px 16px;margin:10px 0;font-size:.79rem;color:#2c4a6e;
                    display:flex;align-items:center;gap:8px">
          <span class="dot pulse" style="color:#eab308">●</span>
          Connecting to Arduino… please wait
        </div>"""
    elif err_msg:
        status_html = f"""
        <div style="background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.2);
                    border-radius:8px;padding:10px 16px;margin:10px 0;
                    font-size:.79rem;color:#f87171">
          ⚠ {err_msg}
        </div>"""

    st.markdown(f"""
    <div class="wait">
      <div class="wait-icon">🔌</div>
      <div class="wait-title">Connect Arduino UNO</div>
      <div class="wait-sub">
        No artificial data will be shown.<br>
        This dashboard displays <b>only real sensor readings</b>
        from your TENG / voltage sensor on <b>A0</b>.<br>
        Shake your hand near the sensor — the system detects
        tremor frequency in real time.
      </div><br>
      <span class="step"><b>1</b> Upload <code>sketch_feb18a.ino</code></span>
      <span class="step"><b>2</b> Connect USB cable</span>
      <span class="step"><b>3</b> Select COM port in sidebar</span>
      <span class="step"><b>4</b> Click ▶ START</span>
      {status_html}
    </div>
    """, unsafe_allow_html=True)

    if ports:
        st.info(f"📡 **Ports detected:** `{'` , `'.join(ports)}`  — select in sidebar → ▶ START")
    else:
        st.warning("No COM ports found. Connect Arduino via USB and refresh.")

# ── STATE 3: Arduino connected — LIVE DASHBOARD ──────────────────────────
else:
    label = st.session_state.cur_label
    freq  = st.session_state.cur_freq
    conf  = st.session_state.cur_conf
    raw   = st.session_state.cur_raw
    hv    = st.session_state.vals
    hf    = st.session_state.freqs
    hl    = st.session_state.labels
    n_samp = r.sample_count() if r else 0
    in_path = TREMOR_LO <= freq <= TREMOR_HI and freq > 0

    # ── sample-collection progress ────────────────────────────
    if n_samp < WINDOW_SIZE:
        pct = int(n_samp / WINDOW_SIZE * 100)
        st.markdown(f"""
        <div style="background:#08131f;border:1px solid #112036;border-radius:8px;
                    padding:10px 18px;margin-bottom:12px;font-size:.79rem;color:#2c4a6e;
                    display:flex;align-items:center;gap:10px">
          <span class="dot pulse" style="color:#00e5b4">●</span>
          Collecting samples from Arduino:
          <b style="color:#e2e8f0">{n_samp} / {WINDOW_SIZE}</b>
          ({pct}%)  — shake your hand
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # LEFT: alert card + freq bar     RIGHT: live signal chart
    # ─────────────────────────────────────────────────────────
    left, right = st.columns([1.0, 2.0])

    with left:
        # ── alert card ────────────────────────────────────────
        if label == -1:
            css, icon, color = "a-idle",  "⏳", "#2c4a6e"
            title = "Analysing…"
            freq_txt = "Collecting signal…"
            sev_txt  = ""
            conf_txt = ""
        elif label == 1:
            css, icon, color = "a-red", "🚨", "#ef4444"
            title    = "TREMOR DETECTED"
            freq_txt = f"⚡ {freq:.2f} Hz — Parkinson band 3–7 Hz"
            sev_txt  = freq_to_severity(freq)
            conf_txt = f"RF confidence: {conf:.0f}%"
        else:
            css, icon, color = "a-green", "✅", "#22c55e"
            title    = "NO TREMOR"
            freq_txt = f"⚡ {freq:.2f} Hz — Normal range"
            sev_txt  = "Normal hand movement"
            conf_txt = f"RF confidence: {conf:.0f}%"

        st.markdown(f"""
        <div class="acard {css}">
          <div class="ac-icon">{icon}</div>
          <div class="ac-title" style="color:{color}">{title}</div>
          <div class="ac-freq">{freq_txt}</div>
          <div class="ac-sev">{sev_txt}</div>
          <div class="ac-conf">{conf_txt}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── frequency bar ─────────────────────────────────────
        band_pct = min(100.0, freq / 12.0 * 100) if freq > 0 else 0.0
        bar_col  = (
            "#22c55e" if freq < TREMOR_LO   else
            "#eab308" if freq < 4.0         else
            "#f97316" if freq < 6.0         else
            "#ef4444"
        )
        ptag = (
            '<span style="color:#ef4444">⚠ Pathological</span>'
            if in_path and freq > 0 else
            '<span style="color:#22c55e">✓ Normal</span>'
            if freq > 0 else ""
        )
        st.markdown(f"""
        <div class="fbar-card">
          <div class="fbar-lbl">Frequency Position</div>
          <div class="fbar-ticks" style="margin-bottom:5px">
            <span>0 Hz</span>
            <span style="color:#ef4444">3–7 Hz</span>
            <span>12 Hz</span>
          </div>
          <div class="fbar-track">
            <div class="fbar-fill" style="width:{band_pct:.1f}%;background:{bar_col}"></div>
          </div>
          <div class="fbar-val" style="color:{bar_col}">
            {freq:.2f} Hz &nbsp; {ptag}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        if len(hv) < 5:
            st.markdown("""
            <div style="background:#08131f;border:1px solid #112036;border-radius:10px;
                        height:220px;display:flex;align-items:center;justify-content:center;
                        flex-direction:column;gap:8px;color:#2c4a6e;font-size:.82rem">
              <span style="font-size:2rem">📡</span>
              Receiving data from Arduino A0…
            </div>""", unsafe_allow_html=True)
        else:
            n_show = min(len(hv), 200)
            x  = list(range(n_show))
            yv = hv[-n_show:]
            yf = hf[-n_show:]
            pt_c = ["#ef4444" if l == 1 else "#22c55e" for l in hl[-n_show:]]

            fig = go.Figure()

            # raw voltage
            fig.add_trace(go.Scatter(
                x=x, y=yv, mode="lines", name="Voltage (V)",
                line=dict(color="rgba(0,229,180,.45)", width=1.2),
                yaxis="y1",
            ))
            # detected frequency
            fig.add_trace(go.Scatter(
                x=x, y=yf, mode="lines", name="Freq (Hz)",
                line=dict(color="#60a5fa", width=2.2),
                yaxis="y2",
            ))
            # Parkinson band shading
            fig.add_hrect(
                yref="y2", y0=TREMOR_LO, y1=TREMOR_HI,
                fillcolor="rgba(239,68,68,.07)", layer="below", line_width=0,
            )
            # 3 Hz and 7 Hz reference lines
            for y_ref, lbl_txt in [(TREMOR_LO, "3 Hz"), (TREMOR_HI, "7 Hz")]:
                fig.add_hline(
                    y=y_ref, yref="y2",
                    line_dash="dot", line_color="rgba(239,68,68,.5)", line_width=1,
                    annotation_text=lbl_txt, annotation_font_size=9,
                    annotation_font_color="#ef4444",
                    annotation_position="top left",
                )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#08131f",
                font=dict(color="#2c4a6e", size=10),
                margin=dict(l=8, r=8, t=36, b=8),
                height=310,
                title=dict(
                    text="Real-time Arduino A0 Signal  +  Detected Tremor Frequency",
                    font_color="#2c4a6e", font_size=11,
                ),
                legend=dict(
                    orientation="h", y=1.13, x=0,
                    bgcolor="rgba(0,0,0,0)", font_size=10,
                ),
                xaxis=dict(showgrid=True, gridcolor="#0a1628",
                           title="Samples (100 Hz)"),
                yaxis=dict(showgrid=True, gridcolor="#0a1628",
                           title="Voltage (V)", side="left", range=[0, 5]),
                yaxis2=dict(showgrid=False, title="Freq (Hz)",
                            overlaying="y", side="right", range=[0, 12]),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    # ── METRIC TILES ──────────────────────────────────────────
    st.markdown("")
    c1, c2, c3, c4, c5 = st.columns(5)

    def _tile(lbl, val, unit="", sub="", color="#fff"):
        return (f'<div class="tile"><div class="tile-lbl">{lbl}</div>'
                f'<div class="tile-val" style="color:{color}">{val}'
                f'<span class="tile-unit"> {unit}</span></div>'
                f'<div class="tile-sub">{sub}</div></div>')

    freq_c   = "#ef4444" if in_path and freq > 0 else ("#22c55e" if freq > 0 else "#2c4a6e")
    lbl_col  = "#ef4444" if label == 1 else ("#22c55e" if label == 0 else "#2c4a6e")
    lbl_txt  = "TREMOR" if label == 1 else ("NORMAL" if label == 0 else "—")
    ac       = st.session_state.alert_count

    with c1:
        st.markdown(_tile("Voltage (A0)", f"{raw:.3f}", "V",
                          "Arduino A0 pin"), unsafe_allow_html=True)
    with c2:
        st.markdown(_tile("Tremor Freq", f"{freq:.2f}", "Hz",
                          "⚠ Pathological" if in_path and freq>0 else "Normal",
                          freq_c), unsafe_allow_html=True)
    with c3:
        st.markdown(_tile("RF Diagnosis", lbl_txt, "",
                          f"{conf:.0f}% confidence", lbl_col),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(_tile("Tremor Events", str(ac), "",
                          "This session",
                          "#ef4444" if ac > 0 else "#22c55e"),
                    unsafe_allow_html=True)
    with c5:
        st.markdown(_tile("Samples", str(st.session_state.tick), "",
                          f"Session {mm:02d}:{ss:02d}"),
                    unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────
    st.markdown("")
    t1, t2, t3 = st.tabs(["🔬 FFT Spectrum", "📈 History", "📋 Log"])

    with t1:
        if len(hv) < WINDOW_SIZE:
            st.info(f"Need {WINDOW_SIZE} samples — have {len(hv)}. Keep monitoring…")
        else:
            window  = np.array(hv[-WINDOW_SIZE:])
            proc    = process_signal(window)
            fx, mg  = compute_fft(proc)
            vline_x = float(np.clip(freq, 0.01, 49.9)) if freq > 0 else 0.5

            f2 = go.Figure()
            f2.add_trace(go.Scatter(
                x=fx.tolist(), y=mg.tolist(),
                fill="tozeroy", mode="lines",
                line=dict(color="#00e5b4", width=1.5),
                fillcolor="rgba(0,229,180,.07)",
            ))
            f2.add_vrect(
                x0=TREMOR_LO, x1=TREMOR_HI,
                fillcolor="rgba(239,68,68,.10)", layer="below", line_width=0,
                annotation_text="Parkinson's band 3–7 Hz",
                annotation_font_color="#ef4444", annotation_font_size=10,
                annotation_position="top left",
            )
            if freq > 0:
                f2.add_vline(
                    x=vline_x, line_color="#f59e0b",
                    line_dash="dash", line_width=2,
                    annotation_text=f"Peak: {freq:.2f} Hz",
                    annotation_font_color="#f59e0b", annotation_font_size=11,
                )
            f2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#08131f",
                font=dict(color="#2c4a6e", size=10),
                margin=dict(l=8, r=8, t=20, b=8), height=250,
                xaxis=dict(showgrid=True, gridcolor="#0a1628",
                           title="Frequency (Hz)", range=[0, 15]),
                yaxis=dict(showgrid=True, gridcolor="#0a1628",
                           title="FFT Magnitude"),
            )
            st.plotly_chart(f2, use_container_width=True,
                            config={"displayModeBar": False})

            feat_d = extract_features(proc)
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown(_tile("Dominant Freq", f"{freq:.2f}", "Hz",
                                  "Pathological" if in_path else "Normal", freq_c),
                            unsafe_allow_html=True)
            with cb:
                bp = feat_d['band_power']
                st.markdown(_tile("Band Power 3–7 Hz",
                                  f"{bp:.4f}", "V²/Hz",
                                  "Tremor energy",
                                  "#ef4444" if bp > 0.3 else "#22c55e"),
                            unsafe_allow_html=True)
            with cc:
                st.markdown(_tile("PSD Peak",
                                  f"{feat_d['psd_peak']:.4f}", "V²/Hz",
                                  "Max power spectral density", "#60a5fa"),
                            unsafe_allow_html=True)

    with t2:
        if len(hf) < 5:
            st.info("Collect more data…")
        else:
            pt_c2 = ["#ef4444" if l == 1 else "#22c55e" for l in hl]
            fh = go.Figure()
            fh.add_trace(go.Scatter(
                y=hf, mode="lines+markers",
                marker=dict(color=pt_c2, size=5),
                line=dict(color="rgba(96,165,250,.3)", width=1),
                name="Tremor Freq",
            ))
            fh.add_hrect(y0=TREMOR_LO, y1=TREMOR_HI,
                         fillcolor="rgba(239,68,68,.07)",
                         layer="below", line_width=0)
            fh.add_hline(y=TREMOR_LO, line_dash="dot",
                         line_color="rgba(239,68,68,.45)", line_width=1)
            fh.add_hline(y=TREMOR_HI, line_dash="dot",
                         line_color="rgba(239,68,68,.45)", line_width=1)
            fh.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#08131f",
                font=dict(color="#2c4a6e", size=10),
                margin=dict(l=8, r=8, t=20, b=8), height=260,
                xaxis=dict(showgrid=True, gridcolor="#0a1628",
                           title="Reading #"),
                yaxis=dict(showgrid=True, gridcolor="#0a1628",
                           title="Hz", range=[0, 12]),
            )
            st.plotly_chart(fh, use_container_width=True,
                            config={"displayModeBar": False})

            total  = len(hl) or 1
            n_trem = sum(1 for l in hl if l == 1)
            n_norm = total - n_trem
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.markdown(_tile("Tremor Readings", str(n_trem), "",
                                  f"{n_trem/total*100:.0f}% of total", "#ef4444"),
                            unsafe_allow_html=True)
            with bc2:
                st.markdown(_tile("Normal Readings", str(n_norm), "",
                                  f"{n_norm/total*100:.0f}% of total", "#22c55e"),
                            unsafe_allow_html=True)
            with bc3:
                avg_f = float(np.mean(hf)) if hf else 0.0
                st.markdown(_tile("Mean Freq", f"{avg_f:.2f}", "Hz",
                                  "Session average", "#60a5fa"),
                            unsafe_allow_html=True)

    with t3:
        if os.path.exists(LOG_FILE):
            try:
                df_l = (pd.read_csv(LOG_FILE)
                        .tail(60)
                        .sort_index(ascending=False)
                        .reset_index(drop=True))
                st.dataframe(df_l, use_container_width=True, height=300)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total",       len(df_l))
                s2.metric("Avg Freq",    f"{df_l['freq_hz'].mean():.2f} Hz")
                s3.metric("Avg Conf",    f"{df_l['confidence'].mean():.0f}%")
                s4.metric("Tremor Hits", len(df_l[df_l["result"] == "Tremor"]))
            except Exception as e:
                st.warning(f"Log error: {e}")
        else:
            st.info("No log yet. Enable logging in sidebar.")


# ═══════════════════════════════════════════════════════════════════════════
# AUTO REFRESH  —  only when live Arduino data is flowing
# ═══════════════════════════════════════════════════════════════════════════
if mon and r and is_connected:
    time.sleep(refresh_ms / 1000.0)
    st.rerun()
