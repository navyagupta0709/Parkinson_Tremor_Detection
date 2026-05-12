"""
app.py — Deploy this to Streamlit Cloud
Subscribes to MQTT broker → shows real-time tremor dashboard
"""

import time
import json
import io
from datetime import datetime
from collections import deque

import streamlit as st
import paho.mqtt.client as mqtt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="TremorWatch",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── MQTT config — must match sender.py ───────────────────────────
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT   = 1883
MQTT_TOPIC  = "tremorwatch/data/navya"   # ← same unique string in sender.py

# ── Session state init ───────────────────────────────────────────
def _init():
    defaults = {
        "connected":     False,
        "tremor_active": False,
        "last_voltage":  0.0,
        "last_time":     0.0,
        "dominant_freq": 0.0,
        "band_power":    0.0,
        "severity":      "Normal",
        "total_events":  0,
        "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "voltages":      deque(maxlen=300),
        "times":         deque(maxlen=300),
        "events":        [],
        "mqtt_client":   None,
        "mqtt_started":  False,
        "_tremor_start": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── MQTT callbacks ───────────────────────────────────────────────
def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        client.subscribe(MQTT_TOPIC, qos=0)
        st.session_state["connected"] = True

def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    st.session_state["connected"] = False

def on_message(client, userdata, msg):
    try:
        d = json.loads(msg.payload.decode())
    except Exception:
        return

    ss = st.session_state
    ss["last_voltage"]  = d.get("v", 0.0)
    ss["last_time"]     = d.get("t", 0.0)
    ss["dominant_freq"] = d.get("freq", 0.0)
    ss["band_power"]    = d.get("power", 0.0)
    ss["severity"]      = d.get("severity", "Normal")
    ss["voltages"].append(d.get("v", 0.0))
    ss["times"].append(d.get("t", 0.0))
    ss["connected"]     = True

    tremor_now = d.get("tremor", False)
    if tremor_now and not ss["tremor_active"]:
        ss["tremor_active"] = True
        ss["_tremor_start"] = datetime.now()
        ss["total_events"] += 1
    elif not tremor_now and ss["tremor_active"]:
        ss["tremor_active"] = False
        if ss["_tremor_start"]:
            dur = (datetime.now() - ss["_tremor_start"]).total_seconds()
            ss["events"].append({
                "id":       ss["total_events"],
                "start":    ss["_tremor_start"].strftime("%H:%M:%S"),
                "duration": round(dur, 1),
                "freq":     ss["dominant_freq"],
                "power":    ss["band_power"],
                "severity": ss["severity"],
            })
            ss["_tremor_start"] = None

def start_mqtt():
    if st.session_state["mqtt_started"]:
        return
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message
    client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()
    st.session_state["mqtt_client"]  = client
    st.session_state["mqtt_started"] = True

start_mqtt()

# ── PDF report ───────────────────────────────────────────────────
def generate_pdf():
    ss  = st.session_state
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    content = []
    title_s = ParagraphStyle("t", fontSize=22, fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#1a1a2e"), spaceAfter=6)
    sub_s   = ParagraphStyle("s", fontSize=10, fontName="Helvetica",
                              textColor=colors.HexColor("#555"), spaceAfter=14)
    content.append(Paragraph("TremorWatch — Session Report", title_s))
    content.append(Paragraph(
        f"Session: {ss['session_start']}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sub_s))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#ccc")))
    content.append(Spacer(1, 0.4*cm))

    summary = [
        ["Parameter", "Value"],
        ["Total Tremor Events", str(ss["total_events"])],
        ["Current Status", "TREMOR ACTIVE" if ss["tremor_active"] else "Normal"],
        ["Dominant Frequency", f"{ss['dominant_freq']} Hz"],
        ["Band Power (3–12Hz)", f"{ss['band_power']:.4f}"],
        ["Last Voltage", f"{ss['last_voltage']:.4f} V"],
    ]
    t = Table(summary, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",   (0,0),(-1,0), colors.white),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("BACKGROUND",  (0,1),(0,-1), colors.HexColor("#f0f4ff")),
        ("FONTNAME",    (0,1),(0,-1), "Helvetica-Bold"),
        ("GRID",        (0,0),(-1,-1), 0.5, colors.HexColor("#ddd")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f9f9ff")]),
        ("TOPPADDING",  (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))
    content.append(t)
    content.append(Spacer(1, 0.5*cm))
    content.append(Paragraph("Tremor Event Log", ParagraphStyle(
        "h2", fontSize=14, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1a2e"), spaceBefore=6, spaceAfter=8)))

    events = ss["events"]
    if events:
        rows = [["#","Start","Duration (s)","Freq (Hz)","Power","Severity"]]
        for ev in events:
            rows.append([str(ev["id"]), ev["start"], str(ev["duration"]),
                         str(ev["freq"]), str(ev["power"]), ev["severity"]])
        et = Table(rows, colWidths=[1*cm,3*cm,3*cm,3*cm,3*cm,3*cm])
        et.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#c0392b")),
            ("TEXTCOLOR", (0,0),(-1,0), colors.white),
            ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",  (0,0),(-1,-1), 9),
            ("GRID",      (0,0),(-1,-1), 0.5, colors.HexColor("#ddd")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#fff5f5")]),
            ("ALIGN",     (0,0),(-1,-1), "CENTER"),
            ("TOPPADDING",(0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ]))
        content.append(et)
    else:
        styles = getSampleStyleSheet()
        content.append(Paragraph("No tremor events recorded.", styles["Normal"]))

    doc.build(content)
    buf.seek(0)
    return buf.read()

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
.stApp { background: #0b0d14 !important; }
.block-container { padding-top: 1.5rem !important; }
.alert-box {
  background: linear-gradient(135deg, #ff2020, #cc0000);
  color: white; padding: 20px 28px; border-radius: 14px;
  font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800;
  text-align: center; letter-spacing: 4px;
  animation: pulse 0.8s ease-in-out infinite alternate;
  margin-bottom: 20px; box-shadow: 0 0 40px rgba(255,32,32,0.5);
}
@keyframes pulse { from { opacity:1; transform:scale(1); } to { opacity:.7; transform:scale(1.01); } }
.normal-box {
  background: #0d2b1a; border: 2px solid #22d47a; color: #22d47a;
  padding: 14px 24px; border-radius: 12px; font-family: 'Syne', sans-serif;
  font-size: 1.1rem; font-weight: 700; text-align: center; margin-bottom: 20px;
}
.metric-card {
  background: #12151f; border: 1px solid #252a3d; border-radius: 14px;
  padding: 18px 20px; text-align: center; height: 110px;
  display: flex; flex-direction: column; justify-content: center;
}
.metric-label { font-size: 0.68rem; color: #6b7594; text-transform: uppercase;
  letter-spacing: 1.5px; font-weight: 600; margin-bottom: 6px; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.9rem;
  font-weight: 700; line-height: 1; }
.metric-unit  { font-size: 0.72rem; color: #6b7594; margin-top: 4px; }
.ev-row { background: #1a1e2e; border: 1px solid #252a3d; border-radius: 10px;
  padding: 10px 16px; margin-bottom: 7px; font-size: 0.84rem;
  display: flex; gap: 16px; align-items: center; }
.sev-Severe   { color:#ff3b3b; font-weight:700; }
.sev-Moderate { color:#ff8c00; font-weight:700; }
.sev-Mild     { color:#ffdd00; font-weight:700; }
.sev-Trace    { color:#22d47a; font-weight:700; }
.logo-text { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
  color:#e2e8f8; letter-spacing:-0.5px; }
.logo-text span { color:#4f8fff; }
</style>
""", unsafe_allow_html=True)

ss = st.session_state

# ── Header ───────────────────────────────────────────────────────
h1, h2, h3 = st.columns([3, 2, 2])
with h1:
    st.markdown('<div class="logo-text">Tremor<span>Watch</span> 🫀</div>',
                unsafe_allow_html=True)
    st.caption(f"Session started: {ss['session_start']}")
with h2:
    if ss["connected"]:
        st.success("🟢 Live — Arduino sending data")
    else:
        st.warning("🟡 Waiting for sender.py on your PC…")
with h3:
    pdf_bytes = generate_pdf()
    st.download_button(
        "⬇ Download PDF Report",
        data=pdf_bytes,
        file_name=f"tremor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

st.divider()

# ── Alert banner ─────────────────────────────────────────────────
if ss["tremor_active"]:
    st.markdown('<div class="alert-box">⚠ &nbsp; TREMOR DETECTED — RED ALERT &nbsp; ⚠</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="normal-box">✅ &nbsp; Signal Normal — No Tremor Detected</div>',
                unsafe_allow_html=True)

# ── Metric cards ─────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, "Voltage", f"{ss['last_voltage']:.3f}", "Volts",   "#4f8fff"),
    (c2, "Status",  "TREMOR" if ss["tremor_active"] else "Normal",
                    "",  "#ff3b3b" if ss["tremor_active"] else "#22d47a"),
    (c3, "Dom. Freq", f"{ss['dominant_freq']:.2f}", "Hz",   "#ff8c00"),
    (c4, "Band Power", f"{ss['band_power']:.4f}", "3–12 Hz","#7c5cff"),
    (c5, "Events",  str(ss["total_events"]), "this session", "#ff3b3b"),
]
for col, label, val, unit, color in cards:
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{color}">{val}</div>'
            f'<div class="metric-unit">{unit}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("&nbsp;", unsafe_allow_html=True)

# ── Waveform ─────────────────────────────────────────────────────
st.markdown("##### 📈 Live Voltage Waveform")
chart_slot = st.empty()
voltages = list(ss["voltages"])
times    = list(ss["times"])
if voltages:
    import pandas as pd
    df = pd.DataFrame({"Time (s)": times, "Voltage (V)": voltages})
    chart_slot.line_chart(df.set_index("Time (s)"), height=240, use_container_width=True)
else:
    chart_slot.info("⏳ No data yet — start sender.py on your PC with Arduino connected.")

# ── Band power bar ────────────────────────────────────────────────
st.markdown("##### 🔊 Tremor Band Power")
bp_pct = min(1.0, ss["band_power"] / 0.4)
st.progress(bp_pct,
            text=f"Band power: {ss['band_power']*100:.2f}%  |  "
                 f"Dominant: {ss['dominant_freq']:.2f} Hz  |  "
                 f"Severity: {ss['severity']}")

# ── Event log ────────────────────────────────────────────────────
ev_col, clr_col = st.columns([6, 1])
with ev_col:
    st.markdown("##### 📋 Tremor Event Log")
with clr_col:
    if st.button("🗑 Clear"):
        ss["events"]       = []
        ss["total_events"] = 0

events = ss["events"]
if events:
    for ev in reversed(events):
        sev = ev["severity"]
        st.markdown(
            f'<div class="ev-row">'
            f'<span style="color:#6b7594;width:30px">#{ev["id"]}</span>'
            f'<span style="color:#e2e8f8">{ev["start"]}</span>'
            f'<span style="color:#aaa">{ev["duration"]}s</span>'
            f'<span style="color:#aaa">{ev["freq"]} Hz</span>'
            f'<span style="color:#aaa">power {ev["power"]}</span>'
            f'<span class="sev-{sev}">{sev}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown('<p style="color:#6b7594;padding:16px 0">No tremor events recorded yet.</p>',
                unsafe_allow_html=True)

# ── Auto-refresh every 300 ms ─────────────────────────────────────
time.sleep(0.3)
st.rerun()
