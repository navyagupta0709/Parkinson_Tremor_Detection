"""
Tremor Detection Dashboard
Reads voltage from Arduino (A0, 100Hz) via Serial,
detects tremors using FFT, streams data to a web dashboard.
"""

import threading
import time
import json
import os
import io
import csv
import serial
import serial.tools.list_ports
import numpy as np
from collections import deque
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_file, request
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BAUD_RATE       = 9600
SAMPLE_RATE     = 100          # Hz (matches Arduino 10ms interval)
BUFFER_SECONDS  = 5            # seconds of data for FFT window
BUFFER_SIZE     = SAMPLE_RATE * BUFFER_SECONDS   # 500 samples

TREMOR_FREQ_LOW  = 3.0         # Hz – tremor band start
TREMOR_FREQ_HIGH = 12.0        # Hz – tremor band end
TREMOR_POWER_THRESHOLD = 0.01  # relative band power to flag tremor
TREMOR_AMP_THRESHOLD   = 0.15  # voltage swing to consider significant

# ─────────────────────────────────────────────
# Shared state (thread-safe via locks)
# ─────────────────────────────────────────────
data_lock   = threading.Lock()
voltage_buf = deque(maxlen=BUFFER_SIZE)   # raw voltages
time_buf    = deque(maxlen=BUFFER_SIZE)   # timestamps (seconds from start)
event_log   = []                          # list of tremor events

state = {
    "connected":      False,
    "port":           None,
    "tremor_active":  False,
    "tremor_start":   None,
    "last_voltage":   0.0,
    "last_time":      0.0,
    "dominant_freq":  0.0,
    "band_power":     0.0,
    "total_events":   0,
    "session_start":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

app = Flask(__name__)

# ─────────────────────────────────────────────
# Tremor Detection
# ─────────────────────────────────────────────
def analyze_tremor(voltages):
    """
    Run FFT on the voltage buffer and return:
        tremor_detected (bool), dominant_freq (Hz), band_power (0-1)
    """
    if len(voltages) < SAMPLE_RATE:          # need at least 1 second
        return False, 0.0, 0.0

    arr = np.array(voltages, dtype=float)
    arr -= arr.mean()                         # remove DC offset

    amplitude_range = arr.max() - arr.min()
    if amplitude_range < TREMOR_AMP_THRESHOLD:
        return False, 0.0, 0.0               # signal too quiet / flat

    fft_vals  = np.abs(np.fft.rfft(arr))
    fft_freqs = np.fft.rfftfreq(len(arr), d=1.0 / SAMPLE_RATE)

    total_power = np.sum(fft_vals ** 2)
    if total_power == 0:
        return False, 0.0, 0.0

    # Band power in tremor range
    mask        = (fft_freqs >= TREMOR_FREQ_LOW) & (fft_freqs <= TREMOR_FREQ_HIGH)
    band_power  = np.sum(fft_vals[mask] ** 2) / total_power

    # Dominant frequency (excluding DC)
    dc_mask      = fft_freqs > 0.5
    dom_idx      = np.argmax(fft_vals * dc_mask)
    dominant_freq = fft_freqs[dom_idx]

    tremor = (band_power >= TREMOR_POWER_THRESHOLD) and \
             (TREMOR_FREQ_LOW <= dominant_freq <= TREMOR_FREQ_HIGH)

    return tremor, float(dominant_freq), float(band_power)


# ─────────────────────────────────────────────
# Serial Reader Thread
# ─────────────────────────────────────────────
def find_arduino_port():
    """Auto-detect the first available serial port that looks like Arduino."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ["arduino", "ch340", "usb serial", "cp210"]):
            return p.device
    # Fallback: first available port
    if ports:
        return ports[0].device
    return None


def serial_reader(port_override=None):
    global state
    port = port_override or find_arduino_port()
    if not port:
        print("[Serial] No Arduino port found. Waiting...")
        while True:
            port = find_arduino_port()
            if port:
                break
            time.sleep(2)

    print(f"[Serial] Connecting to {port} @ {BAUD_RATE} baud …")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)                          # let Arduino reset
        ser.flushInput()
        with data_lock:
            state["connected"] = True
            state["port"]      = port
        print(f"[Serial] Connected to {port}")
    except Exception as e:
        print(f"[Serial] Could not open {port}: {e}")
        with data_lock:
            state["connected"] = False
        return

    analysis_counter = 0

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            t_sec   = float(parts[0])
            voltage = float(parts[1])

            with data_lock:
                time_buf.append(t_sec)
                voltage_buf.append(voltage)
                state["last_voltage"] = voltage
                state["last_time"]    = t_sec

            analysis_counter += 1
            if analysis_counter >= 10:          # analyze every 10 samples (100ms)
                analysis_counter = 0
                with data_lock:
                    vbuf_copy = list(voltage_buf)

                detected, dom_freq, band_power = analyze_tremor(vbuf_copy)

                with data_lock:
                    state["dominant_freq"] = round(dom_freq, 2)
                    state["band_power"]    = round(band_power, 4)

                    if detected and not state["tremor_active"]:
                        state["tremor_active"] = True
                        state["tremor_start"]  = datetime.now()
                        state["total_events"] += 1
                        print(f"[ALERT] Tremor detected! freq={dom_freq:.1f}Hz power={band_power:.3f}")

                    elif not detected and state["tremor_active"]:
                        duration = (datetime.now() - state["tremor_start"]).total_seconds()
                        event_log.append({
                            "id":        state["total_events"],
                            "start":     state["tremor_start"].strftime("%H:%M:%S"),
                            "duration":  round(duration, 1),
                            "freq":      round(dom_freq, 2),
                            "power":     round(band_power, 4),
                            "severity":  classify_severity(dom_freq, band_power),
                        })
                        state["tremor_active"] = False
                        state["tremor_start"]  = None
                        print(f"[INFO] Tremor ended. Duration={duration:.1f}s")

        except serial.SerialException as e:
            print(f"[Serial] Disconnected: {e}")
            with data_lock:
                state["connected"]     = False
                state["tremor_active"] = False
            break
        except (ValueError, IndexError):
            continue


def classify_severity(freq, power):
    if power > 0.4:
        return "Severe"
    elif power > 0.2:
        return "Moderate"
    elif power > 0.05:
        return "Mild"
    return "Trace"


# ─────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with data_lock:
        return jsonify({
            "connected":      state["connected"],
            "port":           state["port"],
            "tremor_active":  state["tremor_active"],
            "last_voltage":   round(state["last_voltage"], 4),
            "last_time":      round(state["last_time"], 3),
            "dominant_freq":  state["dominant_freq"],
            "band_power":     state["band_power"],
            "total_events":   state["total_events"],
            "session_start":  state["session_start"],
        })


@app.route("/api/waveform")
def api_waveform():
    """Return the last N samples for live chart rendering."""
    n = int(request.args.get("n", 200))
    with data_lock:
        t_list = list(time_buf)[-n:]
        v_list = list(voltage_buf)[-n:]
    return jsonify({"time": t_list, "voltage": v_list})


@app.route("/api/events")
def api_events():
    with data_lock:
        events = list(event_log)
    return jsonify(events)


@app.route("/api/clear_events", methods=["POST"])
def clear_events():
    with data_lock:
        event_log.clear()
        state["total_events"] = 0
    return jsonify({"status": "cleared"})


@app.route("/api/report")
def download_report():
    """Generate and return a PDF report of the session."""
    with data_lock:
        events   = list(event_log)
        snap     = dict(state)
        vbuf_now = list(voltage_buf)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles  = getSampleStyleSheet()
    content = []

    # Title
    title_style = ParagraphStyle("title", fontSize=22, fontName="Helvetica-Bold",
                                 textColor=colors.HexColor("#1a1a2e"), spaceAfter=6)
    sub_style   = ParagraphStyle("sub",   fontSize=11, fontName="Helvetica",
                                 textColor=colors.HexColor("#555555"), spaceAfter=14)

    content.append(Paragraph("Tremor Detection Report", title_style))
    content.append(Paragraph(
        f"Session started: {snap['session_start']} &nbsp;|&nbsp; "
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sub_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    content.append(Spacer(1, 0.4*cm))

    # Summary box
    summary_data = [
        ["Parameter", "Value"],
        ["Arduino Port",           snap.get("port") or "—"],
        ["Total Tremor Events",    str(snap["total_events"])],
        ["Current Status",         "🔴 TREMOR ACTIVE" if snap["tremor_active"] else "🟢 Normal"],
        ["Dominant Frequency",     f"{snap['dominant_freq']} Hz"],
        ["Band Power (3–12 Hz)",   f"{snap['band_power']:.4f}"],
        ["Sample Rate",            f"{SAMPLE_RATE} Hz"],
        ["FFT Window",             f"{BUFFER_SECONDS} s ({BUFFER_SIZE} samples)"],
    ]
    t = Table(summary_data, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 11),
        ("BACKGROUND",  (0,1), (0,-1), colors.HexColor("#f0f4ff")),
        ("FONTNAME",    (0,1), (0,-1), "Helvetica-Bold"),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.white, colors.HexColor("#f9f9ff")]),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    content.append(t)
    content.append(Spacer(1, 0.6*cm))

    # Events table
    content.append(Paragraph("Tremor Event Log", ParagraphStyle(
        "h2", fontSize=15, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1a2e"), spaceBefore=6, spaceAfter=8)))

    if events:
        ev_data = [["#", "Start Time", "Duration (s)", "Freq (Hz)", "Band Power", "Severity"]]
        for ev in events:
            sev_color = {"Severe": "#ff4444", "Moderate": "#ff8800",
                         "Mild": "#ffcc00", "Trace": "#44aa44"}.get(ev["severity"], "#888888")
            ev_data.append([
                str(ev["id"]), ev["start"],
                str(ev["duration"]), str(ev["freq"]),
                str(ev["power"]), ev["severity"],
            ])
        et = Table(ev_data, colWidths=[1.2*cm, 3*cm, 2.8*cm, 2.8*cm, 3*cm, 3.2*cm])
        et.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#c0392b")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.white, colors.HexColor("#fff5f5")]),
            ("TOPPADDING",  (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ]))
        content.append(et)
    else:
        content.append(Paragraph(
            "No tremor events recorded in this session.",
            ParagraphStyle("note", fontSize=11, textColor=colors.gray)))

    content.append(Spacer(1, 0.8*cm))

    # Notes
    content.append(Paragraph("Clinical Notes", ParagraphStyle(
        "h2", fontSize=15, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1a2e"), spaceBefore=6, spaceAfter=8)))
    notes_text = (
        "This report was generated automatically by the wearable tremor detection system. "
        "Tremor detection is based on spectral analysis of the piezoelectric/analog sensor signal "
        f"in the {TREMOR_FREQ_LOW}–{TREMOR_FREQ_HIGH} Hz band (pathological tremor range). "
        "A relative band power ≥ {:.0f}% with amplitude swing ≥ {:.2f}V is flagged as a tremor event. "
        "This data is intended for clinical review and should not replace professional diagnosis."
    ).format(TREMOR_POWER_THRESHOLD * 100, TREMOR_AMP_THRESHOLD)
    content.append(Paragraph(notes_text, styles["Normal"]))

    doc.build(content)
    buf.seek(0)
    fname = f"tremor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name=fname)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Start serial reader in background thread
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()
    print("[App] Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
