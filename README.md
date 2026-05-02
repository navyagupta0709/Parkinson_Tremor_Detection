Parkinson_Tremor_Detection


 Parkinson's IoT Wearable Monitoring System
A full-stack Streamlit web application that simulates a real-time IoT wearable monitoring system for Parkinson's tremor detection — based on the research notebook using KNN, SVM, and Random Forest classifiers.

🏗️ System Architecture
⌚ Wearable  ──►  📶 WiFi Gateway  ──►  ☁️ Internet  ──►  🖥️ Web Server  ──►  👤 Dashboard  ──►  🚨 Alerts
  (Glove/Watch)                          (Cloud)          (ML Processing)      (Streamlit)
The app replicates the full IoT pipeline shown in the architecture diagram:

Wearable prototype (smartwatch / sensor glove) collects tremor, heart rate, temperature, SpO₂, and 3-axis accelerometer data
WiFi Gateway transmits data (simulated RSSI, latency)
Web Server runs ML classification (pathological / non-pathological)
User Dashboard shows live charts, alerts, and session stats


🚀 How to Run
bash# 1. Clone / download the project
cd parkinson_iot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
The app opens at http://localhost:8501 in your browser.

✨ Features
📡 Device Control

Toggle wearable ON/OFF from the sidebar
Live WiFi signal strength (dBm) and gateway latency display
Connection status pill (ONLINE / OFFLINE)

📊 Live Sensor Data
SensorDescriptionHeart RateSimulated bpm with severity-based variationTemperatureCore body temperature (°C)Tremor FrequencyDominant FFT frequency (4–7 Hz = pathological)Tremor AmplitudePeak shake intensity (g)SpO₂Blood oxygen saturation (%)Accelerometer3-axis (X/Y/Z) movement
🚨 Smart Alerts
Alerts fire automatically when:

Tremor frequency ≥ 4 Hz (warning) or ≥ 6 Hz (critical)
Heart rate < 50 or > 110 bpm
Temperature outside 35.5–37.8 °C
SpO₂ < 97% (warning) or < 95% (critical)

📈 Monitoring Charts (4 tabs)

Vitals History — rolling time-series for all 4 key metrics
Tremor Waveform — synthesised raw signal with noise and baseline drift
FFT Spectrum — frequency domain with pathological range highlighted
Accelerometer — 3-axis live plot

🤖 ML Classification
Rule-based ensemble mimicking the notebook's Random Forest / SVM / KNN output:

Non-Pathological
Borderline — Monitor
Pathological — Moderate
Pathological — Severe

With confidence score displayed.
💾 Data Logging

Auto-logs every reading to logs/sensor_log.csv
Download full session report as CSV
Clear log button in sidebar

▶️ Live Streaming Mode
Toggle Live Streaming Mode in the sidebar to auto-refresh at 1–10 second intervals (configurable).

📁 Project Structure
parkinson_iot/
│── app.py              # Main Streamlit dashboard
│── utils.py            # Sensor simulation, FFT, anomaly detection, logging
│── requirements.txt    # Python dependencies
│── README.md           # This file
│── logs/
│     └── sensor_log.csv   # Auto-generated sensor log

🔬 Scientific Background
Based on the Parkinson's Tremor Detection research:

Pathological tremor range: 4–7 Hz (resting tremor)
Features: mean, std, RMS, peak-to-peak, dominant FFT frequency, spectral power
Classifiers: KNN, SVM (RBF kernel), Random Forest — achieving >90% accuracy
Sampling frequency: 100 Hz | Window: 200 samples (2 sec) | 50% overlap

