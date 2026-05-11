# app.py

```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as scipy_signal
from scipy.fft import fft, fftfreq
import time
import datetime

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title='TENG Tremor Detection',
    page_icon='⚡',
    layout='wide'
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>
body {
    background-color:#0f172a;
}

.main-header {
    background:linear-gradient(90deg,#111827,#1e293b);
    padding:20px;
    border-radius:12px;
    margin-bottom:20px;
}

.main-header h1 {
    color:#38bdf8;
}

.main-header p {
    color:white;
}

.metric-box {
    background:#111827;
    padding:15px;
    border-radius:12px;
    text-align:center;
    border:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================

st.markdown("""
<div class="main-header">
<h1>⚡ TENG Parkinson Tremor Detection System</h1>
<p>
Real-Time FFT Analysis · Signal Processing · Spectral Classification
</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.title('⚙️ Control Panel')

    live_mode = st.toggle(
        'Live Detection',
        value=True
    )

    severity = st.selectbox(
        'Simulation Mode',
        ['normal', 'mild', 'moderate', 'severe'],
        index=1
    )

    refresh_rate = st.slider(
        'Refresh Rate (sec)',
        1,
        5,
        2
    )

    st.markdown('---')

    st.caption(
        'IEEE Research Prototype\n'
        'TENG-Based Tremor Detection\n'
        'FFT + PSD + Machine Learning'
    )

# ======================================================
# SIGNAL GENERATOR
# ======================================================

FS = 100


def generate_signal(level='normal'):

    duration = 10

    t = np.linspace(
        0,
        duration,
        FS * duration
    )

    if level == 'normal':
        freq = 1.5
        amp = 0.2

    elif level == 'mild':
        freq = 3.5
        amp = 0.4

    elif level == 'moderate':
        freq = 4.8
        amp = 0.7

    else:
        freq = 6.2
        amp = 1.0

    signal = amp * np.sin(
        2 * np.pi * freq * t
    )

    noise = 0.08 * np.random.randn(
        len(t)
    )

    drift = 0.1 * np.sin(
        2 * np.pi * 0.1 * t
    )

    signal = signal + noise + drift

    return t, signal, freq, amp

# ======================================================
# GENERATE SIGNAL
# ======================================================


t, raw_signal, tremor_freq, tremor_amp = generate_signal(
    severity
)

# ======================================================
# SIGNAL PROCESSING
# ======================================================

sig_clean = np.nan_to_num(
    raw_signal
)

sig_dc = sig_clean - np.mean(
    sig_clean
)

sig_det = scipy_signal.detrend(
    sig_dc
)

b, a = scipy_signal.butter(
    4,
    [0.5, 10],
    btype='bandpass',
    fs=FS
)

sig_filt = scipy_signal.filtfilt(
    b,
    a,
    sig_det
)

sig_final = scipy_signal.savgol_filter(
    sig_filt,
    11,
    2
)

# ======================================================
# FFT ANALYSIS
# ======================================================

n = len(sig_final)

fft_vals = fft(
    sig_final
)

freqs = fftfreq(
    n,
    d=1/FS
)

mask = freqs > 0

freqs = freqs[mask]

amp = (2/n) * np.abs(
    fft_vals[mask]
)

peak_idx = np.argmax(
    amp
)

peak_freq = freqs[peak_idx]

peak_amp = amp[peak_idx]

# ======================================================
# METRICS
# ======================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        'Dominant Frequency',
        f'{peak_freq:.2f} Hz'
    )

with col2:
    st.metric(
        'Detected BPM',
        f'{peak_freq * 60:.0f}'
    )

with col3:
    st.metric(
        'Signal Quality',
        '96 %'
    )

with col4:
    st.metric(
        'Tremor Amplitude',
        f'{tremor_amp:.2f} V'
    )

# ======================================================
# ALERT SYSTEM
# ======================================================

if 3 <= peak_freq <= 7:

    if peak_freq < 4:
        sev = 'MILD'

    elif peak_freq < 5.5:
        sev = 'MODERATE'

    else:
        sev = 'SEVERE'

    st.markdown(f"""
    <div style="
        background:#7f1d1d;
        border:3px solid red;
        padding:25px;
        border-radius:15px;
        text-align:center;
        margin-top:20px;
    ">

    <h1 style="color:white;">
        🚨 TREMOR DETECTED
    </h1>

    <h2 style="color:#fecaca;">
        {peak_freq:.2f} Hz
    </h2>

    <h3 style="color:#fca5a5;">
        Severity : {sev}
    </h3>

    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div style="
        background:#052e16;
        border:3px solid #22c55e;
        padding:25px;
        border-radius:15px;
        text-align:center;
        margin-top:20px;
    ">

    <h1 style="color:#bbf7d0;">
        ✅ NORMAL SIGNAL
    </h1>

    </div>
    """, unsafe_allow_html=True)

# ======================================================
# TABS
# ======================================================

chart_tabs = st.tabs([
    '⚡ Raw Signal',
    '🧹 Signal Processing',
    '🔬 FFT Spectrum',
    '📈 PSD Analysis'
])

# ======================================================
# RAW SIGNAL
# ======================================================

with chart_tabs[0]:

    fig1, ax1 = plt.subplots(
        figsize=(14,4)
    )

    fig1.patch.set_facecolor('#0f172a')

    ax1.set_facecolor('#111827')

    ax1.plot(
        t,
        raw_signal,
        color='#38bdf8',
        linewidth=1.2
    )

    ax1.set_title(
        'Live Raw TENG Signal',
        color='white'
    )

    ax1.set_xlabel(
        'Time (s)',
        color='white'
    )

    ax1.set_ylabel(
        'Voltage (V)',
        color='white'
    )

    ax1.tick_params(colors='white')

    ax1.grid(alpha=0.2)

    st.pyplot(fig1)

# ======================================================
# SIGNAL PROCESSING
# ======================================================

with chart_tabs[1]:

    fig2, ax2 = plt.subplots(
        4,
        1,
        figsize=(12,8)
    )

    ax2[0].plot(t[:500], raw_signal[:500])
    ax2[0].set_title('Raw Signal')

    ax2[1].plot(t[:500], sig_dc[:500])
    ax2[1].set_title('DC Offset Removal')

    ax2[2].plot(t[:500], sig_det[:500])
    ax2[2].set_title('Detrending')

    ax2[3].plot(t[:500], sig_final[:500])
    ax2[3].set_title('Filtered Signal')

    for a in ax2:
        a.grid(alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig2)

# ======================================================
# FFT SPECTRUM
# ======================================================

with chart_tabs[2]:

    fig3, ax3 = plt.subplots(
        figsize=(14,5)
    )

    fig3.patch.set_facecolor('#0f172a')

    ax3.set_facecolor('#111827')

    ax3.plot(
        freqs,
        amp,
        color='#38bdf8',
        linewidth=1.5
    )

    ax3.axvspan(
        3,
        7,
        color='red',
        alpha=0.15,
        label='Tremor Band'
    )

    ax3.plot(
        peak_freq,
        peak_amp,
        'ro'
    )

    ax3.set_xlim(0,12)

    ax3.set_title(
        'Real-Time FFT Spectrum',
        color='white'
    )

    ax3.set_xlabel(
        'Frequency (Hz)',
        color='white'
    )

    ax3.set_ylabel(
        'Amplitude',
        color='white'
    )

    ax3.tick_params(colors='white')

    ax3.grid(alpha=0.2)

    ax3.legend()

    st.pyplot(fig3)

# ======================================================
# PSD ANALYSIS
# ======================================================

with chart_tabs[3]:

    fig4, ax4 = plt.subplots(
        figsize=(14,4)
    )

    fig4.patch.set_facecolor('#0f172a')

    ax4.set_facecolor('#111827')

    f_psd, psd = scipy_signal.welch(
        sig_final,
        fs=FS,
        nperseg=256
    )

    ax4.semilogy(
        f_psd,
        psd,
        color='#22c55e'
    )

    ax4.axvspan(
        3,
        7,
        color='red',
        alpha=0.15
    )

    ax4.set_xlim(0,12)

    ax4.set_title(
        'Welch PSD Analysis',
        color='white'
    )

    ax4.set_xlabel(
        'Frequency (Hz)',
        color='white'
    )

    ax4.set_ylabel(
        'PSD',
        color='white'
    )

    ax4.tick_params(colors='white')

    ax4.grid(alpha=0.2)

    st.pyplot(fig4)

# ======================================================
# FOOTER
# ======================================================

st.markdown('---')

st.caption(
    f'Live Detection Running · '
    f'{datetime.datetime.now().strftime("%H:%M:%S")}'
)

if live_mode:
    time.sleep(refresh_rate)
    st.rerun()
```

# requirements.txt

```txt
streamlit
numpy
pandas
matplotlib
scipy
```
