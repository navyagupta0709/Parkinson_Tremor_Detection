"""
signal_processing.py
Exact pipeline from Parkinson_Tremor_Detection notebook (Cells 5,7,9,11)
FS=100 Hz, WINDOW=200 samples, Butterworth bandpass 0.5–10 Hz
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfiltfilt, detrend, savgol_filter, welch

FS          = 100.0    # Arduino: 10 ms interval → 100 Hz
WINDOW_SIZE = 200      # 2-second window
STEP_SIZE   = 100      # 50% overlap
FEAT_COLS   = ['mean','std','rms','energy','dom_freq','sp_entropy','psd_peak','band_power']

# Parkinson's resting tremor: 3–7 Hz
TREMOR_LO = 3.0
TREMOR_HI = 7.0


# ─────────────────────────────────────────────
# Cell 5: Outlier removal
# ─────────────────────────────────────────────
def remove_outliers(sig: np.ndarray) -> np.ndarray:
    q1, q3 = np.percentile(sig, [25, 75])
    iqr     = q3 - q1
    return sig[(sig >= q1 - 3*iqr) & (sig <= q3 + 3*iqr)]


# ─────────────────────────────────────────────
# Cell 7: Signal processing pipeline (SOP)
# ─────────────────────────────────────────────
def process_signal(sig: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    1. DC offset removal
    2. Detrend
    3. 4th-order Butterworth bandpass 0.5–10 Hz
    4. Savitzky-Golay smoothing (win=11, poly=3)
    """
    s = sig.astype(float).copy()
    if len(s) < 20:
        return s
    s -= np.mean(s)
    s  = detrend(s)
    try:
        sos = butter(4, [0.5, 10.0], btype='bandpass', fs=fs, output='sos')
        s   = sosfiltfilt(sos, s)
        s   = savgol_filter(s, window_length=11, polyorder=3)
    except Exception:
        pass
    return s


# ─────────────────────────────────────────────
# Cell 9: Feature extraction (exact 8 features)
# ─────────────────────────────────────────────
def extract_features(window: np.ndarray, fs: float = FS) -> dict:
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None)) or (lambda y,x: np.sum((y[:-1]+y[1:])*np.diff(x)/2))
    n      = len(window)

    mean    = float(np.mean(window))
    std     = float(np.std(window, ddof=1))
    rms     = float(np.sqrt(np.mean(window**2)))
    energy  = float(np.sum(window**2) / n)

    freqs     = fftfreq(n, d=1.0/fs)
    fft_vals  = np.abs(fft(window))
    pos       = freqs > 0
    fp, fv    = freqs[pos], fft_vals[pos]
    dom_freq  = float(fp[np.argmax(fv)]) if len(fp) else 0.0

    psd_norm   = fv / (fv.sum() + 1e-12)
    sp_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    f_w, psd  = welch(window, fs=fs, nperseg=min(n, 64))
    psd_peak  = float(np.max(psd))
    band      = (f_w >= TREMOR_LO) & (f_w <= TREMOR_HI)
    band_pwr  = float(_trapz(psd[band], f_w[band])) if band.any() else 0.0

    return {
        'mean':       mean,
        'std':        std,
        'rms':        rms,
        'energy':     energy,
        'dom_freq':   dom_freq,
        'sp_entropy': sp_entropy,
        'psd_peak':   psd_peak,
        'band_power': band_pwr,
    }


def features_to_vec(feat: dict) -> np.ndarray:
    return np.array([feat[c] for c in FEAT_COLS])


# ─────────────────────────────────────────────
# Cell 11: FFT spectrum
# ─────────────────────────────────────────────
def compute_fft(sig: np.ndarray, fs: float = FS):
    n        = len(sig)
    fft_vals = np.abs(fft(sig))
    freqs    = fftfreq(n, d=1.0/fs)
    pos      = freqs > 0
    return freqs[pos], fft_vals[pos]


def dominant_freq(sig: np.ndarray, fs: float = FS) -> float:
    fp, fv = compute_fft(sig, fs)
    return float(fp[np.argmax(fv)]) if len(fp) else 0.0


# ─────────────────────────────────────────────
# Binary classification helper
# label2: 0=Non-Tremor (<3 Hz), 1=Tremor (≥3 Hz)
# ─────────────────────────────────────────────
def freq_to_binary(freq_hz: float) -> int:
    return 1 if freq_hz >= TREMOR_LO else 0


def freq_to_severity(freq_hz: float) -> str:
    if freq_hz < TREMOR_LO:
        return "Non-Tremor"
    elif freq_hz < 4.0:
        return "Mild Tremor"
    elif freq_hz < 6.0:
        return "Moderate Tremor"
    else:
        return "Severe Tremor"
