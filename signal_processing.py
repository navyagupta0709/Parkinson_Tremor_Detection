"""
signal_processing.py
====================
Exact signal processing pipeline from Parkinson_Tremor_Detection notebook.

Arduino sketch_feb18a.ino sends:
    "<time_s> <voltage_V>\n"   at 100 Hz, 9600 baud
    e.g.  "1.230 2.4812"

This module:
  1. Pre-processes raw voltage windows (DC removal → detrend → bandpass → smooth)
  2. Extracts 8 features used by the ML model
  3. Computes FFT spectrum for display
"""

import numpy as np
from scipy.fft       import fft, fftfreq
from scipy.signal    import butter, sosfiltfilt, detrend, savgol_filter, welch

# ── constants matching Arduino + notebook ─────────────────────
FS          = 100.0   # sketch_feb18a: samplingInterval = 10ms → 100 Hz
WINDOW_SIZE = 200     # 2-second analysis window
STEP_SIZE   = 100     # 50 % overlap

# Parkinson's resting tremor band (ICD-10 clinical definition)
TREMOR_LO = 3.0       # Hz
TREMOR_HI = 7.0       # Hz

# 8 feature names (must match train_model.py FEAT_COLS)
FEAT_COLS = [
    'mean', 'std', 'rms', 'energy',
    'dom_freq', 'sp_entropy', 'psd_peak', 'band_power',
]


# ──────────────────────────────────────────────────────────────
# 1.  Signal Pre-processing  (notebook Cell 7)
# ──────────────────────────────────────────────────────────────
def process_signal(sig: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Full SOP pipeline:
      1. DC offset removal  (subtract mean)
      2. Detrend            (remove slow baseline drift)
      3. Butterworth bandpass  0.5 – 10 Hz, 4th order
      4. Savitzky-Golay smoothing  (window=11, poly=3)
    """
    s = sig.astype(float).copy()
    if len(s) < 20:
        return s
    s -= np.mean(s)                                              # DC removal
    s  = detrend(s)                                              # detrend
    sos = butter(4, [0.5, 10.0], btype='bandpass',
                 fs=fs, output='sos')
    s   = sosfiltfilt(sos, s)                                    # bandpass
    s   = savgol_filter(s, window_length=11, polyorder=3)        # smooth
    return s


# ──────────────────────────────────────────────────────────────
# 2.  Feature Extraction  (notebook Cell 9)
# ──────────────────────────────────────────────────────────────
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """numpy-version-safe trapezoidal integration."""
    fn = getattr(np, 'trapezoid',
                 getattr(np, 'trapz',
                         lambda y, x: float(np.sum((y[:-1]+y[1:]) * np.diff(x) / 2))))
    return float(fn(y, x))


def extract_features(window: np.ndarray, fs: float = FS) -> dict:
    """
    Extracts the 8 features used by the RF / SVM / KNN models.
    Window should already be processed by process_signal().
    """
    n = len(window)

    # time-domain
    mean   = float(np.mean(window))
    std    = float(np.std(window, ddof=1))
    rms    = float(np.sqrt(np.mean(window ** 2)))
    energy = float(np.sum(window ** 2) / n)

    # FFT — dominant frequency
    freqs    = fftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(fft(window))
    pos      = freqs > 0
    fp, fv   = freqs[pos], fft_vals[pos]
    dom_freq = float(fp[np.argmax(fv)]) if len(fp) else 0.0

    # spectral entropy
    psd_norm   = fv / (fv.sum() + 1e-12)
    sp_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    # Welch PSD
    f_w, psd  = welch(window, fs=fs, nperseg=min(n, 64))
    psd_peak  = float(np.max(psd))

    # band power  3 – 7 Hz
    band     = (f_w >= TREMOR_LO) & (f_w <= TREMOR_HI)
    band_pwr = _trapz(psd[band], f_w[band]) if band.any() else 0.0

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
    """Convert feature dict → numpy array in FEAT_COLS order."""
    return np.array([feat[c] for c in FEAT_COLS])


# ──────────────────────────────────────────────────────────────
# 3.  FFT Spectrum  (notebook Cell 11)
# ──────────────────────────────────────────────────────────────
def compute_fft(sig: np.ndarray, fs: float = FS):
    """Returns (positive_freqs, magnitudes) for plotting."""
    n        = len(sig)
    fft_vals = np.abs(fft(sig))
    freqs    = fftfreq(n, d=1.0 / fs)
    pos      = freqs > 0
    return freqs[pos], fft_vals[pos]


def dominant_freq(sig: np.ndarray, fs: float = FS) -> float:
    fp, fv = compute_fft(sig, fs)
    return float(fp[np.argmax(fv)]) if len(fp) else 0.0


# ──────────────────────────────────────────────────────────────
# 4.  Clinical helpers
# ──────────────────────────────────────────────────────────────
def freq_to_severity(freq_hz: float) -> str:
    """Map dominant frequency → clinical severity label."""
    if freq_hz < TREMOR_LO:
        return "Non-Tremor"
    elif freq_hz < 4.0:
        return "Mild Tremor"
    elif freq_hz < 6.0:
        return "Moderate Tremor"
    else:
        return "Severe Tremor"
