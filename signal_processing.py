
import numpy as np
from scipy.signal import butter, sosfiltfilt, savgol_filter, welch, find_peaks

# ── Constants ────────────────────────────────────────────────────────────────
FS               = 100.0          # Sampling frequency (Hz) — matches Arduino
TREMOR_BAND_LOW  = 3.0            # Parkinson tremor band start (Hz)
TREMOR_BAND_HIGH = 7.0            # Parkinson tremor band end   (Hz)
WINDOW_SIZE      = 200            # 2-second window @ 100 Hz
STEP_SIZE        = 100            # 50 % overlap


# ── Butterworth Bandpass Filter ───────────────────────────────────────────────
def butter_bandpass(lowcut: float, highcut: float,
                    fs: float = FS, order: int = 4):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq],
                 btype="band", output="sos")
    return sos


def bandpass_filter(signal: np.ndarray,
                    lowcut: float = 0.5,
                    highcut: float = 12.0,
                    fs: float     = FS) -> np.ndarray:
    """Remove DC drift and high-frequency noise."""
    if len(signal) < 15:          # need ≥ 3× filter order samples
        return signal.copy()
    sos = butter_bandpass(lowcut, highcut, fs)
    return sosfiltfilt(sos, signal)


# ── Savitzky-Golay Smoothing ──────────────────────────────────────────────────
def smooth_signal(signal: np.ndarray,
                  window_length: int = 11,
                  polyorder: int     = 3) -> np.ndarray:
    if len(signal) < window_length:
        return signal.copy()
    return savgol_filter(signal, window_length, polyorder)


# ── Dominant Frequency via Welch PSD ─────────────────────────────────────────
def estimate_frequency(signal: np.ndarray, fs: float = FS) -> float:
    """
    Returns dominant frequency in the 1–12 Hz range.
    Uses Welch method (same as notebook).
    """
    if len(signal) < 32:
        return 0.0
    nperseg = min(len(signal), 256)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    # Focus on physiologically meaningful tremor range
    mask = (freqs >= 1.0) & (freqs <= 12.0)
    if not np.any(mask):
        return 0.0
    dominant = freqs[mask][np.argmax(psd[mask])]
    return float(dominant)


# ── Tremor Band Power Ratio ───────────────────────────────────────────────────
def tremor_band_power(signal: np.ndarray, fs: float = FS) -> float:
    """
    Returns ratio of power in 3–7 Hz band to total 1–12 Hz power.
    Range: 0.0 → 1.0
    """
    if len(signal) < 32:
        return 0.0
    nperseg = min(len(signal), 256)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    total_mask  = (freqs >= 1.0) & (freqs <= 12.0)
    tremor_mask = (freqs >= TREMOR_BAND_LOW) & (freqs <= TREMOR_BAND_HIGH)

    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    total_power  = _trapz(psd[total_mask],  freqs[total_mask])
    tremor_power = _trapz(psd[tremor_mask], freqs[tremor_mask])

    if total_power < 1e-12:
        return 0.0
    return float(np.clip(tremor_power / total_power, 0.0, 1.0))


# ── Amplitude Metrics ─────────────────────────────────────────────────────────
def compute_amplitude(signal: np.ndarray) -> float:
    """Peak-to-peak amplitude of the filtered signal."""
    if len(signal) < 2:
        return 0.0
    return float(np.ptp(signal))


def compute_rms(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


# ── Peak Analysis ─────────────────────────────────────────────────────────────
def count_peaks(signal: np.ndarray, height_factor: float = 0.5) -> int:
    if len(signal) < 3:
        return 0
    threshold = np.mean(signal) + height_factor * np.std(signal)
    peaks, _ = find_peaks(signal, height=threshold, distance=int(FS / 8))
    return len(peaks)


# ── Signal Quality Index (0-100) ──────────────────────────────────────────────
def signal_quality(signal: np.ndarray) -> float:
    """
    Heuristic quality score based on:
    - SNR proxy (signal variance vs noise floor)
    - Clipping detection (values at ADC rails)
    - Length adequacy
    """
    if len(signal) < 10:
        return 0.0

    # Clipping penalty
    clip_ratio = np.mean((signal <= 0.05) | (signal >= 4.95))
    clip_penalty = clip_ratio * 50.0

    # Variance-based score
    std = np.std(signal)
    if std < 1e-6:          # flat/dead signal
        return max(0.0, 10.0 - clip_penalty)
    snr_score = min(50.0, std * 25.0)

    # Length score
    len_score = min(50.0, len(signal) / WINDOW_SIZE * 50.0)

    quality = snr_score + len_score - clip_penalty
    return float(np.clip(quality, 0.0, 100.0))


# ── Full Feature Vector (matches notebook feature extraction) ─────────────────
def extract_features(raw_window: np.ndarray, fs: float = FS) -> dict:
    """
    Extract the same feature set used in model training.
    Returns a dict with all features.
    """
    # 1. Bandpass filter
    filtered = bandpass_filter(raw_window)
    smoothed = smooth_signal(filtered)

    # 2. Time-domain features
    mean_val  = float(np.mean(smoothed))
    std_val   = float(np.std(smoothed))
    rms_val   = compute_rms(smoothed)
    amp_val   = compute_amplitude(smoothed)
    peaks     = count_peaks(smoothed)

    # 3. Frequency-domain features
    dom_freq    = estimate_frequency(smoothed, fs)
    band_power  = tremor_band_power(smoothed, fs)

    # 4. Quality
    sq          = signal_quality(raw_window)

    return {
        "mean":         mean_val,
        "std":          std_val,
        "rms":          rms_val,
        "amplitude":    amp_val,
        "peak_count":   float(peaks),
        "dom_freq_hz":  dom_freq,
        "band_power":   band_power,
        "signal_quality": sq,
        # processed arrays (not fed to ML but used for plotting)
        "_filtered":    filtered,
        "_smoothed":    smoothed,
    }
