"""Generic per-pass signal features (used by baselines B2/B3 only)."""
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch


def channel_features(x, fs, f_tooth, name):
    x = np.asarray(x, dtype=float)
    f = {}
    f[f"f_{name}_mean"] = np.mean(x)
    f[f"f_{name}_std"] = np.std(x)
    f[f"f_{name}_rms"] = np.sqrt(np.mean(x ** 2))
    f[f"f_{name}_skew"] = skew(x)
    f[f"f_{name}_kurt"] = kurtosis(x)
    f[f"f_{name}_p2p"] = np.ptp(x)
    nper = int(min(len(x), 4096))
    fr, pxx = welch(x - np.mean(x), fs=fs, nperseg=nper)
    tot = np.trapezoid(pxx, fr) + 1e-20
    f[f"f_{name}_centroid"] = np.trapezoid(fr * pxx, fr) / tot
    for k in (1, 2, 3):
        fc = k * f_tooth
        if fc < fs / 2:
            band = (fr > fc * 0.95) & (fr < fc * 1.05)
            f[f"f_{name}_E{k}"] = np.log10(np.trapezoid(pxx[band], fr[band]) + 1e-20) if band.sum() > 1 else np.nan
    # relative energy in four equal bands up to Nyquist
    edges = np.linspace(0, fs / 2, 5)
    for i in range(4):
        band = (fr >= edges[i]) & (fr < edges[i + 1])
        f[f"f_{name}_band{i}"] = np.trapezoid(pxx[band], fr[band]) / tot if band.sum() > 1 else np.nan
    return f


def pass_features(X, fs, f_tooth, names):
    out = {}
    for x, n in zip(X, names):
        out.update(channel_features(x, fs, f_tooth, n))
    return out
