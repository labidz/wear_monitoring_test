"""Mechanics-normalised wear indicator.

Linear edge-force milling model with a flank-wear-land force proportional to VB:
    mean load, fresh tool:  S0 = K_tc * h_m * a_p * c + K_te0 * (a_p + b_e) * c
    wear-induced increase:  dS = tau_w * VB * (a_p + b_w) * c
c collects engaged-tooth count and speed; it cancels in the ratio r = dS / S0.
Neglecting the small fresh-edge contribution of the nose (b_e ~ 0) gives
    VB = alpha * r * (h_m + h_0) * a_p / (a_p + b_w),
with alpha = K_tc / tau_w, h_0 = K_te0 / K_tc and b_w the wear-land length outside a_p.
"""
import numpy as np
import pandas as pd


def mean_chip_thickness(fz, ae, D):
    """Mean uncut chip thickness over the engagement arc (peripheral milling, one-sided entry)."""
    ae = np.minimum(ae, D)
    phi = np.arccos(1.0 - 2.0 * ae / D)          # engagement angle
    return fz * (1.0 - np.cos(phi)) / phi


def mean_chip_thickness_symmetric(fz, ae, D):
    """Face milling with the cutter centred on the workpiece."""
    ae = np.minimum(ae, D)
    p1 = np.arccos(ae / D)                        # entry angle from the feed direction normal
    span = np.pi - 2.0 * p1
    return fz * 2.0 * np.cos(p1) / span


def load_rise(df, load_col="S", n_ref=1):
    """r = S / S_ref - 1, with S_ref the first valid pass of each tool (unlabelled reference);
    n_ref > 1: median of the first n_ref passes (robust-reference sensitivity analysis, amendment 4)."""
    df = df.sort_values(["tool", "pass"]).copy()
    ref = df.groupby("tool")[load_col].transform(lambda s: s.iloc[:n_ref].median())
    df["r"] = df[load_col] / ref - 1.0
    return df


def geometry_factor(df, h0, bw):
    return (df["h_m"].values + h0) * df["ap"].values / (df["ap"].values + bw)


def physics_lambda(df, h0, bw):
    return df["r"].values * geometry_factor(df, h0, bw)
