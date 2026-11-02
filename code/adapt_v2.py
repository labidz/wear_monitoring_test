"""Protocol v2 confirmatory datasets -> common per-pass tables (rules of amendment v2-1).

    python adapt_v2.py NUAA <nuaa_orthogonal_bundle_high_resolution.csv>
    python adapt_v2.py VICOMTECH <Dataset_Machine_tool_wear.csv>
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys
import numpy as np
import pandas as pd
from scipy.signal import resample
from features import pass_features
from physics import mean_chip_thickness, load_rise

ROOT = _REPO + "/work"


def nuaa(path):
    d = pd.read_csv(path)
    CH = ["torsion", "axial_force", "bending_moment_x", "bending_moment_y", "vibration1", "vibration2"]
    D, Z, AE = 10.0, 4, 1.0          # not documented in the bundle; constants only rescale h_m (amendment v2-1 c)
    rows, wins = [], []
    for (exp, k), g in d.groupby(["experiment_tag", "experiment_csv_n"], sort=True):
        g = g.sort_values("timestamp")
        P = g.spindle_power.values
        idle = np.percentile(P, 5)
        act = P > idle + 0.5 * (np.percentile(P, 90) - idle)
        seg = g[act] if act.sum() >= 20 else g
        blades = g[[f"wear_blade_{i}" for i in range(1, 5)]].max(axis=1).values
        mid = len(g) // 2
        r = dict(dataset="NUAA", tool=exp, group=exp, **{"pass": int(k)}, time=g.timestamp.values[mid] / 60.0,
                 VB=float(blades[mid]), mat="Ti6Al4V", ap=float(g.axial_cutting_depth.iloc[0]),
                 fz=float(g.feed_per_tooth.iloc[0]), n=float(g.spindle_speed.iloc[0]), D=D, z=Z, ae=AE,
                 S=float(np.mean(seg.torsion.values)), S_power=float(np.mean(seg.spindle_power.values) - idle),
                 seg_frac=float(act.mean()))
        r["vc"] = np.pi * D * r["n"] / 1000.0
        X = seg[CH].values.T.astype(float)
        r.update(pass_features(X, 15.0, r["n"] * Z / 60.0, CH))
        rows.append(r)
        wins.append(resample(X, 256, axis=1).astype(np.float32))
    t = pd.DataFrame(rows)
    t["h_m"] = mean_chip_thickness(t.fz.values, t.ae.values, t.D.values)
    t = t.loc[:, t.notna().any()]
    fcols = [c for c in t.columns if c.startswith("f_")]
    t[fcols] = t[fcols].fillna(0.0)
    t["f_S"] = t.S
    t["win_idx"] = np.arange(len(t))
    return t, np.stack(wins)


def vicomtech(path):
    d = pd.read_csv(path, sep=None, engine="python")
    rows = []
    feats = [c for c in d.columns if c not in ("tool", "Vb")]
    for tool, g in d.groupby("tool", sort=False):
        g = g.reset_index(drop=True)
        for i, row in g.iterrows():
            r = dict(dataset="VICOMTECH", tool=f"V{tool}", group=f"V{tool}", **{"pass": i + 1}, time=(i + 1) / 60.0,
                     VB=row.Vb / 1000.0, mat="steel_19NiMoCr6", ap=2.0, fz=0.1, n=row["CON.A.SREAL.S"],
                     D=np.nan, z=1, ae=np.nan, vc=200.0, h_m=0.1, S=row["F_c_MEAN"], S_torque=row["TV2.S_MEAN"])
            r.update({"f_" + c: row[c] for c in feats})
            rows.append(r)
    t = pd.DataFrame(rows)
    fcols = [c for c in t.columns if c.startswith("f_")]
    t[fcols] = t[fcols].fillna(0.0)      # skew/kurtosis of constant segments (amendment v2-3)
    t["win_idx"] = -1
    return t, None


if __name__ == "__main__":
    ds, path = sys.argv[1], sys.argv[2]
    t, X = (nuaa if ds == "NUAA" else vicomtech)(path)
    for name, col in [(ds, "S"), (ds + "_SENSORLESS", "S_power" if ds == "NUAA" else "S_torque")]:
        tt = t.copy()
        tt["S"] = tt[col]
        bad = tt.S.isna()
        if bad.any():
            print(name, "dropped passes with missing load:", tt.loc[bad, ["tool", "pass"]].values.tolist())
        tt = tt[~bad].copy()
        tt["f_S"] = tt["S"]
        tt = load_rise(tt, n_ref=3)                 # v2 reference: median of the first three passes
        tt.to_csv(f"{ROOT}/tab_{name}.csv", index=False)
        if X is not None:
            np.savez_compressed(f"{ROOT}/win_{name}.npz", X=X)
    print(t.groupby("group").agg(passes=("pass", "size"), vb0=("VB", "first"), vbmax=("VB", "max"),
                                 tmax=("time", "max"), S0=("S", "first")).round(3).to_string())
