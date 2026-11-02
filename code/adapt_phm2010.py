"""PHM 2010 milling data challenge -> common per-pass table + raw windows.

Röders RFM760; 6 mm ball-nose tungsten-carbide end mill, 3 flutes; stainless steel HRC52.
10400 rpm, feed 1555 mm/min, radial depth 0.125 mm, axial depth 0.2 mm (constant for all cuts).
Each cut file: 7 columns at 50 kHz (Fx, Fy, Fz [N], Vx, Vy, Vz [g], AE-RMS [V]).
Wear files: cut, flute_1, flute_2, flute_3 in 1e-3 mm. Label VB = max over flutes (mm).
Usage: python adapt_phm2010.py <root folder containing c1, c4, c6 and *_wear.csv> <out_csv> <out_npz>
"""
import glob
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from features import pass_features
from physics import mean_chip_thickness, load_rise

FS, N_RPM, Z, FEED = 50_000.0, 10_400.0, 3, 1555.0
AP, AE, R_BALL = 0.2, 0.125, 3.0
D_EFF = 2.0 * np.sqrt(2 * R_BALL * AP - AP ** 2)     # effective diameter at the axial depth
F_TOOTH = N_RPM / 60.0 * Z
NAMES = ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE"]
WIN = 2048


def find(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return hits


def steady(x):
    """Steady-cut part: samples where the in-plane force envelope exceeds half its 95th percentile."""
    fxy = np.hypot(x[:, 0], x[:, 1])
    k = 500
    env = np.convolve(fxy, np.ones(k) / k, mode="same")
    act = np.where(env > 0.5 * np.percentile(env, 95))[0]
    a, b = act[0], act[-1]
    trim = int(0.1 * (b - a))
    return a + trim, b - trim


def main(root, out_csv, out_npz):
    rows, wins = [], []
    for tool in ["c1", "c4", "c6"]:
        wf = [p for p in find(root, f"{tool}_wear.csv")]
        wear = pd.read_csv(wf[0])
        wear.columns = [c.strip().lower() for c in wear.columns]
        vb = wear[[c for c in wear.columns if c.startswith("flute")]].max(1).values / 1000.0
        files = [p for p in find(root, f"c_{tool[1]}_*.csv")]
        files = sorted(files, key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
        t_cum = 0.0
        for i, p in enumerate(files):
            x = pd.read_csv(p, header=None).values.astype(float)
            a, b = steady(x)
            seg = x[a:b]
            dur = len(x) / FS / 60.0                          # min
            t_cum += dur
            fxy = np.hypot(seg[:, 0], seg[:, 1])
            r = dict(dataset="PHM2010", tool=tool, group=tool, **{"pass": i + 1}, time=t_cum,
                     VB=vb[i] if i < len(vb) else np.nan, mat="stainless_HRC52", ap=AP, ae=AE, D=D_EFF, z=Z,
                     fz=FEED / (N_RPM * Z), n=N_RPM, vc=np.pi * D_EFF * N_RPM / 1000.0,
                     S=float(np.mean(fxy)))
            r.update(pass_features(seg.T, FS, F_TOOTH, NAMES))
            rows.append(r)
            mid = seg[len(seg) // 2 - int(FS / 2): len(seg) // 2 + int(FS / 2)]   # 1 s centred window
            wins.append(resample_poly(mid, WIN, len(mid), axis=0).T.astype(np.float32))
            if (i + 1) % 50 == 0:
                print(tool, i + 1, flush=True)
    t = pd.DataFrame(rows)
    t["h_m"] = 0.0
    t["h_m"] = mean_chip_thickness(t.fz.values, t.ae.values, t.D.values)
    t["f_S"] = t["S"]
    t["win_idx"] = np.arange(len(t))
    t = load_rise(t)
    t.to_csv(out_csv, index=False)
    np.savez_compressed(out_npz, X=np.stack(wins))
    print(t.groupby("tool").agg(passes=("pass", "size"), labelled=("VB", "count"), vbmax=("VB", "max")))


if __name__ == "__main__":
    main(*sys.argv[1:4])
