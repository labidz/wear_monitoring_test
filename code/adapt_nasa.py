"""NASA Ames milling -> common per-pass table + raw signal windows.

Machine: Matsuura MC-510V; 70 mm face mill, 6 KC710 inserts; 826 rpm.
Settings: depth of cut 0.75/1.5 mm, feed 0.25/0.5 mm/rev, cast iron (1) / steel (2).
Width of cut is not documented; a centred engagement a_e = 0.7 D is assumed (only scales h_m).
"""
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import resample
from physics import mean_chip_thickness_symmetric, load_rise
from features import pass_features

D, Z, N_RPM = 70.0, 6, 826.0
CH_RAW = ["smcAC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
WIN = 1024
SETTINGS = {(1.5, .5, 1): "S1", (.75, .5, 1): "S2", (.75, .25, 1): "S3", (1.5, .25, 1): "S4",
            (1.5, .5, 2): "S5", (1.5, .25, 2): "S6", (.75, .25, 2): "S7", (.75, .5, 2): "S8"}


def main(mat_path, runs_csv, out_csv, out_npz):
    runs = pd.read_csv(runs_csv)
    runs = runs[(runs.valid == True) & (runs.case != 6)].copy()
    m = sio.loadmat(mat_path)["mill"]
    recs = {(int(m[0, i]["case"][0, 0]), int(m[0, i]["run"][0, 0])): m[0, i] for i in range(m.shape[1])}

    t = pd.DataFrame({
        "dataset": "NASA",
        "tool": "N" + runs.case.astype(str),
        "group": [SETTINGS[(a, f, mt)] for a, f, mt in zip(runs.DOC, runs.feed, runs.material)],
        "pass": runs.run.values,
        "time": runs.time.values,                       # cumulative cutting time, min
        "VB": runs.VB.values,                           # mm
        "mat": np.where(runs.material == 1, "cast_iron", "steel"),
        "ap": runs.DOC.values, "fz": runs.feed.values / Z, "n": N_RPM, "D": D, "z": Z,
    })
    t["ae"] = 0.7 * D
    t["vc"] = np.pi * D * N_RPM / 1000.0
    t["h_m"] = mean_chip_thickness_symmetric(t.fz.values, t.ae.values, D)
    # spindle torque component of the AC motor current
    t["S"] = np.sqrt(np.clip(runs.ac_rms_cut.values ** 2 - runs.ac_rms_idle.values ** 2, 1e-12, None))
    t["f_S"] = t["S"]

    wins, feats = [], []
    for (cs, rn, a, b) in zip(runs.case, runs.run, runs.seg_a, runs.seg_b):
        r = recs[(cs, rn)]
        x = np.stack([r[k].ravel().astype(float)[int(a):int(b)] for k in CH_RAW])
        x[0] = x[0] - np.mean(r["smcAC"].ravel()[100:600])
        feats.append(pass_features(x, 250.0, N_RPM / 60.0 * Z, CH_RAW))
        wins.append(resample(x, WIN, axis=1).astype(np.float32))
    F = pd.DataFrame(feats)
    for c in F.columns:
        t[c] = F[c].values
    t["win_idx"] = np.arange(len(t))
    t = load_rise(t)
    t.to_csv(out_csv, index=False)
    np.savez_compressed(out_npz, X=np.stack(wins))
    print(t.groupby("group").agg(tools=("tool", "nunique"), passes=("pass", "size"),
                                 labelled=("VB", "count")).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
