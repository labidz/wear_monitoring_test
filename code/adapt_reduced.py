"""Turn the Kaggle-reduced PHM2010 and Denkena (LUH) outputs into the common per-pass tables.

    python adapt_reduced.py PHM2010 <dir with phm2010_reduced.csv, phm2010_wear.csv, phm2010_windows.npz>
    python adapt_reduced.py DENKENA <dir with luh_reduced.csv, luh_windows.npz> [force]
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys
import numpy as np
import pandas as pd
from physics import mean_chip_thickness, load_rise

ROOT = _REPO + "/work"


def phm(d):
    R = pd.read_csv(f"{d}/phm2010_reduced.csv")
    W = pd.read_csv(f"{d}/phm2010_wear.csv")
    W.columns = [c.strip().lower() for c in W.columns]
    W["VB"] = W[[c for c in W.columns if c.startswith("flute")]].max(1) / 1000.0
    R = R.merge(W[["tool", "cut", "VB"]], on=["tool", "cut"], how="left").sort_values(["tool", "cut"])
    FS, N, Z, FEED, AP, AE, RB = 50_000.0, 10_400.0, 3, 1555.0, 0.2, 0.125, 3.0
    D = 2 * np.sqrt(2 * RB * AP - AP ** 2)
    t = pd.DataFrame({"dataset": "PHM2010", "tool": R.tool, "group": R.tool, "pass": R.cut,
                      "time": R.groupby("tool").n_samples.cumsum() / FS / 60.0, "VB": R.VB,
                      "mat": "stainless_HRC52", "ap": AP, "ae": AE, "D": D, "z": Z, "fz": FEED / (N * Z), "n": N,
                      "vc": np.pi * D * N / 1000.0, "S": R.S})
    t["h_m"] = mean_chip_thickness(t.fz.values, t.ae.values, t.D.values)
    for c in [c for c in R.columns if c.startswith("f_")]:
        t[c] = R[c].values
    t["f_S"] = t.S
    t["win_idx"] = R.win_idx.values
    X = np.load(f"{d}/phm2010_windows.npz")["X"].astype(np.float32)
    return load_rise(t), X


def denkena(d, load="torque"):
    R = pd.read_csv(f"{d}/luh_reduced.csv").sort_values(["machine", "tool", "run"])
    D, Z, VC, VF, AE, AP = 8.0, 4, 200.0, 1592.0, 4.0, 2.0
    n = VC * 1000 / (np.pi * D)
    t = pd.DataFrame({"dataset": "DENKENA", "tool": "T" + R.tool.astype(str), "group": "M" + R.machine.astype(str),
                      "pass": R.run, "time": R.cumulated_tool_contact_time / 60.0, "VB": R.wear / 1000.0,
                      "mat": "cast_iron_GJS600", "ap": AP, "ae": AE, "D": D, "z": Z, "fz": VF / (n * Z), "n": n,
                      "vc": VC})
    t["S"] = (R.S_torque if load == "torque" else R.S_force).values
    t["h_m"] = mean_chip_thickness(t.fz.values, t.ae.values, D)
    for c in [c for c in R.columns if c.startswith("f_")]:
        t[c] = R[c].values
    t["f_S"] = t.S
    t["win_idx"] = R.win_idx.values
    Z_ = np.load(f"{d}/luh_windows.npz")
    X = Z_["S"].astype(np.float32)          # dynamometer windows (3 x 1024) for the CNN baseline
    return load_rise(t), X


if __name__ == "__main__":
    ds, d = sys.argv[1], sys.argv[2]
    if ds == "PHM2010":
        t, X = phm(d)
        name = "PHM2010"
    else:
        load = sys.argv[3] if len(sys.argv) > 3 else "torque"
        t, X = denkena(d, load)
        name = "DENKENA" if load == "torque" else "DENKENA_F"
    t.to_csv(f"{ROOT}/tab_{name}.csv", index=False)
    np.savez_compressed(f"{ROOT}/win_{name}.npz", X=X)
    print(t.groupby(["group", "tool"]).agg(passes=("pass", "size"), labelled=("VB", "count"), vbmax=("VB", "max"),
                                           tmax=("time", "max")).to_string())
