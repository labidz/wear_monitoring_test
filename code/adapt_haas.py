"""Protocol v3: Haas VF-1 (Piecuch & Zabinski, Sci Data 2025) -> common per-pass tables.

    python adapt_haas.py <FeatureAndMetadata_Milling.csv>

Writes tab_HAAS.csv (groups = tools, primary) and tab_HAAS_LOCO.csv (groups = cutting conditions, secondary).
The label is the consumed life fraction, stored in the column VB so that all model code runs unchanged.
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys
import numpy as np
import pandas as pd
from physics import mean_chip_thickness, load_rise

ROOT = _REPO + "/work"
N, FEED, D, Z = 3200.0, 640.0, 10.0, 4          # constant for all runs (data descriptor, Table 3)


def main(path):
    d = pd.read_csv(path, sep=";", header=1)
    for c in ("RDOC", "HardnessMean", "CycleToFailureNormalized"):    # metadata use a decimal comma
        d[c] = d[c].astype(str).str.replace(",", ".").astype(float)
    d = d[d.TollIndex != 11].copy()                # broke in its first cycle (outlier per data owners)
    sig = [c for c in d.columns if c.startswith(("Accelerometer", "Current"))]
    spindle_std = [f"Current - Spindle L{k} - std" for k in (1, 2, 3)]
    t = pd.DataFrame({
        "dataset": "HAAS",
        "tool": "H" + d.TollIndex.astype(str),
        "pass": d.NumberOfCycle.astype(int),
        "time": d.NumberOfCycle.astype(float),
        "VB": 1.0 - d.CycleToFailureNormalized.astype(float),
        "mat": "42CrMo4_T" + d.MillingToolType.astype(str),
        "ap": d.ADOC.astype(float), "ae": d.RDOC.astype(float),
        "fz": FEED / (N * Z), "vc": np.pi * D * N / 1000.0, "n": N, "D": D, "z": Z,
        "c_hard": d.HardnessMean.astype(float), "c_holder": d.ToolHolderLength.astype(float),
        "S": d[spindle_std].mean(axis=1).values,
        "file": d.FileName.values,
    })
    t["h_m"] = mean_chip_thickness(t.fz.values, t.ae.values, t.D.values)
    t["cond"] = ("T" + d.MillingToolType.astype(str) + "_ap" + d.ADOC.astype(str) + "_ae"
                 + d.RDOC.astype(str) + "_L" + d.ToolHolderLength.astype(str)).values
    F = d[sig].astype(float).fillna(0.0)
    F.columns = ["f_" + c.replace(" ", "") for c in sig]
    t = pd.concat([t.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    t["f_S"] = t.S
    t["win_idx"] = -1
    t = load_rise(t, n_ref=3)
    assert t.VB.between(0, 1).all() and t.S.gt(0).all() and t.r.notna().all()
    for name, grp in [("HAAS", "tool"), ("HAAS_LOCO", "cond")]:
        tt = t.copy()
        tt["group"] = tt[grp]
        tt.to_csv(f"{ROOT}/tab_{name}.csv", index=False)
    print(t.groupby("tool").agg(cond=("cond", "first"), cycles=("pass", "size"), life=("pass", "max"),
                                S0=("S", "first")).to_string())
    print("conditions:", t.cond.nunique(), "| features:", len(F.columns), "| rows:", len(t))


if __name__ == "__main__":
    main(sys.argv[1])
