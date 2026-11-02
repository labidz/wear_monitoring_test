"""Pre-registered sensitivity analyses (protocol v1 amendments 2 and 4; v2 sensorless variants; v3 LOCO).

Hannover: (a) without tool T8 (reference not from a fresh tool); (b) metrics only at the microscope measurement points,
i.e. the breakpoints of each tool's piecewise-linear label trajectory (plus first and last pass).
Variants evaluated as separate tables: *_R3 (three-pass reference), DENKENA_F (dynamometer force),
*_SENSORLESS (sensorless load), HAAS_LOCO (leave one condition out).  Writes results/sensitivity.csv.
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import os
import numpy as np
import pandas as pd
from primary import load, paired

ROOT = _REPO + ""
MODELS = ["B2", "B3", "P1", "P2", "P3"]


def breakpoints(t, tol=0.0006):
    """Passes where the label leaves the straight line through its two neighbours by more than tol (labels are
    rounded to 1 µm): the microscope measurement points of the owners' piecewise-linear interpolation."""
    keep = []
    for _, g in t.sort_values(["tool", "pass"]).groupby("tool"):
        v, x = g.VB.values, g.time.values
        w = (x[1:-1] - x[:-2]) / np.maximum(x[2:] - x[:-2], 1e-9)
        k = np.r_[True, np.abs(v[1:-1] - (v[:-2] + w * (v[2:] - v[:-2]))) > tol, True]
        keep += list(zip(g.tool[k], g["pass"][k]))
    return set(keep)


def contrasts(ds, L, tag):
    rows = []
    for m in MODELS:
        if m in set(L.model) and "B1" in set(L.model):
            r = paired(L, m, "B1", np.random.default_rng(0))
            rows.append(dict(analysis=tag, dataset=ds, model=m, ref="B1", **r))
    return rows


def main():
    rows = []
    L = load("DENKENA")
    rows += contrasts("DENKENA", L[L.tool != "T8"], "Hannover without T8")
    t = pd.read_csv(f"{ROOT}/work/tab_DENKENA.csv")
    bp = breakpoints(t)
    key = list(zip(L.tool, L["pass"]))
    rows += contrasts("DENKENA", L[[k in bp for k in key]], "Hannover, measured points only")
    for ds, tag in [("NASA_R3", "three-pass reference"), ("PHM2010_R3", "three-pass reference"),
                    ("DENKENA_R3", "three-pass reference"), ("DENKENA_F", "dynamometer force"),
                    ("VICOMTECH_SENSORLESS", "sensorless (spindle torque)"),
                    ("NUAA_SENSORLESS", "sensorless (spindle power)"), ("HAAS_LOCO", "leave one condition out")]:
        if os.path.isdir(f"{ROOT}/results/{ds}"):
            try:
                rows += contrasts(ds, load(ds), tag)
            except ValueError as e:
                print(ds, "skipped:", e)
    T = pd.DataFrame(rows)
    T.to_csv(f"{ROOT}/results/sensitivity.csv", index=False)
    pd.set_option("display.width", 220)
    print(T[["analysis", "dataset", "model", "rmse", "rmse_ref", "diff", "diff_lo", "diff_hi", "ratio"]]
          .round(4).to_string(index=False))
    print("measured points kept:", len(bp))


if __name__ == "__main__":
    main()
