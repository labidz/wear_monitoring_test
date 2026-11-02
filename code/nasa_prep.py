"""NASA Ames milling data (Agogino & Goebel, 2007): per-run signal reduction.

Reads mill.mat and writes one row per run with segment bounds, idle and
in-cut current levels, and per-channel statistics of the steady cut.
Physics quantities are derived later (physics.py); this file only measures.
"""
import sys
import numpy as np
import pandas as pd
import scipy.io as sio

FS = 250.0          # Hz, sampling rate of all channels
CH = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]


def segment(dc):
    """Steady-cut segment from the spindle DC current.

    idle = median of samples 100..600 (spindle running, tool in air);
    active = dc > idle + 0.5 (p90 - idle); trimmed 10 % at both ends.
    """
    idle = float(np.median(dc[100:600]))
    p90 = float(np.percentile(dc, 90))
    act = np.where(dc > idle + 0.5 * (p90 - idle))[0]
    if len(act) < 200 or p90 - idle < 0.3:
        return None
    a, b = act[0], act[-1]
    trim = int(0.1 * (b - a))
    return a + trim, b - trim, idle


def run_stats(rec):
    dc = rec["smcDC"].ravel().astype(float)
    seg = segment(dc)
    out = dict(case=int(rec["case"][0, 0]), run=int(rec["run"][0, 0]),
               VB=float(rec["VB"][0, 0]) if rec["VB"].size else np.nan,
               time=float(rec["time"][0, 0]) if rec["time"].size else np.nan,
               DOC=float(rec["DOC"][0, 0]), feed=float(rec["feed"][0, 0]),
               material=int(rec["material"][0, 0]), n=len(dc))
    if seg is None:
        out["valid"] = False
        return out
    a, b, idle_dc = seg
    out.update(valid=True, seg_a=a, seg_b=b, dc_idle=idle_dc,
               dc_clip_frac=float(np.mean(dc[a:b] >= 9.95)))
    ac = rec["smcAC"].ravel().astype(float)
    ac = ac - np.mean(ac[100:600])            # remove sensor offset (idle mean)
    out["ac_rms_idle"] = float(np.sqrt(np.mean(ac[100:600] ** 2)))
    out["ac_rms_cut"] = float(np.sqrt(np.mean(ac[a:b] ** 2)))
    for k in CH:
        x = rec[k].ravel().astype(float)[a:b]
        out[k + "_mean"] = float(np.mean(x))
        out[k + "_rms"] = float(np.sqrt(np.mean(x ** 2)))
        out[k + "_std"] = float(np.std(x))
    return out


def main(mat, dst):
    m = sio.loadmat(mat)["mill"]
    rows = [run_stats(m[0, i]) for i in range(m.shape[1])]
    df = pd.DataFrame(rows).sort_values(["case", "run"]).reset_index(drop=True)
    df.to_csv(dst, index=False)
    print(df.groupby("case").agg(n=("run", "size"), valid=("valid", "sum"),
                                 clip=("dc_clip_frac", "mean")).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
