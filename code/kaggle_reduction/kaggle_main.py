import glob, os, numpy as np, pandas as pd, pyarrow.parquet as pq
from multiprocessing import Pool
from scipy.signal import resample_poly, welch
from scipy.stats import kurtosis, skew
OUT = "/kaggle/working"
P = glob.glob("/kaggle/input/**/*.parquet", recursive=True)[0]
FS_S, FS_C = 25_000.0, 500.0
CTRL_IN = ["torque_spindle", "torque_axis_x", "torque_axis_y", "force_axis_x", "force_axis_y",
           "position_control_deviation_axis_x", "position_control_deviation_axis_y"]
CTRL = ["torque_spindle", "drive_x", "drive_y", "position_control_deviation_axis_x", "position_control_deviation_axis_y"]
SENS = ["force_sensor_x", "force_sensor_y", "force_sensor_z"]
POS = ["tool_position_x", "tool_position_y", "tool_position_z"]
META = ["machine", "tool", "run", "cumulated_tool_contact_time", "wear"]


def chf(x, fs, name):
    f = {f"f_{name}_mean": np.mean(x), f"f_{name}_std": np.std(x), f"f_{name}_rms": np.sqrt(np.mean(x ** 2)),
         f"f_{name}_skew": skew(x), f"f_{name}_kurt": kurtosis(x), f"f_{name}_p2p": np.ptp(x)}
    fr, p = welch(x - np.mean(x), fs=fs, nperseg=int(min(len(x), 4096)))
    tot = np.trapezoid(p, fr) + 1e-20
    f[f"f_{name}_centroid"] = np.trapezoid(fr * p, fr) / tot
    f[f"f_{name}_fpeak"] = fr[np.argmax(p[1:]) + 1]
    e = np.linspace(0, fs / 2, 5)
    for i in range(4):
        b = (fr >= e[i]) & (fr < e[i + 1])
        f[f"f_{name}_band{i}"] = np.trapezoid(p[b], fr[b]) / tot
    return f


def segment(y, frac_idle=0.08):
    """cutting segment from a load-like signal: above idle + 0.5 (p90 - idle), trimmed 10 %."""
    n = len(y)
    k = max(int(frac_idle * n), 5)
    idle = np.median(np.r_[y[:k], y[-k:]])
    p90 = np.percentile(y, 90)
    act = np.where(y > idle + 0.5 * (p90 - idle))[0]
    if len(act) < 10:
        return 0, n, idle, False
    a, b = act[0], act[-1]
    t = int(0.1 * (b - a))
    return a + t, b - t, idle, True


def one(rg):
    pf = pq.ParquetFile(P)
    cols = META + CTRL_IN + SENS + POS
    tab = pf.read_row_group(rg, columns=cols)
    meta = {k: tab.column(k).to_pylist() for k in META}
    lists = {}
    for c in CTRL_IN + SENS + POS:
        col = tab.column(c).combine_chunks()
        lists[c] = (col.offsets.to_numpy(), col.values.to_numpy(zero_copy_only=False))
    del tab
    rows, ws, wc = [], [], []
    for i in range(len(meta["run"])):
        r = {c: lists[c][1][lists[c][0][i]:lists[c][0][i + 1]] for c in lists}
        nT = len(r["torque_spindle"])
        for ax in ("x", "y"):   # M1: feed-drive torque; M2, M3: feed-drive force (data descriptor, Table 2)
            a_ = r["torque_axis_" + ax]
            r["drive_" + ax] = a_ if len(a_) == nT else (r["force_axis_" + ax] if len(r["force_axis_" + ax]) == nT
                                                       else np.zeros(nT))
        try:
            o, w_s, w_c = one_row(r, {k: meta[k][i] for k in META}, rg)
        except Exception as e:
            print("skip", rg, i, {k: meta[k][i] for k in META}, repr(e)[:120], flush=True)
            continue
        rows.append(o)
        ws.append(w_s)
        wc.append(w_c)
    return rows, ws, wc


def one_row(r, o, rg):
    if True:
        o["rg"] = rg
        # --- sensor (dynamometer) channel
        F = np.stack([np.asarray(r[c], dtype=float) for c in SENS])
        fxy = np.hypot(F[0], F[1])
        env = np.convolve(fxy, np.ones(250) / 250, mode="same")
        a, b, idle_f, ok_f = segment(env)
        o.update(n_sens=F.shape[1], sa=a, sb=b, ok_sens=ok_f, S_force=float(np.mean(fxy[a:b]) - idle_f))
        for i, c in enumerate(SENS):
            o.update(chf(F[i, a:b], FS_S, c))
        m, h = (a + b) // 2, int(min(FS_S / 2, (b - a) // 2))
        w_s = resample_poly(F[:, m - h:m + h], 1024, 2 * h, axis=1).astype(np.float16)
        # --- machine control channel
        T = np.asarray(r["torque_spindle"], dtype=float)
        sgn = np.sign(np.median(T) - np.median(np.r_[T[:20], T[-20:]])) or 1.0
        Tl = sgn * T
        a2, b2, idle_t, ok_t = segment(np.convolve(Tl, np.ones(10) / 10, mode="same"))
        o.update(n_ctrl=len(T), ca=a2, cb=b2, ok_ctrl=ok_t, T_idle=idle_t, S_torque=float(np.mean(Tl[a2:b2]) - idle_t),
                 T_cut=float(np.mean(Tl[a2:b2])))
        C = np.stack([np.asarray(r[c], dtype=float) for c in CTRL])
        for i, c in enumerate(CTRL):
            o.update(chf(C[i, a2:b2], FS_C, c))
        m2, h2 = (a2 + b2) // 2, int(min(FS_C / 2, (b2 - a2) // 2))
        w_c = resample_poly(C[:, m2 - h2:m2 + h2], 256, 2 * h2, axis=1).astype(np.float16)
        # --- kinematics: feed speed and mean position in the cut
        X = np.stack([np.asarray(r[c], dtype=float) for c in POS])
        v = np.hypot(np.diff(X[0, a2:b2]), np.diff(X[1, a2:b2])) * FS_C
        o.update(vf_mm_s=float(np.median(v)), **{f"mean_{c}": float(np.mean(X[i, a2:b2])) for i, c in enumerate(POS)},
                 **{f"min_{c}": float(np.min(X[i])) for i, c in enumerate(POS)},
                 **{f"max_{c}": float(np.max(X[i])) for i, c in enumerate(POS)})
        fr, p = welch(fxy[a:b] - np.mean(fxy[a:b]), fs=FS_S, nperseg=8192)
        o["f_tooth_peak"] = fr[np.argmax(p[(fr > 20)]) + np.argmax(fr > 20)]
        assert w_s.shape == (3, 1024) and w_c.shape == (5, 256), (w_s.shape, w_c.shape)
        return o, w_s, w_c


def run_luh():
    n = pq.ParquetFile(P).num_row_groups
    rows, ws, wc = [], [], []
    with Pool(3, maxtasksperchild=8) as pool:
        for i, (r, a, c) in enumerate(pool.imap(one, range(n))):
            rows += r
            ws += a
            wc += c
            if i % 10 == 0:
                print("luh rg", i, len(rows), flush=True)
    T = pd.DataFrame(rows)
    T["win_idx"] = np.arange(len(T))
    T.to_csv(f"{OUT}/luh_reduced.csv", index=False)
    np.savez_compressed(f"{OUT}/luh_windows.npz", S=np.stack(ws), C=np.stack(wc))
    print("LUH DONE", T.shape, flush=True)
    print(T.groupby(["machine", "tool"]).agg(runs=("run", "size"), wmax=("wear", "max"), tmax=("cumulated_tool_contact_time", "max"),
                                              S_t=("S_torque", "median"), S_f=("S_force", "median"), vf=("vf_mm_s", "median"),
                                              ftp=("f_tooth_peak", "median"), okc=("ok_ctrl", "mean")).round(2).to_string(), flush=True)



import glob, os, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import resample_poly, welch
from scipy.stats import kurtosis, skew
FS, N_RPM, Z = 50_000.0, 10_400.0, 3
F_TOOTH = N_RPM / 60.0 * Z
NAMES = ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE"]
OUT = "/kaggle/working"


def chf_phm(x, fs, ft, name):
    f = {f"f_{name}_mean": np.mean(x), f"f_{name}_std": np.std(x), f"f_{name}_rms": np.sqrt(np.mean(x ** 2)),
         f"f_{name}_skew": skew(x), f"f_{name}_kurt": kurtosis(x), f"f_{name}_p2p": np.ptp(x)}
    fr, p = welch(x - np.mean(x), fs=fs, nperseg=int(min(len(x), 4096)))
    tot = np.trapezoid(p, fr) + 1e-20
    f[f"f_{name}_centroid"] = np.trapezoid(fr * p, fr) / tot
    for k in (1, 2, 3):
        b = (fr > .95 * k * ft) & (fr < 1.05 * k * ft)
        f[f"f_{name}_E{k}"] = np.log10(np.trapezoid(p[b], fr[b]) + 1e-20) if b.sum() > 1 else np.nan
    e = np.linspace(0, fs / 2, 5)
    for i in range(4):
        b = (fr >= e[i]) & (fr < e[i + 1])
        f[f"f_{name}_band{i}"] = np.trapezoid(p[b], fr[b]) / tot
    return f


def steady(x):
    fxy = np.hypot(x[:, 0], x[:, 1])
    env = np.convolve(fxy, np.ones(500) / 500, mode="same")
    act = np.where(env > 0.5 * np.percentile(env, 95))[0]
    a, b = act[0], act[-1]
    t = int(0.1 * (b - a))
    return a + t, b - t


def one_phm(args):
    tool, p = args
    x = pd.read_csv(p, header=None).values.astype(float)
    a, b = steady(x)
    seg = x[a:b]
    r = {"tool": tool, "cut": int(os.path.basename(p).split("_")[-1].split(".")[0]), "n_samples": len(x),
         "seg_a": a, "seg_b": b, "S": float(np.mean(np.hypot(seg[:, 0], seg[:, 1])))}
    for c, n in enumerate(NAMES):
        r.update(chf_phm(seg[:, c], FS, F_TOOTH, n))
    m = len(seg) // 2
    h = int(min(FS / 2, m))
    w = resample_poly(seg[m - h:m + h], 2048, 2 * h, axis=0).T.astype(np.float32)
    return r, w


def run_phm():
    root = glob.glob("/kaggle/input/**/phm*", recursive=True)
    root = "/kaggle/input"
    jobs, wear = [], []
    for tool in ["c1", "c4", "c6"]:
        fs = glob.glob(f"{root}/**/c_{tool[1]}_*.csv", recursive=True)
        fs = sorted({os.path.basename(p): p for p in fs}.values(), key=lambda p: int(p.split("_")[-1][:-4]))
        jobs += [(tool, p) for p in fs]
        w = pd.read_csv(glob.glob(f"{root}/**/{tool}_wear.csv", recursive=True)[0])
        w.insert(0, "tool", tool)
        wear.append(w)
        print(tool, len(fs), flush=True)
    rows, wins = [], []
    with Pool(os.cpu_count()) as pool:
        for i, (r, w) in enumerate(pool.imap(one_phm, jobs, chunksize=4)):
            r["win_idx"] = i
            rows.append(r)
            wins.append(w)
            if i % 100 == 0:
                print("phm", i, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/phm2010_reduced.csv", index=False)
    np.savez_compressed(f"{OUT}/phm2010_windows.npz", X=np.stack(wins))
    pd.concat(wear).to_csv(f"{OUT}/phm2010_wear.csv", index=False)
    print("PHM DONE", len(rows), flush=True)



import time
if __name__ == "__main__":
    t0=time.time()
    run_phm()
    print("t",time.time()-t0, flush=True)
    run_luh()
    print("t",time.time()-t0, flush=True)
