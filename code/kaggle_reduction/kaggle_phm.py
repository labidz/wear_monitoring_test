import glob, os, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import resample_poly, welch
from scipy.stats import kurtosis, skew
FS, N_RPM, Z = 50_000.0, 10_400.0, 3
F_TOOTH = N_RPM / 60.0 * Z
NAMES = ["Fx", "Fy", "Fz", "Vx", "Vy", "Vz", "AE"]
OUT = "/kaggle/working"


def chf(x, fs, ft, name):
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


def one(args):
    tool, p = args
    x = pd.read_csv(p, header=None).values.astype(float)
    a, b = steady(x)
    seg = x[a:b]
    r = {"tool": tool, "cut": int(os.path.basename(p).split("_")[-1].split(".")[0]), "n_samples": len(x),
         "seg_a": a, "seg_b": b, "S": float(np.mean(np.hypot(seg[:, 0], seg[:, 1])))}
    for c, n in enumerate(NAMES):
        r.update(chf(seg[:, c], FS, F_TOOTH, n))
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
        for i, (r, w) in enumerate(pool.imap(one, jobs, chunksize=4)):
            r["win_idx"] = i
            rows.append(r)
            wins.append(w)
            if i % 100 == 0:
                print("phm", i, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/phm2010_reduced.csv", index=False)
    np.savez_compressed(f"{OUT}/phm2010_windows.npz", X=np.stack(wins))
    pd.concat(wear).to_csv(f"{OUT}/phm2010_wear.csv", index=False)
    print("PHM DONE", len(rows), flush=True)


run_phm()
