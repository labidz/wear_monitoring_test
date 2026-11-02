"""Metrics, cluster-bootstrap confidence intervals, interval coverage and tool-change decisions.
Usage:  python analyse.py <DATASET>   (reads results/<DATASET>_preds.csv, work/tab_<DATASET>.csv)
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys
import json
import numpy as np
import pandas as pd

ROOT = _REPO + ""
VB_LIM = {"NASA": 0.30, "NUAA": 0.20, "PHM2010": 0.15, "DENKENA": 0.15, "NONASTREDA": 0.110, "VICOMTECH": 0.25, "HAAS": 0.90}
B = 2000


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def cluster_boot(L, models, ref, seed=0):
    """RMSE CI per model and paired RMSE difference vs `ref`, resampling held-out groups."""
    rng = np.random.default_rng(seed)
    folds = L.fold.unique()
    err = {m: {f: (g.y - g.yhat).values for f, g in L[L.model == m].groupby("fold")} for m in models}
    out = {}
    stats = {m: [] for m in models}
    diffs = {m: [] for m in models}
    for _ in range(B):
        s = rng.choice(folds, size=len(folds), replace=True)
        r_ref = rmse(np.concatenate([err[ref][f] for f in s])) if ref in err else np.nan
        for m in models:
            r = rmse(np.concatenate([err[m][f] for f in s]))
            stats[m].append(r)
            diffs[m].append(r - r_ref)
    for m in models:
        e = np.concatenate(list(err[m].values()))
        out[m] = dict(rmse=rmse(e), mae=float(np.mean(np.abs(e))), n=len(e),
                      ci=list(np.percentile(stats[m], [2.5, 97.5])),
                      d_ref=float(rmse(e) - rmse(np.concatenate(list(err[ref].values())))) if ref in err else np.nan,
                      d_ci=list(np.percentile(diffs[m], [2.5, 97.5])))
    return out


def crossing_time(t, y, lim):
    """First time the labelled wear curve reaches lim (linear interpolation); NaN if never."""
    for i in range(1, len(y)):
        if y[i] >= lim > y[i - 1]:
            return t[i - 1] + (lim - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1])
    return t[0] if y[0] >= lim else np.nan


def decisions(P, lim, model):
    """Late-change rate and tool-life utilisation for three policies."""
    D = P[P.model == model]
    res = {k: dict(late=0, n=0, util=[]) for k in ["upper", "point", "fixed"]}
    tools = D.tool.unique()
    # fixed-life policy: median crossing time of the other folds' tools (same model rows hold all tools)
    tc_all = {}
    for tl, g in D.groupby("tool"):
        g = g.sort_values("pass")
        lab = g[g.y.notna()]
        tc_all[tl] = (crossing_time(lab.time.values, lab.y.values, lim), g.fold.iloc[0])
    for tl in tools:
        g = D[D.tool == tl].sort_values("pass")
        tc, fold = tc_all[tl]
        if np.isnan(tc) or tc <= 0:
            continue
        others = [v[0] for k, v in tc_all.items() if v[1] != fold and not np.isnan(v[0])]
        t_fixed = np.median(others) if others else np.inf
        trig = {"upper": g.hi.values >= lim, "point": g.yhat.values >= lim, "fixed": g.time.values >= t_fixed}
        for k, tr in trig.items():
            t_ch = g.time.values[np.argmax(tr)] if tr.any() else np.inf
            res[k]["n"] += 1
            if t_ch > tc:
                res[k]["late"] += 1
            else:
                res[k]["util"].append(t_ch / tc)
    return {k: dict(n=v["n"], late_rate=v["late"] / max(v["n"], 1),
                    utilisation=float(np.mean(v["util"])) if v["util"] else np.nan) for k, v in res.items()}


def main(ds, ref="B1"):
    import glob
    P = pd.concat([pd.read_csv(f) for f in sorted([f for f in glob.glob(f"{ROOT}/results/{ds}/*_preds.csv") if "partial" not in f])])
    L = P[P.y.notna()].copy()
    models = list(dict.fromkeys(L.model))
    boot = cluster_boot(L, models, ref)
    rows = []
    for m in models:
        g = L[L.model == m]
        cov = float(((g.y >= g.lo) & (g.y <= g.hi)).mean())
        covn = float(((g.y >= g.lo_naive) & (g.y <= g.hi_naive)).mean())
        per_fold_cov = g.groupby("fold").apply(lambda h: ((h.y >= h.lo) & (h.y <= h.hi)).mean())
        seeds = [c for c in g.columns if c.startswith("seed") and g[c].notna().all()]
        seed_rmse = [rmse((g.y - g[c]).values) for c in seeds]
        dec = decisions(P, VB_LIM[ds.split("_")[0]], m)
        b = boot[m]
        rows.append(dict(model=m, rmse_um=1e3 * b["rmse"], ci_lo=1e3 * b["ci"][0], ci_hi=1e3 * b["ci"][1],
                         mae_um=1e3 * b["mae"], d_vs_ref_um=1e3 * b["d_ref"], d_ci_lo=1e3 * b["d_ci"][0],
                         d_ci_hi=1e3 * b["d_ci"][1], coverage=cov, min_group_cov=float(per_fold_cov.min()),
                         width_um=1e3 * float((g.hi - g.lo).mean()), naive_coverage=covn,
                         seed_rmse_mean=1e3 * np.mean(seed_rmse) if seed_rmse else np.nan,
                         seed_rmse_sd=1e3 * np.std(seed_rmse) if seed_rmse else np.nan,
                         late_upper=dec["upper"]["late_rate"], util_upper=dec["upper"]["utilisation"],
                         late_point=dec["point"]["late_rate"], util_point=dec["point"]["utilisation"],
                         late_fixed=dec["fixed"]["late_rate"], util_fixed=dec["fixed"]["utilisation"],
                         n_tools_dec=dec["upper"]["n"], n=b["n"]))
    T = pd.DataFrame(rows)
    T.to_csv(f"{ROOT}/results/{ds}/summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    return T


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
