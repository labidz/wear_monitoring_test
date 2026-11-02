"""Leave-one-group-out evaluation with nested model selection, group-conformal intervals and
tool-change decisions.  Usage:  python evaluate.py <DATASET> [model keys ...]
Reads  work/tab_<DATASET>.csv  (+ work/win_<DATASET>.npz)  and writes  results/<DATASET>_*.csv
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import json
import sys
import time as _time
import numpy as np
import pandas as pd
from models import TimeOnly, Process, SensorML, DeepCNN, Physics, Hybrid, PhysicsResidual, ClockLoadFusion

ROOT = _REPO + ""
ALPHA = 0.10
VB_LIM = {"NASA": 0.30, "NUAA": 0.20, "PHM2010": 0.15, "DENKENA": 0.15, "NONASTREDA": 0.110}

MODELS = {
    "B0": (TimeOnly, {}),
    "B1": (Process, {}),
    "B2": (SensorML, {"normalise": False}),
    "B3": (SensorML, {"normalise": True}),
    "B4": (DeepCNN, {}),
    "P1": (Physics, {}),
    "P2": (PhysicsResidual, {}),
    # ablations of P2
    "A_noTime": (PhysicsResidual, {"use_time": False}),
    "A_noGeom": (PhysicsResidual, {"use_geom": False}),
    "A_noNorm": (PhysicsResidual, {"use_norm": False}),
    "A_noMono": (PhysicsResidual, {"monotone": False}),
    # originally pre-registered P2 (free partially monotone network); replaced by amendment 1
    "P2orig": (Hybrid, {}),
    # protocol v2
    "P3": (ClockLoadFusion, {}),
    "A3_noMono": (ClockLoadFusion, {"monotone": False}),
}
NN_KEYS = {"B4", "P2", "A_noTime", "A_noGeom", "A_noNorm", "A_noMono", "P2orig"}


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def oof(cls, kw, df, X, groups):
    """Out-of-group predictions on df (every group predicted by a model trained on the others)."""
    pred = pd.Series(np.nan, index=df.index)
    for g in groups:
        tr, va = df[df.group != g], df[df.group == g]
        if tr.VB.notna().sum() < 3:
            continue
        pred.loc[va.index] = cls(**kw).fit(tr, X).predict(va, X)
    return pred


def select(cls, base, df, X, key):
    """Inner leave-one-group-out selection over the model's grid (single seed for neural models)."""
    groups = df.group.unique()
    L = df.VB.notna()
    if cls.__name__ == "ClockLoadFusion":      # fast path: reuse the model's own inner out-of-group predictions
        m = cls(**base).fit(df, X)
        return {}, m.oof_
    best = None
    for g in cls.grid:
        kw = dict(base, **g)
        if key in NN_KEYS:
            kw["seeds"] = (0,)
        p = oof(cls, kw, df, X, groups)
        ok = L & p.notna()
        score = rmse(df.VB[ok], p[ok]) if ok.any() else np.inf
        if best is None or score < best[0]:
            best = (score, g, p)
    return best[1], best[2]


def conformal_scale(y, p):
    """|residual| ~ c0 + c1 * yhat (non-negative), floor at 5 % of the label range."""
    r = np.abs(y - p)
    A = np.stack([np.ones_like(p), np.clip(p, 0, None)], 1)
    c, *_ = np.linalg.lstsq(A, r, rcond=None)
    c = np.clip(c, 0, None)
    c[0] = max(c[0], 0.05 * (np.max(y) - np.min(y)) / 4)
    return c


def conformal_quantile(scores, alpha=ALPHA):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def run(dataset, keys):
    import os
    os.makedirs(f"{ROOT}/results/{dataset}", exist_ok=True)
    df = pd.read_csv(f"{ROOT}/work/tab_{dataset}.csv")
    try:
        X = np.load(f"{ROOT}/work/win_{dataset}.npz")["X"]
    except FileNotFoundError:
        X = None
    rows, summary, cost = [], [], []
    import os
    for key in keys:
        cls, base = MODELS[key]
        if X is None and key == "B4":
            continue
        # per-fold checkpoint: background jobs are capped at 2 h, so a restarted run resumes after the last
        # completed fold (every fold is deterministic given its training groups, so results are unchanged)
        part_p = f"{ROOT}/results/{dataset}/{key}_partial_preds.csv"
        part_c = f"{ROOT}/results/{dataset}/{key}_partial_cost.csv"
        done = set()
        if os.path.exists(part_p) and os.path.exists(part_c):
            pc = pd.read_csv(part_c)
            done = set(pc.fold.astype(str))
            pp = pd.read_csv(part_p)
            pp = pp[pp.fold.astype(str).isin(done)]
            rows.append(pp)
            cost.extend(pc.to_dict("records"))
            print(f"{dataset} {key}: resuming, {len(done)} folds done", flush=True)
        for g in df.group.unique():
            if str(g) in done:
                continue
            tr, te = df[df.group != g].copy(), df[df.group == g].copy()
            t0 = _time.time()
            hp, cal = select(cls, base, tr, X, key)
            kw = dict(base, **hp)
            # calibration scores: out-of-group predictions of the selected configuration on the training pool,
            # reused from the inner selection loop (single seed for neural models; amendment 3)
            m = cls(**kw).fit(tr, X)
            t_fit = _time.time() - t0
            t1 = _time.time()
            p = m.predict(te, X)
            t_pred = (_time.time() - t1) / max(len(te), 1)
            ok = tr.VB.notna() & cal.notna()
            c = conformal_scale(tr.VB[ok].values, cal[ok].values)
            q = conformal_quantile(np.abs(tr.VB[ok].values - cal[ok].values) / (c[0] + c[1] * np.clip(cal[ok].values, 0, None)))
            half = q * (c[0] + c[1] * np.clip(p, 0, None))
            # naive comparator: split conformal on a random 30 % of training passes (ignores groups)
            rng = np.random.default_rng(0)
            Lidx = tr.index[tr.VB.notna()]
            calidx = rng.choice(Lidx, size=max(int(0.3 * len(Lidx)), 5), replace=False)
            m_naive = cls(**dict(kw, **({"seeds": (0,)} if key in NN_KEYS else {}))).fit(tr.drop(calidx), X)
            pn = m_naive.predict(tr.loc[calidx], X)
            qn = conformal_quantile(np.abs(tr.VB.loc[calidx].values - pn))
            per_seed = None
            if key in NN_KEYS:
                per_seed = m.predict(te, X, per_seed=True)
            out = pd.DataFrame({"model": key, "fold": g, "tool": te.tool.values, "pass": te["pass"].values,
                                "time": te.time.values, "y": te.VB.values, "yhat": p,
                                "lo": np.clip(p - half, 0, None), "hi": p + half,
                                "lo_naive": np.clip(p - qn, 0, None), "hi_naive": p + qn})
            if per_seed is not None:
                for s in range(per_seed.shape[0]):
                    out[f"seed{s}"] = per_seed[s]
            rows.append(out)
            cost.append({"model": key, "fold": g, "fit_s": t_fit, "pred_ms_per_pass": 1e3 * t_pred, "hp": json.dumps(hp),
                         "inner_rmse": rmse(tr.VB[ok], cal[ok])})   # selects the best baseline (protocol v1, sec. 5)
            out.to_csv(part_p, mode="a", header=not os.path.exists(part_p), index=False)
            pd.DataFrame([cost[-1]]).to_csv(part_c, mode="a", header=not os.path.exists(part_c), index=False)
            print(f"{dataset} {key} fold {g}: rmse={rmse(te.VB.dropna(), pd.Series(p, te.index)[te.VB.notna()]):.4f} "
                  f"hp={hp} fit={t_fit:.1f}s", flush=True)
        P = pd.concat([r for r in rows if len(r) and r.model.iloc[0] == key])
        P.to_csv(f"{ROOT}/results/{dataset}/{key}_preds.csv", index=False)
        pd.DataFrame([c for c in cost if c["model"] == key]).to_csv(f"{ROOT}/results/{dataset}/{key}_cost.csv",
                                                                     index=False)
        for f in (part_p, part_c):
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    ds = sys.argv[1]
    keys = sys.argv[2:] or list(MODELS)
    run(ds, keys)
