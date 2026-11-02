"""Pre-registered contrasts for every dataset (paired cluster bootstrap over held-out groups, B = 2000).

v1 (NASA, PHM2010, DENKENA): P2 vs BB, the best of B1-B4 chosen per outer fold by its inner leave-one-group-out RMSE.
v2/v3 (NUAA, VICOMTECH, HAAS): P3 vs B1 and P3 vs B3 (H2/H3: supported only if both CIs exclude zero).
Secondary everywhere: P1, P2, P3 and BB vs B1.  Ratios (model RMSE / reference RMSE) are added for the forest plot.

    python primary.py NASA PHM2010 DENKENA NUAA VICOMTECH HAAS HAAS_LOCO
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import glob
import os
import sys
import numpy as np
import pandas as pd

ROOT = _REPO + ""
B = 2000
PRIMARY = {"NASA": [("P2", "BB")], "PHM2010": [("P2", "BB")], "DENKENA": [("P2", "BB")],
           "NUAA": [("P3", "B1"), ("P3", "B3")], "VICOMTECH": [("P3", "B1"), ("P3", "B3")],
           "HAAS": [("P3", "B1"), ("P3", "B3")]}
SECONDARY = [("P1", "B1"), ("P2", "B1"), ("P3", "B1"), ("BB", "B1"), ("B2", "B1"), ("B3", "B1")]


def load(ds):
    P = pd.concat([pd.read_csv(f) for f in sorted([f for f in glob.glob(f"{ROOT}/results/{ds}/*_preds.csv") if "partial" not in f])])
    P["fold"] = P.fold.astype(str)
    return P[P.y.notna()]


def best_baseline(ds, L):
    """Per outer fold, the baseline (B1-B4) with the lowest inner-CV RMSE; returns rows relabelled 'BB'."""
    scores = []
    for k in ("B1", "B2", "B3", "B4"):
        f = f"{ROOT}/results/{ds}/{k}_inner.csv"
        if os.path.exists(f) and k in set(L.model):
            s = pd.read_csv(f)
            s["fold"], s["model"] = s.fold.astype(str), k
            scores.append(s)
    if not scores:
        return None, None
    S = pd.concat(scores, ignore_index=True)
    pick = S.loc[S.groupby("fold").inner_rmse.idxmin()]
    choice = dict(zip(pick.fold, pick.model))
    parts = [L[(L.model == m) & (L.fold == f)] for f, m in choice.items()]
    bb = pd.concat(parts).assign(model="BB")
    return bb, choice


def paired(L, m, ref, rng):
    folds = np.array(sorted(L.fold.unique()))
    e = {k: {f: (g.y - g.yhat).values for f, g in L[L.model == k].groupby("fold")} for k in (m, ref)}
    if set(e[m]) != set(e[ref]):
        raise ValueError(f"{m} and {ref} cover different folds")

    def r(k, s):
        return np.sqrt(np.mean(np.concatenate([e[k][f] for f in s]) ** 2))

    d, q = [], []
    for _ in range(B):
        s = rng.choice(folds, size=len(folds), replace=True)
        a, b = r(m, s), r(ref, s)
        d.append(a - b)
        q.append(a / b)
    a, b = r(m, folds), r(ref, folds)
    return dict(rmse=a, rmse_ref=b, diff=a - b, diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5),
                ratio=a / b, ratio_lo=np.percentile(q, 2.5), ratio_hi=np.percentile(q, 97.5))


def main(datasets):
    rows = []
    for ds in datasets:
        L = load(ds)
        bb, choice = best_baseline(ds, L)
        if bb is not None:
            L = pd.concat([L, bb])
        base = ds.split("_")[0] if ds != "HAAS_LOCO" else "HAAS"
        todo = [(m, r, "primary") for m, r in PRIMARY.get(base, [])] + [(m, r, "secondary") for m, r in SECONDARY]
        seen = set()
        for m, ref, kind in todo:
            if (m, ref) in seen or m not in set(L.model) or ref not in set(L.model):
                continue
            seen.add((m, ref))
            res = paired(L, m, ref, np.random.default_rng(0))
            rows.append(dict(dataset=ds, kind=kind if ds != "HAAS_LOCO" else "secondary", model=m, ref=ref, **res,
                             bb_choice=";".join(f"{k}:{v}" for k, v in sorted(choice.items())) if choice and "BB" in (m, ref) else ""))
    T = pd.DataFrame(rows)
    T.to_csv(f"{ROOT}/results/contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.drop(columns="bb_choice").round(4).to_string(index=False))
    return T


if __name__ == "__main__":
    main(sys.argv[1:])
