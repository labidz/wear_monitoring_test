"""Backfill the inner leave-one-group-out RMSE of the selected configuration for runs made before evaluate.py
recorded it.  Needed for the pre-registered primary comparison 'P2 vs the best of B1-B4 chosen on inner CV'.

    python inner_scores.py <DATASET> B1 B2 B3 B4
Writes results/<DATASET>/<key>_inner.csv (fold, inner_rmse).  Deterministic models reproduce the original selection;
neural models use seed 0, as in the selection loop.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from evaluate import MODELS, NN_KEYS, ROOT, oof, rmse

ds, keys = sys.argv[1], sys.argv[2:]
df = pd.read_csv(f"{ROOT}/work/tab_{ds}.csv")
try:
    X = np.load(f"{ROOT}/work/win_{ds}.npz")["X"]
except FileNotFoundError:
    X = None
for key in keys:
    cpath = f"{ROOT}/results/{ds}/{key}_cost.csv"
    if not os.path.exists(cpath):
        continue
    cost = pd.read_csv(cpath)
    if "inner_rmse" in cost.columns and cost.inner_rmse.notna().all():
        cost[["fold", "inner_rmse"]].to_csv(f"{ROOT}/results/{ds}/{key}_inner.csv", index=False)
        continue
    cls, base = MODELS[key]
    rows = []
    for _, c in cost.iterrows():
        tr = df[df.group.astype(str) != str(c.fold)]
        kw = dict(base, **json.loads(c.hp))
        if key in NN_KEYS:
            kw["seeds"] = (0,)
        p = oof(cls, kw, tr, X, tr.group.unique())
        ok = tr.VB.notna() & p.notna()
        rows.append(dict(fold=c.fold, inner_rmse=rmse(tr.VB[ok], p[ok])))
        print(ds, key, c.fold, round(rows[-1]["inner_rmse"], 4), flush=True)
    pd.DataFrame(rows).to_csv(f"{ROOT}/results/{ds}/{key}_inner.csv", index=False)
