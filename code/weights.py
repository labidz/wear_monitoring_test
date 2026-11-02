"""Explanatory diagnostic (not pre-registered): the fusion weight w that P3 chose in each outer fold.

w = 1 means the inner leave-one-group-out loop trusted only the clock, w = 0 only the load law.  The refit is
deterministic (boosting with a fixed seed, least squares), so it reproduces the weights used for the held-out
predictions.      python weights.py NASA PHM2010 DENKENA NUAA VICOMTECH HAAS HAAS_LOCO
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys
import pandas as pd
from models import ClockLoadFusion

ROOT = _REPO + ""
rows = []
for ds in sys.argv[1:]:
    df = pd.read_csv(f"{ROOT}/work/tab_{ds}.csv")
    for g in df.group.unique():
        m = ClockLoadFusion().fit(df[df.group != g])
        rows.append(dict(dataset=ds, fold=g, w=m.w, clock=str(m.clock_hp)))
        print(ds, g, m.w, m.clock_hp, flush=True)
W = pd.DataFrame(rows)
W.to_csv(f"{ROOT}/results/p3_weights.csv", index=False)
print(W.groupby("dataset").w.describe()[["mean", "min", "max"]].round(2).to_string())
