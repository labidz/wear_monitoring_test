"""Label-only life scatter: pooled within-condition SD of log time to reach a common wear level (no model involved).

w* = min over tools of the tool's maximum label (so every tool reaches it), capped at VB_lim; the time to reach w*
is linearly interpolated along each tool's passes.  Conditions with one tool contribute nothing.
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys
import numpy as np
import pandas as pd

LIM = {"NASA": 0.30, "PHM2010": 0.15, "DENKENA": 0.15, "NUAA": 0.20, "VICOMTECH": 0.25, "HAAS": 0.90}


def t_reach(g, w):
    g = g.sort_values("pass")
    v = np.maximum.accumulate(g.VB.values)
    t = g.time.values
    k = np.argmax(v >= w)
    if k == 0:
        return t[0]
    return t[k - 1] + (w - v[k - 1]) / max(v[k] - v[k - 1], 1e-12) * (t[k] - t[k - 1])


def scatter(ds):
    t = pd.read_csv(f"{_REPO}/work/tab_{ds}.csv")
    t = t[t.VB.notna()]
    cond_cols = [c for c in ["mat", "fz", "ap", "ae", "vc", "c_holder"] if c in t.columns]
    w = min(t.groupby("tool").VB.max().min(), LIM[ds])
    rows = []
    for tool, g in t.groupby("tool"):
        key = tuple(str(x) for x in g[cond_cols].iloc[0].tolist())
        rows.append(dict(tool=tool, cond=key, t=t_reach(g, w)))
    r = pd.DataFrame(rows)
    r["logt"] = np.log(r.t)
    ss, dof = 0.0, 0
    for _, g in r.groupby("cond"):
        if len(g) > 1:
            ss += ((g["logt"] - g["logt"].mean()) ** 2).sum()
            dof += len(g) - 1
    return dict(dataset=ds, w_star=w, tools=len(r), conditions=r.cond.nunique(), dof=dof,
                sigma_logT=np.sqrt(ss / dof) if dof else np.nan)


if __name__ == "__main__":
    out = pd.DataFrame([scatter(d) for d in sys.argv[1:]])
    print(out.round(3).to_string(index=False))
    out.to_csv(_REPO + "/results/life_scatter.csv", index=False)
