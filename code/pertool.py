"""Post-hoc explanatory analysis: per held-out tool, physics advantage vs. deviation of the tool's wear rate
from the wear rate of the training tools (the quantity a clock cannot know)."""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import numpy as np, pandas as pd
ROOT = _REPO + ""
rows = []
for ds in ["NASA", "PHM2010", "DENKENA", "NUAA", "VICOMTECH"]:
    t = pd.read_csv(f"{ROOT}/work/tab_{ds}.csv")
    L = t[t.VB.notna()]
    # mean wear rate per tool (mm/min) from first to last labelled pass
    rate = L.sort_values("pass").groupby("tool").apply(
        lambda g: (g.VB.iloc[-1] - g.VB.iloc[0]) / max(g.time.iloc[-1] - g.time.iloc[0], 1e-9))
    try:
        pc = pd.read_csv(f"{ROOT}/results/{ds}/B1_preds.csv")
        pp = pd.read_csv(f"{ROOT}/results/{ds}/P1_preds.csv")
    except FileNotFoundError:
        continue
    for tool in rate.index:
        a, b = pc[(pc.tool == tool) & pc.y.notna()], pp[(pp.tool == tool) & pp.y.notna()]
        if len(a) == 0:
            continue
        fold = a.fold.iloc[0]
        train_tools = [k for k in rate.index if k not in set(L[L.group == fold].tool)]
        dev = np.log(rate[tool] / np.median(rate[train_tools]))
        rows.append(dict(dataset=ds, tool=tool, abs_log_rate_dev=abs(dev), rmse_clock=np.sqrt(np.mean((a.y - a.yhat) ** 2)),
                         rmse_phys=np.sqrt(np.mean((b.y - b.yhat) ** 2)), vb_range=a.y.max() - a.y.min()))
R = pd.DataFrame(rows)
R["rel_adv"] = (R.rmse_clock - R.rmse_phys) / R.vb_range.clip(lower=1e-3)
R.to_csv(f"{ROOT}/results/pertool_explanatory.csv", index=False)
from scipy.stats import spearmanr
print(R.groupby("dataset").agg(n=("tool", "size"), dev=("abs_log_rate_dev", "median"), adv=("rel_adv", "median")).round(3))
ok = R.dataset != "NUAA"
print("Spearman (all but NUAA):", spearmanr(R[ok].abs_log_rate_dev, R[ok].rel_adv))
print("Spearman (all):", spearmanr(R.abs_log_rate_dev, R.rel_adv))
