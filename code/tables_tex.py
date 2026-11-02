"""Write manuscript/tables/tab_results.tex from results/<DS>/summary.csv and results/p3_weights.csv.

RMSE per model and dataset (µm; Haas in % of tool life).  Bold: lowest RMSE in the column.
Superscripts mark paired 95 % cluster-bootstrap CIs vs the clock B1 that exclude zero: + better, - worse.
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import os
import pandas as pd

ROOT = _REPO + ""
DS = [("NASA", "NASA$^\\dagger$", 1.0), ("PHM2010", "PHM", 1.0), ("DENKENA", "Hann.", 1.0),
      ("NUAA", "NUAA", 1.0), ("VICOMTECH", "Mondr.", 1.0), ("HAAS", "Haas", 0.1)]
ROWS = [("B0", "B0 time only"), ("B1", "B1 clock"), ("B2", "B2 features"), ("B3", "B3 norm. features"),
        ("B4", "B4 1D-CNN"), ("P1", "P1 load law"), ("P2", "P2 anchored net"), ("P3", "P3 fusion")]


def cell(S, k, scale):
    if k not in S.index:
        return "--"
    s = S.loc[k]
    v = scale * s.rmse_um
    txt = f"{v:.1f}" if scale < 1 else f"{v:.0f}"
    sig = ""
    if k != "B1" and s.d_ci_hi < 0:
        sig = "$^{+}$"
    elif k != "B1" and s.d_ci_lo > 0:
        sig = "$^{-}$"
    return txt, v, sig


def main():
    W = pd.read_csv(f"{ROOT}/results/p3_weights.csv") if os.path.exists(f"{ROOT}/results/p3_weights.csv") else None
    cols = {}
    for ds, _, scale in DS:
        S = pd.read_csv(f"{ROOT}/results/{ds}/summary.csv").set_index("model")
        cols[ds] = {k: cell(S, k, scale) for k, _ in ROWS}
    lines = [r"\begin{table}[t]", r"\caption{Held-out RMSE (\textmu m; Haas: \% of tool life). Bold: lowest in column."
             r" $^{+/-}$: paired 95\,\% CI of the difference to the clock B1 excludes zero (better/worse)."
             r" $^\dagger$development set. Last row: mean fusion weight on the clock.}",
             r"\label{tab:results}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\begin{tabular}{@{}l" + "r" * len(DS) + "@{}}", r"\toprule",
             "Model & " + " & ".join(n for _, n, _ in DS) + r" \\", r"\midrule"]
    for k, name in ROWS:
        vals = []
        for ds, _, _ in DS:
            c = cols[ds][k]
            if c == "--":
                vals.append("--")
                continue
            best = min(v[1] for v in cols[ds].values() if v != "--")
            t = rf"\textbf{{{c[0]}}}" if abs(c[1] - best) < 1e-9 else c[0]
            vals.append(t + c[2])
        lines.append(name + " & " + " & ".join(vals) + r" \\")
    if W is not None:
        lines.append(r"\midrule")
        wv = [f"{W[W.dataset == ds].w.mean():.2f}" if (W.dataset == ds).any() else "--" for ds, _, _ in DS]
        lines.append(r"P3 weight $\bar w$ on clock & " + " & ".join(wv) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(f"{ROOT}/manuscript/tables", exist_ok=True)
    open(f"{ROOT}/manuscript/tables/tab_results.tex", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
