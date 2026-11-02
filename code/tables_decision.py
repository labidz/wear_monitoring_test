"""Table 3: tool-change policies on the datasets with at least nine tools reaching the limit.

Fixed life = median crossing time of the tools in the other folds (today's practice); upper bound = change at the first
pass whose 90 % group-conformal upper bound reaches the limit.  Late = changed after the true crossing.
Utilisation = change time / true crossing time, averaged over the tools changed in time.
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import pandas as pd

ROOT = _REPO + ""
DS = [("NASA", "NASA$^\\dagger$"), ("VICOMTECH", "Mondragon"), ("HAAS", "Haas")]
POL = [("fixed", "B1", "Fixed tool life"), ("upper", "B1", "Clock B1, upper bound"),
       ("upper", "P3", "Fusion P3, upper bound"), ("upper", "B2", "Features B2, upper bound"),
       ("upper", "B3", "Norm. features B3, upper bound")]


def main():
    S = {d: pd.read_csv(f"{ROOT}/results/{d}/summary.csv").set_index("model") for d, _ in DS}
    lines = [r"\begin{table}[t]",
             r"\caption{Tool-change policies: tools changed late (after the true limit crossing) and mean tool-life "
             r"utilisation of the others. Limits: $VB$ 0.30 and 0.25\,mm; 90\,\% of life (Haas).}",
             r"\label{tab:decision}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}l" + "cc" * len(DS) + "@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for _, n in DS) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DS))),
             "Policy & " + " & ".join(["late", "util."] * len(DS)) + r" \\", r"\midrule"]
    for pol, model, name in POL:
        cells = []
        for d, _ in DS:
            s = S[d].loc[model]
            n = int(s.n_tools_dec)
            late = round(s[f"late_{pol}"] * n)
            cells += [f"{late}/{n}", f"{100 * s[f'util_{pol}']:.0f}\\,\\%"]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{ROOT}/manuscript/tables/tab_decision.tex", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
