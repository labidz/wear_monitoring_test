"""Assemble manuscript tables from results/<DS>/summary.csv."""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import pandas as pd

ROOT = _REPO + ""
NAMES = {"B0": "B0 time only", "B1": "B1 process model", "B2": "B2 signal features", "B3": "B3 normalised features",
         "B4": "B4 end-to-end CNN", "P1": "P1 physics law", "P2": "P2 physics-anchored",
         "A_noNorm": "P2 without fresh-tool ratio", "A_noGeom": "P2 without mechanics term",
         "A_noTime": "P2 without time input", "A_noMono": "P2 without running maximum",
         "P2orig": "P2 as pre-registered (free monotone net)"}


def load(ds):
    return pd.read_csv(f"{ROOT}/results/{ds}/summary.csv").set_index("model")


def main_table(dss):
    rows = []
    for k in ["B0", "B1", "B2", "B3", "B4", "P1", "P2"]:
        r = {"model": NAMES[k]}
        for ds in dss:
            S = load(ds)
            if k in S.index:
                s = S.loc[k]
                r[ds] = f"{s.rmse_um:.0f} [{s.ci_lo:.0f}, {s.ci_hi:.0f}]"
        rows.append(r)
    return pd.DataFrame(rows)


def ablation_table(dss):
    rows = []
    for k in ["P2", "A_noNorm", "A_noGeom", "A_noTime", "A_noMono", "P2orig"]:
        r = {"model": NAMES[k]}
        for ds in dss:
            S = load(ds)
            if k in S.index:
                r[ds] = f"{S.loc[k].rmse_um:.0f}"
        rows.append(r)
    return pd.DataFrame(rows)


def decision_table(dss, keys=("B1", "B3", "B4", "P2")):
    rows = []
    for ds in dss:
        S = load(ds)
        for k in keys:
            if k in S.index:
                s = S.loc[k]
                rows.append({"dataset": ds, "model": NAMES[k], "coverage": round(s.coverage, 2),
                             "min_group_cov": round(s.min_group_cov, 2), "width_um": round(s.width_um),
                             "naive_cov": round(s.naive_coverage, 2), "late_upper": round(s.late_upper, 2),
                             "util_upper": round(s.util_upper, 2), "late_point": round(s.late_point, 2),
                             "util_point": round(s.util_point, 2), "n_tools": int(s.n_tools_dec)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    dss = sys.argv[1:]
    pd.set_option("display.width", 250)
    print(main_table(dss).to_string(index=False))
    print()
    print(ablation_table(dss).to_string(index=False))
    print()
    print(decision_table(dss).to_string(index=False))
