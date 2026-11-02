"""Cross-dataset figures.

fig_clock_vs_load: label against cutting time (top) and against the held-out load-law estimate P1 (bottom).
fig_forest:        RMSE ratio to the clock model B1 with 95 % cluster-bootstrap CIs (results/contrasts.csv).
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, _REPO + "/code")
from figures import C, MK, INK, MUTED, GRID, ROOT  # noqa: E402  (shared rcParams and validated palette)

NAMES = {"NASA": "NASA Ames (dev.)", "PHM2010": "PHM 2010", "DENKENA": "Hannover", "NUAA": "NUAA",
         "VICOMTECH": "Mondragon", "HAAS": "Haas", "HAAS_LOCO": "Haas (new condition)"}


def _preds(ds, key):
    return pd.read_csv(f"{ROOT}/results/{ds}/{key}_preds.csv")


def _rmse(P):
    P = P[P.y.notna()]
    return float(np.sqrt(np.mean((P.y - P.yhat) ** 2)))


def fig_clock_vs_load(panels, out):
    fig, ax = plt.subplots(2, len(panels), figsize=(7.1, 3.1), squeeze=False)
    for j, (ds, tlabel, ylab, scale, unit) in enumerate(panels):
        t = pd.read_csv(f"{ROOT}/work/tab_{ds}.csv")
        t = t[t.VB.notna()]
        for _, g in t.groupby("tool"):
            g = g.sort_values("pass")
            ax[0, j].plot(g.time, scale * g.VB, color=C[0], lw=0.6, alpha=0.55)
        P1, B1 = _preds(ds, "P1"), _preds(ds, "B1")
        P1 = P1[P1.y.notna()]
        ax[1, j].scatter(scale * P1.yhat, scale * P1.y, s=2.5, color=C[1], lw=0, alpha=0.35, rasterized=True)
        hi = scale * max(P1.y.max(), P1.yhat.quantile(0.995)) * 1.05
        ax[1, j].plot([0, hi], [0, hi], color=MUTED, lw=0.7, ls="--")
        ax[1, j].set_xlim(0, hi)
        ax[1, j].set_ylim(0, scale * P1.y.max() * 1.08)
        ax[0, j].set_title(NAMES[ds], color=INK)
        ax[0, j].set_xlabel(tlabel)
        ax[1, j].set_xlabel(f"Load-law estimate P1 ({unit})")
        for a in ax[:, j]:
            a.set_ylabel(ylab)
            a.grid(color=GRID, lw=0.5)
        ax[0, j].text(0.03, 0.97, f"clock B1: {scale * _rmse(B1):.0f} {unit}", transform=ax[0, j].transAxes,
                      ha="left", va="top", fontsize=6.3, color=INK,
                      bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax[1, j].text(0.03, 0.97, f"load law P1: {scale * _rmse(P1):.0f} {unit}", transform=ax[1, j].transAxes,
                      ha="left", va="top", fontsize=6.3, color=INK,
                      bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    fig.savefig(out, dpi=600)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)


def fig_forest(datasets, out, series=(("P1", "load law P1", 0), ("P3", "fusion P3", 1),
                                      ("B2", "features B2", 2), ("B3", "norm. features B3", 3))):
    T = pd.read_csv(f"{ROOT}/results/contrasts.csv")
    T = T[T.ref == "B1"]
    fig, ax = plt.subplots(figsize=(3.45, 0.42 * len(datasets) + 0.7))
    off = np.linspace(-0.27, 0.27, len(series))
    for i, ds in enumerate(datasets):
        y0 = len(datasets) - 1 - i
        for (k, lab, c), dy in zip(series, off):
            r = T[(T.dataset == ds) & (T.model == k)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.ratio, y0 - dy, xerr=[[r.ratio - r.ratio_lo], [r.ratio_hi - r.ratio]], fmt=MK[c],
                        ms=3.6, color=C[c], mec="white", mew=0.5, elinewidth=1.0, capsize=0,
                        label=lab if i == 0 else None)
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ticks = [0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{x:g}" for x in ticks])
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([NAMES[d] for d in datasets[::-1]])
    ax.set_xlabel("RMSE relative to the clock B1 (log scale)")
    ax.set_xlim(0.28, 7.5)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.2,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "clock":
        fig_clock_vs_load([("NASA", "Cutting time (min)", "$VB$ (µm)", 1e3, "µm"),
                           ("DENKENA", "Cutting time (min)", "$VB$ (µm)", 1e3, "µm"),
                           ("VICOMTECH", "Cutting time (min)", "$VB$ (µm)", 1e3, "µm"),
                           ("HAAS", "Cycles", "Consumed life (%)", 1e2, "%")],
                          f"{ROOT}/manuscript/figs/fig_clock_vs_load.pdf")
    else:
        fig_forest(sys.argv[2:], f"{ROOT}/manuscript/figs/fig_forest.pdf")
