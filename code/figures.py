"""Figures for the manuscript (print-friendly, colour-blind-checked palette, marker shapes as secondary encoding)."""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, _REPO + "/code")
from models import Physics, PhysicsResidual  # noqa: E402

ROOT = _REPO + ""
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated categorical slots 1-4
MK = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.2, "savefig.dpi": 600, "axes.spines.top": False,
                     "axes.spines.right": False})


def tab(ds):
    return pd.read_csv(f"{ROOT}/work/tab_{ds}.csv")


def preds(ds):
    return pd.concat([pd.read_csv(f) for f in [f for f in glob.glob(f"{ROOT}/results/{ds}/*_preds.csv") if "partial" not in f]])


def fig_collapse(datasets, out):
    """Wear versus raw load rise r (top) and versus the mechanics-normalised alpha*lambda (bottom)."""
    fig, ax = plt.subplots(2, len(datasets), figsize=(7.1, 3.6), squeeze=False)
    for j, (ds, label, groupname) in enumerate(datasets):
        t = tab(ds)
        L = t[t.VB.notna()]
        ph = Physics().fit(t)
        lam = ph.lam(L) * np.array([ph.alpha[m] for m in L.mat])
        lam = pd.Series(lam, index=L.index)
        if ds == "NASA":     # colour = work material, marker = (a_p, f) combination; never cycle hues
            mats = sorted(L.mat.unique())
            combos = sorted({(a, round(f * 6, 2)) for a, f in zip(L.ap, L.fz)})
            for (m, a, f), G in L.groupby([L.mat, L.ap, (L.fz * 6).round(2)]):
                kc, km = mats.index(m), combos.index((a, f))
                lab = f"{m.replace('_', ' ')}, $a_p$={a}, $f$={f}"
                ax[0, j].scatter(G.r, G.VB, s=7, marker=MK[km], color=C[kc], alpha=0.8, lw=0, label=lab)
                ax[1, j].scatter(lam[G.index], G.VB, s=7, marker=MK[km], color=C[kc], alpha=0.8, lw=0)
        else:
            for g_i, (g, G) in enumerate(L.groupby("group")):
                ax[0, j].scatter(G.r, G.VB, s=5, marker=MK[g_i], color=C[g_i], alpha=0.6, lw=0, label=g)
                ax[1, j].scatter(lam[G.index], G.VB, s=5, marker=MK[g_i], color=C[g_i], alpha=0.6, lw=0)
        lim = max(L.VB.max(), np.nanmax(lam)) * 1.05
        ax[1, j].plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
        ax[0, j].set_title(label, color=INK)
        ax[0, j].set_xlabel("Load rise $r$ (–)")
        ax[1, j].set_xlabel(r"$\alpha\lambda$ (mm)")
        for a in ax[:, j]:
            a.set_ylabel("Measured $VB$ (mm)")
            a.grid(color=GRID, lw=0.5)
        ax[0, j].legend(title=groupname, frameon=False, ncol=2, markerscale=1.5, handletextpad=0.2,
                        columnspacing=0.6, title_fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)


def fig_trajectories(panels, out, keys=("B1", "P2")):
    """Held-out wear trajectories with group-conformal 90 % bands for one tool per dataset."""
    fig, ax = plt.subplots(1, len(panels), figsize=(7.1, 1.9), squeeze=False)
    for j, (ds, tool, title, lim) in enumerate(panels):
        P = preds(ds)
        for i, k in enumerate(keys):
            D = P[(P.model == k) & (P.tool == tool)].sort_values("pass")
            if i == len(keys) - 1:
                ax[0, j].fill_between(D.time, D.lo, D.hi, color=C[0], alpha=0.18, lw=0, label=f"{k} 90 % band")
            ax[0, j].plot(D.time, D.yhat, color=C[i if i < 1 else 0] if k == "P2" else C[1], lw=1.0, label=k)
        lab = P[(P.model == keys[-1]) & (P.tool == tool) & P.y.notna()].sort_values("pass")
        ax[0, j].scatter(lab.time, lab.y, s=6, color=INK, lw=0, zorder=5, label="measured VB")
        ax[0, j].axhline(lim, color=MUTED, lw=0.7, ls=":")
        ax[0, j].set_title(title, color=INK)
        ax[0, j].set_xlabel("Cutting time (min)")
        ax[0, j].set_ylabel("$VB$ (mm)")
        ax[0, j].grid(color=GRID, lw=0.5)
    ax[0, 0].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "collapse_nasa"
    if which == "collapse_nasa":
        fig_collapse([("NASA", "NASA Ames (development)", "Setting")], f"{ROOT}/manuscript/figs/fig_collapse_nasa.pdf")
