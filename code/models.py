"""Wear estimators: baselines B0-B4, physics P1 and hybrid P2.

Every model exposes  fit(df, X)  and  predict(df, X)  where df is the per-pass table
(only rows with VB are used for fitting) and X the raw-window array indexed by df.win_idx.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from physics import geometry_factor

import os as _os
torch.set_num_threads(int(_os.environ.get("TORCH_THREADS", "2")))   # 1 when several jobs share the CPU
COND = ["fz", "ap", "h_m", "vc"]


def materials(df):
    return sorted(df["mat"].unique())


def extra_cond(df):
    """Further conditions known before cutting (protocol v3: hardness, holder length); absent elsewhere."""
    return [c for c in df.columns if c.startswith("c_")]


def cond_matrix(df, mats):
    c = df[COND + extra_cond(df)].values.astype(float)
    m = np.stack([(df["mat"] == k).values.astype(float) for k in mats], 1) if mats else np.zeros((len(df), 0))
    return np.hstack([c, m])


def sensor_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


def normalised_features(df):
    """Per-tool ratio of each sensor feature to the tool's first pass (log ratio)."""
    out = {}
    for c in sensor_cols(df):
        ref = df.groupby("tool")[c].transform("first")
        x, rf = df[c].values, ref.values
        pos = (x > 0) & (rf > 0)
        out["n_" + c] = np.where(pos, np.log(np.where(pos, x, 1) / np.where(pos, rf, 1)), x - rf)
    return pd.DataFrame(out, index=df.index)


def running_max(df, yhat):
    s = pd.Series(yhat, index=df.index)
    order = df.sort_values(["tool", "pass"]).index
    return s.loc[order].groupby(df.loc[order, "tool"]).cummax().reindex(df.index).values


# ----------------------------------------------------------------------------- baselines
class TimeOnly:
    name = "B0 time-only"
    grid = [{}]

    def __init__(self, **kw):
        self.m = IsotonicRegression(increasing=True, out_of_bounds="clip")

    def fit(self, df, X=None):
        L = df[df.VB.notna()]
        self.m.fit(L.time.values, L.VB.values)
        return self

    def predict(self, df, X=None):
        return self.m.predict(df.time.values)


class Process:
    """Time + cutting conditions. 'gbm' or 'taylor' (log-linear wear law)."""
    name = "B1 process"
    grid = [{"kind": "taylor"}, {"kind": "gbm", "depth": 2}, {"kind": "gbm", "depth": 3}]

    def __init__(self, kind="gbm", depth=2):
        self.kind, self.depth = kind, depth

    def _x(self, df):
        if self.kind == "taylor":
            c = np.log(df[["fz", "ap", "vc"]].values)
            m = np.stack([(df["mat"] == k).values for k in self.mats], 1).astype(float)
            e = df[extra_cond(df)].values.astype(float)
            return np.hstack([np.log(df.time.values + 1e-3)[:, None], c, m, e])
        return np.hstack([df.time.values[:, None], cond_matrix(df, self.mats)])

    def fit(self, df, X=None):
        self.mats = materials(df)
        L = df[df.VB.notna()]
        if self.kind == "taylor":
            self.m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            self.m.fit(self._x(L), np.log(L.VB.values + 0.02))
        else:
            self.m = GradientBoostingRegressor(n_estimators=300, max_depth=self.depth, learning_rate=0.05,
                                               subsample=0.8, random_state=0)
            self.m.fit(self._x(L), L.VB.values)
        return self

    def predict(self, df, X=None):
        p = self.m.predict(self._x(df))
        return np.clip(np.exp(p) - 0.02, 0, None) if self.kind == "taylor" else np.clip(p, 0, None)


class SensorML:
    """Per-pass sensor statistics + time + conditions; optional per-tool normalisation (B3)."""
    grid = [{"learner": "gbm"}, {"learner": "rf"}, {"learner": "mlp"}]

    def __init__(self, learner="gbm", normalise=False):
        self.learner, self.normalise = learner, normalise
        self.name = "B3 normalised-feature ML" if normalise else "B2 sensor-feature ML"

    def _x(self, df):
        f = normalised_features(df).values if self.normalise else df[sensor_cols(df)].values
        return np.hstack([f, df.time.values[:, None], cond_matrix(df, self.mats)])

    def fit(self, df, X=None):
        self.mats = materials(df)
        L = df[df.VB.notna()]
        if self.learner == "gbm":
            m = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                                          random_state=0)
        elif self.learner == "rf":
            m = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=2)
        else:
            m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=3000,
                                                            early_stopping=False, random_state=0))
        self.m = m.fit(self._x(L), L.VB.values)
        return self

    def predict(self, df, X=None):
        return np.clip(self.m.predict(self._x(df)), 0, None)


class _CNN(nn.Module):
    def __init__(self, c_in, n_side):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(c_in, 16, 9, stride=2, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.h = nn.Sequential(nn.Linear(32 + n_side, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, s):
        return nn.functional.softplus(self.h(torch.cat([self.f(x).squeeze(-1), s], 1))).squeeze(-1)


class DeepCNN:
    """B4: end-to-end 1D-CNN on raw signal windows + time + conditions; 5-seed ensemble."""
    name = "B4 end-to-end 1D-CNN"
    grid = [{"epochs": 60}, {"epochs": 150}]

    def __init__(self, epochs=60, seeds=(0, 1, 2, 3, 4), lr=3e-3):
        self.epochs, self.seeds, self.lr = epochs, seeds, lr

    def _prep(self, df, X):
        x = X[df.win_idx.values]
        x = (x - self.xm) / self.xs
        s = (np.hstack([df.time.values[:, None], cond_matrix(df, self.mats)]) - self.sm) / self.ss
        return torch.tensor(x, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)

    def fit(self, df, X):
        self.mats = materials(df)
        L = df[df.VB.notna()]
        xl = X[L.win_idx.values]
        self.xm = xl.mean((0, 2), keepdims=True)[0]
        self.xs = xl.std((0, 2), keepdims=True)[0] + 1e-6
        s = np.hstack([L.time.values[:, None], cond_matrix(L, self.mats)])
        self.sm, self.ss = s.mean(0), s.std(0) + 1e-6
        xt, st = self._prep(L, X)
        y = torch.tensor(L.VB.values, dtype=torch.float32)
        self.nets = []
        for sd in self.seeds:
            torch.manual_seed(sd)
            net = _CNN(xt.shape[1], st.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-3)
            n = len(y)
            for ep in range(self.epochs):
                net.train()
                perm = torch.randperm(n)
                for i in range(0, n, 32):
                    b = perm[i:i + 32]
                    if len(b) < 2:
                        continue
                    opt.zero_grad()
                    loss = ((net(xt[b], st[b]) - y[b]) ** 2).mean()
                    loss.backward()
                    opt.step()
            net.eval()
            self.nets.append(net)
        return self

    def predict(self, df, X, per_seed=False):
        xt, st = self._prep(df, X)
        with torch.no_grad():
            p = np.stack([net(xt, st).numpy() for net in self.nets])
        return p if per_seed else p.mean(0)


# ----------------------------------------------------------------------------- proposed
class Physics:
    """P1: VB = alpha_m * r * (h_m + h0) * ap / (ap + bw); constants by least squares on training data."""
    name = "P1 physics-only"
    grid = [{}]

    def fit(self, df, X=None):
        self.mats = materials(df)
        L = df[df.VB.notna()]
        mi = np.array([self.mats.index(m) for m in L.mat])
        k = len(self.mats)

        def res(p):
            a, h0, bw = np.exp(p[:k]), np.exp(p[k]), np.exp(p[k + 1])
            return a[mi] * L.r.values * geometry_factor(L, h0, bw) - L.VB.values

        p0 = np.r_[np.zeros(k) + np.log(5.0), np.log(0.05), np.log(0.5)]
        lb = np.r_[np.full(k, -10.0), np.log(1e-3), np.log(1e-3)]
        ub = np.r_[np.full(k, 10.0), np.log(1.0), np.log(20.0)]
        sol = least_squares(res, p0, bounds=(lb, ub), loss="soft_l1", f_scale=0.05)
        self.alpha = dict(zip(self.mats, np.exp(sol.x[:k])))
        self.h0, self.bw = float(np.exp(sol.x[k])), float(np.exp(sol.x[k + 1]))
        return self

    def lam(self, df):
        return df.r.values * geometry_factor(df, self.h0, self.bw)

    def predict(self, df, X=None):
        a_default = np.median(list(self.alpha.values()))
        a = np.array([self.alpha.get(m, a_default) for m in df.mat])
        return np.clip(a * self.lam(df), 0, None)


class _Corr(nn.Module):
    """Bounded multiplicative correction exp(kappa * tanh(g(z))) in [1/2, 2] for kappa = log 2."""

    def __init__(self, n_in, h, kappa=np.log(2.0)):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(n_in, h), nn.Tanh(), nn.Linear(h, 1))
        nn.init.zeros_(self.g[2].weight)
        nn.init.zeros_(self.g[2].bias)
        self.kappa = kappa

    def forward(self, z):
        return torch.exp(self.kappa * torch.tanh(self.g(z))).squeeze(-1)


class PhysicsResidual:
    """P2: physics-anchored network  VB = alpha_m * lambda * exp(kappa * tanh(g(z))).

    z = cutting conditions, material and (optionally) normalised cutting time.  The physics term fixes
    the dependence on the measured load rise, so the estimate extrapolates linearly in lambda; the network
    can only rescale it within [1/2, 2].  The estimate is therefore strictly increasing in lambda, and a
    running maximum along each tool's passes enforces irreversibility of wear.
    """
    name = "P2 physics-anchored hybrid (proposed)"
    grid = [{"h": 8, "wd": 1e-2}, {"h": 16, "wd": 1e-2}, {"h": 16, "wd": 1e-1}]

    def __init__(self, h=8, wd=1e-2, epochs=500, seeds=(0, 1, 2, 3, 4), use_time=True, use_geom=True,
                 use_norm=True, monotone=True):
        self.h, self.wd, self.epochs, self.seeds = h, wd, epochs, seeds
        self.use_time, self.use_geom, self.use_norm, self.monotone = use_time, use_geom, use_norm, monotone

    def _z(self, df):
        z = cond_matrix(df, self.mats)
        if self.use_time:
            z = np.hstack([z, (df.time.values / self.t_ref)[:, None]])
        return z

    def _base(self, df):
        if not self.use_norm:   # ablation: raw load level scaled by one constant per material
            a = np.array([self.raw_a.get(m, np.median(list(self.raw_a.values()))) for m in df.mat])
            return a * df.S.values
        if not self.use_geom:   # ablation: ratio without the mechanics term
            a = np.array([self.r_a.get(m, np.median(list(self.r_a.values()))) for m in df.mat])
            return a * np.clip(df.r.values, 0, None)
        return self.phys.predict(df)

    def fit(self, df, X=None):
        self.mats = materials(df)
        L = df[df.VB.notna()]
        self.t_ref = float(L.time.max())
        self.phys = Physics().fit(df)
        rr = np.clip(L.r.values, 0, None)
        self.r_a = {m: float((rr[L.mat == m] * L.VB[L.mat == m]).sum() / max((rr[L.mat == m] ** 2).sum(), 1e-9))
                    for m in self.mats}
        self.raw_a = {m: float((L.S[L.mat == m] * L.VB[L.mat == m]).sum() / (L.S[L.mat == m] ** 2).sum())
                      for m in self.mats}
        z = self._z(L)
        self.zm, self.zs = z.mean(0), z.std(0) + 1e-6
        zt = torch.tensor((z - self.zm) / self.zs, dtype=torch.float32)
        b = torch.tensor(self._base(L), dtype=torch.float32)
        y = torch.tensor(L.VB.values, dtype=torch.float32)
        self.nets = []
        for sd in self.seeds:
            torch.manual_seed(sd)
            net = _Corr(zt.shape[1], self.h)
            opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=self.wd)
            for ep in range(self.epochs):
                opt.zero_grad()
                loss = ((b * net(zt) - y) ** 2).mean()
                loss.backward()
                opt.step()
            self.nets.append(net)
        return self

    def predict(self, df, X=None, per_seed=False):
        zt = torch.tensor((self._z(df) - self.zm) / self.zs, dtype=torch.float32)
        b = self._base(df)
        with torch.no_grad():
            p = np.stack([b * net(zt).numpy() for net in self.nets])
        p = np.clip(p, 0, None)
        if self.monotone:
            p = np.stack([running_max(df, q) for q in p])
        return p if per_seed else p.mean(0)


class _MonoNet(nn.Module):
    """Partially monotone MLP: non-decreasing in the monotone inputs for any condition vector."""

    def __init__(self, n_mono, n_free, h, monotone=True):
        super().__init__()
        self.wm = nn.Parameter(torch.randn(n_mono, h) * 0.3)
        self.wf = nn.Linear(n_free, h)
        self.wo = nn.Parameter(torch.randn(h, 1) * 0.3)
        self.bo = nn.Parameter(torch.zeros(1))
        self.pos = nn.functional.softplus if monotone else (lambda w: w)

    def forward(self, xm, xf):
        hdn = torch.tanh(xm @ self.pos(self.wm) + self.wf(xf))
        return nn.functional.softplus(hdn @ self.pos(self.wo) + self.bo).squeeze(-1)


class Hybrid:
    """P2: monotone neural mapping of (lambda, r, time) conditioned on cutting conditions; 5-seed ensemble."""
    name = "P2 hybrid (proposed)"
    grid = [{"h": 16, "wd": 1e-3}, {"h": 32, "wd": 1e-3}, {"h": 16, "wd": 1e-2}]

    def __init__(self, h=16, wd=1e-3, epochs=600, seeds=(0, 1, 2, 3, 4), use_time=True, use_geom=True,
                 use_norm=True, monotone=True):
        self.h, self.wd, self.epochs, self.seeds = h, wd, epochs, seeds
        self.use_time, self.use_geom, self.use_norm, self.monotone = use_time, use_geom, use_norm, monotone

    def _inputs(self, df):
        if self.use_norm:
            lam = self.phys.lam(df) if self.use_geom else df.r.values
            mono = [lam, df.r.values]
        else:  # ablation: raw (non-normalised) load level instead of the load-rise ratio
            mono = [df.S.values, df.S.values]
        if self.use_time:
            mono.append(df.time.values)
        xm = np.stack(mono, 1)
        xf = cond_matrix(df, self.mats)
        return xm, xf

    def fit(self, df, X=None):
        self.mats = materials(df)
        self.phys = Physics().fit(df)
        L = df[df.VB.notna()]
        xm, xf = self._inputs(L)
        self.ms = np.abs(xm).max(0) + 1e-6          # positive scaling keeps monotone direction
        self.fm, self.fs = xf.mean(0), xf.std(0) + 1e-6
        xm_t = torch.tensor(xm / self.ms, dtype=torch.float32)
        xf_t = torch.tensor((xf - self.fm) / self.fs, dtype=torch.float32)
        y = torch.tensor(L.VB.values, dtype=torch.float32)
        self.nets = []
        for sd in self.seeds:
            torch.manual_seed(sd)
            net = _MonoNet(xm_t.shape[1], xf_t.shape[1], self.h, self.monotone)
            opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=self.wd)
            for ep in range(self.epochs):
                opt.zero_grad()
                loss = ((net(xm_t, xf_t) - y) ** 2).mean()
                loss.backward()
                opt.step()
            self.nets.append(net)
        return self

    def predict(self, df, X=None, per_seed=False, post=True):
        xm, xf = self._inputs(df)
        xm_t = torch.tensor(xm / self.ms, dtype=torch.float32)
        xf_t = torch.tensor((xf - self.fm) / self.fs, dtype=torch.float32)
        with torch.no_grad():
            p = np.stack([net(xm_t, xf_t).numpy() for net in self.nets])
        if post and self.monotone:
            p = np.stack([running_max(df, q) for q in p])
        return p if per_seed else p.mean(0)


class ClockLoadFusion:
    """P3 (protocol v2): log-linear pool of a wear-clock model (B1, time + conditions) and the mechanics law (P1).

        log(VB + eps) = w log(VB_clock + eps) + (1 - w) log(VB_load + eps),   eps = 0.01 mm

    The weight w is chosen by leave-one-group-out on the training pool (grid step 0.05), so it reflects how well
    each source transfers to an unseen group. The clock model's own form (Taylor law or boosting) is chosen in the
    same inner loop. A running maximum along each tool's passes enforces irreversibility of wear.
    """
    name = "P3 clock-load fusion"
    grid = [{}]
    EPS = 0.01

    def __init__(self, monotone=True, w=None):
        self.monotone, self.w_fixed = monotone, w

    @staticmethod
    def _oof(df, make):
        p = pd.Series(np.nan, index=df.index)
        for g in df.group.unique():
            tr, va = df[df.group != g], df[df.group == g]
            if tr.VB.notna().sum() >= 3:
                p.loc[va.index] = make().fit(tr).predict(va)
        return p

    def fit(self, df, X=None):
        L = df.VB.notna()
        # clock model: choose its form by inner leave-one-group-out
        best = None
        for g in Process.grid:
            p = self._oof(df, lambda g=g: Process(**g))
            ok = L & p.notna()
            s = np.sqrt(np.mean((p[ok] - df.VB[ok]) ** 2)) if ok.any() else np.inf
            if best is None or s < best[0]:
                best = (s, g, p)
        self.clock_hp, pc = best[1], best[2]
        pl = self._oof(df, Physics)
        if self.w_fixed is None:
            ok = L & pc.notna() & pl.notna()
            ws = np.round(np.arange(0, 1.0001, 0.05), 2)
            err = [np.sqrt(np.mean((self._pool(w, pc[ok].values, pl[ok].values) - df.VB[ok].values) ** 2))
                   for w in ws]
            self.w = float(ws[int(np.argmin(err))]) if ok.sum() > 0 else 0.5
        else:
            self.w = self.w_fixed
        self.clock = Process(**self.clock_hp).fit(df)
        self.load = Physics().fit(df)
        # out-of-group pooled predictions on the training pool (used for conformal calibration, amendment v2-2)
        oof = pd.Series(np.clip(self._pool(self.w, pc.values, pl.values), 0, None), index=df.index)
        self.oof_ = oof.where(pc.notna() & pl.notna())
        return self

    def _pool(self, w, a, b):
        e = self.EPS
        return np.exp(w * np.log(np.clip(a, 0, None) + e) + (1 - w) * np.log(np.clip(b, 0, None) + e)) - e

    def predict(self, df, X=None):
        p = np.clip(self._pool(self.w, self.clock.predict(df), self.load.predict(df)), 0, None)
        return running_max(df, p) if self.monotone else p
