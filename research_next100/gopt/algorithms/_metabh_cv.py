"""Collective variables (CVs) and the well-tempered metadynamics bias used by metabh.

All CVs are energy-agnostic, dimensionless and computed from geometry only:

* lnV                       log(volume per atom) (variable cell only)
* cn_<A><B>                 smooth coordination per species pair, normalised per search atom
                            (sum over pairs = mean coordination number; unlike pairs encode
                            the chemical order of alloys)
* q4, q6                    mean local Steinhardt bond-order parameters
* rg, k2                    radius of gyration / (mean r_cov * n^(1/3)) and the relative shape
                            anisotropy kappa^2 (finite systems only)
* cnsub                     mean number of substrate neighbours per search atom (supported)
* h                         mean height above the top substrate layer / (2 mean r_cov) (surface)

Bonds use a cosine switch between 1.15 and 1.40 x (r_cov,i + r_cov,j).
"""
import numpy as np
from scipy.spatial import cKDTree

from ..problem import _images

A_IN, B_OUT = 1.15, 1.40
SIG_CN, SIG_Q, SIG_RG, SIG_K2, SIG_SUB, SIG_H, SIG_V = 0.3, 0.04, 0.03, 0.05, 0.3, 0.1, 0.04


def _switch(x):
    t = np.clip((x - A_IN) / (B_OUT - A_IN), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


class CV:
    def __init__(self, p):
        self.p = p
        self.s = s = p.search_idx
        self.m = m = len(s)
        sym = np.array(p.symbols)[s]
        species = sorted(set(sym.tolist()))
        self.spec = np.array([species.index(x) for x in sym], int)
        nsp = len(species)
        self.rc = p.rcov[s]
        self.r_unit = float(self.rc.mean())
        pairs = [(a, b) for a in range(nsp) for b in range(a, nsp)]
        pid = np.zeros((nsp, nsp), int)
        for k, (a, b) in enumerate(pairs):
            pid[a, b] = pid[b, a] = k
        self.npair = len(pairs)
        self.pid = pid[self.spec[:, None], self.spec[None, :]]
        self.rsum = self.rc[:, None] + self.rc[None, :]
        self.rcut = B_OUT * float(self.rsum.max())
        self.pbc = np.asarray(p.pbc, bool)
        self.cell0 = np.array(p.cell0, float)
        self.inv0 = np.linalg.inv(self.cell0)
        names, sig = [], []
        if p.variable_cell:
            names.append("lnV")
            sig.append(SIG_V)
        for a, b in pairs:
            names.append(f"cn_{species[a]}{species[b]}")
            sig.append(SIG_CN)
        names += ["q4", "q6"]
        sig += [SIG_Q, SIG_Q]
        self.tree = None
        self.ztop = None
        if not p.variable_cell:
            names += ["rg", "k2"]
            sig += [SIG_RG, SIG_K2]
            if p.n_sub:
                shifts = _images(self.pbc) @ self.cell0
                xyz = (p.sub_pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
                self.sub_rc = np.tile(p.rcov[: p.n_sub], len(shifts))
                self.tree = cKDTree(xyz)
                self.sub_cut = B_OUT * float(self.rc.max() + p.rcov[: p.n_sub].max())
                names.append("cnsub")
                sig.append(SIG_SUB)
                if p.category == "surface":
                    z = p.sub_pos[:, 2]
                    reg = p.region or {}
                    if reg.get("type") == "box" and np.any(z < reg["lo"][2]):
                        z = z[z < reg["lo"][2]]
                    self.ztop = float(z.max())
                    names.append("h")
                    sig.append(SIG_H)
        self.names = names
        self.sigma = np.array(sig, float)
        self.d = len(names)

    # ------------------------------------------------------------ bonds
    def _bonds(self, x, cell):
        """Bonds among search atoms: (centre i, neighbour j, unit vector, weight)."""
        m = self.m
        if self.p.variable_cell:
            inv = np.linalg.inv(cell)
            fr = x @ inv
            fr -= np.floor(fr)
            xc = fr @ cell
            heights = 1.0 / np.linalg.norm(inv, axis=0)
            nrep = np.minimum(np.ceil(self.rcut / np.maximum(heights, 1e-3)).astype(int), 3)
            g = np.meshgrid(*[np.arange(-k, k + 1) for k in nrep], indexing="ij")
            sh = np.stack([a.ravel() for a in g], axis=1) @ cell
            d = xc[None, :, None, :] + sh[None, None, :, :] - xc[:, None, None, :]
            r = np.sqrt(np.einsum("ijsk,ijsk->ijs", d, d))
            xr = r / self.rsum[:, :, None]
            mask = (xr < B_OUT) & (r > 1e-6)
            ci, cj, cs = np.nonzero(mask)
            dv = d[ci, cj, cs]
            rr = r[ci, cj, cs]
            w = _switch(xr[ci, cj, cs])
        else:
            d = x[None, :, :] - x[:, None, :]
            if self.pbc.any():
                f = d @ self.inv0
                f -= np.where(self.pbc, np.round(f), 0.0)
                d = f @ self.cell0
            r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
            r[np.arange(m), np.arange(m)] = np.inf
            xr = r / self.rsum
            ci, cj = np.nonzero(xr < B_OUT)
            dv = d[ci, cj]
            rr = r[ci, cj]
            w = _switch(xr[ci, cj])
        u = dv / np.maximum(rr, 1e-9)[:, None]
        return ci, cj, u, w

    def _ql(self, ci, u, w):
        m = self.m
        nb = len(ci)
        if nb == 0:
            return 0.0, 0.0
        counts = np.bincount(ci, minlength=m)
        start = np.cumsum(counts) - counts
        rep = counts[ci]
        a = np.repeat(np.arange(nb), rep)
        offs = np.arange(len(a)) - np.repeat(np.cumsum(rep) - rep, rep)
        b = start[ci[a]] + offs
        c = np.einsum("ij,ij->i", u[a], u[b])
        c2 = c * c
        c4 = c2 * c2
        p4 = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0
        p6 = (231.0 * c4 * c2 - 315.0 * c4 + 105.0 * c2 - 5.0) / 16.0
        ww = w[a] * w[b]
        ca = ci[a]
        n4 = np.bincount(ca, weights=ww * p4, minlength=m)
        n6 = np.bincount(ca, weights=ww * p6, minlength=m)
        den = np.bincount(ci, weights=w, minlength=m) ** 2
        ok = den > 1e-9
        if not ok.any():
            return 0.0, 0.0
        q4 = np.sqrt(np.clip(n4[ok] / den[ok], 0.0, None)).mean()
        q6 = np.sqrt(np.clip(n6[ok] / den[ok], 0.0, None)).mean()
        return float(q4), float(q6)

    # ------------------------------------------------------------ CV vector
    def __call__(self, pos, cell=None):
        p = self.p
        cell = self.cell0 if cell is None or not p.variable_cell else np.asarray(cell, float)
        x = np.asarray(pos, float)[self.s]
        out = []
        if p.variable_cell:
            out.append(np.log(abs(np.linalg.det(cell)) / max(self.m, 1)))
        ci, cj, u, w = self._bonds(x, cell)
        cn = np.bincount(self.pid[ci, cj], weights=w, minlength=self.npair) / max(self.m, 1)
        out.extend(cn.tolist())
        out.extend(self._ql(ci, u, w))
        if not p.variable_cell:
            d = x - x[0]
            if self.pbc.any():
                f = d @ self.inv0
                f -= np.where(self.pbc, np.round(f), 0.0)
                d = f @ self.cell0
            xu = x[0] + d
            y = xu - xu.mean(axis=0)
            lam = np.clip(np.linalg.eigvalsh(y.T @ y / max(self.m, 1)), 0.0, None)
            tr = lam.sum()
            out.append(np.sqrt(tr) / (self.r_unit * max(self.m, 1) ** (1.0 / 3.0)))
            out.append(1.5 * (lam ** 2).sum() / tr ** 2 - 0.5 if tr > 1e-12 else 0.0)
            if self.tree is not None:
                fr = xu @ self.inv0
                fr -= np.where(self.pbc, np.floor(fr), 0.0)
                xw = fr @ self.cell0
                k = int(min(24, self.tree.n))
                dist, idx = self.tree.query(xw, k=k, distance_upper_bound=self.sub_cut)
                dist = np.asarray(dist).reshape(len(xw), -1)
                idx = np.asarray(idx).reshape(len(xw), -1)
                ok = idx < self.tree.n
                csub = 0.0
                if ok.any():
                    rs = self.rc[:, None].repeat(k, axis=1)[ok] + self.sub_rc[idx[ok]]
                    csub = float(_switch(dist[ok] / rs).sum()) / self.m
                out.append(csub)
                if self.ztop is not None:
                    out.append((xu[:, 2].mean() - self.ztop) / (2.0 * self.r_unit))
        v = np.asarray(out, float)
        v[~np.isfinite(v)] = 0.0
        return v


class Bias:
    """Sum of Gaussians with per-Gaussian widths (well-tempered heights)."""

    def __init__(self, d, cap=1024):
        self.d = d
        self.k = 0
        self.c = np.zeros((cap, d))
        self.isg = np.zeros((cap, d))
        self.h = np.zeros(cap)

    def __call__(self, s):
        k = self.k
        if k == 0:
            return 0.0
        z = (s[None, :] - self.c[:k]) * self.isg[:k]
        return float(self.h[:k] @ np.exp(-0.5 * np.einsum("ij,ij->i", z, z)))

    def deposit(self, s, sigma, w0, dT):
        v = self(s)
        h = w0 * np.exp(-v / dT) if dT > 0 else w0
        if self.k == len(self.h):
            n = 2 * len(self.h)
            self.c = np.resize(self.c, (n, self.d))
            self.isg = np.resize(self.isg, (n, self.d))
            self.h = np.resize(self.h, n)
        self.c[self.k] = s
        self.isg[self.k] = 1.0 / np.maximum(sigma, 1e-9)
        self.h[self.k] = h
        self.k += 1
        return h
