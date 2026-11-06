"""Geometry and learned site-energy model for the cut-and-regrow search (growth.py).

No energy calls here. Everything works on an *environment*: the search atoms that
are currently present (the "anchors", always first) plus fixed points around
them (frozen substrate atoms incl. periodic images, or periodic images of the
anchors themselves for variable-cell bulk).

Site energies come from a per-atom surrogate fitted on the fly to the free NEP
per-atom energies (``ctx.atomic_energies``):

    eps_a(c) = w0 + sum_b w1_b c_b + sum_b w2_b c_b^2 + wT T^2 + ws sqrt(T),

where c_b is the smooth coordination of an atom by neighbours of species b
(switch at 1.25 x covalent-radius sum) and T = sum_b c_b. The predicted cost
of growing an atom of species a at site x is eps_a(c(x)) plus the change of the
neighbours' eps when they gain the new neighbour.
"""
import itertools

import numpy as np
from scipy.spatial import cKDTree

from ..problem import _images

SW_X0, SW_K, SW_CUT = 1.25, 25.0, 1.6


def switch(x):
    return 1.0 / (1.0 + np.exp(np.clip(SW_K * (x - SW_X0), -60.0, 60.0)))


def _pairs(tree_a, tree_b, r):
    d = tree_a.sparse_distance_matrix(tree_b, r, output_type="ndarray")
    return d["i"].astype(np.int64), d["j"].astype(np.int64), d["v"]


def _unit(v, rng):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    bad = n[..., 0] < 1e-6
    if np.any(bad):
        r = rng.normal(size=(int(bad.sum()), 3))
        v = np.array(v, float)
        v[bad] = r
        n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / n


class Env:
    __slots__ = ("P", "sp", "home", "m", "c", "tree", "cell", "bulk")


class Geo:
    """Static data of a problem: species ids, radius sums, substrate images."""

    def __init__(self, p):
        self.p = p
        elems = sorted(set(p.symbols))
        self.elems = elems
        self.ns = ns = len(elems)
        self.sp = np.array([elems.index(s) for s in p.symbols])
        rc = np.zeros(ns)
        for i, s in enumerate(p.symbols):
            rc[elems.index(s)] = p.rcov[i]
        self.d0 = rc[:, None] + rc[None, :]
        self.bond = self.d0.copy()      # learned contact distances (placement targets)
        self.bond_n = np.zeros((ns, ns))
        self.s_idx = p.search_idx
        self.sp_s = self.sp[p.search_idx]
        self.search_species = np.unique(self.sp_s)
        self.bulk = bool(p.variable_cell)
        self.cut = SW_CUT * self.d0.max()
        self.has_sub = p.n_sub > 0 and not self.bulk
        if self.has_sub:
            shifts = _images(p.pbc) @ p.cell0
            self.sub_pts = (p.sub_pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
            self.sub_home = np.tile(np.arange(p.n_sub), len(shifts))
            self.sub_tree = cKDTree(self.sub_pts)
            sps = self.sp[: p.n_sub]
            i, j, r = _pairs(cKDTree(p.sub_pos), self.sub_tree, self.cut)
            ok = r > 1e-6
            i, j, r = i[ok], j[ok], r[ok]
            spj = sps[self.sub_home[j]]
            s = switch(r / self.d0[sps[i], spj])
            self.c_sub = np.bincount(i * ns + spj, weights=s,
                                     minlength=p.n_sub * ns).reshape(p.n_sub, ns)

    def env_radius(self):
        if self.bulk:
            return max(self.cut, 1.5 * self.bond.max()) + 0.2
        return max(1.5 * self.bond.max(), self.bond.max() + self.cut) + 0.3

    # ------------------------------------------------------------ environments
    def build(self, X, sps, cell):
        """Environment of present search atoms X (unwrapped or home-cell) of species sps."""
        X = np.asarray(X, float).reshape(-1, 3)
        sps = np.asarray(sps, int)
        m = len(X)
        ns = self.ns
        env = Env()
        env.m, env.cell, env.bulk = m, cell, self.bulk
        if self.bulk:
            inv = np.linalg.inv(cell)
            frac = X @ inv
            frac -= np.floor(frac)
            X = frac @ cell
            heights = 1.0 / np.linalg.norm(inv, axis=0)
            f = self.env_radius() / heights
            nr = np.ceil(f).astype(int)
            sh = np.array(list(itertools.product(*[range(-k, k + 1) for k in nr])), float)
            sh = sh[np.any(sh != 0, axis=1)]
            F = frac[None, :, :] + sh[:, None, :]
            ok = np.all((F >= -f) & (F <= 1 + f), axis=2)
            si, ai = np.nonzero(ok)
            P = np.vstack([X, F[si, ai] @ cell]) if len(si) else X
            home = np.concatenate([np.arange(m), ai])
            sp = sps[home]
            tP = cKDTree(P)
            i, j, r = _pairs(cKDTree(X), tP, self.cut)
            ok = j != i
            i, j, r = i[ok], j[ok], r[ok]
            s = switch(r / self.d0[sp[i], sp[j]])
            ch = np.bincount(i * ns + sp[j], weights=s, minlength=m * ns).reshape(m, ns)
            env.P, env.sp, env.home, env.tree = P, sp, home, tP
            env.c = ch[home]
            return env
        if self.has_sub and m:
            lists = self.sub_tree.query_ball_point(X, r=self.env_radius())
            sel = np.unique(np.fromiter(itertools.chain.from_iterable(lists), np.int64))
        else:
            sel = np.zeros(0, np.int64)
        hb = self.sub_home[sel] if len(sel) else np.zeros(0, np.int64)
        P = np.vstack([X, self.sub_pts[sel]]) if len(sel) else X
        sp = np.concatenate([sps, self.sp[hb]])
        home = np.concatenate([-1 - np.arange(m), hb])
        c = np.zeros((len(P), ns))
        if len(sel):
            c[m:] = self.c_sub[hb]
        tP = cKDTree(P)
        if m:
            i, j, r = _pairs(cKDTree(X), tP, self.cut)
            ok = j != i
            i, j, r = i[ok], j[ok], r[ok]
            s = switch(r / self.d0[sp[i], sp[j]])
            c[:m] += np.bincount(i * ns + sp[j], weights=s, minlength=m * ns).reshape(m, ns)
            sub = j >= m
            if np.any(sub):
                add = np.bincount((j[sub] - m) * ns + sp[i[sub]], weights=s[sub],
                                  minlength=(len(P) - m) * ns).reshape(len(P) - m, ns)
                c[m:] += add
        env.P, env.sp, env.home, env.tree, env.c = P, sp, home, tP, c
        return env

    # -------------------------------------------------------------- candidates
    def candidates(self, env, a, rng, n_atop=2, n_bridge=2):
        """Hollow (3-atom trilateration), bridge and atop sites for species a."""
        P, sp, m = env.P, env.sp, env.m
        if m == 0:
            return np.zeros((0, 3))
        ba = self.bond[a]
        rmax = 0.75 * 2 * ba.max()
        tA = cKDTree(P[:m])
        i, j, r = _pairs(tA, env.tree, rmax)
        ok = (j > i) & (r < 0.75 * (ba[sp[i]] + ba[sp[j]])) & (r > 1e-6)
        i, j = i[ok], j[ok]
        # local frame: dense adjacency among anchors and their neighbours
        loc = np.unique(np.concatenate([np.arange(m), j]))
        L = len(loc)
        pos = P[loc]
        lsp = sp[loc]
        D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        lim = 0.75 * (ba[lsp][:, None] + ba[lsp][None, :])
        adj = (D < lim) & (D > 1e-6)
        cen = np.where(adj.any(1)[:, None],
                       (adj @ pos) / np.maximum(adj.sum(1), 1)[:, None], pos - 1.0)
        out = _unit(pos - cen, rng)
        sites = []
        # atop
        da = ba[lsp[:m]][:, None]
        sites.append(pos[:m] + out[:m] * da)
        for _ in range(n_atop - 1):
            sites.append(pos[:m] + _unit(out[:m] + rng.normal(size=(m, 3)), rng) * da)
        # bridges and hollows from edges (anchor first)
        li = np.searchsorted(loc, i)
        lj = np.searchsorted(loc, j)
        if len(li):
            A, B = pos[li], pos[lj]
            ra, rb = ba[lsp[li]], ba[lsp[lj]]
            ax = B - A
            d = np.linalg.norm(ax, axis=1)
            ex = ax / d[:, None]
            x = (ra ** 2 - rb ** 2 + d ** 2) / (2 * d)
            rho2 = ra ** 2 - x ** 2
            okb = rho2 > 0.04
            base = A + x[:, None] * ex
            rho = np.sqrt(np.maximum(rho2, 0))[:, None]
            for t in range(n_bridge):
                v = out[li] + out[lj] if t == 0 else rng.normal(size=(len(li), 3))
                v = v - np.einsum("ij,ij->i", v, ex)[:, None] * ex
                v = _unit(v, rng)
                sites.append((base + rho * v)[okb])
            # triangles (li < lj < lk all adjacent, li an anchor)
            common = adj[li] & adj[lj]
            common[:, :] &= np.arange(L)[None, :] > lj[:, None]
            e_idx, lk = np.nonzero(common)
            if len(e_idx):
                A, B, C = pos[li[e_idx]], pos[lj[e_idx]], pos[lk]
                rA, rB, rC = ba[lsp[li[e_idx]]], ba[lsp[lj[e_idx]]], ba[lsp[lk]]
                s1, s2, okt = _trilaterate(A, B, C, rA, rB, rC)
                sites.append(s1[okt])
                sites.append(s2[okt])
        X = np.vstack(sites) if sites else np.zeros((0, 3))
        if env.bulk and len(X):
            inv = np.linalg.inv(env.cell)
            fr = X @ inv
            X = (fr - np.floor(fr)) @ env.cell
        if len(X) > 1:
            _, u = np.unique(np.round(X / 0.15).astype(np.int64), axis=0, return_index=True)
            X = X[np.sort(u)]
        return X

    # ---------------------------------------------------------------- scoring
    def score(self, env, X, a, model, overlap=0.8):
        """Predicted energy change for growing species a at sites X (inf = overlap)."""
        C = len(X)
        if C == 0:
            return np.zeros(0)
        ns = self.ns
        i, j, r = _pairs(cKDTree(X), env.tree, self.cut)
        spj = env.sp[j]
        bad = r < np.maximum(overlap * self.bond[a, spj], self.p.dmin_factor * self.d0[a, spj])
        badc = np.bincount(i[bad], minlength=C) > 0
        s = switch(r / self.d0[a, spj])
        cnew = np.bincount(i * ns + spj, weights=s, minlength=C * ns).reshape(C, ns)
        e = model.energy(np.full(C, a), cnew)
        cj = env.c[j]
        cj2 = cj.copy()
        cj2[:, a] += s
        de = model.energy(spj, cj2) - model.energy(spj, cj)
        e = e + np.bincount(i, weights=de, minlength=C)
        nb = np.bincount(i, minlength=C)
        e[badc | (nb == 0)] = np.inf
        return e

    # ----------------------------------------------------------- bond lengths
    def learn_bonds(self, env, rate=0.2):
        """Update contact distances from the nearest neighbours of present search atoms."""
        m = env.m
        if m == 0:
            return
        rmax = 2.2 * self.d0.max()
        i, j, r = _pairs(cKDTree(env.P[:m]), env.tree, rmax)
        ok = (j != i) & (r > 1e-6)
        i, j, r = i[ok], j[ok], r[ok]
        if not len(i):
            return
        spi, spj = env.sp[i], env.sp[j]
        key = (i * self.ns + spj)
        order = np.lexsort((r, key))
        key_s = key[order]
        first = np.ones(len(order), bool)
        first[1:] = key_s[1:] != key_s[:-1]
        sel = order[first]
        for a in np.unique(spi[sel]):
            for b in np.unique(spj[sel]):
                mm = (spi[sel] == a) & (spj[sel] == b)
                if not np.any(mm):
                    continue
                dmed = float(np.median(r[sel][mm]))
                if dmed > 2.0 * self.d0[a, b]:
                    continue
                if self.bond_n[a, b] == 0:
                    new = dmed
                else:
                    new = (1 - rate) * self.bond[a, b] + rate * dmed
                self.bond[a, b] = self.bond[b, a] = new
                self.bond_n[a, b] += 1
                self.bond_n[b, a] = self.bond_n[a, b]


def _trilaterate(A, B, C, ra, rb, rc):
    ex = B - A
    d = np.linalg.norm(ex, axis=1)
    ex = ex / d[:, None]
    ac = C - A
    i = np.einsum("ij,ij->i", ex, ac)
    ey = ac - i[:, None] * ex
    jn = np.linalg.norm(ey, axis=1)
    ok = jn > 1e-3
    jn = np.where(ok, jn, 1.0)
    ey = ey / jn[:, None]
    ez = np.cross(ex, ey)
    x = (ra ** 2 - rb ** 2 + d ** 2) / (2 * d)
    y = (ra ** 2 - rc ** 2 + i ** 2 + jn ** 2) / (2 * jn) - (i / jn) * x
    z2 = ra ** 2 - x ** 2 - y ** 2
    ok &= z2 > 0.0
    z = np.sqrt(np.maximum(z2, 0.0))[:, None]
    base = A + x[:, None] * ex + y[:, None] * ey
    return base + z * ez, base - z * ez, ok


class SiteModel:
    """Per-species ridge model of NEP per-atom energies on coordination features."""

    def __init__(self, ns, default_slope=-0.5):
        self.ns = ns
        self.np_ = 2 * ns + 3
        self.A = np.zeros((ns, self.np_, self.np_))
        self.b = np.zeros((ns, self.np_))
        self.cnt = np.zeros(ns, int)
        self.W = np.zeros((ns, self.np_))
        self.W[:, 1:1 + ns] = default_slope
        self.fitted = np.zeros(ns, bool)

    def phi(self, c):
        T = c.sum(axis=1, keepdims=True)
        return np.hstack([np.ones_like(T), c, c * c, T * T, np.sqrt(np.maximum(T, 0.0))])

    def energy(self, sp, c):
        return np.einsum("ij,ij->i", self.phi(c), self.W[sp])

    def add(self, sp, c, e):
        ok = np.isfinite(e) & (np.abs(e) < 1e3)
        if not np.any(ok):
            return
        X = self.phi(c[ok])
        sp, e = sp[ok], e[ok]
        for a in np.unique(sp):
            mm = sp == a
            Xa = X[mm]
            self.A[a] += Xa.T @ Xa
            self.b[a] += Xa.T @ e[mm]
            self.cnt[a] += int(mm.sum())

    def fit(self):
        k = self.np_
        for a in range(self.ns):
            if self.cnt[a] < 2 * k:
                continue
            A = self.A[a]
            lam = 1e-5 * np.diag(A) + 1e-9
            try:
                self.W[a] = np.linalg.solve(A + np.diag(lam), self.b[a])
                self.fitted[a] = True
            except np.linalg.LinAlgError:
                pass
