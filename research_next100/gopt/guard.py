"""Short-range repulsive guard added to the NEP89 energy.

Universal ML potentials can have unphysical "holes" at short interatomic
distances (e.g. NEP89 collapses Ti4O8 to 4.8 A^3/atom with Ti-Ti = 1.9 A, 0.66
eV/atom below anatase). The guard adds

    E_g = sum_pairs k/2 * (r0_ij - r_ij)^2   for r_ij < r0_ij = factor * (rcov_i + rcov_j)

over pairs that involve at least one search atom (substrate-substrate pairs
are fixed and physical). It is exactly zero for physical structures (bonds are
typically >= 0.85 x the covalent-radius sum) and only removes collapsed
configurations from the landscape.

A benchmark may raise r0 for chosen element pairs (problem.json "guard_pairs",
e.g. {"O-O": 1.9}: r0 = max(factor * rcov sum, 1.9)). This removes NEP89
anion-anion holes (O-O or N-N bonded collapsed states far below the physical
minimum), while the shortest physical anion-anion distances are >= 2.2 A.

Surface benchmarks also get a flat-bottom wall k/2 * (z' - z_w)^2 on each search
atom whose height z' above the slab (taken modulo the period along the surface
normal, with the cut in the slab mid-plane) exceeds z_w = top face + half the
vacuum (Problem.z_wall). It stops atoms from crossing the vacuum to the frozen
bottom face of the periodic image and is zero for atoms nearer the top face.
"""
import numpy as np
from scipy.spatial import cKDTree

from .problem import _images


class Guard:
    def __init__(self, p, factor=0.75, k=20.0):
        self.p = p
        self.factor = float(factor)
        self.k = float(k)
        self.s = p.search_idx
        self.rc = p.rcov
        self.rmax = self.factor * 2 * self.rc[self.s].max() if len(self.s) else 0.0
        self.r0_ss = self.factor * (self.rc[self.s][:, None] + self.rc[self.s][None, :])
        self.r0sq_ss = self.r0_ss ** 2
        self.triu = np.triu(np.ones((len(self.s), len(self.s)), bool), 1)
        self.inv0 = np.linalg.inv(p.cell0)
        if p.variable_cell:
            n = p.n
            self.r0_all = self.factor * (self.rc[:, None] + self.rc[None, :])
            self.r0sq_all = self.r0_all ** 2
            ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            self.i_lt_j = ii < jj
            self.i_eq_j = ii == jj
            self._grids = {}
        self.tree = None
        if p.n_sub and not p.variable_cell:
            shifts = _images(p.pbc) @ p.cell0
            self.sub_xyz = (p.sub_pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
            self.sub_idx = np.tile(np.arange(p.n_sub), len(shifts))
            self.tree = cKDTree(self.sub_xyz)
            self.rmax_sub = self.factor * (self.rc[self.s].max() + self.rc[: p.n_sub].max())
        self.zw = getattr(p, "z_wall", None)
        self.floor = None
        pairs = getattr(p, "guard_pairs", None) or {}
        if pairs:
            # r0 floors through a species-pair table; r0 = max(factor * rcov sum, floor)
            species = sorted(set(p.symbols))
            self.tid = np.array([species.index(x) for x in p.symbols])
            self.floor = np.zeros((len(species), len(species)))
            for (a, b), v in pairs.items():
                if a in species and b in species:
                    ia, ib = species.index(a), species.index(b)
                    self.floor[ia, ib] = self.floor[ib, ia] = v
            fmax = self.floor.max()
            ts = self.tid[self.s]
            self.r0_ss = np.maximum(self.r0_ss, self.floor[ts[:, None], ts[None, :]])
            self.r0sq_ss = self.r0_ss ** 2
            self.rmax = max(self.rmax, fmax)
            if p.variable_cell:
                self.r0_all = np.maximum(self.r0_all, self.floor[self.tid[:, None], self.tid[None, :]])
                self.r0sq_all = self.r0_all ** 2
            if self.tree is not None:
                self.rmax_sub = max(self.rmax_sub, fmax)

    def __call__(self, pos, cell):
        """Return (energy, forces[n,3] or None, virial[3,3], per-atom energies or None)."""
        pos = np.asarray(pos, float)
        terms = self._bulk_pairs(pos, cell) if self.p.variable_cell else self._fixed_pairs(pos)
        wall = self._wall(pos) if self.zw is not None else None
        vir = np.zeros((3, 3))
        if not terms and wall is None:
            return 0.0, None, vir, None
        e = 0.0
        f = np.zeros((self.p.n, 3))
        ea = np.zeros(self.p.n)
        for i, j, d, r, r0 in terms:
            x = r0 - r
            ep = 0.5 * self.k * x * x
            e += float(ep.sum())
            np.add.at(ea, i, 0.5 * ep)
            np.add.at(ea, j, 0.5 * ep)
            fi = (self.k * x / np.maximum(r, 1e-6))[:, None] * d  # force on i; d = r_i - r_j
            np.add.at(f, i, fi)
            np.add.at(f, j, -fi)
            vir += d.T @ fi
        if wall is not None:
            # external field on fixed-cell slabs: no virial contribution
            idx, x = wall
            ep = 0.5 * self.k * x * x
            e += float(ep.sum())
            ea[idx] += ep
            f[idx] -= (self.k * x)[:, None] * self.zw["normal"][None, :]
        return e, f, vir, ea

    def _wall(self, pos):
        zw = self.zw
        h = pos[self.s] @ zw["normal"]
        x = zw["mid"] + np.mod(h - zw["mid"], zw["period"]) - (zw["top"] + zw["height"])
        m = x > 0
        if not m.any():
            return None
        return self.s[m], x[m]

    def _fixed_pairs(self, pos):
        p = self.p
        s = self.s
        xs = pos[s]
        out = []
        # search-search with the minimum image (cells are much larger than 2*r0)
        d = xs[:, None, :] - xs[None, :, :]
        if np.any(p.pbc):
            fr = d @ self.inv0
            fr -= np.where(p.pbc, np.round(fr), 0.0)
            d = fr @ p.cell0
        r2 = np.einsum("ijx,ijx->ij", d, d)
        a, b = np.nonzero((r2 < self.r0sq_ss) & self.triu)
        if len(a):
            out.append((s[a], s[b], d[a, b], np.sqrt(r2[a, b]), self.r0_ss[a, b]))
        # search-substrate through a KD-tree of the substrate and its 26 images
        if self.tree is not None:
            fr = xs @ self.inv0
            xw = xs - np.where(p.pbc, np.floor(fr), 0.0) @ p.cell0
            hits = self.tree.query_ball_point(xw, self.rmax_sub)
            ii = np.fromiter((k for k, h in enumerate(hits) for _ in h), int)
            if len(ii):
                jj = np.fromiter((x for h in hits for x in h), int)
                dv = xw[ii] - self.sub_xyz[jj]
                rr = np.linalg.norm(dv, axis=1)
                js = self.sub_idx[jj]
                r0 = self.factor * (self.rc[s[ii]] + self.rc[js])
                if self.floor is not None:
                    r0 = np.maximum(r0, self.floor[self.tid[s[ii]], self.tid[js]])
                m = rr < r0
                if m.any():
                    out.append((s[ii[m]], js[m], dv[m], rr[m], r0[m]))
        return out

    def _bulk_pairs(self, pos, cell):
        inv = np.linalg.inv(cell)
        frac = pos @ inv
        frac -= np.floor(frac)
        heights = 1.0 / np.linalg.norm(inv, axis=0)
        key = tuple(np.minimum(np.ceil(self.rmax / heights).astype(int), 4))
        g = self._grids.get(key)
        if g is None:
            grids = [np.arange(-m, m + 1) for m in key]
            sh = np.array(np.meshgrid(*grids, indexing="ij")).reshape(3, -1).T.astype(float)
            positive = (sh[:, 0] > 0) | ((sh[:, 0] == 0) & (sh[:, 1] > 0)) | \
                ((sh[:, 0] == 0) & (sh[:, 1] == 0) & (sh[:, 2] > 0))
            # each unordered pair once: i<j with every shift, i==j with "positive" shifts
            once = self.i_lt_j[None] | (self.i_eq_j[None] & positive[:, None, None])
            g = self._grids[key] = (sh, once)
        sh, once = g
        # d[k, i, j] = r_i - (r_j + shift_k)
        d = ((frac[:, None, :] - frac[None, :, :])[None] - sh[:, None, None, :]) @ cell
        r2 = np.einsum("kijx,kijx->kij", d, d)
        k, i, j = np.nonzero((r2 < self.r0sq_all[None]) & once)
        if not len(k):
            return []
        return [(i, j, d[k, i, j], np.sqrt(r2[k, i, j]), self.r0_all[i, j])]
