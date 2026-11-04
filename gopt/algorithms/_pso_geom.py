"""Coordinate spaces and structure alignment for the CALYPSO-style PSO (pso.py).

A particle position X is a (rows, 3) array:

* ``free``  (gas-phase clusters): centred Cartesian coordinates of the search
  atoms. Targets are aligned by translation + rotation/reflection (Kabsch) +
  per-species permutation (Hungarian), iterated from several starting frames.
* ``abs``   (supported / zeolite clusters, any problem with a substrate):
  absolute Cartesian coordinates; targets are aligned by per-species
  permutation only, displacements use the minimum image.
* ``cryst`` (bulk): fractional coordinates in the Niggli-reduced cell plus two
  rows of lattice parameters [ln a, ln b, ln c] and [alpha, beta, gamma] (rad).
  Targets are aligned by an origin shift + per-species permutation.

``Space.align(Y, X)`` returns the displacement D (same shape as X) that moves
X onto the aligned copy of Y, i.e. the (pbest - x) / (lbest - x) terms of the
PSO velocity update.
"""
import numpy as np
from ase.build.tools import niggli_reduce_cell
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .. import ops


def _wrap(f):
    return f - np.round(f)


def _kabsch(a, b, proper=False):
    """Orthogonal R minimising sum |a_i - R b_i|^2 (rows; a, b centred)."""
    u, _, vt = np.linalg.svd(b.T @ a)
    r = vt.T @ u.T
    if proper and np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return r


class Space:
    def __init__(self, p, vmax=0.15):
        self.p = p
        s = p.search_idx
        self.ns = len(s)
        sym = np.array(p.symbols)[s]
        self.groups = [np.where(sym == x)[0] for x in sorted(set(sym))]
        self.var_cell = bool(p.variable_cell)
        if p.n_sub == 0 and (p.variable_cell or p.category == "bulk"):
            self.mode = "cryst"
        elif p.n_sub == 0:
            self.mode = "free"
        else:
            self.mode = "abs"
        if self.mode == "cryst":
            self.vmax = np.full((self.ns + 2, 3), float(vmax))
            if not self.var_cell:
                self.vmax[self.ns:] = 0.0
        else:
            # CALYPSO's V_max is a fraction of the cell edge; for clusters the analogous
            # length is the diameter of a close-packed cluster of n_search atoms.
            rc = float(np.mean(p.rcov[s]))
            self.size = 2.0 * rc * (self.ns / 0.74) ** (1.0 / 3.0)
            self.vmax = np.full((self.ns, 3), float(vmax) * self.size)
        self.center = self._region_center()
        self.mask_pbc = np.asarray(p.pbc, bool)
        self.inv0 = np.linalg.inv(p.cell0) if self.mask_pbc.any() else None
        rs = p.rcov[s]
        self.bond_ss = 1.35 * (rs[:, None] + rs[None, :])
        np.fill_diagonal(self.bond_ss, 0.0)

    # ------------------------------------------------------------- helpers
    def _region_center(self):
        p = self.p
        r = p.region
        if r.get("type") == "sphere":
            return np.asarray(r["center"], float)
        if r.get("type") == "box":
            return 0.5 * (np.asarray(r["lo"], float) + np.asarray(r["hi"], float))
        return 0.5 * p.cell0.sum(axis=0)

    def random_velocity(self, rng):
        return rng.uniform(-1.0, 1.0, self.vmax.shape) * self.vmax

    def clamp(self, v):
        return np.clip(v, -self.vmax, self.vmax)

    # ------------------------------------------------------- encode/decode
    def encode(self, pos, cell):
        """Relaxed structure -> (X, standardised pos, cell, basis change M or None)."""
        p = self.p
        s = p.search_idx
        if self.mode == "free":
            pos = ops.unwrap_search(p, pos, p.cell0)
            x = pos[s] - pos[s].mean(axis=0)
            pos = np.array(pos)
            pos[s] = self.center + x
            return x, pos, p.cell0, None
        if self.mode == "abs":
            pos = ops.unwrap_search(p, pos, p.cell0)
            return np.array(pos[s]), pos, p.cell0, None
        cell = np.array(cell if cell is not None else p.cell0, float)
        frac = np.asarray(pos)[s] @ np.linalg.inv(cell)
        m = None
        if self.var_cell:
            try:
                _, op = niggli_reduce_cell(cell)
                m = np.asarray(op, float).T
                if np.allclose(m, np.eye(3)):
                    m = None
                else:
                    cell = m @ cell
                    frac = frac @ np.linalg.inv(m)
            except Exception:
                m = None
            par = cell_to_cellpar(cell)
            cell = cellpar_to_cell(par)
            lat = np.concatenate([np.log(par[:3]), np.radians(par[3:])]).reshape(2, 3)
        else:
            lat = np.zeros((2, 3))
        frac = frac - np.floor(frac)
        out = np.array(pos, float)
        out[s] = frac @ cell
        return np.vstack([frac, lat]), out, cell, m

    def decode(self, X, base_pos, base_cell):
        """Particle position -> (full pos, cell); None if the lattice is invalid."""
        p = self.p
        s = p.search_idx
        pos = np.array(base_pos, float)
        if self.mode == "free":
            pos[s] = self.center + X - X.mean(axis=0)
            return pos, p.cell0
        if self.mode == "abs":
            pos[s] = X
            return pos, p.cell0
        frac = X[: self.ns] - np.floor(X[: self.ns])
        cell = np.array(base_cell, float)
        if self.var_cell:
            lat = X[self.ns:].ravel()
            par = np.concatenate([np.exp(lat[:3]), np.degrees(lat[3:])])
            cell = None
            a = par[3:]
            if (np.all((a > 30.0) & (a < 150.0)) and a.sum() < 355.0
                    and np.all(2 * a < a.sum() - 1.0)):
                try:
                    c = cellpar_to_cell(par)
                    vol = abs(np.linalg.det(c))
                    h = 1.0 / np.linalg.norm(np.linalg.inv(c).T, axis=1)
                    if np.all(np.isfinite(c)) and vol > 1e-3 and h.min() > 0.3 * vol ** (1 / 3):
                        cell = c
                except Exception:
                    cell = None
            if cell is None:
                return None
        pos[s] = frac @ cell
        return pos, cell

    @staticmethod
    def transform_velocity(v, m, ns):
        """Velocity after the basis change ``cell' = M cell`` of a crystal."""
        if m is None:
            return v
        v = np.array(v)
        v[:ns] = v[:ns] @ np.linalg.inv(m)
        v[ns:] = 0.0
        return v

    def ctx_basis(self, cell, v):
        """ctx.relax restarts a skewed cell (spacing < 0.4 V^(1/3)) from its Minkowski
        basis; reduce it here (same call, idempotent) so that the velocity follows."""
        cell = np.asarray(cell, float)
        h = 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1).max()
        if h >= 0.4 * abs(np.linalg.det(cell)) ** (1 / 3):
            return cell, v
        rc, op = minkowski_reduce(cell, self.p.pbc)
        m = np.asarray(op, float)
        if np.allclose(m, np.eye(3)):
            return cell, v
        return np.array(rc, float), self.transform_velocity(v, m, self.ns)

    # ------------------------------------------------------------ alignment
    def _assign(self, cost_fn):
        """Per-species Hungarian assignment; cost_fn(g) -> (|g|,|g|) cost matrix."""
        perm = np.arange(self.ns)
        total = 0.0
        for g in self.groups:
            c = cost_fn(g)
            if len(g) == 1:
                total += float(c[0, 0])
                continue
            r, col = linear_sum_assignment(c)
            perm[g[r]] = g[col]
            total += float(c[r, col].sum())
        return perm, total

    def align(self, Y, X, rng):
        """Displacement D (shape of X) from X to the best-aligned copy of Y."""
        if self.mode == "free":
            return self._align_free(Y, X, rng)
        if self.mode == "abs":
            return self._align_abs(Y, X)
        return self._align_cryst(Y, X)

    def _align_abs(self, Y, X):
        p = self.p
        d = Y[None, :, :] - X[:, None, :]
        if self.mask_pbc.any():
            f = d @ self.inv0
            f[..., self.mask_pbc] -= np.round(f[..., self.mask_pbc])
            d = f @ p.cell0
        d2 = np.einsum("ijk,ijk->ij", d, d)
        perm, _ = self._assign(lambda g: d2[np.ix_(g, g)])
        return d[np.arange(self.ns), perm]

    def _align_free(self, Y, X, rng, n_random=6, iters=12):
        x = X - X.mean(axis=0)
        y = Y - Y.mean(axis=0)
        n = self.ns
        if n == 1:
            return x * 0.0
        _, ex = np.linalg.eigh(x.T @ x)
        _, ey = np.linalg.eigh(y.T @ y)
        starts = []
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    starts.append(ex @ np.diag([sx, sy, sz]) @ ey.T)
        for _ in range(n_random):
            r = ops.random_rotation(rng)
            starts.append(r)
            starts.append(-r)
        best = (np.inf, None)
        for r in starts:
            prev = None
            for _ in range(iters):
                yr = y @ r.T
                d = yr[None, :, :] - x[:, None, :]
                d2 = np.einsum("ijk,ijk->ij", d, d)
                perm, cost = self._assign(lambda g: d2[np.ix_(g, g)])
                if prev is not None and np.array_equal(perm, prev):
                    break
                prev = perm
                r = _kabsch(x, y[perm])
            yr = y[perm] @ r.T
            cost = float(((yr - x) ** 2).sum())
            if cost < best[0]:
                best = (cost, yr)
                if cost < 1e-4 * n:
                    break
        return best[1] - x

    def _align_cryst(self, Y, X):
        ns = self.ns
        fx, fy = X[:ns], Y[:ns]
        if self.var_cell:
            lat = X[ns:].ravel()
            cell = cellpar_to_cell(np.concatenate([np.exp(lat[:3]), np.degrees(lat[3:])]))
        else:
            cell = self.p.cell0
        g0 = min(self.groups, key=len)
        a = g0[0]
        best = (np.inf, None, None)
        for b in g0:
            t = fx[a] - fy[b]
            perm, cost = self._assign(lambda g: self._cost_cryst(fx[g], fy[g] + t, cell))
            if cost < best[0]:
                best = (cost, t, perm)
        _, t, perm = best
        # refine the origin shift by the mean residual and re-assign once
        t = t + _wrap(fy[perm] + t - fx).mean(axis=0)
        perm, _ = self._assign(lambda g: self._cost_cryst(fx[g], fy[g] + t, cell))
        D = np.zeros_like(X)
        D[:ns] = _wrap(fy[perm] + t - fx)
        if self.var_cell:
            D[ns:] = Y[ns:] - X[ns:]
        return D

    @staticmethod
    def _cost_cryst(fx, fy, cell):
        d = _wrap(fy[None, :, :] - fx[:, None, :]) @ cell
        return np.einsum("ijk,ijk->ij", d, d)

    # --------------------------------------------------------------- repair
    def merge_fragments(self, pos, step=0.3, max_iter=60):
        """Gas-phase clusters: translate disconnected fragments onto the main cluster."""
        p = self.p
        s = p.search_idx
        x = np.array(pos[s], float)
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        ncomp, lab = connected_components(csr_matrix(d < self.bond_ss), directed=False)
        if ncomp <= 1:
            return pos
        sizes = np.bincount(lab)
        main = list(np.where(lab == np.argmax(sizes))[0])
        for c in np.argsort(-sizes):
            if c == np.argmax(sizes):
                continue
            frag = np.where(lab == c)[0]
            for _ in range(max_iter):
                u = x[main].mean(axis=0) - x[frag].mean(axis=0)
                nu = np.linalg.norm(u)
                if nu < 1e-8:
                    break
                dd = np.linalg.norm(x[frag][:, None, :] - x[main][None, :, :], axis=-1)
                if np.any(dd < self.bond_ss[np.ix_(frag, main)]):
                    break
                x[frag] += step * u / nu
            main.extend(frag.tolist())
        out = np.array(pos, float)
        out[s] = x
        return out


class FPArchive:
    """Fingerprints of known structures for CALYPSO's similarity check (Algorithm 1).

    Non-bulk fingerprints (sorted distance lists) are subsampled to <= ``dim``
    evenly spaced entries, which approximates their mean absolute difference at
    a fraction of the cost for large clusters.
    """

    def __init__(self, p, cap=3000, dim=128):
        self.p = p
        self.cap = cap
        self.dim = dim
        self.sel = None
        self.data = None
        self.n = 0

    def _reduce(self, fp):
        fp = np.asarray(fp, float)
        if self.sel is None:
            if not self.p.variable_cell and len(fp) > self.dim:
                self.sel = np.unique(np.linspace(0, len(fp) - 1, self.dim).round().astype(int))
            else:
                self.sel = slice(None)
        return fp[self.sel]

    def add(self, fp):
        fp = self._reduce(fp)
        if self.data is None:
            self.data = np.zeros((64, len(fp)))
        if self.n == len(self.data):
            if self.n >= self.cap:
                # keep the newest half
                keep = self.data[self.n // 2: self.n].copy()
                self.data[: len(keep)] = keep
                self.n = len(keep)
            else:
                self.data = np.vstack([self.data, np.zeros_like(self.data)])
        self.data[self.n] = fp
        self.n += 1

    def min_dist(self, fp):
        if not self.n:
            return np.inf
        fp = self._reduce(fp)
        a = self.data[: self.n]
        if self.p.variable_cell:
            return float(1.0 - (a @ fp).max())
        return float(np.abs(a - fp).mean(axis=1).min())
