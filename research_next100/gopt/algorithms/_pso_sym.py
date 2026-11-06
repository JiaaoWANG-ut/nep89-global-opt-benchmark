"""Symmetry-constrained random crystals for the CALYPSO-style PSO (pso.py).

CALYPSO draws its random crystal structures (first generation and the random
fraction of every generation) with space-group symmetry: a random space group,
a lattice of its crystal system and atoms on Wyckoff positions that reproduce
the composition. Without Wyckoff tables (no spglib/pyxtal in the environment)
the sites are generated from ASE's symmetry operations (setting 1; hexagonal
axes for the R groups, unique axis b for monoclinic groups):

* general position: a uniform random point (only if its multiplicity fits);
* fixed set of one operation (mirror plane, rotation axis, inversion or
  rotoinversion centre): a random point averaged over the cyclic group of the
  operation, using the lattice shift that makes the operation of finite order;
* special points without free parameters (e.g. site symmetry 222 or 23): a
  random point of the grid {0, 1/8, ..., 7/8, 1/6, 1/3, 2/3, 5/6}^3.

The multiplicity of a site is its number of distinct images. Orbits are added
per species (largest count first) while they fit the remaining count and keep
every distance >= dmin (the same minimum distances as Problem.random_structure).
``SymGen.structure`` returns (pos, cell) or None; it never calls ctx.
"""
import numpy as np
from ase.geometry import cellpar_to_cell
from ase.spacegroup import Spacegroup

from ..problem import min_image_distances

_GRID = np.array(sorted(set([i / 8.0 for i in range(8)] + [1 / 6, 1 / 3, 2 / 3, 5 / 6])))
_SHIFTS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)


def _wrap(f):
    return f - np.floor(f)


class SymGen:
    def __init__(self, p, max_groups=8, max_tries=40, p_general=0.4, p_fixed=0.4):
        self.p = p
        s = p.search_idx
        sym = np.array(p.symbols)[s]
        species = sorted(set(sym), key=lambda x: (-(sym == x).sum(), x))
        self.groups = [(x, s[sym == x]) for x in species]
        self.max_groups = int(max_groups)
        self.max_tries = int(max_tries)
        self.p_general = float(p_general)
        self.p_fixed = float(p_fixed)
        self.cache = {}
        self.n_ok = 0
        self.n_fail = 0

    # ----------------------------------------------------------- groups
    def _group(self, no):
        g = self.cache.get(no)
        if g is not None:
            return g
        ops = Spacegroup(int(no)).get_symop()
        R = np.array([o[0] for o in ops], float)
        T = np.array([o[1] for o in ops], float)
        eye = np.eye(3)
        fixed = []
        for r, t in zip(R, T):
            if np.allclose(r, eye):
                continue
            k, m = 1, r.copy()
            while not np.allclose(m, eye) and k < 7:
                m = m @ r
                k += 1
            if k >= 7:
                continue
            S = np.zeros((3, 3))
            m = np.eye(3)
            for _ in range(k):
                S += m
                m = m @ r
            # g_n = (r, t - n) has finite order (hence a fixed set) iff S (t - n) = 0
            v = (t[None, :] - _SHIFTS) @ S.T
            ok = np.abs(v).max(axis=1) < 1e-8
            if ok.any():
                fixed.append((r, t[None, :] - _SHIFTS[ok], k))
        g = dict(no=int(no), R=R, T=T, fixed=fixed, nsym=len(R))
        self.cache[no] = g
        return g

    def _site(self, g, rng, general):
        u = rng.random()
        if general and u < self.p_general:
            return rng.random(3)
        if g["fixed"] and u < self.p_general + self.p_fixed:
            r, tns, k = g["fixed"][rng.integers(len(g["fixed"]))]
            tn = tns[rng.integers(len(tns))]
            z = rng.random(3)
            acc = np.zeros(3)
            for _ in range(k):
                acc += z
                z = r @ z + tn
            return _wrap(acc / k)
        return _GRID[rng.integers(len(_GRID), size=3)]

    @staticmethod
    def _orbit(g, x):
        f = _wrap(np.einsum("mij,j->mi", g["R"], x) + g["T"] + 1e-9) - 1e-9
        key = np.round(f * 1e4).astype(np.int64) % 10000
        _, first = np.unique(key, axis=0, return_index=True)
        return _wrap(f[np.sort(first)])

    # ----------------------------------------------------------- lattice
    def _cell(self, no, rng):
        p = self.p
        vol = float(p.vpa) * p.n_search * rng.uniform(0.9, 1.3)
        for _ in range(50):
            L = rng.uniform(0.6, 1.6, 3)
            A = np.array([90.0, 90.0, 90.0])
            if no <= 2:
                A = rng.uniform(60.0, 120.0, 3)
            elif no <= 15:
                A[1] = rng.uniform(60.0, 120.0)
            elif no <= 74:
                pass
            elif no <= 142:
                L[1] = L[0]
            elif no <= 194:
                L[1] = L[0]
                A[2] = 120.0
            else:
                L[:] = L[0]
            try:
                cell = cellpar_to_cell(np.concatenate([L, A]))
            except Exception:
                continue
            v = abs(np.linalg.det(cell))
            if not np.isfinite(v) or v < 1e-3:
                continue
            cell = cell * (vol / v) ** (1.0 / 3.0)
            h = 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)
            if h.min() >= 0.45 * vol ** (1.0 / 3.0):
                return cell
        return None

    # --------------------------------------------------------- structure
    def structure(self, rng):
        """Random symmetric crystal (pos, cell) with the exact composition, or None."""
        p = self.p
        for _ in range(self.max_groups):
            no = int(rng.integers(1, 231))
            g = self._group(no)
            cell = self._cell(no, rng)
            if cell is None:
                continue
            placed_x = np.zeros((0, 3))
            placed_r = np.zeros(0)
            out = {}
            ok = True
            for x, idx in self.groups:
                rc = float(p.rcov[idx[0]])
                need = len(idx)
                pts = []
                tries = 0
                while need > 0 and tries < self.max_tries:
                    tries += 1
                    orb = self._orbit(g, self._site(g, rng, g["nsym"] <= need))
                    m = len(orb)
                    if m > need:
                        continue
                    cart = orb @ cell
                    if m > 1:
                        d = min_image_distances(cart, cart, cell, p.pbc)
                        d[np.arange(m), np.arange(m)] = np.inf
                        if d.min() < p.dmin_factor * 2.0 * rc:
                            continue
                    if len(placed_x):
                        d = min_image_distances(cart, placed_x, cell, p.pbc)
                        if np.any(d < p.dmin_factor * (rc + placed_r[None, :])):
                            continue
                    pts.append(cart)
                    placed_x = np.vstack([placed_x, cart])
                    placed_r = np.concatenate([placed_r, np.full(m, rc)])
                    need -= m
                if need > 0:
                    ok = False
                    break
                out[x] = np.vstack(pts)
            if not ok:
                continue
            pos = np.zeros((p.n, 3))
            for x, idx in self.groups:
                pos[idx] = out[x][rng.permutation(len(idx))]
            self.n_ok += 1
            return pos, cell
        self.n_fail += 1
        return None
