"""Artificial bee colony (ABC) for atomistic structures, ABCluster style.

Food sources are locally relaxed structures. Each cycle has an employed-bee
phase (one neighbour search per source), an onlooker-bee phase (SN neighbour
searches on sources drawn with probability proportional to fitness) and a
scout-bee phase (sources not improved for ``limit`` generations are replaced
by new random structures). The neighbour of source x_i is
v = x_i + phi * (x_i - x_k) applied to a random subset of atoms (and the
lattice for variable-cell bulk) after aligning x_k onto x_i.
"""
import itertools

import numpy as np
from ase.geometry import cell_to_cellpar, cellpar_to_cell, minkowski_reduce
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components

from gopt import ops
from gopt.problem import min_image_distances

LABEL = "Artificial bee colony (ABCluster)"
FAMILY = "baseline"
REFERENCE = ("Karaboga & Basturk, J. Glob. Optim. 39, 459 (2007); "
             "Zhang & Dolg, Phys. Chem. Chem. Phys. 17, 24173 (2015) (ABCluster)")
DESCRIPTION = (
    "Artificial bee colony on the relaxed energy surface (ABCluster). SN=10 food "
    "sources (random structures relaxed with L-BFGS). Employed phase: for every source "
    "x_i a neighbour v = x_i + phi*(x_i - x_k) (k != i random, phi ~ U[-1,1]) is built, "
    "relaxed and greedily replaces x_i if lower in energy. Onlooker phase: SN more "
    "neighbour searches on sources drawn with p_i = fit_i/sum(fit), fit_i = 1/(1+E_i-E_min). "
    "Scout phase: every source not improved for limit=3 generations is replaced by a new "
    "random relaxed structure. Before differencing, x_k is aligned onto x_i: per-species "
    "Hungarian assignment, plus Procrustes rotation (reflections allowed, multi-start) "
    "for gas-phase clusters, substrate-lattice translation for supported clusters, and "
    "Minkowski-reduced cells + signed axis permutation + origin shift for bulk "
    "(fractional coordinates and lattice vectors; variable-cell relaxation). Overlaps are "
    "repaired and detached atoms/fragments re-seated on the cluster; for supported "
    "clusters, atoms that a move carries into the far half of the periodic vacuum gap "
    "(towards the underside of the frozen slab) are re-seated the same way (stats "
    "n_far_side). Deviations from the "
    "papers (documented): (1) the move perturbs a random subset of atoms (size uniform "
    "in 1..N; bulk: the lattice is one more unit), each with its own phi applied to the "
    "whole atom vector, instead of one scalar coordinate (a one-coordinate change is "
    "undone by the relaxation); (2) the fitness uses E - min(E) because raw total "
    "energies are large negative numbers for which 1+|f| gives uniform selection; "
    "(3) SN and limit at the low end of the ranges recommended in the ABCluster manual "
    "(sec. 2.4.3: SN 10-300, g_limit 3-5 generations without improvement, every "
    "exhausted member replaced) since a run affords only ~100-1000 relaxations; "
    "Karaboga's limit = SN*D would never trigger; (4) improvements below 1 meV "
    "(relaxation noise) count as failures. If a geometric step fails, the source is "
    "randomly displaced instead (counted in stats n_fallback).")
DEFAULTS = dict(sn=10, limit=3)

E_IMPROVE = 1e-3     # eV; smaller "improvements" are relaxation noise
MAX_PERM = 6         # bulk: lattice-basis candidates tried in the alignment
MAX_SHIFT = 8        # bulk: origin-shift candidates tried per basis
_SIGNS = np.array(list(itertools.product((1.0, -1.0), repeat=3)))
_P48 = np.array([np.diag(s)[list(perm)] for perm in itertools.permutations(range(3))
                 for s in _SIGNS])  # signed permutation matrices (rows: new axes)


def _wrap(f):
    return f - np.round(f)


def _std_cell(cell):
    """Right-handed lower-triangular cell with the same metric (a along x)."""
    return cellpar_to_cell(cell_to_cellpar(np.asarray(cell, float)))


def _min_height(cell):
    return float(1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0).max())


def _surface_translations(p, depth=2.0, tol=0.25):
    """In-plane substrate lattice translations (within the cell) mapping the slab onto itself."""
    out = [np.zeros(3)]
    if p.n_sub == 0:
        return np.array(out)
    X = np.asarray(p.sub_pos, float)
    sym = np.asarray(p.sub_symbols)
    top = np.flatnonzero(X[:, 2] > X[:, 2].max() - depth)
    species, counts = np.unique(sym[top], return_counts=True)
    ref = top[sym[top] == species[np.argmin(counts)]]
    inv = np.linalg.inv(p.cell0)
    F = X @ inv
    same = sym[top][:, None] == sym[None, :]
    for b in ref[1:]:
        t = X[b] - X[ref[0]]
        if abs(t[2]) > tol:
            continue
        t[2] = 0.0
        D = (F[top] + t @ inv)[:, None, :] - F[None, :, :]
        D -= np.where(p.pbc, np.round(D), 0.0)
        r = np.linalg.norm(D @ p.cell0, axis=-1)
        r[~same] = np.inf
        if np.all(r.min(axis=1) < tol):
            out.append(t)
    return np.array(out)


def _vacuum_window(p):
    """(f_top, gap, near_top) of the slab's vacuum gap along the third cell axis.

    The vacuum spans fractional c in (f_top, f_top + gap) (mod 1); near_top tells
    whether the search region lies in its lower half (next to the face at f_top).
    """
    inv = np.linalg.inv(p.cell0)
    fz = np.sort((np.asarray(p.sub_pos, float) @ inv)[:, 2] % 1.0)
    gaps = np.diff(np.append(fz, fz[0] + 1.0))
    j = int(np.argmax(gaps))
    f_top, gap = float(fz[j]), float(gaps[j])
    reg = p.region or {}
    if reg.get("type") == "box":
        c = 0.5 * (np.asarray(reg["lo"], float) + np.asarray(reg["hi"], float))
    elif reg.get("type") == "sphere":
        c = np.asarray(reg["center"], float)
    else:
        return None
    u = ((c @ inv)[2] - f_top) % 1.0
    if not 0.0 < u < gap:
        return None
    return f_top, gap, bool(u <= 0.5 * gap)


class _Source:
    """A food source: relaxed structure plus its alignment representation."""
    __slots__ = ("pos", "cell", "e", "T", "f")

    def __init__(self, pos, cell, e, T=None, f=None):
        self.pos, self.cell, self.e, self.T, self.f = pos, cell, e, T, f


class _Geometry:
    """Alignment and neighbour generation for one problem (no energy calls)."""

    def __init__(self, p):
        self.p = p
        self.s = p.search_idx
        self.ns = len(self.s)
        sym = np.array(p.symbols)[self.s]
        self.groups = [np.flatnonzero(sym == x) for x in sorted(set(sym))]
        self.small = min(self.groups, key=len)
        self.pbc = np.asarray(p.pbc, bool)
        self.inv0 = np.linalg.inv(p.cell0)
        if p.variable_cell or p.category == "bulk":
            self.mode = "bulk"
        elif p.category == "cluster" and p.n_sub == 0:
            self.mode = "cluster"
        elif p.category == "surface":
            self.mode = "surface"
        else:
            self.mode = "frame"
        self.lat = None
        self.vac = None
        self.n_far = 0
        if self.mode == "surface":
            try:
                self.lat = _surface_translations(p)
            except Exception:
                self.lat = np.zeros((1, 3))
            if p.n_sub:
                try:
                    self.vac = _vacuum_window(p)
                except Exception:
                    self.vac = None
        self.bond = p.bond_cut(self.s[:, None], self.s[None, :])

    # ------------------------------------------------------------ sources
    def source(self, pos, cell, e):
        p = self.p
        cell = np.array(cell, float)
        if self.mode != "bulk":
            return _Source(ops.unwrap_search(p, pos, cell), cell, float(e))
        rcell = T = cell
        if p.variable_cell:
            try:
                rcell = minkowski_reduce(cell)[0]
                if np.linalg.det(rcell) < 0:
                    rcell = -rcell
                T = _std_cell(rcell)
            except Exception:  # pure geometry, no energy calls inside
                rcell, T = cell, _std_cell(cell) if np.linalg.det(cell) > 0 else cell
        f = np.asarray(pos, float) @ np.linalg.inv(rcell)
        f -= np.floor(f)
        return _Source(np.array(pos, float), cell, float(e), T, f)

    # ---------------------------------------------------------- utilities
    def _mic(self, d):
        if not self.pbc.any():
            return d
        f = d @ self.inv0
        f -= np.where(self.pbc, np.round(f), 0.0)
        return f @ self.p.cell0

    def _assign(self, A, B, mic=False, metric=None):
        """Per-species Hungarian assignment: B[perm] best matches A (squared distances).

        mic: Cartesian minimum image in the fixed cell; metric: A, B are fractional
        coordinates (periodic) and metric is the cell used for distances.
        """
        perm = np.arange(len(A))
        total = 0.0
        for g in self.groups:
            d = A[g][:, None, :] - B[g][None, :, :]
            if metric is not None:
                d = _wrap(d) @ metric
            elif mic:
                d = self._mic(d)
            c = np.einsum("ijk,ijk->ij", d, d)
            if len(g) == 1:
                total += float(c[0, 0])
                continue
            r, q = linear_sum_assignment(c)
            perm[g[r]] = g[q]
            total += float(c[r, q].sum())
        return perm, total

    # ---------------------------------------------------------- alignment
    def _align_cluster(self, A, B):
        """Aligned copy of B (row j matches A[j]): Hungarian + orthogonal Procrustes."""
        ca = A.mean(axis=0)
        a = A - ca
        b = B - B.mean(axis=0)
        starts = [np.eye(3)]
        if self.ns >= 3:
            Ea = np.linalg.eigh(a.T @ a)[1]
            Eb = np.linalg.eigh(b.T @ b)[1]
            starts += [Eb @ np.diag(sg) @ Ea.T for sg in _SIGNS]
        best, best_cost = None, np.inf
        for O in starts:
            prev = None
            for _ in range(5):
                perm, _c = self._assign(a, b @ O)
                if prev is not None and np.array_equal(perm, prev):
                    break
                u, _s, vt = np.linalg.svd(b[perm].T @ a)
                O = u @ vt
                prev = perm
            bb = b[perm] @ O
            cost = float(((a - bb) ** 2).sum())
            if cost < best_cost:
                best, best_cost = bb, cost
        return ca + best

    def _align_surface(self, A, B):
        d = A.mean(axis=0) - B.mean(axis=0)
        d[2] = 0.0
        if self.lat is not None and len(self.lat):
            r = self._mic(d[None, :] - self.lat)
            B = B + (d - r[np.argmin((r ** 2).sum(axis=1))])
        perm, _c = self._assign(A, B, mic=True)
        return A - self._mic(A - B[perm])

    def _align_frame(self, A, B):
        perm, _c = self._assign(A, B, mic=True)
        return A - self._mic(A - B[perm])

    def _align_bulk(self, si, sk, rng):
        """Returns (fractional difference f_i - f_k(aligned) for search atoms, aligned cell of k)."""
        s = self.s
        fi, fk = si.f[s], sk.f[s]
        Ti, Tk = si.T, sk.T
        if self.p.variable_cell:
            Gi = Ti @ Ti.T
            G = np.einsum("pab,bc,pdc->pad", _P48, Tk @ Tk.T, _P48)
            dist = np.abs(G - Gi).sum(axis=(1, 2)) / np.trace(Gi)
            ties = np.flatnonzero(dist <= dist.min() + 0.02)
            first = int(np.argmin(dist))
            others = ties[ties != first]
            if len(others) > MAX_PERM - 1:
                others = rng.choice(others, MAX_PERM - 1, replace=False)
            cands = [first] + list(others)
        else:
            cands = [None]
        g0 = self.small
        a0 = g0[rng.integers(len(g0))]
        best = (np.inf, None, None, None)
        for c in cands:
            fk2 = fk if c is None else fk @ _P48[c].T
            bs = g0 if len(g0) <= MAX_SHIFT else rng.choice(g0, MAX_SHIFT, replace=False)
            for b in bs:
                fk3 = fk2 + (fi[a0] - fk2[b])
                perm, cost = self._assign(fi, fk3, metric=Ti)
                if cost < best[0]:
                    best = (cost, c, fk3, perm)
        _cost, c, fk3, perm = best
        d = _wrap(fi - fk3[perm])
        d = _wrap(d - d.mean(axis=0))  # remove the residual rigid shift
        Tk2 = Tk if c is None else _std_cell(_P48[c] @ Tk)
        return d, Tk2

    # ---------------------------------------------------------- neighbour
    @staticmethod
    def _subset(rng, n_units):
        """Random subset of units (size uniform in 1..n) with one phi ~ U[-1,1] each."""
        m = int(rng.integers(1, n_units + 1))
        return rng.choice(n_units, m, replace=False), rng.uniform(-1.0, 1.0, m)

    def neighbour(self, si, sk, rng):
        """v = x_i + phi (x_i - x_k) on a random subset of atoms (bulk: + lattice)."""
        if self.mode == "bulk":
            return self._neighbour_bulk(si, sk, rng)
        s = self.s
        A = si.pos[s]
        try:
            B = sk.pos[s]
            if self.mode == "cluster":
                Bk = self._align_cluster(A, B)
            elif self.mode == "surface":
                Bk = self._align_surface(A, B)
            else:
                Bk = self._align_frame(A, B)
        except (ValueError, np.linalg.LinAlgError):
            Bk = A - self._mic(A - sk.pos[s])
        sel, phi = self._subset(rng, self.ns)
        new = np.array(si.pos)
        new[s[sel]] += phi[:, None] * (A[sel] - Bk[sel])
        return self.repair(new, si.cell, rng), si.cell

    def _neighbour_bulk(self, si, sk, rng):
        p = self.p
        s = self.s
        try:
            d, Tk = self._align_bulk(si, sk, rng)
        except (ValueError, np.linalg.LinAlgError):
            d, Tk = _wrap(si.f[s] - sk.f[s]), si.T
        nu = self.ns + (1 if p.variable_cell else 0)
        sel, phi = self._subset(rng, nu)
        at = sel < self.ns
        f = np.array(si.f)
        f[s[sel[at]]] += phi[at][:, None] * d[sel[at]]
        T = si.T
        if not at.all():
            ph = float(phi[~at][0])
            v0 = abs(np.linalg.det(si.T))
            for _ in range(4):
                Tc = si.T + ph * (si.T - Tk)
                v = np.linalg.det(Tc)
                if 0.5 * v0 < v < 2.0 * v0 and _min_height(Tc) > 0.35 * v ** (1 / 3):
                    T = Tc
                    break
                ph *= 0.5
        if p.variable_cell:
            pos = f @ T  # whole system follows the (re-oriented) cell
        else:
            pos = np.array(si.pos)
            pos[s] = f[s] @ T  # frozen atoms keep their coordinates
        return ops.repair_overlaps(p, pos, T), T

    def _far_side(self, pos):
        """Search atoms in the far half of the periodic vacuum gap (slab underside)."""
        if self.vac is None:
            return self.s[:0]
        f_top, gap, near_top = self.vac
        v = ((pos[self.s] @ self.inv0)[:, 2] - f_top) % 1.0
        far = (v > 0.5 * gap) if near_top else (v < 0.5 * gap)
        return self.s[far & (v > 0.0) & (v < gap)]

    def repair(self, pos, cell, rng):
        """Remove overlaps; re-seat detached atoms and minor fragments on the cluster."""
        p = self.p
        pos = ops.repair_overlaps(p, pos, cell)
        lost = ops.detached_atoms(p, pos, cell)
        far = self._far_side(pos)
        if len(far):
            # a difference move carried atoms across the vacuum towards the
            # slab's underside: re-seat them on the cluster like detached atoms
            self.n_far += 1
            lost = np.union1d(lost, far).astype(int)
        if len(lost):
            # relocate_atoms returns None if every search atom is detached and
            # p.place fails; keep the overlap-free structure in that case
            new = ops.relocate_atoms(p, pos, rng, lost, cell)
            pos = pos if new is None else new
        if p.connected and self.ns > 2:
            s = self.s
            d = min_image_distances(pos[s], pos[s], cell, self.pbc)
            n, lab = connected_components(d < self.bond, directed=False)
            if n > 1:
                minor = s[lab != np.argmax(np.bincount(lab))]
                new = ops.relocate_atoms(p, pos, rng, minor, cell)
                pos = pos if new is None else new
        return pos


# ====================================================================== ABC
def _new_source(ctx, rng, geo, pop):
    """Random structure (initialisation / scout bee), relaxed; retried if aborted."""
    p = ctx.p
    while True:
        try:
            pos, cell = p.random_structure(rng)
        except RuntimeError:
            if not pop:
                raise
            src = pop[int(rng.integers(len(pop)))]
            pos, cell = ops.displace(p, src.pos, rng, 1.0, cell=src.cell), src.cell
        res = ctx.relax(pos, cell)
        if not res.aborted:
            return geo.source(res.pos, res.cell, res.e)
        ctx.stats["n_aborted"] += 1


def _forage(ctx, rng, geo, pop, i):
    """One employed/onlooker neighbour search on source i with greedy selection."""
    k = int(rng.integers(len(pop) - 1))
    k += k >= i
    src = pop[i]
    try:
        pos, cell = geo.neighbour(src, pop[k], rng)
    except Exception:  # pure geometry (no ctx calls): never let it crash the run
        ctx.stats["n_fallback"] = ctx.stats.get("n_fallback", 0) + 1
        pos, cell = ops.displace(ctx.p, src.pos, rng, 0.5, cell=src.cell), src.cell
    if geo.vac is not None:
        ctx.stats["n_far_side"] = geo.n_far
    res = ctx.relax(pos, cell)
    if res.aborted:
        ctx.stats["n_aborted"] += 1
        return False
    if res.e < src.e - E_IMPROVE:
        pop[i] = geo.source(res.pos, res.cell, res.e)
        return True
    return False


def run(ctx, rng, sn=10, limit=3):
    geo = _Geometry(ctx.p)
    sn = max(2, int(sn))
    limit = max(1, int(limit))
    st = ctx.stats
    for key in ("cycles", "n_init", "n_scout", "emp_try", "emp_ok", "onl_try", "onl_ok",
                "n_aborted"):
        st.setdefault(key, 0)
    if geo.vac is not None:
        st.setdefault("n_far_side", 0)
    pop = []
    for _ in range(sn):
        pop.append(_new_source(ctx, rng, geo, pop))
        st["n_init"] += 1
    trial = np.zeros(sn, int)
    while True:
        improved = np.zeros(sn, bool)
        # employed bees: one neighbour search per food source
        for i in range(sn):
            st["emp_try"] += 1
            if _forage(ctx, rng, geo, pop, i):
                improved[i] = True
                st["emp_ok"] += 1
        # onlooker bees: sources drawn with probability proportional to fitness
        e = np.array([x.e for x in pop])
        fit = 1.0 / (1.0 + (e - e.min()))
        for i in rng.choice(sn, size=sn, p=fit / fit.sum()):
            st["onl_try"] += 1
            if _forage(ctx, rng, geo, pop, int(i)):
                improved[i] = True
                st["onl_ok"] += 1
        # scout bees: abandon sources not improved for `limit` generations
        trial = np.where(improved, 0, trial + 1)
        for i in np.flatnonzero(trial >= limit):
            pop[i] = _new_source(ctx, rng, geo, pop)
            trial[i] = 0
            st["n_scout"] += 1
        st["cycles"] += 1
        st["pop_best"] = round(float(min(x.e for x in pop)), 4)
