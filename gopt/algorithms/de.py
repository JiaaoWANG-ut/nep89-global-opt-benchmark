"""Differential evolution (DE/rand/1/bin) with local relaxation of every trial (Lamarckian)."""
import itertools

import numpy as np
from ase.geometry import cell_to_cellpar, cellpar_to_cell, minkowski_reduce
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .. import ops
from ..problem import _images

LABEL = "Differential evolution"
FAMILY = "baseline"
REFERENCE = ("Storn & Price, J. Glob. Optim. 11, 341 (1997); Lamarckian DE for atomic clusters "
             "(PDECO): Chen, Jiang, Li, Li & Wang, J. Comput. Chem. 34, 1046 (2013); DE for crystal "
             "structures (IM2ODE): Zhang, Gao, Chen, Xiang & Gong, Comput. Mater. Sci. 98, 51 (2015)")
DESCRIPTION = (
    "Classic DE/rand/1/bin (Storn & Price) on a population of NP=20 locally relaxed structures "
    "(Lamarckian: every trial is relaxed with L-BFGS, variable-cell for bulk, and its minimum "
    "is what competes). Synchronous generations: for every target x_i a mutant "
    "v = a + F(b - c) is built from three other random members (F=0.5) and binomially "
    "crossed with x_i (CR=0.9, genes = atoms, i.e. 3-vectors, plus the cell as one gene for "
    "bulk; one random gene always comes from v); the relaxed trial replaces x_i only if it is "
    "lower (greedy one-to-one selection). Because atoms are permutable, b, c and the target are "
    "first aligned to the base a: per-species Hungarian assignment, plus rotation/reflection "
    "(Kabsch, principal-axes and random starts) for gas-phase clusters, integer lattice "
    "translations for clusters on/in periodic hosts (no rotation), and for bulk a "
    "Minkowski-reduced, standard-oriented cell with fractional coordinates aligned by origin "
    "shift/inversion; cell genes are lattice-vector differences. Deviations from the "
    "continuous-optimisation original, all needed for atoms: NP=20 instead of 5-10 x dimension "
    "(each member costs a relaxation); trials that relax onto a structure already in the "
    "population (fingerprint, ops.same_structure) are rejected and duplicates of the initial "
    "random population are re-drawn (max 2 NP draws), otherwise the Lamarckian population "
    "collapses and b-c vanishes; constraint handling: mutant atoms overlapping the frozen host "
    "fall back to the target's coordinates (DE bounce-back), remaining overlaps are pushed "
    "apart, detached atoms are re-seated on the cluster, implausible mutant cells (volume "
    "outside 0.5-2x the parents', flat) fall back to the target's cell.")
DEFAULTS = dict(pop_size=20, F=0.5, CR=0.9, rot_starts=4, init_draws=2.0)

_BAD = 1e5
_SIGNS = list(itertools.product((1.0, -1.0), repeat=3))


class _Ind:
    """Population member: relaxed structure plus its DE coordinates.

    x: fixed cell -> unwrapped Cartesian search-atom coordinates (m, 3);
       bulk -> fractional coordinates of all atoms (n, 3) in the canonical cell c.
    """
    __slots__ = ("e", "pos", "cell", "fp", "x", "c")

    def __init__(self, e, pos, cell, fp, x, c):
        self.e, self.pos, self.cell, self.fp, self.x, self.c = e, pos, cell, fp, x, c


def _wrap(d):
    return d - np.round(d)


def _kabsch(P, Q):
    """Orthogonal R (reflections allowed) minimising sum |p_k - R q_k|^2."""
    u, _, vt = np.linalg.svd(Q.T @ P)
    return vt.T @ u.T


def _axes(X):
    return np.linalg.eigh(X.T @ X)[1]


def _min_height(cell):
    return float(1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0).max())


def _canon(pos, cell):
    """Minkowski-reduced cell in standard orientation + wrapped fractional coordinates."""
    try:
        rcell, _ = minkowski_reduce(np.array(cell, float))
    except Exception:
        rcell = np.array(cell, float)
    frac = np.asarray(pos) @ np.linalg.inv(rcell)
    frac -= np.floor(frac)
    return frac, cellpar_to_cell(cell_to_cellpar(rcell))


class _Geo:
    """DE representation, alignment and trial construction for one problem."""

    def __init__(self, p, rot_starts=4):
        self.p = p
        self.s = p.search_idx
        self.m = len(self.s)
        sym = np.array(p.symbols)[self.s]
        self.groups = [np.where(sym == x)[0] for x in sorted(set(sym))]
        self.bulk = bool(p.variable_cell)
        self.rot = (not self.bulk) and p.category == "cluster" and p.n_sub == 0
        self.cell0 = np.array(p.cell0, float)
        self.inv0 = np.linalg.inv(self.cell0)
        self.pbc = np.array(p.pbc, bool)
        self.rot_starts = int(rot_starts)
        self.tree = None
        if p.n_sub and not self.bulk:
            # frozen host + its periodic images, for the mutant-overlap test
            shifts = _images(p.pbc) @ self.cell0
            xyz = (p.sub_pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
            self.tree = cKDTree(xyz)
            self.sub_idx = np.tile(np.arange(p.n_sub), len(shifts))
            self.sub_xyz = xyz
            self.r_sub = p.dmin_factor * (p.rcov[self.s].max() + p.rcov[: p.n_sub].max())

    # ------------------------------------------------------------ members
    def make(self, res):
        p = self.p
        pos, cell = np.array(res.pos, float), np.array(res.cell, float)
        fp = ops.fingerprint(p, pos, cell)
        if self.bulk:
            x, c = _canon(pos, cell)
        else:
            x, c = ops.unwrap_search(p, pos, self.cell0)[self.s], self.cell0
        return _Ind(float(res.e), pos, cell, fp, x, c)

    # ---------------------------------------------------------- alignment
    def _assign(self, ref, mov, metric=None):
        """Per-species optimal assignment: mov[perm] matches ref. Returns (perm, cost)."""
        perm = np.arange(self.m)
        cost = 0.0
        for g in self.groups:
            d = ref[g][:, None, :] - mov[g][None, :, :]
            if metric is not None:
                d = _wrap(d) @ metric
            C = np.einsum("ijk,ijk->ij", d, d)
            if len(g) == 1:
                cost += float(C[0, 0])
                continue
            r, c = linear_sum_assignment(C)
            perm[g[r]] = g[c]
            cost += float(C[r, c].sum())
        return perm, cost

    def _align_rot(self, ref, mov, rng):
        """Gas-phase cluster: translation + rotation/reflection + permutation onto ref.

        Starts: the 8 principal-axes frames (sign choices) and a few random rotations,
        screened by one assignment each; the best 3 are refined by alternating
        Hungarian assignment and Kabsch superposition.
        """
        cr = ref.mean(axis=0)
        R0 = ref - cr
        M0 = mov - mov.mean(axis=0)
        er, em = _axes(R0), _axes(M0)
        starts = [er @ np.diag(sg) @ em.T for sg in _SIGNS]
        starts += [ops.random_rotation(rng) for _ in range(self.rot_starts)]
        scored = []
        for rot in starts:
            X = M0 @ rot.T
            perm, cost = self._assign(R0, X)
            scored.append((cost, X, perm))
        scored.sort(key=lambda z: z[0])
        best_cost, best = np.inf, None
        for cost, X, perm in scored[:3]:
            for _ in range(6):
                X = X @ _kabsch(R0, X[perm]).T
                perm2, cost = self._assign(R0, X)
                if np.array_equal(perm2, perm):
                    break
                perm = perm2
            if cost < best_cost:
                best_cost, best = cost, X[perm2]
        return best + cr

    def _align_fixed(self, ref, mov):
        """Cluster on/in a periodic host: integer lattice translation + permutation."""
        f = (ref.mean(axis=0) - mov.mean(axis=0)) @ self.inv0
        mov = mov + np.where(self.pbc, np.round(f), 0.0) @ self.cell0
        perm, _ = self._assign(ref, mov)
        return mov[perm]

    def _align_frac(self, ref, metric, mov, rng):
        """Bulk: fractional coordinates of mov (all atoms) aligned onto ref (search atoms).

        Rigid origin shift (and optionally inversion, an exact symmetry of the PES) +
        per-species assignment; returns search-atom coordinates continuous with ref.
        """
        R = ref[self.s]
        M = mov[self.s]
        g0 = min(self.groups, key=len)
        anchors = g0 if len(g0) <= 6 else rng.choice(g0, 6, replace=False)
        lm = g0[int(rng.integers(len(g0)))]
        best_cost, best = np.inf, None
        for sgn in (1.0, -1.0):
            Mi = sgn * M
            for k in anchors:
                sh = R[k] - Mi[lm]
                for _ in range(3):
                    perm, _ = self._assign(R, Mi + sh, metric)
                    sh = sh + _wrap(R - (Mi[perm] + sh)).mean(axis=0)
                perm, cost = self._assign(R, Mi + sh, metric)
                if cost < best_cost:
                    best_cost, best = cost, R - _wrap(R - (Mi[perm] + sh))
        return best

    # -------------------------------------------------------------- trials
    def trial(self, t, a, b, c, F, CR, rng):
        """Mutant a + F (b - c), binomial crossover with target t, constraint handling."""
        if self.bulk:
            return self._trial_bulk(t, a, b, c, F, CR, rng)
        p = self.p
        A = a.x
        if self.rot:
            B, C, T = (self._align_rot(A, z.x, rng) for z in (b, c, t))
        else:
            B, C, T = (self._align_fixed(A, z.x) for z in (b, c, t))
        V = A + F * (B - C)
        mask = rng.random(self.m) < CR
        mask[rng.integers(self.m)] = True
        U = np.where(mask[:, None], V, T)
        if self.tree is not None:
            bad = mask & self._host_overlap(U)
            U[bad] = T[bad]
        pos = np.array(a.pos, float)
        pos[self.s] = U
        pos = ops.repair_overlaps(p, pos, self.cell0)
        lost = ops.detached_atoms(p, pos, self.cell0)
        if len(lost):
            pos = ops.relocate_atoms(p, pos, rng, lost, self.cell0)
            if pos is None:
                raise ValueError("re-seating detached atoms failed")
        return pos, self.cell0

    def _host_overlap(self, X):
        """Mask of search atoms closer than dmin to any host atom."""
        p = self.p
        f = X @ self.inv0
        xw = X - np.where(self.pbc, np.floor(f), 0.0) @ self.cell0
        out = np.zeros(len(X), bool)
        for k, hits in enumerate(self.tree.query_ball_point(xw, self.r_sub)):
            if hits:
                j = self.sub_idx[hits]
                d = np.linalg.norm(self.sub_xyz[hits] - xw[k], axis=1)
                out[k] = bool(np.any(d < p.dmin(self.s[k], j)))
        return out

    def _trial_bulk(self, t, a, b, c, F, CR, rng):
        p = self.p
        A = a.x[self.s]
        B, C, T = (self._align_frac(a.x, a.c, z.x, rng) for z in (b, c, t))
        V = A + F * (B - C)
        mask = rng.random(self.m + 1) < CR
        mask[rng.integers(self.m + 1)] = True
        U = np.where(mask[: self.m, None], V, T)
        cell = t.c
        if mask[self.m]:
            cv = a.c + F * (b.c - c.c)
            vols = [abs(np.linalg.det(z.c)) for z in (a, b, c)]
            v = np.linalg.det(cv)
            if 0.5 * min(vols) < v < 2.0 * max(vols) and _min_height(cv) > 0.3 * v ** (1 / 3):
                cell = cv
        frac = np.array(a.x, float)
        frac[self.s] = U
        frac -= np.floor(frac)
        pos = ops.repair_overlaps(p, frac @ cell, cell)
        return pos, np.array(cell, float)


def _ok(res):
    return (not res.aborted) and np.isfinite(res.e) and res.e < _BAD


def _dup(p, ind, pop, skip=-1):
    for j, z in enumerate(pop):
        if j != skip and ops.same_structure(p, ind.e, ind.fp, z.e, z.fp):
            return j
    return -1


def _member(geo, res):
    """Population member from a relaxation result, or None if its geometry is unusable."""
    try:
        ind = geo.make(res)
    except Exception:  # e.g. degenerate cell in the canonicalisation
        return None
    if not (np.all(np.isfinite(ind.x)) and np.all(np.isfinite(ind.c)) and np.all(np.isfinite(ind.fp))):
        return None
    return ind


def run(ctx, rng, pop_size=20, F=0.5, CR=0.9, rot_starts=4, init_draws=2.0):
    p = ctx.p
    geo = _Geo(p, rot_starts)
    NP = max(4, int(pop_size))
    st = dict(gen=0, trials=0, replaced=0, dup=0, invalid=0, fallback=0, init_draws=0)
    ctx.stats.update(st)

    # initial population: random structures, relaxed; duplicates re-drawn (max init_draws*NP)
    pop = []
    while len(pop) < NP:
        st["init_draws"] += 1
        ctx.stats.update(st)
        pos, cell = p.random_structure(rng)
        res = ctx.relax(pos, cell)
        ind = _member(geo, res) if _ok(res) else None
        if ind is None:
            st["invalid"] += 1
            continue
        if st["init_draws"] <= init_draws * NP and _dup(p, ind, pop) >= 0:
            st["dup"] += 1
            continue
        pop.append(ind)
    ctx.stats.update(st)

    while True:
        new = list(pop)
        for i in range(NP):
            r1, r2, r3 = rng.choice([j for j in range(NP) if j != i], 3, replace=False)
            try:
                pos, cell = geo.trial(pop[i], pop[r1], pop[r2], pop[r3], F, CR, rng)
                if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(cell))):
                    raise ValueError("non-finite trial")
            except Exception:  # geometry failure: fall back to a random structure
                st["fallback"] += 1
                pos, cell = p.random_structure(rng)
            res = ctx.relax(pos, cell)
            st["trials"] += 1
            if not _ok(res):
                st["invalid"] += 1
            elif res.e < pop[i].e:
                ind = _member(geo, res)
                if ind is None:
                    st["invalid"] += 1
                elif _dup(p, ind, new, skip=i) >= 0:
                    st["dup"] += 1
                else:
                    new[i] = ind
                    st["replaced"] += 1
            ctx.stats.update(st)
        pop = new
        st["gen"] += 1
        es = np.array([z.e for z in pop])
        st.update(pop_min=round(float(es.min()), 4), pop_max=round(float(es.max()), 4))
        ctx.stats.update(st)
