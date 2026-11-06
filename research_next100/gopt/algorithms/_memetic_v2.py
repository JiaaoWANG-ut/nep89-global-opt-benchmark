"""Adaptive memetic algorithm (AMA): GA recombination + basin-hopping chains with
credit-based allocation of ionic steps, meta-Lamarckian move selection and
fingerprint niching."""
import bisect

import numpy as np
from ase.geometry import cell_to_cellpar, cellpar_to_cell, minkowski_reduce

from .. import ops
from ..context import BudgetExhausted

LABEL = "Adaptive memetic (GA + BH chains)"
FAMILY = "innovative"
REFERENCE = ("Moscato, Caltech C3P Report 826 (1989); Ong & Keane, IEEE TEVC 8, 99 (2004) and "
             "Ong, Lim & Chen, IEEE Comput. Intell. Mag. 5, 24 (2010) (meta-Lamarckian / adaptive "
             "memetic algorithms); Deaven & Ho, PRL 75, 288 (1995); Johnston, Dalton Trans. 4193 "
             "(2003); Oganov & Glass, JCP 124, 244704 (2006); Wales & Doye, JPCA 101, 5111 (1997)")
DESCRIPTION = (
    "A small, niched population of relaxed structures (size 4 + 0.15 n_search, <= 12) is evolved by "
    "three competing operators: global recombination (Deaven-Ho cut-and-splice with a "
    "registry-preserving lateral shift on surfaces; USPEX-like lattice heredity with cell "
    "averaging for bulk), local intensification by short Metropolis basin-hopping chains started "
    "from (preferably good, not yet exhausted) members, and random immigrants. Novelty: (i) the "
    "ionic-step budget is allocated online by credit assignment - each operator earns a "
    "rank-based reward (1 for a new global best, 2^-k/2 for entering the population at rank k, "
    "0 for duplicates/rejections) and is chosen with probability proportional to its "
    "exponentially-forgotten reward per ionic step (with a floor), so the method behaves like a "
    "GA, a BH or a random search depending on which currently pays off; (ii) BH chains are "
    "Lamarckian and self-terminating (they run while they keep improving, at most `patience` "
    "non-improving relaxations), and their move type (shake, twist, energy-guided surface "
    "relocation, unlike-atom swap, rigid move on supports, lattice strain) is chosen by a second "
    "meta-Lamarckian credit level; (iii) diversity is kept by fingerprint niching (duplicates "
    "are merged, near-duplicates compete only with their niche) and by elitist immigration of "
    "random structures into the worse half after stagnation.")
DEFAULTS = dict(pop_base=4.0, pop_per_atom=0.15, pop_max=12, kT=0.1, step=0.3, patience=3,
                chain_max=12, decay=0.9, p_min=0.08, move_p_min=0.1, stagnation=6.0, niche=2.5,
                strain=0.08, xo_mut=0.1, adapt=True)

_EPS_BEST = 1e-3  # eV, improvement that counts as a new best


class _Ind:
    __slots__ = ("pos", "cell", "e", "fp", "ea", "nfail")

    def __init__(self, pos, cell, e, fp, ea):
        self.pos, self.cell, self.e, self.fp, self.ea = pos, cell, e, fp, ea
        self.nfail = 0


class _Credit:
    """Probability matching on exponentially forgotten reward per ionic step."""

    def __init__(self, names, r0, c0, decay, p_min, adapt=True):
        self.names = list(names)
        self.adapt = bool(adapt)
        self.r0, self.c0 = r0, c0
        self.decay = decay
        self.p_min = min(p_min, 0.5 / max(len(self.names), 1))
        self.used = np.zeros(len(self.names), int)
        self.steps = np.zeros(len(self.names), int)
        self.gain = np.zeros(len(self.names))
        self.reset()

    def reset(self):
        self.R = np.full(len(self.names), self.r0, float)
        self.C = np.full(len(self.names), self.c0, float)

    def probs(self):
        k = len(self.names)
        if not self.adapt:
            return np.full(k, 1.0 / k)
        v = self.R / np.maximum(self.C, 1e-9)
        s = v.sum()
        if not np.isfinite(s) or s <= 0:
            return np.full(k, 1.0 / k)
        q = self.p_min + (1.0 - k * self.p_min) * v / s
        return q / q.sum()

    def pick(self, rng):
        return int(rng.choice(len(self.names), p=self.probs()))

    def update(self, i, reward, cost):
        self.R[i] = self.decay * self.R[i] + reward
        self.C[i] = self.decay * self.C[i] + max(cost, 1)
        self.used[i] += 1
        self.steps[i] += cost
        self.gain[i] += reward


class _Memetic:
    def __init__(self, ctx, rng, pop_base, pop_per_atom, pop_max, kT, step, patience, chain_max,
                 decay, p_min, move_p_min, stagnation, niche, strain, xo_mut, adapt):
        self.ctx, self.rng = ctx, rng
        p = self.p = ctx.p
        self.bulk = bool(p.variable_cell)
        self.cluster_like = not self.bulk and p.category != "bulk"
        self.N = int(np.clip(round(pop_base + pop_per_atom * p.n_search), 4, max(4, pop_max)))
        self.kT, self.patience, self.chain_max = kT, int(patience), int(chain_max)
        self.strain, self.xo_mut = strain, xo_mut
        s = p.search_idx
        self.d0 = float(2.0 * np.mean(p.rcov[s]))  # typical bond length
        self.step = step * self.d0
        self.step_lo, self.step_hi = 0.1 * self.d0, 0.8 * self.d0
        self.dfp = 0.01 if self.bulk else 0.03
        self.r_niche = niche * self.dfp
        self.sym = np.array(p.symbols)
        self.multi = len(set(self.sym[s])) > 1
        self.stag_limit = max(15, int(round(stagnation * self.N)))
        self.decay, self.p_min, self.move_p_min = decay, p_min, move_p_min
        self.adapt = adapt
        self.arch_e, self.arch = [], []  # archive of distinct minima (sorted by energy)
        self.cur_op = "init"
        self.best_log = []
        moves = ["shake"]
        if self.bulk:
            moves.append("strain")
        else:
            if p.n_search >= 4:
                moves.append("twist")
            moves.append("surf")
            if p.n_sub:
                moves.append("rigid")
        if self.multi:
            moves.append("swap")
        self.move_names = moves
        self.pop = []
        self.since_best = 0
        self.n_immig = 0
        self.n_dup = 0
        self.c_rel = 50.0
        self._setup_registry()

    # ------------------------------------------------------------ helpers
    def _setup_registry(self):
        """Top-layer sublattice of a slab: lateral shifts that keep adsorption-site registry."""
        self.reg = None
        p = self.p
        if p.category != "surface" or not p.n_sub:
            return
        try:
            z = p.sub_pos[:, 2]
            top = np.where(z > z.max() - 0.7)[0]
            syms = self.sym[top]
            vals, cnt = np.unique(syms, return_counts=True)
            top = top[syms == vals[np.argmax(cnt)]]
            xy = p.sub_pos[top]
            self.reg = (xy - xy[0])[:, :2]
            self.reg_cell = p.cell0[:2, :2]
            self.reg_inv = np.linalg.inv(self.reg_cell)
        except Exception:
            self.reg = None

    def _registry_shift(self, v):
        """Lateral sublattice vector closest to the desired 2D shift v."""
        if self.reg is None:
            return np.array([v[0], v[1], 0.0])
        w = v[None, :2] - self.reg
        f = w @ self.reg_inv
        f -= np.round(f)
        w = f @ self.reg_cell
        j = int(np.argmin((w ** 2).sum(axis=1)))
        t = v[:2] - w[j]
        return np.array([t[0], t[1], 0.0])

    def _std_cell(self, pos, cell):
        """Minkowski-reduced, standard-orientation cell (energy invariant)."""
        if not self.bulk or self.p.n_sub:
            return pos, cell
        try:
            rcell, _ = minkowski_reduce(cell)
            new = cellpar_to_cell(cell_to_cellpar(rcell))
            frac = np.asarray(pos) @ np.linalg.inv(rcell)
            frac -= np.floor(frac)
            out = frac @ new
            if np.all(np.isfinite(out)) and np.all(np.isfinite(new)):
                return out, new
        except Exception:
            pass
        return pos, cell

    def _make(self, res):
        """Population member from a relaxation result (None if unusable)."""
        if res.aborted or not np.isfinite(res.e) or res.e > 1e5:
            return None
        ea = self.ctx.atomic_energies()
        pos, cell = self._std_cell(np.array(res.pos), np.array(res.cell))
        try:
            fp = ops.fingerprint(self.p, pos, cell)
        except Exception:
            return None
        if ea is not None:
            ea = np.asarray(ea)[self.p.search_idx]
        return _Ind(pos, cell, float(res.e), fp, ea)

    def _relax(self, pos, cell):
        """Relax; returns (member or None, ionic steps used)."""
        n0 = self.ctx.nsteps
        res = self.ctx.relax(pos, cell)
        return self._make(res), self.ctx.nsteps - n0

    def _fix_detached(self, pos, cell):
        if not self.cluster_like:
            return pos
        lost = ops.detached_atoms(self.p, pos, cell)
        if len(lost):
            pos = ops.relocate_atoms(self.p, pos, self.rng, lost, cell)
        return pos

    def _valid(self, pos, cell):
        return (pos is not None and np.all(np.isfinite(pos)) and
                (cell is None or (np.all(np.isfinite(cell)) and abs(np.linalg.det(cell)) > 1e-3)))

    # --------------------------------------------------------- population
    def _same(self, a, b):
        return ops.same_structure(self.p, a.e, a.fp, b.e, b.fp, dfp=self.dfp)

    def _lookup(self, ind):
        """Archived minimum identical to ind (or None)."""
        lo = bisect.bisect_left(self.arch_e, ind.e - 0.0101)
        hi = bisect.bisect_right(self.arch_e, ind.e + 0.0101)
        for k in range(lo, hi):
            if self._same(ind, self.arch[k]):
                return self.arch[k]
        return None

    def _archive(self, ind):
        if len(self.arch) < 20000:
            k = bisect.bisect_left(self.arch_e, ind.e)
            self.arch_e.insert(k, ind.e)
            self.arch.insert(k, _Ind(None, None, ind.e, ind.fp, None))

    def offer(self, ind):
        """Niche-based insertion. Returns (reward, status).

        Only novel minima (not in the archive of everything seen) earn a reward; a
        rediscovery may only refresh the coordinates of the identical member."""
        if ind is None:
            return 0.0, "bad"
        pop = self.pop
        best_before = pop[0].e if pop else np.inf
        if self._lookup(ind) is not None:
            for j, m in enumerate(pop):
                if self._same(ind, m):
                    if ind.e < m.e:
                        ind.nfail = m.nfail
                        pop[j] = ind
                        pop.sort(key=lambda x: x.e)
                    break
            self.n_dup += 1
            return 0.0, "dup"
        self._archive(ind)
        if pop:
            d = np.array([ops.fp_distance(self.p, ind.fp, m.fp) for m in pop])
            j = int(np.argmin(d))
        else:
            d, j = None, -1
        if d is not None and d[j] < self.r_niche:
            if ind.e < pop[j].e:
                pop[j] = ind
            else:
                return 0.0, "niche"
        elif len(pop) < self.N:
            pop.append(ind)
        elif ind.e < pop[-1].e:
            pop[-1] = ind
        else:
            return 0.0, "reject"
        pop.sort(key=lambda x: x.e)
        if ind.e < best_before - _EPS_BEST:
            self.since_best = 0
            if len(self.best_log) < 60:
                self.best_log.append([int(self.ctx.nsteps), self.cur_op])
            return 1.0, "best"
        k = next(i for i, m in enumerate(pop) if m is ind)
        return 0.5 * 0.5 ** k, "in"

    def _fitness(self):
        e = np.array([m.e for m in self.pop])
        if len(e) == 1 or e.max() - e.min() < 1e-9:
            return np.full(len(e), 1.0 / len(e))
        rho = (e - e.min()) / (e.max() - e.min())
        f = 0.5 * (1.0 - np.tanh(2.0 * rho - 1.0))
        return f / f.sum()

    # ---------------------------------------------------------- operators
    def random_member(self):
        p = self.p
        try:
            pos, cell = p.random_structure(self.rng)
        except Exception:
            base = self.pop[0] if self.pop else None
            if base is None:
                raise
            pos, cell = self._move("shake", base, big=True)
        return self._relax(pos, cell)

    def _cut_splice(self, A, B):
        """Deaven-Ho cut-and-splice (composition preserving)."""
        p, rng = self.p, self.rng
        cell = p.cell0
        a = ops.unwrap_search(p, A.pos, cell)
        b = ops.unwrap_search(p, B.pos, cell)
        s = p.search_idx
        sym = self.sym
        n = ops.random_unit(rng)
        if p.category == "surface":
            n[2] = 0.0
            nn = np.linalg.norm(n)
            n = n / nn if nn > 1e-8 else np.array([1.0, 0.0, 0.0])
        ca, cb = a[s].mean(axis=0), b[s].mean(axis=0)
        if p.category == "surface":
            shift = self._registry_shift(ca - cb)
        else:
            shift = ca - cb
        pa = (a[s] - ca) @ n
        pb = (b[s] + shift - ca) @ n
        need = {}
        for x in sym[s]:
            need[x] = need.get(x, 0) + 1
        n_from_a = int(np.clip((pa >= 0).sum(), 1, len(s) - 1))
        chosen = []
        for k in np.argsort(-pa)[:n_from_a]:
            i = s[k]
            if need[sym[i]] > 0:
                chosen.append((sym[i], a[i]))
                need[sym[i]] -= 1
        for k in np.argsort(pb):
            i = s[k]
            if need[sym[i]] > 0:
                chosen.append((sym[i], b[i] + shift))
                need[sym[i]] -= 1
        child = np.array(a)
        slots = {x: [i for i in s if sym[i] == x] for x in set(sym[s])}
        for x, xyz in chosen:
            child[slots[x].pop(0)] = xyz
        child = ops.repair_overlaps(p, child, cell)
        return child, cell

    def _heredity(self, A, B):
        """Lattice heredity (Oganov & Glass): slab of A + complementary slab of B."""
        p, rng = self.p, self.rng
        sym = self.sym
        fa = A.pos @ np.linalg.inv(A.cell)
        fb = B.pos @ np.linalg.inv(B.cell)
        ax = int(rng.integers(3))
        w = rng.uniform(0.3, 0.7)
        fa[:, ax] = (fa[:, ax] - rng.random()) % 1.0
        fb[:, ax] = (fb[:, ax] - rng.random()) % 1.0
        t = rng.uniform(0.3, 0.7)
        cell = t * A.cell + (1 - t) * B.cell
        vol = abs(np.linalg.det(cell))
        vmin = min(abs(np.linalg.det(A.cell)), abs(np.linalg.det(B.cell)))
        heights = 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)
        if not np.isfinite(vol) or vol < 0.7 * vmin or heights.min() < 0.35 * vol ** (1 / 3):
            cell = A.cell.copy()
        frac = np.zeros((p.n, 3))
        for x in set(sym):
            idx = np.where(sym == x)[0]
            ina = idx[fa[idx, ax] < w]
            inb = idx[fb[idx, ax] >= w]
            # distance to the slab boundaries (atoms near a cut are dropped first)
            da = np.minimum(fa[ina, ax], w - fa[ina, ax])
            db = np.minimum(fb[inb, ax] - w, 1.0 - fb[inb, ax])
            cand = [(da[k] + 1e-3 * rng.random(), fa[i]) for k, i in enumerate(ina)]
            cand += [(db[k] + 1e-3 * rng.random(), fb[i]) for k, i in enumerate(inb)]
            cand.sort(key=lambda c: -c[0])
            cand = [c[1] for c in cand[:len(idx)]]
            if len(cand) < len(idx):
                # fill from the unused atoms of both parents
                rest = [fa[i] for i in idx if fa[i, ax] >= w] + [fb[i] for i in idx if fb[i, ax] < w]
                order = rng.permutation(len(rest))
                for k in order[:len(idx) - len(cand)]:
                    cand.append(rest[k])
            for k, i in enumerate(idx):
                frac[i] = cand[k]
        pos = frac @ cell
        pos = ops.repair_overlaps(p, pos, cell)
        return pos, cell

    def crossover(self):
        f = self._fitness()
        i, j = self.rng.choice(len(self.pop), size=2, replace=False, p=f)
        A, B = self.pop[i], self.pop[j]
        if self.rng.random() < 0.5:
            A, B = B, A
        if self.bulk:
            pos, cell = self._heredity(A, B)
        else:
            pos, cell = self._cut_splice(A, B)
        if self.rng.random() < self.xo_mut:
            tmp = _Ind(pos, cell, A.e, A.fp, None)
            pos, cell = self._move("swap" if self.multi and self.rng.random() < 0.5 else "shake",
                                   tmp, small=True)
        pos = self._fix_detached(pos, cell)
        return pos, cell

    def _move(self, kind, ind, big=False, small=False):
        """Perturb a member; returns (pos, cell) (not relaxed)."""
        p, rng = self.p, self.rng
        pos, cell = np.array(ind.pos), np.array(ind.cell)
        s = p.search_idx
        step = self.step * (2.0 if big else 0.5 if small else 1.0)
        if kind == "shake":
            if self.bulk:
                pos, cell = ops.strain_cell(pos, cell, rng, self.strain)
            pos = ops.displace(p, pos, rng, step, cell=cell)
        elif kind == "strain":
            pos, cell = ops.strain_cell(pos, cell, rng, 2.0 * self.strain)
            pos = ops.displace(p, pos, rng, 0.25 * step, cell=cell)
        elif kind == "swap":
            pos = ops.swap_unlike(p, pos, rng, nswap=1 + int(rng.random() < 0.3))
            pos = ops.displace(p, pos, rng, 0.1 * self.d0, cell=cell)
        elif kind == "twist":
            pos = ops.unwrap_search(p, pos, cell)
            c = pos[s].mean(axis=0)
            if p.category == "surface":
                axis = np.array([0.0, 0.0, 1.0])
                side = (pos[s, 2] - c[2]) > 0
                if side.sum() == 0 or side.sum() == len(s):
                    nrm = ops.random_unit(rng)
                    nrm[2] = 0.0
                    nrm /= max(np.linalg.norm(nrm), 1e-8)
                    side = (pos[s] - c) @ nrm > 0
            else:
                axis = ops.random_unit(rng)
                side = (pos[s] - c) @ axis > 0
            if side.sum() == 0:
                side[rng.integers(len(s))] = True
            rot = ops.rotation_about(axis, rng.uniform(0.25, 1.75) * np.pi)
            idx = s[side]
            pos[idx] = (pos[idx] - c) @ rot.T + c
            pos = ops.repair_overlaps(p, pos, cell)
        elif kind == "surf":
            ea = ind.ea
            if ea is None or len(ea) != len(s):
                k = rng.choice(len(s), size=1, replace=False)
            else:
                z = np.array(ea, float)
                for x in set(self.sym[s]):
                    m = self.sym[s] == x
                    z[m] -= z[m].mean()
                nk = min(len(s) - 1, 1 + int(rng.random() < 0.3))
                top = np.argsort(-z)[:min(len(s), 3)]
                k = rng.choice(top, size=max(1, min(nk, len(top))), replace=False)
            pos = ops.relocate_atoms(p, pos, rng, s[np.atleast_1d(k)], cell)
            if pos is None:
                pos = ops.displace(p, np.array(ind.pos), rng, step, cell=cell)
        elif kind == "rigid":
            pos = ops.unwrap_search(p, pos, cell)
            c = pos[s].mean(axis=0)
            if p.category == "surface":
                rot = ops.rotation_about([0, 0, 1], rng.uniform(0, 2 * np.pi))
                t = ops.random_unit(rng)
                t[2] = 0.0
            else:
                rot = ops.random_rotation(rng)
                t = ops.random_unit(rng)
            pos[s] = (pos[s] - c) @ rot.T + c + t * rng.uniform(0.0, 0.5) * self.d0
            pos = ops.repair_overlaps(p, pos, cell)
        else:
            pos = ops.displace(p, pos, rng, step, cell=cell)
        pos = self._fix_detached(pos, cell)
        return pos, cell

    # --------------------------------------------------------------- main
    def bh_chain(self, arm):
        """Lamarckian Metropolis BH chain from a member; credits each relaxation."""
        rng = self.rng
        n = len(self.pop)
        ranks = np.arange(n)
        w = np.exp(-ranks / max(1.0, n / 4.0)) * np.array([0.5 ** min(m.nfail, 10) for m in self.pop])
        start = self.pop[int(rng.choice(n, p=w / w.sum()))]
        cur = start
        chain_best = start.e
        stall = 0
        improved = False
        for _ in range(self.chain_max):
            mi = self.moves.pick(rng)
            kind = self.move_names[mi]
            self.cur_op = "bh:" + kind
            pos, cell = self._safe_move(kind, cur)
            new, used = self._relax(pos, cell)
            r, status = self.offer(new)
            self.arms.update(arm, r, used)
            self._count_relax(r)
            if new is None:
                self.moves.update(mi, 0.0, used)
                stall += 1
            else:
                same = self._same(new, cur)
                better = new.e < cur.e - _EPS_BEST and not same
                self.moves.update(mi, 1.0 if better else 0.0, used)
                if kind == "shake":
                    if same:
                        self.step *= 1.15
                    elif new.e > cur.e + self.kT:
                        self.step *= 0.93
                    self.step = float(np.clip(self.step, self.step_lo, self.step_hi))
                de = new.e - cur.e
                if not same and (de < 0 or rng.random() < np.exp(-de / self.kT)):
                    cur = new
                if new.e < chain_best - _EPS_BEST:
                    chain_best = new.e
                    stall = 0
                    improved = True
                else:
                    stall += 1
            if stall >= self.patience:
                break
        if not improved:
            start.nfail += 1

    def _safe_move(self, kind, ind):
        try:
            pos, cell = self._move(kind, ind)
            if self._valid(pos, cell):
                return pos, cell
        except BudgetExhausted:
            raise
        except Exception:
            pass
        return ops.displace(self.p, np.array(ind.pos), self.rng, self.step, cell=ind.cell), ind.cell

    def _count_relax(self, r):
        if r < 1.0:
            self.since_best += 1

    def immigrate(self):
        """Elitist restart of the worse half after stagnation."""
        self.n_immig += 1
        self.cur_op = "immig"
        keep = max(1, self.N - max(1, self.N // 2))
        self.pop = self.pop[:keep]
        for m in self.pop:
            m.nfail = 0
        for _ in range(self.N - keep):
            new, used = self.random_member()
            r, _ = self.offer(new)
            self.arms.update(self.arm_index["rnd"], r, used)
        self.arms.reset()
        self.moves.reset()
        self.since_best = 0

    def loop(self):
        ctx, rng = self.ctx, self.rng
        steps = []
        for _ in range(self.N):
            new, used = self.random_member()
            steps.append(used)
            self.offer(new)
        self.c_rel = float(np.clip(np.mean(steps), 5, 500))
        arm_names = ["xo", "bh", "rnd"]
        self.arm_index = {a: i for i, a in enumerate(arm_names)}
        self.arms = _Credit(arm_names, 0.25, self.c_rel, self.decay, self.p_min, self.adapt)
        self.moves = _Credit(self.move_names, 0.25, self.c_rel, self.decay, self.move_p_min,
                             self.adapt)
        while True:
            if len(self.pop) < 2:
                self.cur_op = "rnd"
                new, used = self.random_member()
                r, _ = self.offer(new)
                self._count_relax(r)
                continue
            a = self.arms.pick(rng)
            name = arm_names[a]
            self.cur_op = name
            if name == "bh":
                self.bh_chain(a)
            else:
                if name == "xo":
                    try:
                        pos, cell = self.crossover()
                        ok = self._valid(pos, cell)
                    except BudgetExhausted:
                        raise
                    except Exception:
                        ok = False
                    if not ok:
                        pos, cell = self._safe_move("shake", self.pop[0])
                    new, used = self._relax(pos, cell)
                else:
                    new, used = self.random_member()
                r, _ = self.offer(new)
                self.arms.update(a, r, used)
                self._count_relax(r)
            if self.since_best >= self.stag_limit:
                self.immigrate()
            ctx.stats.update(
                pop=self.N, arm_used=dict(zip(arm_names, self.arms.used.tolist())),
                arm_steps=dict(zip(arm_names, self.arms.steps.tolist())),
                arm_gain=dict(zip(arm_names, np.round(self.arms.gain, 2).tolist())),
                arm_p=dict(zip(arm_names, np.round(self.arms.probs(), 3).tolist())),
                move_used=dict(zip(self.move_names, self.moves.used.tolist())),
                move_gain=dict(zip(self.move_names, np.round(self.moves.gain, 2).tolist())),
                step=round(self.step, 3), immig=self.n_immig, dup=self.n_dup,
                best_by=self.best_log[-1][1] if self.best_log else "init",
                best_log=self.best_log, archive=len(self.arch))


def run(ctx, rng, **params):
    kw = dict(DEFAULTS)
    kw.update(params)
    _Memetic(ctx, rng, **kw).loop()
