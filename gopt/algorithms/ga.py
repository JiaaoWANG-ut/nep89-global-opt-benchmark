"""Genetic algorithm: steady-state cut-and-splice cluster GA (BCGA / ASE-GA style).

Population of locally relaxed structures (Lamarckian GA). New candidates are
either Deaven-Ho cut-and-splice children of two roulette-selected parents or
mutants of one parent; every candidate is relaxed and replaces the worst member
if it is better and not a duplicate. Bulk problems use the periodic slab cut in
fractional coordinates with lattice heredity and strain mutation.
"""
import math

import numpy as np
from ase.build.tools import niggli_reduce_cell

from .. import ops

LABEL = "Genetic algorithm"
FAMILY = "baseline"
REFERENCE = ("Deaven & Ho, PRL 75, 288 (1995); Johnston, Dalton Trans. 2003, 4193 (BCGA); "
             "Vilhelmsen & Hammer, JCP 141, 044711 (2014) (ASE-GA); "
             "Abraham & Probert, PRB 73, 224104 (2006)")
DESCRIPTION = (
    "Steady-state Lamarckian GA with a population of pop_size=20 relaxed structures. "
    "Parents are drawn by roulette wheel on Johnston's tanh fitness "
    "F=0.5[1-tanh(2rho-1)] of the normalised energy rho, multiplied by the ASE-GA history "
    "factors 1/sqrt(1+n_paired) and 1/sqrt(1+n_duplicates_found). A fraction p_mut=0.2 of the new "
    "candidates are mutants of one parent; the rest are crossover children. "
    "Clusters: Deaven-Ho cut-and-splice (random plane through the parent centres, composition kept). "
    "For gas-phase clusters parent B is moved onto A's centre. For supported clusters the plane is "
    "vertical and B is only shifted sideways, by the surface lattice vector closest to the centre "
    "offset so the adsorption sites stay the same. Zeolite clusters are not translated (ASE-GA). "
    "Cluster mutations, one picked at random: rattle (ASE: each atom with p=0.4 by up to +-0.8 A), "
    "mirror half of the cluster through a random plane (ASE MirrorMutation), "
    "twist one half by a random angle (BCGA rotation), relocate 1..ceil(0.2N) random atoms to "
    "surface sites (BCGA atom replacement), and permutation of unlike atoms (ASE, ceil(0.33N/2) "
    "swaps; only for alloys). "
    "Bulk: periodic slab cut of Abraham-Probert in fractional coordinates. Parent A gives the "
    "atoms between 1/4 and 3/4 along a random cell axis and B gives the rest, after random "
    "fractional offsets. The lattice is inherited as w*h_A+(1-w)*h_B with volume w*V_A+(1-w)*V_B, "
    "w~U(0,1), from Niggli-reduced parents. Bulk mutations: strain (ASE/USPEX, symmetric "
    "sigma=0.7, rescaled to the parent volume, flat cells rejected), rattle and permutation. "
    "Each candidate is relaxed with L-BFGS. A duplicate (ops.same_structure) replaces its twin "
    "only if lower; any other candidate replaces the worst member only if better. "
    "Deviations from the papers: "
    "steady-state replacement as in ASE-GA instead of BCGA generations; mutants are separate "
    "offspring; at most 2*pop_size random relaxations are spent filling the initial population; "
    "detached atoms are re-seated before relaxation (as in bh); "
    "the GA restarts from a new random population after stall_gens*pop_size candidates without a "
    "new best (BCGA convergence criterion, giving independent GA runs within the budget).")
DEFAULTS = dict(pop_size=20, p_mut=0.2, fitness="tanh", alpha=3.0, history=True,
                rattle_strength=0.8, rattle_prop=0.4, perm_prob=0.33, relocate_frac=0.2,
                strain_sigma=0.7, stall_gens=10)

BAD_E = 1e5          # relaxations ending above this (NaN guard energy) are discarded
INIT_FACTOR = 2      # at most INIT_FACTOR*pop_size random relaxations for the initial population
MIN_HEIGHT = 0.45    # bulk: reject cells with an interplanar spacing < MIN_HEIGHT*V^(1/3)
REGISTRY_TOL = 0.3   # A; tolerance for surface-lattice translations
REGISTRY_MAX = 2.0   # A; larger registry mismatch -> continuous lateral translation


# ---------------------------------------------------------------- helpers
def _heights(cell):
    return 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)


def _reduce(pos, cell):
    """Niggli-reduced cell in standard orientation; atoms wrapped (distances preserved)."""
    try:
        rc, C = niggli_reduce_cell(np.asarray(cell, float))
        basis = C.T @ cell
        std = np.array(rc, float)
        if not np.all(np.isfinite(std)) or abs(np.linalg.det(std)) < 1e-6:
            return pos, cell
        frac = np.linalg.solve(basis.T, np.asarray(pos, float).T).T % 1.0
        return frac @ std, std
    except Exception:
        return np.asarray(pos, float), np.asarray(cell, float)


def _horizontal_unit(rng):
    for _ in range(100):
        v = rng.normal(size=3)
        v[2] = 0.0
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            return v / nv
    return np.array([1.0, 0.0, 0.0])


class _Member:
    __slots__ = ("pos", "cell", "e", "fp", "n_paired", "looks_like")

    def __init__(self, pos, cell, e, fp):
        self.pos, self.cell, self.e, self.fp = pos, cell, e, fp
        self.n_paired = 0
        self.looks_like = 0


class _GA:
    def __init__(self, ctx, rng, pop_size, p_mut, fitness, alpha, history, rattle_strength,
                 rattle_prop, perm_prob, relocate_frac, strain_sigma, stall_gens):
        self.ctx, self.rng, self.p = ctx, rng, ctx.p
        p = self.p
        self.pop_size = max(2, int(pop_size))
        self.p_mut = float(p_mut)
        self.fitness, self.alpha, self.history = fitness, float(alpha), bool(history)
        self.rattle_strength, self.rattle_prop = float(rattle_strength), float(rattle_prop)
        self.perm_prob, self.relocate_frac = float(perm_prob), float(relocate_frac)
        self.strain_sigma = float(strain_sigma)
        self.stall = int(stall_gens) * self.pop_size if stall_gens else 0
        self.s = p.search_idx
        self.ns = len(self.s)
        self.sym = np.array(p.symbols)
        self.ssym = self.sym[self.s]
        self.species = sorted(set(self.ssym.tolist()))
        self.counts = {x: int((self.ssym == x).sum()) for x in self.species}
        self.slots = {x: self.s[self.ssym == x] for x in self.species}
        self.multi = len(self.species) > 1
        self.periodic = bool(p.variable_cell or p.category == "bulk")
        self.surface = p.category == "surface" and not self.periodic
        self.free_cluster = p.n_sub == 0 and not self.periodic
        self.inv0 = np.linalg.inv(p.cell0)
        if self.periodic:
            self.mut_ops = (["strain"] if p.variable_cell else []) + ["rattle"]
        else:
            self.mut_ops = ["rattle", "mirror", "twist", "relocate"]
        if self.multi:
            self.mut_ops.append("permute")
        self.surf_T = self._surface_translations() if self.surface else None
        self.pop = []
        self.best_e = np.inf
        self.since_best = 0
        st = ctx.stats.setdefault("ga", {})
        # per operator: [tried, entered population (new or better twin), duplicate, new best]
        self.opstats = st.setdefault("ops", {})
        st.setdefault("restarts", 0)
        st["n_reg_T"] = 0 if self.surf_T is None else int(len(self.surf_T))
        self.st = st

    # ------------------------------------------------------------ geometry
    def _mic(self, d):
        f = np.asarray(d, float) @ self.inv0
        f = f - np.round(f) * self.p.pbc
        return f @ self.p.cell0

    def _surface_translations(self):
        """Lateral translations mapping the top substrate layer onto itself."""
        p = self.p
        if p.n_sub == 0:
            return None
        sub = np.asarray(p.sub_pos, float)
        z = sub[:, 2]
        reg = p.region or {}
        if reg.get("type") == "box":
            zc = 0.5 * (reg["lo"][2] + reg["hi"][2])
        elif reg.get("type") == "sphere":
            zc = reg["center"][2]
        else:
            zc = z.max() + 1.0
        top = z >= z.max() - 1.0 if zc >= z.mean() else z <= z.min() + 1.0
        idx = np.where(top)[0]
        ssym = np.array(p.sub_symbols)[idx]
        sp, cnt = np.unique(ssym, return_counts=True)
        ref_sp = sp[np.argmax(cnt)]
        cand = idx[ssym == ref_sp]
        ref = cand[0]
        P = sub[idx]
        test = np.arange(len(idx))
        if len(test) > 64:
            test = self.rng.choice(test, 64, replace=False)
        same = ssym[test][:, None] == ssym[None, :]
        out = []
        for j in cand[:400]:
            t = sub[j] - sub[ref]
            t[2] = 0.0
            d = self._pair_dist(P[test] + t, P)
            if np.all(np.where(same, d, np.inf).min(axis=1) < REGISTRY_TOL):
                out.append(t)
        return np.array(out) if out else None

    def _pair_dist(self, a, b):
        d = a[:, None, :] - b[None, :, :]
        f = d @ self.inv0
        f -= np.round(f) * self.p.pbc
        return np.linalg.norm(f @ self.p.cell0, axis=-1)

    def _surface_shift(self, d):
        """Lateral shift close to d that preserves the substrate registry if possible."""
        d = np.array([d[0], d[1], 0.0])
        if self.surf_T is None:
            return d
        r = d[None, :] - self.surf_T
        f = r @ self.inv0
        f -= np.round(f) * self.p.pbc
        rm = f @ self.p.cell0
        k = int(np.argmin(np.linalg.norm(rm[:, :2], axis=1)))
        if np.linalg.norm(rm[k, :2]) > REGISTRY_MAX:
            return d
        out = d - rm[k]
        out[2] = 0.0
        return out

    # ------------------------------------------------------------ population
    def _relax(self, pos, cell):
        res = self.ctx.relax(pos, cell)
        if res.aborted or not np.isfinite(res.e) or res.e >= BAD_E:
            return None
        pos, cell = np.array(res.pos, float), np.array(res.cell, float)
        if self.p.variable_cell:
            pos, cell = _reduce(pos, cell)
        return _Member(pos, cell, float(res.e), ops.fingerprint(self.p, pos, cell))

    def _add(self, m):
        """Steady-state insertion; returns 'new', 'twin+' (better duplicate), 'dup', 'worse', 'fail'."""
        if m is None:
            return "fail"
        for q in self.pop:
            if ops.same_structure(self.p, m.e, m.fp, q.e, q.fp):
                q.looks_like += 1
                if m.e < q.e:
                    q.pos, q.cell, q.e, q.fp = m.pos, m.cell, m.e, m.fp
                    return "twin+"
                return "dup"
        if len(self.pop) < self.pop_size:
            self.pop.append(m)
            return "new"
        w = int(np.argmax([q.e for q in self.pop]))
        if m.e < self.pop[w].e:
            self.pop[w] = m
            return "new"
        return "worse"

    def _fit(self):
        e = np.array([q.e for q in self.pop])
        span = e.max() - e.min()
        rho = (e - e.min()) / span if span > 1e-12 else np.zeros(len(e))
        if self.fitness == "exp":
            f = np.exp(-self.alpha * rho)
        else:
            f = 0.5 * (1.0 - np.tanh(2.0 * rho - 1.0))
        if self.history:
            npair = np.array([q.n_paired for q in self.pop], float)
            nlike = np.array([q.looks_like for q in self.pop], float)
            f = f / np.sqrt(1.0 + npair) / np.sqrt(1.0 + nlike)
        return f

    def _roulette(self, f, exclude=-1):
        """Johnston/ASE roulette: random member accepted with probability f/f_max."""
        n = len(f)
        fmax = max(f.max(), 1e-300)
        for _ in range(100 * n + 100):
            t = int(self.rng.integers(n))
            if t != exclude and f[t] > self.rng.random() * fmax:
                return t
        others = [k for k in range(n) if k != exclude]
        return int(self.rng.choice(others))

    # ------------------------------------------------------------ crossover
    def _xover_cluster(self, A, B):
        p, rng, s = self.p, self.rng, self.s
        cell = p.cell0
        a = ops.unwrap_search(p, A.pos, cell)
        b = ops.unwrap_search(p, B.pos, cell)
        ca, cb = a[s].mean(axis=0), b[s].mean(axis=0)
        if self.free_cluster:
            nrm = ops.random_unit(rng)
            shift = ca - cb  # Deaven-Ho: both planes through the parent centres
        else:
            d = ca - cb
            dm = self._mic(d)
            shift = d - dm  # lattice vector only: B unchanged physically
            if self.surface:
                nrm = _horizontal_unit(rng)
                shift = shift + self._surface_shift(dm)
            else:
                nrm = ops.random_unit(rng)
        bs = b[s] + shift
        point = ca if self.free_cluster else 0.5 * (ca + bs.mean(axis=0))
        da = (a[s] - point) @ nrm
        db = (bs - point) @ nrm
        need = dict(self.counts)
        chosen = []
        n_from_a = int(np.clip((da >= 0).sum(), 1, self.ns - 1))
        for k in np.argsort(-da)[:n_from_a]:
            x = self.ssym[k]
            if need[x] > 0:
                chosen.append((x, a[s[k]]))
                need[x] -= 1
        for k in np.argsort(db):
            x = self.ssym[k]
            if need[x] > 0:
                chosen.append((x, bs[k]))
                need[x] -= 1
        child = np.array(a)
        fill = {x: 0 for x in self.species}
        for x, xyz in chosen:
            child[self.slots[x][fill[x]]] = xyz
            fill[x] += 1
        return ops.repair_overlaps(p, child, cell), np.array(cell)

    def _xover_bulk(self, A, B):
        """Abraham-Probert periodic slab cut in fractional coordinates + lattice heredity."""
        p, rng, s = self.p, self.rng, self.s
        fa = (A.pos[s] @ np.linalg.inv(A.cell))
        fb = (B.pos[s] @ np.linalg.inv(B.cell))
        va, vb = abs(np.linalg.det(A.cell)), abs(np.linalg.det(B.cell))
        pos = cell = None
        for _ in range(20):
            k = int(rng.integers(3))
            offs = []
            for _par in range(2):
                o = np.where(rng.random(3) < 0.05, rng.random(3), 0.0)  # ASE p2 in-plane shift
                o[k] = rng.random()  # ASE p1=1: always shifted along the cut normal
                offs.append(o)
            ga = (fa + offs[0]) % 1.0
            gb = (fb + offs[1]) % 1.0
            in_a = (ga[:, k] >= 0.25) & (ga[:, k] < 0.75)
            in_b = ~((gb[:, k] >= 0.25) & (gb[:, k] < 0.75))
            frac = np.zeros((self.ns, 3))
            n_from = [0, 0]
            for x in self.species:
                lx = np.where(self.ssym == x)[0]
                used = [(0, i) for i in lx if in_a[i]] + [(1, i) for i in lx if in_b[i]]
                unused = [(0, i) for i in lx if not in_a[i]] + [(1, i) for i in lx if not in_b[i]]
                while len(used) > len(lx):
                    used.pop(int(rng.integers(len(used))))
                while len(used) < len(lx):
                    used.append(unused.pop(int(rng.integers(len(unused)))))
                for slot, (src, i) in zip(lx, used):
                    frac[slot] = ga[i] if src == 0 else gb[i]
                    n_from[src] += 1
            if self.ns > 1 and min(n_from) == 0:
                continue
            if p.variable_cell:
                w = rng.random()
                c = w * A.cell + (1.0 - w) * B.cell
                vol = abs(np.linalg.det(c))
                if vol < 1e-6:
                    continue
                c = c * ((w * va + (1.0 - w) * vb) / vol) ** (1.0 / 3.0)
                if _heights(c).min() < MIN_HEIGHT * abs(np.linalg.det(c)) ** (1 / 3):
                    continue
            else:
                c = np.array(p.cell0)
            pos, cell = np.array(A.pos), c
            pos[s] = frac @ cell
            if p.min_dist_ok(pos, cell):
                return pos, cell
        if pos is None:  # no valid cut/cell found: fall back to a strained/rattled parent
            return self._rattle(A)
        return ops.repair_overlaps(p, pos, cell), cell

    # ------------------------------------------------------------ mutations
    def _rattle(self, M):
        """ASE RattleMutation: each atom with prob rattle_prop, uniform +-rattle_strength."""
        rng, s = self.rng, self.s
        pos = np.array(M.pos)
        sel = rng.random(self.ns) < self.rattle_prop
        if not sel.any():
            sel[int(rng.integers(self.ns))] = True
        pos[s[sel]] += self.rattle_strength * (2.0 * rng.random((int(sel.sum()), 3)) - 1.0)
        return ops.repair_overlaps(self.p, pos, M.cell), np.array(M.cell)

    def _permute(self, M):
        """ASE PermutationMutation: ceil(N*perm_prob/2) swaps of unlike atoms."""
        nswap = max(1, int(math.ceil(self.ns * self.perm_prob / 2.0)))
        pos = ops.swap_unlike(self.p, M.pos, self.rng, nswap)
        return ops.repair_overlaps(self.p, pos, M.cell), np.array(M.cell)

    def _mirror(self, M):
        """ASE MirrorMutation: keep half of each species, mirror it through a random plane."""
        p, rng, s = self.p, self.rng, self.s
        cell = p.cell0
        pos = ops.unwrap_search(p, M.pos, cell)
        c = pos[s].mean(axis=0)
        nrm = _horizontal_unit(rng) if self.surface else ops.random_unit(rng)
        d = (pos[s] - c) @ nrm
        new = np.array(pos)
        for x in self.species:
            lx = np.where(self.ssym == x)[0]
            order = lx[np.argsort(d[lx])]
            m = len(order)
            h = m // 2
            src = order[:h]          # farthest on the kept side
            dst = order[m - h:]      # replaced by the mirror images
            xs = pos[s[src]]
            new[s[dst]] = xs - 2.0 * ((xs - c) @ nrm)[:, None] * nrm[None, :]
        return ops.repair_overlaps(p, new, cell), np.array(cell)

    def _twist(self, M):
        """BCGA rotation mutation: rotate one half of the cluster by a random angle."""
        p, rng, s = self.p, self.rng, self.s
        cell = p.cell0
        pos = ops.unwrap_search(p, M.pos, cell)
        c = pos[s].mean(axis=0)
        sel = None
        axis = np.array([0.0, 0.0, 1.0])
        if self.surface:
            up = pos[s, 2] > c[2] + 0.5
            if 0 < up.sum() < self.ns:
                sel = up
        for _ in range(50):
            if sel is not None:
                break
            if self.surface:
                n = _horizontal_unit(rng)
            else:
                n = axis = ops.random_unit(rng)
            side = (pos[s] - c) @ n > 0
            if 0 < side.sum() < self.ns:
                sel = side
        if sel is None:
            sel = np.zeros(self.ns, bool)
            sel[int(rng.integers(self.ns))] = True
        rot = ops.rotation_about(axis, rng.uniform(0.0, 2.0 * np.pi))
        idx = s[sel]
        new = np.array(pos)
        new[idx] = (pos[idx] - c) @ rot.T + c
        return ops.repair_overlaps(p, new, cell), np.array(cell)

    def _relocate(self, M):
        """BCGA atom replacement: move 1..ceil(relocate_frac*N) random atoms to surface sites."""
        p, rng = self.p, self.rng
        kmax = max(1, min(self.ns - 1, int(math.ceil(self.relocate_frac * self.ns))))
        k = int(rng.integers(1, kmax + 1))
        idx = rng.choice(self.s, size=k, replace=False)
        pos = ops.relocate_atoms(p, M.pos, rng, idx, p.cell0)
        if pos is None:
            return self._rattle(M)
        return pos, np.array(p.cell0)

    def _strain(self, M):
        """ASE/USPEX strain mutation of the lattice vectors, volume kept, atoms follow."""
        p, rng, s = self.p, self.rng, self.s
        cell0 = np.array(M.cell)
        vol = abs(np.linalg.det(cell0))
        frac = M.pos @ np.linalg.inv(cell0)
        for _ in range(100):
            S = np.eye(3)
            for i in range(3):
                for j in range(i + 1):
                    r = rng.normal(0.0, self.strain_sigma)
                    if i == j:
                        S[i, i] += r
                    else:
                        S[i, j] += 0.5 * r
                        S[j, i] += 0.5 * r
            cell = S @ cell0
            v = abs(np.linalg.det(cell))
            if not np.isfinite(v) or v < 1e-6 * vol:
                continue
            cell = cell * (vol / v) ** (1.0 / 3.0)
            pos, cell = _reduce(frac @ cell, cell)
            if _heights(cell).min() < MIN_HEIGHT * vol ** (1.0 / 3.0):
                continue
            if p.min_dist_ok(pos, cell):
                return pos, cell
        return self._rattle(M)

    # ------------------------------------------------------------ main loop
    def _reseat(self, pos, cell):
        if self.periodic:
            return pos
        lost = ops.detached_atoms(self.p, pos, cell)
        if len(lost):
            new = ops.relocate_atoms(self.p, pos, self.rng, lost, cell)
            if new is not None:
                return new
        return pos

    def _record(self, op, status, e):
        r = self.opstats.setdefault(op, [0, 0, 0, 0])
        r[0] += 1
        if status in ("new", "twin+"):
            r[1] += 1
        elif status == "dup":
            r[2] += 1
        if e is not None and e < self.best_e - 1e-3:
            if np.isfinite(self.best_e):
                r[3] += 1
            self.best_e = e
            self.since_best = 0
        else:
            self.since_best += 1

    def _step(self, op, pos, cell):
        m = self._relax(self._reseat(pos, cell), cell)
        status = self._add(m)
        self._record(op, status, None if m is None else m.e)

    def run(self):
        p, rng = self.p, self.rng
        tries = 0
        while len(self.pop) < self.pop_size and tries < INIT_FACTOR * self.pop_size:
            tries += 1
            pos, cell = p.random_structure(rng)
            m = self._relax(pos, cell)
            self._record("init", self._add(m), None if m is None else m.e)
        self.since_best = 0
        while True:
            if self.stall and self.since_best >= self.stall:
                self.st["restarts"] += 1
                return  # converged: the runner restarts the GA with a new population
            if not self.pop:
                pos, cell = p.random_structure(rng)
                self._step("random", pos, cell)
                continue
            if len(self.pop) >= 2 and self.ns >= 2 and rng.random() >= self.p_mut:
                f = self._fit()
                i = self._roulette(f)
                j = self._roulette(f, exclude=i)
                A, B = self.pop[i], self.pop[j]
                A.n_paired += 1
                B.n_paired += 1
                if self.periodic:
                    pos, cell = self._xover_bulk(A, B)
                else:
                    pos, cell = self._xover_cluster(A, B)
                self._step("xover", pos, cell)
            else:
                M = self.pop[self._roulette(self._fit())]
                op = self.mut_ops[int(rng.integers(len(self.mut_ops)))]
                pos, cell = getattr(self, "_" + op)(M)
                self._step(op, pos, cell)
            self.st["pop"] = len(self.pop)


def run(ctx, rng, pop_size=20, p_mut=0.2, fitness="tanh", alpha=3.0, history=True,
        rattle_strength=0.8, rattle_prop=0.4, perm_prob=0.33, relocate_frac=0.2,
        strain_sigma=0.7, stall_gens=10):
    _GA(ctx, rng, pop_size, p_mut, fitness, alpha, history, rattle_strength, rattle_prop,
        perm_prob, relocate_frac, strain_sigma, stall_gens).run()
