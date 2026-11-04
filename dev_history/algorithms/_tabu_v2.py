"""Tabu multi-walker basin hopping (TMW-BH).

A small population of basin-hopping walkers shares one tabu memory of every
local minimum found so far. The memory is a hash of relaxed energies (bins of
``de``) refined by a species-resolved fingerprint, so identifying a minimum
costs microseconds and no ionic steps. The memory is used in five ways:

1. frequency penalty: Metropolis acceptance on E + lam*log(1+visits) (energy
   landscape paving on minima, not on energies), so walkers drift out of
   over-sampled basins; a short recency tabu list forbids immediate cycling;
2. escalation: a trial that relaxes back into a known minimum enlarges that
   walker's next perturbation (Goedecker-style feedback, per walker);
3. stagnation: revisits (and, at half weight, non-improving new minima) fill
   a stagnation counter; an exhausted walker is re-seeded either at random or
   at an unoccupied archive minimum drawn with weight exp(-E/kT)/(1+visits);
4. merging: a walker that moves into the minimum occupied by another walker
   is redundant (same funnel) and is re-seeded;
5. scheduling: the next move goes to the walker with the best discounted-UCB
   score, rewarding new minima that improve the walker or enter the global
   top-n list (bandit allocation of the step budget).

Moves are BH moves (uniform displacement of all search atoms, random cell
strain for bulk, re-seating of detached atoms) plus swaps of unlike atoms of
the same chemical class (metal/metal or nonmetal/nonmetal) for multi-element
systems.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

from .. import ops
from ..context import RelaxResult, _fmax, _relax_cell, lbfgs

LABEL = "Tabu multi-walker BH"
FAMILY = "innovative"
REFERENCE = ("Glover, ORSA J. Comput. 1, 190 (1989) and 2, 4 (1990) [tabu search]; "
             "Cvijovic & Klinowski, Science 267, 664 (1995) [taboo search for continuous "
             "global optimisation]; Wales & Doye, J. Phys. Chem. A 101, 5111 (1997) "
             "[basin hopping]; Hansmann & Wille, Phys. Rev. Lett. 88, 068105 (2002) "
             "[energy landscape paving]; Goedecker, J. Chem. Phys. 120, 9911 (2004) "
             "[history feedback of minima hopping]; Garivier & Moulines, ALT 2011, "
             "LNCS 6925, 174 [discounted UCB bandits]")
DESCRIPTION = (
    "Several basin-hopping walkers share a tabu memory of all relaxed minima "
    "(energy-binned hash + species-resolved distance fingerprint; no ionic steps). "
    "Acceptance is Metropolis on E + lam*log(1+visits) with a short recency tabu "
    "list, so walkers leave over-sampled basins; a trial that falls back into a "
    "known minimum enlarges that walker's next step; walkers whose trials keep "
    "revisiting known minima (stagnation) or that move into another walker's "
    "minimum (same funnel) are re-seeded either randomly or at an unoccupied archive "
    "minimum drawn with weight exp(-E/kT)/(1+visits); the next move is given to the "
    "walker with the best discounted-UCB score for recent new low minima. Moves are "
    "BH displacements (+cell strain for bulk) and same-class swaps for alloys. "
    "Unlike BH/ptbh/minima hopping, the search memory is shared, drives both "
    "acceptance and budget allocation, and turns BH smoothly into multi-start search "
    "on landscapes where single walkers keep re-finding the same minima.")
DEFAULTS = dict(n_walkers=4, kT=0.1, step=0.55, target_accept=0.5, adapt_every=10,
                strain=0.05, lam=0.05, tenure=3, patience=6.0, p_random=0.5, merge=True,
                esc=1.3, esc_max=2.0, p_swap=0.2, ucb_c=0.5, discount=0.9, n_top=5,
                restart_arm=False, p_reloc=0.4, sel_T=0.1, abort_known=True, ab_fmax=0.15,
                ab_de=0.05, ab_dfp=0.5)

_NONMETALS = frozenset("H He B C N O F Ne Si P S Cl Ar As Se Br Kr Te I Xe".split())
_BAD_E = 1e5


def _images(pbc):
    r = [(-1, 0, 1) if q else (0,) for q in pbc]
    return np.array([(i, j, k) for i in r[0] for j in r[1] for k in r[2]], float)


class _Fingerprint:
    """Species-resolved sorted distances (fixed cell) or ops RDF fingerprint (bulk)."""

    def __init__(self, p):
        self.p = p
        self.bulk = bool(p.variable_cell)
        self.s = p.search_idx
        if self.bulk:
            return
        sym = np.array(p.symbols)[self.s]
        species = sorted(set(sym))
        iu, ju = np.triu_indices(len(self.s), 1)
        self.iu, self.ju = iu, ju
        self.groups = []
        for a, x in enumerate(species):
            for y in species[a:]:
                m = ((sym[iu] == x) & (sym[ju] == y)) | ((sym[iu] == y) & (sym[ju] == x))
                if m.any():
                    self.groups.append(np.where(m)[0])
        self.sp = [np.where(sym == x)[0] for x in species]
        self.pbc = np.asarray(p.pbc, bool)
        self.cell = np.asarray(p.cell0, float)
        self.inv = np.linalg.inv(self.cell)
        self.tree = None
        if p.n_sub:
            shifts = _images(self.pbc) @ self.cell
            sub = (np.asarray(p.sub_pos)[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
            self.tree = cKDTree(sub)

    def __call__(self, pos, cell):
        if self.bulk:
            return ops.fingerprint(self.p, np.asarray(pos), cell)
        x = np.asarray(pos)[self.s]
        d = x[self.iu] - x[self.ju]
        if self.pbc.any():
            f = d @ self.inv
            f -= np.where(self.pbc, np.round(f), 0.0)
            d = f @ self.cell
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        parts = [np.sort(r[g]) for g in self.groups]
        if self.tree is not None:
            f = x @ self.inv
            xw = x - np.where(self.pbc, np.floor(f), 0.0) @ self.cell
            dsub = self.tree.query(xw)[0]
            parts += [np.sort(dsub[g]) for g in self.sp]
        return np.concatenate(parts) if parts else np.zeros(1)

    def distance(self, a, b):
        if self.bulk:
            return float(1.0 - np.dot(a, b))
        return float(np.mean(np.abs(a - b)))


class _Memory:
    """Tabu memory: every distinct relaxed minimum with its visit count."""

    def __init__(self, p, de=0.01, dfp=None):
        self.p = p
        self.fpr = _Fingerprint(p)
        self.de = float(de)
        self.dfp = (0.01 if p.variable_cell else 0.03) if dfp is None else float(dfp)
        self.free = np.asarray(p.free_idx)
        self.base = np.zeros((p.n, 3))
        if p.n_sub:
            self.base[: p.n_sub] = p.sub_pos
        self.bins = {}
        self.e, self.fp, self.x, self.cell, self.ea = [], [], [], [], []
        self.visits, self.seeded = [], []
        self.top = []  # lowest energies, ascending, len <= n_top

    def __len__(self):
        return len(self.e)

    def fingerprint(self, pos, cell):
        try:
            return self.fpr(pos, cell)
        except Exception:  # never let a descriptor problem kill the run
            return np.zeros(1)

    def match(self, e, fp, de_lo, de_hi, dfp):
        """Known minimum j with -de_lo <= e - E_j <= de_hi and fingerprint distance < dfp."""
        hit, best = -1, dfp
        for bb in range(int(math.floor((e - de_hi) / self.de)), int(math.floor((e + de_lo) / self.de)) + 1):
            for i in self.bins.get(bb, ()):
                if -de_lo <= e - self.e[i] <= de_hi and len(self.fp[i]) == len(fp):
                    d = self.fpr.distance(fp, self.fp[i])
                    if d < best:
                        hit, best = i, d
        return hit

    def has_near(self, e, de_lo, de_hi):
        for bb in range(int(math.floor((e - de_hi) / self.de)), int(math.floor((e + de_lo) / self.de)) + 1):
            for i in self.bins.get(bb, ()):
                if -de_lo <= e - self.e[i] <= de_hi:
                    return True
        return False

    def find_or_add(self, e, pos, cell, n_top, ea=None):
        """Return (index, is_new, is_top) and count the visit."""
        fp = self.fingerprint(pos, cell)
        b = int(math.floor(e / self.de))
        hit, best = -1, np.inf
        for bb in (b - 1, b, b + 1):
            for i in self.bins.get(bb, ()):
                if abs(self.e[i] - e) <= self.de and len(self.fp[i]) == len(fp):
                    d = self.fpr.distance(fp, self.fp[i])
                    if d < self.dfp and d < best:
                        hit, best = i, d
        if hit >= 0:
            self.visits[hit] += 1
            return hit, False, False
        i = len(self.e)
        self.bins.setdefault(b, []).append(i)
        self.e.append(float(e))
        self.fp.append(fp)
        self.x.append(np.array(np.asarray(pos)[self.free], float))
        self.cell.append(np.array(cell, float))
        self.ea.append(None if ea is None else np.array(ea, float))
        self.visits.append(1)
        self.seeded.append(0)
        is_top = len(self.top) < n_top or e < self.top[-1]
        if is_top:
            self.top.append(float(e))
            self.top.sort()
            del self.top[n_top:]
        return i, True, is_top

    def structure(self, i):
        pos = self.base.copy()
        pos[self.free] = self.x[i]
        return pos, self.cell[i].copy()


class _Walker:
    __slots__ = ("cur", "e", "best", "stag", "esc", "S", "N", "recent")

    def __init__(self, i, e, esc=1.0, prior=0.5):
        self.cur, self.e, self.best = i, e, e
        self.stag, self.esc = 0.0, esc
        self.S, self.N = prior, 1.0
        self.recent = []


def _swap_pairs(p):
    sym = np.array(p.symbols)[p.search_idx]
    species = sorted(set(sym))
    out = []
    for a, x in enumerate(species):
        for y in species[a + 1:]:
            if (x in _NONMETALS) == (y in _NONMETALS):
                out.append((p.search_idx[sym == x], p.search_idx[sym == y]))
    return out




class _Track:
    """ctx stand-in for the variable-cell optimiser that remembers the last evaluated geometry."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.x = None
        self.c = None

    def evaluate(self, pos, cell=None):
        self.x, self.c = pos, cell
        return self.ctx.evaluate(pos, cell)

    def __getattr__(self, name):  # everything else (p, last_virial, ...) comes from ctx
        return getattr(self.ctx, name)


def run(ctx, rng, n_walkers=4, kT=0.1, step=0.55, target_accept=0.5, adapt_every=10,
        strain=0.05, lam=0.05, tenure=3, patience=6.0, p_random=0.5, merge=True, esc=1.3,
        esc_max=2.0, p_swap=0.2, ucb_c=0.5, discount=0.9, n_top=5, restart_arm=False,
        p_reloc=0.4, sel_T=0.1, abort_known=True, ab_fmax=0.15, ab_de=0.05, ab_dfp=0.5):
    """Steps are in units of the mean covalent bond length of the search atoms."""
    p = ctx.p
    s_idx = p.search_idx
    bond = float(2.0 * np.mean(p.rcov[s_idx]))
    mem = _Memory(p)
    pairs = _swap_pairs(p)
    reseat = (not p.variable_cell) and p.category != "bulk"
    can_reloc = reseat and p.n_search >= 2 and p_reloc > 0
    ssym = np.array(p.symbols)[s_idx]
    groups = [np.where(ssym == x)[0] for x in sorted(set(ssym))]
    ab_dfp = float(ab_dfp) * mem.dfp
    n_walkers = max(1, int(n_walkers))
    n_top = max(1, int(n_top))
    walkers = []
    g = dict(step=float(step) * bond, n_try=0, n_acc=0, n_new=0, n_rev=0, n_back=0, n_fail=0,
             n_reseed=0, n_merge=0, n_swap=0, n_reloc=0, n_restart=0, n_adopt=0,
             n_tabu_abort=0)
    rarm = _Walker(-1, np.inf) if restart_arm else None  # bandit arm "random restart"
    track = _Track(ctx)

    def fun_fixed(x):
        track.x = x
        return ctx.evaluate(x)

    def relax(pos, cell):
        """ctx.relax equivalent (same L-BFGS, same step counting) with the tabu early stop:
        a relaxation that reaches the bottom of an already known basin (small forces, energy
        just above a stored minimum, matching fingerprint) is stopped; returns (res, j)."""
        hit = [-1]
        ab = None
        if abort_known and len(mem):
            def ab(it, e, fm):
                if it < 1 or fm > ab_fmax or not mem.has_near(e, 0.005, ab_de):
                    return False
                fp = mem.fingerprint(track.x, track.c if p.variable_cell else p.cell0)
                j = mem.match(e, fp, 0.005, ab_de, ab_dfp)
                if j >= 0:
                    hit[0] = j
                    return True
                return False
        if p.variable_cell:
            res = _relax_cell(track, pos, p.cell0 if cell is None else cell, p.fmax,
                              p.max_relax_steps, "lbfgs", ab)
        else:
            x, e, f, conv, n, aborted = lbfgs(fun_fixed, np.array(pos, float), p.fmax,
                                              p.max_relax_steps, abort=ab)
            res = RelaxResult(x, p.cell0.copy(), e, conv, n, _fmax(f), aborted)
        ctx.n_relax += 1
        if not res.aborted:
            ctx.minima.append(round(float(res.e), 5))
        return res, hit[0]

    def relax_register(pos, cell):
        res, j = relax(pos, cell)
        if j >= 0:  # stopped inside a known basin: a revisit whose convergence tail is saved
            mem.visits[j] += 1
            g["n_rev"] += 1
            g["n_tabu_abort"] += 1
            return res, j, False, False
        if res.aborted or not np.isfinite(res.e) or res.e >= _BAD_E:
            return res, -1, False, False
        ea = ctx.atomic_energies()
        ea = None if ea is None or len(ea) != p.n else ea[s_idx]
        i, new, top = mem.find_or_add(res.e, res.pos, res.cell, n_top, ea)
        if new:
            g["n_new"] += 1
        else:
            g["n_rev"] += 1
        return res, i, new, top

    def pick_atoms(i_min, k):
        """k search atoms, drawn with weight exp(z/sel_T), z = per-atom NEP energy above the
        median of its species in minimum i_min (uniform if no atomic energies are stored)."""
        ea = mem.ea[i_min]
        if ea is None or len(ea) != p.n_search or not np.all(np.isfinite(ea)):
            return rng.choice(p.n_search, size=k, replace=False)
        z = np.empty(p.n_search)
        for gi in groups:
            z[gi] = ea[gi] - np.median(ea[gi])
        w = np.maximum(np.exp((z - z.max()) / max(sel_T, 1e-6)), 1e-12)
        return rng.choice(p.n_search, size=k, replace=False, p=w / w.sum())

    def perturb(wk, pos, cell, s):
        pos, cell = np.array(pos), np.array(cell)
        if pairs and rng.random() < p_swap:
            a, b = pairs[rng.integers(len(pairs))]
            i, j = rng.choice(a), rng.choice(b)
            pos[[i, j]] = pos[[j, i]]
            g["n_swap"] += 1
            return pos, cell
        if can_reloc and rng.random() < p_reloc:
            # surface relocation of weakly bound atoms; more atoms after revisits
            k = min(1 if wk.esc < 1.5 else 2, p.n_search - 1)
            idx = s_idx[pick_atoms(wk.cur, k)]
            g["n_reloc"] += 1
            return ops.relocate_atoms(p, pos, rng, idx, cell), cell
        if p.variable_cell:
            pos, cell = ops.strain_cell(pos, cell, rng, strain)
        pos = ops.displace(p, pos, rng, s, cell=cell)
        if reseat:
            lost = ops.detached_atoms(p, pos, cell)
            if len(lost):
                pos = ops.relocate_atoms(p, pos, rng, lost, cell)
        return pos, cell

    def occupied(i, skip=None):
        return any(w is not skip and w.cur == i for w in walkers)

    def random_minimum():
        for _ in range(3):
            try:
                pos, cell = p.random_structure(rng)
            except RuntimeError:
                continue
            res, i, new, top = relax_register(pos, cell)
            if i >= 0:
                return i
        return -1

    def archive_minimum(skip):
        cand = [i for i in range(len(mem)) if not occupied(i, skip)]
        if skip is not None and skip.cur in cand and len(cand) > 1:
            cand.remove(skip.cur)
        if not cand:
            return -1
        e = np.array([mem.e[i] for i in cand])
        cnt = np.array([mem.visits[i] + 3 * mem.seeded[i] for i in cand], float)
        logw = -(e - e.min()) / kT - np.log1p(cnt)
        w = np.exp(logw - logw.max())
        i = cand[int(rng.choice(len(cand), p=w / w.sum()))]
        mem.seeded[i] += 1
        return i

    def reseed(w):
        g["n_reseed"] += 1
        i, e_esc = -1, 1.0
        if len(mem) > n_walkers and rng.random() >= p_random:
            i, e_esc = archive_minimum(w), esc_max
        if i < 0:
            i, e_esc = random_minimum(), 1.0
        if i < 0:  # could not generate anything: keep the walker where it is
            w.stag = 0.0
            return
        w.cur, w.e, w.best = i, mem.e[i], mem.e[i]
        w.stag, w.esc, w.S, w.N, w.recent = 0.0, e_esc, 0.5, 1.0, []

    while True:
        if len(walkers) < n_walkers:
            i = random_minimum()
            if i < 0:
                if not len(mem):
                    raise RuntimeError("tabu: could not create a valid starting minimum")
                i = int(np.argmin(mem.e))
            walkers.append(_Walker(i, mem.e[i]))
            continue

        # ---- bandit: discounted UCB over walkers (+ the random-restart arm)
        arms = walkers + [rarm] if rarm is not None else walkers
        tot = 0.0
        for w in arms:
            w.S *= discount
            w.N *= discount
            tot += w.N
        lg = math.log(tot + 1.0)
        k = max(range(len(arms)), key=lambda q: arms[q].S / arms[q].N
                + ucb_c * math.sqrt(lg / arms[q].N) + 1e-9 * rng.random())
        w = arms[k]
        if w is rarm:
            g["n_restart"] += 1
            n0 = len(mem)
            i = random_minimum()
            reward = 1.0 if (i >= 0 and len(mem) > n0 and mem.e[i] <= mem.top[-1]) else 0.0
            if reward:  # promising new funnel: it replaces the most stagnant walker
                v = max(walkers, key=lambda q: (q.stag, q.e))
                if not occupied(i):
                    g["n_adopt"] += 1
                    v.cur, v.e, v.best = i, mem.e[i], mem.e[i]
                    v.stag, v.esc, v.S, v.N, v.recent = 0.0, 1.0, 0.5, 1.0, []
            rarm.S += reward
            rarm.N += 1.0
            ctx.stats.update(g, archive=len(mem), accept_rate=g["n_acc"] / max(g["n_try"], 1))
            continue

        # ---- trial move from the walker's current minimum
        pos, cell = mem.structure(w.cur)
        s_eff = float(min(g["step"] * w.esc, bond))
        tpos, tcell = perturb(w, pos, cell, s_eff)
        res, i, new, top = relax_register(tpos, tcell)
        g["n_try"] += 1
        reward = 0.0
        if i < 0:
            g["n_fail"] += 1
            w.stag += 1.0
            w.esc = min(w.esc * esc, esc_max)
        else:
            back = i == w.cur
            accepted = False
            if back:
                g["n_back"] += 1
            elif i not in w.recent:
                d = (mem.e[i] + lam * math.log1p(mem.visits[i])) - \
                    (w.e + lam * math.log1p(mem.visits[w.cur]))
                accepted = d <= 0 or rng.random() < math.exp(-d / kT)
            if accepted or back:
                g["n_acc"] += 1
            good = new and (mem.e[i] < w.best - 1e-3 or top)
            if good:
                reward = 1.0
                w.stag = 0.0
            else:
                w.stag += 0.5 if new else 1.0
            if rarm is not None:  # common currency for all arms: new global top-n minima
                reward = 1.0 if (new and top) else 0.0
            w.esc = 1.0 if new else min(w.esc * esc, esc_max)
            w.best = min(w.best, mem.e[i])
            if accepted:
                w.recent.append(w.cur)
                if len(w.recent) > tenure:
                    del w.recent[0]
                w.cur, w.e = i, mem.e[i]
                if merge and occupied(i, w):
                    g["n_merge"] += 1
                    w.stag = patience
        w.S += reward
        w.N += 1.0

        # ---- BH step adaptation (population-wide)
        if g["n_try"] % adapt_every == 0:
            rate = g["n_acc"] / g["n_try"]
            g["step"] = float(np.clip(g["step"] * (1.1 if rate > target_accept else 0.9),
                                      0.05 * bond, 0.75 * bond))
        if w.stag >= patience:
            reseed(w)
        ctx.stats.update(g, archive=len(mem), accept_rate=g["n_acc"] / g["n_try"])
