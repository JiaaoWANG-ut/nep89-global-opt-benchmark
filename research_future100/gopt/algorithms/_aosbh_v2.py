"""Adaptive operator selection basin hopping (AOS-BH)."""
import numpy as np

from .. import ops
from ..context import BudgetExhausted
from ..problem import min_image_distances

LABEL = "AOS basin hopping"
FAMILY = "innovative"
REFERENCE = (
    "Wales & Doye, J. Phys. Chem. A 101, 5111 (1997) [basin hopping]; "
    "Fialho, Da Costa, Schoenauer & Sebag, Ann. Math. Artif. Intell. 60, 25 (2010) "
    "[bandit-based adaptive operator selection]; Thierens, GECCO 2005, 1539 [adaptive "
    "pursuit]; Xia et al., IJCAI 2015 [budgeted Thompson sampling]; Raj & Kalyani, "
    "arXiv:1707.09727 (2017) [discounted Thompson sampling]; Rondina & Da Silva, J. Chem. "
    "Inf. Model. 53, 2282 (2013) [operator-rich basin hopping]; Deaven & Ho, Phys. Rev. "
    "Lett. 75, 288 (1995) [cut-and-splice]; Lyakhov et al., Comput. Phys. Commun. 184, "
    "1172 (2013) [success-adapted variation operators]")
DESCRIPTION = (
    "Basin hopping in which every step chooses its move online from a category-aware "
    "portfolio (global rattle, local rattle of a neighbourhood, relocation of the "
    "least-bound atoms (species-normalised NEP atomic energies) to the most coordinated "
    "free surface/void site, twist of half the cluster, rigid rotation/translation of the "
    "cluster on the support or in the pore, swap of unlike atoms, cell strain for bulk, "
    "cut-and-splice with an archived low minimum, and a fresh random structure). The "
    "choice is a budgeted, discounted Thompson-sampling bandit: each move type keeps "
    "a discounted Beta posterior of its success (new best minimum > downhill hop > new "
    "accepted minimum) and a discounted mean cost in ionic steps, and the move with the "
    "largest sampled success per ionic step is played (plus a uniform exploration floor). "
    "Hence the method behaves like random search on easy landscapes and like a "
    "surface-move BH on rugged ones without per-benchmark tuning. Amplitudes adapt per "
    "move towards ~20% returns to the current minimum, the Metropolis temperature adapts "
    "towards a target acceptance of non-returning trials, relaxations whose predicted "
    "final energy (current energy minus an empirically calibrated 95% quantile of the "
    "remaining drop at fmax thresholds) is out of Metropolis reach are aborted early, and "
    "the walker restarts from a random structure or a perturbed archived minimum after "
    "a size-dependent number of moves without improvement.")
DEFAULTS = dict(kT=0.1, target_accept=0.25, gamma=0.97, eps=0.1, abort_q=0.95, calib=0.2,
                patience0=30, patience_n=2.0, strain=0.05)

_LEVELS = (0.5, 0.25, 0.12)
_ESIG = 0.01  # eV; minima closer in energy count as the same (relaxation noise)


class _Min:
    __slots__ = ("pos", "cell", "e", "ae")

    def __init__(self, pos, cell, e, ae):
        self.pos, self.cell, self.e, self.ae = pos, cell, e, ae


class _Bandit:
    """Discounted Thompson sampling with costs (success per ionic step)."""

    def __init__(self, names, rng, gamma, eps):
        self.names = list(names)
        k = len(self.names)
        self.rng, self.gamma, self.eps = rng, gamma, eps
        self.S = np.zeros(k)
        self.F = np.zeros(k)
        self.C = np.zeros(k)
        self.N = np.zeros(k)
        self.c_prior = 50.0
        self.pulls = np.zeros(k, int)
        self.rsum = np.zeros(k)
        self.csum = np.zeros(k)

    def choose(self, avail):
        idx = np.where(avail)[0]
        if self.rng.random() < self.eps:
            return int(self.rng.choice(idx))
        theta = self.rng.beta(1.0 + self.S, 1.0 + self.F)
        cost = (self.C + self.c_prior) / (self.N + 1.0)
        score = np.where(avail, theta / cost, -np.inf)
        return int(np.argmax(score))

    def update(self, a, r, steps):
        g = self.gamma
        self.S *= g
        self.F *= g
        self.C *= g
        self.N *= g
        self.S[a] += r
        self.F[a] += 1.0 - r
        self.C[a] += steps
        self.N[a] += 1.0
        self.pulls[a] += 1
        self.rsum[a] += r
        self.csum[a] += steps


class _AOSBH:
    def __init__(self, ctx, rng, prm):
        self.ctx, self.rng, self.prm = ctx, rng, prm
        p = self.p = ctx.p
        self.s = p.search_idx
        self.ns = len(self.s)
        self.bulk = bool(p.variable_cell) or p.category == "bulk"
        self.surface = p.category == "surface"
        sym = np.array(p.symbols)[self.s]
        self.groups = [np.where(sym == x)[0] for x in sorted(set(sym))]
        self.multi = len(self.groups) > 1
        self.rcov_s = p.rcov[self.s]
        self.d0 = 2.0 * float(np.mean(self.rcov_s))
        d0 = self.d0
        # move amplitudes: (value, lo, hi)
        self.amp = {"rattle": [0.25 * d0, 0.08 * d0, 0.9 * d0],
                    "local": [0.4 * d0, 0.15 * d0, 1.0 * d0],
                    "rigid": [0.5 * d0, 0.15 * d0, 1.2 * d0],
                    "strain": [0.06, 0.02, 0.15]}
        arms = ["rattle", "jump"]
        if self.ns >= 3:
            arms.append("local")
        if not self.bulk and self.ns >= 3:
            arms.append("twist")
        if p.category in ("surface", "zeolite"):
            arms.append("rigid")
        if self.multi:
            arms.append("swap")
        if self.bulk:
            arms.append("strain")
        arms.append("random")
        if not self.bulk and self.ns >= 3:
            arms.append("splice")
        self.arms = arms
        self.bandit = _Bandit(arms, rng, prm["gamma"], prm["eps"])
        self.kT = float(prm["kT"])
        self.kT_lo, self.kT_hi = 0.2 * self.kT, 5.0 * self.kT
        self.tails = {f: [] for f in _LEVELS}
        self.tail_q = {f: None for f in _LEVELS}
        self.archive = []
        self.best_e = np.inf
        self.patience = int(prm["patience0"] + prm["patience_n"] * self.ns)
        self.n_abort = self.n_ret = self.n_restart = self.n_fail = self.n_moves = 0
        self.n_acc = 0
        self.cost_sum = 0.0
        self.cost_n = 0

    # ------------------------------------------------------------ helpers
    def zscore(self, ae):
        z = np.zeros(self.ns)
        if ae is None or len(ae) != self.p.n:
            return z
        v = np.asarray(ae, float)[self.s]
        for g in self.groups:
            x = v[g]
            z[g] = (x - x.mean()) / (x.std() + 0.05)
        return z

    def unwrap(self, m):
        if self.bulk:
            return np.array(m.pos, float)
        return ops.unwrap_search(self.p, m.pos, m.cell)

    def finish(self, pos, cell):
        """Re-seat detached atoms (non-bulk) and validate."""
        p = self.p
        if pos is None or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(cell)):
            return None
        if not self.bulk:
            lost = ops.detached_atoms(p, pos, cell)
            if len(lost):
                pos = ops.relocate_atoms(p, pos, self.rng, lost, cell)
                if pos is None or not np.all(np.isfinite(pos)):
                    return None
        return pos, cell

    # ------------------------------------------------------------ moves
    def strain(self, pos, cell, sigma):
        """Random symmetric strain with principal stretches bounded to [0.8, 1.25]."""
        eps = self.rng.normal(0.0, sigma, (3, 3))
        w, v = np.linalg.eigh(0.5 * (eps + eps.T))
        w = np.clip(w, -0.2, 0.25)
        new = cell @ (np.eye(3) + (v * w) @ v.T)
        return np.asarray(pos) @ np.linalg.solve(cell, new), new

    def op_rattle(self, m):
        pos, cell = np.array(m.pos), np.array(m.cell)
        if self.bulk:
            pos, cell = self.strain(pos, cell, self.prm["strain"])
        return ops.displace(self.p, pos, self.rng, self.amp["rattle"][0], cell=cell), cell

    def op_local(self, m):
        p, rng = self.p, self.rng
        pos, cell = self.unwrap(m), np.array(m.cell)
        X = pos[self.s]
        if self.bulk:
            D = min_image_distances(X, X, cell, p.pbc)
        else:
            D = np.linalg.norm(X[:, None] - X[None], axis=-1)
        if rng.random() < 0.5:
            z = self.zscore(m.ae)
            w = np.exp(1.5 * (z - z.max()))
            c = int(rng.choice(self.ns, p=w / w.sum()))
        else:
            c = int(rng.integers(self.ns))
        order = np.argsort(D[c])
        lim = 1.35 * (self.rcov_s[c] + self.rcov_s)
        nb = [k for k in order if D[c, k] < lim[k]]
        kmax = max(2, self.ns // 3)
        nb = nb[:kmax]
        if len(nb) < 2:
            nb = list(order[:2])
        pos = ops.displace(p, pos, rng, self.amp["local"][0], idx=self.s[nb], cell=cell)
        return pos, cell

    def site(self, pos, cell, i, present):
        """Most coordinated non-overlapping site for atom i next to the present atoms."""
        p, rng = self.p, self.rng
        pres = np.where(present)[0]
        anchors = np.intersect1d(pres, self.s)
        if len(anchors) == 0:
            anchors = pres
        if len(anchors) == 0:
            return None
        M = 40
        js = anchors[rng.integers(len(anchors), size=M)]
        u = ops.random_unit(rng, M)
        if not self.bulk:
            c = pos[anchors].mean(axis=0)
            out = pos[js] - c
            nrm = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.where(nrm > 1e-6, out / np.maximum(nrm, 1e-6), 0.0)
            u = out + 0.8 * u
            u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        r = (p.rcov[i] + p.rcov[js]) * rng.uniform(0.95, 1.08, M)
        X = pos[js] + u * r[:, None]
        if self.bulk:
            X = np.vstack([X, rng.random((M // 2, 3)) @ cell])
        d = min_image_distances(X, pos[pres], cell, p.pbc)
        rc = p.rcov[i] + p.rcov[pres]
        ok = np.all(d >= 0.8 * rc[None], axis=1)
        if not ok.any():
            ok = np.all(d >= p.dmin_factor * rc[None], axis=1)
            if not ok.any():
                return None
        score = (d < 1.3 * rc[None]).sum(axis=1) + 0.5 * rng.random(len(X))
        score[~ok] = -np.inf
        return X[int(np.argmax(score))]

    def op_jump(self, m):
        p, rng = self.p, self.rng
        pos, cell = self.unwrap(m), np.array(m.cell)
        k = 1
        if self.ns >= 10 and rng.random() < 0.4:
            k = 2
        k = min(k, self.ns - 1) if self.ns > 1 else 1
        z = self.zscore(m.ae)
        w = np.exp(1.5 * (z - z.max()))
        pick = rng.choice(self.ns, size=k, replace=False, p=w / w.sum())
        moved = self.s[pick]
        present = np.ones(p.n, bool)
        present[moved] = False
        for i in moved:
            x = self.site(pos, cell, i, present)
            if x is None:
                return None
            pos[i] = x
            present[i] = True
        return pos, cell

    def op_twist(self, m):
        p, rng = self.p, self.rng
        pos, cell = self.unwrap(m), np.array(m.cell)
        X = pos[self.s]
        c = X.mean(axis=0)
        n = np.array([0.0, 0.0, 1.0]) if self.surface else ops.random_unit(rng)
        h = (X - c) @ n
        sel = h > (np.median(h) if self.surface else 0.0)
        if sel.sum() == 0 or sel.sum() == self.ns:
            return None
        R = ops.rotation_about(n, rng.uniform(0.25, 1.75) * np.pi)
        pos[self.s[sel]] = (X[sel] - c) @ R.T + c
        return ops.repair_overlaps(p, pos, cell), cell

    def op_rigid(self, m):
        p, rng = self.p, self.rng
        pos, cell = self.unwrap(m), np.array(m.cell)
        X = pos[self.s]
        c = X.mean(axis=0)
        t = self.amp["rigid"][0] * np.sqrt(rng.random())
        if self.surface:
            if rng.random() < 0.5:
                R = ops.rotation_about([0.0, 0.0, 1.0], rng.uniform(0, 2 * np.pi))
            else:
                R = ops.random_rotation(rng)
            Y = (X - c) @ R.T + c
            phi = rng.uniform(0, 2 * np.pi)
            Y[:, 0] += t * np.cos(phi)
            Y[:, 1] += t * np.sin(phi)
            Y[:, 2] += X[:, 2].min() - Y[:, 2].min()
        else:
            Y = (X - c) @ ops.random_rotation(rng).T + c + ops.random_unit(rng) * t
        pos[self.s] = Y
        return ops.repair_overlaps(p, pos, cell), cell

    def op_swap(self, m):
        k = 1 if self.rng.random() < 0.7 else 2
        return ops.swap_unlike(self.p, m.pos, self.rng, k), np.array(m.cell)

    def op_strain(self, m):
        pos, cell = self.strain(np.array(m.pos), np.array(m.cell), self.amp["strain"][0])
        return ops.displace(self.p, pos, self.rng, 0.04 * self.d0, cell=cell), cell

    def op_random(self, m):
        return self.p.random_structure(self.rng)

    def op_splice(self, m):
        cands = [a for a in self.archive[:8] if abs(a.e - m.e) > _ESIG]
        if not cands:
            return None
        b = cands[int(self.rng.integers(len(cands)))]
        return ops.cut_and_splice(self.p, m.pos, b.pos, self.rng, m.cell), np.array(m.cell)

    def make(self, name, m):
        try:
            out = getattr(self, "op_" + name)(m)
            if out is not None:
                out = self.finish(*out)
        except Exception:  # geometry failure in an operator: fall back to rattle
            out = None
        if out is None:
            self.n_fail += 1
            try:
                out = self.finish(*self.op_rattle(m))
            except Exception:
                out = None
            if out is None:
                out = (np.array(m.pos), np.array(m.cell))
        return out

    # ------------------------------------------------------------ relaxation
    def relax(self, pos, cell, e_ref=None):
        """Relax; abort early if the predicted minimum is out of Metropolis reach.

        The remaining energy drop after the first point with fmax < f is learned from
        a random subset of relaxations that are never aborted (unbiased calibration).
        """
        ctx = self.ctx
        rec = {}
        flag = [False]
        calib = e_ref is None or self.rng.random() < self.prm["calib"]
        thr = None if calib else e_ref + 3.0 * self.kT
        q = self.tail_q

        def cb(i, e, fm):
            for f in _LEVELS:
                if fm < f and f not in rec:
                    rec[f] = e
                    if thr is not None and i >= 2 and q[f] is not None and e - q[f] > thr:
                        flag[0] = True
                        return True
            return False

        n0 = ctx.nsteps
        try:
            res = ctx.relax(pos, cell, abort=cb)
        except BudgetExhausted:
            raise
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            # safety net (e.g. neighbour-list overflow for a pathological cell)
            self.n_fail += 1
            steps = ctx.nsteps - n0
            return _Min(np.array(pos), np.array(cell), np.inf, None), True, False, steps
        steps = ctx.nsteps - n0
        ae = ctx.atomic_energies()
        if calib and not res.aborted:
            for f, e in rec.items():
                t = self.tails[f]
                t.append(e - res.e)
                if len(t) > 100:
                    del t[0]
                if len(t) >= 15:
                    q[f] = float(np.quantile(t, self.prm["abort_q"])) + 0.05
        self.cost_sum += steps
        self.cost_n += 1
        self.bandit.c_prior = self.cost_sum / self.cost_n
        m = _Min(np.array(res.pos), np.array(res.cell), float(res.e), ae)
        return m, bool(res.aborted), flag[0], steps

    def archive_add(self, m):
        for k, a in enumerate(self.archive):
            if abs(a.e - m.e) < _ESIG:
                if m.e < a.e:
                    self.archive[k] = m
                    self.archive.sort(key=lambda a: a.e)
                return
        self.archive.append(m)
        self.archive.sort(key=lambda a: a.e)
        del self.archive[20:]

    def restart(self):
        rng = self.rng
        self.n_restart += 1
        out = None
        if len(self.archive) >= 3 and rng.random() < 0.5:
            k = int(rng.integers(1, min(len(self.archive), 8)))
            base = self.archive[k]
            try:
                pos = ops.displace(self.p, base.pos, rng, 2.0 * self.amp["rattle"][0],
                                   cell=base.cell)
                out = self.finish(pos, np.array(base.cell))
            except Exception:
                out = None
        if out is None:
            out = self.p.random_structure(rng)
        m, aborted, _, _ = self.relax(*out)
        return m, aborted

    # ------------------------------------------------------------ main loop
    def run(self):
        ctx, rng, prm = self.ctx, self.rng, self.prm
        cur, aborted, _, _ = self.relax(*self.p.random_structure(rng))
        while aborted:
            cur, aborted, _, _ = self.relax(*self.p.random_structure(rng))
        self.archive_add(cur)
        self.best_e = min(self.best_e, cur.e)
        walker_best = cur.e
        since = 0
        A = prm["target_accept"]
        avail = np.ones(len(self.arms), bool)
        i_splice = self.arms.index("splice") if "splice" in self.arms else None
        while True:
            if i_splice is not None:
                avail[i_splice] = len(self.archive) >= 3
            a = self.bandit.choose(avail)
            name = self.arms[a]
            pos, cell = self.make(name, cur)
            trial, aborted, early, steps = self.relax(pos, cell, cur.e)
            self.n_moves += 1
            self.n_abort += early
            de = trial.e - cur.e
            ret = (not aborted) and abs(de) < _ESIG
            self.n_ret += ret
            if aborted:
                acc = False
            else:
                acc = de < 0 or rng.random() < np.exp(-de / self.kT)
            if aborted:
                r = 0.0
            elif trial.e < self.best_e - _ESIG:
                r = 1.0
            elif de < -_ESIG:
                r = 0.6
            elif acc and not ret:
                r = 0.15
            else:
                r = 0.0
            self.bandit.update(a, r, steps)
            if name in self.amp:
                amp = self.amp[name]
                amp[0] = float(np.clip(amp[0] * (1.08 if ret else 0.98), amp[1], amp[2]))
            if not ret:
                self.kT *= np.exp(-0.08 * (1 - A)) if acc else np.exp(0.08 * A)
                self.kT = float(np.clip(self.kT, self.kT_lo, self.kT_hi))
            if not aborted:
                self.archive_add(trial)
                self.best_e = min(self.best_e, trial.e)
            if acc:
                cur = trial
                self.n_acc += 1
            if cur.e < walker_best - _ESIG:
                walker_best = cur.e
                since = 0
            else:
                since += 1
            if since >= self.patience:
                new, ab = self.restart()
                if not ab:
                    cur = new
                    self.archive_add(cur)
                    self.best_e = min(self.best_e, cur.e)
                walker_best = cur.e
                since = 0
            self.report()

    def report(self):
        b = self.bandit
        self.ctx.stats.update(
            moves=self.n_moves, accept_rate=round(self.n_acc / max(self.n_moves, 1), 4),
            returns=self.n_ret, early_aborts=self.n_abort, restarts=self.n_restart,
            op_fail=self.n_fail, kT=round(self.kT, 4),
            amp={k: round(v[0], 3) for k, v in self.amp.items()},
            arms={n: [int(b.pulls[k]), round(float(b.rsum[k]), 2), int(b.csum[k])]
                  for k, n in enumerate(b.names)},
            tail_q={str(f): (None if v is None else round(v, 3)) for f, v in self.tail_q.items()})


def run(ctx, rng, **params):
    prm = dict(DEFAULTS)
    prm.update(params)
    _AOSBH(ctx, rng, prm).run()
