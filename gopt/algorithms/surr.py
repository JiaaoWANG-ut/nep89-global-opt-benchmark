"""Online-surrogate pre-relaxation basin hopping (key ``surr``).

Every NEP call made by this run (energy, forces and, for variable cells, the virial)
is recorded and used to fit a linear moment-tensor surrogate online
(``_surr_model.py``). The surrogate is then used in two ways that cost no ionic steps:

1. candidate screening: a batch of BH-type candidates (walker / elite perturbations,
   atom relocations, crossovers, homotop swaps, random structures) is relaxed on the
   surrogate, candidates that fall back into the walker's basin or into an already
   visited NEP minimum are discarded by fingerprint, and only the lowest predicted
   minimum is relaxed with NEP;
2. surrogate-Newton local relaxation: each NEP step jumps to the minimum (inside a
   trust region) of the surrogate corrected to reproduce the NEP energy and forces of
   the current point, so that a pre-relaxed candidate converges in a few ionic steps.

Unscreened cycles (one draw from the same operator mix, NEP-relaxed directly) are
mixed in with a probability that grows with the measured out-of-sample error of the
surrogate at NEP minima (and always while it is untrained or when the wall-time
governor says the surrogate is too expensive).

Robustness rules: the first ``n_warm`` minima come from random structures (diverse
training data, random-search-like start); surrogate minima predicted more than
``hole`` eV/atom below every NEP energy seen so far are discarded as model holes; the
online error estimate is capped per sample so one outlier cannot disable screening;
the screening walk step is adapted so that about half of the walk candidates leave
the walker's basin on the surrogate ("escape" control - acceptance is meaningless
for novelty-filtered moves). The governor keeps the run on a ``wall_target``
schedule: behind schedule, a cycle does no surrogate work at all (no refit, no
screening, plain NEP relaxation of an unscreened candidate).
"""
import time

import numpy as np

from .. import ops
from ..context import BudgetExhausted, RelaxResult
from ..problem import min_image_distances
from ._surr_model import RidgeAccumulator, Surrogate, relax_batch

LABEL = "Online-surrogate pre-relaxation BH"
FAMILY = "innovative"
REFERENCE = (
    "Wales & Doye, J. Phys. Chem. A 101, 5111 (1997) [basin hopping]; "
    "Shapeev, Multiscale Model. Simul. 14, 1153 (2016) [moment tensor potentials]; "
    "Baskes, PRB 46, 2727 (1992) [MEAM angular densities]; "
    "Jacobsen, Jorgensen & Hammer, PRL 120, 026102 (2018) [on-the-fly learned structure search]; "
    "Kolsbjerg, Peterson & Hammer, PRB 97, 195424 (2018) [NN-accelerated global optimisation]; "
    "Podryabinkin, Tikhonov, Shapeev & Oganov, PRB 99, 064114 (2019) [MLIP pre-relaxation "
    "of candidates in crystal structure prediction]; "
    "del Rio, Mortensen & Jacobsen, PRB 100, 104103 (2019) [GPMin surrogate-accelerated "
    "local optimisation]; Bisbo & Hammer, PRL 124, 086102 (2020) [GOFEE, contrasted]")
DESCRIPTION = (
    "Basin hopping in which NEP ionic steps are spent only on fully relaxed, pre-screened "
    "minima. A linear moment-tensor/MEAM-like surrogate (per-atom energies from radial "
    "densities and l<=3 angular invariants of species-resolved Gaussian shells around "
    "every search atom, frozen-host atoms included as neighbours; exact guard prior) is "
    "refitted online by ridge regression on the energies, forces and virials of all NEP "
    "steps of the run (no pretrained data). Each cycle relaxes ~12 candidates "
    "(perturbations, relocations, crossover, swaps, random) on the surrogate for free, "
    "drops those that fall back into the walker's or an already visited basin "
    "(species-resolved fingerprints), and NEP-relaxes only the lowest predicted one. "
    "Novel: the NEP relaxation itself is surrogate-Newton - every ionic step jumps to "
    "the trust-region minimum of the surrogate corrected to the current NEP energy and "
    "forces (variable cell included), so minima cost a few ionic steps; Metropolis "
    "acceptance as BH. Unlike GOFEE (global GP, LCB single points) all NEP data are "
    "converged minima. Robust rules: random warm-up minima, model-hole filter, capped "
    "online error, walk step tuned to the surrogate basin-escape rate. Falls back to "
    "plain BH when the out-of-sample surrogate error at minima is large or the "
    "wall-time governor triggers.")
DEFAULTS = dict(kT=0.1, step=0.5, strain=0.05, n_cand=12, n_warm=3, s_fmax=0.1, s_steps=150,
                keep=8, newton=True, sn_steps=25, p_plain=0.1, err_good=0.03, err_bad=0.1,
                wall_target=2400.0, adapt="escape", hole=0.5, warm_random=True, plain_mix=1.0)

_BAD = 1e5


def run(ctx, rng, **params):
    kw = dict(DEFAULTS)
    kw.update(params)
    alg = _Surr(ctx, rng, **kw)
    had = "evaluate" in vars(ctx)
    inner = ctx.evaluate

    def evaluate(pos, cell=None):
        e, f = inner(pos, cell)
        alg.record(pos, cell, e, f)
        return e, f

    ctx.evaluate = evaluate
    try:
        alg.main()
    finally:
        if had:
            ctx.evaluate = inner
        else:
            del ctx.evaluate


class _Min:
    def __init__(self, pos, cell, e, fp, ae=None):
        self.pos, self.cell, self.e, self.fp, self.ae = pos, cell, e, fp, ae


class _Surr:
    def __init__(self, ctx, rng, kT, step, strain, n_cand, n_warm, s_fmax, s_steps, keep,
                 newton, sn_steps, p_plain, err_good, err_bad, wall_target, adapt="escape",
                 hole=0.5, warm_random=True, plain_mix=1.0):
        self.ctx, self.rng, self.p = ctx, rng, ctx.p
        p = self.p
        self.kT, self.step, self.strain = float(kT), float(step), float(strain)
        self.step_bh = float(step)
        self.adapt, self.hole, self.warm_random = str(adapt), float(hole), bool(warm_random)
        self.plain_mix = float(plain_mix)
        self.plain_bh = False
        self.n_wk = self.n_esc = 0
        self.n_acc_bh = self.n_try_bh = 0
        self.n_cand, self.n_warm = int(n_cand), int(n_warm)
        self.s_fmax, self.s_steps, self.keep = float(s_fmax), int(s_steps), int(keep)
        self.newton, self.sn_steps = bool(newton), int(sn_steps)
        self.p_plain, self.err_good, self.err_bad = float(p_plain), float(err_good), float(err_bad)
        self.wall_target = float(wall_target)
        self.S = np.asarray(p.search_idx)
        self.nS = len(self.S)
        self.vc = bool(p.variable_cell)
        self.bulk = self.vc or p.category == "bulk"
        sym = np.array(p.symbols)
        self.ssym = sym[self.S]
        self.spec = sorted(set(self.ssym))
        self.multi = len(self.spec) > 1
        self.m = Surrogate(p)
        self.acc = RidgeAccumulator(self.m.P)
        self.trained = False
        self.traj, self.pending = [], []
        self.n_data = 0
        self.e_shift = None
        self.ebest = np.inf
        self.last_ae = None
        self.known = []          # fingerprints of NEP minima
        self.known_e = []
        self.elite = []          # distinct low NEP minima
        self.walker = None
        self.err_min = None
        self.err_rank = None
        self.t_sur = 0.0
        self.sn_ok = 1.0
        self.t_sn = 0.0
        self.t_screen = 0.0
        self.t_fit = 0.0
        self.n_relax = 0
        self.n_acc = self.n_try = 0
        self.cnt = dict(cycles=0, sur=0, plain=0, sn=0, sn_fallback=0, dup_ref=0, dup_known=0,
                        no_novel=0, invalid=0, refits=0, errors=0, governor=0, hole=0)
        self.steps_sur = []
        self.steps_plain = []
        rc = p.rcov
        rs = rc[self.S]
        self.dmin_ss = p.dmin_factor * (rs[:, None] + rs[None, :])
        self.bond_ss = 1.6 * (rs[:, None] + rs[None, :])
        np.fill_diagonal(self.dmin_ss, 0.0)
        np.fill_diagonal(self.bond_ss, 0.0)
        self.rs = rs
        if self.m.tree is not None:
            nimg = len(self.m.sub_xyz) // p.n_sub
            self.rsub = np.tile(rc[: p.n_sub], nimg)
        self.pairs = []
        for a_i, a in enumerate(self.spec):
            for b in self.spec[a_i:]:
                ia = np.where(self.ssym == a)[0]
                ib = np.where(self.ssym == b)[0]
                self.pairs.append((ia, ib, a == b))
        self.tol_img = 0.01 if self.bulk else 0.03
        self.tol_known = 0.015 if self.bulk else 0.05

    # ------------------------------------------------------------ data
    def record(self, pos, cell, e, f):
        if not np.isfinite(e) or e >= _BAD:
            return
        pos = np.asarray(pos, float)
        X = pos[self.S].copy()
        C = np.array(cell if (self.vc and cell is not None) else self.p.cell0, float)
        V = np.array(self.ctx.last_virial, float) if self.vc else None
        self.traj.append((X, C, float(e), np.array(f, float)[self.S].copy(), V))
        ae = self.ctx.last_atomic_e
        if ae is not None:
            self.last_ae = np.asarray(ae, float)[self.S].copy()
        if e < self.ebest:
            self.ebest = float(e)

    def flush(self):
        tr, self.traj = self.traj, []
        n = len(tr)
        if not n:
            return
        k = min(n, self.keep)
        idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
        self.pending.extend(tr[i] for i in idx)
        if len(self.pending) > 3000:  # backlog while the governor blocks refits
            del self.pending[:-3000]

    def refit(self, force=False):
        need = max(8, int(0.05 * self.n_data))
        if not self.pending or (len(self.pending) < need and not force):
            return
        t0 = time.time()
        fr, self.pending = self.pending, []
        m, nS, P = self.m, self.nS, self.m.P
        if self.e_shift is None:
            self.e_shift = min(f[2] for f in fr)
        sigE, sigF, sigV = 0.01 * nS, 0.15, 0.03 * nS
        cap = self.ebest + 3.0 * nS
        fr = [f for f in fr if f[2] < cap and np.abs(f[3]).max() < 50.0]
        for k0 in range(0, len(fr), 16):
            ch = fr[k0:k0 + 16]
            B = len(ch)
            X = np.array([f[0] for f in ch])
            C = np.array([f[1] for f in ch])
            E = np.array([f[2] for f in ch])
            F = np.array([f[3] for f in ch])
            pl = m.build(X, C)
            XE, XF, XV, Ep, Fp, Vp = m.design(pl, X, C, want_virial=self.vc)
            w = 1.0 / (1.0 + ((E - self.ebest) / (0.1 * nS)) ** 2)
            rows = [XE, XF.reshape(-1, P)]
            ys = [E - self.e_shift - Ep, (F - Fp).ravel()]
            sws = [np.sqrt(w) / sigE, np.repeat(np.sqrt(w / (3 * nS)) / sigF, 3 * nS)]
            if self.vc:
                V = np.array([f[4] for f in ch]).reshape(B, 9)
                rows.append(XV.reshape(-1, P))
                ys.append((V - Vp).ravel())
                sws.append(np.repeat(np.sqrt(w / 9.0) / sigV, 9))
            A = np.vstack(rows)
            y = np.concatenate(ys)
            sw = np.concatenate(sws)
            self.acc.validate(A, y, sw)
            self.acc.add(A, y, sw)
            self.n_data += B
        if self.n_data:
            wnew = self.acc.solve()
            if np.all(np.isfinite(wnew)):
                m.w = wnew
                self.trained = True
            self.cnt["refits"] += 1
        self.t_sur += time.time() - t0
        self.t_fit += time.time() - t0

    def predict(self, X, C):
        X = np.asarray(X, float)[None]
        C = np.asarray(C, float)[None]
        pl = self.m.build(X, C)
        E, F, V = self.m.energy_forces(pl, X, C, want_virial=self.vc)
        return float(E[0]) + (self.e_shift or 0.0), F[0], (None if V is None else V[0])

    # ------------------------------------------------------------ geometry helpers
    def _wrap(self, X):
        p = self.p
        fr = X @ self.m.inv0
        fr -= np.where(p.pbc, np.floor(fr), 0.0)
        return fr @ p.cell0

    def fingerprint(self, pos, cell):
        if self.bulk:
            return ops.fingerprint(self.p, pos, cell)
        X = np.asarray(pos)[self.S]
        d = min_image_distances(X, X, self.p.cell0, self.p.pbc)
        parts = []
        for ia, ib, same in self.pairs:
            if same:
                iu = np.triu_indices(len(ia), 1)
                parts.append(np.sort(d[np.ix_(ia, ia)][iu]))
            else:
                parts.append(np.sort(d[np.ix_(ia, ib)].ravel()))
        if self.m.tree is not None:
            dsub, _ = self.m.tree.query(self._wrap(X), k=1)
            for a in self.spec:
                parts.append(np.sort(dsub[self.ssym == a]))
        return np.concatenate(parts)

    def fpdist(self, a, b):
        if self.bulk:
            return 1.0 - np.asarray(b) @ a
        return np.abs(np.asarray(b) - a).mean(axis=-1)

    def valid(self, pos, cell):
        p = self.p
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(cell)):
            return False
        if self.bulk:
            return p.min_dist_ok(pos, cell)
        X = np.asarray(pos)[self.S]
        d = min_image_distances(X, X, p.cell0, p.pbc)
        np.fill_diagonal(d, np.inf)
        if np.any(d < self.dmin_ss):
            return False
        att = np.any(d < self.bond_ss, axis=1)
        if self.m.tree is not None:
            k = min(4, len(self.m.sub_xyz))
            dist, j = self.m.tree.query(self._wrap(X), k=k)
            dist = dist.reshape(self.nS, -1)
            j = j.reshape(self.nS, -1)
            rsum = self.rs[:, None] + self.rsub[j]
            if np.any(dist < p.dmin_factor * rsum):
                return False
            att |= np.any(dist < 1.6 * rsum, axis=1)
        return bool(att.all())

    # ------------------------------------------------------------ moves
    def _disp(self, pos, step):
        pos = np.array(pos, float)
        r = step * self.rng.random(self.nS) ** (1 / 3)
        pos[self.S] += ops.random_unit(self.rng, self.nS) * r[:, None]
        return pos

    def _pick_elite(self):
        n = len(self.elite)
        w = np.exp(-0.5 * np.arange(n))
        return self.elite[int(self.rng.choice(n, p=w / w.sum()))]

    def make_one(self):
        rng, p = self.rng, self.p
        wk = self.walker
        u = rng.random()
        if self.bulk:
            table = [("walk", 0.3 if self.multi else 0.4), ("elite", 0.2), ("rand", 0.4)]
            if self.multi:
                table.append(("swap", 0.1))
        else:
            table = [("walk", 0.35), ("elite", 0.15), ("reloc", 0.15), ("rand", 0.15),
                     ("cross", 0.1), ("swap", 0.1)]
            if not self.multi:
                table[0] = ("walk", 0.45)
                table.pop()
        tot = sum(t[1] for t in table)
        acc = 0.0
        op = table[0][0]
        for name, pr in table:
            acc += pr / tot
            if u < acc:
                op = name
                break
        if op == "rand":
            pos, cell = p.random_structure(rng)
            return pos, cell, op
        base = wk
        if op in ("elite", "cross") and self.elite:
            base = self._pick_elite()
        pos, cell = np.array(base.pos), np.array(base.cell)
        if op == "cross" and len(self.elite) >= 2:
            other = self._pick_elite()
            if other is base:
                other = wk
            pos = ops.cut_and_splice(p, pos, other.pos, rng, cell)
            return pos, cell, op
        if op == "swap":
            pos = ops.swap_unlike(p, pos, rng, nswap=int(rng.integers(1, 3)))
            return self._disp(pos, 0.15), cell, op
        if op == "reloc":
            k = int(rng.integers(1, max(1, min(3, self.nS // 4)) + 1))
            if base.ae is not None and rng.random() < 0.5:
                sel = np.argsort(-base.ae)[:k]
            else:
                sel = rng.choice(self.nS, k, replace=False)
            pos = ops.relocate_atoms(p, pos, rng, self.S[sel], cell)
            return pos, cell, op
        if self.vc:
            pos, cell = ops.strain_cell(pos, cell, rng, self.strain * rng.uniform(0.5, 1.5))
        f = rng.uniform(0.6, 1.6) if op == "walk" else rng.uniform(0.5, 1.2)
        return self._disp(pos, self.step * f), cell, op

    def plain_candidate(self):
        """Unscreened candidate: random during warm-up, else (probability ``plain_mix``)
        one draw from the screening operator mix, otherwise a classic BH displacement."""
        p, rng = self.p, self.rng
        self.plain_bh = False
        if self.warm_random and self.n_relax < self.n_warm:
            return p.random_structure(rng)
        pos = None
        if rng.random() < self.plain_mix:
            try:
                pos, cell, _ = self.make_one()
            except Exception:
                self.cnt["errors"] += 1
                pos = None
        if pos is None:
            self.plain_bh = True
            pos, cell = np.array(self.walker.pos), np.array(self.walker.cell)
            if self.vc:
                pos, cell = ops.strain_cell(pos, cell, rng, self.strain)
            pos = ops.displace(p, pos, rng, self.step_bh, cell=cell)
        if not self.vc and p.category != "bulk":
            lost = ops.detached_atoms(p, pos, cell)
            if len(lost):
                pos = ops.relocate_atoms(p, pos, rng, lost, cell)
        return pos, cell

    # ------------------------------------------------------------ surrogate screening
    def screen(self):
        """Relax candidates on the surrogate; return (pos, cell, E_pred, dE_pred) or None."""
        t0 = time.time()
        cands, opsu = [], []
        for _ in range(self.n_cand):
            try:
                pos, cell, op = self.make_one()
            except BudgetExhausted:
                raise
            except Exception:
                pos, cell, op = self._disp(self.walker.pos, self.step), np.array(self.walker.cell), "walk"
            cands.append((np.asarray(pos, float), np.asarray(cell, float)))
            opsu.append(op)
        refs = [self.walker] + [e for e in self.elite[:2] if e is not self.walker]
        for r in refs:
            cands.append((np.array(r.pos), np.array(r.cell)))
            opsu.append("ref")
        X = np.array([c[0][self.S] for c in cands])
        C = np.array([c[1] if self.vc else self.p.cell0 for c in cands])
        Xr, Cr, Es, fm, ok, _ = relax_batch(self.m, X, C, fmax=self.s_fmax, steps=self.s_steps,
                                            maxstep=0.2)
        nc = self.n_cand
        ref_fp = []
        e_wk = Es[nc] if np.isfinite(Es[nc]) else None
        for k in range(nc, len(cands)):
            q = np.array(cands[k][0])
            q[self.S] = Xr[k]
            ref_fp.append(self.fingerprint(q, Cr[k] if self.vc else self.p.cell0))
        fps = {}

        def cand_fp(k):
            if k not in fps:
                q = np.array(cands[k][0])
                q[self.S] = Xr[k]
                fps[k] = (q, self.fingerprint(q, Cr[k] if self.vc else self.p.cell0))
            return fps[k]

        if self.adapt == "escape":
            # walk step adapted so that about half of the walk moves leave the walker's
            # basin on the surrogate (acceptance is meaningless for novelty-filtered moves)
            for k in range(nc):
                if opsu[k] == "walk" and ok[k] and np.isfinite(Es[k]):
                    self.n_wk += 1
                    self.n_esc += float(self.fpdist(cand_fp(k)[1], ref_fp[0])) >= self.tol_img
            if self.n_wk >= 20:
                self.step *= 1.1 if self.n_esc < 0.5 * self.n_wk else 0.9
                self.step = float(np.clip(self.step, 0.1, 2.0))
                self.n_wk = self.n_esc = 0
        # predicted energies far below every NEP energy seen so far are model holes
        e_hole = self.ebest - (self.e_shift or 0.0) - self.hole * self.nS
        known = np.array(self.known) if self.known else None
        order = np.argsort(np.where(np.isfinite(Es[:nc]), Es[:nc], np.inf))
        out = None
        for k in order:
            if not ok[k] or not np.isfinite(Es[k]):
                self.cnt["invalid"] += 1
                continue
            if Es[k] < e_hole:
                self.cnt["hole"] += 1
                continue
            pos, fp = cand_fp(k)
            pos = np.array(pos)
            cell = Cr[k] if self.vc else self.p.cell0
            if not self.valid(pos, cell):
                self.cnt["invalid"] += 1
                continue
            if min(float(self.fpdist(fp, r)) for r in ref_fp) < self.tol_img:
                self.cnt["dup_ref"] += 1
                continue
            if known is not None and float(np.min(self.fpdist(fp, known))) < self.tol_known:
                self.cnt["dup_known"] += 1
                continue
            de = None if e_wk is None else float(Es[k] - e_wk)
            out = (pos, np.array(cell), float(Es[k]), de, opsu[k])
            break
        self.t_sur += time.time() - t0
        self.t_screen += time.time() - t0
        return out

    # ------------------------------------------------------------ NEP relaxations
    def relax_sn(self, pos, cell):
        """Surrogate-Newton relaxation: each ionic step goes to the trust-region minimum
        of the surrogate corrected to the NEP energy/forces of the current point."""
        ctx, p, m, S = self.ctx, self.p, self.m, self.S
        vc = self.vc
        cf = float(self.nS)
        base = np.array(pos, float)
        C = np.array(cell if vc else p.cell0, float)

        def nep(X, Cx):
            q = base.copy()
            q[S] = X
            e, f = ctx.evaluate(q, Cx if vc else None)
            G = np.array(f, float)[S]
            if vc:
                v = np.array(ctx.last_virial, float)
                G = np.vstack([G, 0.5 * (v + v.T) / cf])
            return float(e), G

        X = base[S].copy()
        e0, G0 = nep(X, C)
        nst = 1
        fm0 = float(np.sqrt((G0 ** 2).sum(1)).max())
        trust = 0.3
        rej = 0
        maxit = min(self.sn_steps, p.max_relax_steps // 2)
        while fm0 >= p.fmax and e0 < _BAD:
            why = ("err" if rej > 90 else "maxit" if nst >= maxit else "rej" if rej >= 2
                   else "trust" if trust < 0.02 else None)
            if why is not None:
                self.cnt["sn_fallback"] += 1
                self.cnt["fb_" + why] = self.cnt.get("fb_" + why, 0) + 1
                self.sn_ok = 0.8 * self.sn_ok
                q = base.copy()
                q[S] = X
                r = ctx.relax(q, C, steps=max(p.max_relax_steps - nst, 10))
                return RelaxResult(r.pos, r.cell, r.e, r.converged, r.nsteps + nst, r.fmax,
                                   r.aborted)
            t0 = time.time()
            try:
                Xb, Cb = X[None], C[None]
                pl = m.build(Xb, Cb)
                Es, Fs, Vs = m.energy_forces(pl, Xb, Cb, want_virial=vc)
                Gs = Fs[0]
                if vc:
                    Gs = np.vstack([Gs, 0.5 * (Vs[0] + Vs[0].T) / cf])
                corr = ((G0 - Gs)[None], np.array([e0 - Es[0]]))
                Xn, Cn, Ec, _, okn, _ = relax_batch(m, Xb, Cb, fmax=0.5 * p.fmax,
                                                    steps=80, maxstep=min(0.2, trust),
                                                    corr=corr, trust_total=trust)
                good = bool(okn[0]) and np.isfinite(Ec[0]) and np.all(np.isfinite(Xn[0]))
            except BudgetExhausted:
                raise
            except Exception:
                self.cnt["errors"] += 1
                good = False
            self.t_sur += time.time() - t0
            self.t_sn += time.time() - t0
            if not good:
                rej = 99
                continue
            e1, G1 = nep(Xn[0], Cn[0])
            nst += 1
            pred = float(Ec[0]) - e0
            act = e1 - e0
            dx = float(np.sqrt(((Xn[0] - X) ** 2).sum(1)).max())
            if e1 < _BAD and act <= 1e-4:
                X, C, e0, G0 = Xn[0], Cn[0], e1, G1
                fm0 = float(np.sqrt((G0 ** 2).sum(1)).max())
                rej = 0
                rho = act / pred if pred < -1e-3 else 1.0
                if rho > 0.5 and dx > 0.8 * trust:
                    trust = min(2.0 * trust, 1.0)
                elif rho < 0.25:
                    trust = 0.5 * trust
            else:
                trust = 0.5 * min(trust, max(dx, 0.01))
                rej += 1
        q = base.copy()
        q[S] = X
        self.cnt["sn"] += 1
        self.sn_ok = 0.8 * self.sn_ok + 0.2
        return RelaxResult(q, C, e0, fm0 < p.fmax, nst, fm0)

    def local_relax(self, pos, cell, sn=True):
        n0 = self.ctx.nsteps
        reliable = self.err_min is not None and self.err_min < self.err_bad
        retry = self.n_relax % 8 == 0
        try:
            if sn and self.newton and self.trained and reliable and (self.sn_ok > 0.4 or retry):
                res = self.relax_sn(pos, cell)
            else:
                res = self.ctx.relax(pos, cell)
        finally:
            self.flush()
        self.n_relax += 1
        return res, self.ctx.nsteps - n0

    # ------------------------------------------------------------ bookkeeping
    def make_min(self, res):
        cell = np.array(res.cell if self.vc else self.p.cell0, float)
        pos = np.array(res.pos, float)
        return _Min(pos, cell, float(res.e), self.fingerprint(pos, cell),
                    None if self.last_ae is None else self.last_ae.copy())

    def register(self, mn):
        if mn.e >= _BAD:
            return
        if self.known:
            d = self.fpdist(mn.fp, np.array(self.known))
            j = int(np.argmin(d))
            if d[j] < 0.5 * self.tol_known and abs(self.known_e[j] - mn.e) < 0.05:
                pass
            else:
                self.known.append(mn.fp)
                self.known_e.append(mn.e)
        else:
            self.known.append(mn.fp)
            self.known_e.append(mn.e)
        if len(self.known) > 1500:
            keep = np.argsort(self.known_e)[:1000]
            self.known = [self.known[i] for i in keep]
            self.known_e = [self.known_e[i] for i in keep]
        for k, el in enumerate(self.elite):
            if float(self.fpdist(mn.fp, el.fp)) < self.tol_known:
                if mn.e < el.e:
                    self.elite[k] = mn
                    self.elite.sort(key=lambda x: x.e)
                return
        self.elite.append(mn)
        self.elite.sort(key=lambda x: x.e)
        del self.elite[8:]

    def surrogate_error(self, mn):
        if not self.trained:
            return
        try:
            es, _, _ = self.predict(mn.pos[self.S], mn.cell)
        except Exception:
            return
        err = abs(es - mn.e) / self.nS
        if np.isfinite(err):
            # capped so that a single outlier (early random structures, model holes) cannot
            # switch the screening off for long
            err = min(err, 2.0 * self.err_bad)
            self.err_min = err if self.err_min is None else 0.8 * self.err_min + 0.2 * err

    def p_plain_now(self):
        if self.err_min is None:
            return 1.0
        x = (self.err_min - self.err_good) / (self.err_bad - self.err_good)
        return float(np.clip(self.p_plain + x * (0.9 - self.p_plain), self.p_plain, 0.9))

    def update_stats(self):
        c = self.cnt
        self.ctx.stats.update(
            c, accept_rate=self.n_acc / max(self.n_try, 1), step=self.step, step_bh=self.step_bh,
            err_min=self.err_min, err_rank=self.err_rank, p_plain=self.p_plain_now(),
            n_data=self.n_data, t_surrogate=round(self.t_sur, 2), t_sn=round(self.t_sn, 2),
            t_screen=round(self.t_screen, 2), t_fit=round(self.t_fit, 2),
            steps_per_min_sur=float(np.mean(self.steps_sur)) if self.steps_sur else None,
            steps_per_min_plain=float(np.mean(self.steps_plain)) if self.steps_plain else None,
            n_elite=len(self.elite), n_known=len(self.known))

    # ------------------------------------------------------------ main loop
    def main(self):
        ctx, p, rng = self.ctx, self.p, self.rng
        pos, cell = p.random_structure(rng)
        res, ns = self.local_relax(pos, cell)
        self.walker = self.make_min(res)
        self.register(self.walker)
        while True:
            self.cnt["cycles"] += 1
            el = time.time() - ctx.t0
            slow = el > self.wall_target * ctx.nsteps / ctx.budget + 60.0
            if slow:
                # behind the wall-time schedule: no surrogate work at all this cycle (the
                # recorded data are kept and fitted once the run is back on schedule)
                self.cnt["governor"] += 1
            else:
                try:
                    self.refit()
                except BudgetExhausted:
                    raise
                except Exception:
                    self.cnt["errors"] += 1
            use = (self.trained and self.n_relax >= self.n_warm and not slow
                   and rng.random() >= self.p_plain_now())
            cand = None
            if use:
                try:
                    cand = self.screen()
                except BudgetExhausted:
                    raise
                except Exception:
                    self.cnt["errors"] += 1
                    cand = None
                if cand is None:
                    self.cnt["no_novel"] += 1
            if cand is not None:
                res, ns = self.local_relax(cand[0], cand[1], sn=not slow)
                self.cnt["sur"] += 1
                self.steps_sur.append(ns)
            else:
                pos, cell = self.plain_candidate()
                res, ns = self.local_relax(pos, cell, sn=not slow)
                self.cnt["plain"] += 1
                self.steps_plain.append(ns)
            mn = self.make_min(res)
            self.surrogate_error(mn)
            if cand is not None and cand[3] is not None:
                er = abs(cand[3] - (mn.e - self.walker.e))
                self.err_rank = er if self.err_rank is None else 0.8 * self.err_rank + 0.2 * er
            self.register(mn)
            self.n_try += 1
            de = mn.e - self.walker.e
            accepted = bool(not res.aborted and mn.e < _BAD
                            and (de < 0 or rng.random() < np.exp(-de / self.kT)))
            if accepted:
                self.walker = mn
                self.n_acc += 1
            if self.adapt != "escape":
                if self.n_try % 10 == 0:
                    rate = self.n_acc / self.n_try
                    self.step *= 1.1 if rate > 0.5 else 0.9
                    self.step = float(np.clip(self.step, 0.1, 2.0))
                self.step_bh = self.step
            elif cand is None and self.plain_bh:
                # plain BH moves keep the classic acceptance-based step control
                self.n_try_bh += 1
                self.n_acc_bh += accepted
                if self.n_try_bh % 10 == 0:
                    rate = self.n_acc_bh / self.n_try_bh
                    self.step_bh *= 1.1 if rate > 0.5 else 0.9
                    self.step_bh = float(np.clip(self.step_bh, 0.1, 2.0))
            self.update_stats()
