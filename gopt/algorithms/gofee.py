"""GOFEE: Global Optimization with First-principles Energy Expressions (baseline)."""
import time

import numpy as np

from .. import ops
from ..context import BAD_ENERGY, BudgetExhausted
from . import _gofee_ops as gops
from ._gofee_fp import Fingerprint
from ._gofee_gp import GP

LABEL = "GOFEE"
FAMILY = "baseline"
REFERENCE = ("M. K. Bisbo & B. Hammer, Phys. Rev. Lett. 124, 086102 (2020); "
             "Phys. Rev. B 105, 245404 (2022)")
DESCRIPTION = (
    "GOFEE with a GPR surrogate trained on all NEP data (energies only). Descriptor: GOFEE/"
    "Valle-Oganov radial (Rc1=6, bin 0.2, sigma 0.2, 1/r^2 weight) + angular (Rc2=4, 30 bins, "
    "sigma 0.2, polynomial cutoff gamma=2) fingerprint centred on the search atoms, host atoms "
    "(with images) enter as neighbours. Kernel: double Gaussian A[exp(-d^2/2l1^2)+0.01 "
    "exp(-d^2/2l2^2)], relative noise 1e-5, prior = repulsive (0.7*cov. radii)^12 term + mean "
    "bias; l1, l2, angular weight eta and A by max. marginal likelihood (every iteration for the "
    "first 50, then every 10). Each iteration: 30 candidates from a population of 10 (random "
    "structure 0.2, rattle 3 atoms within 4 A 0.6, cut-and-splice 0.1, permutation 0.1 if "
    "multi-element else rattle; bulk: strain 0.1 instead of part of rattle), relaxed in the "
    "surrogate LCB E-2*sigma (batched L-BFGS, fmax 0.1, max_relax_dist 4 A); population members "
    "relaxed in the surrogate mean are added as candidates; certainty filter (sigma<0.7 sqrt(K0)); "
    "the lowest-LCB candidate gets one NEP single point plus the GOFEE dual point "
    "(0.1*F*min(1/5,1/Fmax)); the lower of the two enters the population (kernel similarity "
    "0.9999). Deviations: (1) every 10 iterations the lowest-energy not-yet-relaxed population "
    "member is relaxed with NEP (ctx.relax, <=100 steps, every 5th step + end point added to the "
    "training set) because hits need structures within 0.02 eV of a NEP minimum; (2) for wall "
    "time: training set capped at 1000 (lowest half + most recent), hyperparameters fitted on "
    "<=300 points, surrogate relaxations capped (candidates 100 steps, population warm-started "
    "from the previous surrogate minimum, fmax 0.01, 50 steps); (3) fingerprint without "
    "volume/count normalisation (eta fitted relative to a block-norm balance instead of eta=20; "
    "l1, l2 bounds [0.01, 100] x the median distance of the first fit instead of [1, 1000]), "
    "host-host terms dropped (constant), search atoms confined to the region inflated by 2 A for "
    "supported/zeolite systems; (4) numerics: features centred before computing distances, "
    "Cholesky jitter up to 1e-2 if needed, NEP failure sentinels not used as data, any "
    "surrogate failure falls back to a random structure.")

DEFAULTS = dict(
    n_init=10, n_cand=30, pop_size=10, kappa=2.0, min_certainty=0.7,
    p_random=0.2, p_rattle=0.6, p_cut=0.1, p_perm=0.1, p_strain=0.1,
    n_rattle=3.0, rattle_range=4.0, n_perm=3.0, blmin=0.7, blmax=1.4,
    relax_fmax=0.1, relax_steps=100, pop_fmax=0.01, pop_steps=50, max_relax_dist=4.0,
    maxstep=0.2, dual_lmax=0.1, dual_fflat=5.0, similarity=0.9999,
    fit_first=50, fit_every=10, nmax=1000, nfit=300,
    nep_every=10, nep_steps=100, traj_stride=5,
)


# ---------------------------------------------------------------- operators
def _rattle(p, pos, cell, rng, P, fp):
    """GOFEE RattleMutation: each search atom with prob. Nrattle/n, uniform in a 4 A sphere."""
    zone = None
    if fp.box is not None or fp.sphere is not None:
        zone = lambda X: fp.inside(X[:, None, :])  # noqa: E731
    return gops.rattle(p, pos, cell, rng, P["n_rattle"], P["rattle_range"], P["blmin"],
                       P["blmax"], inside=zone)


def _permute(p, pos, rng, P):
    s = p.search_idx
    nsw = max(1, int(rng.binomial(len(s), min(1.0, P["n_perm"] / len(s)))))
    return ops.swap_unlike(p, pos, rng, nswap=nsw)


def _cut_splice(p, a, b, rng):
    (pa, ca), (pb, cb) = a, b
    if p.variable_cell:
        pb = np.array(pb, float)
        pb[p.search_idx] = (pb[p.search_idx] @ np.linalg.inv(cb)) @ ca
    return ops.cut_and_splice(p, pa, pb, rng, ca), ca


def _valid(p, pos, cell, P):
    return gops.valid(p, pos, cell, P["blmin"], P["blmax"])


def _make_candidate(p, pop, rng, P, fp):
    """One new candidate (full positions, cell) from the population."""
    multi = len(set(p.search_symbols)) > 1
    for _ in range(10):
        u = rng.random()
        i, j = rng.permutation(len(pop))[:2] if len(pop) > 1 else (0, 0)
        a, b = pop[i], pop[j]
        pr, pc, pp = P["p_random"], P["p_cut"], P["p_perm"] if multi else 0.0
        ps = P["p_strain"] if p.variable_cell else 0.0
        cell = a["cell"]
        if u < pr:
            return gops.random_structure(p, rng)
        elif u < pr + pc and len(pop) > 1:
            pos, cell = _cut_splice(p, (a["pos"], a["cell"]), (b["pos"], b["cell"]), rng)
        elif u < pr + pc + pp:
            pos = _permute(p, a["pos"], rng, P)
        elif u < pr + pc + pp + ps:
            pos, cell = ops.strain_cell(a["pos"], a["cell"], rng, sigma=0.1)
            pos = np.array(pos)
            pos[: p.n_sub] = a["pos"][: p.n_sub]
        else:
            pos = _rattle(p, a["pos"], cell, rng, P, fp)
        if _valid(p, pos, cell, P):
            return pos, np.array(cell, float)
    return gops.random_structure(p, rng)


# ---------------------------------------------------------------- surrogate relaxation
def _objective(fp, gp, X, nl, kappa):
    F, erep, back = fp.forward(X, nl)
    need_std = bool(np.any(kappa > 0))
    mu, sig, dmu, dsig = gp.predict(F, erep, grad=True, std=need_std)
    if need_std:
        E = mu - kappa * sig
        G = dmu - kappa[:, None] * dsig
    else:
        sig = np.zeros_like(mu)
        E, G = mu, dmu
    gX = back(G, np.ones(len(E)))
    return E, gX, mu, sig


def surrogate_relax(fp, gp, X0, cells, kappa, fmax, steps, maxstep=0.2, max_dist=4.0, mem=10,
                    h0=1.0 / 70.0):
    """Batched L-BFGS minimisation of mu - kappa*sigma for search-atom positions X0 (B,n,3).

    Per structure: stops at max|grad| < fmax, when leaving the allowed zone, or when an atom
    moved more than max_dist from its start (the last accepted position is kept).
    Returns X, objective, mu, sigma, number of batched evaluations.
    """
    X0 = np.array(X0, float)
    B, n = X0.shape[:2]
    kappa = np.broadcast_to(np.asarray(kappa, float), (B,)).copy()
    fmax = np.broadcast_to(np.asarray(fmax, float), (B,)).copy()
    steps = np.broadcast_to(np.asarray(steps, int), (B,)).copy()
    X = X0.copy()
    out_E, out_mu, out_sig = np.zeros(B), np.zeros(B), np.zeros(B)
    active = np.ones(B, bool)
    batch = np.arange(B)
    nl = fp.build(X[batch], cells[batch])
    Xnl = X[batch].copy()
    E, G, mu, sig = _objective(fp, gp, X[batch], nl, kappa[batch])
    out_E[batch], out_mu[batch], out_sig[batch] = E, mu, sig
    Ecur, Gcur = np.zeros(B), np.zeros((B, n, 3))
    Ecur[batch], Gcur[batch] = E, G
    D3 = 3 * n
    Sh, Yh, rho = np.zeros((mem, B, D3)), np.zeros((mem, B, D3)), np.zeros((mem, B))
    ptr = 0
    gamma = np.full(B, h0)
    scale = np.ones(B)
    nev = 1
    for it in range(int(steps.max()) if B else 0):
        gn = np.linalg.norm(Gcur, axis=2).max(axis=1)
        active &= (gn >= fmax) & (steps > it)
        if not active.any():
            break
        act = np.nonzero(active)[0]
        # rebuild the batch / neighbour list when it shrank a lot or atoms moved > skin/2
        moved = np.linalg.norm(X[batch] - Xnl, axis=2).max() if len(batch) else 0.0
        if len(act) < 0.75 * len(batch) or moved > 0.5 * fp.skin:
            batch = act
            nl = fp.build(X[batch], cells[batch])
            Xnl = X[batch].copy()
        # two-loop recursion (all structures at once, rho=0 entries are inert)
        q = Gcur[batch].reshape(len(batch), D3).copy()
        order = [(ptr - 1 - k) % mem for k in range(mem)]
        alph = np.zeros((mem, len(batch)))
        for k in order:
            alph[k] = rho[k, batch] * np.einsum("ij,ij->i", Sh[k, batch], q)
            q -= alph[k][:, None] * Yh[k, batch]
        r = gamma[batch][:, None] * q
        for k in order[::-1]:
            beta = rho[k, batch] * np.einsum("ij,ij->i", Yh[k, batch], r)
            r += (alph[k] - beta)[:, None] * Sh[k, batch]
        d = -r
        g = Gcur[batch].reshape(len(batch), D3)
        bad = np.einsum("ij,ij->i", d, g) >= 0
        if bad.any():
            d[bad] = -h0 * g[bad]
            rho[:, batch[bad]] = 0.0
        d = d.reshape(len(batch), n, 3)
        dl = np.linalg.norm(d, axis=2).max(axis=1)
        f = np.minimum(1.0, maxstep / np.maximum(dl, 1e-12)) * scale[batch]
        d *= f[:, None, None]
        d[~active[batch]] = 0.0
        Xt = X[batch] + d
        Et, Gt, mut, sigt = _objective(fp, gp, Xt, nl, kappa[batch])
        nev += 1
        dE = np.einsum("ij,ij->i", d.reshape(len(batch), D3), g)
        acc = active[batch] & (Et <= Ecur[batch] + 1e-4 * dE + 1e-9)
        far = np.linalg.norm(Xt - X0[batch], axis=2).max(axis=1) > max_dist
        out = ~fp.inside(Xt)
        stop = acc & (far | out)
        acc &= ~stop
        rej = active[batch] & ~acc & ~stop
        active[batch[stop]] = False
        ia = batch[acc]
        if len(ia):
            sv = d[acc].reshape(len(ia), D3)
            yv = (Gt[acc] - Gcur[ia]).reshape(len(ia), D3)
            sy = np.einsum("ij,ij->i", sv, yv)
            good = sy > 1e-12
            Sh[ptr, ia] = sv
            Yh[ptr, ia] = yv
            rho[ptr, ia] = np.where(good, 1.0 / np.where(good, sy, 1.0), 0.0)
            yy = np.einsum("ij,ij->i", yv, yv)
            gamma[ia] = np.where(good, sy / np.maximum(yy, 1e-30), gamma[ia])
            X[ia] = Xt[acc]
            Ecur[ia], Gcur[ia] = Et[acc], Gt[acc]
            out_E[ia], out_mu[ia], out_sig[ia] = Et[acc], mut[acc], sigt[acc]
            scale[ia] = np.minimum(1.0, 2.0 * scale[ia])
        others = np.setdiff1d(np.arange(B), ia)
        Sh[ptr, others] = 0.0
        Yh[ptr, others] = 0.0
        rho[ptr, others] = 0.0
        ptr = (ptr + 1) % mem
        ir = batch[rej]
        if len(ir):
            rho[:, ir] = 0.0
            gamma[ir] = h0
            scale[ir] *= 0.5
            active[ir[scale[ir] < 1e-3]] = False
    return X, out_E, out_mu, out_sig, nev


# ---------------------------------------------------------------- population
def _pop_add(pop, ent, gp, P):
    """GOFEE Population.add: replace a similar member if lower in energy, else insert."""
    size = P["pop_size"]
    if len(pop) >= size and ent["e"] >= max(m["e"] for m in pop):
        return False
    if pop and gp.l1 is not None:
        Fp = np.array([m["Fref"] if m["Fref"] is not None else m["F"] for m in pop])
        sim = gp.similarity(ent["F"][None], Fp)[0]
        for i in np.argsort([m["e"] for m in pop])[::-1]:
            if sim[i] > P["similarity"]:
                if ent["e"] < pop[i]["e"]:
                    pop[i] = ent
                    pop.sort(key=lambda m: m["e"])
                    return True
                return False
    pop.append(ent)
    pop.sort(key=lambda m: m["e"])
    del pop[size:]
    return True


def _pop_dedup(pop, gp, P):
    """Remove the higher-energy member of pairs whose surrogate minima look alike."""
    if len(pop) < 2 or gp.l1 is None:
        return
    Fp = np.array([m["Fref"] if m["Fref"] is not None else m["F"] for m in pop])
    sim = gp.similarity(Fp, Fp)
    drop = set()
    for i in range(len(pop)):
        if i in drop:
            continue
        for j in range(i + 1, len(pop)):
            if j not in drop and sim[i, j] > P["similarity"]:
                drop.add(j)
    if drop:
        pop[:] = [m for k, m in enumerate(pop) if k not in drop]


def _count(st, key, v=1):
    st[key] = st.get(key, 0) + v


# ---------------------------------------------------------------- main loop
def run(ctx, rng, **kw):
    P = dict(DEFAULTS)
    P.update(kw)
    p = ctx.p
    st = ctx.stats
    s = p.search_idx
    bulk = bool(p.variable_cell)
    fp = Fingerprint(p)
    gp = GP(fp.D, fp.Drad, nmax=P["nmax"], nfit=P["nfit"])
    pop = []
    st.setdefault("gofee_D", fp.D)
    st.setdefault("gofee_env", int(len(fp.env)))

    def evaluate(pos, cell):
        return ctx.evaluate(pos, cell if bulk else None)

    def learn(structs, energies):
        t = time.time()
        F, er = fp.features(structs)
        y = np.asarray(energies, float)
        y = np.where(y < 0.1 * BAD_ENERGY, y, np.nan)  # NEP failure sentinels are not data
        try:
            gp.add(F, y, er)
        except Exception as exc:  # numerical failure: the GP is refitted next iteration
            gp.ready = False
            _count(st, "gp_add_errors")
            st["last_error"] = repr(exc)[:200]
        st["t_train"] = round(st.get("t_train", 0.0) + time.time() - t, 2)
        st["n_train"] = len(gp)
        return F

    def entry(pos, cell, e, F, relaxed=False):
        return dict(pos=fp.canonical(pos, cell), cell=np.array(cell, float), e=float(e), F=F,
                    Xref=None, Fref=None, relaxed=relaxed)

    def random_point():
        try:
            pos, cell = gops.random_structure(p, rng)
        except Exception:
            pos, cell = p.random_structure(rng)
        e, _ = evaluate(pos, cell)
        F = learn([(pos, cell)], [e])
        _pop_add(pop, entry(pos, cell, e, F[0]), gp, P)
        _count(st, "n_random_fallback")

    def nep_relax(m):
        """True NEP relaxation of a population member; trajectory points become data."""
        m["relaxed"] = True
        traj = []
        orig = ctx.evaluate

        def rec(x, c=None):
            e, f = orig(x, c)
            traj.append((np.array(x, float), None if c is None else np.array(c, float), e))
            return e, f

        ctx.evaluate = rec
        try:
            res = ctx.relax(m["pos"], m["cell"] if bulk else None, steps=P["nep_steps"])
        finally:
            ctx.evaluate = orig
        _count(st, "n_nep_relax")
        _count(st, "nep_relax_steps", len(traj))
        sub = traj[:-1][:: P["traj_stride"]]
        structs = [(x, m["cell"] if c is None else c) for x, c, _ in sub]
        energies = [e for _, _, e in sub]
        cell_end = res.cell if bulk else p.cell0
        structs.append((res.pos, cell_end))
        energies.append(res.e)
        F = learn(structs, energies)
        _pop_add(pop, entry(res.pos, cell_end, res.e, F[-1], relaxed=True), gp, P)

    def nep_step(pos, cell, it):
        # ------------------------------------------------ NEP evaluation + dual point
        e1, f1 = evaluate(pos, cell)
        structs, energies = [(pos, cell)], [e1]
        fs = f1[s]
        fm = float(np.sqrt((fs ** 2).sum(axis=1).max())) if np.all(np.isfinite(fs)) else 0.0
        if fm > 1e-8:
            pos2 = np.array(pos)
            pos2[s] += P["dual_lmax"] * fs * min(1.0 / P["dual_fflat"], 1.0 / fm)
            e2, _ = evaluate(pos2, cell)
            structs.append((pos2, cell))
            energies.append(e2)
        _count(st, "gofee_steps", len(structs))
        F = learn(structs, energies)
        j = int(np.argmin(energies))
        _pop_add(pop, entry(structs[j][0], cell, energies[j], F[j]), gp, P)
        # ------------------------------------------------ occasional true NEP relaxation
        if P["nep_every"] and it % P["nep_every"] == 0:
            todo = [m for m in pop if not m["relaxed"]]
            if todo:
                nep_relax(min(todo, key=lambda m: m["e"]))
        if gp.l1 is not None and it % 10 == 0:
            st.update(gp_l1=round(gp.l1, 4), gp_l2=round(gp.l2, 4), gp_eta=round(gp.eta, 4),
                      gp_A=round(gp.A, 3), gp_fallbacks=gp.nfallback,
                      pop_e=[round(m["e"], 3) for m in pop[:3]])

    # ------------------------------------------------ initial structures
    for _ in range(P["n_init"]):
        pos, cell = gops.random_structure(p, rng)
        e, _ = evaluate(pos, cell)
        F = learn([(pos, cell)], [e])
        _pop_add(pop, entry(pos, cell, e, F[0]), gp, P)

    it = 0
    nerr = 0
    while True:
        it += 1
        st["iterations"] = it
        try:
            t0 = time.time()
            if not gp.ready or it <= P["fit_first"] or it % P["fit_every"] == 0:
                gp.fit()
            t1 = time.time()
            st["t_fit"] = round(st.get("t_fit", 0.0) + t1 - t0, 2)
            if not gp.ready:
                raise RuntimeError("GP not ready")
            # candidates from the population + population members (warm-started)
            cands = [_make_candidate(p, pop, rng, P, fp) for _ in range(P["n_cand"])]
            npop = len(pop)
            base, cells, X0 = [], [], []
            for m in pop:
                base.append(m["pos"])
                cells.append(m["cell"] if bulk else p.cell0)
                X0.append(m["Xref"] if m["Xref"] is not None else m["pos"][s])
            for pos, cell in cands:
                pc = fp.canonical(pos, cell)
                base.append(pc)
                cells.append(cell if bulk else p.cell0)
                X0.append(pc[s])
            X0, cells = np.array(X0), np.array(cells, float)
            fp.check_extent(X0)
            nc = len(cands)
            kap = np.r_[np.zeros(npop), np.full(nc, P["kappa"])]
            fmx = np.r_[np.full(npop, P["pop_fmax"]), np.full(nc, P["relax_fmax"])]
            stp = np.r_[np.full(npop, P["pop_steps"]), np.full(nc, P["relax_steps"])]
            t2 = time.time()
            Xr, _, mu, sig, nev = surrogate_relax(fp, gp, X0, cells, kap, fmx, stp,
                                                  maxstep=P["maxstep"],
                                                  max_dist=P["max_relax_dist"])
            st["t_relax"] = round(st.get("t_relax", 0.0) + time.time() - t2, 2)
            _count(st, "surrogate_evals", nev)
            # pop refinement: sigma at the refined points (relaxation used kappa=0)
            if npop:
                Fr, er = fp.features([(np.vstack([b[: p.n_sub], x]) if p.n_sub else x, c)
                                      for b, x, c in zip(base[:npop], Xr[:npop], cells[:npop])])
                mu_p, sig_p, _, _ = gp.predict(Fr, er)
                mu[:npop], sig[:npop] = mu_p, sig_p
                for k, m in enumerate(pop):
                    m["Xref"], m["Fref"] = Xr[k], Fr[k]
                _pop_dedup(pop, gp, P)
            # certainty filter on new candidates (GOFEE), acquisition over all
            cert = sig[npop:] / np.sqrt(gp.k0())
            thr = P["min_certainty"]
            for _ in range(5):
                keep = cert < thr
                if keep.any():
                    break
                thr = thr + (1.0 - thr) / 2.0
            ok = np.r_[np.ones(npop, bool), keep]
            acq = np.where(ok & np.isfinite(mu) & np.isfinite(sig), mu - P["kappa"] * sig, np.inf)
            if not np.isfinite(acq).any():
                raise RuntimeError("no finite acquisition value")
            k = int(np.argmin(acq))
            pos = np.array(base[k], float)
            pos[s] = Xr[k]
            cell = cells[k]
            st["last_pick"] = "pop" if k < npop else "cand"
            _count(st, "pick_pop" if k < npop else "pick_cand")
            nerr = 0
        except BudgetExhausted:
            raise
        except Exception as exc:  # surrogate machinery failed: keep sampling
            nerr += 1
            _count(st, "errors")
            st["last_error"] = repr(exc)[:200]
            try:
                random_point()
            except BudgetExhausted:
                raise
            except Exception as exc2:
                st["last_error"] = repr(exc2)[:200]
            if nerr > 20:
                gp.l1 = None  # force a fresh hyperparameter fit
            continue
        try:
            nep_step(pos, cell, it)
        except BudgetExhausted:
            raise
        except Exception as exc:
            _count(st, "errors")
            st["last_error"] = repr(exc)[:200]
