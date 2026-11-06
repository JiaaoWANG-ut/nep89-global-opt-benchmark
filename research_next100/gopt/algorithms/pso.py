"""Particle swarm optimisation in the CALYPSO style (local PSO + random structures)."""
import numpy as np

from .. import ops
from ._pso_geom import FPArchive, Space
from ._pso_pg import PGGen
from ._pso_sym import SymGen

LABEL = "Particle swarm optimisation (CALYPSO)"
FAMILY = "baseline"
REFERENCE = ("Wang, Lv, Zhu & Ma, Phys. Rev. B 82, 094116 (2010); Comput. Phys. Commun. 183, "
             "2063 (2012); Lv, Wang, Zhu & Ma, J. Chem. Phys. 137, 084104 (2012); Wang et al., "
             "J. Phys.: Condens. Matter 27, 203203 (2015)")
DESCRIPTION = (
    "CALYPSO-style PSO on locally relaxed structures. Random structures (generation 0 = "
    "pop_size of them, and the random share of every later generation): variable-cell bulk "
    "uses CALYPSO's symmetry-constrained generation (random space group 1-230, lattice of its "
    "crystal system, atoms on general or special positions built from ASE's symmetry "
    "operations, exact composition, dmin checks; fallback p.random_structure); gas-phase "
    "clusters use CALYPSO's point-group-constrained generation (random point group among "
    "the 32 crystallographic ones plus C5, C5v, C5h, D5, D5h, D5d, I, Ih that fit the "
    "composition; orbits of random points on the general position, mirror planes, rotation "
    "axes or the centre inside a compact sphere; dmin, connectivity and region checks; "
    "fallback p.random_structure); supported and zeolite clusters use p.random_structure. "
    "Each generation the lowest-energy 60% of the unique relaxed structures (duplicates "
    "removed with ops.fingerprint + energy instead of CALYPSO's bond-characterisation "
    "matrix) move by v <- w v + c1 r1 (pbest - x) + c2 r2 (lbest - x), x <- x + v with "
    "c1 = c2 = 2, scalar r1, r2 ~ U[0,1], per-component clamp |v| <= V_max and w decreasing "
    "linearly 0.9 -> 0.4 over the ionic-step budget (CALYPSO: over MaxStep generations); "
    "the other 40% are new random structures with random velocities. Default topology is "
    "CALYPSO's local PSO: lbest = best pbest among the 3 particles nearest in fingerprint "
    "space (topology='global' uses gbest). x is the relaxed structure; every candidate is "
    "relaxed with L-BFGS (one stage instead of CALYPSO's 4-stage DFT relaxations). Bulk: x = "
    "fractional coordinates in the Niggli-reduced cell + lattice parameters (ln a,b,c; "
    "angles in rad), both evolved as in the 2015 CALYPSO review, V_max = 0.15 (CALYPSO "
    "suggests 0.1-0.2); clusters: Cartesian, V_max = "
    "0.15 x diameter of a close-packed cluster of n_search atoms. Added for permutable atoms "
    "(not in CALYPSO): pbest/lbest are aligned onto x by per-species Hungarian assignment, "
    "plus Kabsch rotation/reflection and centring for gas-phase clusters, or an origin "
    "shift for crystals (supported/zeolite clusters: absolute positions, minimum image). As "
    "in CALYPSO's Algorithm 1, candidates too similar to known structures are regenerated "
    "(new r1, r2; after 5 tries the velocity is re-initialised randomly). Constraint repair "
    "(overlaps pushed apart, detached atoms re-seated, gas-cluster fragments merged) "
    "replaces rejection. CALYPSO's halting rule (10 generations without a better gbest) "
    "restarts the swarm. pop_size = 0 -> 20..30 growing with n_search (CALYPSO default 30).")
DEFAULTS = dict(pop_size=0, pso_ratio=0.6, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax=0.15,
                topology="local", n_neigh=3, halt_gens=10, max_regen=5, sym_random=True,
                pg_random=True)


def auto_pop(n_search):
    return int(20 + round(10 * np.clip((n_search - 8) / 22.0, 0.0, 1.0)))


class _Particle:
    __slots__ = ("X", "pos", "cell", "e", "fp", "v", "pb_X", "pb_e", "src")


def run(ctx, rng, pop_size=0, pso_ratio=0.6, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax=0.15,
        topology="local", n_neigh=3, halt_gens=10, max_regen=5, sym_random=True,
        pg_random=True):
    p = ctx.p
    sp = Space(p, vmax)
    symgen = None
    try:  # pure geometry set-up, no ctx calls inside
        if sym_random and sp.mode == "cryst" and p.variable_cell and p.vpa:
            symgen = SymGen(p)
        elif pg_random and sp.mode == "free":
            symgen = PGGen(p)
    except Exception:
        symgen = None
    N = int(pop_size) if pop_size else auto_pop(p.n_search)
    N = max(N, 2)
    n_pso = int(np.clip(round(pso_ratio * N), 1, N))
    dfp = 0.01 if p.variable_cell else 0.03
    st = ctx.stats
    st["swarms"] = st.get("swarms", 0) + 1
    for k in ("gens", "n_pso", "n_rand", "n_regen", "n_vreinit", "n_dup", "n_invalid", "n_fail",
              "imp_pso", "imp_rand"):
        st.setdefault(k, 0)
    st.update(pop_size=N, n_pso_per_gen=n_pso)
    n0 = ctx.nsteps
    horizon = max(1, ctx.budget - n0)
    archive = FPArchive(p)
    g = {"e": np.inf, "X": None, "improved": False}

    def make(res, v, parent, src):
        q = _Particle()
        q.src, q.e = src, np.inf
        if res.aborted or not np.isfinite(res.e) or res.e >= 1e5:
            # collapsed cell or broken structure: dropped from the swarm
            st["n_invalid"] += 1
            return q
        try:
            X, pos, cell, m = sp.encode(res.pos, res.cell)
            if sp.mode == "cryst":
                v = sp.transform_velocity(v, m, sp.ns)
            fp = ops.fingerprint(p, pos, cell)
        except Exception:  # pure geometry, no ctx calls inside
            st["n_invalid"] += 1
            return q
        q.X, q.pos, q.cell, q.v, q.fp = X, pos, cell, v, fp
        q.e = float(res.e)
        if parent is not None and parent.pb_e <= q.e:
            q.pb_X, q.pb_e = parent.pb_X, parent.pb_e
        else:
            q.pb_X, q.pb_e = X, q.e
        archive.add(q.fp)
        if q.e < g["e"] - 1e-3:
            g["improved"] = True
            st["imp_" + src] += 1
        if q.e < g["e"]:
            g["e"], g["X"] = q.e, X
            st["gbest_src"] = src
        return q

    def random_particle():
        out = None
        if symgen is not None:
            try:  # pure geometry, no ctx calls inside
                out = symgen.structure(rng)
            except Exception:
                out = None
            st["n_sym"] = st.get("n_sym", 0) + (out is not None)
        pos, cell = out if out is not None else p.random_structure(rng)
        res = ctx.relax(pos, cell)
        st["n_rand"] += 1
        return make(res, sp.random_velocity(rng), None, "rand")

    def repair(pos, cell):
        if sp.mode == "free":
            pos = sp.merge_fragments(pos)
        pos = ops.repair_overlaps(p, pos, cell)
        if sp.mode != "cryst":
            lost = ops.detached_atoms(p, pos, cell)
            if len(lost):
                new = ops.relocate_atoms(p, pos, rng, lost, cell)
                if new is not None:
                    pos = new
        return pos

    def build(par, Xn):
        """Decoded and repaired candidate; ok=False if the parent's lattice had to be kept."""
        out = sp.decode(Xn, par.pos, par.cell)
        ok = out is not None
        if not ok:  # invalid lattice: keep the parent's lattice parameters
            Xn = np.array(Xn)
            Xn[sp.ns:] = par.X[sp.ns:]
            out = sp.decode(Xn, par.pos, par.cell)
        pos, cell = out
        return repair(pos, cell), cell, ok

    def candidate(par, lb_X, w, gen_arch):
        dp = 0.0 if par.pb_X is par.X else sp.align(par.pb_X, par.X, rng)
        dl = 0.0 if lb_X is par.X else sp.align(lb_X, par.X, rng)
        for attempt in range(max_regen + 1):
            if attempt < max_regen:
                v = sp.clamp(w * par.v + c1 * rng.random() * dp + c2 * rng.random() * dl)
            else:
                # stagnated particle (x = pbest = lbest, v ~ 0): random velocity
                v = sp.random_velocity(rng)
                st["n_vreinit"] += 1
            pos, cell, ok = build(par, par.X + v)
            if not ok:
                v = np.array(v)
                v[sp.ns:] = 0.0
            fp = ops.fingerprint(p, pos, cell)
            if archive.min_dist(fp) > dfp and gen_arch.min_dist(fp) > dfp:
                break
            st["n_regen"] += 1
        gen_arch.add(fp)
        if sp.mode == "cryst" and p.variable_cell:
            cell, v = sp.ctx_basis(cell, v)
        return pos, cell, v

    def select(pop):
        kept = []
        for q in sorted((q for q in pop if np.isfinite(q.e)), key=lambda q: q.e):
            if any(ops.same_structure(p, q.e, q.fp, k.e, k.fp) for k in kept):
                st["n_dup"] += 1
                continue
            kept.append(q)
            if len(kept) == n_pso:
                break
        return kept

    def attractors(parents):
        if topology == "global" or len(parents) < 2:
            return [g["X"]] * len(parents)
        n = len(parents)
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = ops.fp_distance(p, parents[i].fp, parents[j].fp)
        out = []
        for i in range(n):
            nb = np.argsort(d[i], kind="stable")[: max(1, n_neigh)]
            if i not in nb:
                nb = np.append(nb, i)
            j = min(nb, key=lambda k: parents[k].pb_e)
            out.append(parents[j].pb_X)
        return out

    pop = [random_particle() for _ in range(N)]
    stall = 0
    while True:
        st["gens"] += 1
        w = w_max - (w_max - w_min) * min(1.0, (ctx.nsteps - n0) / horizon)
        st["w"] = round(float(w), 4)
        g["improved"] = False
        parents = select(pop)
        lbest = attractors(parents)
        gen_arch = FPArchive(p)
        new = []
        for par, lb in zip(parents, lbest):
            try:
                pos, cell, v = candidate(par, lb, w, gen_arch)
            except Exception:  # pure geometry, no ctx calls inside
                st["n_fail"] += 1
                new.append(random_particle())
                continue
            res = ctx.relax(pos, cell)
            st["n_pso"] += 1
            new.append(make(res, v, par, "pso"))
        while len(new) < N:
            new.append(random_particle())
        pop = new
        stall = 0 if g["improved"] else stall + 1
        if stall >= halt_gens:
            return
