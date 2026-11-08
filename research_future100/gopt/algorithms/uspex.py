"""USPEX-style evolutionary algorithm (baseline)."""
import numpy as np

from .. import ops
from . import _uspex_ops as U

LABEL = "USPEX evolutionary algorithm"
FAMILY = "baseline"
REFERENCE = ("A. R. Oganov & C. W. Glass, J. Chem. Phys. 124, 244704 (2006); "
             "A. O. Lyakhov, A. R. Oganov, H. T. Stokes & Q. Zhu, Comput. Phys. Commun. 184, 1172 (2013); "
             "A. O. Lyakhov, A. R. Oganov & M. Valle, Comput. Phys. Commun. 181, 1623 (2010) [softmutation]; "
             "A. O. Lyakhov & A. R. Oganov, Phys. Rev. B 84, 092103 (2011) [bond hardness]; "
             "A. R. Oganov & M. Valle, J. Chem. Phys. 130, 104504 (2009) [fingerprints]; "
             "Q. Zhu, L. Li, A. R. Oganov & P. B. Allen, Phys. Rev. B 87, 195317 (2013) [surfaces]")
DESCRIPTION = (
    "Generational evolutionary algorithm in the style of USPEX. Population size = 2*n_search rounded "
    "to 10 (USPEX default), limited to [10, 30]. The first generation is random; every later "
    "generation = the best keep_frac*pop distinct structures (keepBestHM, USPEX default 0.15; not "
    "re-relaxed, reoptOld=0) + pop new offspring, each locally relaxed (L-BFGS): heredity 50%, "
    "random 20%, softmutation 20%, permutation 10% (multi-species only), lattice mutation 10% "
    "(variable cell only), fractions renormalised. Parents = best 60% after niching (bestFrac), "
    "chosen with probability linear in fitness rank. Heredity: clusters are cut by a random plane "
    "(vertical on surfaces) at a random position (25-75% of the atoms from parent A) and completed "
    "with the atoms on the other side of parent B (centred on A; gas phase: randomly rotated); "
    "crystals join a slab of random position/thickness of A (along a random lattice vector of the "
    "Minkowski-reduced standardised cells; parents randomly shifted) with the complementary slab of "
    "B, lattice = random weighted average, composition repaired with the farthest unused parent "
    "atoms. Softmutation moves atoms along the softest normal mode of a bond-hardness spring model "
    "(Lyakhov & Oganov 2011; frozen substrate atoms are fixed anchors, rigid modes projected out), "
    "largest displacement 3x mean covalent radius (mutationDegree); an individual softmutated again "
    "uses the next softer non-degenerate mode. Lattice mutation: symmetric strain (I+eps), eps_i ~ "
    "N(0, 0.5) clipped to [-1, 1] (Oganov & Glass 2006, mutationRate), plus a 1 A soft-mode "
    "displacement (DisplaceInLatmutation). Crystals from heredity, lattice mutation and random "
    "generation are rescaled to the mean volume of the 5 best structures. Duplicates: Oganov-Valle "
    "fingerprint cosine distance < 0.008 for crystals, energy + ops.fingerprint "
    "(ops.same_structure) for clusters. Aborted relaxations (cell collapse) and non-finite energies "
    "are discarded. stopCrit: after n_search generations without improvement "
    "the run returns and the runner starts a fresh one. Deviations from USPEX: random structures "
    "from p.random_structure (no symmetric/topological generator); population capped at 30 "
    "(USPEX: 60) because of the step budgets; hard-constraint violations are repaired (push-apart; "
    "softmutation amplitude shrunk by 0.8, at most 5x, until the minimum-distance constraint holds) "
    "instead of regenerated; detached atoms/fragments of clusters are re-seated on the cluster "
    "(benchmark rule); no local-order biasing; single-stage relaxation; valence electron counts of "
    "the hardness model from the periodic-table group.")
DEFAULTS = dict(pop_per_atom=2.0, pop_min=10, pop_max=30, best_frac=0.6, keep_frac=0.15,
                frac_heredity=0.5, frac_random=0.2, frac_softmut=0.2, frac_perm=0.1,
                frac_latmut=0.1, slice_lo=0.25, slice_hi=0.75, soft_degree=3.0, soft_tries=5,
                soft_shrink=0.8, lat_sigma=0.5, lat_soft=1.0, swap_frac=0.5, n_vol=5,
                fp_tol=0.008, fp_rmax=10.0, fp_delta=0.08, fp_sigma=0.03, stop_crit=None,
                de_improve=1e-3)


BAD_E = 1e5  # ctx returns 1e6 for non-finite NEP results; such relaxations are discarded


def _schedule(weights, n, rng):
    """Operator list for one generation: counts by largest remainder, random order."""
    names = list(weights)
    w = np.array([weights[k] for k in names], float)
    raw = w / w.sum() * n
    cnt = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - cnt), kind="stable")[: n - cnt.sum()]:
        cnt[k] += 1
    lst = [name for name, c in zip(names, cnt) for _ in range(c)]
    return [lst[i] for i in rng.permutation(len(lst))]


def run(ctx, rng, **params):
    q = dict(DEFAULTS)
    q.update(params)
    p = ctx.p
    st = ctx.stats
    st["runs"] = st.get("runs", 0) + 1
    for key in ("n_off", "n_best"):
        st.setdefault(key, {})
    s = p.search_idx
    N = p.n_search
    bulk = U.is_bulk(p)
    vc = bool(p.variable_cell)
    multi = len(p.species) > 1
    pop = int(np.clip(10 * np.floor(q["pop_per_atom"] * N / 10 + 0.5), q["pop_min"], q["pop_max"]))
    n_par = max(2, int(round(q["best_frac"] * pop)))
    n_keep = max(1, int(q["keep_frac"] * pop + 0.5))
    stop_crit = int(q["stop_crit"] or N)
    st.update(pop=pop, n_par=n_par, n_keep=n_keep)

    weights = {"heredity": q["frac_heredity"], "random": q["frac_random"],
               "softmut": q["frac_softmut"]}
    if multi:
        weights["permutation"] = q["frac_perm"]
    if vc:
        weights["latmut"] = q["frac_latmut"]
    weights = {k: v for k, v in weights.items() if v > 0}

    sym = np.array(p.symbols)[s]
    counts = [int((sym == x).sum()) for x in p.species]
    max_swaps = sum(min(a, b) for i, a in enumerate(counts) for b in counts[i + 1:])
    how_many_swaps = max(1, int(round(q["swap_frac"] * max_swaps)))
    soft_amp = q["soft_degree"] * float(np.mean(p.rcov[s]))
    min_len = 2.0 * float(p.rcov[s].max())
    nval = np.array([U.n_valence(z) for z in p.numbers], float)
    S0 = U.substrate_bv_sums(p)

    # ------------------------------------------------------------ helpers
    def fingerprint(pos, cell):
        try:
            if bulk:
                return U.ov_fingerprint(p, pos, cell, q["fp_rmax"], q["fp_delta"], q["fp_sigma"])
            return ops.fingerprint(p, pos, cell)
        except Exception:  # pathological geometry: unique fingerprint (never a duplicate)
            st["fp_errors"] = st.get("fp_errors", 0) + 1
            return None

    def same(a, b):
        if a["fp"] is None or b["fp"] is None:
            return False
        if bulk:
            return U.cos_distance(a["fp"], b["fp"]) < q["fp_tol"]
        return ops.same_structure(p, a["e"], a["fp"], b["e"], b["fp"])

    def niche(inds):
        out = []
        for d in sorted(inds, key=lambda d: d["e"]):
            if not any(same(d, o) for o in out):
                out.append(d)
        st["n_dup"] = st.get("n_dup", 0) + len(inds) - len(out)
        return out

    def crystal(pos, cell, v_target=None):
        """Bulk offspring: optional volume rescaling, Minkowski-reduced basis (same lattice; keeps
        the 27-image minimum-image distances exact), overlap repair."""
        if vc and v_target:
            pos, cell = U.rescale_volume(pos, cell, v_target)
        cell = U.reduced(cell) if vc else cell
        return ops.repair_overlaps(p, pos, cell), cell

    def random_child(v_target):
        pos, cell = p.random_structure(rng)
        return crystal(pos, cell, v_target) if bulk else (pos, cell)

    def modes(ind):
        if ind["modes"] is None:
            ind["modes"] = U.soft_modes(p, ind["pos"], ind["cell"], nval, S0)
        return ind["modes"]

    def soft_displace(ind, group, amp):
        """Displace along a mode group; shrink amplitude until the min-distance constraint holds."""
        new = np.array(ind["pos"], float)
        for _ in range(int(q["soft_tries"]) + 1):
            new = np.array(ind["pos"], float)
            new[s] += U.mode_displacement(p, group, amp, rng)
            if p.min_dist_ok(new, ind["cell"]):
                break
            amp *= q["soft_shrink"]
        return new

    def finalize(pos, cell):
        return crystal(pos, cell) if bulk else (U.finalize(p, pos, cell, rng), cell)

    def make_child(op, parents, prob, v_target):
        """Returns (pos, cell, operator actually used)."""
        if op == "heredity" and len(parents) >= 2:
            i, j = rng.choice(len(parents), 2, replace=False, p=prob)
            A, B = parents[i], parents[j]
            if bulk:
                res = U.heredity_bulk(p, A["pos"], A["cell"], B["pos"], B["cell"], rng,
                                      v_target or U.volume(A["cell"]), min_len,
                                      q["slice_lo"], q["slice_hi"], vc)
                if res is not None:
                    return (*crystal(*res), op)
            else:
                pos = U.heredity_cluster(p, A["pos"], B["pos"], rng, A["cell"],
                                         q["slice_lo"], q["slice_hi"])
                return (*finalize(pos, A["cell"]), op)
        elif op in ("softmut", "permutation", "latmut") and parents:
            P = parents[rng.choice(len(parents), p=prob)]
            if op == "softmut":
                groups = modes(P)
                if groups:
                    g = groups[P["nsoft"] % len(groups)]
                    P["nsoft"] += 1
                    pos = soft_displace(P, g, soft_amp)
                    return (*finalize(pos, P["cell"]), op)
            elif op == "permutation":
                nswap = int(rng.integers(1, how_many_swaps + 1))
                pos = ops.swap_unlike(p, P["pos"], rng, nswap)
                return (*finalize(pos, P["cell"]), op)
            else:
                groups = modes(P)
                pos = soft_displace(P, groups[0], q["lat_soft"]) if groups else P["pos"]
                res = U.lattice_mutation(p, pos, P["cell"], rng, q["lat_sigma"],
                                         v_target or U.volume(P["cell"]), min_len)
                if res is not None:
                    return (*crystal(*res), op)
        pos, cell = random_child(v_target)
        return pos, cell, "random"

    def relax(pos, cell, op):
        res = ctx.relax(pos, cell)
        st["n_off"][op] = st["n_off"].get(op, 0) + 1
        if (res.aborted or not np.isfinite(res.e) or res.e >= BAD_E
                or not np.all(np.isfinite(res.pos)) or not np.all(np.isfinite(res.cell))):
            st["n_abort"] = st.get("n_abort", 0) + 1
            return None
        cell = U.reduced(res.cell) if vc else np.array(res.cell)
        return dict(pos=np.array(res.pos), cell=cell, e=float(res.e),
                    fp=fingerprint(res.pos, cell), modes=None, nsoft=0, op=op)

    # ------------------------------------------------------- first generation
    popl = []
    for _ in range(pop):
        pos, cell = p.random_structure(rng)
        ind = relax(pos, cell, "random")
        if ind is not None:
            popl.append(ind)
    best = np.inf
    stall = 0
    while True:
        uniq = niche(popl)
        if uniq and uniq[0]["e"] < best - q["de_improve"]:
            if np.isfinite(best):
                op = uniq[0]["op"]
                st["n_best"][op] = st["n_best"].get(op, 0) + 1
            best = uniq[0]["e"]
            stall = 0
        else:
            stall += 1
        st["gens"] = st.get("gens", 0) + 1
        if stall >= stop_crit:
            return  # USPEX stopCrit reached: the runner starts a new run
        parents = uniq[:n_par]
        keep = uniq[:n_keep]
        prob = np.arange(len(parents), 0, -1, dtype=float)
        prob /= prob.sum() if len(prob) else 1.0
        v_target = None
        if vc and uniq:
            v_target = float(np.mean([U.volume(d["cell"]) for d in uniq[: int(q["n_vol"])]]))
        children = []
        for op in _schedule(weights, pop, rng):
            try:
                pos, cell, used = make_child(op, parents, prob, v_target)
            except Exception:  # geometry-only code (no ctx calls): fall back to a random structure
                st["op_errors"] = st.get("op_errors", 0) + 1
                pos, cell = random_child(v_target)
                used = "random"
            ind = relax(pos, cell, used)
            if ind is not None:
                children.append(ind)
        popl = keep + children
