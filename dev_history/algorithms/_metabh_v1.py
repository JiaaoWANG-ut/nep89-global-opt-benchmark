"""Metadynamics-biased basin hopping (MetaBH)."""
import numpy as np

from .. import ops
from ._metabh_cv import CV, Bias

LABEL = "Metadynamics-biased basin hopping"
FAMILY = "innovative"
REFERENCE = ("Laio & Parrinello, PNAS 99, 12562 (2002); Barducci, Bussi & Parrinello, PRL 100, "
             "020603 (2008); Iwamatsu & Okabe, Chem. Phys. Lett. 399, 396 (2004); Wales & Doye, "
             "J. Phys. Chem. A 101, 5111 (1997); cf. energy-landscape paving, Hansmann & Wille, "
             "PRL 88, 068105 (2002)")
DESCRIPTION = (
    "Basin hopping whose Metropolis test uses E + V(s) instead of E. s is a low-dimensional, "
    "energy-agnostic structural descriptor of each local minimum (smooth coordination per "
    "species pair - which also encodes the chemical order of alloys -, mean Steinhardt q4/q6, "
    "radius of gyration and shape anisotropy for finite systems, substrate coordination and "
    "height for supported clusters, log volume per atom for bulk) with generic dimensionless "
    "Gaussian widths. V is a history-dependent sum of Gaussians deposited at the walker's "
    "minimum after every move, with well-tempered heights w0*exp(-V/dT) so that the bias "
    "saturates: regions of structure space where the walker has spent time are progressively "
    "filled, pushing it into unexplored funnels, while the true NEP energy alone defines the "
    "best structure. Moves: all-atom displacements (cell strain for bulk) with a step adapted "
    "to 50% unbiased acceptance, relocation of high-energy atoms (per-atom NEP energies), "
    "unlike-atom swaps, rigid rotations of supported/confined clusters, and occasional jumps "
    "(random restarts accepted unconditionally, with a size-dependent probability and after "
    "stagnation) that keep the bias memory.")
DEFAULTS = dict(kT=0.1, step=0.6, w0=0.05, dT=1.0, sigma_scale=1.0, w_trial=0.0,
                jump_c=2.0, jump_max=0.5, stag_c=1.0, stag_min=10, p_reloc=0.25, p_swap=0.25,
                p_rot=0.1, strain=0.05, target_accept=0.5)


def run(ctx, rng, kT=0.1, step=0.6, w0=0.05, dT=1.0, sigma_scale=1.0, w_trial=0.0, jump_c=2.0,
        jump_max=0.5, stag_c=1.0, stag_min=10, p_reloc=0.25, p_swap=0.25, p_rot=0.1, strain=0.05,
        target_accept=0.5):
    p = ctx.p
    m = p.n_search
    s_idx = p.search_idx
    bulk = bool(p.variable_cell)
    finite = not bulk and p.category != "bulk"
    multi = len(p.species) > 1
    sym = np.array(p.symbols)[s_idx]
    spec_masks = [sym == x for x in p.species]
    cv = CV(p)
    bias = Bias(cv.d)
    sig = cv.sigma * float(sigma_scale)
    p_jump = float(min(jump_max, jump_c / max(m, 1)))
    n_stag = int(max(stag_min, round(stag_c * m)))
    st = ctx.stats
    nk = {k: 0 for k in ("disp", "reloc", "swap", "rot", "jump")}
    na = dict(nk)
    ni = dict(nk)

    def relax(pos, cell):
        r = ctx.relax(pos, cell)
        return r, ctx.atomic_energies()

    state = {}

    def random_min():
        while True:
            try:
                pos, cell = p.random_structure(rng)
            except RuntimeError:
                if not state:
                    raise
                pos, cell = ops.displace(p, state["r"].pos, rng, 2.0), np.array(state["r"].cell)
            r, ea = relax(pos, cell)
            if not r.aborted:
                return r, ea

    def worst_atoms(ea, k):
        """k search atoms chosen by species-normalised per-atom energy (or at random)."""
        if ea is None or rng.random() < 0.5:
            return rng.choice(s_idx, size=k, replace=False)
        e = np.array(ea, float)[s_idx]
        for msk in spec_masks:
            e[msk] -= e[msk].mean()
        return s_idx[np.argsort(-e)[:k]]

    def propose(r, ea):
        pos, cell = np.array(r.pos), np.array(r.cell)
        u = rng.random()
        ps = p_swap if multi else 0.0
        pr = p_rot if finite and p.n_sub else 0.0
        if u < ps:
            kind = "swap"
            pos = ops.swap_unlike(p, pos, rng, nswap=1 if rng.random() < 0.7 else 2)
        elif u < ps + p_reloc:
            kind = "reloc"
            k = 1 + int(rng.integers(max(1, m // 8)))
            idx = worst_atoms(ea, k)
            if finite:
                pos = ops.relocate_atoms(p, pos, rng, idx, cell)
            else:
                new = p.place(pos, idx, rng, cell, connected=False)
                pos = new if new is not None else pos
                pos = ops.displace(p, pos, rng, 0.3 * step, cell=cell)
        elif u < ps + p_reloc + pr:
            kind = "rot"
            pos = ops.rotate_subset(p, pos, rng, s_idx, cell=cell)
            if rng.random() < 0.5:
                v = ops.random_unit(rng)
                if p.category == "surface":
                    v[2] = 0.0
                pos[s_idx] += v * rng.uniform(0.3, 1.0) * 2.0 * cv.r_unit
                pos = ops.repair_overlaps(p, pos, cell)
        else:
            kind = "disp"
            if bulk:
                pos, cell = ops.strain_cell(pos, cell, rng, strain)
            pos = ops.displace(p, pos, rng, step, cell=cell)
        if finite:
            lost = ops.detached_atoms(p, pos, cell)
            if len(lost):
                pos = ops.relocate_atoms(p, pos, rng, lost, cell)
        return kind, pos, cell

    r, ea = random_min()
    state.update(r=r, ea=ea, s=cv(r.pos, r.cell))
    seg_best = r.e
    since = 0
    n_moves = n_acc = 0
    acc_win = []
    bias.deposit(state["s"], sig, w0, dT)
    while True:
        jump = since >= n_stag or rng.random() < p_jump
        if jump:
            kind = "jump"
            nk[kind] += 1
            r, ea = random_min()
            state.update(r=r, ea=ea, s=cv(r.pos, r.cell))
            seg_best = r.e
            since = 0
            na[kind] += 1
        else:
            kind, pos, cell = propose(state["r"], state["ea"])
            nk[kind] += 1
            tr, tea = relax(pos, cell)
            n_moves += 1
            improved = False
            if not tr.aborted:
                s_t = cv(tr.pos, tr.cell)
                de = tr.e - state["r"].e
                db = de + bias(s_t) - bias(state["s"])
                if kind == "disp":
                    acc_win.append(1.0 if de <= 0 else float(np.exp(-de / kT)))
                if db <= 0 or rng.random() < np.exp(-db / kT):
                    state.update(r=tr, ea=tea, s=s_t)
                    n_acc += 1
                    na[kind] += 1
                elif w_trial > 0:
                    bias.deposit(s_t, sig, w0 * w_trial, dT)
                if tr.e < seg_best - 1e-3:
                    seg_best = tr.e
                    improved = True
                    ni[kind] += 1
            since = 0 if improved else since + 1
            if len(acc_win) >= 10:
                step *= 1.1 if np.mean(acc_win) > target_accept else 0.9
                step = float(np.clip(step, 0.1, 2.0))
                acc_win = []
        bias.deposit(state["s"], sig, w0, dT)
        st.update(moves=n_moves, accept_rate=n_acc / max(n_moves, 1), step=step,
                  n_gauss=bias.k, v_cur=bias(state["s"]), p_jump=p_jump, n_stag=n_stag,
                  n_kind=nk, n_acc_kind=na, n_improve_kind=ni)
