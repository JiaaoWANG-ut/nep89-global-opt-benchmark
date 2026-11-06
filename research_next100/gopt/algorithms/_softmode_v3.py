"""Soft-mode guided hopping, version 3 snapshot (v2 + energy-ranked relocation moves).

Campaign dev-softmode-v3 ran this behaviour (as alg softmode, bit-identical). Rejected:
relocation helped su_Pt8_MgO100 only slightly (ERT 1868 vs 2037) and hurt ze_Pt4_CHA
(227 vs 121) and cl_Pt13 (248 vs 192). Run with ``--alg _softmode_v3``.
"""
import numpy as np

from .. import ops
from ._softmode_hess import ModelHessian
from ._softmode_opt import PreconRelaxer

LABEL = "Soft-mode guided hopping (v3 dev)"
FAMILY = "innovative"
REFERENCE = "see softmode.py"
DESCRIPTION = "Development snapshot of softmode: v2 plus adaptive energy-ranked relocation moves."
DEFAULTS = dict(kT=0.1, amp=0.3, amp_min=0.05, amp_max=1.5, target_escape=0.65, adapt_rate=0.15,
                n_modes=6, beta=1.0, noise=0.1, p_swap=0.25, swap_adapt=True, p_swap_min=0.05,
                p_swap_max=0.5, p_reloc=0.25, p_reloc_min=0.05, p_reloc_max=0.5, reloc_rank=0.6,
                patience=10, patience_per_atom=1.5, strain=0.1)


def _seat(p, pos, cell, rng):
    """Re-seat detached search atoms (non-bulk)."""
    if p.variable_cell or p.category == "bulk":
        return pos
    lost = ops.detached_atoms(p, pos, cell)
    if len(lost):
        pos = ops.relocate_atoms(p, pos, rng, lost, cell)
    return pos


def _pick_weak(p, ea, rng, rank_q):
    """Search atoms (global indices) to relocate, ranked by species-relative site energy."""
    s = p.search_idx
    sym = np.array(p.symbols)[s]
    z = np.array(ea, float)
    for x in set(sym):
        m = sym == x
        z[m] -= np.median(z[m])
    kmax = max(1, len(s) // 4)
    k = int(min(rng.geometric(0.6), kmax))
    order = np.argsort(-z)
    w = rank_q ** np.arange(len(s))
    w /= w.sum()
    pick = rng.choice(len(s), size=k, replace=False, p=w)
    return s[order[pick]]


def _update(prob, good, lo, hi, target=0.3, rate=0.1):
    return float(np.clip(prob * np.exp(rate * ((1.0 if good else 0.0) - target)), lo, hi))


def run(ctx, rng, kT=0.1, amp=0.3, amp_min=0.05, amp_max=1.5, target_escape=0.65, adapt_rate=0.15,
        n_modes=6, beta=1.0, noise=0.1, p_swap=0.25, swap_adapt=True, p_swap_min=0.05,
        p_swap_max=0.5, p_reloc=0.25, p_reloc_min=0.05, p_reloc_max=0.5, reloc_rank=0.6,
        patience=10, patience_per_atom=1.5, strain=0.1):
    p = ctx.p
    model = ModelHessian(p)
    relaxer = PreconRelaxer(ctx, model)
    multi = len(p.species) > 1
    scale = 2.0 * float(np.mean(p.rcov[p.search_idx]))  # typical bond length
    a_cur = amp
    ps = p_swap if multi else 0.0
    pr = 0.0 if (p.variable_cell or p.category == "bulk" or p.n_search < 3) else p_reloc
    pat = int(max(patience, round(patience_per_atom * p.n_search)))
    st = ctx.stats
    st.setdefault("restarts", -1)
    for key in ("soft", "soft_esc", "soft_acc", "swap", "swap_acc", "reloc", "reloc_acc"):
        st.setdefault(key, 0)

    def site_e():
        ea = ctx.atomic_energies()
        return None if ea is None else ea[p.search_idx]

    def fresh():
        pos, cell = p.random_structure(rng)
        st["restarts"] += 1
        return relaxer.relax(pos, cell)

    cur = fresh()
    ea_cur = site_e()
    fp_cur = ops.fingerprint(p, cur.pos, cur.cell)
    wbest = cur.e
    stag = 0
    modes = None  # soft modes cached for the current minimum
    while True:
        if stag >= pat:
            cur = fresh()
            ea_cur = site_e()
            fp_cur = ops.fingerprint(p, cur.pos, cur.cell)
            wbest, stag, modes = cur.e, 0, None
            continue
        pos, cell = np.array(cur.pos), np.array(cur.cell)
        kind = "swap" if (ps > 0 and rng.random() < ps) else "soft"
        if kind == "soft" and pr > 0 and ea_cur is not None and rng.random() < pr:
            kind = "reloc"
        if kind == "soft":
            if modes is None:
                modes = model.soft_modes(pos, cell)
            pos, cell = model.jump(pos, cell, modes, rng, a_cur * scale, n_modes, beta, noise,
                                   strain * a_cur / amp)
        elif kind == "swap":
            pos = ops.swap_unlike(p, pos, rng, 1)
        else:
            pos = ops.relocate_atoms(p, pos, rng, _pick_weak(p, ea_cur, rng, reloc_rank), cell)
        pos = _seat(p, pos, cell, rng)
        trial = relaxer.relax(pos, cell)
        ea_t = site_e()
        de = trial.e - cur.e
        st[kind] += 1
        if trial.aborted:
            escaped, fp_t = True, None  # unphysical cell: a different basin
        else:
            fp_t = ops.fingerprint(p, trial.pos, trial.cell)
            escaped = not ops.same_structure(p, cur.e, fp_cur, trial.e, fp_t, de=0.02)
        if kind == "soft":
            st["soft_esc"] += int(escaped)
            a_cur *= np.exp(adapt_rate * ((0.0 if escaped else 1.0) - (1.0 - target_escape)))
            a_cur = float(np.clip(a_cur, amp_min, amp_max))
        acc = (not trial.aborted) and (de < 0 or rng.random() < np.exp(-de / kT))
        if acc:
            st[kind + "_acc"] += 1
            if escaped:
                modes = None
            cur, fp_cur, ea_cur = trial, fp_t, ea_t
        if kind == "swap" and swap_adapt:
            # swaps pay off for alloys (accepted new arrangements), not for ionic systems
            ps = _update(ps, acc and escaped, p_swap_min, p_swap_max)
        elif kind == "reloc":
            pr = _update(pr, acc and escaped, p_reloc_min, p_reloc_max)
        if cur.e < wbest - 0.005:
            wbest, stag = cur.e, 0
        else:
            stag += 1
        st["amp"] = round(a_cur, 4)
        st["p_swap"] = round(ps, 4)
        st["p_reloc"] = round(pr, 4)
