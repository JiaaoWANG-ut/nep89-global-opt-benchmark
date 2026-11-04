"""Soft-mode guided hopping, version 2 snapshot (campaign dev-softmode-v2).

v2 = model Hessian with anchors/strain modes (``_softmode_hess``), preconditioned
L-BFGS (``_softmode_opt``), adaptive swaps; optional ceiling abort (rejected).
Equivalent to ``softmode`` with p_reloc=0. Run with ``--alg _softmode_v2``.
"""
import numpy as np

from .. import ops
from ._softmode_hess import ModelHessian
from ._softmode_opt import PreconRelaxer

LABEL = "Soft-mode guided hopping (v2 snapshot)"
FAMILY = "innovative"
REFERENCE = "see softmode.py"
DESCRIPTION = "Snapshot of softmode v2 (campaign dev-softmode-v2); see softmode.py."
DEFAULTS = dict(kT=0.1, amp=0.3, amp_min=0.05, amp_max=1.5, target_escape=0.65, adapt_rate=0.15,
                n_modes=6, beta=1.0, noise=0.1, p_swap=0.25, swap_adapt=True, p_swap_min=0.05,
                p_swap_max=0.5, patience=10, patience_per_atom=1.5, precon=True, strain=0.1,
                abort_de=None, abort_f=0.5)


def _seat(p, pos, cell, rng):
    """Re-seat detached search atoms (non-bulk)."""
    if p.variable_cell or p.category == "bulk":
        return pos
    lost = ops.detached_atoms(p, pos, cell)
    if len(lost):
        pos = ops.relocate_atoms(p, pos, rng, lost, cell)
    return pos


def run(ctx, rng, kT=0.1, amp=0.3, amp_min=0.05, amp_max=1.5, target_escape=0.65, adapt_rate=0.15,
        n_modes=6, beta=1.0, noise=0.1, p_swap=0.25, swap_adapt=True, p_swap_min=0.05,
        p_swap_max=0.5, patience=10, patience_per_atom=1.5, precon=True, strain=0.1,
        abort_de=None, abort_f=0.5):
    p = ctx.p
    model = ModelHessian(p)
    relaxer = PreconRelaxer(ctx, model, abort_f=abort_f) if precon else None
    multi = len(p.species) > 1
    scale = 2.0 * float(np.mean(p.rcov[p.search_idx]))  # typical bond length
    a_cur = amp
    ps = p_swap if multi else 0.0
    pat = int(max(patience, round(patience_per_atom * p.n_search)))
    st = ctx.stats
    st.setdefault("restarts", -1)
    for key in ("soft", "soft_esc", "soft_acc", "swap", "swap_acc", "ceil"):
        st.setdefault(key, 0)

    def relax(pos, cell, ceiling=None):
        if relaxer is None:
            return ctx.relax(pos, cell)
        return relaxer.relax(pos, cell, ceiling=ceiling)

    def fresh():
        pos, cell = p.random_structure(rng)
        st["restarts"] += 1
        return relax(pos, cell)

    cur = fresh()
    fp_cur = ops.fingerprint(p, cur.pos, cur.cell)
    wbest = cur.e
    stag = 0
    modes = None  # soft modes cached for the current minimum
    while True:
        if stag >= pat:
            cur = fresh()
            fp_cur = ops.fingerprint(p, cur.pos, cur.cell)
            wbest, stag, modes = cur.e, 0, None
            continue
        pos, cell = np.array(cur.pos), np.array(cur.cell)
        kind = "swap" if (ps > 0 and rng.random() < ps) else "soft"
        if kind == "soft":
            if modes is None:
                modes = model.soft_modes(pos, cell)
            pos, cell = model.jump(pos, cell, modes, rng, a_cur * scale, n_modes, beta, noise,
                                   strain * a_cur / amp)
        else:
            pos = ops.swap_unlike(p, pos, rng, 1)
        pos = _seat(p, pos, cell, rng)
        ceiling = None if abort_de is None else cur.e + abort_de
        trial = relax(pos, cell, ceiling)
        de = trial.e - cur.e
        st[kind] += 1
        if trial.aborted:
            # abandoned far above the current minimum (or unphysical cell): a different basin
            escaped, fp_t = True, None
            st["ceil"] += int(ceiling is not None and trial.e > ceiling)
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
            cur, fp_cur = trial, fp_t
        if kind == "swap" and swap_adapt:
            # swaps pay off for alloys (accepted new arrangements), not for ionic systems
            good = acc and escaped
            ps = float(np.clip(ps * np.exp(0.1 * ((1.0 if good else 0.0) - 0.3)), p_swap_min,
                               p_swap_max))
        if cur.e < wbest - 0.005:
            wbest, stag = cur.e, 0
        else:
            stag += 1
        st["amp"] = round(a_cur, 4)
        st["p_swap"] = round(ps, 4)
