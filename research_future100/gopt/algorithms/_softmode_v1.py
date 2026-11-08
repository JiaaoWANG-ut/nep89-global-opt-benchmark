"""Soft-mode guided hopping, version 1 snapshot (campaign dev-softmode).

v1 model Hessian (``_softmode_model``), plain ctx.relax, fixed swap probability.
Run with ``--alg _softmode_v1``; the final algorithm is ``softmode``.
"""
import numpy as np

from .. import ops
from ._softmode_model import ModelHessian

LABEL = "Soft-mode guided hopping (v1 snapshot)"
FAMILY = "innovative"
REFERENCE = "see softmode.py"
DESCRIPTION = "Snapshot of softmode v1 (campaign dev-softmode); see softmode.py."
DEFAULTS = dict(kT=0.1, amp=0.35, amp_min=0.05, amp_max=1.5, target_escape=0.75, adapt_rate=0.15,
                n_modes=6, beta=1.0, noise=0.1, p_swap=0.25, patience=10, patience_per_atom=1.5)


def _seat(p, pos, cell, rng):
    """Re-seat detached search atoms (non-bulk)."""
    if p.variable_cell or p.category == "bulk":
        return pos
    lost = ops.detached_atoms(p, pos, cell)
    if len(lost):
        pos = ops.relocate_atoms(p, pos, rng, lost, cell)
    return pos


def run(ctx, rng, kT=0.1, amp=0.35, amp_min=0.05, amp_max=1.5, target_escape=0.75, adapt_rate=0.15,
        n_modes=6, beta=1.0, noise=0.1, p_swap=0.25, patience=10, patience_per_atom=1.5):
    p = ctx.p
    model = ModelHessian(p)
    multi = len(p.species) > 1
    scale = 2.0 * float(np.mean(p.rcov[p.search_idx]))  # typical bond length
    a_cur = amp
    pat = int(max(patience, round(patience_per_atom * p.n_search)))
    st = ctx.stats
    st.setdefault("restarts", -1)
    for key in ("soft", "soft_esc", "soft_acc", "swap", "swap_acc"):
        st.setdefault(key, 0)

    def fresh():
        pos, cell = p.random_structure(rng)
        st["restarts"] += 1
        return ctx.relax(pos, cell)

    cur = fresh()
    fp_cur = ops.fingerprint(p, cur.pos, cur.cell)
    wbest = cur.e
    stag = 0
    modes = None  # (lam, V, L) cached for the current minimum
    while True:
        if stag >= pat:
            cur = fresh()
            fp_cur = ops.fingerprint(p, cur.pos, cur.cell)
            wbest, stag, modes = cur.e, 0, None
            continue
        pos, cell = np.array(cur.pos), np.array(cur.cell)
        kind = "swap" if (multi and rng.random() < p_swap) else "soft"
        if kind == "soft":
            if modes is None:
                modes = model.soft_modes(pos, cell)
            pos, cell = model.jump(pos, cell, modes, rng, a_cur * scale, n_modes, beta, noise)
        else:
            pos = ops.swap_unlike(p, pos, rng, 1)
        pos = _seat(p, pos, cell, rng)
        trial = ctx.relax(pos, cell)
        de = trial.e - cur.e
        fp_t = ops.fingerprint(p, trial.pos, trial.cell)
        escaped = not ops.same_structure(p, cur.e, fp_cur, trial.e, fp_t, de=0.02)
        st[kind] += 1
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
        if cur.e < wbest - 0.005:
            wbest, stag = cur.e, 0
        else:
            stag += 1
        st["amp"] = round(a_cur, 4)
