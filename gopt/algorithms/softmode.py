"""Soft-mode guided hopping (SMH).

Basin hopping in which the trial moves follow the soft collective modes of the
current minimum instead of random rattles of every atom.

* Model Hessian (zero ionic steps, ``_softmode_hess``): exponential pair springs
  k_ij = exp(-3 (r_ij / r0_ij - 1)), r0_ij = rcov_i + rcov_j (Lindh/Packwood model
  Hessian, anisotropic-network-model springs); frozen substrate/framework atoms act
  as anchors, bulk systems get the six cell-strain components (Schur complement)
  so soft elastic/shear directions are included. Rigid-body modes are projected
  out and the softest modes are obtained by diagonalisation (~1-2 ms).
* Soft-mode jump: a random combination of the 6 softest modes weighted like
  harmonic thermal amplitudes (lambda^-1/2) plus 10 % noise, scaled to a maximum
  atomic displacement amp * 2<rcov>. ``amp`` adapts online so that a target
  fraction (65 %) of jumps escapes into a new basin; bulk jumps add a soft strain.
* Relaxation (``_softmode_opt``): L-BFGS preconditioned by the same model Hessian
  (same convergence test as ``ctx.relax``; ~38 % fewer steps per relaxation on
  clusters).
* Multi-element systems also use unlike-atom swaps whose probability tunes itself
  towards a 30 % rate of accepted escapes (high for alloys, low for ionic systems).
* Metropolis acceptance at kT; a walker that has not improved by 5 meV for
  max(10, 1.5 n_search) moves restarts from a fresh random structure.

Tested and rejected during development (snapshots ``_softmode_v1/v2/v3``):
energy-ranked relocation of weakly bound atoms (v3), ceiling aborts and loose
trial relaxations, finite-difference/Davidson refinement of the model modes.
"""
import numpy as np

from .. import ops
from ._softmode_hess import ModelHessian
from ._softmode_opt import PreconRelaxer

LABEL = "Soft-mode guided hopping"
FAMILY = "innovative"
REFERENCE = ("Lyakhov, Oganov & Valle, Comput. Phys. Commun. 181, 1623 (2010) [softmutation]; "
             "Schoenborn, Goedecker, Roy & Oganov, J. Chem. Phys. 130, 144108 (2009) "
             "[minima-hopping softening]; Henkelman & Jonsson, J. Chem. Phys. 111, 7010 (1999) "
             "[dimer curvature]; Lindh et al., Chem. Phys. Lett. 241, 423 (1995) [model Hessian]; "
             "Packwood et al., J. Chem. Phys. 144, 164109 (2016) [preconditioned optimisation]; "
             "Atilgan et al., Biophys. J. 80, 505 (2001) [anisotropic network model]; "
             "Wales & Doye, J. Phys. Chem. A 101, 5111 (1997) [basin hopping]")
DESCRIPTION = (
    "Basin hopping whose moves follow the soft collective modes of the current minimum "
    "instead of random rattles. A free (zero-step) bond-stiffness model Hessian is built "
    "from covalent radii (exponential pair springs, Lindh/ANM style; frozen substrate atoms "
    "act as anchors; bulk adds the six strain components so soft elastic/shear directions "
    "are included); rigid-body modes are projected out and the softest modes are found by "
    "diagonalisation. A jump is a random combination of the six softest modes with "
    "harmonic weights ~lambda^-1/2 plus small noise, scaled to a maximum atomic "
    "displacement that adapts online to keep 65% of jumps escaping into new basins. The "
    "same model Hessian preconditions the L-BFGS relaxations (about a third fewer steps "
    "on clusters). Alloys add unlike-atom swaps with a self-tuned probability; Metropolis "
    "acceptance and restarts from random structures after stagnation.")
DEFAULTS = dict(kT=0.1, amp=0.3, amp_min=0.05, amp_max=1.5, target_escape=0.65, adapt_rate=0.15,
                n_modes=6, beta=1.0, noise=0.1, strain=0.1, p_swap=0.25, swap_adapt=True,
                p_swap_min=0.05, p_swap_max=0.5, patience=10, patience_per_atom=1.5)


def _seat(p, pos, cell, rng):
    """Re-seat detached search atoms (non-bulk)."""
    if p.variable_cell or p.category == "bulk":
        return pos
    lost = ops.detached_atoms(p, pos, cell)
    if len(lost):
        pos = ops.relocate_atoms(p, pos, rng, lost, cell)
    return pos


def _update(prob, good, lo, hi, target=0.3, rate=0.1):
    """Multiplicative self-tuning of a move probability towards `target` successes."""
    return float(np.clip(prob * np.exp(rate * ((1.0 if good else 0.0) - target)), lo, hi))


def run(ctx, rng, kT=0.1, amp=0.3, amp_min=0.05, amp_max=1.5, target_escape=0.65, adapt_rate=0.15,
        n_modes=6, beta=1.0, noise=0.1, strain=0.1, p_swap=0.25, swap_adapt=True,
        p_swap_min=0.05, p_swap_max=0.5, patience=10, patience_per_atom=1.5):
    p = ctx.p
    model = ModelHessian(p)
    relaxer = PreconRelaxer(ctx, model)
    scale = 2.0 * float(np.mean(p.rcov[p.search_idx]))  # typical bond length
    a_cur = amp
    ps = p_swap if len(p.species) > 1 else 0.0
    pat = int(max(patience, round(patience_per_atom * p.n_search)))
    st = ctx.stats
    st.setdefault("restarts", -1)
    for key in ("soft", "soft_esc", "soft_acc", "swap", "swap_acc"):
        st.setdefault(key, 0)

    def fresh():
        pos, cell = p.random_structure(rng)
        st["restarts"] += 1
        return relaxer.relax(pos, cell)

    cur = fresh()
    fp_cur = ops.fingerprint(p, cur.pos, cur.cell)
    wbest = cur.e
    stag = 0
    modes = None  # soft modes of the current minimum (cached until it changes)
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
        trial = relaxer.relax(pos, cell)
        de = trial.e - cur.e
        st[kind] += 1
        if trial.aborted:
            escaped, fp_t = True, None  # unphysical cell: a different basin, never accepted
        else:
            fp_t = ops.fingerprint(p, trial.pos, trial.cell)
            escaped = not ops.same_structure(p, cur.e, fp_cur, trial.e, fp_t, de=0.02)
        if kind == "soft":
            # escape-rate control of the jump amplitude
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
            ps = _update(ps, acc and escaped, p_swap_min, p_swap_max)
        if cur.e < wbest - 0.005:
            wbest, stag = cur.e, 0
        else:
            stag += 1
        st["amp"] = round(a_cur, 4)
        st["p_swap"] = round(ps, 4)
