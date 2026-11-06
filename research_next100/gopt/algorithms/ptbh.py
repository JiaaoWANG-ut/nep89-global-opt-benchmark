"""Parallel-tempering basin hopping (replica-exchange basin hopping, BHPT)."""
import numpy as np
from ase.geometry import minkowski_reduce

from .. import ops

LABEL = "Parallel-tempering basin hopping"
FAMILY = "baseline"
REFERENCE = ("Strodel, Lee, Whittleston & Wales, J. Am. Chem. Soc. 132, 13300 (2010) "
             "(basin-hopping parallel tempering, GMIN keyword BHPT); Wales & Doye, "
             "J. Phys. Chem. A 101, 5111 (1997); Swendsen & Wang, Phys. Rev. Lett. 57, "
             "2607 (1986)")
DESCRIPTION = (
    "M = 6 basin-hopping replicas at temperatures on a geometric ladder kT = 0.05 ... "
    "0.5 eV (0.05, 0.08, 0.13, 0.2, 0.32, 0.5 eV; the BH reference value 0.1 eV lies inside "
    "the ladder; the same ladder for every system) share the ionic-step budget and are "
    "advanced round-robin, one BH step each per sweep. Every replica uses the move set of "
    "the `bh` baseline (uniform displacement of all search atoms within radius `step`, "
    "bulk: plus a random cell strain; detached atoms re-seated), an L-BFGS quench and "
    "Metropolis acceptance at its own temperature; the step size belongs to the "
    "temperature slot and adapts every 10 of that slot's moves by x1.1/x0.9 towards 50% "
    "acceptance (clipped to 0.1-2 A), using the acceptance of those last 10 moves as GMIN "
    "does (bh's cumulative ratio stays above 50% at every temperature here, so all slots "
    "would share one step). After each sweep a replica-exchange round is attempted with "
    "probability 0.5 (the exchange probability of Strodel et al.; one round per ~12 BH "
    "steps): neighbouring temperature pairs, alternating even/odd pairs (GMIN option "
    "SETS), swap their current minima with probability "
    "min(1, exp((1/kT_i - 1/kT_j)(E_i - E_j))), so low-energy minima found by hot "
    "replicas migrate to the cold ones. Deviations from the paper: replicas run "
    "interleaved in one process (MPI-parallel in GMIN) on one shared budget; 6 replicas "
    "on a fixed ladder instead of 32 on a hand-chosen per-system range; a replica whose "
    "current minimum came from an aborted variable-cell relaxation restarts from a new "
    "random structure; variable-cell states are moved from their Minkowski-reduced cell "
    "(same crystal and same move; without it the accepted cells drift to flat "
    "representations and after ~30k steps up to 20% of the relaxations hit the thin-cell "
    "abort of ctx.relax).")
DEFAULTS = dict(n_replicas=6, kT_min=0.05, kT_max=0.5, exchange_prob=0.5, step=0.5,
                target_accept=0.5, adapt_every=10, strain=0.05)

_BIG = 1e9  # effective energy of a minimum from an aborted relaxation


def _reduce_cell(p, pos, cell):
    """The same crystal in its Minkowski-reduced cell, atoms wrapped into it."""
    try:
        rcell, _ = minkowski_reduce(cell, p.pbc)
        frac = pos @ np.linalg.inv(rcell)
    except Exception:
        return pos, cell
    frac[:, p.pbc] -= np.floor(frac[:, p.pbc])
    return frac @ rcell, rcell


def _perturb(p, cur, rng, step, strain):
    """bh.py move: (cell strain), uniform displacement, re-seat detached atoms."""
    pos, cell = np.array(cur.pos), np.array(cur.cell)
    if p.variable_cell:
        pos, cell = _reduce_cell(p, pos, cell)
        pos, cell = ops.strain_cell(pos, cell, rng, strain)
    pos = ops.displace(p, pos, rng, step, cell=cell)
    if not p.variable_cell and p.category != "bulk":
        lost = ops.detached_atoms(p, pos, cell)
        if len(lost):
            pos = ops.relocate_atoms(p, pos, rng, lost, cell)
    return pos, cell


def _energy(state):
    return _BIG if state.aborted else float(state.e)


def _start(ctx, rng, states):
    """Relaxed random structure (fallback: strongly perturbed copy of another replica)."""
    p = ctx.p
    try:
        pos, cell = p.random_structure(rng)
    except RuntimeError:
        pool = [s for s in states if not s.aborted] or states
        if not pool:
            raise
        pos, cell = _perturb(p, pool[int(rng.integers(len(pool)))], rng, 1.0, 0.05)
    return ctx.relax(pos, cell)


def run(ctx, rng, n_replicas=6, kT_min=0.05, kT_max=0.5, exchange_prob=0.5, step=0.5,
        target_accept=0.5, adapt_every=10, strain=0.05):
    p = ctx.p
    m = max(int(n_replicas), 1)
    adapt_every = max(int(adapt_every), 1)
    kts = float(kT_min) * (float(kT_max) / float(kT_min)) ** (np.arange(m) / max(m - 1, 1))
    steps = np.full(m, float(step))
    n_try = np.zeros(m, int)
    n_acc = np.zeros(m, int)
    w_acc = np.zeros(m, int)  # acceptances since the slot's last step adjustment
    ex_try = np.zeros(max(m - 1, 1), int)
    ex_acc = np.zeros(max(m - 1, 1), int)
    stats = ctx.stats
    stats.update(kT=[round(float(k), 4) for k in kts], sweeps=0, moves=0)

    states = []
    for _ in range(m):
        states.append(_start(ctx, rng, states))
    energies = [_energy(s) for s in states]

    sweep = 0
    parity = 0
    n_restart = 0
    while True:
        for k in range(m):
            cur = states[k]
            if cur.aborted:  # no usable minimum (variable-cell relaxation aborted)
                states[k] = _start(ctx, rng, states)
                energies[k] = _energy(states[k])
                n_restart += 1
                continue
            trial = ctx.relax(*_perturb(p, cur, rng, steps[k], strain))
            n_try[k] += 1
            if not trial.aborted:
                de = float(trial.e) - energies[k]
                if de < 0 or rng.random() < np.exp(-de / kts[k]):
                    states[k] = trial
                    energies[k] = float(trial.e)
                    n_acc[k] += 1
                    w_acc[k] += 1
            if n_try[k] % adapt_every == 0:
                rate = w_acc[k] / adapt_every
                w_acc[k] = 0
                steps[k] = float(np.clip(steps[k] * (1.1 if rate > target_accept else 0.9),
                                         0.1, 2.0))
        sweep += 1
        if m > 1 and rng.random() < exchange_prob:
            for i in range(parity, m - 1, 2):
                j = i + 1
                delta = (1.0 / kts[i] - 1.0 / kts[j]) * (energies[i] - energies[j])
                ex_try[i] += 1
                if delta >= 0 or rng.random() < np.exp(delta):
                    states[i], states[j] = states[j], states[i]
                    energies[i], energies[j] = energies[j], energies[i]
                    ex_acc[i] += 1
            parity ^= 1
        stats.update(
            sweeps=sweep, moves=int(n_try.sum()),
            accept_rate=[round(float(a / max(t, 1)), 3) for a, t in zip(n_acc, n_try)],
            step=[round(float(s), 3) for s in steps],
            exchange_rate=[round(float(a / max(t, 1)), 3) for a, t in zip(ex_acc, ex_try)],
            exchanges=int(ex_acc.sum()), restarts=n_restart)
