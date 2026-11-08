"""Basin hopping (Monte Carlo minimisation)."""
import numpy as np

from .. import ops

LABEL = "Basin hopping"
FAMILY = "baseline"
REFERENCE = "Wales & Doye, J. Phys. Chem. A 101, 5111 (1997); Li & Scheraga, PNAS 84, 6611 (1987)"
DESCRIPTION = ("Metropolis walk on the relaxed energy surface. Every move displaces all "
               "search atoms uniformly within a sphere of radius `step` (bulk: plus a "
               "random cell strain), relaxes with L-BFGS and accepts with "
               "exp(-dE/kT). Every 10 moves the step is scaled by 1.1 or 0.9 depending on "
               "whether the acceptance ratio of those 10 moves is above 50% (as in GMIN); "
               "detached atoms are re-seated on the cluster surface.")
DEFAULTS = dict(kT=0.1, step=0.5, target_accept=0.5, adapt_every=10, strain=0.05)


def run(ctx, rng, kT=0.1, step=0.5, target_accept=0.5, adapt_every=10, strain=0.05):
    p = ctx.p
    pos, cell = p.random_structure(rng)
    cur = ctx.relax(pos, cell)
    n_acc = n_try = w_acc = 0
    while True:
        new_pos, new_cell = np.array(cur.pos), np.array(cur.cell)
        if p.variable_cell:
            new_pos, new_cell = ops.strain_cell(new_pos, new_cell, rng, strain)
        new_pos = ops.displace(p, new_pos, rng, step, cell=new_cell)
        if not p.variable_cell and p.category != "bulk":
            lost = ops.detached_atoms(p, new_pos, new_cell)
            if len(lost):
                new_pos = ops.relocate_atoms(p, new_pos, rng, lost, new_cell)
        trial = ctx.relax(new_pos, new_cell)
        n_try += 1
        de = trial.e - cur.e
        if not trial.aborted and (de < 0 or rng.random() < np.exp(-de / kT)):
            cur = trial
            n_acc += 1
            w_acc += 1
        if n_try % adapt_every == 0:
            # acceptance ratio of the last adapt_every moves (a cumulative ratio lags and lets the step drift)
            step *= 1.1 if w_acc / adapt_every > target_accept else 0.9
            step = float(np.clip(step, 0.1, 2.0))
            w_acc = 0
        ctx.stats.update(accept_rate=n_acc / n_try, step=step, moves=n_try)
