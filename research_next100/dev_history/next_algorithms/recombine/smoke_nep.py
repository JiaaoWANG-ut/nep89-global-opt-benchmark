"""Tiny true-NEP integration check; no runner registration or result files."""
import numpy as np

from gopt.context import BudgetExhausted, Context
from gopt.problem import Problem
from gopt.algorithms.relink_v2 import run


def smoke(name, budget, relax_steps):
    p = Problem(name)
    p.max_relax_steps = relax_steps
    ctx = Context(p, budget=budget)
    try:
        run(ctx, np.random.default_rng(42), rotations=1)
    except BudgetExhausted:
        pass
    assert ctx.nsteps == ctx.budget
    assert np.isfinite(ctx.best_e)
    assert ctx.n_bad == 0
    assert ctx.stats["path_relink"]["images"] > 0
    if p.n_sub:
        assert ctx.stats["path_relink"]["host"] > 0
    assert ctx.n_frozen_fix == 0
    print(name, "NEP smoke:", ctx.nsteps, "steps, best =", ctx.best_e,
          "stats =", ctx.stats["path_relink"])


def main():
    smoke("cl_Pt13", 90, 8)
    smoke("bu_Si8", 45, 4)
    smoke("su_Pt8_MgO100", 45, 4)
    smoke("ze_Pt4_CHA", 45, 4)


if __name__ == "__main__":
    main()
