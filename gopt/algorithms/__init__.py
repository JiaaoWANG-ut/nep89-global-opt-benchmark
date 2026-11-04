"""Algorithm registry.

Every algorithm is a module ``gopt/algorithms/<key>.py`` defining

    LABEL = "Basin hopping"            # display name
    FAMILY = "baseline" | "innovative"
    REFERENCE = "Wales & Doye, JPCA 101, 5111 (1997)"
    DESCRIPTION = "one-paragraph description"
    DEFAULTS = {...}                   # default keyword parameters
    def run(ctx, rng, **params): ...   # search until ctx.evaluate raises BudgetExhausted

``run`` may return early; the runner then calls it again (a restart) until the
budget is spent. Parameters must be the same for every benchmark (no
per-benchmark tuning); they may depend on generic problem properties
(category, number of search atoms, species, variable_cell).
"""
import importlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def available():
    return sorted(f[:-3] for f in os.listdir(HERE)
                  if f.endswith(".py") and not f.startswith("_"))


def get(key):
    return importlib.import_module(f"gopt.algorithms.{key}")
