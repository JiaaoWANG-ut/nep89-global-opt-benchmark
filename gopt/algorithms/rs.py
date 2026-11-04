"""Random search (AIRSS-style random sampling + local relaxation)."""
LABEL = "Random search"
FAMILY = "baseline"
REFERENCE = "Pickard & Needs, J. Phys.: Condens. Matter 23, 053201 (2011)"
DESCRIPTION = ("Independent random structures (random cell for bulk) inside the search "
               "region with minimum-distance and connectivity constraints, each relaxed "
               "with L-BFGS. No memory between trials.")
DEFAULTS = {}


def run(ctx, rng):
    p = ctx.p
    while True:
        pos, cell = p.random_structure(rng)
        ctx.relax(pos, cell)
