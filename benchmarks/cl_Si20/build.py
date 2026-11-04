#!/usr/bin/env python3
"""Build benchmarks/cl_Si20 (gas-phase Si20 cluster).

    python benchmarks/cl_Si20/build.py          # write problem.json
    python benchmarks/cl_Si20/build.py --refs   # + relax literature motifs with NEP89+guard

Cubic box with 2*RADIUS + >= 16 A edge, i.e. >= 12 A of vacuum around anything
generated in the search sphere. Reference motifs are rough hand-built
geometries that are relaxed (fmax 0.01 eV/A) to compare with the search.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

NAME = "cl_Si20"
SYMBOLS = ["Si"] * 20
RADIUS = 5.0
BOX = 28.0
SPEC = dict(
    name=NAME,
    title="Si20 cluster",
    category="cluster",
    description=('Gas-phase Si20. Ion-mobility measurements and DFT show prolate Si_n clusters'
        ' built from stacked tricapped-trigonal-prism units up to n ~ 25, with '
        'compact isomers close in energy around n = 20 (Ho et al., Nature 392, 582 '
        '(1998)). NEP89 has two near-degenerate putative minima, a compact oblate GM '
        'and a prolate isomer only 2 meV higher (tight relaxation), both with '
        '4-7-fold coordinated Si and Si-Si >= 2.41 A, so a hit (E <= E_best + e_tol) '
        'can be either isomer.'),
    search_symbols=SYMBOLS,
    cell=[[BOX, 0, 0], [0, BOX, 0], [0, 0, BOX]],
    pbc=[True, True, True],
    region=dict(type="sphere", center=[BOX / 2] * 3, radius=RADIUS),
    connected=True,
    budget=34000,
    fmax=0.05,
    e_tol=0.02,
)


def references():
    return {}


def relax_refs():
    from gopt.context import Context
    from gopt.problem import Problem, min_image_distances
    p = Problem(NAME)
    for label, xyz in references().items():
        ctx = Context(p, budget=20000)
        pos = np.array(xyz, float) - np.mean(xyz, axis=0) + np.diag(p.cell0) / 2
        r = ctx.relax(pos, fmax=0.01, steps=5000)
        d = min_image_distances(r.pos, r.pos, p.cell0, p.pbc)
        np.fill_diagonal(d, np.inf)
        cn = (d < 1.25 * (p.rcov[:, None] + p.rcov[None, :])).sum(1)
        eg = ctx.guard(r.pos, p.cell0)[0] if ctx.guard is not None else 0.0
        print(f"{label:40s} E={r.e:.4f} conv={r.converged} steps={r.nsteps} "
              f"dmin={d.min():.3f} CN={np.bincount(cn).tolist()} guard={eg:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(HERE, "problem.json"), "w") as f:
        json.dump(SPEC, f, indent=1)
    print("wrote", os.path.join(HERE, "problem.json"))
    if args.refs:
        relax_refs()
