#!/usr/bin/env python3
"""Build benchmarks/cl_Ti6O12 (gas-phase (TiO2)6 cluster).

    python benchmarks/cl_Ti6O12/build.py          # write problem.json
    python benchmarks/cl_Ti6O12/build.py --refs   # + relax reference motifs with NEP89+guard

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

NAME = "cl_Ti6O12"
SYMBOLS = ["Ti"] * 6 + ["O"] * 12
RADIUS = 4.5
BOX = 26.0
SPEC = dict(
    name=NAME,
    title="(TiO2)6 cluster",
    category="cluster",
    description=('Gas-phase (TiO2)6, a standard titania-nanocluster global-optimisation '
        'target; DFT global minima of small (TiO2)n are compact, with 4-/5-fold '
        'coordinated Ti and one or two terminal Ti=O groups (Qu & Kroes, JPCB 110, '
        '8998 (2006); Lamiel-Garcia et al., Nanoscale 9, 1049 (2017)). The NEP89 '
        'putative GM has these features (two titanyl O at 1.69 A, four 4- and two '
        '5-fold Ti, nine 4-membered Ti2O2 rings, no O-O bonds), but its identity with'
        ' the DFT GM was not verified; the default repulsive guard (Ti-O r0 = 1.695 '
        'A) is marginally active on the titanyl bonds (0.3 meV).'),
    search_symbols=SYMBOLS,
    cell=[[BOX, 0, 0], [0, BOX, 0], [0, 0, BOX]],
    pbc=[True, True, True],
    region=dict(type="sphere", center=[BOX / 2] * 3, radius=RADIUS),
    connected=True,
    budget=47500,
    fmax=0.05,
    e_tol=0.02,
)


def references():
    refs = {}
    # three (TiO2)2 C2h-like rhombi (Ti2O2 ring + 2 terminal O) stacked into a
    # rough chain: only a sanity check of Ti=O / Ti-O bond lengths
    ti, o = [], []
    for k in range(3):
        z = 3.2 * k
        ti += [[1.40, 0.0, z], [-1.40, 0.0, z]]
        o += [[0.0, 1.30, z], [0.0, -1.30, z], [3.05, 0.0, z + 0.5], [-3.05, 0.0, z - 0.5]]
    refs["3 x (TiO2)2 rhombus stack"] = np.array(ti + o)
    return refs


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
