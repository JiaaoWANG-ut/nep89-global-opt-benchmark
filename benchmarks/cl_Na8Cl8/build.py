#!/usr/bin/env python3
"""Build benchmarks/cl_Na8Cl8 (gas-phase (NaCl)8 cluster).

    python benchmarks/cl_Na8Cl8/build.py          # write problem.json
    python benchmarks/cl_Na8Cl8/build.py --refs   # + relax literature motifs with NEP89+guard

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

NAME = "cl_Na8Cl8"
SYMBOLS = ["Na"] * 8 + ["Cl"] * 8
RADIUS = 5.5
BOX = 30.0
SPEC = dict(
    name=NAME,
    title="(NaCl)8 cluster",
    category="cluster",
    description=('Gas-phase (NaCl)8. Small alkali-halide clusters adopt rocksalt-cuboid '
        'structures in ab initio and Tosi-Fumi studies (Aguado et al., PRB 56, 15353 '
        '(1997); Doye & Wales, PRB 59, 2292 (1999)), and the NEP89 putative GM is the'
        ' 2x2x4 rocksalt cuboid (Na-Cl 2.56-2.65 A). The landscape is soft: fmax = '
        '0.05 relaxations of the same GM differ by up to 0.016 eV (0.3 meV after '
        'tight relaxation), hence e_tol = 0.05 eV.'),
    search_symbols=SYMBOLS,
    cell=[[BOX, 0, 0], [0, BOX, 0], [0, 0, BOX]],
    pbc=[True, True, True],
    region=dict(type="sphere", center=[BOX / 2] * 3, radius=RADIUS),
    connected=True,
    budget=84000,
    fmax=0.05,
    e_tol=0.05,
)


def _ordered(sites):
    a = [x for m, x in sites if m]
    b = [x for m, x in sites if not m]
    assert len(a) == 8 and len(b) == 8
    return np.array(a + b, float)


def references():
    refs = {}
    d = 2.65
    refs["2x2x4 rocksalt cuboid"] = _ordered(
        [((i + j + k) % 2 == 0, [i * d, j * d, k * d])
         for i in range(2) for j in range(2) for k in range(4)])
    r = 2.6 / (2 * np.sin(np.pi / 8))
    refs["octagonal tube (2 x (NaCl)4 ring)"] = _ordered(
        [((k + layer) % 2 == 0, [r * np.cos(np.pi * k / 4), r * np.sin(np.pi * k / 4), layer * 2.7])
         for layer in range(2) for k in range(8)])
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
