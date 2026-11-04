#!/usr/bin/env python3
"""Build benchmarks/cl_Mg8O8 (gas-phase (MgO)8 cluster).

    python benchmarks/cl_Mg8O8/build.py          # write problem.json
    python benchmarks/cl_Mg8O8/build.py --refs   # + relax literature motifs with NEP89+guard

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

NAME = "cl_Mg8O8"
SYMBOLS = ["Mg"] * 8 + ["O"] * 8
RADIUS = 4.5
BOX = 26.0
SPEC = dict(
    name=NAME,
    title="(MgO)8 cluster",
    category="cluster",
    description=('Gas-phase (MgO)8. IR spectroscopy combined with DFT global optimisation '
        '(Haertelt et al., PCCP 14, 2849 (2012)) assigns cage-like structures '
        'containing hexagonal (MgO)3 rings, not rocksalt cuboids, to (MgO)n with n !='
        ' 3k. NEP89 instead favours rocksalt fragments: its putative GM (a 3x3x2 '
        'rocksalt slab missing one corner MgO pair; Mg 3-5-fold coordinated, Mg-O '
        '1.94-2.31 A, no O-O or Mg-Mg bonds) lies 0.25 eV below the 2x2x4 cuboid and '
        '2.7 eV below a hand-built hexagonal-ring cage.'),
    search_symbols=SYMBOLS,
    cell=[[BOX, 0, 0], [0, BOX, 0], [0, 0, BOX]],
    pbc=[True, True, True],
    region=dict(type="sphere", center=[BOX / 2] * 3, radius=RADIUS),
    connected=True,
    budget=7000,
    fmax=0.05,
    e_tol=0.02,
)


def _ordered(sites):
    """sites: list of (is_Mg, xyz) -> positions in SYMBOLS order (Mg first)."""
    mg = [x for m, x in sites if m]
    o = [x for m, x in sites if not m]
    assert len(mg) == 8 and len(o) == 8
    return np.array(mg + o, float)


def references():
    refs = {}
    d = 2.0
    # 2x2x4 rocksalt cuboid
    refs["2x2x4 rocksalt cuboid"] = _ordered(
        [((i + j + k) % 2 == 0, [i * d, j * d, k * d])
         for i in range(2) for j in range(2) for k in range(4)])
    # octagonal tube: two stacked (MgO)4 rings, alternating species vertically
    r = 1.95 / (2 * np.sin(np.pi / 8))
    refs["octagonal tube (2 x (MgO)4 ring)"] = _ordered(
        [((k + layer) % 2 == 0, [r * np.cos(np.pi * k / 4), r * np.sin(np.pi * k / 4), layer * 2.0])
         for layer in range(2) for k in range(8)])
    # hexagonal tube fragment: (MgO)3 rings x 2 + (MgO)2 capping square on one end
    rh = 1.95
    sites = [((k + layer) % 2 == 0, [rh * np.cos(np.pi * k / 3), rh * np.sin(np.pi * k / 3), layer * 2.0])
             for layer in range(2) for k in range(6)]
    sites += [(True, [0.95, 0.0, 4.0]), (False, [-0.95, 0.0, 4.0]),
              (False, [0.0, 0.95, -2.0]), (True, [0.0, -0.95, -2.0])]
    refs["hexagonal tube + caps"] = _ordered(sites)
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
