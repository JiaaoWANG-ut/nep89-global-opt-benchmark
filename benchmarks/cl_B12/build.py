#!/usr/bin/env python3
"""Build benchmarks/cl_B12 (gas-phase B12 cluster).

    python benchmarks/cl_B12/build.py          # write problem.json
    python benchmarks/cl_B12/build.py --refs   # + relax literature motifs with NEP89+guard

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

NAME = "cl_B12"
SYMBOLS = ["B"] * 12
RADIUS = 4.0
BOX = 24.0
SPEC = dict(
    name=NAME,
    title="B12 cluster",
    category="cluster",
    description=('Gas-phase B12. Photoelectron spectroscopy and ab initio calculations '
        'establish a quasi-planar C3v global minimum (Zhai et al., Nat. Mater. 2, 827'
        ' (2003)), with 3D isomers such as the icosahedron several eV higher. NEP89 '
        'reverses this ordering: its putative GM is the Ih icosahedron (B-B 1.73 A, '
        'the building block of bulk boron), 2.7 eV below the relaxed quasi-planar C3v'
        ' structure, so this benchmark probes search on the NEP89 landscape rather '
        'than boron-cluster chemistry.'),
    search_symbols=SYMBOLS,
    cell=[[BOX, 0, 0], [0, BOX, 0], [0, 0, BOX]],
    pbc=[True, True, True],
    region=dict(type="sphere", center=[BOX / 2] * 3, radius=RADIUS),
    connected=True,
    budget=20000,
    fmax=0.05,
    e_tol=0.02,
)


def references():
    refs = {}
    # quasi-planar C3v: inner triangle + 9-membered outer ring, slightly convex
    inner = [[0.95 * np.cos(t), 0.95 * np.sin(t), 0.3] for t in 2 * np.pi * np.arange(3) / 3]
    outer = [[2.55 * np.cos(t), 2.55 * np.sin(t), -0.1] for t in 2 * np.pi * (np.arange(9) + 0.5) / 9]
    refs["quasi-planar (C3v), DFT/PES GM"] = np.array(inner + outer)
    # icosahedron (alpha-boron unit)
    phi = (1 + 5 ** 0.5) / 2
    ico = []
    for a in (-1, 1):
        for b in (-1, 1):
            ico += [[0, a, b * phi], [a, b * phi, 0], [b * phi, 0, a]]
    refs["icosahedron"] = np.array(ico) * 1.75 / 2
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
