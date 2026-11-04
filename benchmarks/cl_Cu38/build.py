#!/usr/bin/env python3
"""Build benchmark cl_Cu38: gas-phase Cu38 cluster.

    python benchmarks/cl_Cu38/build.py [--reference]

Writes problem.json next to this file. The cluster sits in a cubic box with
>= 14 A of vacuum around the generation sphere; the sphere radius gives the
same packing density as cl_Pt13 (1.305 x radius of a bulk-density sphere of the
cluster atoms). --reference relaxes the literature global-minimum motif
(truncated octahedron) with the benchmark potential (NEP89 + guard) and prints its energy.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, reference_states

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

NAME = "cl_Cu38"
TITLE = "Cu38 cluster"
SYMBOLS = ["Cu"] * 38
DESCRIPTION = ("Gas-phase Cu38. For many-body potentials (Gupta, Sutton-Chen) the GM is the Oh fcc "
               "truncated octahedron, separated from icosahedral/decahedral structures as in the "
               "LJ38 double funnel (Doye & Wales, New J. Chem. 22, 733 (1998); Cambridge Cluster "
               "Database). NEP89 agrees; the best non-fcc structure found lies 0.11 eV higher.")
BUDGET = 150000
E_TOL = 0.02


def sizes(symbols):
    vol = sum(reference_states[atomic_numbers[s]]["a"] ** 3 / 4 for s in symbols)  # fcc bulk
    r_bulk = (3 * vol / (4 * math.pi)) ** (1 / 3)
    radius = round(1.305 * r_bulk, 1)
    return radius, float(math.ceil(2 * radius + 14))


def spec():
    radius, box = sizes(SYMBOLS)
    return dict(name=NAME, title=TITLE, category="cluster", description=DESCRIPTION,
                search_symbols=SYMBOLS, cell=[[box, 0, 0], [0, box, 0], [0, 0, box]],
                pbc=[True, True, True],
                region=dict(type="sphere", center=[box / 2] * 3, radius=radius),
                connected=True, budget=BUDGET, fmax=0.05, e_tol=E_TOL)


def references():
    from ase.cluster import Octahedron
    return {"truncated octahedron (Oh)": Octahedron("Cu", 4, 1, latticeconstant=3.61)}


def relax_reference(atoms):
    from gopt.context import Context
    from gopt.problem import Problem, min_image_distances
    p = Problem(NAME)
    ctx = Context(p, budget=10 ** 6)
    pos = atoms.positions - atoms.positions.mean(axis=0) + np.diag(p.cell0) / 2
    r = ctx.relax(pos, fmax=0.005, steps=5000)
    d = min_image_distances(r.pos, r.pos, p.cell0, p.pbc)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    coord = (d < 1.2 * np.median(nn)).sum(axis=1)
    return r, d.min(), np.bincount(coord).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true")
    a = ap.parse_args()
    s = spec()
    with open(os.path.join(HERE, "problem.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(f"{NAME}: box {s['cell'][0][0]} A, sphere radius {s['region']['radius']} A, "
          f"budget {s['budget']}")
    if a.reference:
        for label, at in references().items():
            r, dmin, coord = relax_reference(at)
            print(f"  {label}: E = {r.e:.4f} eV, converged={r.converged}, "
                  f"min dist {dmin:.3f} A, coordination histogram {coord}")


if __name__ == "__main__":
    main()
