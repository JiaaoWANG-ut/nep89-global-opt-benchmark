#!/usr/bin/env python3
"""Build benchmark cl_Pt19: gas-phase Pt19 cluster.

    python benchmarks/cl_Pt19/build.py [--reference]

Writes problem.json next to this file. The cluster sits in a cubic box with
>= 14 A of vacuum around the generation sphere; the sphere radius gives the
same packing density as cl_Pt13 (1.305 x radius of a bulk-density sphere of the
cluster atoms). --reference relaxes the DFT global minimum (capped three-layer
trigonal prism) and compact motifs (double icosahedron, fcc octahedron) with the
benchmark potential (NEP89 + guard) and prints their energies; the NEP89
putative GM (a hollow deltahedral cage) is lower than all of them.
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

NAME = "cl_Pt19"
TITLE = "Pt19 cluster"
SYMBOLS = ["Pt"] * 19
DESCRIPTION = ("Gas-phase Pt19. PBE and hybrid DFT give a Pt18 three-layer trigonal prism capped on "
               "a triangular face (C3v; Kumar & Kawazoe, PRB 77, 205418 (2008); J. Mol. Model. (2024), "
               "doi:10.1007/s00894-024-06099-5). NEP89 instead prefers a hollow 19-vertex deltahedral "
               "cage (all atoms 5- or 6-coordinated, Pt-Pt 2.53-2.57 A), 1.5 eV below the relaxed DFT "
               "structure and 0.66 eV below the fcc octahedron, so this landscape is NEP89-specific.")
BUDGET = 10000
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


def double_icosahedron(symbol, r):
    """Two interpenetrating 13-atom icosahedra (D5h, 19 atoms), unrelaxed."""
    s5 = np.sqrt(5.0)
    pts = [(0, 0, 0), (0, 0, r), (0, 0, -r), (0, 0, 2 * r)]
    for k in range(5):
        a, b = np.radians(72 * k), np.radians(72 * k + 36)
        pts.append((2 * r / s5 * np.cos(b), 2 * r / s5 * np.sin(b), -r / s5))
        pts.append((np.sqrt(3) / 2 * r * np.cos(a), np.sqrt(3) / 2 * r * np.sin(a), r / 2))
        pts.append((2 * r / s5 * np.cos(b), 2 * r / s5 * np.sin(b), r + r / s5))
    return Atoms(symbol * len(pts), positions=pts)


def capped_prism(symbol, d=2.6, h=2.55):
    """DFT GM (isomer 19.1): three eclipsed 6-atom triangular layers (Pt18, D3h) plus one atom
    capping the centre of a triangular face (C3v), unrelaxed."""
    a1, a2 = np.array([d, 0, 0]), np.array([d / 2, d * np.sqrt(3) / 2, 0])
    pts = [i * a1 + j * a2 + [0, 0, k * h] for k in range(3) for i in range(3) for j in range(3)
           if i + j <= 2]
    top = np.array(pts[-6:])
    pts.append(top.mean(axis=0) + [0, 0, d * np.sqrt(2 / 3)])
    return Atoms(symbol * len(pts), positions=pts)


def references():
    from ase.cluster import Octahedron
    return {"capped trigonal prism (C3v, DFT GM)": capped_prism("Pt"),
            "double icosahedron (D5h)": double_icosahedron("Pt", 2.65),
            "fcc octahedron (Oh)": Octahedron("Pt", 3, 0, latticeconstant=3.92)}


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
