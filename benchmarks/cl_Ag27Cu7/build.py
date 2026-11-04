#!/usr/bin/env python3
"""Build benchmark cl_Ag27Cu7: gas-phase Ag27Cu7 nanoalloy.

    python benchmarks/cl_Ag27Cu7/build.py [--reference]

Writes problem.json next to this file. The cluster sits in a cubic box with
>= 14 A of vacuum around the generation sphere; the sphere radius gives the
same packing density as cl_Pt13 (1.305 x radius of a bulk-density sphere of the
cluster atoms). --reference relaxes the literature global-minimum motif
(core-shell polyicosahedron) with the benchmark potential (NEP89 + guard) and prints its energy.
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

NAME = "cl_Ag27Cu7"
TITLE = "Ag27Cu7 nanoalloy"
SYMBOLS = ["Ag"] * 27 + ["Cu"] * 7
DESCRIPTION = ("Gas-phase Ag27Cu7 nanoalloy (geometry plus homotop / chemical-ordering search). "
               "With the Gupta potential the GM is a perfect core-shell polyicosahedron (pIh7, D5h) "
               "with a Cu7 pentagonal-bipyramid core under a 27-atom Ag shell (Rossi et al., PRL 93, "
               "105503 (2004)); NEP89 reproduces this structure.")
BUDGET = 100000
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


def pih7(a_core=2.55, a_shell=2.8):
    """Rossi et al. pIh7: Cu7 pentagonal bipyramid, each Cu the centre of an icosahedron.

    Shell (27 Ag): 2 axial caps, 2 x 5 atoms capping the core faces above/below the
    equator, 5 equatorial atoms between ring Cu and 2 x 5 atoms outside each ring Cu.
    """
    rho = a_core / (2 * np.sin(np.radians(36)))
    h = np.sqrt(a_core ** 2 - rho ** 2)
    core = np.array([(0, 0, h), (0, 0, -h)] + [(rho * np.cos(np.radians(72 * k)),
                                                rho * np.sin(np.radians(72 * k)), 0)
                                               for k in range(5)])
    shell = [np.array((0, 0, h + a_shell)), np.array((0, 0, -h - a_shell))]

    def apex(p1, p2, p3, away):
        c = (p1 + p2 + p3) / 3
        n = np.cross(p2 - p1, p3 - p1)
        n /= np.linalg.norm(n)
        return c + np.sign(np.dot(n, c - away)) * n * a_shell * np.sqrt(2 / 3)
    c36 = np.cos(np.radians(36))
    for k in range(5):
        rk, rk1 = core[2 + k], core[2 + (k + 1) % 5]
        shell.append(apex(core[0], rk, rk1, core[1]))
        shell.append(apex(core[1], rk, rk1, core[0]))
        ang = np.radians(36 + 72 * k)
        r = rho * c36 + np.sqrt(a_shell ** 2 - rho ** 2 * (1 - c36 ** 2))
        shell.append(np.array((r * np.cos(ang), r * np.sin(ang), 0)))
        ang = np.radians(72 * k)
        rr = rho + a_shell * np.cos(np.radians(35))
        for z in (1, -1):
            shell.append(np.array((rr * np.cos(ang), rr * np.sin(ang),
                                   z * a_shell * np.sin(np.radians(35)))))
    return Atoms(["Ag"] * 27 + ["Cu"] * 7, positions=np.vstack([np.array(shell), core]))


def references():
    return {"core-shell pIh7 (D5h, literature GM)": pih7()}


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
