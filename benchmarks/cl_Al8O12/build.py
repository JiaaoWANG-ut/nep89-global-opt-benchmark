#!/usr/bin/env python3
"""Build benchmarks/cl_Al8O12 (gas-phase (Al2O3)4 cluster).

    python benchmarks/cl_Al8O12/build.py          # write problem.json
    python benchmarks/cl_Al8O12/build.py --refs   # + NEP89 bulk corundum reference

Cubic box with 2*RADIUS + >= 16 A edge, i.e. >= 12 A of vacuum around anything
generated in the search sphere. --refs relaxes alpha-Al2O3 (cell + positions)
with NEP89 to compare the cluster energy per formula unit with the bulk.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

NAME = "cl_Al8O12"
SYMBOLS = ["Al"] * 8 + ["O"] * 12
RADIUS = 4.5
BOX = 26.0
SPEC = dict(
    name=NAME,
    title="(Al2O3)4 cluster",
    category="cluster",
    description=('Gas-phase (Al2O3)4, an alumina dust-nucleation and gas-phase IR target; DFT '
        'global minima of small (Al2O3)n are compact cages of 4-membered Al2O2 and '
        '6-membered Al3O3 rings with 3-5-fold Al and 2-4-fold O (Rahane, Deshpande & '
        'Kumar, JPCC 115, 18111 (2011)), and for n = 4 the B3LYP GM is a low-symmetry'
        ' (C1) cage while PBE0 prefers a D3d isomer (Gobrecht et al., A&A 658, A167 '
        '(2022)). The NEP89 putative GM has these features (Cs cage; five 3-fold and '
        'three 4-fold Al; ten 2-fold, one 3-fold and one 4-fold O; two Al2O2 and '
        'eight Al3O3 rings; Al-O 1.71-1.98 A; no O-O or Al-Al bonds), but its '
        'identity with the DFT GM was not verified. A Cs isomer 0.20 eV higher (one '
        '2-fold Al) forms a competing funnel that traps some basin-hopping runs.'),
    search_symbols=SYMBOLS,
    cell=[[BOX, 0, 0], [0, BOX, 0], [0, 0, BOX]],
    pbc=[True, True, True],
    region=dict(type="sphere", center=[BOX / 2] * 3, radius=RADIUS),
    connected=True,
    budget=69000,
    fmax=0.05,
    e_tol=0.02,
)


def bulk_reference(fmax=0.005):
    """NEP89 alpha-Al2O3 (corundum): energy per Al2O3 unit, after cell relaxation."""
    import contextlib
    import io

    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    from ase.spacegroup import crystal
    from calorine.calculators import CPUNEP

    from gopt.nep import NEP89
    at = crystal(["Al", "O"], [(0, 0, 0.35216), (0.30624, 0, 0.25)], spacegroup=167,
                 cellpar=[4.759, 4.759, 12.991, 90, 90, 120], primitive_cell=True)
    with contextlib.redirect_stdout(io.StringIO()):
        at.calc = CPUNEP(NEP89)
    BFGS(FrechetCellFilter(at), logfile=None).run(fmax=fmax, steps=1000)
    n_fu = at.get_chemical_symbols().count("Al") // 2
    d = at.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    print(f"alpha-Al2O3 NEP89: {len(at)} atoms, E = {at.get_potential_energy() / n_fu:.4f} eV/Al2O3, "
          f"V = {at.get_volume() / len(at):.2f} A^3/atom, shortest Al-O {d.min():.3f} A")
    return at.get_potential_energy() / n_fu


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(HERE, "problem.json"), "w") as f:
        json.dump(SPEC, f, indent=1)
    print("wrote", os.path.join(HERE, "problem.json"))
    if args.refs:
        bulk_reference()
