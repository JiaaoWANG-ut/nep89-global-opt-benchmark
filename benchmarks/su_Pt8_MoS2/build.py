"""Build su_Pt8_MoS2: Pt8 on a frozen 2H(1H)-MoS2 monolayer (6x6 cell, 108 atoms).

In-plane lattice constant and S-S thickness from a NEP89 relaxation of the 3-atom
cell (a = 3.150 A, exp. 3.160 A for 2H-MoS2).

    python benchmarks/su_Pt8_MoS2/build.py [--json-only]
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from gopt.context import lbfgs  # noqa: E402
from gopt.nep import NEP89, FastNEP  # noqa: E402

VACUUM = 20.0  # A between the top substrate atom and the bottom of the next slab image


def nep_calc():
    from calorine.calculators import CPUNEP
    with contextlib.redirect_stdout(io.StringIO()):
        return CPUNEP(NEP89)


def relax_cell(atoms, mask=None, fmax=0.002):
    """NEP89 relaxation of positions + cell (mask: Voigt components allowed to relax)."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    atoms.calc = nep_calc()
    BFGS(FrechetCellFilter(atoms, mask=mask), logfile=None).run(fmax=fmax, steps=2000)
    e = atoms.get_potential_energy()
    atoms.calc = None
    return atoms, e


def relax_fixed(atoms, fixed, fmax=0.01, steps=5000):
    """NEP89 L-BFGS relaxation at fixed cell; atoms with fixed=True do not move."""
    fn = FastNEP(atoms.get_chemical_symbols(), atoms.cell.array, atoms.positions)

    def fun(x):
        e, f, v, ea = fn.compute(x)
        f[fixed] = 0.0
        return e, f
    x, e, f, conv, n, ab = lbfgs(fun, atoms.positions.copy(), fmax, steps)
    atoms.positions = x
    fm = float(np.sqrt((f ** 2).sum(axis=1).max()))
    print(f"  slab relaxation: converged={conv} steps={n} E={e:.4f} eV fmax={fm:.4f}")
    return atoms


def layer_index(z, tol=0.3):
    """Atomic-layer index (0 = bottom) from z coordinates."""
    order = np.argsort(z)
    idx = np.zeros(len(z), int)
    k, last = 0, z[order[0]]
    for i in order:
        if z[i] - last > tol:
            k += 1
        idx[i] = k
        last = z[i]
    return idx


def finish_slab(slab):
    """Bottom at z=0, VACUUM above the top, sorted by z, no calculator."""
    slab.calc = None
    slab.info = {}
    slab.pbc = True
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    slab.cell[2] = [0.0, 0.0, slab.positions[:, 2].max() + VACUUM]
    slab.wrap()
    return slab[np.argsort(slab.positions[:, 2], kind="stable")]


def write_problem(spec_extra):
    """problem.json from the substrate file (region: box above the top layer)."""
    sub = read(os.path.join(HERE, "substrate.extxyz"))
    c = sub.cell.array
    top = float(sub.positions[:, 2].max())
    spec = dict(name=NAME, title=TITLE, category="surface", description=DESCRIPTION,
                search_symbols=SEARCH, substrate_file="substrate.extxyz", pbc=[True] * 3,
                # a1 is along x, so [0,|a1|] x [0,a2_y] is a fundamental domain of the 2D lattice
                region=dict(type="box", lo=[0.0, 0.0, top + REGION_Z[0]],
                            hi=[float(c[0, 0]), float(c[1, 1]), top + REGION_Z[1]]),
                connected=True, budget=BUDGET, fmax=0.05, e_tol=E_TOL)
    with open(os.path.join(HERE, "problem.json"), "w") as fh:
        json.dump(spec, fh, indent=1)
    lat = np.linalg.norm(c[:2, :2], axis=1)
    print(f"{NAME}: {len(sub)} substrate atoms + {len(SEARCH)} search atoms, cell |a1|,|a2| = "
          f"{lat.round(3).tolist()} A, gamma = {sub.cell.cellpar()[5]:.1f}, c = {c[2, 2]:.2f} A, "
          f"top z = {top:.3f}, vacuum = {c[2, 2] - (top - sub.positions[:, 2].min()):.2f} A")


def main(build):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-only", action="store_true",
                    help="only rewrite problem.json from the existing substrate.extxyz")
    a = ap.parse_args()
    if not a.json_only:
        slab = finish_slab(build())
        write(os.path.join(HERE, "substrate.extxyz"), slab)
    write_problem({})

NAME = "su_Pt8_MoS2"
TITLE = "Pt8 / MoS2 monolayer"
DESCRIPTION = ('Pt8 on a frozen 1H-MoS2 monolayer (6x6, 108 atoms; NEP89 a = 3.150 A vs exp. 3.160 A), a model of Pt/MoS2 catalysts; DFT finds that Pt_n clusters on the MoS2 basal plane become 3D from n = 5 and that Pt particles bind to S (Saidi, Cryst. Growth Des. 15, 642 (2015)). The NEP89 putative GM is a 3D 5+3 Pt bilayer whose five lower Pt sit above Mo atoms, each bonded to three S (Pt-S 2.31-2.35 A, Pt-Pt 2.58-2.60 A); the lowest competing isomer, a 4+4 bilayer, is 0.16 eV higher.')
SEARCH = ["Pt"] * 8
REGION_Z = (1.5, 7.5)
BUDGET = 27500
E_TOL = 0.02


def build():
    from ase.build import mx2
    m = mx2("MoS2", kind="2H", a=3.16, thickness=3.17, vacuum=10.0)
    m.pbc = True
    m, _ = relax_cell(m, mask=[1, 1, 0, 0, 0, 1])
    a = float(m.cell.lengths()[0])
    z = m.positions[:, 2]
    print(f"  NEP89 MoS2 a = {a:.4f} A (exp. 3.160 A, {100 * (a / 3.160 - 1):+.2f}%), "
          f"S-S thickness {z.max() - z.min():.3f} A (exp. 3.17 A)")
    slab = m.repeat((6, 6, 1))
    return relax_fixed(slab, np.zeros(len(slab), bool))


if __name__ == "__main__":
    main(build)
