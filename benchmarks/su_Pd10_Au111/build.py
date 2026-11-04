"""Build su_Pd10_Au111: Pd10 on a frozen 4-layer Au(111) slab (7x7 surface cell).

Au lattice constant from a NEP89 cell relaxation (4.1105 A, exp. 4.078 A); top two
layers relaxed with NEP89 (bottom two fixed), then the whole slab is frozen.

    python benchmarks/su_Pd10_Au111/build.py [--json-only]
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

NAME = "su_Pd10_Au111"
TITLE = "Pd10 / Au(111)"
DESCRIPTION = ('Pd10 on a frozen 4-layer Au(111) slab (7x7, 196 Au; NEP89 a(Au) = 4.111 A vs exp. 4.078 A), a model of Pd-Au bimetallic catalysts and PdAu(111) surface alloys (Maroun et al., Science 293, 1811 (2001); Chen et al., Science 310, 291 (2005)). The NEP89 putative GM is a flat Pd10 island with every Pd in an fcc hollow (the same island in hcp hollows is 0.24 eV higher), in line with the 2D close-packed Pd islands seen by STM on Au(111) (Casari et al., Phys. Rev. B 79, 195402 (2009)); the frozen substrate excludes the Pd-Au place exchange (surface alloying) observed at room temperature.')
SEARCH = ["Pd"] * 10
REGION_Z = (1.5, 7.5)
BUDGET = 15000
E_TOL = 0.02


def build():
    from ase.build import bulk, fcc111
    au, _ = relax_cell(bulk("Au", "fcc", a=4.08, cubic=True))
    a = float(au.cell.lengths()[0])
    print(f"  NEP89 Au a = {a:.4f} A (exp. 4.078 A, {100 * (a / 4.078 - 1):+.2f}%)")
    slab = fcc111("Au", size=(7, 7, 4), a=a, vacuum=10.0, orthogonal=False, periodic=True)
    lay = layer_index(slab.positions[:, 2])
    return relax_fixed(slab, lay <= 1)


if __name__ == "__main__":
    main(build)
