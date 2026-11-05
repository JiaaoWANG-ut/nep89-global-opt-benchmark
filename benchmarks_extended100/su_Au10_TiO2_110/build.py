"""Build su_Au10_TiO2_110: Au10 on a frozen stoichiometric rutile TiO2(110) slab.

Reproducible recipe (~60 s on one core):
1. NEP89 (calorine CPUNEP) relaxation of bulk rutile TiO2 (cell + positions).
2. (110) slab of 3 stoichiometric O-Ti2O2-O trilayers with bridging-O rows along
   [001]; (3x6) surface cell = 3 x [1-10] (19.5 A) by 6 x [001] (17.7 A),
   324 atoms, 20 A vacuum. (In a (3x5) cell, 14.8 A along the rows, Au8 and Au10
   formed chains/upright ribbons along [001] bonded to their own periodic image.)
3. NEP89 relaxation of the top two trilayers (bottom trilayer fixed); the whole
   slab is frozen in the benchmark.
4. problem.json: search region = box above the bridging-O rows covering the cell.

    python benchmarks/su_Au10_TiO2_110/build.py                  # full rebuild
    python benchmarks/su_Au10_TiO2_110/build.py --problem-only   # rewrite problem.json only
"""
import contextlib
import io
import json
import os
import re
import sys

import numpy as np
from ase.build import surface
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS
from ase.spacegroup import crystal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from calorine.calculators import CPUNEP  # noqa: E402

from gopt.context import lbfgs  # noqa: E402
from gopt.nep import NEP89, FastNEP  # noqa: E402

NAME = os.path.basename(HERE)
_el, _n = re.match(r"([A-Z][a-z]?)(\d+)$", NAME.split("_")[1]).groups()
CLUSTER = [_el] * int(_n)  # cluster from the directory name, e.g. su_Pt10_CeO2_111
A_EXP, C_EXP, U_EXP = 4.594, 2.959, 0.3048   # rutile, experiment
REP = (3, 6)             # repeats along [1-10] (6.5 A) and [001] (2.96 A)
N_TRILAYERS = 3
VACUUM = 20.0
REGION_DZ = (1.4, 8.0)   # search box: z in [top + dz0, top + dz1] (top = bridging O)
BUDGET = 30000
E_TOL = 0.02
TITLE = f"{CLUSTER[0]}{len(CLUSTER)} / rutile TiO2(110)"
DESCRIPTION = f"{CLUSTER[0]}{len(CLUSTER)} cluster on a frozen 3-trilayer stoichiometric rutile TiO2(110) slab."


def calc():
    with contextlib.redirect_stdout(io.StringIO()):
        return CPUNEP(NEP89)


def relax_bulk(atoms, fmax=0.005):
    atoms.calc = calc()
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=1000)
    atoms.calc = None
    return atoms


def relax_slab(slab, fixed, fmax=0.02, steps=3000):
    fn = FastNEP(slab.get_chemical_symbols(), slab.cell.array, slab.positions)

    def fun(x):
        e, f, v, ea = fn.compute(x)
        f[fixed] = 0.0
        return e, f
    x, e, f, conv, n, ab = lbfgs(fun, slab.positions.copy(), fmax, steps)
    slab.positions = x
    print(f"  slab relaxation: converged={conv} steps={n} E={e:.4f} eV")
    return slab


def plane_heights(z, tol=0.3):
    zs = np.sort(z)
    planes = [[zs[0]]]
    for v in zs[1:]:
        if v - planes[-1][-1] < tol:
            planes[-1].append(v)
        else:
            planes.append([v])
    return np.array([np.mean(p) for p in planes])


def build_substrate():
    rut = relax_bulk(crystal(["Ti", "O"], basis=[(0, 0, 0), (U_EXP, U_EXP, 0)], spacegroup=136,
                             cellpar=[A_EXP, A_EXP, C_EXP, 90, 90, 90]))
    a, _, c = rut.cell.cellpar()[:3]
    print(f"NEP89 rutile a = {a:.4f} A (exp {A_EXP}, {100 * (a / A_EXP - 1):+.2f}%), "
          f"c = {c:.4f} A (exp {C_EXP}, {100 * (c / C_EXP - 1):+.2f}%)")
    # thick slab, keep complete O-Ti2O2-O trilayers (each O joins its nearest Ti plane)
    thick = surface(rut, (1, 1, 0), N_TRILAYERS + 2, vacuum=10.0)
    z = thick.positions[:, 2]
    sym = np.array(thick.get_chemical_symbols())
    ti_planes = plane_heights(z[sym == "Ti"])
    owner = np.argmin(np.abs(z[:, None] - ti_planes[None, :]), axis=1)
    slab = thick[(owner >= 1) & (owner <= N_TRILAYERS)]
    assert slab.get_chemical_formula() == f"O{4 * N_TRILAYERS}Ti{2 * N_TRILAYERS}"
    slab = slab.repeat((REP[0], REP[1], 1))
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    cell = slab.cell.array.copy()
    cell[2] = [0.0, 0.0, slab.positions[:, 2].max() + VACUUM]
    slab.set_cell(cell, scale_atoms=False)
    slab.pbc = True
    ti_z = slab.positions[np.array(slab.get_chemical_symbols()) == "Ti", 2]
    fixed = slab.positions[:, 2] < ti_z.min() + 1.6  # bottom trilayer
    relax_slab(slab, fixed)
    slab = slab[np.argsort(slab.positions[:, 2], kind="stable")]
    return slab


def write_problem(slab):
    c = slab.cell.array
    top = float(slab.positions[:, 2].max())
    width = float(np.linalg.norm(c[0, :2]))
    height = float(abs(np.linalg.det(c[:2, :2])) / width)
    spec = dict(
        name=NAME, title=TITLE, category="surface", description=DESCRIPTION,
        search_symbols=CLUSTER, substrate_file="substrate.extxyz", pbc=[True] * 3,
        region=dict(type="box", lo=[0.0, 0.0, round(top + REGION_DZ[0], 4)],
                    hi=[round(width, 4), round(height, 4), round(top + REGION_DZ[1], 4)]),
        connected=True, budget=BUDGET, fmax=0.05, e_tol=E_TOL)
    with open(os.path.join(HERE, "problem.json"), "w") as f:
        json.dump(spec, f, indent=1)
    print(f"{NAME}: {len(slab)} substrate + {len(CLUSTER)} search atoms, "
          f"cellpar {np.round(slab.cell.cellpar(), 3).tolist()}, top z {top:.3f}")


if __name__ == "__main__":
    path = os.path.join(HERE, "substrate.extxyz")
    if "--problem-only" in sys.argv:
        slab = read(path)
    else:
        slab = build_substrate()
        write(path, slab)
    write_problem(slab)
