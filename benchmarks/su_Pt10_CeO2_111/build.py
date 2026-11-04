"""Build su_Pt10_CeO2_111: Pt10 on a frozen O-terminated CeO2(111) slab.

Reproducible recipe (~30 s on one core):
1. NEP89 (calorine CPUNEP) relaxation of bulk fluorite CeO2 (cell + positions).
2. (111) slab with 3 stoichiometric O-Ce-O trilayers (O-terminated on both sides),
   (5x5) hexagonal surface cell (a_s = a/sqrt2), 225 atoms, 20 A vacuum.
3. NEP89 relaxation of the top two trilayers (bottom trilayer fixed); the whole
   slab is frozen in the benchmark.
4. problem.json: search region = box above the surface covering the whole cell.

    python benchmarks/su_Pt10_CeO2_111/build.py                  # full rebuild
    python benchmarks/su_Pt10_CeO2_111/build.py --problem-only   # rewrite problem.json only
"""
import contextlib
import io
import json
import os
import re
import sys

import numpy as np
from ase.build import bulk, surface
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from calorine.calculators import CPUNEP  # noqa: E402

from gopt.context import lbfgs  # noqa: E402
from gopt.nep import NEP89, FastNEP  # noqa: E402

NAME = os.path.basename(HERE)
_el, _n = re.match(r"([A-Z][a-z]?)(\d+)$", NAME.split("_")[1]).groups()
CLUSTER = [_el] * int(_n)  # cluster from the directory name, e.g. su_Pt10_CeO2_111
A_EXP = 5.411            # CeO2 fluorite, experiment (A)
NREP = 5                 # (NREP x NREP) surface cell
N_TRILAYERS = 3
VACUUM = 20.0
REGION_DZ = (1.5, 8.0)   # search box: z in [top + dz0, top + dz1]
BUDGET = 20000           # ~3x the median BH hit step of the pilot (pilot-bsu1, 40000 steps)
E_TOL = 0.02
TITLE = f"{CLUSTER[0]}{len(CLUSTER)} / CeO2(111)"
DESCRIPTION = (
    "Pt10 on a frozen O-terminated CeO2(111) slab (3 O-Ce-O trilayers, 5x5 cell, 225 atoms; "
    "NEP89 a = 5.438 A vs exp. 5.411 A), a model of Pt/ceria oxidation and three-way catalysts. "
    "The NEP89 putative GM is a 3D Pt7+3 bilayer bound to surface O (Pt-O 1.99 A), and NEP89 "
    "Pt8 is planar, in line with the DFT 2D-to-3D transition above Pt8 for Pt_N on CeO2(111) "
    "(Paz-Borbon et al., Phys. Chem. Chem. Phys. 19, 17845 (2017)). NEP89 is charge-neutral: "
    "the Pt-to-ceria electron transfer and Ce4+ -> Ce3+ reduction found by DFT are not described, "
    "and the frozen slab excludes O spillover and vacancy formation."
)


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
    ceo2 = relax_bulk(bulk("CeO2", "fluorite", a=A_EXP))
    a = ceo2.cell.cellpar()[0] * np.sqrt(2)  # primitive fcc edge -> cubic a
    print(f"NEP89 CeO2 a = {a:.4f} A (exp {A_EXP}, {100 * (a / A_EXP - 1):+.2f}%)")
    # thick slab, then keep complete O-Ce-O trilayers (each O joins its nearest Ce plane)
    thick = surface(ceo2, (1, 1, 1), N_TRILAYERS + 2, vacuum=10.0)
    z = thick.positions[:, 2]
    sym = np.array(thick.get_chemical_symbols())
    ce_planes = plane_heights(z[sym == "Ce"])
    owner = np.argmin(np.abs(z[:, None] - ce_planes[None, :]), axis=1)
    slab = thick[(owner >= 1) & (owner <= N_TRILAYERS)]
    assert slab.get_chemical_formula() == f"Ce{N_TRILAYERS}O{2 * N_TRILAYERS}"
    slab = slab.repeat((NREP, NREP, 1))
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    cell = slab.cell.array.copy()
    cell[2] = [0.0, 0.0, slab.positions[:, 2].max() + VACUUM]
    slab.set_cell(cell, scale_atoms=False)
    slab.pbc = True
    ce_z = slab.positions[np.array(slab.get_chemical_symbols()) == "Ce", 2]
    fixed = slab.positions[:, 2] < ce_z.min() + 1.2  # bottom trilayer
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
