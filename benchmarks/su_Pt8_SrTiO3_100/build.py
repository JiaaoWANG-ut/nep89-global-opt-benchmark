"""Build su_Pt8_SrTiO3_100: Pt8 on a frozen TiO2-terminated SrTiO3(001) slab.

Reproducible recipe (~20 s on one core):
1. NEP89 (calorine CPUNEP) relaxation of cubic perovskite SrTiO3.
2. Stoichiometric (001) slab of 4 atomic layers SrO/TiO2/SrO/TiO2 (TiO2 on top; SrO
   and TiO2 planes are charge-neutral, so the slab is non-polar), (4x4) surface
   cell (15.6 A), 160 atoms, 20 A vacuum.
3. NEP89 relaxation of the top two layers (bottom SrO/TiO2 fixed); the whole slab
   is frozen in the benchmark.
4. problem.json: search region = box above the TiO2 termination covering the cell.

    python benchmarks/su_Pt8_SrTiO3_100/build.py                  # full rebuild
    python benchmarks/su_Pt8_SrTiO3_100/build.py --problem-only   # rewrite problem.json only
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
A_EXP = 3.905            # SrTiO3 cubic perovskite, experiment (A)
NREP = 4                 # (NREP x NREP) surface cell
N_CELLS = 2              # SrO/TiO2 bilayers
VACUUM = 20.0
REGION_DZ = (1.5, 7.5)   # search box: z in [top + dz0, top + dz1]
BUDGET = 13000           # ~3x the median BH hit step of the pilot (pilot-bsu1, 40000 steps)
E_TOL = 0.02
TITLE = f"{CLUSTER[0]}{len(CLUSTER)} / SrTiO3(001)"
DESCRIPTION = (
    "Pt8 on a frozen TiO2-terminated SrTiO3(001) slab (SrO/TiO2/SrO/TiO2, 4x4 cell, 160 atoms; "
    "NEP89 a = 3.908 A vs exp. 3.905 A), a model of Pt/SrTiO3 photocatalysts and "
    "perovskite-supported Pt. The NEP89 putative GM is a single buckled Pt8 layer (6 Pt on the "
    "surface, Pt-O 2.19 A, Pt-Ti 2.61 A; 2 Pt 1.7 A higher), in line with DFT, which finds the "
    "TiO2 termination favourable for epitaxial Pt(100) films (Asthagiri and Sholl, J. Chem. Phys. "
    "116, 9914 (2002)). NEP89 is charge-neutral and the frozen, unreconstructed slab excludes the "
    "surface reconstructions and O vacancies of real SrTiO3(001)."
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


def build_substrate():
    sto = relax_bulk(crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                             spacegroup=221, cellpar=[A_EXP] * 3 + [90] * 3))
    a = sto.cell.cellpar()[0]
    print(f"NEP89 SrTiO3 a = {a:.4f} A (exp {A_EXP}, {100 * (a / A_EXP - 1):+.2f}%)")
    slab = surface(sto, (0, 0, 1), N_CELLS, vacuum=10.0)  # SrO / TiO2 / ... / TiO2
    z = slab.positions[:, 2]
    top_plane = np.array(slab.get_chemical_symbols())[np.abs(z - z.max()) < 0.1]
    assert sorted(set(top_plane)) == ["O", "Ti"], top_plane
    slab = slab.repeat((NREP, NREP, 1))
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    cell = slab.cell.array.copy()
    cell[2] = [0.0, 0.0, slab.positions[:, 2].max() + VACUUM]
    slab.set_cell(cell, scale_atoms=False)
    slab.pbc = True
    fixed = slab.positions[:, 2] < a * (N_CELLS - 1) - 0.1  # bottom N_CELLS-1 bilayers
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
