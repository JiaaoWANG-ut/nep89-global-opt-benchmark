"""Build su_Au8_MgO100: Au8 on a frozen 3-layer MgO(001) slab.

Reproducible recipe (~20 s on one core):
1. NEP89 (calorine CPUNEP) relaxation of bulk rock-salt MgO.
2. (001) slab of 3 atomic layers in a (6x6) cell of the primitive surface lattice
   (a_s = a/sqrt2, cell edges along <110>, 17.8 A, 216 atoms), 20 A vacuum.
   (NEP89 Au wets MgO: in a (5x5) cell, 14.9 A, the putative minimum of Au8 was a
   flat two-row ribbon bonded to its own periodic image; the 3x3 conventional slab
   of su_Pt8_MgO100 is only 12.6 A wide.)
3. NEP89 relaxation of the top two layers (bottom layer fixed); the whole slab
   is frozen in the benchmark.
4. problem.json: search region = box above the surface covering the whole cell.

    python benchmarks/su_Au8_MgO100/build.py                  # full rebuild
    python benchmarks/su_Au8_MgO100/build.py --problem-only   # rewrite problem.json only
"""
import contextlib
import io
import json
import os
import re
import sys

import numpy as np
from ase import Atoms
from ase.build import bulk
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
A_EXP = 4.212            # MgO rock salt, experiment (A)
NREP = 6                 # (NREP x NREP) primitive surface cells
N_LAYERS = 3
VACUUM = 20.0
REGION_DZ = (1.6, 7.5)   # search box: z in [top + dz0, top + dz1]
BUDGET = 19500           # ~3x the median BH hit step at e_tol 0.015 (pilot-bsu1-v2, 40000 steps)
# Distinct flat isomers lie only 0.019 eV (tight relaxation) above the GM, while runs that
# found the GM itself scatter by up to 0.011 eV (fmax 0.05), so e_tol 0.05 would count other
# isomers as hits; 0.015 separates both groups in the pilot (pilot.json, notes).
E_TOL = 0.015
TITLE = f"{CLUSTER[0]}{len(CLUSTER)} / MgO(001)"
DESCRIPTION = (
    "Au8 on a frozen 3-layer MgO(001) slab (6x6 primitive surface cells, 216 atoms; NEP89 "
    "a = 4.203 A vs exp. 4.212 A), the classic model of size-selected Au_n/MgO catalysts "
    "(Ricci et al., Phys. Rev. Lett. 97, 036106 (2006)). The NEP89 putative GM is a flat Au8 "
    "island with all 8 Au in contact with the surface (Au-Au 2.74-2.83 A); NEP89 puts single Au "
    "atoms on top of O as DFT does, but places the best 3D isomers found (fcc 6+2 bilayers) "
    ">= 0.69 eV above the flat GM, whereas DFT finds 3D Au8 on bulk MgO(001) and planar Au8 only "
    "on metal-supported MgO films that charge the cluster (Ricci et al. 2006), i.e. NEP89 "
    "over-stabilises wetting. e_tol is 0.015 eV because distinct flat isomers lie 0.019 eV above "
    "the GM."
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
    mgo = relax_bulk(bulk("MgO", "rocksalt", a=A_EXP, cubic=True))
    a = mgo.cell.cellpar()[0]
    print(f"NEP89 MgO a = {a:.4f} A (exp {A_EXP}, {100 * (a / A_EXP - 1):+.2f}%)")
    b = a / np.sqrt(2)  # primitive surface lattice constant (Mg-Mg along <110>)
    sym, pos = [], []
    for l in range(N_LAYERS):
        for i in range(NREP):
            for j in range(NREP):
                for k, (dx, dy) in enumerate(((0.0, 0.0), (0.5, 0.5))):
                    sym.append("Mg" if (k + l) % 2 == 0 else "O")
                    pos.append([(i + dx) * b, (j + dy) * b, l * a / 2])
    slab = Atoms(sym, positions=pos, cell=[NREP * b, NREP * b, (N_LAYERS - 1) * a / 2 + VACUUM],
                 pbc=True)
    fixed = slab.positions[:, 2] < 0.1  # bottom layer
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
