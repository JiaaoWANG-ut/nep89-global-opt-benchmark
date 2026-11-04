"""Build su_Pt10_TiO2a101: Pt10 on a frozen stoichiometric anatase TiO2(101) slab.

Reproducible recipe (~40 s on one core):
1. NEP89 (calorine CPUNEP) relaxation of bulk anatase TiO2 (cell + positions).
2. (101) slab of 3 stoichiometric TiO2 layers cut at the widest interlayer gap
   (standard saw-tooth termination with 2-fold O / 5-fold Ti rows along [010]).
   Oblique, nearly square surface supercell spanned by 4 x [010] (15.3 A) and
   (3/2)[-101] - (1/2)[010] (15.2 A, gamma ~ 97 deg; both cell heights >= 15 A),
   i.e. 12 primitive (101) cells, 216 atoms, 20 A vacuum.
3. NEP89 relaxation of the top two layers (bottom layer fixed); the whole slab is
   frozen in the benchmark.
4. problem.json: search region = box above the surface covering the whole cell.

    python benchmarks/su_Pt10_TiO2a101/build.py                  # full rebuild
    python benchmarks/su_Pt10_TiO2a101/build.py --problem-only   # rewrite problem.json only
"""
import contextlib
import io
import json
import os
import re
import sys

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.neighborlist import neighbor_list
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
A_EXP, C_EXP, U_EXP = 3.785, 9.514, 0.2081   # anatase, experiment
N_LAYERS = 3
VACUUM = 20.0
REGION_DZ = (1.4, 7.5)   # search box: z in [top + dz0, top + dz1] (top = 2-fold O)
BUDGET = 21500           # ~3x the median BH hit step of the pilot (pilot-bsu1-v2, 40000 steps)
E_TOL = 0.02
TITLE = f"{CLUSTER[0]}{len(CLUSTER)} / anatase TiO2(101)"
DESCRIPTION = (
    "Pt10 on a frozen 3-layer stoichiometric anatase TiO2(101) slab (oblique 15.2 x 15.3 A cell, "
    "216 atoms; NEP89 a = 3.819 A, c = 9.296 A vs exp. 3.785, 9.514 A), the majority facet of "
    "anatase-supported Pt photo- and thermal catalysts. The NEP89 putative GM is a 3D Pt5+5 "
    "bilayer anchored on the terrace by 5-fold Ti (Pt-Ti 2.49-2.65 A) and 2-fold O (Pt-O 2.09 A), "
    "consistent with the strong Pt binding on anatase(101) terraces found by DFT and STM (Gong et "
    "al., J. Am. Chem. Soc. 130, 370 (2008)). Caveats: NEP89 Pt-Ti contacts are short (2.76 A in "
    "Pt3Ti), NEP89 is charge-neutral and its anatase c axis is 2.3% too short."
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


def inplane_supercell(slab, S1, S2, reps=(4, 8)):
    """Re-express a periodic slab in the in-plane cell (S1, S2) (Cartesian xy vectors
    that are lattice vectors of the slab) and rotate S1 onto x."""
    big = slab.repeat((reps[0], reps[1], 1))
    cell = np.array([[S1[0], S1[1], 0.0], [S2[0], S2[1], 0.0], slab.cell[2]])
    frac = np.linalg.solve(cell.T, big.positions.T).T
    frac[:, :2] -= np.floor(frac[:, :2])
    key = np.c_[np.round(frac[:, :2] * 1e4).astype(int) % 10000,
                np.round(big.positions[:, 2] * 1e3).astype(int)]
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    pos = frac[first] @ cell
    th = -np.arctan2(S1[1], S1[0])
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    out = Atoms(np.array(big.get_chemical_symbols())[first], positions=pos @ rot.T,
                cell=cell @ rot.T, pbc=True)
    if np.linalg.det(out.cell.array) < 0:
        raise ValueError("left-handed cell")
    return out


def build_substrate():
    ana = relax_bulk(crystal(["Ti", "O"], basis=[(0, 0, 0), (0, 0, U_EXP)], spacegroup=141,
                             cellpar=[A_EXP, A_EXP, C_EXP, 90, 90, 90]))
    a, _, c = ana.cell.cellpar()[:3]
    print(f"NEP89 anatase a = {a:.4f} A (exp {A_EXP}, {100 * (a / A_EXP - 1):+.2f}%), "
          f"c = {c:.4f} A (exp {C_EXP}, {100 * (c / C_EXP - 1):+.2f}%)")
    thick = surface(ana, (1, 0, 1), N_LAYERS + 2, vacuum=10.0)
    e1, e2 = thick.cell[0, :2], thick.cell[1, :2]  # [-101]-type (10.0 A) and [010] (3.8 A)
    assert abs(np.linalg.norm(e2) - a) < 1e-3, thick.cell
    # cut between layers at the widest interlayer gap (O-O, ~1.2 A)
    z = thick.positions[:, 2]
    zs = np.sort(z)
    gaps = np.diff(zs)
    cuts = 0.5 * (zs[1:] + zs[:-1])[gaps > 0.9 * gaps.max()]
    slab = thick[(z > cuts[0]) & (z < cuts[N_LAYERS])]
    assert slab.get_chemical_formula() == f"O{8 * N_LAYERS}Ti{4 * N_LAYERS}", slab.get_chemical_formula()
    # primitive 2D lattice: p1 = [010], p2 = (e1 + e2)/2 (body centring)
    p1, p2 = e2, 0.5 * (e1 + e2)
    S1, S2 = 3 * p2 - 2 * p1, 4 * p1
    slab = inplane_supercell(slab, S1, S2)
    assert len(slab) == 12 * 6 * N_LAYERS, len(slab)
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    cell = slab.cell.array.copy()
    cell[2] = [0.0, 0.0, slab.positions[:, 2].max() + VACUUM]
    slab.set_cell(cell, scale_atoms=False)
    fixed = slab.positions[:, 2] < 2.6  # bottom layer
    relax_slab(slab, fixed)
    slab = slab[np.argsort(slab.positions[:, 2], kind="stable")]
    # report surface coordination (Ti-O < 2.4 A) of the top layer
    i, j = neighbor_list("ij", slab, 2.4)
    sym = np.array(slab.get_chemical_symbols())
    m = sym[i] != sym[j]
    cn = np.bincount(i[m], minlength=len(slab))
    top = slab.positions[:, 2] > slab.positions[:, 2].max() - 2.0
    for s in ("Ti", "O"):
        sel = top & (sym == s)
        print(f"  top-layer {s}: coordination {np.bincount(cn[sel]).tolist()}")
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
