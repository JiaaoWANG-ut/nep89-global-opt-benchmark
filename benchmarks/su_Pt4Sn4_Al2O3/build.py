"""Build su_Pt4Sn4_Al2O3: Pt4Sn4 on a frozen Al-terminated alpha-Al2O3(0001) slab
((2sqrt3 x 2sqrt3)R30 surface cell, 16.5 A, 4 O3 layers = 12 atomic layers, Al96O144).

Corundum cell from a NEP89 relaxation (a = 4.753 A, c = 12.895 A; exp. 4.759 / 12.991 A).
Stoichiometric single-Al termination on both sides (Al-O3-Al-Al-O3-Al-...); the top
six atomic layers are relaxed with NEP89 (bottom six fixed), then everything is frozen.

    python benchmarks/su_Pt4Sn4_Al2O3/build.py [--json-only]
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

NAME = "su_Pt4Sn4_Al2O3"
TITLE = "Pt4Sn4 / alpha-Al2O3(0001)"
DESCRIPTION = ('Pt4Sn4 on a frozen Al-terminated alpha-Al2O3(0001) slab ((2sqrt3 x 2sqrt3)R30, Al96O144; NEP89 a/c = 4.753/12.895 A vs exp. 4.759/12.991 A), a model of Pt-Sn/Al2O3 propane-dehydrogenation catalysts (sub-nanometre Pt and PtSn clusters on alumina: Baxter et al., ACS Catal. 7, 3322 (2017); Liu et al., ACS Catal. 13, 2802 (2023)). In the NEP89 putative GM the oxophilic Sn atoms bind surface O (Sn-O 2.37-2.55 A) and carry the Pt atoms on top (only Pt-Sn and Pt-Pt bonds, no Sn-Sn); note that NEP89 relaxes the surface Al 0.4 A below the top O plane (DFT: about 0.1 A above).')
SEARCH = ["Pt"] * 4 + ["Sn"] * 4
REGION_Z = (1.5, 7.5)
BUDGET = 15500
E_TOL = 0.02


def build():
    from ase.build import make_supercell
    from ase.geometry import cellpar_to_cell
    from ase.spacegroup import crystal
    bulk = crystal(["Al", "O"], basis=[(0, 0, 0.3523), (0.3064, 0, 0.25)], spacegroup=167,
                   cellpar=[4.759, 4.759, 12.991, 90, 90, 120])
    bulk, _ = relax_cell(bulk)
    a, c = bulk.cell.cellpar()[[0, 2]]
    print(f"  NEP89 alpha-Al2O3 a = {a:.4f} A ({100 * (a / 4.759 - 1):+.2f}%), "
          f"c = {c:.4f} A ({100 * (c / 12.991 - 1):+.2f}%) (exp. 4.759 / 12.991 A)")
    b2 = bulk.repeat((1, 1, 2))
    lay = layer_index(b2.positions[:, 2], tol=0.2)
    sym = np.array(b2.get_chemical_symbols())
    kinds = ["O" if sym[lay == k][0] == "O" else "Al" for k in range(lay.max() + 1)]
    # start at an Al layer that sits on an Al layer and is covered by an O3 layer
    start = next(k for k in range(1, len(kinds) - 12)
                 if kinds[k] == "Al" and kinds[k - 1] == "Al" and kinds[k + 1] == "O")
    keep = (lay >= start) & (lay < start + 12)
    one = b2[keep]
    assert one.get_chemical_formula() == "Al8O12", one.get_chemical_formula()
    slab = make_supercell(one, [[4, 2, 0], [-2, 2, 0], [0, 0, 1]])  # (2sqrt3 x 2sqrt3)R30
    slab.set_cell(cellpar_to_cell(slab.cell.cellpar()), scale_atoms=True)
    slab.wrap()
    lay = layer_index(slab.positions[:, 2], tol=0.2)
    assert lay.max() == 11
    return relax_fixed(slab, lay <= 5)


if __name__ == "__main__":
    main(build)
