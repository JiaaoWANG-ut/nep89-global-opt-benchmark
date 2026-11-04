#!/usr/bin/env python3
"""Build benchmarks/bu_Zn12O12: variable-cell crystal-structure prediction of ZnO (Z=12).

Writes problem.json and relaxes the experimental ground state and competing
polymorphs on the benchmark energy surface (NEP89 + short-range guard, through
gopt.context.Context and its native variable-cell relaxation). The reference
energies go to pilot.json["reference_polymorphs"] (other pilot.json keys are
kept), the relaxed structures to reference_polymorphs.extxyz. Polymorphs whose
cell is not commensurate with the benchmark cell are relaxed in their own cell
(fits_cell = false).

    python benchmarks/bu_Zn12O12/build.py
"""
import json
import os
import sys
import tempfile

import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.data import atomic_numbers, covalent_radii
from ase.io import write
from ase.neighborlist import neighbor_list
from ase.spacegroup import crystal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from gopt.context import Context  # noqa: E402
from gopt.problem import Problem  # noqa: E402

NAME = os.path.basename(HERE)
SYMBOLS = ["Zn"] * 12 + ["O"] * 12
VPA = 13.1  # 1.1 x experimental wurtzite (11.90 A^3/atom)
SPEC = dict(
    name=NAME,
    title="ZnO (Zn12O12) crystal",
    description=(
        "Crystal structure prediction of ZnO with 12 formula units (24 atoms) per cell. The "
        "experimental ground state wurtzite (P6_3mc) is also the NEP89 global minimum; the "
        "nearly degenerate tetrahedral stacking polytypes (4H +5, zinc blende +7 meV/atom) and "
        "rocksalt (+47 meV/atom) are competing minima."),
    category="bulk",
    search_symbols=SYMBOLS,
    cell=(np.eye(3) * round((VPA * len(SYMBOLS)) ** (1 / 3), 3)).tolist(),
    pbc=[True, True, True],
    variable_cell=True,
    volume_per_atom=VPA,
    region={"type": "cell"},
    budget=20000,
    fmax=0.05,
    e_tol=0.02,
)


def polytype(seq, a=3.2495, c=5.2069, u=0.3819):
    """ZnO tetrahedral polytype with stacking sequence seq ('AB' = 2H/wurtzite,
    'ABCB' = 4H, 'ABC' = 3C/zinc blende), built from the wurtzite bilayer."""
    lat = {"A": (0, 0), "B": (1 / 3, 2 / 3), "C": (2 / 3, 1 / 3)}
    n = len(seq)
    sp = [(*lat[s], k / n) for k, s in enumerate(seq)]
    sp += [(*lat[s], (k + 2 * u) / n % 1.0) for k, s in enumerate(seq)]
    return Atoms(["Zn"] * n + ["O"] * n, scaled_positions=sp,
                 cell=cellpar_to_cell([a, a, n * c / 2, 90, 90, 120]), pbc=True)


def polymorphs():
    """Experimental / known structures (any multiple of the formula unit)."""
    return {
        # wurtzite, P6_3mc (Kisi & Elcombe, Acta Cryst. C45, 1867 (1989))
        "wurtzite": crystal(["Zn", "O"], basis=[(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.3819)],
                            spacegroup=186, cellpar=[3.2495, 3.2495, 5.2069, 90, 90, 120]),
        "polytype_4H": polytype("ABCB"),
        # zinc blende, F-43m (a = 4.58 A, metastable thin films)
        "zinc_blende": crystal(["Zn", "O"], basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
                               spacegroup=216, cellpar=[4.58] * 3 + [90] * 3, primitive_cell=True),
        # rocksalt, Fm-3m (high pressure, a = 4.28 A)
        "rocksalt": crystal(["Zn", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                            spacegroup=225, cellpar=[4.28] * 3 + [90] * 3, primitive_cell=True),
    }

# ----------------------------------------------------------------- generic part
def supercell(atoms, n):
    """Most compact repetition of ``atoms`` with n atoms, or None."""
    k, r = divmod(n, len(atoms))
    if r:
        return None
    best = None
    for a in range(1, k + 1):
        for b in range(1, k // a + 1):
            if k % (a * b):
                continue
            c = k // (a * b)
            L = atoms.cell.lengths() * (a, b, c)
            if best is None or L.max() / L.min() < best[0]:
                best = (L.max() / L.min(), (a, b, c))
    return atoms.repeat(best[1])


def problem_for(atoms, tmp):
    """Problem with the benchmark settings for the composition of ``atoms``."""
    order = list(dict.fromkeys(SYMBOLS))
    syms = sorted(atoms.get_chemical_symbols(), key=order.index)
    if syms == SYMBOLS:
        return Problem(NAME, root=os.path.dirname(HERE))
    name = "ref_" + "".join(f"{s}{syms.count(s)}" for s in order)
    os.makedirs(os.path.join(tmp, name), exist_ok=True)
    with open(os.path.join(tmp, name, "problem.json"), "w") as f:
        json.dump(dict(SPEC, name=name, search_symbols=syms), f)
    return Problem(name, root=tmp)


def relax(p, atoms, rattle=0.0, seed=0, fmax=0.005):
    """Variable-cell relaxation on the benchmark PES (restarted until converged)."""
    sy = np.array(atoms.get_chemical_symbols())
    idx, used = [], np.zeros(len(sy), bool)
    for s in p.symbols:  # reorder to the problem's species order
        k = np.where((sy == s) & ~used)[0][0]
        used[k] = True
        idx.append(k)
    pos, cell = atoms.positions[idx].copy(), atoms.cell.array.copy()
    if rattle:
        rng = np.random.default_rng(seed)
        eps = rng.normal(scale=0.5 * rattle, size=(3, 3))
        F = np.eye(3) + 0.5 * (eps + eps.T)
        pos, cell = pos @ F.T + rng.normal(scale=rattle, size=pos.shape), cell @ F.T
    ctx = Context(p, budget=10 ** 9)
    e_old = np.inf
    for _ in range(6):
        r = ctx.relax(pos, cell, fmax=fmax, steps=3000)
        pos, cell = r.pos, r.cell
        if r.converged and abs(e_old - r.e) < 1e-4:
            break
        e_old = r.e
    return r, ctx.n_guard


def describe(atoms):
    """Minimum distance and coordination numbers (bonds to unlike species for compounds,
    cutoff 1.25 x sum of covalent radii, all periodic images counted)."""
    sy = np.array(atoms.get_chemical_symbols())
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))
    cut = {(a, b): 1.25 * (covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]])
           for a in species for b in species if a != b or len(species) == 1}
    i, j = neighbor_list("ij", atoms, cut)
    cn = np.bincount(i, minlength=len(atoms))
    dmin = neighbor_list("d", atoms, 3.0).min()
    return float(dmin), {sp: sorted(set(cn[sy == sp].tolist())) for sp in species}


def main():
    with open(os.path.join(HERE, "problem.json"), "w") as f:
        json.dump(SPEC, f, indent=1)
    n = len(SYMBOLS)
    refs, frames = {}, []
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        for name, at0 in polymorphs().items():
            at = supercell(at0, n)
            fits = at is not None  # else: computed in its own cell, not reachable here
            at = at if fits else at0
            p = problem_for(at, tmp)
            r1, ng = relax(p, at)
            r2, _ = relax(p, at, rattle=0.03, seed=1)  # symmetry-broken start
            r = r1 if r1.e <= r2.e + 1e-3 else r2  # keep the lower minimum
            vol = abs(np.linalg.det(r.cell))
            at_r = p.to_atoms(r.pos, r.cell)
            dmin, coord = describe(at_r)
            refs[name] = dict(e_per_atom=round(r.e / p.n, 5), volume_per_atom=round(vol / p.n, 3),
                              e_cell=round(r.e / p.n * n, 4), n_atoms=p.n, fits_cell=fits,
                              e_per_atom_ideal_start=round(r1.e / p.n, 5),
                              e_per_atom_rattled_start=round(r2.e / p.n, 5), dmin=round(dmin, 3),
                              coordination=coord, converged=bool(r.converged), guard_calls=ng)
            at_r.info.update(name=name, energy=float(r.e), e_per_atom=float(r.e / p.n))
            frames.append(at_r)
            print(f"{name:22s} E/atom {r.e / p.n:9.4f} (ideal {r1.e / p.n:9.4f} rattled "
                  f"{r2.e / p.n:9.4f}) V/atom {vol / p.n:6.2f} dmin {dmin:.3f} CN {coord}")
    e0 = min(v["e_per_atom"] for v in refs.values())
    for v in refs.values():
        v["de_mev_per_atom"] = round(1000 * (v["e_per_atom"] - e0), 1)
    path = os.path.join(HERE, "pilot.json")
    pilot = json.load(open(path)) if os.path.exists(path) else {}
    pilot["reference_polymorphs"] = refs
    with open(path, "w") as f:
        json.dump(pilot, f, indent=1)
    write(os.path.join(HERE, "reference_polymorphs.extxyz"), frames)


if __name__ == "__main__":
    main()
