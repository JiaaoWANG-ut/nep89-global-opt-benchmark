"""Build the bulk benchmark bu_Mg4Cu8 (MgCu2 Laves phase, 4 formula units per cell).

    python benchmarks/bu_Mg4Cu8/build.py

Writes problem.json, references.extxyz (relaxed reference structures) and the
"reference_polymorphs"/"decomposition" entries of pilot.json (other pilot.json
fields, i.e. the pilot statistics, are kept). Reference structures are relaxed
(positions + cell, fmax 0.005 eV/A) with the benchmark energy (NEP89 + the
short-range guard) and relaxer through gopt.Problem/gopt.Context.
"""
import json
import os
import shutil
import sys

import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal
from ase.io import write

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from gopt.context import Context  # noqa: E402
from gopt.problem import Problem, min_image_distances  # noqa: E402

NAME = os.path.basename(HERE)
SYMBOLS = ["Mg"] * 4 + ["Cu"] * 8
VPA = 16.0  # ~1.1 x experimental 14.54 A^3/atom (C15, a = 7.04 A)
BUDGET = 60000
E_TOL = 0.02
SPEC = dict(
    name=NAME, title="MgCu2 (Mg4Cu8) Laves-phase crystal", category="bulk",
    description=("Variable-cell crystal-structure prediction of the Laves phase MgCu2 with four "
                 "formula units (12 atoms) per cell: a size-mismatched, topologically "
                 "close-packed intermetallic. Experimental ground state: cubic C15 "
                 "(Fd-3m); the hexagonal C14 (MgZn2-type) polytype and AlB2/MoSi2-type "
                 "AB2 structures compete. NEP89 reproduces C15 as the global minimum, but C14 "
                 "is only 0.5 meV/atom higher (within e_tol, so it also counts as a hit)."),
    search_symbols=SYMBOLS)


# ----------------------------------------------------------------- helpers
def relax(atoms, fmax=0.005, rattle=0.01, seed=0):
    """NEP89+guard variable-cell relaxation through a throw-away Problem/Context."""
    at = atoms.copy()
    at.positions += np.random.default_rng(seed).normal(scale=rattle, size=at.positions.shape)
    tmp = os.path.join(HERE, "_tmp_ref")
    os.makedirs(os.path.join(tmp, "ref"), exist_ok=True)
    spec = dict(name="ref", category="bulk", search_symbols=at.get_chemical_symbols(),
                cell=at.cell.array.tolist(), pbc=[True] * 3, variable_cell=True,
                volume_per_atom=at.get_volume() / len(at), region={"type": "cell"})
    with open(os.path.join(tmp, "ref", "problem.json"), "w") as f:
        json.dump(spec, f)
    try:
        p = Problem("ref", root=tmp)
        ctx = Context(p, budget=10 ** 8)
        pos, cell = at.positions.copy(), at.cell.array.copy()
        for _ in range(8):
            r = ctx.relax(pos, cell, fmax=fmax, steps=3000)
            pos, cell = r.pos, r.cell
            if r.converged:
                break
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = p.to_atoms(pos, cell)
    n = len(out)
    out.info.update(energy=float(r.e), converged=bool(r.converged))
    return out, float(r.e) / n, abs(np.linalg.det(cell)) / n


def describe(at, cut=1.25):
    """Coordination (all periodic images, bond < cut x covalent-radius sum) and min distances."""
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list
    zs = sorted(set(at.numbers.tolist()))
    i = neighbor_list("i", at, {(a, b): cut * (covalent_radii[a] + covalent_radii[b])
                                for a in zs for b in zs})
    coord = np.bincount(i, minlength=len(at))
    d = min_image_distances(at.positions, at.positions, at.cell.array, [True] * 3)
    np.fill_diagonal(d, np.inf)
    sym = np.array(at.get_chemical_symbols())
    parts = []
    for a in dict.fromkeys(sym):
        m = sym == a
        cn = {int(c): int(k) for c, k in zip(*np.unique(coord[m], return_counts=True))}
        dm = ",".join(f"{b}{d[m][:, sym == b].min():.2f}" for b in dict.fromkeys(sym))
        parts.append(f"{a}: CN{cn} dmin {dm}")
    return "; ".join(parts) + f"; cellpar {np.round(at.cell.cellpar(), 2).tolist()}"


def write_outputs(polymorphs, decomposition=None, notes=None):
    """polymorphs: {label: Atoms or (Atoms, relax kwargs)} (same composition; any cell size);
    decomposition: {label: [(Atoms, count)]} energies of phase mixtures."""
    a = (VPA * len(SYMBOLS)) ** (1 / 3)
    spec = dict(SPEC)
    spec.update(cell=[[round(a, 4), 0, 0], [0, round(a, 4), 0], [0, 0, round(a, 4)]],
                pbc=[True] * 3, variable_cell=True, volume_per_atom=VPA, region={"type": "cell"},
                budget=BUDGET, fmax=0.05, e_tol=E_TOL)
    order = ["name", "title", "category", "description", "search_symbols", "cell", "pbc",
             "variable_cell", "volume_per_atom", "region", "budget", "fmax", "e_tol"]
    with open(os.path.join(HERE, "problem.json"), "w") as f:
        json.dump({k: spec[k] for k in order}, f, indent=1)
    Problem(NAME)  # validates the spec
    refs, frames = {}, []
    for label, at in polymorphs.items():
        at, kw = at if isinstance(at, tuple) else (at, {})
        r, epa, vpa = relax(at, **kw)
        refs[label] = dict(e_per_atom=round(epa, 5), volume_per_atom=round(vpa, 3), n_atoms=len(r))
        r.info["label"] = label
        frames.append(r)
        print(f"{label:28s} n={len(r):3d} E/atom={epa:9.4f} V/atom={vpa:7.2f}  {describe(r)}",
              flush=True)
    write(os.path.join(HERE, "references.extxyz"), frames)
    dec = {}
    for label, parts in (decomposition or {}).items():
        e = n = 0.0
        for at, cnt in parts:
            _, epa, _ = relax(at)
            e += cnt * epa * len(at)
            n += cnt * len(at)
        dec[label] = dict(e_per_atom=round(e / n, 5))
        print(f"{label:28s} E/atom={e / n:9.4f}", flush=True)
    path = os.path.join(HERE, "pilot.json")
    pilot = json.load(open(path)) if os.path.exists(path) else {}
    pilot["reference_polymorphs"] = refs
    if dec:
        pilot["decomposition"] = dec
    if notes:
        pilot["reference_notes"] = notes
    with open(path, "w") as f:
        json.dump(pilot, f, indent=1)


# -------------------------------------------------------------- structures
def main():
    a = 7.04
    c15 = crystal(["Mg", "Cu"], basis=[(1 / 8, 1 / 8, 1 / 8), (1 / 2, 1 / 2, 1 / 2)], spacegroup=227,
                  setting=2, cellpar=[a, a, a, 90, 90, 90], primitive_cell=True)
    ah = a / np.sqrt(2)
    c14 = crystal(["Mg", "Cu", "Cu"], basis=[(1 / 3, 2 / 3, 0.063), (0, 0, 0), (-0.17, -0.34, .25)],
                  spacegroup=194, cellpar=[ah, ah, ah * 1.633, 90, 90, 120])
    alb2 = crystal(["Mg", "Cu"], basis=[(0, 0, 0), (1 / 3, 2 / 3, .5)], spacegroup=191,
                   cellpar=[4.2, 4.2, 3.4, 90, 90, 120])
    mosi2 = crystal(["Mg", "Cu"], basis=[(0, 0, 0), (0, 0, 1 / 3)], spacegroup=139,
                    cellpar=[3.2, 3.2, 8.0, 90, 90, 90], primitive_cell=True)
    polymorphs = {"C15 Laves (MgCu2, exp. ground state)": c15.repeat((2, 1, 1)),
                  "C14 Laves (MgZn2-type)": c14,
                  "C32 (AlB2-type)": alb2.repeat((2, 2, 1)),
                  "C11b (MoSi2-type)": mosi2.repeat((2, 2, 1))}
    write_outputs(polymorphs, {"hcp Mg + fcc Cu": [(bulk("Mg", "hcp", a=3.21, c=5.21), 2),
                                                   (bulk("Cu", "fcc", a=3.61), 8)]})


if __name__ == "__main__":
    main()
