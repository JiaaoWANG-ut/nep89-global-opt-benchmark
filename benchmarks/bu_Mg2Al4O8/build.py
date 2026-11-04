"""Build the bulk benchmark bu_Mg2Al4O8 (spinel MgAl2O4, 2 formula units = fcc primitive cell).

    python benchmarks/bu_Mg2Al4O8/build.py

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

import itertools

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
SYMBOLS = ["Mg"] * 2 + ["Al"] * 4 + ["O"] * 8
VPA = 10.4  # ~1.1 x experimental 9.43 A^3/atom (a = 8.083 A)
BUDGET = 28000
E_TOL = 0.02
SPEC = dict(
    name=NAME, title="MgAl2O4 spinel (Mg2Al4O8) crystal", category="bulk",
    description=("Variable-cell crystal-structure prediction of the spinel MgAl2O4 with two "
                 "formula units (14 atoms, the fcc primitive cell of spinel) per cell. "
                 "Experimental ground state: normal spinel (Fd-3m; Mg tetrahedral, Al "
                 "octahedral); inverse cation orderings and MgO + Al2O3 (corundum) compete. "
                 "NEP89 caveat: the inverse spinel (Al tetrahedral, Mg + Al octahedral) is "
                 "62 meV/atom below normal spinel, which is even 49 meV/atom above MgO + "
                 "Al2O3, so the NEP89 global minimum is the inverse spinel."),
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
def spinel():
    """Normal spinel MgAl2O4 (Fd-3m, origin choice 2), 14-atom fcc primitive cell."""
    return crystal(["Mg", "Al", "O"], basis=[(.125, .125, .125), (.5, .5, .5), (.2624, .2624, .2624)],
                   spacegroup=227, setting=2, cellpar=[8.0832] * 3 + [90] * 3, primitive_cell=True)


def main():
    normal = spinel()
    sym = np.array(normal.get_chemical_symbols())
    mg, al = np.where(sym == "Mg")[0], np.where(sym == "Al")[0]
    best = None
    for pair in itertools.combinations(al, 2):  # inverse: Al on 8a, Mg + Al on 16d
        inv = normal.copy()
        s = sym.copy()
        s[mg], s[list(pair)] = "Al", "Mg"
        inv.set_chemical_symbols(s)
        _, epa, _ = relax(inv)
        print(f"inverse ordering {tuple(map(int, pair))}: E/atom={epa:.5f}", flush=True)
        if best is None or epa < best[0]:
            best = (epa, inv)
    polymorphs = {"normal spinel (Fd-3m, exp. ground state)": normal,
                  "inverse spinel (lowest 14-atom ordering, NEP89 global minimum)": best[1]}
    mgo = bulk("MgO", "rocksalt", a=4.212)
    corundum = crystal(["Al", "O"], basis=[(0, 0, 0.3523), (0.3064, 0, 0.25)], spacegroup=167,
                       cellpar=[4.759, 4.759, 12.991, 90, 90, 120], primitive_cell=True)
    write_outputs(polymorphs, {"MgO (rocksalt) + Al2O3 (corundum)": [(mgo, 2), (corundum, 1)]})


if __name__ == "__main__":
    main()
