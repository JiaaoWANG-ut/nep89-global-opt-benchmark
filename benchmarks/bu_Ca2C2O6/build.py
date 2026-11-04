"""Build the bulk benchmark bu_Ca2C2O6 (CaCO3, 2 formula units = rhombohedral primitive cell of calcite).

    python benchmarks/bu_Ca2C2O6/build.py

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
from ase import Atoms
from ase.build import bulk
from ase.geometry import cellpar_to_cell
from ase.spacegroup import crystal
from ase.io import write

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from gopt.context import Context  # noqa: E402
from gopt.problem import Problem, min_image_distances  # noqa: E402

NAME = os.path.basename(HERE)
SYMBOLS = ["Ca"] * 2 + ["C"] * 2 + ["O"] * 6
VPA = 13.5  # ~1.1 x experimental 12.26 A^3/atom (calcite)
BUDGET = 100000
E_TOL = 0.02
SPEC = dict(
    name=NAME, title="CaCO3 (Ca2C2O6) crystal", category="bulk",
    description=("Variable-cell crystal-structure prediction of CaCO3 with two formula units (10 "
                 "atoms) per cell: planar CO3 anions must assemble from random atoms. "
                 "Experimental ground state: calcite (R-3c, 10-atom primitive cell); "
                 "aragonite and vaterite are metastable. NEP89 caveats: the putative NEP89 "
                 "global minimum is a vaterite-like layered polymorph (CO3 planes normal "
                 "to the Ca layers, +6% volume) 7.4 meV/atom below calcite, and CaO + CO2 "
                 "is only 0.32 eV/f.u. above calcite (exp. ~1.8 eV), a frequent trap."),
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
    calcite = crystal(["Ca", "C", "O"], basis=[(0, 0, 0), (0, 0, .25), (0.2578, 0, .25)],
                      spacegroup=167, cellpar=[4.988, 4.988, 17.061, 90, 90, 120],
                      primitive_cell=True)
    # aragonite, Pnma setting (a, b, c) = Pmcn (c, a, b) of de Villiers, Am. Mineral. 56, 758 (1971)
    aragonite = crystal(["Ca", "C", "O", "O"],
                        basis=[(0.75985, .25, 0.41500), (-0.08518, .25, 0.76211),
                               (-0.09566, .25, 0.92224), (-0.08646, 0.47347, 0.68065)],
                        spacegroup=62, cellpar=[5.7407, 4.9611, 7.9672, 90, 90, 90])
    # putative NEP89 global minimum found in the pilot (BH), Niggli-reduced
    vat = Atoms("Ca2C2O6", cell=cellpar_to_cell([4.266, 4.269, 7.967, 89.852, 89.87, 63.925]),
                scaled_positions=[(0.4034, 0.7170, 0.6835), (0.4020, 0.7163, 0.1839),
                                  (0.7620, 0.0270, 0.9337), (0.0925, 0.3566, 0.4329),
                                  (0.2590, 0.2768, 0.2916), (0.8420, 0.8619, 0.7922),
                                  (0.7547, 0.5162, 0.4336), (0.2609, 0.2775, 0.5735),
                                  (0.6010, 0.3646, 0.9346), (0.8422, 0.8574, 0.0741)], pbc=True)
    polymorphs = {"calcite (exp. ground state)": calcite,
                  "aragonite (20 atoms/cell)": aragonite,
                  "vaterite-like layered (NEP89 global minimum, pilot)": vat}
    cao = bulk("CaO", "rocksalt", a=4.81)
    co2 = Atoms("CO2", positions=[(0, 0, 0), (0, 0, 1.16), (0, 0, -1.16)], cell=[9.0, 9.2, 9.4],
                pbc=True)
    write_outputs(polymorphs, {"CaO (rocksalt) + CO2 (molecule)": [(cao, 2), (co2, 2)]})


if __name__ == "__main__":
    main()
