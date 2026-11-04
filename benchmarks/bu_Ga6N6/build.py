"""Build the bulk benchmark bu_Ga6N6 (GaN, 6 formula units per cell).

    python benchmarks/bu_Ga6N6/build.py

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
from ase.build import bulk, make_supercell
from ase.spacegroup import crystal
from ase.io import write

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from gopt.context import Context  # noqa: E402
from gopt.problem import Problem, min_image_distances  # noqa: E402

NAME = os.path.basename(HERE)
SYMBOLS = ["Ga"] * 6 + ["N"] * 6
VPA = 12.6  # ~1.1 x experimental 11.42 A^3/atom (wurtzite)
BUDGET = 10000
E_TOL = 0.02
SPEC = dict(
    name=NAME, title="GaN (Ga6N6) crystal", category="bulk",
    description=("Variable-cell crystal-structure prediction of GaN with six formula units (12 "
                 "atoms) per cell. Experimental ground state: wurtzite (P6_3mc); zinc "
                 "blende (~10 meV/atom higher in DFT), the 6H polytype and the high-pressure "
                 "rocksalt phase compete. NEP89 caveat: NEP89 puts zinc blende 33 meV/atom "
                 "below wurtzite (wrong polytype ordering), so its global minimum is zinc blende."),
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
def polytype(stack, a, layer, u, cation, anion):
    """Hexagonal AX polytype; cation layers stacked as ``stack`` ('AB' wurtzite,
    'ABC' zinc blende, 'ABCB' 4H, ...), anion 2*u layer spacings above its cation."""
    xy = {"A": (0, 0), "B": (1 / 3, 2 / 3), "C": (2 / 3, 1 / 3)}
    n = len(stack)
    frac = [(*xy[s], k / n) for k, s in enumerate(stack)]
    frac += [(x, y, z + 2 * u / n) for x, y, z in frac]
    cell = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, layer * n]]
    return Atoms([cation] * n + [anion] * n, scaled_positions=frac, cell=cell, pbc=True)


def main():
    a, layer, u = 3.189, 5.185 / 2, 0.377
    h55 = crystal(["Ga", "N"], basis=[(1 / 3, 2 / 3, .25), (2 / 3, 1 / 3, .25)], spacegroup=194,
                  cellpar=[3.38, 3.38, 4.12, 90, 90, 120])
    rs = make_supercell(bulk("GaN", "rocksalt", a=4.22), np.diag([3, 2, 1]))
    polymorphs = {"wurtzite (exp. ground state)": polytype("ABABAB", a, layer, u, "Ga", "N"),
                  "zinc blende": polytype("ABCABC", a, layer, u, "Ga", "N"),
                  "6H polytype": polytype("ABCACB", a, layer, u, "Ga", "N"),
                  "rocksalt (high pressure)": rs,
                  "h-BN-like 5-5 layered": h55.repeat((3, 1, 1))}
    alpha_ga = crystal("Ga", basis=[(0, 0.1549, 0.0810)], spacegroup=64,
                        cellpar=[4.5192, 7.6586, 4.5258, 90, 90, 90])
    n2 = Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.10)], cell=[9.0, 9.2, 9.4], pbc=True)
    write_outputs(polymorphs, {"alpha-Ga + N2": [(alpha_ga, 0.75), (n2, 3)]})


if __name__ == "__main__":
    main()
