"""Build the bulk benchmark bu_Sr4Ti4O12 (SrTiO3 perovskite, 4 formula units per cell).

    python benchmarks/bu_Sr4Ti4O12/build.py

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
from ase.build import bulk, make_supercell
from ase.spacegroup import crystal
from ase.io import write

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from gopt.context import Context  # noqa: E402
from gopt.problem import Problem, min_image_distances  # noqa: E402

NAME = os.path.basename(HERE)
SYMBOLS = ["Sr"] * 4 + ["Ti"] * 4 + ["O"] * 12
VPA = 13.1  # ~1.1 x experimental 11.91 A^3/atom (a = 3.905 A)
BUDGET = 22000
E_TOL = 0.02
# NEP89 has a short-range Sr-O hole: with the default guard (0.75) the lowest Sr2Ti2O6
# structure had six Sr-O contacts at 1.93 A (0.74 x covalent-radius sum; real 2.6-2.8 A).
GUARD = 0.80
SPEC = dict(
    name=NAME, title="SrTiO3 (Sr4Ti4O12) crystal", category="bulk",
    description=("Variable-cell crystal-structure prediction of the perovskite SrTiO3 with four "
                 "formula units (20 atoms) per cell, enough for octahedral tilt systems "
                 "(a0a0c-, I4/mcm: the experimental low-temperature phase; cubic Pm-3m above "
                 "105 K; a-b+a- Pnma as in CaTiO3); ilmenite and hexagonal-perovskite "
                 "arrangements compete. In NEP89 tilts gain < 1 meV/atom, so cubic and tilted "
                 "perovskites are nearly degenerate. guard_factor 0.80: NEP89 has a "
                 "short-range Sr-O hole (with the default 0.75 an unphysical Sr2Ti2O6 "
                 "structure with Sr-O = 1.93 A was 0.046 eV below the perovskite)."),
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
                volume_per_atom=at.get_volume() / len(at), region={"type": "cell"},
                guard_factor=GUARD)
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
                budget=BUDGET, fmax=0.05, e_tol=E_TOL, guard_factor=GUARD)
    order = ["name", "title", "category", "description", "search_symbols", "cell", "pbc",
             "variable_cell", "volume_per_atom", "region", "budget", "fmax", "e_tol",
             "guard_factor"]
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
def tilt(at, axis, deg, a):
    """Antiphase (R-point) octahedral rotation about ``axis`` of a perovskite supercell."""
    at = at.copy()
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    sym = np.array(at.get_chemical_symbols())
    ti = np.where(sym == "Ti")[0]
    cell = at.cell.array
    pos = at.positions
    for o in np.where(sym == "O")[0]:
        k = ti[np.argmin(min_image_distances(pos[o], pos[ti], cell, [True] * 3)[0])]
        f = (pos[o] - pos[k]) @ np.linalg.inv(cell)
        v = (f - np.round(f)) @ cell
        sign = (-1) ** int(np.floor(pos[k] / a + 1e-6).sum())
        pos[o] = pos[o] + sign * np.radians(deg) * np.cross(axis, v)
    at.positions = pos
    return at


def main():
    a = 3.905
    cub = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (.5, .5, .5), (.5, .5, 0)], spacegroup=221,
                  cellpar=[a, a, a, 90, 90, 90])
    r10 = make_supercell(cub, [[1, 1, 0], [1, 0, 1], [0, 1, 1]])  # allows R-point tilts
    r20 = make_supercell(cub, [[1, 1, 0], [-1, 1, 0], [0, 0, 2]])  # I4/mcm conventional cell
    sc = 1.045  # FeTiO3 volume -> SrTiO3 volume
    ilm = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0.3554), (0, 0, 0.1464), (0.317, 0.020, 0.245)],
                  spacegroup=148, cellpar=[5.088 * sc, 5.088 * sc, 14.08 * sc, 90, 90, 120],
                  primitive_cell=True)
    h2 = crystal(["Sr", "Ti", "O"], basis=[(1 / 3, 2 / 3, .75), (0, 0, 0), (0.15, 0.30, .25)],
                 spacegroup=194, cellpar=[5.65, 5.65, 4.70, 90, 90, 120])
    # CaTiO3 (Pbnm, Sasaki et al. 1987) in the standard Pnma setting, scaled to the SrTiO3 volume
    so = (59.55 / (5.3796 * 5.4423 * 7.6401 / 4)) ** (1 / 3)
    pnma = crystal(["Sr", "Ti", "O", "O"],
                   basis=[(0.0349, 0.25, -0.0065), (0.5, 0, 0), (0.4842, 0.25, 0.0711),
                          (0.2884, 0.0372, 0.7109)],
                   spacegroup=62, cellpar=[5.4423 * so, 7.6401 * so, 5.3796 * so, 90, 90, 90])
    polymorphs = {"perovskite a0a0c- tilt (I4/mcm, exp. low T)": tilt(r20, [0, 0, 1], 6, a),
                  "perovskite cubic (Pm-3m, exp. room T)": (r20, dict(rattle=0.0)),
                  "perovskite a-a-a- tilt (R-3c)": tilt(r10, [1, 1, 1], 6, a),
                  "perovskite a-b+a- tilt (Pnma, CaTiO3-type)": pnma,
                  "ilmenite-type (R-3)": ilm,
                  # unstable in NEP89+guard: any rattle collapses it to an open 3/4-coordinated
                  # framework, hence a symmetry-preserving relaxation (face-sharing Ti-Ti 2.52 A)
                  "2H hexagonal perovskite (BaNiO3-type, symmetric)": (h2, dict(rattle=0.0))}
    sro = bulk("SrO", "rocksalt", a=5.16)
    rutile = crystal(["Ti", "O"], basis=[(0, 0, 0), (0.305, 0.305, 0)], spacegroup=136,
                     cellpar=[4.594, 4.594, 2.959, 90, 90, 90])
    anatase = crystal(["Ti", "O"], basis=[(0, 0, 0), (0, 0, 0.208)], spacegroup=141,
                      cellpar=[3.785, 3.785, 9.514, 90, 90, 90], setting=1, primitive_cell=True)
    write_outputs(polymorphs, {"SrO (rocksalt) + TiO2 (anatase)": [(sro, 2), (anatase, 1)],
                               "SrO (rocksalt) + TiO2 (rutile)": [(sro, 2), (rutile, 1)]})


if __name__ == "__main__":
    main()
