#!/usr/bin/env python3
"""Build benchmarks/bu_Si4O8: variable-cell crystal-structure prediction of SiO2 (Z=4).

Writes problem.json and relaxes the experimental ground state and competing
polymorphs on the benchmark energy surface (NEP89 + short-range guard, through
gopt.context.Context and its native variable-cell relaxation). The reference
energies go to pilot.json["reference_polymorphs"] (other pilot.json keys are
kept), the relaxed structures to reference_polymorphs.extxyz. Polymorphs whose
cell is not commensurate with the benchmark cell are relaxed in their own cell
(fits_cell = false).

    python benchmarks/bu_Si4O8/build.py
"""
import json
import os
import sys
import tempfile

import numpy as np
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
SYMBOLS = ["Si"] * 4 + ["O"] * 8
VPA = 13.8  # 1.1 x experimental alpha-quartz (12.56 A^3/atom)
SPEC = dict(
    name=NAME,
    title="SiO2 (Si4O8) crystal",
    description=(
        "Crystal structure prediction of SiO2 with 4 formula units (12 atoms) per cell, which "
        "fits alpha-cristobalite, tridymite and stishovite but not the experimental ground state "
        "alpha-quartz (Z=3; 29.6 meV/atom above the GM with NEP89). NEP89 over-stabilises open "
        "corner-sharing SiO4 frameworks (alpha-cristobalite 24.5 meV/atom below quartz): the "
        "putative NEP89 global minimum is an all-silica ABW zeolite framework (coordination "
        "sequence 4,10,21,36,54,...; 17.3 A^3/atom, 23% less dense than quartz), 5.1 meV/atom "
        "below alpha-cristobalite."),
    category="bulk",
    search_symbols=SYMBOLS,
    cell=(np.eye(3) * round((VPA * len(SYMBOLS)) ** (1 / 3), 3)).tolist(),
    pbc=[True, True, True],
    variable_cell=True,
    volume_per_atom=VPA,
    region={"type": "cell"},
    budget=43000,
    fmax=0.05,
    e_tol=0.02,
)


def polymorphs():
    """Experimental structures (any multiple of the formula unit)."""
    return {
        # alpha-quartz, P3_221 (Levien, Prewitt & Weidner, Am. Mineral. 65, 920 (1980))
        "alpha_quartz": crystal(["Si", "O"], basis=[(0.4697, 0, 2 / 3), (0.4135, 0.2669, 0.7858)],
                                spacegroup=154, cellpar=[4.9134, 4.9134, 5.4052, 90, 90, 120]),
        # alpha-cristobalite, P4_12_12 (Downs & Palmer, Am. Mineral. 79, 9 (1994))
        "alpha_cristobalite": crystal(["Si", "O"], basis=[(0.3003, 0.3003, 0),
                                                          (0.2392, 0.1044, 0.1787)],
                                      spacegroup=92, cellpar=[4.9709, 4.9709, 6.9278, 90, 90, 90]),
        # ideal high tridymite, P6_3/mmc (Si 4f z=0.062, O 2c + 6g; relaxes to a distorted variant)
        "tridymite": crystal(["Si", "O", "O"], basis=[(1 / 3, 2 / 3, 0.0620), (1 / 3, 2 / 3, 0.25),
                                                      (0.5, 0, 0)],
                             spacegroup=194, cellpar=[5.03, 5.03, 8.22, 90, 90, 120]),
        # stishovite, P4_2/mnm (rutile type; Ross et al., Am. Mineral. 75, 739 (1990))
        "stishovite": crystal(["Si", "O"], basis=[(0, 0, 0), (0.3062, 0.3062, 0)],
                              spacegroup=136, cellpar=[4.1773, 4.1773, 2.6655, 90, 90, 90]),
        # silica sodalite (SOD framework, IZA idealised coordinates), Im-3m primitive cell
        "sodalite_SiO2": crystal(["Si", "O"], basis=[(0.25, 0.5, 0), (0.1467, 0.5, 0.8533)],
                                 spacegroup=229, cellpar=[8.965] * 3 + [90] * 3,
                                 primitive_cell=True),
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
