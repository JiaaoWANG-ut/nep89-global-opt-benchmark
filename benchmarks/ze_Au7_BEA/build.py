#!/usr/bin/env python3
"""Build the zeolite benchmark ze_Au7_BEA: Au cluster at a channel intersection of all-silica beta.

    python benchmarks/ze_Au7_BEA/build.py [--n 7] [--out DIR] [--reuse]

Standalone, reproducible recipe (same style as ze_Pt4_CHA in scripts/build_dev_benchmarks.py):
1. download the IZA-SC CIF of *BEA (polymorph A, P4122, 192 atoms; the europe
   mirror has no BEA.cif, the america mirror does) unless present;
2. NEP89 (calorine CPUNEP) variable-cell + position relaxation (ASE BFGS on a
   FrechetCellFilter, fmax 0.005 eV/A) of the tetragonal cell (interplanar
   spacings ~12.7 x 12.7 x 26 A, used as is);
3. largest empty sphere of the framework (grid + Nelder-Mead refinement) = the
   intersection of the 12-ring channels; region radius = distance to the nearest
   framework atom - 2.2 A;
4. write substrate.extxyz, problem.json and framework.json (geometry checks).

--reuse keeps an existing substrate.extxyz in --out and only redoes steps 3-4.
"""
import argparse
import contextlib
import io
import itertools
import json
import os
import sys
import time
import urllib.request

import numpy as np
from ase.build import make_supercell
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from gopt.nep import NEP89  # noqa: E402

# ------------------------------------------------------------------ configuration
CODE = "BEA"
METAL, N_DEFAULT = "Au", 7
SUPERCELL = None                                 # applied after relaxing the CIF cell
PRIMITIVE_FCC = False                            # reduce an F-centred CIF cell before relaxing
CLEARANCE = 2.2                                  # region radius = cavity radius - CLEARANCE
# pilot (pilot.json): Au6 too easy (BH ERT 2.1k), Au8 too hard (ERT 121k); budget = 3x median BH hit
BUDGET, E_TOL = 12000, 0.02
TITLE = "{metal}{n} @ BEA (channel intersection)"
DESCRIPTION = (
    "{metal}{n} cluster at an intersection of the 12-ring channels of a frozen all-silica zeolite "
    "beta (polymorph A; NEP89-relaxed tetragonal cell, 192 framework atoms). Metal clusters "
    "immobilised inside beta crystals resist sintering (Wang et al., Nat. Catal. 1, 540 (2018)). "
    "Caveat: NEP89 is not fitted to metal/zeolite interfaces and the framework is rigid, so "
    "cluster-wall binding is qualitative (NEP89 GM: planar Au7 spanning the intersection).")
IZA_MIRRORS = ("https://europe.iza-structure.org/IZA-SC/cif/{}.cif",
               "https://america.iza-structure.org/IZA-SC/cif/{}.cif")


def fetch_cif(path):
    if os.path.exists(path):
        return path
    for url in IZA_MIRRORS:
        try:
            data = urllib.request.urlopen(url.format(CODE), timeout=60).read()
        except Exception:
            continue
        if data.lstrip().startswith(b"data_"):
            with open(path, "wb") as f:
                f.write(data)
            return path
    raise SystemExit(f"could not download {CODE}.cif from the IZA-SC database")


def nep_calc():
    from calorine.calculators import CPUNEP
    with contextlib.redirect_stdout(io.StringIO()):
        return CPUNEP(NEP89)


def primitive_fcc(at):
    """Rhombohedral primitive cell of an F-centred cubic cell (4x fewer atoms)."""
    cell = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) @ at.cell.array
    frac = np.linalg.solve(cell.T, at.positions.T).T % 1.0
    keep = []
    for i, f in enumerate(frac):
        if all(np.linalg.norm(((f - frac[j] + 0.5) % 1.0 - 0.5) @ cell) > 0.1 for j in keep):
            keep.append(i)
    out = at[keep]
    out.set_cell(cell)
    out.set_scaled_positions(frac[keep])
    return out


def relax_cell(at, fmax=0.005, steps=3000):
    at.calc = nep_calc()
    opt = BFGS(FrechetCellFilter(at), logfile=None)
    conv = opt.run(fmax=fmax, steps=steps)
    print(f"  NEP89 variable-cell relaxation: converged={conv} steps={opt.nsteps} "
          f"E={at.get_potential_energy():.4f} eV")
    return at


def heights(cell):
    return 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)


def framework_report(at, ref):
    sym = np.array(at.get_chemical_symbols())
    si = sym == "Si"
    i, j, d, D = neighbor_list("ijdD", at, 2.0)
    cn = np.bincount(i, minlength=len(at))
    m = si[i] & ~si[j]

    def angles(center_mask):
        out = []
        for c in np.where(center_mask)[0]:
            v = D[i == c]
            for a, b in itertools.combinations(range(len(v)), 2):
                cosang = v[a] @ v[b] / np.linalg.norm(v[a]) / np.linalg.norm(v[b])
                out.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        return np.array(out)

    osio, siosi = angles(si), angles(~si)
    st = lambda x: dict(mean=round(float(x.mean()), 4), min=round(float(x.min()), 4),  # noqa: E731
                        max=round(float(x.max()), 4))
    vpt = at.get_volume() / si.sum()
    vpt_ref = ref.get_volume() / (np.array(ref.get_chemical_symbols()) == "Si").sum()
    f = at.get_forces() if at.calc is not None else None
    return dict(
        code=CODE, n_atoms=len(at), formula=at.get_chemical_formula(),
        cellpar=[round(float(x), 4) for x in at.cell.cellpar()],
        cellpar_iza=[round(float(x), 4) for x in ref.cell.cellpar()],
        heights=[round(float(x), 3) for x in heights(at.cell.array)],
        volume_per_T=round(float(vpt), 3), volume_per_T_iza=round(float(vpt_ref), 3),
        volume_ratio_nep_iza=round(float(vpt / vpt_ref), 4),
        si_cn=sorted(set(cn[si].tolist())), o_cn=sorted(set(cn[~si].tolist())),
        si_o=st(d[m]), o_si_o=st(osio), si_o_si=st(siosi),
        energy_per_SiO2=(round(float(at.get_potential_energy() / si.sum()), 5)
                         if at.calc is not None else None),
        max_force=(round(float(np.linalg.norm(f, axis=1).max()), 5) if f is not None else None))


def _tree(at):
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float) @ at.cell.array
    return cKDTree((at.positions[None] + shifts[:, None]).reshape(-1, 3))


def _grid(at, spacing):
    n = np.maximum(np.ceil(at.cell.lengths() / spacing).astype(int), 4)
    return np.stack(np.meshgrid(*[(np.arange(k) + 0.5) / k for k in n], indexing="ij"),
                    -1).reshape(-1, 3)


def largest_cavity(at, spacing=0.3, refine=10):
    """Centre and radius (distance to the nearest atom centre) of the largest empty sphere."""
    cell = at.cell.array
    tree = _tree(at)
    pts = _grid(at, spacing) @ cell
    d = tree.query(pts)[0]
    best = (None, -1.0)
    for k in np.argsort(-d)[:refine]:
        r = minimize(lambda x: -tree.query(x)[0], pts[k], method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-7, maxiter=4000))
        if -r.fun > best[1] + 1e-4:
            best = (r.x, -r.fun)
    frac = np.linalg.solve(cell.T, best[0]) % 1.0
    c = frac @ cell
    return c, float(tree.query(c)[0])


def cavity_list(at, spacing=0.4, min_radius=3.5):
    """Distinct local maxima of the empty-sphere radius (for documentation)."""
    cell = at.cell.array
    tree = _tree(at)
    n = np.maximum(np.ceil(at.cell.lengths() / spacing).astype(int), 4)
    g = _grid(at, spacing)
    d = tree.query(g @ cell)[0]
    peak = (maximum_filter(d.reshape(n), size=3, mode="wrap").ravel() == d) & (d >= min_radius)
    out = []
    for k in np.where(peak)[0][np.argsort(-d[peak])]:
        r = minimize(lambda x: -tree.query(x)[0], g[k] @ cell, method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-6))
        f = np.linalg.solve(cell.T, r.x) % 1.0
        if all(np.linalg.norm(((f - q + 0.5) % 1.0 - 0.5) @ cell) > 1.0 for q, _ in out):
            out.append((f, -r.fun))
    return [dict(frac=[round(float(x), 4) for x in f], radius=round(float(rr), 3)) for f, rr in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="number of metal atoms")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--reuse", action="store_true", help="keep an existing substrate.extxyz")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--e-tol", type=float, default=E_TOL)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    name = f"ze_{METAL}{a.n}_{CODE}"
    cif = fetch_cif(os.path.join(HERE, f"{CODE}.cif"))
    ref = read(cif)
    sub_file = os.path.join(a.out, "substrate.extxyz")
    reuse = a.reuse and os.path.exists(sub_file)
    if reuse:
        fw = read(sub_file)
        fw.calc = nep_calc()
    else:
        fw = read(cif)
        print(f"{CODE}: CIF {len(fw)} atoms {fw.get_chemical_formula()} "
              f"cellpar {np.round(fw.cell.cellpar(), 3).tolist()}")
        if PRIMITIVE_FCC:
            fw = primitive_fcc(fw)
            print(f"  primitive cell: {len(fw)} atoms")
        fw = relax_cell(fw)
        if SUPERCELL is not None:
            e1 = fw.get_potential_energy() / len(fw)
            fw = make_supercell(fw, np.array(SUPERCELL))
            fw.calc = nep_calc()
            print(f"  supercell {len(fw)} atoms; dE/atom vs relaxed cell "
                  f"{fw.get_potential_energy() / len(fw) - e1:.2e} eV")
        fw.wrap()
        sub = fw.copy()
        sub.calc = None
        sub.info.clear()
        sub.arrays = {k: v for k, v in sub.arrays.items() if k in ("numbers", "positions")}
        write(sub_file, sub)
        # continue from the written file, so a full rebuild and --reuse see identical coordinates
        # (symmetry-equivalent cavities tie and coordinate rounding decides which one is picked)
        fw = read(sub_file)
        fw.calc = nep_calc()
    rep = framework_report(fw, ref)
    center, rad = largest_cavity(fw)
    rep["cavity_center"] = [round(float(x), 6) for x in center]
    rep["cavity_radius"] = round(rad, 4)
    rep["cavities"] = cavity_list(fw)
    print(json.dumps(rep, indent=1))
    with open(os.path.join(a.out, "framework.json"), "w") as f:
        json.dump(rep, f, indent=1)
    spec = dict(
        name=name, title=TITLE.format(metal=METAL, n=a.n), category="zeolite",
        description=DESCRIPTION.format(metal=METAL, n=a.n),
        search_symbols=[METAL] * a.n, substrate_file="substrate.extxyz", pbc=[True] * 3,
        region=dict(type="sphere", center=[round(float(x), 6) for x in center],
                    radius=round(rad - CLEARANCE, 2)),
        connected=True, budget=a.budget, fmax=0.05, e_tol=a.e_tol)
    with open(os.path.join(a.out, "problem.json"), "w") as f:
        json.dump(spec, f, indent=1)
    print(f"wrote {a.out} ({name}); {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
