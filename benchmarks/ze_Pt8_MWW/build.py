"""Build ze_Pt8_MWW: Pt8 in the 12-ring supercage of all-silica MWW (MCM-22 / ITQ-1).

Reproducible pipeline (run from anywhere, ~1 min on one core):
  1. download the IZA-SC CIF (DLS-optimised SiO2 framework) if it is missing;
  2. hexagonal P6/mmm cell (216 atoms; heights 12.5, 12.5, 25.2 A -> no supercell);
  3. NEP89 variable-cell + position relaxation (calorine CPUNEP, BFGS on a
     FrechetCellFilter, fmax 0.01 eV/A); the relaxed cell is cached in
     framework_nep89.extxyz (use --force to recompute);
  4. framework checks (Si-O, O-Si-O, Si-O-Si, volume vs IZA, energy vs alpha-quartz);
  5. largest empty point of the pore system (grid + Nelder-Mead refinement of the
     distance to the nearest framework atom) = centre of the supercage (framework
     translated so that this point is the cell centre);
  6. search region: a box along the supercage axis (c) that covers the central bulge
     and both 12-ring cups (REGION_BOX half-widths; >= 99.9 % of the box is >= 2.2 A
     from every framework atom). The axis profile of the empty radius is 5.97 A at the
     centre, 4.73 A at the waists (z = +-3.5 A) and 5.10 A in the cups (z = +-6 A); the
     pilot GM sits in a cup, outside the original central sphere (radius 3.77 A);
  7. write substrate.extxyz, problem.json and build_log.json.
"""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import warnings

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase.spacegroup import crystal
from scipy.optimize import minimize
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from gopt.nep import NEP89  # noqa: E402

# ------------------------------------------------------------------ settings
NAME = "ze_Pt8_MWW"
TITLE = "Pt8 @ MWW"
CODE = "MWW"
METAL, N_METAL = "Pt", 8
PRIMITIVE_C = False          # C-centred cell -> primitive cell (half the atoms)
REPEAT = (1, 1, 1)          # supercell of the relaxed (primitive) cell
REGION_MARGIN = 2.2         # region radius = (largest empty-sphere radius) - margin (as ze_Pt4_CHA)
REGION_BOX = (2.0, 2.0, 7.0)  # Cartesian half-widths of a box along the supercage axis (None: sphere)
BUDGET = 5300   # 3 x median BH hit step (1782, 20 seeds) of campaign pilot-bze2-v3 (box region), >= 5000
E_TOL = 0.01    # GM-basin runs (<= +0.005 eV) re-relax (fmax 0.002) to E_best; closest isomer: tight minimum +0.0204 eV
DESCRIPTION = (
    'Pt8 cluster in the 12-ring supercage (7.1 x 18.2 A, 10-ring windows) of a frozen all-silica '
    'MWW (MCM-22/ITQ-1) framework (NEP89-relaxed hexagonal cell, 216 atoms); subnanometric Pt '
    'clusters encapsulated in MCM-22 cavities are sinter-resistant, size-selective catalysts (Liu '
    'et al., Nat. Mater. 16, 132 (2017)). Framework Al/H+ sites and framework relaxation are '
    'absent, NEP89 places this framework 0.8 kJ/mol SiO2 below alpha-quartz and attracts Pt to '
    'the cage wall by ~0.7 eV/atom even at Pt-O distances > 3 A, so the global minimum is a '
    'model-PES optimum.')


def cif_path():
    path = os.path.join(HERE, f"{CODE}.cif")
    if not os.path.exists(path):
        url = f"https://europe.iza-structure.org/IZA-SC/cif/{CODE}.cif"
        subprocess.run(["curl", "-s", "-f", url, "-o", path], check=True)
    return path


def read_iza():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at = read(cif_path())
    assert set(at.get_chemical_symbols()) == {"Si", "O"}, at.get_chemical_formula()
    return at


def primitive_c(at):
    """Primitive cell of a C-centred (1/2,1/2,0) lattice: a1=(a-b)/2, a2=(a+b)/2, a3=c."""
    P = np.array([[0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    cell = P @ at.cell.array
    frac = at.positions @ np.linalg.inv(cell)
    frac -= np.floor(frac)
    keep = []
    for k, f in enumerate(frac):
        dup = False
        for j in keep:
            d = f - frac[j]
            d -= np.round(d)
            if np.linalg.norm(d @ cell) < 0.05:
                dup = True
                break
        if not dup:
            keep.append(k)
    prim = at[keep]
    prim.set_cell(cell)
    prim.set_scaled_positions(frac[keep])
    assert len(prim) * 2 == len(at), (len(prim), len(at))
    return prim


def calculator():
    from calorine.calculators import CPUNEP
    with contextlib.redirect_stdout(io.StringIO()):
        return CPUNEP(NEP89)


def relax_cell(at, fmax=0.01, steps=3000):
    at = at.copy()
    at.calc = calculator()
    opt = BFGS(FrechetCellFilter(at), logfile=None)
    conv = opt.run(fmax=fmax, steps=steps)
    e = at.get_potential_energy()
    at.calc = None
    return at, e, bool(conv), opt.nsteps


def quartz_energy():
    """NEP89 energy per SiO2 of relaxed alpha-quartz (reference for framework stability)."""
    q = crystal(["Si", "O"], basis=[(0.4697, 0.0, 2 / 3), (0.4135, 0.2669, 0.1191 + 2 / 3)],
                spacegroup=154, cellpar=[4.913, 4.913, 5.405, 90, 90, 120])
    assert len(q) == 9
    q, e, conv, n = relax_cell(q, fmax=0.005)
    return e / (len(q) // 3), q.get_volume() / (len(q) // 3)


def framework_checks(at):
    sym = np.array(at.get_chemical_symbols())
    i, j, d, D = neighbor_list("ijdD", at, 2.0)
    m = (sym[i] == "Si") & (sym[j] == "O")
    mo = (sym[i] == "O") & (sym[j] == "Si")
    cn_si = np.bincount(i[m], minlength=len(at))[sym == "Si"]
    cn_o = np.bincount(i[mo], minlength=len(at))[sym == "O"]

    def angles(center_mask, pair_mask):
        out = []
        for c in np.where(center_mask)[0]:
            v = D[(i == c) & pair_mask]
            for a in range(len(v)):
                for b in range(a + 1, len(v)):
                    cs = v[a] @ v[b] / np.linalg.norm(v[a]) / np.linalg.norm(v[b])
                    out.append(np.degrees(np.arccos(np.clip(cs, -1, 1))))
        return np.array(out)

    osio = angles(sym == "Si", m)
    siosi = angles(sym == "O", mo)
    r = lambda x: round(float(x), 4)  # noqa: E731
    return dict(si_o_mean=r(d[m].mean()), si_o_min=r(d[m].min()), si_o_max=r(d[m].max()),
                o_si_o_mean=r(osio.mean()), o_si_o_std=r(osio.std()), o_si_o_min=r(osio.min()),
                o_si_o_max=r(osio.max()), si_o_si_mean=r(siosi.mean()), si_o_si_min=r(siosi.min()),
                si_o_si_max=r(siosi.max()), cn_si=sorted(set(cn_si.tolist())),
                cn_o=sorted(set(cn_o.tolist())))


def heights(cell):
    return 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)


def largest_cavity(at, spacing=0.25, start=None):
    """Point of the pore system farthest from every framework atom (and that distance).

    Global grid search, or only a local refinement from ``start`` if given."""
    cell = at.cell.array
    pos = at.get_positions(wrap=True)
    sh = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float)
    tree = cKDTree((pos[None] + (sh @ cell)[:, None]).reshape(-1, 3))
    if start is None:
        n = np.maximum(np.ceil(at.cell.lengths() / spacing).astype(int), 2)
        g = np.stack(np.meshgrid(*[(np.arange(k) + 0.5) / k for k in n], indexing="ij"),
                     -1).reshape(-1, 3)
        pts = g @ cell
        d, _ = tree.query(pts)
        start = pts[int(np.argmax(d))]
    res = minimize(lambda x: -tree.query(x)[0], np.asarray(start, float), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000))
    x = res.x
    f = x @ np.linalg.inv(cell)
    x = (f - np.floor(f)) @ cell
    return x, float(tree.query(x)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="redo the NEP89 framework relaxation")
    a = ap.parse_args()
    iza = read_iza()
    unit = primitive_c(iza) if PRIMITIVE_C else iza
    n_t = int((np.array(unit.get_chemical_symbols()) == "Si").sum())
    cache = os.path.join(HERE, "framework_nep89.extxyz")
    log = {}
    if os.path.exists(cache) and not a.force:
        fw = read(cache)
        old = os.path.join(HERE, "build_log.json")
        if os.path.exists(old):
            log.update(json.load(open(old)).get("relaxation", {}))
        print("using cached relaxed framework", cache)
    else:
        fw, e, conv, nst = relax_cell(unit)
        eq, vq = quartz_energy()
        log = dict(converged=conv, steps=nst, e_per_sio2=round(e / n_t, 5),
                   e_quartz_per_sio2=round(eq, 5),
                   de_vs_quartz_kJmol=round((e / n_t - eq) * 96.485, 2),
                   v_quartz_per_sio2=round(vq, 3))
        fw.info.clear()
        write(cache, fw)
        print(f"relaxed framework: converged={conv} steps={nst} E/SiO2={e / n_t:.4f} eV, "
              f"quartz {eq:.4f} eV -> {log['de_vs_quartz_kJmol']} kJ/mol")
    v_iza = unit.get_volume() / n_t
    v_nep = fw.get_volume() / n_t
    checks = framework_checks(fw)
    print("cellpar IZA ", np.round(unit.cell.cellpar(), 3).tolist())
    print("cellpar NEP ", np.round(fw.cell.cellpar(), 3).tolist())
    print(f"volume/T: IZA {v_iza:.2f}  NEP89 {v_nep:.2f} A^3  ratio {v_nep / v_iza:.4f}")
    print("checks", checks)

    sub = fw.repeat(REPEAT)
    sub.wrap()
    sub.info.clear()
    h = heights(sub.cell.array)
    print("supercell atoms", len(sub), "heights", np.round(h, 2).tolist())
    assert h.min() >= 12.0, h
    center, rad = largest_cavity(sub)
    # translate the framework so that the cavity centre is the cell centre (no wrapping issues)
    sub.translate(0.5 * sub.cell.array.sum(axis=0) - center)
    sub.wrap()
    center, rad2 = largest_cavity(sub, start=0.5 * sub.cell.array.sum(axis=0))
    assert abs(rad2 - rad) < 1e-3, (rad, rad2)
    radius = round(rad - REGION_MARGIN, 2)
    print(f"cavity centre {np.round(center, 4).tolist()}  empty-sphere radius {rad:.3f} A  "
          f"-> region radius {radius}")
    write(os.path.join(HERE, "substrate.extxyz"), sub)
    region = dict(type="sphere", center=[round(float(x), 4) for x in center], radius=radius)
    box_stats = None
    if REGION_BOX is not None:
        hw = np.array(REGION_BOX, float)
        lo, hi = np.round(center - hw, 4), np.round(center + hw, 4)
        region = dict(type="box", lo=lo.tolist(), hi=hi.tolist())
        sh = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float)
        tree = cKDTree((sub.positions[None] + (sh @ sub.cell.array)[:, None]).reshape(-1, 3))
        pts = lo + np.random.default_rng(0).random((20000, 3)) * (hi - lo)
        d = tree.query(pts)[0]
        box_stats = dict(d_min=round(float(d.min()), 3), d_p05=round(float(np.percentile(d, 5)), 3),
                         d_median=round(float(np.median(d)), 3),
                         frac_ge_2p2=round(float((d >= 2.2).mean()), 4))
        print("box region", region, "distance to framework", box_stats)
    spec = dict(name=NAME, title=TITLE, category="zeolite", description=DESCRIPTION,
                search_symbols=[METAL] * N_METAL, substrate_file="substrate.extxyz",
                pbc=[True, True, True], region=region,
                connected=True, budget=BUDGET, fmax=0.05, e_tol=E_TOL)
    with open(os.path.join(HERE, "problem.json"), "w") as fh:
        json.dump(spec, fh, indent=1)
    out = dict(code=CODE, n_framework=len(sub), n_search=N_METAL, repeat=REPEAT,
               primitive=PRIMITIVE_C, relaxation=log,
               cellpar_iza=np.round(unit.cell.cellpar(), 4).tolist(),
               cellpar_nep89=np.round(fw.cell.cellpar(), 4).tolist(),
               vol_per_T_iza=round(v_iza, 3), vol_per_T_nep89=round(v_nep, 3),
               vol_ratio=round(v_nep / v_iza, 4), checks=checks,
               heights=np.round(h, 3).tolist(), cavity_center=np.round(center, 4).tolist(),
               cavity_radius=round(rad, 3), region_radius=radius, region_box=box_stats)
    with open(os.path.join(HERE, "build_log.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
