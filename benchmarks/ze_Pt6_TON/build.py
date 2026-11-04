"""Build ze_Pt6_TON: Pt6 in the 1D 10-ring channel of all-silica TON (ZSM-22 / Theta-1).

Reproducible pipeline (run from anywhere, < 1 min on one core):
  1. download the IZA-SC CIF (DLS-optimised SiO2 framework) if it is missing;
  2. conventional Cmcm cell (72 atoms, c = 5.3 A along the channel);
  3. NEP89 variable-cell + position relaxation (calorine CPUNEP, BFGS on a
     FrechetCellFilter, fmax 0.01 eV/A) started from a slightly rattled structure
     (RATTLE, fixed seed); the relaxed cell is cached in framework_nep89.extxyz
     (readable, 1e-8 A rounding) and bit-exactly in framework_nep89.traj, which the
     cached path reads so that the substrate (and the problem hash) is identical to
     a --force rebuild (use --force to recompute). Without the rattle the
     idealised Cmcm IZA model relaxes to a symmetric saddle point with 1/3 of the
     Si-O-Si angles linear,
     ~12 kJ/mol SiO2 above the symmetry-broken minimum and with a ~10 % larger cell;
     the symmetry-broken cell (framework density 19.66 T/1000 A^3) matches the type
     material Theta-1 (FD 19.7);
  4. framework checks (Si-O, O-Si-O, Si-O-Si, volume vs IZA, energy vs alpha-quartz);
  5. 1x1x3 supercell (c = 3 x 5.0 A after relaxation) so that every cell height is >= 12 A;
  6. largest empty point of the channel (grid + Nelder-Mead refinement of the
     distance to the nearest framework atom) = centre of the search region; the
     framework is translated so that this point is the cell centre; the search
     region is a box elongated along the channel axis (REGION_BOX half-widths,
     chosen so that >= 95 % of the box is >= 2.2 A from every framework atom);
  7. write substrate.extxyz, problem.json and build_log.json.

Pt replaces the originally planned Ir: in NEP89 single Ir atoms sink into the
silica walls (Ir-O 1.97 A, Ir-Si 2.40 A, 2.6 eV below an Ir atom in the channel),
so the Ir4@TON putative global minimum was four isolated wall-embedded atoms.
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
from ase import Atoms
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
NAME = "ze_Pt6_TON"
TITLE = "Pt6 @ TON"
CODE = "TON"
METAL, N_METAL = "Pt", 6
PRIMITIVE_C = False          # C-centred cell -> primitive cell (half the atoms)
REPEAT = (1, 1, 3)          # supercell of the relaxed (primitive) cell
REGION_MARGIN = 2.2         # region radius = (largest empty-sphere radius) - margin (as ze_Pt4_CHA)
REGION_BOX = (1.5, 1.0, 3.5)  # Cartesian half-widths of a box region along the channel (None: sphere)
RATTLE = 0.05               # A, Gaussian rattle (seed 0) before the framework relaxation (None: keep symmetry)
BUDGET = 17900  # 3 x median BH hit step (5954, 20 seeds) of campaign pilot-bze2-v4 (BH 14/20, RS 16/20)
E_TOL = 0.01    # GM-basin runs re-relax (fmax 0.002) to E_best; closest isomers: tight minima +0.0130 / +0.0143 eV
DESCRIPTION = (
    'Pt6 cluster in the one-dimensional 10-ring channel (4.6 x 5.7 A) of a frozen all-silica TON '
    '(ZSM-22/Theta-1) framework (NEP89-relaxed from a symmetry-broken start, 1x1x3 supercell, '
    '216 atoms; density within 0.3 % of Theta-1); Pt/ZSM-22 is the classic shape-selective '
    'n-alkane hydroisomerisation catalyst (Martens and co-workers). Framework Al/H+ sites and '
    'framework relaxation are absent, NEP89 binds Pt to siloxane O (Pt-O ~2.2 A) and places this '
    'framework 3.6 kJ/mol SiO2 below alpha-quartz, so the global minimum is a model-PES optimum '
    '(Pt is used because NEP89 spuriously embeds Ir/Rh atoms into silica walls).')


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
    cache_exact = os.path.join(HERE, "framework_nep89.traj")
    log = {}
    if os.path.exists(cache) and not a.force:
        fw = read(cache)
        if os.path.exists(cache_exact):  # restore the exact (unrounded) cell and positions
            ex = read(cache_exact)
            fw.set_cell(ex.cell)
            fw.positions = ex.positions
        fw.calc = None
        old = os.path.join(HERE, "build_log.json")
        if os.path.exists(old):
            log.update(json.load(open(old)).get("relaxation", {}))
        print("using cached relaxed framework", cache)
    else:
        start = unit.copy()
        if RATTLE:
            start.rattle(stdev=RATTLE, seed=0)
        fw, e, conv, nst = relax_cell(start)
        eq, vq = quartz_energy()
        log = dict(rattle=RATTLE, converged=conv, steps=nst, e_per_sio2=round(e / n_t, 5),
                   e_quartz_per_sio2=round(eq, 5),
                   de_vs_quartz_kJmol=round((e / n_t - eq) * 96.485, 2),
                   v_quartz_per_sio2=round(vq, 3))
        fw.info.clear()
        write(cache, fw)
        write(cache_exact, Atoms(fw.numbers, positions=fw.positions, cell=fw.cell, pbc=fw.pbc))
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
