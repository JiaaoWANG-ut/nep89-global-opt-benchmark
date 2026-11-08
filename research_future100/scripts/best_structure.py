#!/usr/bin/env python3
"""Write the lowest-energy structure(s) found for a benchmark.

    python scripts/best_structure.py --bench bu_Si8 [--campaign 'dev*'] [--alg bh] [--top 3] -o best.extxyz

Prints energy, source run, nearest-neighbour distances and coordination of the
search atoms, so that the result can be checked for unphysical features.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ase.io import write  # noqa: E402

from gopt.problem import ROOT, Problem, min_image_distances  # noqa: E402


def structure(p, r):
    pos = np.zeros((p.n, 3))
    pos[: p.n_sub] = p.sub_pos
    pos[p.free_idx] = np.array(r["best_free_pos"])
    cell = np.array(r["best_cell"]) if r.get("best_cell") else p.cell0
    return p.to_atoms(pos, cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--campaign", default="*")
    ap.add_argument("--alg", default="*")
    ap.add_argument("--top", type=int, default=1, help="distinct energies (>10 meV apart)")
    ap.add_argument("-o", "--output")
    a = ap.parse_args()
    p = Problem(a.bench)
    runs = []
    for f in glob.glob(os.path.join(ROOT, "results", a.campaign, a.bench, a.alg, "s*.json")):
        r = json.load(open(f))
        if r.get("bench_hash") == p.hash and r.get("best_free_pos") is not None:
            r["file"] = os.path.relpath(f, ROOT)
            runs.append(r)
    runs.sort(key=lambda r: r["best_e"])
    picked = []
    for r in runs:
        if all(abs(r["best_e"] - q["best_e"]) > 0.01 for q in picked):
            picked.append(r)
        if len(picked) == a.top:
            break
    frames = []
    for r in picked:
        at = structure(p, r)
        s = p.search_idx
        d = min_image_distances(at.positions[s], at.positions, at.cell.array, p.pbc)
        d[np.arange(len(s)), s] = np.inf
        coord = (d < 1.25 * (p.rcov[s][:, None] + p.rcov[None, :])).sum(axis=1)
        vol = abs(np.linalg.det(at.cell.array))
        print(f"E={r['best_e']:.4f} eV ({r['best_e'] / p.n_search:.4f}/search atom) {r['file']}")
        print(f"   min dist {d.min():.3f} A, NN dist mean {d.min(axis=1).mean():.3f} A, "
              f"coordination {np.bincount(coord).tolist()}"
              + (f", vol/atom {vol / p.n:.2f} A^3, cellpar {np.round(at.cell.cellpar(), 2).tolist()}"
                 if p.variable_cell else ""))
        at.info["energy"] = r["best_e"]
        frames.append(at)
    if a.output and frames:
        write(a.output, frames)


if __name__ == "__main__":
    main()
