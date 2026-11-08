#!/usr/bin/env python3
"""Validate the putative global minima: the lowest-energy runs of every benchmark.

    python scripts/validate_gm.py [--top 3] [--benches a,b] [--jobs 8]

For the --top lowest-energy runs of each benchmark (all campaigns, current
benchmark hash) the stored structure is rebuilt and checked:

* dE      recomputed NEP89+guard energy minus the stored best_e (must be ~0;
          a mismatch means the run evaluated something it did not store,
          e.g. moved frozen host atoms)
* dmin    shortest distance involving a search atom / covalent-radius sum
* Eg      energy of the short-range guard in the structure (must be ~0: a minimum
          that leans on the guard sits in an NEP89 short-distance hole)
* comp    connected components of the search atoms (must be 1 if connected)
* region  all search atoms inside the benchmark's sampling region (information
          only: relaxed clusters may legitimately leave the sampling region)
* vpa     volume per atom / spec volume_per_atom (variable-cell benchmarks)

Also compare E/atom of gas-phase clusters with the bulk element/compound by hand:
a cluster below its bulk phase signals an NEP89 artefact (e.g. small carbon clusters).
Exit code 1 if a lowest-energy run (rank 1) fails a check.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.context import Context  # noqa: E402
from gopt.report.data import load_all  # noqa: E402
from gopt.report.gallery import min_dist_ratio  # noqa: E402


def rebuild(p, path):
    with open(path) as f:
        r = json.load(f)
    pos = np.zeros((p.n, 3))
    pos[: p.n_sub] = p.sub_pos
    pos[p.free_idx] = np.array(r["best_free_pos"])
    cell = np.array(r["best_cell"]) if r.get("best_cell") else p.cell0
    return pos, cell, r


def check(p, path, nep_cache):
    pos, cell, r = rebuild(p, path)
    key = p.name
    if key not in nep_cache:
        nep_cache[key] = Context(p, budget=10 ** 9)
    ctx = nep_cache[key]
    e, _ = ctx.evaluate(pos, cell if p.variable_cell else None)
    eg = float(ctx.guard(pos, cell)[0]) if ctx.guard is not None else 0.0
    at = p.to_atoms(pos, cell)
    out = dict(dE=e - r["best_e"], dmin=min_dist_ratio(at, p), eg=eg, comp=p.cluster_components(pos, cell),
               region=all(p.in_region(x, cell) for x in pos[p.search_idx]),
               vpa=(abs(np.linalg.det(cell)) / p.n / p.vpa) if (p.variable_cell and p.vpa) else None,
               frozen_fix=r.get("n_frozen_fix"))
    bad = []
    if abs(out["dE"]) > 2e-3:
        bad.append("energy")
    if out["dmin"] < p.guard_factor + 0.02:
        bad.append("dmin")
    if out["eg"] > 0.01:
        bad.append("guard")
    if p.connected and out["comp"] != 1:
        bad.append("disconnected")
    if out["vpa"] is not None and not 0.6 <= out["vpa"] <= 1.6:
        bad.append("volume")
    out["bad"] = bad
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--benches")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    recs, probs = load_all(jobs=a.jobs)
    sel = a.benches.split(",") if a.benches else sorted(probs)
    by = {}
    for r in recs:
        if r["best_e"] is not None and np.isfinite(r["best_e"]):
            by.setdefault(r["bench"], []).append(r)
    cache, fail = {}, []
    print(f"{'benchmark':22s} rk {'E (eV)':>12s} {'dE':>9s} {'dmin':>5s} {'Eg':>6s} comp reg  vpa   run")
    for b in sel:
        if b not in by:
            print(f"{b:22s} (no runs)")
            continue
        p = probs[b]
        rs = sorted(by[b], key=lambda r: r["best_e"])[: a.top]
        for k, r in enumerate(rs, 1):
            try:
                c = check(p, r["file"], cache)
            except Exception as exc:  # noqa: BLE001
                print(f"{b:22s} {k:2d} error: {exc}")
                if k == 1:
                    fail.append(b)
                continue
            v = f"{c['vpa']:.2f}" if c["vpa"] is not None else "  - "
            flag = ("  <-- " + ",".join(c["bad"])) if c["bad"] else ""
            print(f"{b:22s} {k:2d} {r['best_e']:12.4f} {c['dE']:9.2e} {c['dmin']:5.2f} {c['eg']:6.3f} {c['comp']:4d} "
                  f"{'y' if c['region'] else 'N':>3s} {v:>5s}  {r['campaign']}/{r['alg_dir']}/s{r['seed']:03d}{flag}")
            if k == 1 and c["bad"]:
                fail.append(b)
    if fail:
        print("FAILED (rank-1 structure):", ", ".join(fail))
        sys.exit(1)


if __name__ == "__main__":
    main()
