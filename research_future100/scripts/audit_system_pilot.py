#!/usr/bin/env python3
"""Audit every stored structure in the predeclared new-system pilot."""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.algorithms import get  # noqa: E402
from gopt.nep import NEP89  # noqa: E402
from gopt.problem import Problem, list_benchmarks, min_image_distances  # noqa: E402
from gopt.run import _alg_hash  # noqa: E402
from scripts.extended100_report import audit_record  # noqa: E402
from scripts.validate_gm import check, rebuild  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    benches = list_benchmarks()
    algs = ("bh", "surr")
    seeds = range(1000, 1005)
    if len(benches) != 4:
        raise ValueError(f"Expected four pilot systems, found {len(benches)}")
    probs = {b: Problem(b) for b in benches}
    manifest = {
        "benchmarks": {b: {"bench_hash": p.hash, "budget": p.budget}
                       for b, p in probs.items()},
        "algorithms": {a: {"alg_hash": _alg_hash(get(a)), "defaults": get(a).DEFAULTS}
                       for a in algs},
        "nep_model": {"file": os.path.basename(NEP89)},
    }
    results = {}
    cache = {}
    for bench, p in probs.items():
        records = []
        for alg in algs:
            for seed in seeds:
                path = os.path.join(ROOT, "results", "nextsystemspilot", bench,
                                    alg, f"s{seed:03d}.json")
                with open(path) as f:
                    record = json.load(f)
                audit_record(record, bench, alg, seed, manifest)
                validation = check(p, path, cache)
                pos, cell, _ = rebuild(p, path)
                _, forces = cache[bench].evaluate(pos, cell if p.variable_cell else None)
                item = {
                    "algorithm": alg, "seed": seed, "status": record["status"],
                    "energy": record["best_e"], "validation": validation,
                    "max_free_force_eV_per_A": float(
                        np.linalg.norm(forces[p.free_idx], axis=1).max()),
                }
                if p.n_sub:
                    d = min_image_distances(pos[p.search_idx], pos[:p.n_sub], cell, p.pbc)
                    item["minimum_search_host_distance"] = float(np.min(d))
                if p.z_wall:
                    wall = p.z_wall
                    z = pos[p.search_idx] @ wall["normal"]
                    dz = (z - wall["top"] + 0.5 * wall["period"]) % wall["period"] \
                        - 0.5 * wall["period"]
                    item["height_relative_to_slab_top"] = [float(np.min(dz)), float(np.max(dz))]
                records.append(item)
        guard_free = [r for r in records if r["validation"]["eg"] <= 0.01]
        if not guard_free:
            raise ValueError(f"{bench}: no guard-free saved pilot structure")
        ref = min(guard_free, key=lambda r: r["energy"])
        hits = [r for r in records if r["energy"] <= ref["energy"] + p.e_tol]
        active = [r for r in hits if r["validation"]["eg"] > 0.01]
        results[bench] = {
            "hash": p.hash, "reference": ref,
            "statuses": dict(Counter(r["status"] for r in records)),
            "candidate_hit_count": len(hits), "guard_active_candidate_hits": len(active),
            "all_stored_structures": records,
        }
        if (ref["validation"]["bad"] or active
                or ref["max_free_force_eV_per_A"] > p.fmax + 0.001):
            raise ValueError(f"{bench}: reference or candidate hit failed checks: "
                             f"{ref['validation']['bad']}, active={len(active)}, "
                             f"max_force={ref['max_free_force_eV_per_A']}")
        print(bench, "E_ref", ref["energy"], "reference checks", ref["validation"],
              "candidate hits", len(hits), "active", len(active))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
