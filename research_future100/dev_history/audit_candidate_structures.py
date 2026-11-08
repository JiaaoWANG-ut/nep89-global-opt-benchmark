#!/usr/bin/env python3
"""Re-evaluate saved candidate-hit structures on compute nodes."""
import argparse
import glob
import json
import multiprocessing
import os
import sys
from collections import Counter

import numpy as np
from ase.geometry import find_mic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.ops import center_of_search  # noqa: E402
from gopt.problem import Problem  # noqa: E402
from scripts.validate_gm import check, rebuild  # noqa: E402

PROBS, CONTEXTS = {}, {}


def inspect(item):
    bench, path = item
    if bench not in PROBS:
        PROBS[bench] = Problem(bench)
    p = PROBS[bench]
    validation = check(p, path, CONTEXTS)
    pos, cell, _ = rebuild(p, path)
    if (not np.isfinite(pos).all() or not np.isfinite(cell).all()
            or not all(np.isfinite(validation[key]) for key in ("dE", "eg", "dmin"))):
        raise ValueError(f"Nonfinite geometry or validation result: {path}")
    result = {
        "benchmark": bench, "path": os.path.relpath(path, ROOT),
        "bad": validation["bad"], "region": validation["region"],
    }
    if p.z_wall:
        wall = p.z_wall
        z = pos[p.search_idx] @ wall["normal"]
        delta = (z - wall["top"] + 0.5 * wall["period"]) % wall["period"] \
            - 0.5 * wall["period"]
        result["minimum_height_above_slab_top_A"] = float(np.min(delta))
    if p.category == "zeolite" and p.region.get("type") == "sphere":
        center = np.asarray(p.region["center"])
        displacement, _ = find_mic(center_of_search(p, pos, cell) - center, cell, pbc=p.pbc)
        result["center_outside_initial_sphere"] = bool(
            np.linalg.norm(displacement) > float(p.region["radius"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--reference-campaign", required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.jobs < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("jobs and optional limit must be positive")
    if args.campaign == args.reference_campaign:
        raise ValueError("Formal and pilot campaigns must be distinct")
    with open(os.path.join(args.report, "summary.json")) as f:
        summary = json.load(f)
    candidates = []
    for bench, ref in summary["references"].items():
        p = Problem(bench)
        for campaign in (args.campaign, args.reference_campaign):
            pattern = os.path.join(ROOT, "results", campaign, bench, "*", "s*.json")
            for path in sorted(glob.glob(pattern)):
                with open(path) as f:
                    r = json.load(f)
                if r.get("bench_hash") != p.hash:
                    raise ValueError(f"Stale candidate: {path}")
                if r["best_e"] <= ref["energy"] + p.e_tol:
                    candidates.append((bench, path))
    if args.limit is not None:
        candidates = candidates[:args.limit]
    if not candidates:
        raise ValueError("No saved candidate-hit structures found")
    counts, bad, outside = Counter(), [], Counter()
    minimum_height = None
    with multiprocessing.Pool(args.jobs) as pool:
        for item in pool.imap_unordered(inspect, candidates, chunksize=8):
            counts[item["benchmark"]] += 1
            if item["bad"]:
                bad.append(item)
            if not item["region"]:
                outside[item["benchmark"] + ":atoms_outside_initial_region"] += 1
            if item.get("center_outside_initial_sphere"):
                outside[item["benchmark"] + ":center_outside_initial_sphere"] += 1
            height = item.get("minimum_height_above_slab_top_A")
            if height is not None:
                minimum_height = height if minimum_height is None else min(minimum_height, height)
    report = {
        "candidate_count": len(candidates), "counts_by_system": dict(counts),
        "failed_checks": sorted(bad, key=lambda x: x["path"]),
        "outside_initial_region": dict(outside),
        "minimum_surface_height_above_slab_top_A": minimum_height,
        "scope": "Saved best structures only; intermediate trace geometries were not saved.",
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "failed_checks"}, indent=2))
    if bad:
        raise SystemExit(f"{len(bad)} candidate structures failed checks; inspect {args.out}")


if __name__ == "__main__":
    main()
