#!/usr/bin/env python3
"""Independently recompute ERT cells and scores from raw formal records."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compute_cell(records, target, budget):
    spent, hits, timeout_hits = 0, 0, 0
    for r in records:
        if r["status"] not in ("ok", "timeout"):
            raise ValueError(f"Unexpected run status {r['status']!r}")
        steps, energies = r["trace_steps"], r["trace_e"]
        if len(steps) != len(energies) or not steps:
            raise ValueError("Invalid trace")
        first = next((s for s, e in zip(steps, energies) if e <= target), None)
        timed = r["status"] == "timeout" or (r.get("stats") or {}).get("timeout", False)
        if first is not None and first <= budget:
            spent += first
            hits += 1
            timeout_hits += int(timed)
        elif timed:
            spent += budget
        else:
            spent += min(budget, r["nsteps"])
    return (spent / hits if hits else math.inf), hits, timeout_hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(ROOT / "benches-eval" / "protocol.json"))
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = Path(args.report)
    with open(args.protocol) as f:
        p = json.load(f)
    with (report / "summary.json").open() as f:
        summary = json.load(f)
    with (report / "metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    cells = {(r["benchmark"], r["algorithm"]): r for r in rows}
    expected_cells = {(b, a) for b in p["benchmarks"] for a in p["algorithms"]}
    if len(cells) != len(rows) or set(cells) != expected_cells:
        raise ValueError("Metric inventory mismatch")
    if summary["seed_range_inclusive"] != p["seed_range_inclusive"]:
        raise ValueError("Seed range mismatch")
    start, stop = p["seed_range_inclusive"]
    calculated, statuses, timeout_hits = {}, Counter(), Counter()
    total_steps = 0
    total_wall = 0.0
    for (bench, alg), row in sorted(cells.items()):
        records = []
        budget = int(row["budget"])
        target = float(row["reference_eV"]) + float(row["tolerance_eV"])
        for seed in range(start, stop + 1):
            path = ROOT / "results" / p["campaign"] / bench / alg / f"s{seed:03d}.json"
            with path.open() as f:
                r = json.load(f)
            if (r["bench"], r["alg"], r["seed"], r["budget"], r["campaign"]) != (
                    bench, alg, seed, budget, p["campaign"]):
                raise ValueError(f"Run identity mismatch: {path}")
            if not math.isfinite(r["wall"]) or r["wall"] < 0:
                raise ValueError(f"Invalid wall time: {path}")
            statuses[r["status"]] += 1
            total_steps += r["nsteps"]
            total_wall += r["wall"]
            records.append(r)
        ert, hits, timed_hits = compute_cell(records, target, budget)
        if int(row["n"]) != len(records) or int(row["hits"]) != hits:
            raise ValueError(f"Hit count mismatch: {bench}/{alg}")
        if not math.isclose(ert, float(row["ert_steps"]), rel_tol=1e-12):
            raise ValueError(f"ERT mismatch: {bench}/{alg}")
        calculated[bench, alg] = ert
        timeout_hits[alg] += timed_hits
    published = {r["algorithm"]: r["score"] for r in summary["ranking"]}
    if set(published) != set(p["algorithms"]):
        raise ValueError("Ranking inventory mismatch")
    for alg in p["algorithms"]:
        score = 0.0
        for bench in p["benchmarks"]:
            best = min(calculated[bench, other] for other in p["algorithms"])
            ert = calculated[bench, alg]
            if not math.isfinite(best) or best <= 0:
                raise ValueError(f"No valid reference ERT for {bench}")
            score += max(0.0, min(1.0, 1 + math.log10(best / ert) / 3)) \
                if math.isfinite(ert) else 0.0
        score *= 100 / len(p["benchmarks"])
        if not math.isclose(score, published[alg], rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError(f"Score mismatch: {alg}")
    if dict(statuses) != summary["statuses"]:
        raise ValueError("Status counts differ from summary")
    result = {
        "runs": sum(statuses.values()), "metric_cells": len(cells),
        "scores": len(published), "statuses": dict(statuses),
        "timeout_hits_by_algorithm": dict(timeout_hits),
        "ionic_steps": total_steps, "summed_wall_hours": total_wall / 3600,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Verified {result['runs']} records, {len(cells)} ERT cells and {len(published)} scores.")


if __name__ == "__main__":
    main()
