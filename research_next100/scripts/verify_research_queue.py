#!/usr/bin/env python3
"""Verify the frozen next100 Cartesian product before starting any worker."""
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.algorithms import available  # noqa: E402
from gopt.problem import list_benchmarks  # noqa: E402
from gopt.run import parse_seeds  # noqa: E402


def main():
    queue = Path(os.environ["FARM_QUEUE"])
    prefix = os.environ["FARM_PREFIX"]
    benches, algs = list_benchmarks(), available()
    expected = {(b, a, s) for b in benches for a in algs for s in range(100)}
    if len(benches) != 4 or len(algs) != 26:
        raise ValueError(f"Unexpected inventory: {len(benches)} systems, {len(algs)} methods")
    for dirname in ("done", "failed"):
        if list((queue / dirname).glob("*.json")):
            raise ValueError(f"Queue already contains {dirname} tasks")
    if list((queue / "running").rglob("*.json")):
        raise ValueError("Queue already has running tasks")
    seen = set()
    tasks = list((queue / "pending").glob("*.json"))
    for path in tasks:
        with path.open() as f:
            task = json.load(f)
        if not task["name"].startswith(prefix + "-") or task["cwd"] != ROOT:
            raise ValueError(f"Unexpected task identity or working directory: {path}")
        args = shlex.split(task["cmd"])
        if args[:3] != ["python", "-m", "gopt.run"]:
            raise ValueError(f"Unexpected runner in {path}")
        opts = args[3:]
        if len(opts) % 2 != 1 or opts[-1] != "--print":
            # make_tasks appends --time-limit after --print for capped runs
            if len(opts) < 3 or opts[-3] != "--print" or opts[-2] != "--time-limit":
                raise ValueError(f"Unexpected command shape in {path}")
            options = dict(zip(opts[:-3:2], opts[1:-3:2]))
            options["--time-limit"] = opts[-1]
        else:
            options = dict(zip(opts[:-1:2], opts[1:-1:2]))
        if set(options) - {"--bench", "--alg", "--seeds", "--campaign", "--time-limit"}:
            raise ValueError(f"Unexpected runner option in {path}")
        bench, alg = options["--bench"], options["--alg"]
        if options["--campaign"] != "next100":
            raise ValueError(f"Wrong campaign in {path}")
        capped = alg in ("gofee", "surr")
        if ((capped and options.get("--time-limit") != "3600.0")
                or (not capped and "--time-limit" in options)):
            raise ValueError(f"Wrong per-seed time limit in {path}")
        seeds = parse_seeds(options["--seeds"])
        if len(seeds) != (2 if capped else 5) or task.get("timeout") != (9000.0 if capped else 21600.0):
            raise ValueError(f"Wrong seed chunk or task timeout in {path}")
        for seed in seeds:
            key = bench, alg, seed
            if key in seen:
                raise ValueError(f"Duplicate run: {key}")
            seen.add(key)
    if seen != expected:
        raise ValueError(f"Run inventory mismatch: missing={len(expected - seen)}, "
                         f"unexpected={len(seen - expected)}")
    print(f"Verified {len(tasks)} tasks and {len(seen)} distinct benchmark/method/seed runs")


if __name__ == "__main__":
    main()
