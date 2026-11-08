#!/usr/bin/env python3
"""Verify an isolated farm queue against a predeclared campaign configuration."""
import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gopt.algorithms import available  # noqa: E402
from gopt.problem import list_benchmarks  # noqa: E402
from gopt.run import parse_seeds  # noqa: E402


def verify(queue, config, benches, algorithms):
    queue = Path(queue)
    if sorted(benches) != sorted(config["benchmarks"]):
        raise ValueError("Benchmark inventory differs from the predeclared protocol")
    if sorted(algorithms) != sorted(config["algorithms"]):
        raise ValueError("Algorithm inventory differs from the predeclared protocol")
    start, stop = config["seed_range_inclusive"]
    if start < 0 or stop < start:
        raise ValueError("Invalid seed range")
    capped = set(config["capped_algorithms"])
    if not capped <= set(algorithms):
        raise ValueError("Unknown capped algorithm")
    expected = {(b, a, s) for b in benches for a in algorithms
                for s in range(start, stop + 1)}
    for dirname in ("done", "failed", "running"):
        if list((queue / dirname).rglob("*.json")):
            raise ValueError(f"Queue already contains {dirname} tasks")
    if list((queue / "control").iterdir()):
        raise ValueError("Queue control directory is not empty")
    seen = set()
    paths = sorted((queue / "pending").glob("*.json"))
    for path in paths:
        with path.open() as f:
            task = json.load(f)
        if not task["name"].startswith(config["prefix"] + "-") or task["cwd"] != str(ROOT):
            raise ValueError(f"Unexpected task identity or working directory: {path}")
        allowed_task_keys = {"name", "cmd", "cwd", "timeout", "priority", "submitted", "ncores"}
        if set(task) - allowed_task_keys or task.get("ncores", 1) != 1:
            raise ValueError(f"Unexpected task resource or execution settings: {path}")
        args = shlex.split(task["cmd"])
        if args[:3] != ["python", "-m", "gopt.run"]:
            raise ValueError(f"Unexpected runner: {path}")
        opts = args[3:]
        if opts.count("--print") != 1:
            raise ValueError(f"Expected --print exactly once: {path}")
        opts.remove("--print")
        if len(opts) % 2:
            raise ValueError(f"Malformed options: {path}")
        pairs = list(zip(opts[::2], opts[1::2]))
        options = dict(pairs)
        if len(options) != len(pairs):
            raise ValueError(f"Duplicate option: {path}")
        alg = options.get("--alg")
        is_capped = alg in capped
        wanted = {"--bench", "--alg", "--seeds", "--campaign"}
        if is_capped:
            wanted.add("--time-limit")
        if set(options) != wanted or options["--campaign"] != config["campaign"]:
            raise ValueError(f"Wrong campaign or runner options: {path}")
        if is_capped and float(options["--time-limit"]) != config["time_limit_seconds"]:
            raise ValueError(f"Wrong per-seed time limit: {path}")
        group = "capped" if is_capped else "other"
        if task.get("timeout") != config[f"{group}_task_timeout_seconds"]:
            raise ValueError(f"Wrong task timeout: {path}")
        seeds = parse_seeds(options["--seeds"])
        chunk_size = config[f"{group}_seeds_per_task"]
        if chunk_size < 1:
            raise ValueError("Invalid chunk size")
        expected_chunks = [list(range(s, min(s + chunk_size, stop + 1)))
                           for s in range(start, stop + 1, chunk_size)]
        if seeds not in expected_chunks:
            raise ValueError(f"Unexpected seed chunk: {path}")
        for seed in seeds:
            key = options["--bench"], alg, seed
            if key in seen:
                raise ValueError(f"Duplicate run: {key}")
            seen.add(key)
    if seen != expected:
        raise ValueError(f"Run inventory mismatch: missing={len(expected - seen)}, "
                         f"unexpected={len(seen - expected)}")
    return len(paths), len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    with open(args.protocol) as f:
        config = json.load(f)
    tasks, runs = verify(args.queue, config, list_benchmarks(), available())
    print(f"Verified {tasks} tasks and {runs} distinct benchmark/method/seed runs")


if __name__ == "__main__":
    main()
