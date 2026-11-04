#!/usr/bin/env python3
"""Submit the main experiment: every algorithm x every benchmark x N seeds.

    python scripts/launch_main.py --algs all --seeds 0-49 [--submit]

Expensive benchmarks (surface, zeolite) get a higher farm priority and fewer
seeds per task so that the long tasks start first and the tail stays short.
Seeds that already have a result are skipped by gopt.run, so re-running this
script (with --suffix) only fills gaps.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.algorithms import available  # noqa: E402
from gopt.problem import BENCH_DIR, list_benchmarks  # noqa: E402

SURROGATE = {"gofee", "surr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algs", default="all")
    ap.add_argument("--benches", default="all")
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--campaign", default="main")
    ap.add_argument("--suffix", default="", help="task-name suffix for gap-filling resubmissions")
    ap.add_argument("--time-limit", type=float, default=3600, help="wall s per seed, surrogate algorithms")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    algs = available() if a.algs == "all" else a.algs.split(",")
    benches = list_benchmarks() if a.benches == "all" else a.benches.split(",")
    lo, hi = map(int, a.seeds.split("-"))
    nseeds = hi - lo + 1
    groups = {}
    for b in benches:
        cat = json.load(open(os.path.join(BENCH_DIR, b, "problem.json")))["category"]
        groups.setdefault(cat in ("surface", "zeolite"), []).append(b)
    total = 0
    for heavy, bs in sorted(groups.items(), reverse=True):
        for alg in algs:
            sur = alg in SURROGATE
            per = 2 if sur else (5 if heavy else 10)
            prio = (60 if heavy else 62) + (1 if sur else 0)
            tmo = per * a.time_limit + 1800 if sur else (6 if heavy else 4) * 3600
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "make_tasks.py"),
                   "--prefix", f"{a.campaign}{a.suffix}", "--campaign", a.campaign, "--algs", alg,
                   "--benches", ",".join(bs), "--seeds", a.seeds, "--per-task", str(per),
                   "--priority", str(prio), "--timeout", str(int(tmo))]
            if sur:
                cmd += ["--time-limit", str(a.time_limit)]
            if a.submit:
                cmd.append("--submit")
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode:
                print(out.stdout, out.stderr)
                sys.exit(out.returncode)
            n = len(bs) * ((nseeds + per - 1) // per)
            total += n
            print(f"{alg:9s} {'heavy' if heavy else 'light'}  {len(bs):3d} benches  {n:5d} tasks  "
                  f"{out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ''}")
    print(json.dumps(dict(algs=len(algs), benches=len(benches), seeds=nseeds, tasks=total,
                          submitted=a.submit)))


if __name__ == "__main__":
    main()
