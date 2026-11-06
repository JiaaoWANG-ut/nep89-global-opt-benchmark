#!/usr/bin/env python3
"""Generate (and optionally submit) farm tasks for gopt runs.

    python scripts/make_tasks.py --prefix dev-ga --campaign dev-ga --algs ga \
        --benches cl_Pt13,su_Pt8_MgO100 --seeds 0-9 --submit

One task per (bench, alg, chunk of --per-task seeds). Task names are
``<prefix>-<bench>-<alg><tag>-s<first seed>``; wait for them with
``python farm/wait.py '<prefix>-*'``.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.problem import list_benchmarks  # noqa: E402
from gopt.run import parse_seeds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--algs", required=True, help="comma list")
    ap.add_argument("--benches", default="all", help="comma list or 'all'")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--per-task", type=int, default=1, help="seeds per farm task")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--params", help="JSON parameter overrides")
    ap.add_argument("--tag", default="")
    ap.add_argument("--time-limit", type=float, help="wall seconds per seed")
    ap.add_argument("--timeout", type=float, help="farm timeout per task (s)")
    ap.add_argument("--priority", type=int, default=50)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    benches = list_benchmarks() if a.benches == "all" else a.benches.split(",")
    seeds = parse_seeds(a.seeds)
    tasks = []
    for b in benches:
        for alg in a.algs.split(","):
            for k in range(0, len(seeds), a.per_task):
                chunk = seeds[k:k + a.per_task]
                sd = ",".join(map(str, chunk))
                cmd = (f"python -m gopt.run --bench {b} --alg {alg} --seeds {sd} "
                       f"--campaign {a.campaign} --print")
                if a.budget:
                    cmd += f" --budget {a.budget}"
                if a.params:
                    cmd += f" --params {shlex.quote(a.params)}"
                if a.tag:
                    cmd += f" --tag={shlex.quote(a.tag)}"
                if a.time_limit:
                    cmd += f" --time-limit {a.time_limit}"
                t = dict(name=f"{a.prefix}-{b}-{alg}{a.tag}-s{chunk[0]:03d}", cmd=cmd, cwd=ROOT,
                         priority=a.priority)
                if a.timeout:
                    t["timeout"] = a.timeout
                tasks.append(t)
    if not a.submit:
        for t in tasks:
            print(json.dumps(t))
        return
    args = [sys.executable, os.path.join(ROOT, "farm", "submit.py"), "--jsonl", "-"]
    if a.force:
        args.append("--force")
    subprocess.run(args, input="\n".join(json.dumps(t) for t in tasks), text=True, check=True)


if __name__ == "__main__":
    main()
