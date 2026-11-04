#!/usr/bin/env python3
"""Quick efficiency table for development campaigns.

    python scripts/devstats.py --campaign dev-ga            # your runs
    python scripts/devstats.py --campaign 'dev-*' --alg bh  # reference runs

E_ref per benchmark is the lowest energy found in ANY campaign (all results/*),
so the table reflects the current putative global minimum.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gopt.analysis import best_known, group, load_runs, metrics  # noqa: E402
from gopt.problem import Problem  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="dev*")
    ap.add_argument("--alg", default="*")
    ap.add_argument("--bench", default="*")
    ap.add_argument("--tol", type=float, help="override the benchmark e_tol")
    a = ap.parse_args()
    allruns = load_runs("*", a.bench)
    ref = best_known(allruns)
    runs = load_runs(a.campaign, a.bench, a.alg)
    print(f"{'bench':24s} {'alg':16s} {'n':>3s} {'SR':>5s} {'ERT':>9s} {'medHit':>8s} "
          f"{'medGap':>7s} {'wall/run':>8s} {'ms/st':>6s} err/tmo  E_ref")
    for (b, alg), rs in sorted(group(runs).items()):
        tol = a.tol if a.tol is not None else Problem(b).e_tol
        m = metrics(rs, ref[b][0], tol)
        ert = f"{m['ert']:9.0f}" if m["ert"] != float("inf") else "      inf"
        print(f"{b:24s} {alg:16s} {m['n']:3d} {m['sr']:5.2f} {ert} {m['med_hit']:8.0f} "
              f"{m['med_gap']:7.3f} {m['wall']:8.1f} {m['ms_per_step']:6.2f} "
              f"{m['n_error']}/{m['n_timeout']}  {ref[b][0]:.3f}")


if __name__ == "__main__":
    main()
