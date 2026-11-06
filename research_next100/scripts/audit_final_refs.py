#!/usr/bin/env python3
"""Re-evaluate and check forces of the final report's reference structures."""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.context import Context  # noqa: E402
from gopt.problem import Problem  # noqa: E402
from gopt.report.data import penalty  # noqa: E402
from scripts.validate_gm import rebuild  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(os.path.join(args.report, "summary.json")) as f:
        summary = json.load(f)
    checks = {}
    for bench, ref in summary["references"].items():
        p = Problem(bench)
        candidates = sorted(glob.glob(os.path.join(ROOT, "results", "*", bench, "*", "s*.json")))
        source = None
        for path in candidates:
            with open(path) as f:
                record = json.load(f)
            if (record.get("bench_hash") == p.hash and record.get("best_e") == ref["energy"]
                    and penalty(p, path) <= 0.01):
                source = path
                break
        if source is None:
            raise ValueError(f"{bench}: cannot locate the report reference")
        pos, cell, r = rebuild(p, source)
        ctx = Context(p, budget=10)
        energy, forces = ctx.evaluate(pos, cell if p.variable_cell else None)
        force = float(np.linalg.norm(forces[p.free_idx], axis=1).max())
        result = {
            "source": os.path.relpath(source, ROOT), "energy": energy,
            "stored_energy": r["best_e"], "max_free_force_eV_per_A": force,
            "fmax_target_eV_per_A": p.fmax,
        }
        if p.variable_cell:
            result["stress_voigt_eV_per_A3"] = ctx.stress(cell).tolist()
        checks[bench] = result
        if (not np.isfinite(energy) or not np.isfinite(force)
                or abs(energy - r["best_e"]) > 0.002 or force > p.fmax + 0.001):
            raise ValueError(f"{bench}: invalid reference energy or force {force:.6g}")
        print(bench, "reference force", force, "eV/A")
    with open(args.out, "w") as f:
        json.dump(checks, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
