#!/usr/bin/env python3
"""Record the frozen state of the benchmark before the main experiment.

    python scripts/freeze_manifest.py [--out docs/freeze_manifest.json] [--frozen-at ISO]

Writes the NEP89 model checksum, a checksum of every shared-code file, every
algorithm's alg_hash (main file + private helper modules, as stored in each
run), its DEFAULTS, and every benchmark's hash, budget and tolerance. A run of
the main experiment is reproducible from the files listed here.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.algorithms import available, get  # noqa: E402
from gopt.nep import NEP89  # noqa: E402
from gopt.problem import Problem, list_benchmarks  # noqa: E402
from gopt.run import _alg_files, _alg_hash  # noqa: E402


def sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "freeze_manifest.json"))
    ap.add_argument("--frozen-at", help="keep this freeze time (ISO) when refreshing the checksums of "
                                        "files that do not affect runs, e.g. the report code")
    a = ap.parse_args()
    shared = {}
    for d in ("gopt", "gopt/report", "farm", "scripts"):
        for fn in sorted(os.listdir(os.path.join(ROOT, d))):
            if fn.endswith(".py"):
                shared[f"{d}/{fn}"] = sha1(os.path.join(ROOT, d, fn))
    algs = {}
    for name in available():
        mod = get(name)
        algs[name] = dict(alg_hash=_alg_hash(mod),
                          files={os.path.relpath(f, ROOT): sha1(f) for f in _alg_files(mod)},
                          defaults=getattr(mod, "DEFAULTS", {}))
    benches = {}
    for b in list_benchmarks():
        p = Problem(b)
        benches[b] = dict(bench_hash=p.hash, category=p.category, n_search=p.n_search, n_atoms=p.n,
                          budget=p.budget, e_tol=p.e_tol, fmax=p.fmax, guard_factor=p.guard_factor,
                          guard_pairs={f"{x}-{y}": v for (x, y), v in p.guard_pairs.items()},
                          z_wall_height=round(p.z_wall["height"], 3) if getattr(p, "z_wall", None) else None)
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    out = dict(frozen_at=a.frozen_at or now, written_at=now,
               nep_model=dict(file=os.path.basename(NEP89), sha1=sha1(NEP89)),
               shared_code=shared, algorithms=algs, benchmarks=benches)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"wrote {a.out}: {len(algs)} algorithms, {len(benches)} benchmarks, {len(shared)} shared files")


if __name__ == "__main__":
    main()
