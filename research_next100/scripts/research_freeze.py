#!/usr/bin/env python3
"""Seal or verify extra research-campaign inputs omitted from landscape hashes."""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.algorithms import available, get  # noqa: E402
from gopt.nep import NEP89  # noqa: E402
from gopt.problem import Problem, list_benchmarks  # noqa: E402
from gopt.run import _alg_hash  # noqa: E402


def digest(path, algorithm):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def spec_digests(p):
    files = {"problem.json": digest(os.path.join(p.dir, "problem.json"), "sha256")}
    if p.spec.get("substrate_file"):
        name = p.spec["substrate_file"]
        files[name] = digest(os.path.join(p.dir, name), "sha256")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    benchdir = os.environ.get("GOPT_BENCH_DIR")
    if not benchdir:
        raise SystemExit("Set GOPT_BENCH_DIR")
    path = os.path.join(benchdir, "freeze_manifest.json")
    with open(path) as f:
        manifest = json.load(f)
    benches, algs = list_benchmarks(), available()
    if set(benches) != set(manifest["benchmarks"]) or set(algs) != set(manifest["algorithms"]):
        raise ValueError("Current inventory differs from the frozen inventory")
    if digest(NEP89, "sha1") != manifest["nep_model"]["sha1"]:
        raise ValueError("Model SHA-1 differs from the freeze")
    for name, expected in manifest["shared_code"].items():
        if digest(os.path.join(ROOT, name), "sha1") != expected:
            raise ValueError(f"Shared code differs from freeze: {name}")
    for name in algs:
        module = get(name)
        expected = manifest["algorithms"][name]
        if _alg_hash(module) != expected["alg_hash"] or module.DEFAULTS != expected["defaults"]:
            raise ValueError(f"Algorithm differs from freeze: {name}")
        for filename, sha in expected["files"].items():
            if digest(os.path.join(ROOT, filename), "sha1") != sha:
                raise ValueError(f"Algorithm file differs from freeze: {filename}")
    current = {}
    for name in benches:
        p = Problem(name)
        expected = manifest["benchmarks"][name]
        if (p.hash != expected["bench_hash"] or p.budget != expected["budget"]
                or p.e_tol != expected["e_tol"] or p.fmax != expected["fmax"]):
            raise ValueError(f"Benchmark differs from freeze: {name}")
        current[name] = spec_digests(p)
    extra = {"model_sha256": digest(NEP89, "sha256"), "benchmark_files_sha256": current}
    for name in ("protocol.json", "pilot_audit.json"):
        filename = os.path.join(benchdir, name)
        if os.path.isfile(filename):
            extra[f"{name}_sha256"] = digest(filename, "sha256")
    if args.seal:
        if "research_integrity" in manifest:
            raise ValueError("Research freeze is already sealed")
        manifest["research_integrity"] = extra
        temp = path + ".tmp"
        with open(temp, "x") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")
        os.replace(temp, path)
        print(f"Sealed {path}")
    else:
        if manifest.get("research_integrity") != extra:
            raise ValueError("Model or complete benchmark input files differ from the sealed freeze")
        print(f"Verified sealed research freeze: {len(benches)} systems, {len(algs)} algorithms")


if __name__ == "__main__":
    main()
