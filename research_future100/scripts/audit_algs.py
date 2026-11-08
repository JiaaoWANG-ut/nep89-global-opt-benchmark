#!/usr/bin/env python3
"""Static fairness audit of algorithm modules (and the private helpers they import).

    python scripts/audit_algs.py [key ...]

Flags lines that may violate the rules in docs/DEVELOPER_GUIDE.md: reading
files/results, benchmark-name branches, direct calculator access, private ctx
attributes, global numpy randomness, tighter-than-default fmax literals. Every
flag must be checked by hand; most are harmless.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALG = os.path.join(ROOT, "gopt", "algorithms")
BENCH = os.path.join(ROOT, "benchmarks")

PATTERNS = [
    ("file-io", r"\bopen\(|json\.load|np\.load\b|np\.loadtxt|pickle|os\.listdir|glob\.|Path\("),
    ("results", r"results|E_ref|e_ref\b|best_known|reference"),
    ("bench-name", r"\.name\b|['\"](cl|su|ze|bu)_"),
    ("calculator", r"calorine|CPUNEP|FastNEP|nep89|\.calc\b|get_potential_energy|get_forces"),
    ("ctx-private", r"ctx\._|\bctx\.(p\._|_)"),
    ("global-rng", r"np\.random\.(rand|randn|random|choice|uniform|normal|seed|shuffle|permutation|randint)\b|"
                   r"(^|[^.\w])random\.(random|choice|uniform|seed|shuffle|randint)\b"),
    ("time", r"\btime\.(time|perf_counter|monotonic)\b"),
    ("e_tol", r"\be_tol\b"),
    ("env", r"os\.environ|sys\.argv"),
]


def bench_names():
    return [d for d in os.listdir(BENCH) if os.path.isdir(os.path.join(BENCH, d)) and d[:3] in
            ("cl_", "su_", "ze_", "bu_")]


def helper_closure(key):
    seen, todo = [], [key]
    while todo:
        k = todo.pop()
        if k in seen:
            continue
        seen.append(k)
        f = os.path.join(ALG, k + ".py")
        if not os.path.exists(f):
            continue
        tree = ast.parse(open(f).read())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom):
                mods = ([node.module.split(".")[-1]] if node.module else []) + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                mods = [a.name.split(".")[-1] for a in node.names]
            for m in mods:
                if m.startswith("_") and os.path.exists(os.path.join(ALG, m + ".py")):
                    todo.append(m)
    return seen


def fmax_flags(src, default=0.05):
    out = []
    for i, line in enumerate(src.splitlines(), 1):
        for m in re.finditer(r"fmax[\w\"']*\s*[:=]\s*([0-9.eE-]+)", line):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if v < default - 1e-12:
                out.append((i, f"fmax literal {v} < {default}", line.strip()))
    return out


def audit(key, names):
    flags = []
    for mod in helper_closure(key):
        f = os.path.join(ALG, mod + ".py")
        if not os.path.exists(f):
            flags.append((mod, 0, "missing", ""))
            continue
        src = open(f).read()
        lines = src.splitlines()
        for i, line in enumerate(lines, 1):
            code = line.split("#", 1)[0]
            if not code.strip() or code.strip().startswith(('"', "'")):
                continue
            for tag, pat in PATTERNS:
                if re.search(pat, code):
                    flags.append((mod, i, tag, line.strip()))
            for b in names:
                if b in code:
                    flags.append((mod, i, "bench-literal", line.strip()))
        for i, msg, line in fmax_flags(src):
            flags.append((mod, i, "fmax", line))
        tree = ast.parse(src)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])
        ext = imports - {"numpy", "scipy", "math", "itertools", "collections", "functools", "heapq",
                         "dataclasses", "typing", "time", "warnings", "copy", "bisect", "operator",
                         "gopt", "ase", "enum", "random", "sys", "os", "re", "json"}
        if ext:
            flags.append((mod, 0, "imports", ", ".join(sorted(ext))))
    return flags


def main():
    sys.path.insert(0, ROOT)
    from gopt.algorithms import available
    keys = sys.argv[1:] or available()
    names = bench_names()
    for k in keys:
        fl = audit(k, names)
        mods = helper_closure(k)
        print(f"== {k}  ({', '.join(mods)})  {len(fl)} flag(s)")
        for mod, i, tag, line in fl:
            print(f"   {tag:13s} {mod}:{i}  {line[:110]}")


if __name__ == "__main__":
    main()
