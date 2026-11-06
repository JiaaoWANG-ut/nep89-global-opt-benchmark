"""Result loading and efficiency metrics.

A run *hits* the global minimum at the first ionic step where its best-so-far
energy is <= E_ref + tol (E_ref: lowest energy found for the benchmark by any
run). Efficiency is measured with the expected running time

    ERT = (sum over runs of steps spent until hit, or the steps used) / #hits

i.e. the expected number of ionic steps per discovery of the global minimum
when unsuccessful runs are restarted; efficiency = 1 / ERT. An unsuccessful
run normally uses its full budget; runs stopped early (time limit, error) are
charged the steps they actually used, as in COCO's restart semantics.
"""
import glob
import json
import os

import numpy as np

from .problem import ROOT, Problem

RESULTS = os.path.join(ROOT, "results")
_SKIP = ("best_free_pos", "best_cell", "error")


def load_runs(campaigns="*", bench="*", alg="*", root=RESULTS, check_hash=True, keep_pos=False):
    """Load result JSONs; drop runs whose benchmark definition has since changed."""
    if isinstance(campaigns, str):
        campaigns = [campaigns]
    files = []
    for c in campaigns:
        files.extend(glob.glob(os.path.join(root, c, bench, alg, "s*.json")))
    hashes = {}
    out = []
    for f in sorted(files):
        try:
            with open(f) as fh:
                r = json.load(fh)
        except Exception:
            continue
        if check_hash:
            b = r["bench"]
            if b not in hashes:
                try:
                    hashes[b] = Problem(b).hash
                except Exception:
                    hashes[b] = None
            if r.get("bench_hash") != hashes[b]:
                continue
        if not keep_pos:
            for k in _SKIP:
                r.pop(k, None)
        r["campaign"] = f.split(os.sep)[-4]
        r["alg_dir"] = f.split(os.sep)[-2]
        out.append(r)
    return out


def best_known(runs):
    """bench -> (lowest energy, (campaign, alg, seed))."""
    best = {}
    for r in runs:
        e = r.get("best_e")
        if e is None or not np.isfinite(e):
            continue
        b = r["bench"]
        if b not in best or e < best[b][0]:
            best[b] = (e, (r["campaign"], r["alg_dir"], r["seed"]))
    return best


def hit_step(run, target):
    for s, e in zip(run.get("trace_steps", []), run.get("trace_e", [])):
        if e <= target:
            return int(s)
    return None


def outcomes(runs, target, budget=None):
    """Per-run (steps charged, hit?) arrays for the ERT of one benchmark/algorithm."""
    spent = np.zeros(len(runs))
    hit = np.zeros(len(runs), bool)
    for i, r in enumerate(runs):
        b = budget or r["budget"]
        h = hit_step(r, target)
        if h is not None and h <= b:
            spent[i], hit[i] = h, True
        elif r.get("status") in ("timeout", "error") or (r.get("stats") or {}).get("timeout"):
            # stopped by the wall-time cap or an error: charge the full budget, so that slow or
            # failing algorithms do not look cheap
            spent[i] = b
        else:
            spent[i] = min(b, r.get("nsteps") or b)
    return spent, hit


def metrics(runs, e_ref, tol, budget=None):
    """Success statistics of a set of runs (same benchmark and algorithm)."""
    n = len(runs)
    if n == 0:
        return None
    spent_i, hit_i = outcomes(runs, e_ref + tol, budget)
    hits = spent_i[hit_i].tolist()
    spent = spent_i.sum()
    ns = len(hits)
    gaps = np.array([r["best_e"] - e_ref for r in runs], float)
    walls = np.array([r.get("wall", np.nan) for r in runs], float)
    steps = np.array([r.get("nsteps", np.nan) for r in runs], float)
    return dict(
        n=n, n_hit=ns, sr=ns / n, ert=(spent / ns) if ns else np.inf,
        med_hit=float(np.median(hits)) if hits else np.nan,
        mean_gap=float(np.mean(gaps)), med_gap=float(np.median(gaps)),
        wall=float(np.nanmean(walls)), ms_per_step=float(np.nansum(walls) / np.nansum(steps) * 1e3),
        n_error=sum(r["status"] == "error" for r in runs),
        n_timeout=sum(r["status"] == "timeout" for r in runs))


def group(runs):
    out = {}
    for r in runs:
        out.setdefault((r["bench"], r["alg_dir"]), []).append(r)
    return out
