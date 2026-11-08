"""Run one algorithm on one benchmark for a list of seeds.

    python -m gopt.run --bench Pt13 --alg bh --seeds 0-9 --campaign main

Each seed is an independent run with the benchmark's ionic-step budget. Results
go to results/<campaign>/<bench>/<alg>/s<seed>.json (skipped if present).
"""
import argparse
import hashlib
import inspect
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback

import numpy as np

from .algorithms import get
from .context import BudgetExhausted, Context
from .nep import NEP89, FastNEP
from .problem import ROOT, Problem


def parse_seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


class _Deadline(BudgetExhausted):
    pass


STALL_S = float(os.environ.get("GOPT_STALL_S", 1800))


class _Stall(BaseException):
    """No ionic step for STALL_S seconds. A BaseException, so that an algorithm's own
    `except Exception` cannot swallow it."""


class _Watchdog:
    """Ends a seed whose algorithm stops making ionic steps (e.g. an endless loop), so that the
    other seeds of a farm task still run. SIGALRM based: active in the main thread only."""

    def __init__(self, ctx, limit=STALL_S, tick=60.0):
        self.ctx, self.limit, self.tick = ctx, limit, min(tick, limit)
        self.on = limit > 0 and threading.current_thread() is threading.main_thread()

    def _check(self, signum, frame):
        n, t = self.ctx.nsteps, time.time()
        if n != self.n:
            self.n, self.t = n, t
        elif t - self.t > self.limit:
            raise _Stall(f"no ionic step for {t - self.t:.0f} s, stalled at step {n}")

    def __enter__(self):
        if self.on:
            self.n, self.t = self.ctx.nsteps, time.time()
            self.old = signal.signal(signal.SIGALRM, self._check)
            signal.setitimer(signal.ITIMER_REAL, self.tick, self.tick)
        return self

    def __exit__(self, *exc):
        if self.on:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.old)
        return False


def _alg_files(mod):
    """The algorithm module plus the private gopt.algorithms helper modules it uses (transitively)."""
    seen, todo = set(), [mod]
    while todo:
        m = todo.pop()
        path = getattr(m, "__file__", None)
        if not path or path in seen:
            continue
        seen.add(path)
        for v in list(vars(m).values()):
            try:
                dm = v if inspect.ismodule(v) else inspect.getmodule(v)
            except Exception:
                continue
            if dm is not None and dm.__name__.startswith("gopt.algorithms."):
                todo.append(dm)
    return sorted(seen)


def _alg_hash(mod):
    # one sha1 over the sorted files, so that a single-file algorithm keeps the hash of its file
    h = hashlib.sha1()
    for path in _alg_files(mod):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


_ALG_HASH = {}


def run_one(problem, alg_key, seed, budget=None, params=None, nep=None, time_limit=None):
    mod = get(alg_key)
    # hash the file once, when the module is first imported: later seeds of the same
    # process keep running that code even if the file is edited meanwhile
    alg_hash = _ALG_HASH.get(mod.__file__)
    if alg_hash is None:
        alg_hash = _ALG_HASH[mod.__file__] = _alg_hash(mod)
    kw = dict(getattr(mod, "DEFAULTS", {}))
    kw.update(params or {})
    ctx = Context(problem, budget, nep)
    if time_limit:
        deadline = time.time() + time_limit
        evaluate = ctx.evaluate

        def timed(pos, cell=None):
            if time.time() > deadline:
                ctx.stats["timeout"] = True
                raise _Deadline()
            return evaluate(pos, cell)
        ctx.evaluate = timed
    rng = np.random.default_rng(seed)
    status, err = "ok", None
    restarts = 0
    try:
        with _Watchdog(ctx):
            while ctx.remaining() > 0:
                before = ctx.nsteps
                mod.run(ctx, rng, **kw)
                if ctx.nsteps == before:
                    break
                restarts += 1
    except _Deadline:
        status = "timeout"
    except BudgetExhausted:
        pass
    except (Exception, _Stall):
        status, err = "error", traceback.format_exc()
    if status == "ok" and ctx.stats.get("timeout"):
        # the algorithm caught the deadline itself
        status = "timeout"
    out = dict(bench=problem.name, alg=alg_key, seed=seed, params=kw, status=status,
               error=err, restarts=restarts, host=socket.gethostname(), model=os.path.basename(NEP89),
               bench_hash=problem.hash, alg_hash=alg_hash)
    out.update(ctx.summary())
    if ctx.best_pos is not None:
        out["best_free_pos"] = np.round(ctx.best_pos[problem.free_idx], 5).tolist()
        out["best_cell"] = np.round(ctx.best_cell, 6).tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--alg", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--params", default="{}", help="JSON dict overriding DEFAULTS")
    ap.add_argument("--campaign", default="dev")
    ap.add_argument("--outroot", default=os.path.join(ROOT, "results"))
    ap.add_argument("--tag", default="", help="suffix for the algorithm directory name")
    ap.add_argument("--time-limit", type=float, help="wall seconds per seed")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--print", action="store_true", help="print a one-line summary per seed")
    argv = list(sys.argv[1:] if argv is None else argv)
    # tags usually start with '-', which argparse would take for an option in "--tag -x"
    for i in range(len(argv) - 1):
        if argv[i] == "--tag":
            argv[i:i + 2] = [f"--tag={argv[i + 1]}"]
            break
    a = ap.parse_args(argv)
    params = json.loads(a.params)
    problem = Problem(a.bench)
    outdir = os.path.join(a.outroot, a.campaign, problem.name, a.alg + a.tag)
    os.makedirs(outdir, exist_ok=True)
    nep = None
    failures = 0
    for seed in parse_seeds(a.seeds):
        path = os.path.join(outdir, f"s{seed:03d}.json")
        if os.path.exists(path) and not a.overwrite:
            continue
        if nep is None:
            pos0, _ = problem.random_structure(np.random.default_rng(12345))
            nep = FastNEP(problem.symbols, problem.cell0, pos0)
        res = run_one(problem, a.alg, seed, a.budget, params, nep, a.time_limit)
        res["tag"] = a.tag
        res["campaign"] = a.campaign
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, path)
        if res["status"] == "error":
            failures += 1
            print(res["error"], file=sys.stderr)
        if a.print:
            print(f"{problem.name} {a.alg}{a.tag} seed={seed} best={res['best_e']:.4f} "
                  f"at step {res['best_step']}/{res['nsteps']} relax={res['n_relax']} "
                  f"wall={res['wall']:.1f}s status={res['status']}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
