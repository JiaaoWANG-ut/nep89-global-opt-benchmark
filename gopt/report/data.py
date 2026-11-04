"""Load results and compute the ranking statistics of the report.

Per benchmark b and algorithm a the expected running time ERT[b, a] (ionic
steps per global-minimum discovery, gopt.analysis) is computed from all seeds.
The per-benchmark score is

    s = clip(1 + log10(ERT_best / ERT) / LOG_RANGE, 0, 1)

(1: most efficient algorithm on b; 0: >= 10**LOG_RANGE times slower or never
successful). An algorithm's overall score is 100 * mean(s) over benchmarks,
i.e. a monotone map of the clipped geometric-mean relative efficiency
GMRE = 10**(LOG_RANGE * (score / 100 - 1)).

E_ref of a benchmark is the lowest energy of any stored run (any campaign) whose
best structure does not lean on the short-range guard or the slab wall (their
energy <= GUARD_TOL). Lower structures in which the guard is active sit in NEP89
short-distance holes that the guard did not close; they are listed in
Study.holes, and their runs are scored by energy like all others.
"""
import fnmatch
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy import stats

from ..analysis import RESULTS
from ..guard import Guard
from ..problem import Problem

DEV = ("cl_Pt13", "cl_Pt7Cu6", "bu_Si8", "bu_Ti4O8", "su_Pt8_MgO100", "ze_Pt4_CHA")
CATS = ("cluster", "surface", "zeolite", "bulk")
LOG_RANGE = 3.0
GUARD_TOL = 0.01  # eV; guard + wall energy above which a structure cannot define E_ref
_KEEP = ("bench", "alg", "seed", "status", "nsteps", "budget", "best_e", "best_step", "wall",
         "n_guard", "n_relax", "n_unique_minima", "n_frozen_fix", "bench_hash", "alg_hash", "tag")


def _read(path):
    try:
        with open(path) as f:
            r = json.load(f)
    except Exception:
        return None
    out = {k: r.get(k) for k in _KEEP}
    if out["status"] == "ok" and (r.get("stats") or {}).get("timeout"):
        out["status"] = "timeout"  # the algorithm caught the wall-time deadline itself
    parts = path.split(os.sep)
    out["campaign"], out["alg_dir"], out["file"] = parts[-4], parts[-2], path
    out["trace_steps"] = np.asarray(r.get("trace_steps") or [], np.int64)
    out["trace_e"] = np.asarray(r.get("trace_e") or [], float)
    return out


def load_all(root=RESULTS, jobs=8):
    """All result records (any campaign) whose benchmark hash is current."""
    files = glob.glob(os.path.join(root, "*", "*", "*", "s*.json"))
    with Pool(jobs) as pool:
        recs = [r for r in pool.map(_read, files, chunksize=64) if r is not None]
    probs = {}
    for b in sorted({r["bench"] for r in recs}):
        try:
            probs[b] = Problem(b)
        except Exception:
            pass
    recs = [r for r in recs if r["bench"] in probs and r["bench_hash"] == probs[r["bench"]].hash]
    return recs, probs


def best_structure(p, path):
    """(positions, cell, record) of the best structure stored in a result file."""
    with open(path) as f:
        r = json.load(f)
    pos = np.zeros((p.n, 3))
    pos[: p.n_sub] = p.sub_pos
    pos[p.free_idx] = np.array(r["best_free_pos"])
    cell = np.array(r["best_cell"]) if r.get("best_cell") else p.cell0
    return pos, cell, r


_GUARDS, _PENALTY = {}, {}


def penalty(p, path):
    """Guard + slab-wall energy (eV) of the best structure stored in a result file;
    inf if the file holds no structure."""
    if p.guard_factor <= 0:
        return 0.0
    key = (p.name, p.hash, path)
    if key not in _PENALTY:
        g = _GUARDS.get(key[:2])
        if g is None:
            g = _GUARDS[key[:2]] = Guard(p, p.guard_factor, p.guard_k)
        try:
            pos, cell, _ = best_structure(p, path)
        except (OSError, KeyError, TypeError, ValueError):
            _PENALTY[key] = np.inf
        else:
            _PENALTY[key] = float(g(pos, cell)[0])
    return _PENALTY[key]


def references(recs, probs, benches):
    """E_ref per benchmark: {b: (energy, record)}, and the lower structures excluded
    because the guard or the wall is active in them: {b: [(record, penalty), ...]}."""
    cand = {}
    for r in recs:
        e = r["best_e"]
        if r["bench"] in benches and e is not None and np.isfinite(e):
            cand.setdefault(r["bench"], []).append(r)
    ref, holes = {}, {}
    for b, rs in cand.items():
        for r in sorted(rs, key=lambda r: (r["best_e"], r["file"])):
            eg = penalty(probs[b], r["file"])
            if eg <= GUARD_TOL:
                ref[b] = (r["best_e"], r)
                break
            if np.isfinite(eg):
                holes.setdefault(b, []).append((r, eg))
    return ref, holes


def _hit(r, target):
    idx = np.flatnonzero(r["trace_e"] <= target)
    return int(r["trace_steps"][idx[0]]) if len(idx) else None


def _charge(r, budget):
    """Steps charged to a run without a hit. A run stopped by the wall-time cap or by an error
    is charged the full budget, so that slow or failing algorithms do not look cheap."""
    if r["status"] in ("timeout", "error"):
        return budget
    return min(budget, r["nsteps"] or budget)


class Study:
    """ERT/SR matrices (benchmarks x algorithms) of one campaign."""

    def __init__(self, recs, probs, campaign="main", algs=None, benches=None):
        self.probs = probs
        runs = [r for r in recs if fnmatch.fnmatch(r["campaign"], campaign)]
        if benches:
            runs = [r for r in runs if r["bench"] in benches]
        self.algs = list(algs) if algs else sorted({r["alg_dir"] for r in runs})
        runs = [r for r in runs if r["alg_dir"] in self.algs]
        ref, holes = references(recs, probs, {r["bench"] for r in runs})
        order = {c: i for i, c in enumerate(CATS)}
        self.benches = sorted({r["bench"] for r in runs} & set(ref),
                              key=lambda b: (order.get(probs[b].category, 9), b))
        self.ref = {b: ref[b] for b in self.benches}
        self.holes = {b: holes[b] for b in self.benches if b in holes}
        self.runs = runs
        nb, na = len(self.benches), len(self.algs)
        full = lambda v=np.nan: np.full((nb, na), v, float)  # noqa: E731
        self.ert, self.sr, self.n, self.n_hit = full(), full(), full(0), full(0)
        self.med_hit, self.med_gap, self.mean_gap = full(), full(), full()
        self.ms_step, self.wall, self.n_err, self.n_tmo = full(), full(), full(0), full(0)
        self.outc, self.hitfrac = {}, {}
        groups = {}
        for r in runs:
            groups.setdefault((r["bench"], r["alg_dir"]), []).append(r)
        bi = {b: i for i, b in enumerate(self.benches)}
        ai = {a: j for j, a in enumerate(self.algs)}
        self._groups = {(bi[b], ai[a]): rs for (b, a), rs in groups.items() if b in bi}
        for (b, a), rs in groups.items():
            if b not in bi:
                continue
            i, j = bi[b], ai[a]
            p = probs[b]
            target = self.ref[b][0] + p.e_tol
            spent, hit, frac = [], [], []
            for r in sorted(rs, key=lambda r: r["seed"]):
                budget = r["budget"] or p.budget
                h = _hit(r, target)
                if h is not None and h <= budget:
                    spent.append(h), hit.append(True), frac.append(h / budget)
                else:
                    spent.append(_charge(r, budget)), hit.append(False)
                    frac.append(np.inf)
            spent, hit = np.array(spent, float), np.array(hit)
            self.outc[i, j] = (spent, hit)
            self.hitfrac[i, j] = np.array(frac)
            n, nh = len(spent), int(hit.sum())
            self.n[i, j], self.n_hit[i, j], self.sr[i, j] = n, nh, nh / n
            self.ert[i, j] = spent.sum() / nh if nh else np.inf
            self.med_hit[i, j] = np.median(spent[hit]) if nh else np.nan
            gaps = np.array([r["best_e"] - self.ref[b][0] if r["best_e"] is not None else np.nan
                             for r in rs], float)
            self.med_gap[i, j], self.mean_gap[i, j] = np.nanmedian(gaps), np.nanmean(gaps)
            wall = np.array([r["wall"] or np.nan for r in rs], float)
            nst = np.array([r["nsteps"] or np.nan for r in rs], float)
            self.wall[i, j] = np.nanmean(wall)
            self.ms_step[i, j] = np.nansum(wall) / np.nansum(nst) * 1e3
            self.n_err[i, j] = sum(r["status"] == "error" for r in rs)
            self.n_tmo[i, j] = sum(r["status"] == "timeout" for r in rs)

    # ---------------------------------------------------------------- subsets
    def ert_at(self, e_tol):
        """ERT matrix for a different absolute success tolerance (eV)."""
        ert = np.full_like(self.ert, np.nan)
        for (i, j), rs in self._groups.items():
            b = self.benches[i]
            target = self.ref[b][0] + e_tol
            spent, nh = 0.0, 0
            for r in rs:
                budget = r["budget"] or self.probs[b].budget
                h = _hit(r, target)
                if h is not None and h <= budget:
                    spent, nh = spent + h, nh + 1
                else:
                    spent += _charge(r, budget)
            ert[i, j] = spent / nh if nh else np.inf
        return ert

    def subset(self, name):
        if name == "all":
            m = [True] * len(self.benches)
        elif name == "test":
            m = [b not in DEV for b in self.benches]
        elif name == "dev":
            m = [b in DEV for b in self.benches]
        elif name in CATS:
            m = [self.probs[b].category == name for b in self.benches]
        else:
            raise ValueError(name)
        return np.array(m, bool)

    def complete(self):
        """Benchmarks run by every algorithm and solved by at least one."""
        return ~np.isnan(self.ert).any(1) & np.isfinite(np.nanmin(self.ert, 1))

    def rows(self, subset):
        return self.subset(subset) & self.complete()

    # ----------------------------------------------------------------- scores
    @staticmethod
    def score_matrix(ert, best=None, upper=1.0):
        e = np.where(np.isnan(ert), np.inf, ert)
        if best is None:
            best = e.min(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(np.isfinite(e) & np.isfinite(best)[:, None], best[:, None] / e, 0.0)
            s = np.clip(1 + np.log10(np.maximum(rel, 1e-300)) / LOG_RANGE, 0, upper)
        s[~np.isfinite(best)] = np.nan
        return s, rel

    def scores(self, subset="test", ert=None):
        rows = self.rows(subset)
        if not rows.any():
            return np.full(len(self.algs), np.nan)
        s, _ = self.score_matrix(self.ert if ert is None else ert)
        return np.nanmean(s[rows], 0) * 100

    @staticmethod
    def gmre(score):
        return 10 ** (LOG_RANGE * (np.asarray(score) / 100 - 1))

    def ranks(self, subset="test"):
        """Per-benchmark ranks by ERT (ties averaged) on the subset's rows."""
        rows = self.rows(subset)
        return np.array([stats.rankdata(self.ert[i]) for i in np.flatnonzero(rows)])

    def friedman(self, subset="test", alpha=0.05):
        R = self.ranks(subset)
        nb, k = R.shape if R.ndim == 2 else (0, len(self.algs))
        if nb < 2 or k < 3:
            return None
        avg = R.mean(0)
        chi2 = 12 * nb / (k * (k + 1)) * (np.sum(avg ** 2) - k * (k + 1) ** 2 / 4)
        p = stats.chi2.sf(chi2, k - 1)
        q = stats.studentized_range.ppf(1 - alpha, k, 1e5) / np.sqrt(2)
        cd = q * np.sqrt(k * (k + 1) / (6 * nb))
        return dict(avg_rank=avg, chi2=chi2, p=p, cd=cd, n=nb, k=k)

    def bootstrap(self, subset="test", nboot=1000, seed=0):
        """Score replicates (nboot x algs) resampling seeds within each cell.

        The per-benchmark reference ERT_best is held at its point estimate (the
        minimum over noisy replicate ERTs would bias every score low), and
        replicate scores are not capped at 1, so the interval reflects the
        sampling uncertainty of each algorithm's own runs.
        """
        rng = np.random.default_rng(seed)
        nb, na = self.ert.shape
        E = np.full((nboot, nb, na), np.nan)
        for (i, j), (sp, ht) in self.outc.items():
            idx = rng.integers(0, len(sp), (nboot, len(sp)))
            num, den = sp[idx].sum(1), ht[idx].sum(1)
            E[:, i, j] = np.where(den > 0, num / np.maximum(den, 1), np.inf)
        rows = self.rows(subset)
        best = np.where(np.isnan(self.ert), np.inf, self.ert).min(1)
        out = np.full((nboot, na), np.nan)
        if rows.any():
            for k in range(nboot):
                s, _ = self.score_matrix(E[k], best=best, upper=None)
                out[k] = np.nanmean(s[rows], 0) * 100
        return out

    def speedup(self, ref_alg="bh", subset="test", ert=None):
        """Geometric-mean ERT ratio ERT_ref / ERT over rows both solved."""
        ert = self.ert if ert is None else ert
        if ref_alg not in self.algs:
            return np.full(len(self.algs), np.nan), np.zeros(len(self.algs), int)
        j0 = self.algs.index(ref_alg)
        rows = self.rows(subset)
        out, cnt = [], []
        for j in range(len(self.algs)):
            m = rows & np.isfinite(ert[:, j]) & np.isfinite(ert[:, j0])
            out.append(np.exp(np.mean(np.log(ert[m, j0] / ert[m, j]))) if m.any() else np.nan)
            cnt.append(int(m.sum()))
        return np.array(out), np.array(cnt)

    def ert_seconds(self):
        return self.ert * self.ms_step / 1e3

    def pooled_sr(self, subset="test"):
        rows = self.rows(subset)
        return self.n_hit[rows].sum(0) / np.maximum(self.n[rows].sum(0), 1)

    def n_solved(self, subset="test"):
        rows = self.rows(subset)
        return (self.n_hit[rows] > 0).sum(0)

    def overhead(self, ref_alg="bh", subset="all"):
        """Geometric-mean wall time per ionic step relative to ref_alg."""
        if ref_alg not in self.algs:
            return np.full(len(self.algs), np.nan)
        j0 = self.algs.index(ref_alg)
        rows = self.rows(subset)
        r = self.ms_step[rows] / self.ms_step[rows, j0][:, None]
        return np.exp(np.nanmean(np.log(r), 0))

    # ------------------------------------------------------------- structures
    def gm_atoms(self, b):
        """Atoms of the putative global minimum of benchmark b."""
        p = self.probs[b]
        pos, cell, r = best_structure(p, self.ref[b][1]["file"])
        at = p.to_atoms(pos, cell)
        at.info["energy"] = r["best_e"]
        at.info["found_by"] = f"{r.get('alg')} seed {r.get('seed')} ({self.ref[b][1]['campaign']})"
        return at
