#!/usr/bin/env python3
"""Build the ranking / efficiency web report (static site).

    python scripts/report.py --campaign main --out site
    python scripts/report.py --campaign dev-ref --primary all --boot 100 --out /tmp/site   # quick test

Algorithm names, types and descriptions come from the module attributes (LABEL, FAMILY,
REFERENCE, DESCRIPTION), overridden by docs/algorithms.json (adds the Chinese text).
"""
import argparse
import csv
import datetime
import importlib
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ase.io import write  # noqa: E402

from gopt.report import gallery, plots  # noqa: E402
from gopt.report.data import CATS, DEV, Study, load_all  # noqa: E402
from gopt.report.html import page  # noqa: E402

BASELINES = ("rs", "bh", "mh", "ga", "pso", "de", "sa", "ptbh", "ssw", "abc", "gofee", "uspex")
CAT_ZH = {"cluster": "团簇", "surface": "表面", "zeolite": "分子筛", "bulk": "体相"}


def load_meta(algs):
    """Display metadata: docs/algorithms.json overrides the module attributes."""
    path = os.path.join(ROOT, "docs", "algorithms.json")
    meta = json.load(open(path)) if os.path.exists(path) else {}
    out = {}
    for k in algs:
        base = k.split("@")[0]
        try:
            mod = importlib.import_module(f"gopt.algorithms.{base}")
        except Exception:
            mod = None
        fam = getattr(mod, "FAMILY", None)
        m = meta.get(k, {})
        en = m.get("en", " ".join(str(getattr(mod, "DESCRIPTION", "")).split()))
        out[k] = dict(name=m.get("name", getattr(mod, "LABEL", k)), short=m.get("short", k),
                      type=m.get("type", fam if fam in ("baseline", "innovative")
                                 else ("baseline" if base in BASELINES else "innovative")),
                      desc=(en, m.get("zh", en)), ref=m.get("ref", getattr(mod, "REFERENCE", "")))
    return out


def findings(st, R, A, primary):
    """Auto-generated key findings (en, zh)."""
    en, zh = [], []
    if not A or not np.isfinite(A[0]["score_t"]):
        return en, zh
    top = A[:3]
    en.append(f"<b>{top[0]['name']}</b> ranks first (score {top[0]['score_t']:.1f}, 95% CI "
              f"[{top[0]['ci_lo']:.1f}, {top[0]['ci_hi']:.1f}], first in {top[0]['p_best']:.0%} of bootstrap "
              f"replicates), followed by " + " and ".join(f"{a['name']} ({a['score_t']:.1f})" for a in top[1:]) + ".")
    zh.append(f"<b>{top[0]['name']}</b> 排名第一（得分 {top[0]['score_t']:.1f}，95% 置信区间 "
              f"[{top[0]['ci_lo']:.1f}, {top[0]['ci_hi']:.1f}]，在 {top[0]['p_best']:.0%} 的 bootstrap 样本中第一），其后为 "
              + "、".join(f"{a['name']}（{a['score_t']:.1f}）" for a in top[1:]) + "。")
    new = [a for a in A if a["type"] == "innovative"]
    base = [a for a in A if a["type"] == "baseline"]
    if new and base:
        bn, bb = new[0], base[0]
        sp, cnt = st.speedup(bb["key"], primary)
        f = sp[st.algs.index(bn["key"])]
        if np.isfinite(f):
            n_both = cnt[st.algs.index(bn["key"])]
            rat_en = f"{f:.2f}&times; fewer" if f >= 1 else f"{1 / f:.2f}&times; more"
            rat_zh = f"为后者的 1/{f:.2f}" if f >= 1 else f"为后者的 {1 / f:.2f} 倍"
            en.append(f"Best new algorithm <b>{bn['name']}</b> vs best baseline <b>{bb['name']}</b>: "
                      f"{rat_en} ionic steps per global-minimum discovery "
                      f"(geometric mean over the {n_both} benchmarks solved by both).")
            zh.append(f"最佳创新算法 <b>{bn['name']}</b> 对比最佳主流算法 <b>{bb['name']}</b>：每发现一次全局最优所需离子步"
                      f"{rat_zh}（两者都解出的 {n_both} 个基准上的几何平均）。")
        nbeat = sum(a["score_t"] > bb["score_t"] for a in new)
        en.append(f"{nbeat} of {len(new)} new algorithms score higher than every baseline.")
        zh.append(f"{len(new)} 个创新算法中有 {nbeat} 个得分高于所有主流算法。")
    if "bh" in st.algs and A[0]["key"] != "bh" and np.isfinite(A[0]["speedup"]):
        b = next(a for a in A if a["key"] == "bh")
        en.append(f"Basin hopping (literature settings) ranks {b['rank']} of {len(A)}; the top algorithm is "
                  f"{A[0]['speedup']:.2f}&times; as efficient as BH (geometric mean of the ERT ratio over the "
                  f"benchmarks solved by both).")
        zh.append(f"盆地跳跃（文献参数）排名第 {b['rank']}/{len(A)}；第一名算法的效率是 BH 的 {A[0]['speedup']:.2f} 倍"
                  f"（两者都解出的基准上 ERT 比值的几何平均）。")
    wins_en, wins_zh = [], []
    for c in CATS:
        if st.rows(c).any():
            s = st.scores(c)
            j = int(np.nanargmax(s))
            nm = R["meta"][st.algs[j]]["name"]
            wins_en.append(f"{c}s &ndash; {nm} ({s[j]:.0f})")
            wins_zh.append(f"{CAT_ZH[c]}——{nm}（{s[j]:.0f}）")
    if wins_en:
        en.append("Category winners: " + "; ".join(wins_en) + ".")
        zh.append("各类别第一：" + "；".join(wins_zh) + "。")
    rob = max(A, key=lambda a: a["pooled_sr"])
    en.append(f"Most robust: <b>{rob['name']}</b> found the global minimum in {rob['pooled_sr']:.0%} of all "
              f"runs and on {rob['solved']}/{R['nt_rows']} benchmarks.")
    zh.append(f"最稳健：<b>{rob['name']}</b> 在 {rob['pooled_sr']:.0%} 的运行中找到全局最优，解出 {rob['solved']}/{R['nt_rows']} 个基准。")
    w = max(A, key=lambda a: a["score_wall"])
    en.append(f"Measured in wall-clock time instead of ionic steps, the best algorithm is <b>{w['name']}</b> "
              f"(score {w['score_wall']:.1f}).")
    zh.append(f"若以墙钟时间而非离子步计，最佳算法为 <b>{w['name']}</b>（得分 {w['score_wall']:.1f}）。")
    hard = sorted(R["benches"], key=lambda b: (b["n_solved"], -(b["best_ert"] / b["budget"])))[:3]
    en.append("Hardest benchmarks: " + ", ".join(f"{b['name']} (solved by {b['n_solved']}/{len(A)})" for b in hard) + ".")
    zh.append("最难的基准：" + "、".join(f"{b['name']}（{b['n_solved']}/{len(A)} 个算法解出）" for b in hard) + "。")
    return en, zh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="main")
    ap.add_argument("--out", default=os.path.join(ROOT, "site"))
    ap.add_argument("--primary", default="test", help="benchmark subset of the headline score")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--algs", help="comma-separated subset")
    ap.add_argument("--ref-alg", default="bh")
    ap.add_argument("--repo", default="")
    ap.add_argument("--no-gallery", action="store_true")
    a = ap.parse_args()
    recs, probs = load_all(jobs=a.jobs)
    st = Study(recs, probs, a.campaign, algs=a.algs.split(",") if a.algs else None)
    if not st.algs or not st.benches:
        sys.exit("no runs found")
    meta = load_meta(st.algs)
    names = [meta[k]["short"] for k in st.algs]
    types = [meta[k]["type"] for k in st.algs]
    na, pr = len(st.algs), a.primary
    img, dat = os.path.join(a.out, "img"), os.path.join(a.out, "data")
    os.makedirs(img, exist_ok=True)
    os.makedirs(dat, exist_ok=True)

    score_t = st.scores(pr)
    boot = st.bootstrap(pr, a.boot)
    # normal interval from the bootstrap SE; replicates re-centred on the point
    # estimate before ranking (removes the small bias of the clipped log score)
    sd = np.nanstd(boot, 0)
    lo, hi = score_t - 1.96 * sd, score_t + 1.96 * sd
    boot = boot - np.nanmean(boot, 0) + score_t
    brank = np.argsort(np.argsort(-np.nan_to_num(boot, nan=-1), 1), 1) + 1
    order = np.argsort(-np.nan_to_num(score_t, nan=-1))
    rank = np.empty(na, int)
    rank[order] = np.arange(1, na + 1)
    R_t = st.ranks(pr)
    avg_rank = R_t.mean(0) if len(R_t) else np.full(na, np.nan)
    speed, _ = st.speedup(a.ref_alg, pr)
    score_all, score_dev = st.scores("all"), st.scores("dev")
    score_wall = st.scores(pr, ert=st.ert_seconds())
    over = st.overhead(a.ref_alg)
    pooled, solved = st.pooled_sr(pr), st.n_solved(pr)
    A = []
    for j in order:
        k = st.algs[j]
        A.append(dict(key=k, rank=int(rank[j]), name=meta[k]["name"], short=names[j], type=types[j],
                      score_t=float(score_t[j]), ci_lo=float(lo[j]), ci_hi=float(hi[j]),
                      p_best=float((brank[:, j] == 1).mean()), gmre=float(st.gmre(score_t[j])),
                      avg_rank=float(avg_rank[j]), solved=int(solved[j]), pooled_sr=float(pooled[j]),
                      speedup=float(speed[j]), score_all=float(score_all[j]), score_dev=float(score_dev[j]),
                      score_wall=float(score_wall[j]), overhead=float(over[j]),
                      desc=meta[k]["desc"], ref=meta[k]["ref"]))

    # benchmarks, structures
    frames, B = [], []
    jbh = st.algs.index(a.ref_alg) if a.ref_alg in st.algs else None
    for i, b in enumerate(st.benches):
        p = probs[b]
        e_ref, rr = st.ref[b]
        at = st.gm_atoms(b)
        dmin = gallery.min_dist_ratio(at, p)
        rel = None
        if not a.no_gallery:
            rel = f"img/gm_{b}.png"
            gallery.render(at, p, os.path.join(a.out, rel))
        at.info.update(bench=b, e_ref=e_ref)
        frames.append(at)
        e = np.where(np.isnan(st.ert[i]), np.inf, st.ert[i])
        jf = int(np.argmin(e))
        hs = st.holes.get(b, [])
        B.append(dict(name=b, category=p.category, title=p.title, description=p.description,
                      guard="" if (p.guard_factor == 0.75 and not p.guard_pairs) else
                      f"guard f = {p.guard_factor:g}" + "".join(
                          f", {x}-{y} floor {v:g} A" for (x, y), v in sorted(p.guard_pairs.items())),
                      n_search=p.n_search, n_total=p.n,
                      budget=p.budget, e_tol=p.e_tol, e_ref=e_ref, dev=b in DEV,
                      found_by=f"{rr['alg_dir']} s{rr['seed']} ({rr['campaign']})",
                      holes=len(hs), hole_e=min((r["best_e"] for r, _ in hs), default=None),
                      hole_algs=sorted({names[st.algs.index(r["alg_dir"])] if r["alg_dir"] in st.algs
                                        else r["alg_dir"] for r, _ in hs}),
                      fastest=names[jf] if np.isfinite(e[jf]) else "-", best_ert=float(e[jf]),
                      bh_ert=float(st.ert[i, jbh]) if jbh is not None else float("nan"),
                      n_solved=int(np.isfinite(e).sum()), dmin=dmin, img=rel))
    write(os.path.join(dat, "gm_structures.extxyz"), frames)

    # figures
    pal = plots.alg_colors(na)
    colors = [pal[rank[j] - 1] for j in range(na)]
    xl = f"efficiency score ({pr} benchmarks; bars: 95% bootstrap CI)"
    plots.rank_bars(names, types, score_t, lo, hi, os.path.join(img, "rank_bars.png"), xl)
    plots.heatmap(st, order, names, os.path.join(img, "heatmap.png"))
    hl = set(order[:10].tolist()) | {st.algs.index(k) for k in ("bh", "rs") if k in st.algs}
    plots.perf_profile(st, pr, order, names, types, colors, os.path.join(img, "perf_profile.png"), hl)
    plots.ecdf(st, pr, order, names, types, colors, os.path.join(img, "ecdf.png"), hl)
    plots.category_matrix(st, order, names, os.path.join(img, "categories.png"))
    fr = st.friedman(pr)
    if fr:
        plots.cd_diagram(fr["avg_rank"], fr["cd"], names, os.path.join(img, "cd.png"))
    # robustness of the ranking to the success tolerance
    sens = []
    for tol in (0.005, 0.01, 0.05, 0.1):
        s = st.scores(pr, ert=st.ert_at(tol))
        ok = np.isfinite(s) & np.isfinite(score_t)
        tau = float(stats.kendalltau(score_t[ok], s[ok])[0]) if ok.sum() > 2 else float("nan")
        top = [names[j] for j in np.argsort(-np.nan_to_num(s, nan=-1))[:3]]
        sens.append(dict(tol=tol, tau=tau, top=top))
    ms_bh = float(np.nanmedian(st.ms_step[:, jbh])) if jbh is not None else float("nan")
    plots.overhead_bars(over, ms_bh, order, names, types, os.path.join(img, "overhead.png"))
    curves = []
    for c in CATS:
        if plots.sr_curves(st, c, order, names, types, colors, os.path.join(img, f"curves_{c}.png")):
            curves.append((c, f"img/curves_{c}.png"))

    # data files
    with open(os.path.join(dat, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "alg", "n", "n_hit", "sr", "ert", "med_hit", "med_gap", "mean_gap",
                    "wall_mean_s", "ms_per_step", "n_error", "n_timeout", "e_ref", "e_tol", "budget"])
        for i, b in enumerate(st.benches):
            for j, k in enumerate(st.algs):
                if (i, j) in st.outc:
                    w.writerow([b, k, int(st.n[i, j]), int(st.n_hit[i, j]), f"{st.sr[i, j]:.3f}",
                                f"{st.ert[i, j]:.1f}", f"{st.med_hit[i, j]:.0f}", f"{st.med_gap[i, j]:.4f}",
                                f"{st.mean_gap[i, j]:.4f}", f"{st.wall[i, j]:.1f}", f"{st.ms_step[i, j]:.3f}",
                                int(st.n_err[i, j]), int(st.n_tmo[i, j]), f"{st.ref[b][0]:.4f}",
                                probs[b].e_tol, probs[b].budget])
    with open(os.path.join(dat, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "alg", "seed", "hit_step", "nsteps", "budget", "best_e", "gap", "wall_s", "status"])
        for r in sorted(st.runs, key=lambda r: (r["bench"], r["alg_dir"], r["seed"])):
            if r["bench"] not in st.ref:
                continue
            e0 = st.ref[r["bench"]][0]
            idx = np.flatnonzero(r["trace_e"] <= e0 + probs[r["bench"]].e_tol)
            h = int(r["trace_steps"][idx[0]]) if len(idx) else ""
            w.writerow([r["bench"], r["alg_dir"], r["seed"], h, r["nsteps"], r["budget"],
                        f"{r['best_e']:.5f}" if r["best_e"] is not None else "",
                        f"{r['best_e'] - e0:.5f}" if r["best_e"] is not None else "",
                        f"{r['wall'] or 0:.1f}", r["status"]])

    nruns = len(st.runs)
    R = dict(nb=len(st.benches), na=na, seeds=int(np.nanmedian(st.n)), nt=int(st.subset("test").sum()),
             nd=int(st.subset("dev").sum()), nt_rows=int(st.rows(pr).sum()), nruns=nruns,
             steps=float(sum(r["nsteps"] or 0 for r in st.runs)),
             corehours=float(sum(r["wall"] or 0 for r in st.runs) / 3600),
             bmin=f"{min(probs[b].budget for b in st.benches):,}",
             bmax=f"{max(probs[b].budget for b in st.benches):,}",
             date=datetime.date.today().isoformat(), camp=a.campaign, repo=a.repo, algs=A, benches=B,
             friedman=fr and dict(chi2=fr["chi2"], p=fr["p"], n=fr["n"], k=fr["k"], cd=fr["cd"]),
             sens=sens, top3=[names[j] for j in order[:3]],
             e_tol=float(np.median([probs[b].e_tol for b in st.benches])),
             curves=curves, meta=meta)
    ran = st.subset(pr) & ~np.isnan(st.ert).any(1)
    R["unsolved"] = [st.benches[i] for i in np.flatnonzero(ran & ~st.complete())]
    R["findings"] = findings(st, R, A, pr)
    notes = os.path.join(ROOT, "docs", "report_notes.json")
    if os.path.exists(notes):
        nj = json.load(open(notes))
        R["notes"] = [nj.get("en", ""), nj.get("zh", "")]
    for lang, fn in (("en", "index.html"), ("zh", "zh.html")):
        with open(os.path.join(a.out, fn), "w") as f:
            f.write(page(lang, R))
    js = {k: v for k, v in R.items() if k not in ("meta",)}

    def _strict(o):
        if isinstance(o, dict):
            return {k: _strict(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_strict(v) for v in o]
        if hasattr(o, "tolist"):
            return _strict(o.tolist())
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    with open(os.path.join(dat, "ranking.json"), "w") as f:
        json.dump(_strict(js), f, indent=1, allow_nan=False, default=str)
    open(os.path.join(a.out, ".nojekyll"), "w").close()
    print(f"report: {a.out}  ({len(st.benches)} benchmarks, {na} algorithms, {nruns} runs)")
    for x in A:
        print(f"  {x['rank']:2d} {x['key']:10s} {x['score_t']:6.1f} [{x['ci_lo']:5.1f},{x['ci_hi']:5.1f}] "
              f"avg_rank {x['avg_rank']:5.2f} solved {x['solved']:3d} SR {x['pooled_sr']:.2f}")


if __name__ == "__main__":
    main()
