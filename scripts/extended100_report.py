#!/usr/bin/env python3
"""Audit the completed extension campaign and write its separate ranking.

Run with GOPT_BENCH_DIR set to the extension's benchmark directory. The report
deliberately does not reuse the original campaign's static methods text.
"""
import argparse
import csv
import glob
import hashlib
import html
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gopt.algorithms import available  # noqa: E402
from gopt.nep import NEP89  # noqa: E402
from gopt.problem import Problem, list_benchmarks  # noqa: E402
from gopt.report.data import GUARD_TOL, Study, _read, penalty  # noqa: E402
from scripts.validate_gm import check  # noqa: E402

CAMPAIGN = "ext100"
SEEDS = 100


def audit_record(r, bench, alg, seed, manifest):
    """Reject missing, stale, malformed or failed campaign records."""
    spec = manifest["benchmarks"][bench]
    expected = {
        "bench": bench, "alg": alg, "seed": seed,
        "bench_hash": spec["bench_hash"],
        "alg_hash": manifest["algorithms"][alg]["alg_hash"],
        "params": manifest["algorithms"][alg]["defaults"],
        "model": manifest["nep_model"]["file"],
        "budget": spec["budget"],
    }
    for key, value in expected.items():
        if r.get(key) != value:
            raise ValueError(f"{bench}/{alg}/s{seed:03d}: {key} mismatch")
    if r.get("status") not in ("ok", "timeout"):
        raise ValueError(f"{bench}/{alg}/s{seed:03d}: status {r.get('status')!r}")
    n = r.get("nsteps")
    e = r.get("best_e")
    if not isinstance(n, int) or not 0 < n <= spec["budget"] or e is None or not np.isfinite(e):
        raise ValueError(f"{bench}/{alg}/s{seed:03d}: invalid step count or energy")
    steps, energies = r.get("trace_steps"), r.get("trace_e")
    if (not isinstance(steps, list) or not steps or not isinstance(energies, list)
            or len(steps) != len(energies)):
        raise ValueError(f"{bench}/{alg}/s{seed:03d}: invalid trace")
    if (not all(isinstance(s, int) and 0 < s <= n for s in steps)
            or any(a >= b for a, b in zip(steps, steps[1:]))
            or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in energies)
            or any(a < b for a, b in zip(energies, energies[1:]))):
        raise ValueError(f"{bench}/{alg}/s{seed:03d}: invalid trace values")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", default=CAMPAIGN)
    ap.add_argument("--out", default=os.path.join(ROOT, "site_extended100"))
    ap.add_argument("--release-url", default="https://github.com/JiaaoWANG-ut/"
                                            "nep89-global-opt-benchmark/releases/tag/v1.1")
    args = ap.parse_args()
    benchdir = os.environ.get("GOPT_BENCH_DIR")
    if not benchdir:
        raise SystemExit("Set GOPT_BENCH_DIR to the extension's benchmark directory")
    manifest_path = os.path.join(benchdir, "freeze_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    def sha1(path):
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha1").hexdigest()

    if os.path.basename(NEP89) != manifest["nep_model"]["file"] or sha1(NEP89) != manifest["nep_model"]["sha1"]:
        raise SystemExit("NEP model differs from the extension freeze")
    for path, digest in manifest["shared_code"].items():
        if sha1(os.path.join(ROOT, path)) != digest:
            raise SystemExit(f"Shared code differs from the extension freeze: {path}")
    benches, algs = list_benchmarks(), available()
    if set(benches) != set(manifest["benchmarks"]) or set(algs) != set(manifest["algorithms"]):
        raise SystemExit("Benchmark or algorithm inventory differs from the extension freeze")
    probs = {b: Problem(b) for b in benches}
    for b in benches:
        if probs[b].hash != manifest["benchmarks"][b]["bench_hash"]:
            raise SystemExit(f"Current benchmark hash differs from freeze: {b}")

    root = os.path.join(ROOT, "results")
    campaign_dir = os.path.join(root, args.campaign)
    expected_paths = set()
    statuses = Counter()
    for b in benches:
        for alg in algs:
            for seed in range(SEEDS):
                path = os.path.join(campaign_dir, b, alg, f"s{seed:03d}.json")
                expected_paths.add(path)
                try:
                    with open(path) as f:
                        r = json.load(f)
                    audit_record(r, b, alg, seed, manifest)
                except (OSError, ValueError, TypeError) as exc:
                    raise SystemExit(f"Campaign audit failed: {path}: {exc}") from exc
                statuses[r["status"]] += 1
    actual_paths = set(glob.glob(os.path.join(campaign_dir, "*", "*", "s*.json")))
    if actual_paths != expected_paths:
        raise SystemExit(f"Unexpected or missing records: {len(actual_paths ^ expected_paths)}")

    recs = []
    for b in benches:
        for path in glob.glob(os.path.join(root, "*", b, "*", "s*.json")):
            r = _read(path)
            if r is None:
                raise SystemExit(f"Unreadable reference candidate: {path}")
            if r["bench_hash"] == probs[b].hash:
                recs.append(r)
    st = Study(recs, probs, args.campaign, algs=algs, benches=benches)
    if set(st.benches) != set(benches) or not np.all(st.n == SEEDS):
        raise SystemExit("Study is missing a guard-free reference or a full 100-seed cell")

    checks, active = {}, {}
    cache = {}
    for b in benches:
        checks[b] = check(probs[b], st.ref[b][1]["file"], cache)
        if checks[b]["bad"]:
            raise SystemExit(f"Reference validation failed for {b}: {checks[b]['bad']}")
        active[b] = [
            {"campaign": r["campaign"], "algorithm": r["alg_dir"], "seed": r["seed"],
             "energy": r["best_e"], "penalty": eg}
            for r in recs if r["bench"] == b and r["best_e"] is not None
            and r["best_e"] <= st.ref[b][0] + probs[b].e_tol
            if (eg := penalty(probs[b], r["file"])) > GUARD_TOL
        ]
    if not st.rows("all").all():
        raise SystemExit("At least one benchmark has no hit; inspect results before ranking")

    scores = st.scores("all")
    boot = st.bootstrap("all", nboot=1000)
    sd = np.nanstd(boot, axis=0)
    out = args.out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "algorithm", "n", "hits", "success_rate", "ert_steps",
                    "reference_eV", "tolerance_eV", "budget", "timeouts", "errors"])
        for i, b in enumerate(st.benches):
            for j, alg in enumerate(st.algs):
                w.writerow([b, alg, int(st.n[i, j]), int(st.n_hit[i, j]), st.sr[i, j],
                            st.ert[i, j], st.ref[b][0], probs[b].e_tol, probs[b].budget,
                            int(st.n_tmo[i, j]), int(st.n_err[i, j])])
    ranking = [
        {"rank": rank, "algorithm": st.algs[j], "score": float(scores[j]),
         "ci_low": float(scores[j] - 1.96 * sd[j]),
         "ci_high": float(scores[j] + 1.96 * sd[j])}
        for rank, j in enumerate(np.argsort(-scores), 1)
    ]
    alg_stats = {
        alg: {"runs": int(st.n[:, j].sum()), "hits": int(st.n_hit[:, j].sum()),
              "timeouts": int(st.n_tmo[:, j].sum())}
        for j, alg in enumerate(st.algs)
    }
    report = {
        "campaign": args.campaign, "runs": len(expected_paths), "seeds": SEEDS,
        "statuses": dict(statuses), "algorithms": alg_stats, "ranking": ranking,
        "references": {b: {"energy": st.ref[b][0], "validation": checks[b],
                            "guard_active_hits": active[b]} for b in benches},
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    rows = "".join(
        f"<tr><td>{r['rank']}</td><td>{html.escape(r['algorithm'])}</td>"
        f"<td>{r['score']:.1f}</td><td>[{r['ci_low']:.1f}, {r['ci_high']:.1f}]</td></tr>"
        for r in ranking
    )
    refs = "".join(
        f"<li>{html.escape(b)}: {st.ref[b][0]:.4f} eV; "
        f"{len(active[b])} guard-active candidate hit structures.</li>"
        for b in benches
    )
    timed = "".join(
        f"<li>{html.escape(alg)}: {s['timeouts']} timeouts in {s['runs']} runs; "
        f"{s['hits']} runs hit the reference window.</li>"
        for alg, s in alg_stats.items() if s["timeouts"]
    ) or "<li>No timeouts.</li>"
    css = (
        "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:auto;"
        "padding:1.5rem;color:#203040}table{border-collapse:collapse;width:100%}"
        "th,td{border-bottom:1px solid #d5dee5;padding:.45rem;text-align:left}"
        "th{background:#e8f0f5}tr:nth-child(even){background:#f7fafc}"
        "a{color:#075b86}h1,h2{color:#14374e}</style>"
    )
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>NEP89 extension: 100 seeds</title>{css}<body>"
        "<h1>NEP89 extension: 100 seeds</h1><p><a href='zh.html'>中文</a></p>"
        f"<p>{len(benches)} additional systems, {len(algs)} algorithms, 100 seeds per cell. "
        "This four-system ranking is separate from the "
        "<a href='../'>published 45-system study</a>; its scores are not directly comparable "
        "because the benchmark sets differ.</p>"
        "<p>Score: 100 times the mean of clip(1 + log10(ERT_best / ERT) / 3, 0, 1). "
        "ERT counts steps to the first energy at or below E_ref + tolerance; timeouts without "
        "a hit are charged the full budget. Intervals use a normal approximation from the standard "
        "deviation of 1000 within-cell bootstrap samples."
        "</p><p>E_ref is the lowest stored energy from any campaign for the current benchmark "
        "hash whose guard and wall energy is at most 0.01 eV. Guard-active runs are scored by "
        "energy, as in the original study; the counts below should be examined for artefacts. "
        "The reference is a putative NEP89 minimum, not a DFT global minimum.</p>"
        "<p>GOFEE and SurrBH had a 3,600-second limit per seed. "
        "Time limits may affect hit rates and rankings, especially for costly systems.</p>"
        "<h2>Ranking</h2><table><thead><tr><th>Rank</th><th>Algorithm</th><th>Score</th>"
        f"<th>95% interval</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<h2>References and guard screening</h2><ul>{refs}</ul>"
        f"<h2>Time limits</h2><ul>{timed}</ul>"
        f"<p>Run statuses: {html.escape(str(dict(statuses)))}. "
        "<a href='metrics.csv'>Per-system metrics</a> · "
        "<a href='summary.json'>Audit and ranking data</a> · "
        f"<a href='{html.escape(args.release_url, quote=True)}'>Raw run records</a>.</p></body></html>"
    )
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(page)
    zh_refs = "".join(
        f"<li>{html.escape(b)}：参考能量 {st.ref[b][0]:.4f} eV；"
        f"候选命中结构中 {len(active[b])} 个 guard 或势墙起作用。</li>"
        for b in benches
    )
    zh_timed = "".join(
        f"<li>{html.escape(alg)}：{s['runs']} 次中 {s['timeouts']} 次超时；"
        f"{s['hits']} 次运行达到参考能量窗口。</li>"
        for alg, s in alg_stats.items() if s["timeouts"]
    ) or "<li>没有超时。</li>"
    zh_page = (
        "<!doctype html><html lang='zh'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>NEP89 扩展实验：100 个种子</title>{css}<body>"
        "<h1>NEP89 扩展实验：100 个种子</h1><p><a href='index.html'>English</a></p>"
        f"<p>{len(benches)} 个新增体系，{len(algs)} 种算法，每种组合 100 个独立种子。"
        "此排名独立于<a href='../zh.html'>已发布的 45 体系研究</a>；"
        "由于基准集合不同，两者的分数不能直接比较。</p>"
        "<p>得分为 100 倍的逐体系平均值：clip(1 + log10(最佳 ERT / ERT) / 3, 0, 1)。"
        "ERT 按首次达到参考能量加容差的离子步计算；超时且未命中的运行计入全部预算。"
        "区间以每个体系与算法的种子进行 1000 次 bootstrap 重采样，再由标准差构造正态近似。</p>"
        "<p>参考能量是所有批次中、当前基准哈希下 guard 与表面势墙能量之和不超过 0.01 eV "
        "的已保存结构最低能量。guard 起作用的运行与原研究一样仍按能量计分；"
        "下方数量需要进一步排查势能空洞。参考结构只是 NEP89 势上的推定最低结构，不是 DFT 全局最优。</p>"
        "<p>GOFEE 和 SurrBH 每个种子的时间上限为 3,600 秒；时间限制可能影响命中率和排名，"
        "尤其是在计算成本较高的体系上。</p>"
        "<h2>排名</h2><table><thead><tr><th>名次</th><th>算法</th><th>得分</th>"
        f"<th>95% 区间</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<h2>参考结构与 guard 筛查</h2><ul>{zh_refs}</ul>"
        f"<h2>时间上限</h2><ul>{zh_timed}</ul>"
        f"<p>运行状态：{html.escape(str(dict(statuses)))}。"
        "<a href='metrics.csv'>逐体系指标</a> · "
        "<a href='summary.json'>审计与排名数据</a> · "
        f"<a href='{html.escape(args.release_url, quote=True)}'>原始运行记录</a>。</p></body></html>"
    )
    with open(os.path.join(out, "zh.html"), "w") as f:
        f.write(zh_page)
    print(f"Audited {len(expected_paths)} runs; wrote {out}")


if __name__ == "__main__":
    main()
