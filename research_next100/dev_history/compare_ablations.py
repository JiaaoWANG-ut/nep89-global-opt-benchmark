#!/usr/bin/env python3
"""Publish descriptive hosted-move comparisons from audited campaign metrics."""
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site_next100"
RESULTS = ROOT / "results/next100"
FAMILIES = (
    ("Diversity archive", "diversity_search", "diversity_v2"),
    ("Path relinking", "path_relink", "relink_v2"),
)

with (SITE / "metrics.csv").open(newline="") as f:
    records = list(csv.DictReader(f))
metrics = {(r["benchmark"], r["algorithm"]): r for r in records}
if len(records) != 104 or len(metrics) != 104:
    raise ValueError("Expected 104 distinct system/method metric cells")
benches = sorted({b for b, _ in metrics})
if len(benches) != 4:
    raise ValueError(f"Expected four systems, found {len(benches)}")
rows = []
rng = np.random.default_rng(20261106)
for family, original, variant in FAMILIES:
    for bench in benches:
        old, new = metrics[bench, original], metrics[bench, variant]
        if int(old["n"]) != 100 or int(new["n"]) != 100:
            raise ValueError(f"Incomplete ablation cell: {bench}/{family}")
        if any(old[key] != new[key] for key in ("reference_eV", "tolerance_eV", "budget")):
            raise ValueError(f"Mismatched ablation targets: {bench}/{family}")
        old_ert, new_ert = float(old["ert_steps"]), float(new["ert_steps"])
        ratio = old_ert / new_ert if math.isfinite(old_ert) and math.isfinite(new_ert) else None
        samples = []
        for alg, metric in ((original, old), (variant, new)):
            spent, hit = [], []
            target = float(metric["reference_eV"]) + float(metric["tolerance_eV"])
            for seed in range(100):
                with (RESULTS / bench / alg / f"s{seed:03d}.json").open() as f:
                    record = json.load(f)
                if record["status"] not in ("ok", "timeout") or record["seed"] != seed:
                    raise ValueError(f"Invalid record: {bench}/{alg}/s{seed:03d}")
                h = next((s for s, e in zip(record["trace_steps"], record["trace_e"])
                          if e <= target), None)
                timed = record["status"] == "timeout" or (
                    record["status"] == "ok" and (record.get("stats") or {}).get("timeout"))
                spent.append(h if h is not None else
                             record["budget"] if timed else record["nsteps"])
                hit.append(h is not None)
            if sum(hit) != int(metric["hits"]):
                raise ValueError(f"Raw hits differ from metrics: {bench}/{alg}")
            ert = sum(spent) / sum(hit) if sum(hit) else math.inf
            if not math.isclose(ert, float(metric["ert_steps"]), rel_tol=1e-12):
                raise ValueError(f"Raw ERT differs from metrics: {bench}/{alg}")
            samples.append((np.asarray(spent), np.asarray(hit)))
        indices = rng.integers(0, 100, size=(10000, 100))
        old_count = samples[0][1][indices].sum(axis=1)
        new_count = samples[1][1][indices].sum(axis=1)
        valid = (old_count > 0) & (new_count > 0)
        lo = hi = None
        if valid.all():
            old_boot = samples[0][0][indices].sum(axis=1) / old_count
            new_boot = samples[1][0][indices].sum(axis=1) / new_count
            lo, hi = map(float, np.quantile(old_boot / new_boot, [0.025, 0.975]))
        rows.append({
            "benchmark": bench, "family": family, "original": original, "variant": variant,
            "original_hits": int(old["hits"]), "variant_hits": int(new["hits"]),
            "original_ert": old_ert, "variant_ert": new_ert,
            "ert_original_over_variant": ratio,
            "ratio_ci_low": lo, "ratio_ci_high": hi,
            "undefined_bootstrap_replicates": int((~valid).sum()),
        })
with (SITE / "ablation.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def display(value):
    return "—" if value is None or not math.isfinite(value) else f"{value:.2f}"


body = "".join(
    "<tr>"
    f"<td>{html.escape(r['benchmark'])}</td><td>{html.escape(r['family'])}</td>"
    f"<td>{r['original_hits']}/100</td><td>{r['variant_hits']}/100</td>"
    f"<td>{display(r['original_ert'])}</td><td>{display(r['variant_ert'])}</td>"
    f"<td>{display(r['ert_original_over_variant'])} "
    f"[{display(r['ratio_ci_low'])}, {display(r['ratio_ci_high'])}]</td></tr>"
    for r in rows
)
page = (
    "<!doctype html><html lang='en'><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>Hosted-move ablations / 宿主位点移动消融</title>"
    "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:auto;"
    "padding:1.5rem;color:#203040}table{border-collapse:collapse;width:100%}"
    "th,td{border-bottom:1px solid #d5dee5;padding:.45rem;text-align:left}"
    "th{background:#e8f0f5}a{color:#075b86}</style><body>"
    "<h1>Hosted-move ablations / 宿主位点移动消融</h1>"
    "<p><a href='index.html'>English report</a> · <a href='zh.html'>中文报告</a> · "
    "<a href='ablation.csv'>CSV</a></p>"
    "<p>Each cell has 100 seeds. ERT is ionic steps per discovery; "
    "original/variant ERT above 1 favors the hosted-move variant. Intervals "
    "are 95% percentile intervals from 10,000 seed-paired bootstrap samples "
    "with fixed references. Random streams can diverge after initialization. "
    "Intervals are omitted if any replicate has zero hits; see CSV for "
    "the number of undefined replicates. This exploratory comparison does not "
    "establish literature novelty or general superiority.</p>"
    "<p>每个单元包含 100 个种子。ERT 是每次发现所需的期望离子步；"
    "原版与改进版 ERT 之比大于 1 表示改进版更快。区间是固定参考能量下，"
    "按种子配对重采样 10,000 次得到的 95% 百分位区间；算法初始化后"
    "随机流可能不同。若任一重采样无命中则不报告区间，未定义次数见 CSV。"
    "这一探索性比较不能证明文献新颖性或普遍优势。</p>"
    "<table><thead><tr><th>System / 体系</th><th>Family / 算法家族</th>"
    "<th>Original hits / 原版命中</th><th>Variant hits / 改进版命中</th>"
    "<th>Original ERT / 原版</th><th>Variant ERT / 改进版</th>"
    "<th>ERT ratio [95% CI] / 比值</th></tr></thead>"
    f"<tbody>{body}</tbody></table></body></html>"
)
(SITE / "ablation.html").write_text(page, encoding="utf-8")
print("Wrote 8 descriptive ablation cells.")
