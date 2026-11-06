#!/usr/bin/env python3
"""Merge the algorithm documentation into docs/algorithms.json (read by scripts/report.py).

Sources: the developers' entries scratch/<key>/algorithms_entry.json (keys name, short, type, en, zh, ref)
and the coordinator's entries for the two reference baselines below.

    python scripts/merge_alg_docs.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("name", "short", "type", "en", "zh", "ref")

BASE = {
    "rs": dict(
        name="Random search", short="RS", type="baseline",
        en="Ab initio random structure searching (AIRSS) style sampling: independent random structures (with a "
           "random cell for bulk crystals) are generated inside the search region with the shared minimum-distance "
           "and connectivity constraints, and each is relaxed with L-BFGS. There is no memory between trials, so "
           "its efficiency is the probability of landing in the global-minimum basin per ionic step; it is the "
           "reference for how much a search strategy gains over pure sampling.",
        zh="仿照 AIRSS（从头随机结构搜索）的随机采样：在搜索区域内按统一的最小原子间距与连通性约束独立生成随机结构"
           "（体相同时随机生成晶胞），每个结构用 L-BFGS 弛豫。各次试验之间没有记忆，因此其效率就是每个离子步落入全局最优"
           "盆地的概率；它是衡量搜索策略相对纯随机采样收益的参照。",
        ref="Pickard & Needs, J. Phys.: Condens. Matter 23, 053201 (2011)"),
    "bh": dict(
        name="Basin hopping", short="BH", type="baseline",
        en="Monte Carlo minimisation (basin hopping) with the standard GMIN settings: a Metropolis walk at "
           "kT = 0.1 eV on the relaxed energy surface. Every move displaces all search atoms uniformly within a "
           "sphere of radius <i>step</i> (bulk: plus a random symmetric cell strain, &sigma; = 0.05), relaxes with "
           "L-BFGS and accepts with probability min(1, exp(&minus;&Delta;E/kT)). Every 10 moves the step is "
           "multiplied by 1.1 or 0.9 depending on whether the acceptance ratio of those 10 moves is above 50% "
           "(bounded to 0.1&ndash;2 &Aring;, initial 0.5 &Aring;); detached cluster atoms are re-seated on the "
           "cluster surface. Literature parameters, not tuned.",
        zh="采用 GMIN 标准设置的蒙特卡洛极小化（盆地跳跃）：在弛豫后的能量面上进行 kT = 0.1 eV 的 Metropolis 行走。"
           "每次移动把所有搜索原子在半径为 <i>step</i> 的球内均匀随机位移（体相再加 &sigma; = 0.05 的随机对称晶胞应变），"
           "用 L-BFGS 弛豫，并以 min(1, exp(&minus;&Delta;E/kT)) 的概率接受。每 10 次移动根据这 10 次移动的接受率是否高于 50% "
           "将步长乘以 1.1 或 0.9（限制在 0.1&ndash;2 &Aring;，初值 0.5 &Aring;）；团簇中脱离的原子被重新放回团簇表面。"
           "采用文献参数，未调参。",
        ref="Li & Scheraga, PNAS 84, 6611 (1987); Wales & Doye, J. Phys. Chem. A 101, 5111 (1997)"),
}


def main():
    out, bad = dict(BASE), 0
    for f in sorted(glob.glob(os.path.join(ROOT, "scratch", "*", "algorithms_entry.json"))):
        key = os.path.basename(os.path.dirname(f))
        if not os.path.exists(os.path.join(ROOT, "gopt", "algorithms", f"{key}.py")):
            print(f"skip {key}: no gopt/algorithms/{key}.py")
            continue
        d = json.load(open(f))
        miss = [k for k in KEYS if not str(d.get(k, "")).strip()]
        if miss or d.get("type") not in ("baseline", "innovative"):
            print(f"{key}: missing {miss} or bad type {d.get('type')!r}")
            bad += 1
        out[key] = {k: d[k] for k in KEYS if k in d}
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "algorithms.json"), "w") as fh:
        json.dump(dict(sorted(out.items())), fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    n = {t: sum(v.get("type") == t for v in out.values()) for t in ("baseline", "innovative")}
    print(f"docs/algorithms.json: {len(out)} algorithms ({n['baseline']} baseline, {n['innovative']} innovative)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
