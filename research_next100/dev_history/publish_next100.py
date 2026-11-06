#!/usr/bin/env python3
"""Add research-specific context and supporting data to the completed report."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site_next100"
BENCH = ROOT / "benches-eval"
with (SITE / "candidate_geometry_audit.json").open() as f:
    audit = json.load(f)
if (audit["candidate_count"] != 10086 or audit["failed_checks"]
        or audit["counts_by_system"]["ze_Au4_CHA"] != 2387
        or audit["outside_initial_region"]["ze_Au4_CHA:atoms_outside_initial_region"] != 2387
        or audit["outside_initial_region"]["ze_Au4_CHA:center_outside_initial_sphere"] != 116):
    raise ValueError("Candidate geometry audit does not match the published disclosure")
ANNOTATIONS = {
    "index.html": (
        "<h2>System selection and experimental methods</h2>"
        "<p>The four systems are Cu13, variable-cell Al8, Pd7 on MgO(001), and "
        "Au4 initialized in a CHA zeolite. These manually selected systems are "
        "not a random or representative sample. The surface and zeolite hosts "
        "were also used during algorithm development, so they are not unseen "
        "hosts. In a separate five-seed pilot with BH and SurrBH, all 40 runs "
        "hit their pilot reference window; the systems may be relatively easy.</p>"
        "<p>The 26 compared methods comprise 22 previously frozen algorithms "
        "and two experimental families, each tested both before and after "
        "adding a hosted-site move. They are not four independent inventions. "
        "A ten-seed surface-development ablation gave 3/10 versus 4/10 hits "
        "for the diversity family, and 3/10 versus 5/10 for path relinking; "
        "these small development samples do not establish an improvement. "
        "The methods combine established ideas; literature novelty and "
        "general superiority have not been established.</p>"
        "<p>The hosts are frozen; the CHA source CIF provenance is unknown. "
        "Initial placement regions are not confinement constraints, and two "
        "of ten zeolite pilot centers lay outside the initial region; that "
        "alone proves neither escape nor confinement. NEP89 is not DFT, and "
        "guard-free structures may still be unphysical. Scores from different "
        "benchmark sets are not directly comparable.</p>"
        "<p><a href='ablation.html'>Hosted-move ablations</a> · "
        "<a href='protocol.json'>Frozen protocol</a> · "
        "<a href='pilot_audit.json'>System pilot audit</a> · "
        "<a href='reference_forces.json'>Final reference force checks</a> · "
        "<a href='freeze_manifest.json'>Frozen input hashes</a>.</p>"
    ),
    "zh.html": (
        "<h2>体系选择与实验算法</h2>"
        "<p>四个体系为 Cu13、变胞 Al8、MgO(001) 表面的 Pd7 和在 CHA "
        "分子筛中初始化的 Au4。它们由人工选择，并非随机或代表性样本。"
        "表面和分子筛宿主也用于算法开发，因此并非未见过的新宿主。"
        "独立的 BH 与 SurrBH 五种子预试验中，40 次运行均命中预试验参考窗口，"
        "这些体系可能相对容易。</p>"
        "<p>26 个比较方法包括 22 个既有冻结算法，以及两个实验算法家族"
        "各自加入宿主位点移动前后的版本；这并非四项相互独立的发明。"
        "十种子表面开发消融中，多样性方法的命中数由 3/10 变为 4/10，"
        "路径重连由 3/10 变为 5/10；样本太少，不能据此认定改进有效。"
        "这些方法组合了已有思想，文献新颖性和普遍优势尚未确立。</p>"
        "<p>宿主固定，CHA 原始 CIF 来源不明。初始采样区域不是硬约束；"
        "十个分子筛预试验结构中有两个团簇中心位于初始区域之外，"
        "这本身既不能证明逃逸，也不能证明仍被限制在原笼中。"
        "NEP89 不是 DFT，guard 不活跃也不能保证结构物理正确。"
        "不同基准集合的得分不能直接比较。</p>"
        "<p><a href='ablation.html'>宿主位点移动消融</a> · "
        "<a href='protocol.json'>冻结实验方案</a> · "
        "<a href='pilot_audit.json'>体系预试验审计</a> · "
        "<a href='reference_forces.json'>最终参考结构受力检查</a> · "
        "<a href='freeze_manifest.json'>冻结输入校验值</a>。</p>"
    ),
}
AUDIT_NOTES = {
    "index.html": (
        "<p>The additional <a href='candidate_geometry_audit.json'>saved-candidate "
        "geometry audit</a> checked 10,086 best structures in the reference "
        "window from the formal campaign and pilot; none failed its energy, "
        "guard, distance, connectivity or applicable volume checks. All 2,387 "
        "zeolite candidates had at least one atom outside the initial region, "
        "and 116 had their cluster center outside the initial sphere. These "
        "region observations alone do not establish cage escape or confinement. "
        "Intermediate trajectory geometries were not saved and cannot be "
        "assessed by this audit.</p>"
    ),
    "zh.html": (
        "<p>补充的<a href='candidate_geometry_audit.json'>已保存候选结构几何审计</a>"
        "检查了正式实验和预试验中落入参考窗口的 10,086 个最佳结构；"
        "能量、guard、距离、连通性和适用的体积检查均无失败。"
        "2,387 个分子筛候选结构均有至少一个原子在初始区域外，其中 116 个"
        "团簇中心位于初始球外；这些现象本身不能证明逃出或仍留在笼中。"
        "轨迹中的中间构型未保存，因此无法通过此审计验证。</p>"
    ),
}

for filename, annotation in ANNOTATIONS.items():
    path = SITE / filename
    page = path.read_text(encoding="utf-8")
    marker = "<h2>Ranking</h2>" if filename == "index.html" else "<h2>排名</h2>"
    if page.count(marker) != 1:
        raise ValueError(f"{path}: expected exactly one ranking heading")
    for block in (annotation, AUDIT_NOTES[filename]):
        if page.count(block) > 1:
            raise ValueError(f"{path}: duplicate research annotation")
        if block not in page:
            page = page.replace(marker, block + marker, 1)
    path.write_text(page, encoding="utf-8")
for name in ("protocol.json", "pilot_audit.json", "freeze_manifest.json"):
    shutil.copyfile(BENCH / name, SITE / name)
if not (SITE / "reference_forces.json").is_file():
    raise FileNotFoundError("Run the final reference-force audit before publishing")
if not (SITE / "ablation.html").is_file():
    raise FileNotFoundError("Run compare_ablations.py before publishing")
print("Added research disclosures and supporting data.")
