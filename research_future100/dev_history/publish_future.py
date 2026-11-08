#!/usr/bin/env python3
"""Add research disclosures and audited supporting data to the completed report."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site_future100"
BENCH = ROOT / "benches-eval"

with (SITE / "independent_verification.json").open() as f:
    verification = json.load(f)
with (SITE / "candidate_geometry_audit.json").open() as f:
    geometry = json.load(f)
with (SITE / "summary.json").open() as f:
    summary = json.load(f)
if verification["runs"] != 15500 or verification["metric_cells"] != 155:
    raise ValueError("Independent verification is incomplete")
if geometry["failed_checks"]:
    raise ValueError("Candidate geometry audit has failed checks")
for name in ("reference_forces.json", "ablation.html", "ablation.csv"):
    if not (SITE / name).is_file():
        raise FileNotFoundError(f"Missing audited support file: {name}")

notes = {
    "index.html": (
        "<h2>System selection and experimental methods</h2>"
        "<p>The five systems are Cu4 and Mg4O4 variable-cell bulk crystals, Ni13, "
        "Ag4 on Cu(111), and Au4 on graphene. They are manually selected "
        "convenience samples, not a random representative set. The Cu and "
        "graphene hosts are ideal frozen structures with illustrative geometry; "
        "adsorption, substrate relaxation and alloying have not been independently "
        "validated. Initial sampling regions are not confinement constraints.</p>"
        "<p>The 31 methods comprise 26 previously evaluated methods plus two "
        "experimental families and three mechanism controls, not five independent "
        "inventions. They combine established ideas; neither literature novelty "
        "nor general superiority has been established. Their earlier five-seed "
        "development comparisons were on previously used systems and are not "
        "independent evidence of generalization.</p>"
        "<p>The six-system pilot used BH and Growth at seeds 30–34; all 60 saved "
        "best structures passed the baseline checks. Only five systems were "
        "selected for this study, whose seeds 100–199 are disjoint from the pilot. "
        "Ag4/MFI was held out because an Ag–Si contact and rigid-framework "
        "interface remain insufficiently validated; a prior Pd4/Al candidate was "
        "excluded after a disconnected saved structure. The full pilot audit "
        "retains the held-out MFI results. Earlier Mg4O4 trials also showed a "
        "short Mg–O contact and an initially short O–O distance; these are neither "
        "proven artefacts nor independently validated chemistry.</p>"
        f"<p>The additional <a href='candidate_geometry_audit.json'>candidate audit</a> "
        f"checked {geometry['candidate_count']} saved best structures within the "
        "reference window and found no failures of its energy, guard, distance, "
        "connectivity or applicable volume checks. Intermediate trace geometries "
        "were not saved; neither this audit nor zero guard energy proves all hits "
        "physical or identifies a global minimum. NEP89 is not DFT; the reported "
        "ionic-step score is not a wall-clock ranking and scores across different "
        "system sets cannot be compared directly.</p>"
        "<p><a href='ablation.html'>Mechanism ablations</a> · "
        "<a href='protocol.json'>Frozen protocol</a> · "
        "<a href='pilot_audit.json'>Full pilot audit</a> · "
        "<a href='reference_forces.json'>Reference force checks</a> · "
        "<a href='independent_verification.json'>Independent metric verification</a> · "
        "<a href='freeze_manifest.json'>Frozen input hashes</a>.</p>"
    ),
    "zh.html": (
        "<h2>体系选择与实验算法</h2>"
        "<p>五个体系为变胞 Cu4 和 Mg4O4 体相、Ni13、Cu(111) 表面的 Ag4 "
        "以及石墨烯上的 Au4。它们由人工选择，并非随机或代表性样本。"
        "铜和石墨烯宿主为理想固定结构，几何参数仅作示例；吸附、宿主弛豫"
        "和合金化尚未得到独立验证。初始采样区域不是硬约束。</p>"
        "<p>31 个方法包括 26 个既有方法、两个实验算法家族及三个机制对照，"
        "并非五项相互独立的发明。它们组合了已有思想，文献新颖性和普遍优势"
        "尚未确立。此前每种组合仅五个种子的开发对比使用了既有开发体系，"
        "不能作为泛化性能的独立证据。</p>"
        "<p>六体系预试验使用 BH 和 Growth、种子 30–34；60 个已保存最佳结构"
        "通过既定检查。正式实验仅选择其中五个体系，种子 100–199 与预试验"
        "不重叠。Ag4/MFI 因 Ag–Si 接触和固定框架界面尚缺乏充分验证而暂缓；"
        "此前 Pd4/Al 的一个已保存结构不连通，因此未纳入。完整预试验审计保留"
        "被暂缓的 MFI 结果。较早的 Mg4O4 小试验还出现较短 Mg–O 接触和初始"
        "较短 O–O 距离：这既未证明模型伪影，也未得到独立化学验证。</p>"
        f"<p>补充的<a href='candidate_geometry_audit.json'>候选结构审计</a>检查了"
        f"参考窗口内 {geometry['candidate_count']} 个已保存最佳结构，能量、guard、"
        "距离、连通性和适用的体积检查均无失败。轨迹中间构型未保存；这一审计"
        "或零 guard 能量均不能证明所有命中结构物理正确，也不能证明全局最优。"
        "NEP89 不是 DFT；离子步得分不是墙钟时间排名，不同体系集合的得分"
        "不能直接比较。</p>"
        "<p><a href='ablation.html'>算法机制消融</a> · "
        "<a href='protocol.json'>冻结实验方案</a> · "
        "<a href='pilot_audit.json'>完整预试验审计</a> · "
        "<a href='reference_forces.json'>参考结构受力检查</a> · "
        "<a href='independent_verification.json'>独立指标复算</a> · "
        "<a href='freeze_manifest.json'>冻结输入校验值</a>。</p>"
    ),
}

for filename, note in notes.items():
    path = SITE / filename
    page = path.read_text(encoding="utf-8")
    marker = "<h2>Ranking</h2>" if filename == "index.html" else "<h2>排名</h2>"
    if page.count(marker) != 1 or page.count(note) > 1:
        raise ValueError(f"{path}: unexpected report structure")
    if note not in page:
        if ("<h2>System selection and experimental methods</h2>" in page
                or "<h2>体系选择与实验算法</h2>" in page):
            raise ValueError(f"{path}: a different disclosure is already present")
        path.write_text(page.replace(marker, note + marker, 1), encoding="utf-8")

outside = geometry["outside_initial_region"]
counts = geometry["counts_by_system"]
systems = ("cl_Ni13", "su_Ag4_Cu111", "su_Au4_graphene")
outside_counts = [outside.get(f"{system}:atoms_outside_initial_region", 0)
                  for system in systems]
if any(n < 0 or n > counts[system] for system, n in zip(systems, outside_counts)):
    raise ValueError("Invalid outside-region candidate count")
timed = summary["algorithms"]["gofee"]["timeouts"]
timed_hits = verification["timeout_hits_by_algorithm"]["gofee"]
if timed_hits != timed or timed > summary["algorithms"]["gofee"]["runs"]:
    raise ValueError("GOFEE timeout-hit counts differ")
ni, ag, au = outside_counts
extra = {
    "index.html": (
        "<h2>Observed timeout and initial-region results</h2>"
        f"<p>GOFEE had {timed} timed-out runs out of "
        f"{summary['algorithms']['gofee']['runs']}; all {timed_hits} had already "
        "hit the energy window before the cap. The ionic-step ranking therefore "
        "retains those hits and does not measure the substantial wall-clock cost.</p>"
        f"<p>Among saved candidate structures in the reference window, at least "
        f"one search atom lay outside the initial sampling box in {ni}/{counts['cl_Ni13']} "
        f"Ni13, {ag}/{counts['su_Ag4_Cu111']} Ag4/Cu(111), and "
        f"{au}/{counts['su_Au4_graphene']} Au4/graphene structures. These counts "
        "include the selected pilot records. The box is an initialization region, "
        "not a confinement constraint, so this observation alone neither proves "
        "an artefact nor validates the interface chemistry.</p>"
    ),
    "zh.html": (
        "<h2>实测超时与初始采样区域情况</h2>"
        f"<p>GOFEE 的 {summary['algorithms']['gofee']['runs']} 次运行中有 "
        f"{timed} 次超时；这 {timed_hits} 次均在达到时间上限前已进入能量窗口。"
        "因此离子步排名保留这些命中，但不衡量其显著的墙钟耗时。</p>"
        f"<p>参考窗口内已保存候选结构中，至少一个搜索原子位于初始采样框外的数量为："
        f"Ni13 {ni}/{counts['cl_Ni13']}，Ag4/Cu(111) {ag}/{counts['su_Ag4_Cu111']}，"
        f"Au4/石墨烯 {au}/{counts['su_Au4_graphene']}。这些计数包括所选预试验记录。"
        "采样框用于初始化而非硬约束；这一现象本身既不能证明势能伪影，"
        "也不能验证界面化学的可靠性。</p>"
    ),
}
for filename, note in extra.items():
    path = SITE / filename
    page = path.read_text(encoding="utf-8")
    marker = "<h2>Ranking</h2>" if filename == "index.html" else "<h2>排名</h2>"
    heading = ("<h2>Observed timeout and initial-region results</h2>"
               if filename == "index.html" else "<h2>实测超时与初始采样区域情况</h2>")
    if page.count(marker) != 1 or page.count(note) > 1:
        raise ValueError(f"{path}: unexpected report structure")
    if note not in page:
        if heading in page:
            raise ValueError(f"{path}: a different observed-results note is already present")
        path.write_text(page.replace(marker, note + marker, 1), encoding="utf-8")
for name in ("protocol.json", "pilot_audit.json", "freeze_manifest.json"):
    shutil.copyfile(BENCH / name, SITE / name)
print("Added audited research disclosures.")
