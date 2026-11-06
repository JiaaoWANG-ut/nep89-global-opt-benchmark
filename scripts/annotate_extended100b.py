#!/usr/bin/env python3
"""Add second-extension selection and model caveats to generated HTML pages."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "site_extended100b"
ANNOTATIONS = {
    "index.html": (
        "<h2>System selection and limitations</h2>"
        "<p>The four systems are Al13, Pt8 on CeO2(111), Au8 in BEA, "
        "and variable-cell Al4O6. They were selected from earlier candidates, "
        "not randomly sampled; Pt8/CeO2 and Al4O6 had previously been set aside "
        "as relatively easy. This small set is not representative of all "
        "structure-search problems. Scores are also not directly comparable "
        "with the <a href='../extension100/'>first 100-seed extension</a>.</p>"
        "<p>The CeO2 slab and BEA framework are frozen. NEP89 does not model "
        "ceria reduction or oxygen vacancies, and metal/zeolite binding is "
        "qualitative. The surface and zeolite reference structures extend "
        "outside their initial sampling regions; relaxation is not constrained "
        "to those regions. Passing the listed checks does not establish "
        "physical correctness or global optimality.</p>"
    ),
    "zh.html": (
        "<h2>体系选择与局限</h2>"
        "<p>四个体系分别为 Al13、CeO2(111) 表面 Pt8、BEA 分子筛中的 Au8 "
        "和变胞 Al4O6。它们选自先前的候选体系，并非随机抽样；"
        "Pt8/CeO2 和 Al4O6 曾因相对容易而未进入原主榜。"
        "这一小样本不能代表所有结构搜索问题；其得分也不能与"
        "<a href='../extension100/zh.html'>第一批 100 种子扩展</a>直接比较。</p>"
        "<p>CeO2 衬底和 BEA 骨架固定不动；NEP89 未描述氧化铈还原或氧空位，"
        "金属与分子筛之间的结合只能定性看待。表面与分子筛的参考结构"
        "有原子超出初始采样区域，弛豫过程并不受该区域约束。"
        "通过上述检查不代表结构在物理上正确或已找到全局最优。</p>"
    ),
}

for filename, annotation in ANNOTATIONS.items():
    path = PAGES / filename
    page = path.read_text(encoding="utf-8")
    if page.count(annotation) == 1:
        continue
    marker = "<h2>Ranking</h2>" if filename == "index.html" else "<h2>排名</h2>"
    if page.count(marker) != 1:
        raise ValueError(f"{path}: expected exactly one ranking heading")
    path.write_text(page.replace(marker, annotation + marker, 1), encoding="utf-8")
    print(f"Annotated {path}")
