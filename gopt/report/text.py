"""Bilingual (en, zh) text blocks of the report."""

T = {
    "title": ("Global structure optimization with NEP89: algorithm ranking & efficiency",
              "基于 NEP89 的全局结构优化：算法排名与效率报告"),
    "subtitle": ("{nb} benchmark systems (gas-phase clusters, supported clusters, metals in zeolites, bulk crystals) "
                 "&middot; {na} algorithms &middot; {seeds} independent runs per algorithm and benchmark &middot; "
                 "efficiency = global-minimum discoveries per ionic step",
                 "{nb} 个基准体系（气相团簇、表面负载团簇、分子筛内贵金属、体相晶体）&middot; {na} 种算法 &middot; "
                 "每个算法×基准 {seeds} 次独立运行 &middot; 效率 = 每个离子步发现全局最优的次数"),
    "other_lang": ("中文", "English"),
    "nav": (["Summary", "Ranking", "Efficiency matrix", "Profiles", "Categories", "Statistics",
             "Wall-clock cost", "Benchmarks", "Algorithms", "Methods", "Downloads"],
            ["摘要", "排名", "效率矩阵", "性能曲线", "分类排名", "统计检验", "墙钟开销", "基准体系",
             "算法", "方法", "下载"]),
    "cards": (["benchmarks", "algorithms", "runs", "ionic steps", "CPU core-hours"],
              ["基准体系", "算法", "运行次数", "离子步", "CPU 核时"]),
    "findings": ("Key findings", "主要结论"),
    "rank_intro": (
        "Algorithms are ranked by the <b>efficiency score</b> on the {nt} held-out benchmarks (the {nd} development "
        "benchmarks, on which the new algorithms were tuned, are excluded). For each benchmark the expected running "
        "time ERT (ionic steps per discovery of the global minimum) of every algorithm is compared with the best "
        "algorithm on that benchmark: score = 100 &times; mean of clip(1 + log<sub>10</sub>(ERT<sub>best</sub>/ERT)/3, 0, 1). "
        "100 means fastest everywhere; every factor 10 slower costs 33 points; failing (or &ge;1000&times; slower) "
        "scores 0. GMRE is the equivalent geometric-mean relative efficiency (ERT<sub>best</sub>/ERT). "
        "Brackets: 95% interval (&plusmn;1.96 bootstrap standard errors, resampling seeds). Click a column header to sort.",
        "排名依据是 {nt} 个<b>留出基准</b>上的<b>效率得分</b>（新算法调参所用的 {nd} 个开发基准不计入）。"
        "对每个基准，比较各算法的期望运行时间 ERT（每发现一次全局最优所需的离子步数）与该基准上最快算法："
        "得分 = 100 &times; 平均[clip(1 + log<sub>10</sub>(ERT<sub>best</sub>/ERT)/3, 0, 1)]。"
        "100 表示在所有基准上都最快；每慢 10 倍扣 33 分；未找到（或慢 &ge;1000 倍）得 0 分。"
        "GMRE 为等价的几何平均相对效率（ERT<sub>best</sub>/ERT）。方括号为 95% 区间（&plusmn;1.96 倍 bootstrap 标准误，对种子重抽样）。点击表头可排序。"),
    "rank_cols": (["#", "Algorithm", "Type", "Score (held-out)", "P(best)", "GMRE", "Avg. rank", "Solved",
                   "Pooled SR", "Speed-up vs BH", "Score (all)", "Score (dev)", "Score (wall-clock)",
                   "Time/step vs BH"],
                  ["#", "算法", "类型", "得分（留出集）", "P(第一)", "GMRE", "平均名次", "解出基准数",
                   "总成功率", "相对 BH 加速", "得分（全部）", "得分（开发集）", "得分（墙钟）", "单步耗时/BH"]),
    "type": ({"baseline": "baseline", "innovative": "new"}, {"baseline": "主流", "innovative": "创新"}),
    "matrix_intro": (
        "ERT of every algorithm divided by the best ERT on each benchmark (1 = fastest; x = global minimum never "
        "found in any run). Rows grouped by category, * marks development benchmarks; columns ordered by overall rank.",
        "各算法 ERT 与该基准最佳 ERT 之比（1 = 最快；x = 所有运行都未找到全局最优）。行按类别分组，* 为开发基准；列按总排名排序。"),
    "profiles_intro": (
        "Left: Dolan&ndash;Mor&eacute; performance profile &ndash; fraction of held-out benchmarks on which an "
        "algorithm's ERT is within a factor &tau; of the best (value at &tau;=1: how often it is the fastest; "
        "right end: robustness). Right: anytime behaviour &ndash; fraction of all held-out runs that have found the "
        "global minimum after a given fraction of the step budget (solid: new algorithms, dashed: baselines).",
        "左：Dolan&ndash;Mor&eacute; 性能曲线——算法 ERT 在最佳值 &tau; 倍以内的留出基准比例（&tau;=1 处为最快的频率，右端反映鲁棒性）。"
        "右：任意时刻性能——在消耗一定比例步数预算后已找到全局最优的留出运行比例（实线：创新算法，虚线：主流算法）。"),
    "cat_intro": (
        "Efficiency score per benchmark category (all benchmarks of the category, including its development "
        "benchmark; number of benchmarks in brackets).",
        "按基准类别计算的效率得分（包含该类别的开发基准；括号内为基准数）。"),
    "stats_intro": (
        "Friedman test on per-benchmark ERT ranks (held-out set): &chi;<sup>2</sup> = {chi2:.1f}, p = {p:.2g}, "
        "{n} benchmarks, {k} algorithms. Critical-difference diagram (Nemenyi, &alpha; = 0.05): algorithms joined by "
        "a red bar are not significantly different in average rank.",
        "对留出集各基准 ERT 名次做 Friedman 检验：&chi;<sup>2</sup> = {chi2:.1f}，p = {p:.2g}，{n} 个基准，{k} 个算法。"
        "临界差异图（Nemenyi，&alpha; = 0.05）：被红线连接的算法平均名次无显著差异。"),
    "sens_title": ("Sensitivity to the success tolerance", "对成功判据 e<sub>tol</sub> 的敏感性"),
    "unsolved": ("Held-out benchmarks on which no run of this experiment reached E<sub>ref</sub> + e<sub>tol</sub> "
                 "and that therefore contribute no score ({n}): {names}.",
                 "本实验中没有任何运行到达 E<sub>ref</sub> + e<sub>tol</sub>、因而不计入得分的留出基准（{n} 个）：{names}。"),
    "cat_short": ({"cluster": "cluster", "surface": "surface", "zeolite": "zeolite", "bulk": "bulk"},
                  {"cluster": "团簇", "surface": "表面", "zeolite": "分子筛", "bulk": "体相"}),
    "cat_long": ({"cluster": "Gas-phase clusters", "surface": "Supported clusters",
                  "zeolite": "Metal clusters in zeolites", "bulk": "Bulk crystals"},
                 {"cluster": "气相团簇", "surface": "表面负载团簇", "zeolite": "分子筛内金属团簇", "bulk": "体相晶体"}),
    "sens_intro": (
        "The ranking is recomputed with other energy tolerances for a hit (same runs, same E<sub>ref</sub>). "
        "Kendall &tau; compares the efficiency scores with those of the primary ranking (1 = identical order).",
        "用不同的命中能量容差重新计算排名（同一批运行、同一 E<sub>ref</sub>）。Kendall &tau; 衡量与主排名得分顺序的一致性（1 = 顺序完全相同）。"),
    "sens_cols": (["e_tol (eV)", "Kendall &tau;", "Top 3"], ["e_tol (eV)", "Kendall &tau;", "前三名"]),
    "primary": ("primary", "主排名"),
    "cost_intro": (
        "Ionic steps are the primary cost measure. Wall-clock time additionally contains each algorithm's own "
        "overhead (surrogate models, fingerprints, population bookkeeping). Bars: geometric mean over benchmarks of "
        "wall time per ionic step relative to basin hopping. The 'Score (wall-clock)' column of the ranking repeats "
        "the efficiency score with ERT measured in seconds.",
        "主要成本指标是离子步数。墙钟时间还包含算法自身开销（代理模型、指纹、种群管理等）。柱状图为各基准上单个离子步墙钟时间相对"
        "盆地跳跃（BH）的几何平均。排名表中“得分（墙钟）”一列是用秒计 ERT 重新计算的效率得分。"),
    "bench_intro": (
        "E<sub>ref</sub> is the putative global minimum: the lowest energy found by any run of any algorithm "
        "(including development runs) in a structure in which the short-range guard and the slab wall are inactive. "
        "&dagger;: lower-energy structures in which the guard is active (NEP89 holes) were found and are excluded "
        "(see Methods). d<sub>min</sub> ratio: shortest distance involving a search atom divided by "
        "the sum of covalent radii (sanity check of the global minimum).",
        "E<sub>ref</sub> 为推定全局最优：所有算法所有运行（含开发阶段）在短程 guard 与表面势墙均不起作用的结构中找到的最低能量。"
        "&dagger;：发现了 guard 起作用的更低能量结构（NEP89 空洞），已排除（见方法）。d<sub>min</sub> 比值：涉及搜索原子的最短距离"
        "除以共价半径之和（全局最优结构合理性检查）。"),
    "hole_note": ("E<sub>ref</sub> excludes {n} lower-energy structures in which the guard is active: NEP89 holes, "
                  "found in the main experiment, that the guard makes shallower but does not remove "
                  "(lowest {e:.3f} eV; runs of {algs}; see Methods)",
                  "E<sub>ref</sub> 未计入 {n} 个 guard 起作用的更低能量结构：主实验中发现的、guard 只使其变浅而未能消除的 "
                  "NEP89 空洞（最低 {e:.3f} eV；来自 {algs} 的运行；见方法）"),
    "bench_cols": (["Benchmark", "Category", "System", "Atoms (search/total)", "Budget", "e_tol (eV)",
                    "E_ref (eV)", "Found by", "Fastest", "Best ERT", "BH ERT", "Solved by", "d_min ratio"],
                   ["基准", "类别", "体系", "原子数（搜索/总数）", "步数预算", "e_tol (eV)", "E_ref (eV)",
                    "最优发现者", "最快算法", "最佳 ERT", "BH ERT", "解出算法数", "d_min 比值"]),
    "gallery": ("Putative global minima (search atoms in full colour, host atoms faded)",
                "推定全局最优结构（搜索原子为全色，载体原子淡化）"),
    "bench_desc": ("Benchmark descriptions and literature context (click to expand)",
                   "基准描述与文献背景（点击展开；描述为英文）"),
    "curves": ("Success rate versus ionic steps", "成功率随离子步变化"),
    "alg_cols": (["Algorithm", "Key", "Type", "Description", "Reference"],
                 ["算法", "代号", "类型", "描述", "参考文献"]),
    "downloads": ("Downloads", "下载"),
    "dl_items": ([("data/ranking.json", "ranking and all summary statistics"),
                  ("data/metrics.csv", "per benchmark &times; algorithm: SR, ERT, median hit step, wall time"),
                  ("data/runs.csv", "per run: hit step, steps used, best energy, wall time"),
                  ("data/gm_structures.extxyz", "putative global-minimum structures of all benchmarks")],
                 [("data/ranking.json", "排名与全部汇总统计"),
                  ("data/metrics.csv", "每个基准×算法：成功率、ERT、中位命中步、墙钟时间"),
                  ("data/runs.csv", "每次运行：命中步、已用步数、最低能量、墙钟时间"),
                  ("data/gm_structures.extxyz", "所有基准的推定全局最优结构")]),
    "dl_release": ("raw records of every run of all campaigns (JSON with energy traces and best structures) and the "
                   "freeze manifest",
                   "所有批次每次运行的原始记录（JSON，含能量轨迹与最优结构）以及冻结清单"),
    "generated": ("Generated {date} from {nruns:,} runs (campaign '{camp}').",
                  "生成于 {date}，基于 {nruns:,} 次运行（campaign '{camp}'）。"),
}

METHODS = (
    """
<h3>Potential energy surface</h3>
<p>All energies and forces come from the NEP89 foundation potential (nep89_20250409) evaluated on CPU with
calorine, plus a short-range repulsive guard E<sub>g</sub> = &Sigma; k/2 (r<sub>0</sub> &minus; r)<sup>2</sup> for
r &lt; r<sub>0</sub> = f (r<sub>cov,i</sub> + r<sub>cov,j</sub>) with f = 0.75, k = 20 eV/&Aring;<sup>2</sup>, applied to pairs
containing a search atom. NEP89 has unphysical low-energy regions at very short interatomic distances (e.g. a
collapsed TiO<sub>2</sub> phase 30 eV below anatase); the guard removes them and is exactly zero for all
physically sensible structures (the shortest real bonds, N&equiv;N or C&equiv;C, are at 0.80 of the covalent-radius
sum).</p>
<p>Before the main experiment the putative global minima of all benchmarks were screened for artefacts (recomputed
energy, shortest distance, guard energy, connectivity, volume) and hunted for holes with extra runs of
symmetry-constrained random search and particle swarm. Where holes were found the guard was strengthened for that
benchmark only, after checking that the physical minima are unchanged: f = 0.80 for TiO<sub>2</sub> and
SrTiO<sub>3</sub> crystals (dense collapsed phases), f = 0.90 for Si<sub>3</sub>N<sub>4</sub> (Si&ndash;Si 1.65 &Aring;
collapse), and an anion&ndash;anion floor r<sub>0</sub> = 1.9 &Aring; for O&ndash;O in the (Al<sub>2</sub>O<sub>3</sub>)<sub>4</sub>
cluster and for N&ndash;N in the Si<sub>3</sub>N<sub>4</sub> and GaN crystals (NEP89 states with O&ndash;O or
N&equiv;N bonds up to 28 eV below the physical minimum; the shortest physical anion&ndash;anion distances there are
&ge; 2.5 &Aring;), and a cation&ndash;anion floor r<sub>0</sub> = 1.45 &Aring; for Si&ndash;O in the SiO<sub>2</sub> crystal
(a NEP89 state with 1.30 &Aring; Si=O bonds 0.12 eV below the physical minimum; physical Si&ndash;O bonds are
&ge; 1.5 &Aring;). The benchmark descriptions list these non-default guard settings.</p>
<p>The NEP89 evaluation is periodic in all three directions, so on the surface benchmarks search atoms could cross
the vacuum and bind to the frozen, bulk-terminated bottom face of the periodic slab image (an Au<sub>8</sub> state on
TiO<sub>2</sub>(110) found this way lies 1.4 eV below the best top-face structure). Each search atom on a surface
benchmark is therefore held by a flat-bottom wall k/2 (z &minus; z<sub>w</sub>)<sup>2</sup> placed half-way across
the vacuum (about 10 &Aring; above the top face; heights are taken modulo the slab period). The wall is zero for
every structure on the top face: in the more than 2000 slab structures stored before the main experiment no search
atom was higher than 8.4 &Aring;, and in none of the 14,387 best structures of all surface runs (main experiment
included) does a search atom reach the wall (highest atom 9.6 &Aring;, in a run that ended 2.3 eV above the global
minimum). Gas-phase clusters were checked in the same way: no low-energy cluster interacts with its periodic
images (extent &le; 12 &Aring; in 24&ndash;30 &Aring; boxes).</p>
<p>After the main experiment the guard and wall energy was computed for every stored best structure that counts as a
hit (47,789 structures from all campaigns). In 20 of them, all from two held-out crystals, the guard is active: they
lie in NEP89 holes that the guard makes shallower but does not remove. In SiO<sub>2</sub> (bu_Si4O8) four runs (one of SYMM, three of PSO) ended 0.4&ndash;8.3 eV below the best
guard-free structure, with eight Si&ndash;O bonds of 1.30 &Aring; (a deeper form of the Si=O state for which the
1.45 &Aring; floor had been introduced). In SrTiO<sub>3</sub> (bu_Sr4Ti4O12) 16 runs of six algorithms ended 0.23 eV
below it, with Sr&ndash;O bonds of 1.97 &Aring; and Ti&ndash;O bonds of 1.80 &Aring;. E<sub>ref</sub> is therefore the
lowest energy of a stored structure in which the guard and the wall are inactive (energy &le; 0.01 eV); for the other
43 benchmarks this is simply the lowest energy found. With the hole energy as E<sub>ref</sub>, only the runs that fell
into a hole would count as hits. The 20 hole runs are scored like all other runs, since their best energy passed
E<sub>ref</sub> + e<sub>tol</sub> (all 16 SrTiO<sub>3</sub> runs and one of the SiO<sub>2</sub> runs first reached the
guard-free minimum). Counting them as failures changes no rank and no score by more than 0.1; removing both
benchmarks from the score swaps only GA and SMH (ranks 5 and 6).</p>
<h3>Cost measure: ionic steps</h3>
<p>Every energy+force evaluation of the full system is one ionic step &ndash; whether it is part of a local
relaxation, molecular dynamics, a single-point screening or a line search. All algorithms run through the same
evaluation context, which counts the steps and stops a run at its budget. Surrogate-model predictions cost no ionic
steps (their cost appears in wall-clock time).</p>
<h3>Success and efficiency</h3>
<p>A run <i>hits</i> the global minimum at the first ionic step at which its best energy so far is within
e<sub>tol</sub> (normally 0.02 eV) of E<sub>ref</sub>, the lowest energy found for the benchmark by any run in a
structure in which the guard and the wall are inactive (see above).
The expected running time ERT = (&Sigma; steps until hit, or steps used if no hit) / (number of hits) is the
expected number of ionic steps per global-minimum discovery when unsuccessful runs are restarted; a run without a hit
that was stopped by the wall-time cap or by an error is charged the full budget. Efficiency = 1/ERT. Each algorithm runs {seeds} independent seeds on every benchmark with the same per-benchmark budget
({bmin}&ndash;{bmax} ionic steps), fixed before the main experiment from pilot runs of basin hopping and random
search.</p>
<h3>Fairness</h3>
<ul>
<li>Same energy function, local-relaxation convergence threshold (f<sub>max</sub> = 0.05 eV/&Aring;; shared L-BFGS and FIRE
implementations), random structure
generator, search region and distance constraints for every algorithm; no algorithm may inspect the benchmark name
or previous results.</li>
<li>Baselines use the parameters recommended in the literature without tuning.</li>
<li>New algorithms were developed and tuned only on six development benchmarks (marked *); the headline ranking
uses the remaining held-out benchmarks.</li>
<li>Uncertainty from the finite number of seeds is quantified by bootstrap resampling of runs (1000 replicates).</li>
</ul>
<h3>Caveats</h3>
<ul>
<li>E<sub>ref</sub> is a putative global minimum of the NEP89 surface (among structures in which the guard is
inactive), not of DFT. NEP89 errors can change
energy orderings (e.g. it predicts the &beta;-tin-like Si<sub>8</sub> cell slightly below diamond).</li>
<li>ERT is noisy when the success rate is low; consult the confidence intervals and the per-benchmark matrix.</li>
<li>The ranking depends on the benchmark mix and budgets; category rankings are given for this reason.</li>
</ul>
""",
    """
<h3>势能面</h3>
<p>所有能量与力均由 NEP89 通用势（nep89_20250409）通过 calorine 在 CPU 上计算，并叠加短程排斥 guard：
E<sub>g</sub> = &Sigma; k/2 (r<sub>0</sub> &minus; r)<sup>2</sup>（r &lt; r<sub>0</sub> = f (r<sub>cov,i</sub> + r<sub>cov,j</sub>)，f = 0.75，
k = 20 eV/&Aring;<sup>2</sup>），只作用于含搜索原子的原子对。NEP89 在极短原子间距处存在非物理的低能区域（例如一种塌缩的
TiO<sub>2</sub> 相比锐钛矿低 30 eV）；guard 消除了这些空洞，并且对所有物理合理结构严格为零（最短的真实键 N&equiv;N、
C&equiv;C 约为共价半径和的 0.80 倍）。</p>
<p>主实验之前，对所有基准的推定全局最优做了伪影筛查（重算能量、最短距离、guard 能量、连通性、体积），并用对称性约束随机搜索与
粒子群算法额外运行专门搜寻“空洞”。发现空洞的基准只对该基准加强 guard，并确认物理极小值不变：TiO<sub>2</sub> 与
SrTiO<sub>3</sub> 晶体 f = 0.80（致密塌缩相），Si<sub>3</sub>N<sub>4</sub> f = 0.90（Si&ndash;Si 1.65 &Aring; 塌缩），
(Al<sub>2</sub>O<sub>3</sub>)<sub>4</sub> 团簇的 O&ndash;O 以及 Si<sub>3</sub>N<sub>4</sub>、GaN 晶体的 N&ndash;N 设阴离子间下限
r<sub>0</sub> = 1.9 &Aring;（NEP89 中含 O&ndash;O 或 N&equiv;N 键的状态比物理极小值低最多 28 eV；这些体系中真实的最短阴离子间距
&ge; 2.5 &Aring;），SiO<sub>2</sub> 晶体的 Si&ndash;O 设阳离子&ndash;阴离子下限 r<sub>0</sub> = 1.45 &Aring;（NEP89 中含 1.30 &Aring;
Si=O 键的状态比物理极小值低 0.12 eV；真实 Si&ndash;O 键长 &ge; 1.5 &Aring;）。基准说明中列出了这些非默认的 guard 设置。</p>
<p>NEP89 计算在三个方向上都是周期性的，因此在表面基准中，搜索原子可能穿过真空层，吸附到周期镜像中冻结的体相截断底面上（用这种方式
找到的 TiO<sub>2</sub>(110) 上 Au<sub>8</sub> 结构比最佳顶面结构低 1.4 eV）。因此表面基准中的每个搜索原子都受一个平底势墙
k/2 (z &minus; z<sub>w</sub>)<sup>2</sup> 约束，势墙位于真空层中点（约在顶面以上 10 &Aring;，高度按板周期取模）。对顶面上的任何结构
势墙严格为零：主实验前保存的 2000 多个表面结构中，搜索原子的最大高度为 8.4 &Aring;；包括主实验在内，全部 14,387 次表面运行的最优结构中没有任何搜索原子触及势墙（最高的原子为 9.6 &Aring;，出现在一个结束时比全局最优高 2.3 eV 的运行中）。气相团簇也做了同样检查：没有低能团簇与其
周期镜像相互作用（尺寸 &le; 12 &Aring;，盒长 24&ndash;30 &Aring;）。</p>
<p>主实验结束后，对所有计为命中的已保存最优结构（所有批次共 47,789 个）计算了 guard 与势墙能量。其中 20 个结构中 guard 起作用，
全部来自两个留出晶体：它们位于 guard 只使其变浅、但没有消除的 NEP89 空洞中。SiO<sub>2</sub>（bu_Si4O8）中有
4 次运行（SYMM 1 次、PSO 3 次）停在比最佳无 guard 结构低 0.4&ndash;8.3 eV 的结构上，其中有 8 根 1.30 &Aring; 的 Si&ndash;O 键
（即当初为之引入 1.45 &Aring; 下限的 Si=O 状态的更深形式）；SrTiO<sub>3</sub>（bu_Sr4Ti4O12）中有 6 个算法的 16 次运行停在比它低
0.23 eV 的结构上，其中 Sr&ndash;O 键长 1.97 &Aring;、Ti&ndash;O 键长 1.80 &Aring;。因此 E<sub>ref</sub> 定义为 guard 与势墙均不起作用
（能量 &le; 0.01 eV）的已保存结构中的最低能量；对其余 43 个基准，它就是找到的最低能量。若以空洞能量作为 E<sub>ref</sub>，只有掉进空洞的
运行才算命中。这 20 次空洞运行与其他运行同样计分，因为它们的最低能量越过了 E<sub>ref</sub> + e<sub>tol</sub>（全部 16 次
SrTiO<sub>3</sub> 运行和 1 次 SiO<sub>2</sub> 运行在此之前先到达了无 guard 的最低结构）。把它们计为失败不改变任何名次，得分变化不超过
0.1；把这两个基准从得分中去掉只交换 GA 与 SMH（第 5、6 名）。</p>
<h3>成本度量：离子步</h3>
<p>每一次对整个体系的能量+力计算记为一个离子步——无论它属于局部弛豫、分子动力学、单点筛选还是线搜索。所有算法都通过同一个
计算上下文调用势函数，由它统一计数，并在达到预算时终止运行。代理模型的预测不计离子步（其开销体现在墙钟时间中）。</p>
<h3>成功与效率</h3>
<p>一次运行在其历史最低能量首次进入 E<sub>ref</sub> + e<sub>tol</sub>（通常 0.02 eV）范围内的离子步记为<i>命中</i>全局最优，
E<sub>ref</sub> 为所有运行在该基准上找到的、guard 与势墙均不起作用的结构中的最低能量（见上文）。期望运行时间 ERT =（各运行命中所用步数之和，未命中则计已用步数）/ 命中次数，
即在失败后重启的策略下每发现一次全局最优所需的期望离子步数；因墙钟上限或错误而中止且未命中的运行计全部预算。效率 = 1/ERT。每个算法在每个基准上用相同预算独立运行 {seeds} 个随机种子
（{bmin}&ndash;{bmax} 离子步），预算在主实验前根据盆地跳跃与随机搜索的预实验确定。</p>
<h3>公平性</h3>
<ul>
<li>所有算法使用相同的能量函数、局部弛豫收敛阈值（f<sub>max</sub> = 0.05 eV/&Aring;；共用的 L-BFGS 与 FIRE 实现）、随机结构生成器、搜索区域与距离约束；
任何算法都不能读取基准名称或已有结果。</li>
<li>主流算法采用文献推荐参数，不做调参。</li>
<li>创新算法只在 6 个开发基准（标 *）上开发与调参；主排名只使用其余留出基准。</li>
<li>有限种子数带来的不确定性通过对运行做 bootstrap 重抽样（1000 次）量化。</li>
</ul>
<h3>局限</h3>
<ul>
<li>E<sub>ref</sub> 是 NEP89 势能面上（guard 不起作用的结构中）的推定全局最优，而非 DFT 全局最优。NEP89 的误差可能改变能量排序（例如它预测
&beta;-tin 型 Si<sub>8</sub> 略低于金刚石结构）。</li>
<li>成功率较低时 ERT 噪声较大，请参考置信区间与逐基准矩阵。</li>
<li>排名取决于基准组成与预算设置，因此同时给出分类排名。</li>
</ul>
""")
