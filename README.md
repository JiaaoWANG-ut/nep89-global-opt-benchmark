# NEP89 global structure-search benchmark

**Report:** [English](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/) ·
[中文](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/zh.html) ·
**Raw data:** [releases](https://github.com/JiaaoWANG-ut/nep89-global-opt-benchmark/releases)

**100-seed extension:** [English](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/extension100/) ·
[中文](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/extension100/zh.html) ·
[extension data](https://github.com/JiaaoWANG-ut/nep89-global-opt-benchmark/releases/tag/v1.1)

**Second 100-seed extension:** [English](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/extension100b/) ·
[中文](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/extension100b/zh.html) ·
[extension data](https://github.com/JiaaoWANG-ut/nep89-global-opt-benchmark/releases/tag/v1.2)

A benchmark of 22 global structure-search algorithms (12 established methods and 10 new ones written for
this study) on 45 problems with the NEP89 foundation potential: 13 gas-phase clusters, 11 clusters
supported on surfaces, 9 metal clusters in zeolites and 12 bulk crystals (variable cell), with 4–55
search atoms and up to 332 atoms in total. Every algorithm ran 50 independent seeds on every benchmark
with the same budget of ionic steps (5,300–150,000 per benchmark). The main measure is efficiency per
ionic step (one NEP89 energy and force evaluation of the whole system): the expected running time
ERT, i.e. the expected number of ionic steps per discovery of the global minimum.

<!-- RESULTS -->

## Results

Main experiment: 49,500 runs (22 algorithms × 45 benchmarks × 50 seeds), 1,616 million ionic steps, 5,465 CPU core-hours. Ranking by the efficiency score on the 39 held-out benchmarks:

| # | Algorithm | Type | Score (95% CI) | GMRE | Speed-up vs BH | Pooled SR | Score (wall-clock) |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | GOFEE (Gaussian-process surrogate search) (`gofee`) | baseline | 82.7 [81.3, 84.2] | 0.304 | 5.44× | 77% | 41.3 |
| 2 | **Online-surrogate pre-relaxation BH** (`surr`) | new | 78.5 [77.7, 79.2] | 0.226 | 4.05× | 93% | 58.3 |
| 3 | **Adaptive memetic algorithm (GA + BH chains)** (`memetic`) | new | 68.4 [67.6, 69.1] | 0.113 | 2.02× | 86% | 84.1 |
| 4 | **Adaptive operator selection basin hopping** (`aosbh`) | new | 67.9 [67.3, 68.6] | 0.109 | 1.96× | 89% | 84.3 |
| 5 | **Soft-mode guided hopping** (`softmode`) | new | 66.9 [66.2, 67.7] | 0.102 | 1.82× | 84% | 82.3 |
| 6 | Genetic algorithm (cut-and-splice) (`ga`) | baseline | 66.4 [65.8, 66.9] | 0.098 | 1.76× | 89% | 82.9 |
| 7 | **Tabu basin hopping** (`tabu`) | new | 64.6 [63.9, 65.3] | 0.086 | 1.55× | 87% | 80.9 |
| 8 | **Symmetry-seeded search** (`symm`) | new | 63.6 [62.9, 64.2] | 0.081 | 1.45× | 82% | 79.4 |
| 9 | Particle swarm optimisation (CALYPSO) (`pso`) | baseline | 63.4 [62.6, 64.3] | 0.080 | 1.44× | 76% | 78.5 |
| 10 | USPEX evolutionary algorithm (`uspex`) | baseline | 62.3 [61.7, 62.9] | 0.074 | 1.32× | 85% | 78.3 |
| 11 | **Energy-guided basin hopping** (`egbh`) | new | 60.5 [59.6, 61.4] | 0.065 | 1.14× | 71% | 76.0 |
| 12 | **Racing basin hopping** (`race`) | new | 59.3 [58.5, 60.1] | 0.060 | 1.05× | 74% | 74.9 |
| 13 | Differential evolution (`de`) | baseline | 59.2 [58.5, 60.0] | 0.060 | 1.07× | 82% | 75.4 |
| 14 | **Cut-and-regrow (learned-site growth)** (`growth`) | new | 59.0 [58.1, 59.9] | 0.059 | 1.37× | 70% | 73.2 |
| 15 | Basin hopping (`bh`) | baseline | 58.2 [57.2, 59.2] | 0.056 | 1.00× | 79% | 75.1 |
| 16 | Artificial bee colony (ABCluster) (`abc`) | baseline | 53.7 [53.0, 54.5] | 0.041 | 0.80× | 71% | 68.9 |
| 17 | **Metadynamics-biased basin hopping** (`metabh`) | new | 53.2 [52.1, 54.3] | 0.039 | 0.71× | 67% | 69.8 |
| 18 | Stochastic surface walking (`ssw`) | baseline | 51.7 [50.6, 52.8] | 0.036 | 0.62× | 61% | 69.0 |
| 19 | Minima hopping (`mh`) | baseline | 50.1 [49.3, 51.0] | 0.032 | 0.57× | 65% | 67.9 |
| 20 | Parallel-tempering basin hopping (`ptbh`) | baseline | 49.1 [48.2, 50.0] | 0.030 | 0.53× | 65% | 66.0 |
| 21 | Random search (`rs`) | baseline | 34.6 [33.3, 36.0] | 0.011 | 0.22× | 35% | 49.1 |
| 22 | Simulated annealing (MD) (`sa`) | baseline | 14.0 [11.9, 16.0] | 0.003 | 0.05× | 12% | 24.2 |

Score 100 = fastest on every benchmark; every factor of 10 in ERT costs 33 points. GMRE: geometric-mean efficiency relative to the best algorithm. Speed-up vs BH: geometric mean of ERT<sub>BH</sub>/ERT over the benchmarks solved by both. Pooled SR: fraction of all held-out runs that found the global minimum.

Key findings:

* **GOFEE (Gaussian-process surrogate search)** ranks first (score 82.7, 95% CI [81.3, 84.2], first in 100% of bootstrap replicates), followed by Online-surrogate pre-relaxation BH (78.5) and Adaptive memetic algorithm (GA + BH chains) (68.4).
* Best new algorithm **Online-surrogate pre-relaxation BH** vs best baseline **GOFEE (Gaussian-process surrogate search)**: 1.34× more ionic steps per global-minimum discovery (geometric mean over the 39 benchmarks solved by both).
* 0 of 10 new algorithms score higher than every baseline.
* Basin hopping (literature settings) ranks 15 of 22; the top algorithm is 5.44× as efficient as BH (geometric mean of the ERT ratio over the benchmarks solved by both).
* Category winners: clusters – GOFEE (Gaussian-process surrogate search) (83); surfaces – GOFEE (Gaussian-process surrogate search) (96); zeolites – GOFEE (Gaussian-process surrogate search) (95); bulks – Symmetry-seeded search (90).
* Most robust: **Online-surrogate pre-relaxation BH** found the global minimum in 93% of all runs and on 39/39 benchmarks.
* Measured in wall-clock time instead of ionic steps, the best algorithm is **Adaptive operator selection basin hopping** (score 84.3).
* Hardest benchmarks: ze_Pt9_MOR (solved by 18/22), cl_Ag27Cu7 (solved by 20/22), cl_Cu38 (solved by 20/22).

Per-benchmark results, success-rate curves, statistics and the global-minimum structures are in the [report](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/).

<!-- /RESULTS -->

## Four-system extension

The separate extension adds Ni19, Au10 on rutile TiO2(110), Pd6 in MFI and a Ga4O6 crystal:
22 algorithms × 4 systems × 100 seeds = 8,800 runs. SurrBH leads this four-system ranking
(86.3), followed by SMH (79.5) and AOS-BH (78.2). All 400 GOFEE runs reached the one-hour
cap, although 278 had already hit the reference energy window; time limits affect the comparison.
The benchmark sets differ, so these scores must not be directly compared with the original ranking.
Reference structures passed the guard, energy, connectivity and applicable volume checks;
NEP89 minima are not DFT global minima. See the [extension report](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/extension100/)
for details and per-system metrics.

## Second four-system extension

Al13, Pt8 on CeO2(111), Au8 in BEA and variable-cell Al4O6 were tested with
22 algorithms and 100 independent seeds per cell: 8,800 runs and 278,118,054
ionic steps. SurrBH leads this separate ranking (86.7), followed by cut-and-regrow
growth (72.1) and the adaptive memetic algorithm (69.4). These systems were
selected from earlier candidates, not randomly sampled; Pt8/CeO2 and Al4O6
were previously set aside as relatively easy. Scores cannot be directly compared
across the different benchmark sets.

GOFEE reached the one-hour cap in 398 of 400 runs, including 243 that had
already hit the reference energy window; two completed without timeout and hit.
An unsuccessful timeout is charged its full ionic-step budget. Time limits can
affect the ranking. The four reference structures passed energy re-evaluation,
guard, minimum-distance, connectivity and applicable volume checks; no stored
candidate hits screened had an active guard or wall. These are putative NEP89
minima, not DFT global minima. The surface and zeolite hosts are fixed, and their
reference structures extend outside the initial sampling regions. See the
[second extension report](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/extension100b/)
for limitations and per-system metrics.

## How the comparison is made

* **One evaluation context for all algorithms** (`gopt/context.py`). It counts ionic steps, stops a
  run at its budget and records the best energy after every step. All algorithms use the same
  relaxers, structure operators and constraints.
* **Hit:** a run hits the global minimum at the first ionic step at which its best energy is within
  `e_tol` (normally 0.02 eV) of E<sub>ref</sub>, the lowest energy found for the benchmark by any run in a
  structure in which the guard and the slab wall are inactive.
* **ERT** = (Σ steps until the hit, or the steps used if there is no hit; a run without a hit that was
  stopped by the wall-time cap or by an error is charged the full budget) / (number of hits). Efficiency = 1/ERT.
* **Score:** score = 100 × mean over the 39 held-out benchmarks of
  clip(1 + log<sub>10</sub>(ERT<sub>best</sub>/ERT)/3, 0, 1), where ERT<sub>best</sub> is the best
  ERT of any algorithm on the benchmark: 100 means the fastest algorithm everywhere, 0 means 1000×
  slower than the best or no hit. The six development benchmarks, on which the new algorithms were
  tuned, are scored separately.
* **NEP89 artefacts:** NEP89 has unphysical short-range holes. A short-range repulsive guard (and,
  for slabs, a vacuum wall) keeps structures out of them. Every putative global minimum was checked
  again by re-evaluation, minimum distances, guard energy, connectivity and density, and the guard
  energy of every structure that counts as a hit was computed after the main experiment. In two held-out
  crystals (SiO<sub>2</sub>, SrTiO<sub>3</sub>) 20 of the 49,500 runs ended in NEP89 holes that the
  guard makes shallower but does not remove; they do not define E<sub>ref</sub>, and counting them as
  failures leaves the ranking unchanged. See the Methods section of the report.

## Repository layout

| path | content |
|---|---|
| `gopt/` | evaluation context, problems, guard, NEP89 wrapper, shared operators, runner, report code |
| `gopt/algorithms/` | the 22 frozen algorithms (`<key>.py`) and their private helpers (`_<key>_*.py`) |
| `benchmarks/<name>/` | `problem.json` (atoms, region, budget, guard), substrate structures, build scripts |
| `scripts/` | task generation, main launch, GM validation, freeze manifest, report |
| `farm/` | file-queue task farm for Slurm: one multi-node job, one core per run |
| `docs/` | developer rules (`DEVELOPER_GUIDE.md`), algorithm texts, report notes, `freeze_manifest.json` |
| `dev_history/` | intermediate algorithm versions written during development (not used) |

`docs/freeze_manifest.json` lists the NEP89 model checksum, checksums of all shared code, the hash
of every algorithm (also stored in every result record) and of every benchmark. It was written at the
code freeze (`frozen_at`) and refreshed at publication (`written_at`) only for files that do not affect
runs (the report code); the NEP89, algorithm, benchmark and runner checksums are those of the freeze.

## Reproduce

Requirements: numpy, scipy, ase, matplotlib and calorine (CPU NEP engine). The study used Python
3.12, numpy 2.5, scipy 1.18, ase 3.29, calorine 4.1 and matplotlib 3.11 on NERSC Perlmutter CPU
nodes.

```bash
# NEP89 model: GPUMD repository, potentials/nep/nep89_20250409/nep89_20250409.txt
export GOPT_NEP_MODEL=/path/to/nep89_20250409.txt
export PYTHONPATH=$PWD OMP_NUM_THREADS=1

# one run: basin hopping on Pt13, seed 0, full budget
python -m gopt.run --bench cl_Pt13 --alg bh --seeds 0 --campaign test --print

# the main experiment on a Slurm cluster (farm/farm.sbatch contains site-specific paths and account)
sbatch farm/farm.sbatch
python scripts/launch_main.py --algs all --seeds 0-49 --campaign main --submit
python farm/wait.py 'main-*'

# the report
python scripts/report.py --campaign main --out site
```

Results are written to `results/<campaign>/<benchmark>/<algorithm>/s<seed>.json`. The raw records
of the study (all campaigns) are attached to the GitHub release.

## 中文简介

新增四个体系、每种算法运行 100 个独立种子的扩展实验见[中文报告](https://jiaaowang-ut.github.io/nep89-global-opt-benchmark/extension100/zh.html)。
扩展排名只基于这四个体系，与原研究的得分不能直接比较。

本项目用 NEP89 通用势比较 22 种全局结构搜索算法（12 种已有方法与 10 种为本研究新设计的方法），覆盖 45 个
基准问题：13 个气相团簇、11 个表面负载团簇、9 个分子筛孔道内的金属团簇和 12 个体相晶体（可变晶胞）；搜索原子
4–55 个，体系总原子数最多 332 个。每个算法在每个基准上用相同的离子步预算独立运行 50 个随机种子。主要指标是
每离子步的搜索效率：期望运行时间 ERT，即每发现一次全局最优所需的期望离子步数（一次离子步为对整个体系的一次
NEP89 能量与力计算）。完整的方法、结果与讨论见上方的中英文报告。

<!-- RESULTS_ZH -->

主要结果（排名表见上方 Results 一节，得分按 39 个留出基准计算）：

* **GOFEE (Gaussian-process surrogate search)** 排名第一（得分 82.7，95% 置信区间 [81.3, 84.2]，在 100% 的 bootstrap 样本中第一），其后为 Online-surrogate pre-relaxation BH（78.5）、Adaptive memetic algorithm (GA + BH chains)（68.4）。
* 最佳创新算法 **Online-surrogate pre-relaxation BH** 对比最佳主流算法 **GOFEE (Gaussian-process surrogate search)**：每发现一次全局最优所需离子步为后者的 1.34 倍（两者都解出的 39 个基准上的几何平均）。
* 10 个创新算法中有 0 个得分高于所有主流算法。
* 盆地跳跃（文献参数）排名第 15/22；第一名算法的效率是 BH 的 5.44 倍（两者都解出的基准上 ERT 比值的几何平均）。
* 各类别第一：团簇——GOFEE (Gaussian-process surrogate search)（83）；表面——GOFEE (Gaussian-process surrogate search)（96）；分子筛——GOFEE (Gaussian-process surrogate search)（95）；体相——Symmetry-seeded search（90）。
* 最稳健：**Online-surrogate pre-relaxation BH** 在 93% 的运行中找到全局最优，解出 39/39 个基准。
* 若以墙钟时间而非离子步计，最佳算法为 **Adaptive operator selection basin hopping**（得分 84.3）。
* 最难的基准：ze_Pt9_MOR（18/22 个算法解出）、cl_Ag27Cu7（20/22 个算法解出）、cl_Cu38（20/22 个算法解出）。

<!-- /RESULTS_ZH -->
