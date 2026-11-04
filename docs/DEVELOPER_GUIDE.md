# Developer guide (for contributors and sub-agents)

Project root: `/global/cfs/cdirs/m4517/jiaao/test/global-opt-game`

Goal: compare global-optimisation (GO) algorithms on ~40 atomistic benchmarks
(gas-phase clusters, supported clusters, metal clusters inside zeolites, bulk
crystal-structure prediction) using the NEP89 machine-learned potential, and
rank them by **efficiency = discoveries of the global minimum per ionic step**.

## 1. Environment

```bash
cd /global/cfs/cdirs/m4517/jiaao/test/global-opt-game
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
```

Available packages: numpy, scipy, ase, calorine, pandas, matplotlib,
scikit-learn. **Do not pip-install anything** into the shared venv.

You are on a shared login node. Local runs must be tiny (single process,
< ~60 s CPU). Everything heavier goes to the task farm.

## 2. Task farm (640 cores, always on)

A 5-node job runs a manager per node that executes queued shell commands, one
core each, with the venv + `PYTHONPATH` already set and `cwd` = project root.

```bash
# gopt runs: one farm task per (bench, alg, seed chunk)
python scripts/make_tasks.py --prefix dev-ga --campaign dev-ga --algs ga \
    --benches cl_Pt13,cl_Pt7Cu6,bu_Si8,bu_Ti4O8,su_Pt8_MgO100,ze_Pt4_CHA --seeds 0-9 \
    --priority 50 --timeout 7200 --submit
# arbitrary command
python farm/submit.py --name build-cl-Au20-pilot --cmd "python benchmarks/cl_Au20/build.py" \
    --priority 30 --timeout 3600
python farm/wait.py 'dev-ga-*' --timeout 7200   # blocks; prints failures + log tails
python farm/submit.py --status                    # queue + node overview
```

* Task names must start with **your own prefix** (given in your assignment).
  Names are unique: resubmitting an existing name is silently skipped; use a
  new name (e.g. add `-v2`) or `--force` (make_tasks/submit) to rerun.
* Logs: `farm/queue/logs/p<priority>_<name>.log`; outcome json in
  `farm/queue/{done,failed}/`.
* Lower priority number runs first (benchmark pilots 20-30, algorithm
  development 50). Never touch `farm/queue/control`, other people's tasks, or
  the SLURM job.
* `gopt.run` skips seeds whose result file already exists; use a new
  `--campaign` (or `--tag`) after changing your code, e.g. `dev-ga-v2`.

## 3. Code layout and ownership

```
gopt/nep.py        FastNEP: thin wrapper around calorine's NEP CPU engine
gopt/problem.py    Problem: benchmark definition + random structure generation
gopt/context.py    Context: the ONLY energy entry point, step counting, relax, md
gopt/guard.py      short-range repulsive guard added to NEP89 (see section 4)
gopt/ops.py        shared structure operators (no energy calls)
gopt/run.py        runner CLI (python -m gopt.run ...)
gopt/analysis.py   hit detection, ERT metrics
gopt/algorithms/   one module per algorithm (rs.py, bh.py are references)
benchmarks/<name>/ problem.json (+ substrate.extxyz, build.py)
results/<campaign>/<bench>/<alg>/sNNN.json
scripts/           make_tasks.py, devstats.py, ...
farm/              task farm
```

**Only create/modify the files you own** (your algorithm module, optional
private helper modules `gopt/algorithms/_<yourkey>_*.py`, your benchmark
directories, scratch files under `scratch/<yourprefix>/`). Do not edit shared
files (`gopt/*.py`, `gopt/algorithms/__init__.py`, `rs.py`, `bh.py`, farm,
scripts, other benchmarks). If you find a bug or need a feature in shared
code, work around it in your own file and describe the issue (with a proposed
patch) in your final report.

## 4. Benchmarks

`benchmarks/<name>/problem.json` (see the docstring of `gopt/problem.py`):

| key | meaning |
|---|---|
| `name` | directory name; prefix `cl_` cluster, `su_` surface, `ze_` zeolite, `bu_` bulk |
| `title`, `description` | display name; description with literature context |
| `category` | `cluster`, `surface`, `zeolite` or `bulk` |
| `search_symbols` | elements of the atoms placed by the search (appended after the substrate) |
| `substrate_file` | extxyz of the pre-relaxed host (surface/zeolite), defines cell+pbc |
| `cell` | 3x3 cell when there is no substrate (cluster box, bulk initial guess) |
| `pbc` | always `[true,true,true]` (NEP CPU treats cells as periodic) |
| `region` | where search atoms are generated: `{"type":"sphere","center":[..],"radius":r}`, `{"type":"box","lo":[..],"hi":[..]}`, `{"type":"cell"}` |
| `variable_cell`, `volume_per_atom` | bulk only: cell is optimised; random cells use this volume |
| `connected` | generated search atoms must form one bonded cluster (default: true unless bulk) |
| `budget` | ionic-step budget of one run |
| `fmax` (0.05 eV/A), `e_tol` (0.02 eV), `max_relax_steps` (500), `dmin_factor` (0.7) | relaxation / success settings |
| `guard_factor` (0.75), `guard_k` (20 eV/A^2) | short-range guard (below); keep the defaults |

Substrate atoms are frozen (`fixed_substrate: "all"`, default). Cluster
benchmarks use a cubic box with >= 12 A of vacuum around the cluster; slabs
have >= 15 A vacuum. Full systems must have < 1500 atoms (prefer < 400; NEP89
costs ~70 us per atom per ionic step on one core).

Dev benchmarks (the only ones algorithms may be tuned on): `cl_Pt13`,
`cl_Pt7Cu6`, `bu_Si8`, `bu_Ti4O8`, `su_Pt8_MgO100`, `ze_Pt4_CHA`.

**Energy = NEP89 + short-range guard.** NEP89 has unphysical holes at short
interatomic distances (e.g. Ti4O8 collapses to 4.8 A^3/atom with Ti-Ti 1.9 A,
7.9 eV below anatase). `gopt/guard.py` adds
`sum k/2 (r0-r)^2` for pairs with `r < r0 = guard_factor*(rcov_i+rcov_j)`
involving at least one search atom. It is exactly zero for physical
structures (tightest real bonds, N≡N/C≡C, are at 0.80 x the radius sum) and is
included in energies, forces, stress and per-atom energies; `ctx` handles it
transparently. Benchmark builders must still check the NEP89 global minimum
for physical sanity: run `python scripts/best_structure.py --campaign <pilot>
--bench <name>` and compare minimum distances / coordination / volume with
experiment or the literature GM. NEP89 can also have wrong phase ordering
(it puts a 6-coordinated beta-tin-like Si8 50 meV/atom below diamond);
document such cases in `description` instead of hiding them.

Per-benchmark guard extras (all part of the benchmark hash): `guard_pairs`
(`{"O-O": 1.9}` raises r0 for an element pair, against anion-anion or too-short
cation-anion holes), and, automatically for every `surface` benchmark, a
flat-bottom wall half-way across the vacuum (`Problem.z_wall`). NEP89 is
periodic along the slab normal too, and without the wall search atoms can cross
the vacuum and bind to the frozen bottom face of the periodic image.

## 5. Algorithms

`gopt/algorithms/<key>.py` must define

```python
LABEL = "Basin hopping"                      # display name
FAMILY = "baseline"                          # or "innovative"
REFERENCE = "Wales & Doye, JPCA 101, 5111 (1997)"
DESCRIPTION = "One paragraph: what it does and how it differs from others."
DEFAULTS = dict(kT=0.1, step=0.5)            # all tunable parameters

def run(ctx, rng, **params):
    """Search until ctx.evaluate raises BudgetExhausted (do not catch it)."""
```

Read `rs.py` and `bh.py` first: they are the reference implementations.

### Context API (`ctx`)
* `ctx.evaluate(pos, cell=None) -> (energy, forces)` — **one ionic step**.
  `pos` is the full (n,3) Cartesian array (substrate first, then search atoms).
  Forces on frozen atoms are zero. Raises `BudgetExhausted` when the budget is
  spent; the runner catches it and stores the result.
* `ctx.relax(pos, cell=None, fmax=None, steps=None, method="lbfgs"|"fire", abort=None)`
  -> `RelaxResult(pos, cell, e, converged, nsteps, fmax, aborted)`; every force
  call inside counts. Variable-cell problems also relax the cell.
  `abort(i, e, fmax_now) -> True` stops a relaxation early (result `aborted`).
* `ctx.md(pos, vel, dt=1.0, nsteps=100, callback=None, cell=None)` velocity
  Verlet (fs, A/fs); `ops.mb_velocities(p, T, rng)` gives initial velocities.
* `ctx.atomic_energies()` NEP per-atom energies of the last evaluation (free).
* `ctx.stress(cell)` Voigt stress of the last evaluation (variable cell).
* `ctx.remaining()`, `ctx.nsteps`, `ctx.best_e`, `ctx.best_pos`.
* `ctx.stats` dict: put small JSON-serialisable diagnostics here.

### Problem API (`p = ctx.p`)
`n, n_sub, n_search, search_idx, free_mask, symbols, numbers, cell0, pbc,
category, variable_cell, region, connected, vpa, rcov, species,
dmin(i, j_idx), bond_cut(i, j_idx), in_region(x), sample_point(rng, cell),
random_cell(rng), place(pos, idx, rng, cell) (re-place atoms idx randomly,
may return None), random_structure(rng) -> (pos, cell), min_dist_ok(pos, cell),
cluster_components(pos, cell), to_atoms(pos, cell)`.

### Shared operators (`from .. import ops`)
`center_of_search, unwrap_search, random_unit, random_rotation, rotation_about,
repair_overlaps, detached_atoms, displace, rotate_subset, swap_unlike,
relocate_atoms, cut_and_splice, strain_cell, mb_velocities, fingerprint,
fp_distance, same_structure`. Read their docstrings in `gopt/ops.py`.

### Rules (fairness)
1. Energies/forces only through `ctx` (no FastNEP/calorine/other calculators).
   A surrogate model trained on data obtained from `ctx` is allowed; its own
   cost is free in ionic steps but shows up in wall time (also reported).
2. The same `DEFAULTS` for every benchmark. Parameters may depend only on
   generic properties (`category`, `n_search`, species, `variable_cell`,
   covalent radii) — never on the benchmark name or known answers. Never read
   `results/` or any reference energy inside an algorithm.
3. Randomness only from the given `rng` (`numpy.random.Generator`).
4. Move only search atoms (and the cell for bulk). Keep species counts.
   Bimetallic/compound systems: atoms of different elements are distinct
   (`ops.swap_unlike` exchanges positions).
5. Baselines: implement the published method faithfully with standard
   parameter values from the literature; only fix values that are clearly
   broken for these systems (document it). Innovative methods may be tuned on
   the six dev benchmarks only.

### Robustness and speed
* Must work for all four categories, single- and multi-species systems,
  4..60 search atoms, 0..1500 substrate atoms, and variable-cell bulk.
* Never crash; never loop without calling `ctx` (if `run` returns, the runner
  restarts it; if a call makes no evaluation the run stops). A watchdog in
  `gopt.run` ends a seed that makes no ionic step for 30 min
  (`GOPT_STALL_S`); the seed is recorded as an error.
* Keep Python overhead small: NEP costs ~0.5 ms (13 atoms) to ~20 ms (300
  atoms) per step; avoid per-step O(n_atoms^2) Python loops. Surrogate methods
  must still finish a default-budget run of `su_Pt8_MgO100` within ~1 h.
* Supported clusters must stay on the surface, zeolite clusters inside the
  pores: re-seat detached atoms (`ops.detached_atoms`, `ops.relocate_atoms`).

### Development loop
```bash
# smoke test (seconds)
python -m gopt.run --bench cl_Pt13 --alg <key> --seeds 0 --budget 2000 --campaign dev-<key> --print --overwrite
# full dev test on the farm
python scripts/make_tasks.py --prefix dev-<key> --campaign dev-<key> --algs <key> \
    --benches cl_Pt13,cl_Pt7Cu6,bu_Si8,bu_Ti4O8,su_Pt8_MgO100,ze_Pt4_CHA --seeds 0-9 --timeout 7200 --submit
python farm/wait.py 'dev-<key>-*'
python scripts/devstats.py --campaign dev-<key>
python scripts/devstats.py --campaign dev-ref        # BH / RS reference runs
```
`devstats` prints success rate (SR), expected running time (ERT, ionic steps
per global-minimum discovery; lower is better), median hit step, median energy
gap to the best-known energy, wall time per run and ms per ionic step.

## 6. Metric

Hit step of a run = first ionic step whose best-so-far energy is within
`e_tol` of the lowest energy found for that benchmark by any run of any
algorithm. The report (`gopt/report/data.py`) takes this reference only from
stored structures in which the guard and the slab wall are inactive (energy
<= 0.01 eV), so that a run that falls into a hole the guard did not close
does not define it; `devstats` uses the plain lowest energy. ERT = (sum over runs of hit step, or the steps used if no hit;
the full budget for a run stopped by the wall-time cap or an error) /
(number of hits). Efficiency = 1/ERT. Main campaign: every algorithm x
benchmark with 50 independent seeds.
