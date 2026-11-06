# Proposed Pd7 / MgO(001) benchmark

**Status: provisional.** A later 10-run system pilot is recorded in
`../pilot_audit.json`; the checks described below predate that pilot. Seven Pd atoms are searched on a
frozen, stoichiometric MgO(001) slab. Supported small Pd clusters on oxide
surfaces are a plausible heterogeneous-catalysis model; seven atoms is an
arbitrary deliberately new size, not a reconstruction of a specific experiment.
This composition is not one of the original 45 or either four-system extension,
nor was a same-named pilot definition found in the candidate inventory. Selection
was manual, not random or statistically representative.

The 144-atom host is byte-for-byte `benchmarks/su_Pt8_MgO100/substrate.extxyz`
(Mg72O72, 3×3 lateral conventional cells, four atomic layers); region and
30,000-ionic-step budget are inherited from that development benchmark, not
from a Pd pilot. `scripts/build_dev_benchmarks.py` describes its original
construction: NEP-relaxed rocksalt bulk, a slab with lower layers held during
host relaxation, and then the *entire* host frozen for search. Reuse means this
is not an independently sourced host, and it overlaps an algorithm-development
system. It is an ideal flat, stoichiometric support, without defects,
temperature, charge or environmental chemistry. No Pd-host accuracy is implied.
The Mg/O/Pd symbols all appear in the NEP89 model header; coverage is not
validation of this interaction. The project model is GPUMD
`potentials/nep/nep89_20250409/nep89_20250409.txt`, SHA-256
`75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1`.

The cell is fully periodic (NEP CPU is fully periodic), even in vacuum: slab
thickness 6.3112 Å, vacuum 20.0953 Å, lateral repeat 12.6097 Å. The search
box starts 1.6 Å above the highest slab atom and ends at z=13.8112 Å, below the
automatic periodic-vacuum wall starting at z=16.3588 Å. A direct wall probe
passed. This soft wall is a penalty, not an absolute topological constraint:
large moves can traverse barriers, and atoms may detach or bind the wrong side.
Periodic image and finite-size effects remain; even image separations above
the 6 Å NEP radial cutoff do not establish convergence of a many-body host.
The guard and wall modify the objective. Before using a result, explicitly
screen positions using minimum-image geometry, connectivity, attachment to
the correct face, guard/wall energy, and energy/force consistency.

## Identity and checks

- `Problem.hash`: `c5519a69befd`. The project hash omits model bytes, region,
  budget, and the automatic surface wall: additionally freeze those separately.
  Loading all definitions in `benchmarks`, `benchmarks_extended100` and
  `benchmarks_extended100b` found no matching hash.
- `problem.json` SHA-256: `e181f970bb469d54ed00c8e599278fa3e2079294b557c7d54f54767edcb8f445`.
- host SHA-256: `1912ffde4fd6838a6b0e5035896a163400d77f6720804899d75475166430ab34`.
- JSON and `Problem` load, composition, fully fixed host, periodicity and NEP
  species checked. Host minimum pair distance 2.0532 Å; per-Mg nearest O
  2.0532–2.1029 Å.
- Predeclared seeds 0, 1, 2: one random connected initial structure, true
  NEP+guard/wall and up to 50 L-BFGS steps each; all are reported, not selected.
  Final guard+wall energies are zero, all clusters connected, minimum Pd–host
  distances 2.0882/2.0824/2.1260 Å, minimum Pd–Pd 2.5681/2.5041/2.6019 Å,
  approximate unwrapped cluster-to-image minima 7.9983/8.3315/9.0610 Å.
  Heights range 8.3167–10.6687, 8.3123–10.7572, 8.3201–10.6306 Å.
  All three were **unconverged** at the deliberately short cap, with max free
  forces 0.65382/0.60014/0.07673 eV/Å. Exclude these snapshots as reference
  minima; do not infer stability or ranking. A separate BH `run_one` at seed 0,
  budget 40, returned `ok` and exactly 40 evaluations.
- Runs used 6.81 CPU seconds (user+system) on the login node.

Reproduce from the project root (single process, hard CPU cap):

```bash
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
( ulimit -t 55; /usr/bin/time -f 'user=%U sys=%S wall=%e' python benches-eval/su_Pd7_MgO100/validate.py )
```

`validate.py` requires the project and model, but the definition and host are
standalone under `GOPT_BENCH_DIR=$PWD/benches-eval`.
The 20k/30k budgets and 0.05 eV/Å, 0.02 eV tolerances match existing nearby
benchmarks, not a new convergence study. For the later campaign, freeze the
full setup and prespecify all seeds and algorithms; audit
all structures. References exclude guard/wall-active structures; escaped,
disconnected, invalid or unconverged structures require investigation and
disclosure. Because intermediate geometries are not all saved, the scoring
trace cannot reliably exclude every anomalous intermediate hit. Escaped configurations can contaminate the
landscape even if excluded from hits; if substantial, exclude the *benchmark*
pending a separately reviewed constraint. The present checks do not prove
physical reliability, DFT agreement, or a global minimum.
