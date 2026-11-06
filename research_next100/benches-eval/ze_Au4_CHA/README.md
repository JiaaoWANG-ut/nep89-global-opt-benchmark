# Proposed Au4 in all-silica CHA

**Status: provisional.** A later 10-run system pilot is recorded in
`../pilot_audit.json`; the checks described below predate that pilot. Four Au atoms are initialized in
a cha cage of a frozen all-silica CHA framework. Small noble-metal clusters in
zeolite cages are a plausible confinement model; this particular Au4/CHA
stoichiometry is an idealized proposal, not a claim of a characterized material
or experimentally established charge state. It is absent from the original
45 and both four-system extensions, and a same-named pilot definition was not
found in the candidate inventory. Selection was manual, not random or
statistically representative.

The 108-atom Si36O72 host is byte-for-byte
`benchmarks/ze_Pt4_CHA/substrate.extxyz`, and its 2.51 Å initialization sphere,
20,000-ionic-step budget and tolerances are inherited from that development
benchmark. This is not an independent host and overlaps an algorithm-development
system. `scripts/build_dev_benchmarks.py` documents the original cell/atomic
NEP relaxation and grid-based cavity search. The builder references a local
CHA CIF but does not preserve a verifiable external CIF source in this
directory; exact crystallographic provenance is therefore **unknown**. Do not
attribute the geometry to a specific experimental structure. The framework is
fully frozen, defect-free and all silica; no Al, counterions, adsorbates,
charge-state dynamics, finite-temperature effects or framework relaxation are
modeled. Au/O/Si appear in the NEP89 header, which does not establish accuracy
for Au–silica interactions. The GPUMD model
`potentials/nep/nep89_20250409/nep89_20250409.txt` SHA-256 is
`75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1`.

All three cell directions are periodic. The initialization sphere is *not* a
confinement constraint during relaxation or subsequent moves. The framework
and short-range guard create only energetic barriers; search atoms can migrate
to other cages, dissociate, or cross framework rings. Investigate escaped,
disconnected or guard-active structures and audit saved putative references
and hits with periodic
geometry and topology, including cage assignments. If escape or an
unphysical NEP attraction affects the landscape materially, exclude the whole
benchmark pending a separately reviewed confinement strategy, not just
inconvenient outcomes. The scoring trace cannot exclude all anomalous
intermediate hits because their geometries are not all saved. Periodic cluster
images and frozen-host effects remain.

## Identity and checks

- `Problem.hash`: `bc00fcb2d4b7`; the hash does not include model bytes,
  initialization region, budget or tolerances. Freeze these separately.
  Loading all definitions in `benchmarks`, `benchmarks_extended100` and
  `benchmarks_extended100b` found no matching hash.
- `problem.json` SHA-256: `97c289de72dae8e06f4924a7e8d45c58a30ddc53d10aaa9708823c918e93b915`.
- host SHA-256: `c96fd568eb7d443699bc2f72d35594af7b2ee4dc50f6d4d81e6b0d35793cd4ca`.
- JSON, `Problem`, host freezing, periodicity and NEP symbols verified. Si–O
  nearest distances 1.6153–1.6154 Å, all Si four-coordinate under 2.1 Å;
  center clearance to nearest host atom 4.707 Å.
- Predeclared seeds 0, 1, 2, all reported: true NEP+guard and up to 50 L-BFGS
  steps. All converged at fmax 0.04231/0.03509/0.03451 eV/Å; guard energies
  zero; one cluster component; minimum Au–host 2.4122/2.3260/2.3557 Å,
  minimum Au–Au 2.6544/2.7174/2.6169 Å, approximate unwrapped cluster-image
  minima 9.3740/10.3048/9.1850 Å. Maximum minimum-image distance from initial
  cage center 3.1425/2.4504/3.3118 Å, exceeding the initialization radius
  for two seeds; this is not itself proof of escape but confirms the sphere
  does not confine relaxation. Center distance alone does not establish cage
  topology. A separate BH `run_one` at seed 0, budget 40, returned `ok` and
  exactly 40 evaluations. These are smoke results, not validated references.
- Runs used 5.82 CPU seconds (user+system) on the login node.

Reproduce from the project root (single process, hard CPU cap):

```bash
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
( ulimit -t 55; /usr/bin/time -f 'user=%U sys=%S wall=%e' python benches-eval/ze_Au4_CHA/validate.py )
```

`validate.py` requires the project and model, but the definition and host are
standalone under `GOPT_BENCH_DIR=$PWD/benches-eval`.
The full model, specification, code and seed plan must be frozen for the
campaign. These brief checks establish neither physical
reliability nor DFT agreement nor a global minimum.
