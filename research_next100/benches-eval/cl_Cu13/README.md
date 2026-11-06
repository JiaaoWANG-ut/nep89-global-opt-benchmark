# Candidate: free Cu13

This description records the initial proposal and short checks, not a result of
the 100-seed campaign. A later 10-run system pilot is recorded separately in
`../pilot_audit.json`. The composition and atom count
do not appear in `benchmarks/`, `benchmarks_extended100/`, or
`benchmarks_extended100b/`; Cu38 already appears in the original set. Cu and
Al are listed in the NEP89 model's 89-element header, but coverage is not a
guarantee of transferability to clusters.

All 13 atoms move in a periodic 24 Å box. NEP CPU is always periodic; the
4.7 Å initial placement sphere gives at least 14.6 Å between images, larger
than the model's 6 Å radial cutoff (angular cutoff 5 Å). `connected` and
the sphere govern **initialization**, not all relaxed or later search
structures. The objective includes the project's default guard
`factor=0.75`, `k=20`; no special pair floors. The proposed 20,000
force-evaluation budget, 0.05 eV/Å threshold and 0.02 eV tolerance match
the analogous original Pt13 setup; they have **not** been calibrated by a
pilot or chosen for a measured success rate.

## Provenance and checks

Local test shapes are ASE `Icosahedron("Cu", 2, latticeconstant=3.615)` and
`Octahedron("Cu", 3, 1, latticeconstant=3.615)` (13-atom icosahedron and
cuboctahedron), rather than literature global-minimum claims. `check.py`
also tests random placements and short random-search runs for **both** seeds
0 and 1, without selecting the favorable seed. Its complete results are in
`check.json`. Reference relaxations converged at -36.94924 and -36.03868 eV;
the random-start relaxations at -35.62149 and -36.21052 eV. Final nearest
distances were 2.393–2.483 Å, all were connected and guard-free, and the
shortest final cluster–image distance was 18.19 Å. The unrelaxed fcc Cu
comparison is -4.28347 eV/atom, below all four tested cluster energies per
atom; it is only a sanity check, not a phase-stability proof.

All four random-search smokes across the two proposed candidates completed
their 80-step budgets; Cu13 used 102 local/re-evaluation steps and 160 smoke
steps with 0 bad evaluations in the former. The recorded CPU time is 2.9 s.
Problem landscape hash `9afb6b94d559`; problem JSON SHA-256
`1f50e06e8695e67cfebfd32f81cdb259a08efea9475e54b074f81ca111006a60`.
Hash identifies the project landscape definition, **not** the model file,
region or budget. Model SHA-256:
`75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1`.

No NEP energy here is DFT, no global minimum or success rate has been
established, and two seeds do not test robustness. The previously observed
SiO2 and SrTiO3 NEP89 holes survived the default guard; inactive guard and
reasonable distance do not rule out other holes. Investigate guard-active,
disconnected, image-interacting or otherwise anomalous structures and
suspicious low energies. The scoring trace does not save every intermediate
geometry and cannot exclude all anomalous intermediate hits.
Pilot references and the system selection remain provisional; the
10-run pilot does not establish a representative global minimum.

## Reproduce from project root

```bash
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m json.tool benches-eval/cl_Cu13/problem.json >/dev/null
timeout 55s python benches-eval/cl_Cu13/check.py
```

The script loads `Problem`, asserts geometry and energy checks, relaxes both
fixed starting shapes and both random seeds using real `FastNEP` via `Context`,
and invokes the actual `rs` runner (`run_one`) twice at 80 ionic steps.
The cap is wall time; the CPU time is reported separately. Rerunning can
change timing and tiny numerical values. `GOPT_NEP_MODEL`, if set, overrides
the default model and changes the reported model hash.
