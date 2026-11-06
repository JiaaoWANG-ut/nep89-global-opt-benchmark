# Candidate: variable-cell Al8

This description records the initial proposal and short checks, not a result
of the 100-seed campaign. A later 10-run system pilot is recorded separately
in `../pilot_audit.json`. Neither its elemental Al8
composition nor name occurs in the original 45 or either four-system
extension. Those sets contain Al13 cluster and Al-oxide crystals, so this is
not independent in element or a random sample. Al is present in the NEP89
89-element header, which does not guarantee accuracy or eliminate potential
holes.

All eight atoms and the periodic cell are free; random cells initially
sample 0.9–1.3 times 16.61 Å³/atom, with geometry rejection in `Problem`.
The 5.103 Å cubic initial cell is a volume guess, **not** a fixed lattice or
structure. ASE's fcc Al with 4.05 Å conventional lattice constant repeated
2×1×1 provides an eight-atom reference; its conventional-cell volume is
16.6075 Å³/atom. The objective uses the standard short-range guard
`factor=0.75`, `k=20` without pair floors. The proposed 20,000 evaluation
budget and thresholds (0.05 eV/Å, 0.02 eV) match the original Si8
variable-cell benchmark and are provisional, not pilot-calibrated.

## Validation and exclusion

`check.json` contains every short result, not a selected successful seed.
The fcc starting structure relaxed to -32.21902 eV, 15.8295 Å³/atom,
2.8183 Å nearest periodic distance, and zero guard energy. Random seeds 0
and 1 relaxed respectively to -32.21829 and -31.76042 eV, with distances
2.8070 and 2.6452 Å and volumes 15.8689 and 16.3411 Å³/atom; both had zero
final guard energy. Both *initial* random configurations had an active
guard (0.34425 and 0.08796 eV), which is allowed during exploration but
must not be hidden or treated as a physical reference. All three local
relaxations report convergence; force and stress components are recorded in
`check.json`. Actual random search, seeds 0 and 1, separately completed 80
evaluations each; 70 local/re-evaluations had zero bad evaluations. Total
recorded CPU time 2.544 s.

The first trial bulk Ge8 was **excluded**, not proposed: the two random
relaxations had 16.3025 and 15.4538 Å³/atom and energies -36.82557 and
-37.82306 eV, below the diamond reference -36.32662 eV at 21.5829 Å³/atom.
That large dense-phase preference (up to 0.187 eV/atom) is an unresolved
physical/model-transferability concern, not proof of a potential hole. The
second Ge result also had force maximum 0.062 eV/Å despite a reported
variable-cell convergence flag. The complete unfiltered Ge screening record
is `excluded_Ge8_check.json`; it used a 5.658 Å Ge diamond conventional
cell, vpa 22.64, and the same two seeds and 100-step local cap.

Problem landscape hash `3c67c25f0823`; problem JSON SHA-256
`33d8c5b049f67c9bb589639851ed87aa825e0836752f90c1bef753c8d01a9dc2`.
The project hash omits model file, initial cell, sampling and budget; the
tested model SHA-256 is
`75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1`.

Two random seeds are not a pilot and no tested structure is certified as the
global minimum. NEP89 is not DFT. Prior SiO2 and SrTiO3 studies found holes
even with the guard; a guard-free, plausible-volume structure is not proof
that a lower hole does not exist. Investigate guard-active, grossly compressed
or otherwise anomalous structures and suspicious low energies; the scoring
trace cannot exclude all anomalous intermediate hits because their geometries
are not all saved. A physical review, independent references, phase comparisons, and broader
sampling remain important. The screening and exclusion make these candidates
a selected convenience sample.

## Reproduce from project root

```bash
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m json.tool benches-eval/bu_Al8/problem.json >/dev/null
timeout 55s python benches-eval/bu_Al8/check.py
```

The script loads `Problem`, checks real `FastNEP` energies, forces, guard
and periodic geometry, relaxes the fcc starting point and two random seeds
using `Context`, and invokes the actual `rs` runner (`run_one`) twice at 80
steps. Timeout is wall time; CPU time is recorded separately. Reruns can
vary slightly. `GOPT_NEP_MODEL`, if set, overrides the default model.
