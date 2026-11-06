# Archive path relinking

`gopt/algorithms/relink_v2.py` is an experimental candidate, not an established
or validated new research method. Its original version is
`gopt/algorithms/path_relink.py`. It combines familiar ideas; no literature survey or
novelty claim has been made.

An energy-sorted archive keeps distinct relaxed results, always preserving the
best; if full, a combined energy-rank and nearest-fingerprint-distance score
prunes it. Rank-biased selection chooses an anchor, then a different archive
member weighted by both quality and distance. Same-species Hungarian matching
aligns the donor with the anchor. Free clusters use centering and multiple
rotation/assignment fits; fixed-host systems use minimum-image distances
without translating the host; host-free periodic bulk also tries origin shifts.
Several independently relaxed, evenly spaced interpolation images form a path
between parents. Bulk cells are interpolated when well-conditioned, otherwise
the anchor cell is retained. Overlaps are repaired without permanently moving
frozen atoms. A periodic random immigrant and a singleton-archive shake
provide exploration. Only `Context.relax` evaluates energies; budget exhaustion
propagates to the runner. Hosted systems additionally attempt rigid
support-relative translations on a fixed schedule. For surfaces the direction
lies in the plane of the first two cell vectors; for other hosts it is 3D.
The displacement length is `host_step` times the mean search-atom covalent
diameter. Up to six random directions and lengths 1, 1/2, and 1/4 of that
length are tried, screening minimum-image endpoint distances against all
nonmoving atoms using the shared minimum-distance rule. If none pass, the
existing small shake is used. The move changes only mobile search atoms, never
the cell or host. The site in the host is therefore explored independently of
archive-parent interpolation; it is not an explicit adsorption-site finder.

Parameters are fixed global defaults:

```
archive_size=8, initial=3, path_points=3, immigrant_every=6,
diversity_weight=0.5, shake=0.35, rotations=3, host_every=2, host_step=1.0
```

`host_every` and `immigrant_every` count search cycles (a path cycle can contain
multiple relaxations); overdue forced operations alternate rather than
starving each other. Invalid
parameters raise `ValueError` before any evaluation: archive size is an
integer >= 2; initial, path points and immigrant interval are integers >= 1;
rotations is an integer >= 0; diversity weight is finite and in [0, 1]; shake
and host step are finite and positive; host interval is an integer >= 1.
Booleans and non-integer numeric values are not accepted
for integer parameters.

## Pilot observation and ablation

The supplied three-seed, 5,000-step pilot reported 0/3 hits on
`su_Pt8_MgO100`, versus 2/3 for BH and 3/3 for memetic, relative to the old
published reference; the candidate hit 3/3 on the other five development
systems. Read-only inspection of those slab result files shows 108, 108 and 153
path-image attempts, only 10, 10 and 13 random attempts, and **zero singleton
shakes**. BH used 167–174 moves, while memetic used BH chains plus
support-aware moves. The old candidate therefore mostly interpolated between
existing archived minima and had no deliberate support-relative move once its
archive had multiple members. Its fixed-cell fingerprint also retains only
sorted search distances and nearest-host distances, which can alias different
support registries. These observations motivate the translation operator; they
do not establish the failure's cause.

`gopt/algorithms/path_relink.py` is the exact pre-change candidate, byte-identical to
the pilot's integrated module at inspection (SHA-256
`f5e2213c4e651fdf1d8da06bae2b09c86183b2b2e0361b043b0ca01e9fdcec33`).
On the subsequent 10-seed surface development ablation, the original and
revised versions hit 3/10 and 5/10 times, respectively; this small comparison
does not establish a gain.

Unlike the existing cut-and-splice GA/USPEX and memetic crossover, this uses
matched geometric interpolation and a multi-image path. Unlike the existing
DE implementation, it does not construct three-parent vector differences or
greedy per-target generations. These distinctions do not establish novelty or
superior performance. Non-converged but non-aborted relaxations are eligible
for the archive, as in other project algorithms.

## Reproduce bounded checks

From the project root:

```sh
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH=$PWD GOPT_BENCH_DIR=$PWD/benches-dev OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s dev_history/next_algorithms/recombine -p 'test_*.py' -v
python dev_history/next_algorithms/recombine/smoke_nep.py
```

The NEP smoke uses seeds fixed at 42, capped local relaxations, and 90/45/45/45
ionic steps on `cl_Pt13`, `bu_Si8`, `su_Pt8_MgO100`, and `ze_Pt4_CHA`,
respectively. These
are integration checks, not performance or convergence measurements. Mock
tests cover all four problem categories, frozen positions, budget termination,
determinism, matching, archive updates, variable-cell fallback, invalid
parameters and degenerate fingerprints. Statistics accumulate over repeated
calls; random, shake, host, and image counters count attempts, including the
attempt on which budget exhaustion occurs, while admissions count completed
relaxations. `host_fallback` counts host attempts that used a shake.

## Integration and limitations

Both variants are available in this isolated research copy; they do not alter
the published 22-method registry.

The fixed-cell fingerprint is species-insensitive, so distinct alloy orderings
may sometimes be merged. Fingerprints and minimum-image matching are imperfect
proxies for structural equivalence; the bulk interpolation uses the parents'
existing lattice bases rather than a canonical crystallographic alignment.
Consequently equivalent differently based cells and difficult rotations may
produce unproductive paths. Hosted translations screen their endpoint only;
they do not guarantee connectivity, confinement, or a valid continuous path,
and their cost grows with the number of host atoms. Overlap repair likewise
does not guarantee connected clusters or confinement. There is no
benchmark-specific tuning or verified advantage. `Context`'s existing
variable-cell relaxation may temporarily strain frozen atoms before
`Context.evaluate` restores their reference positions; the candidate restores
frozen host coordinates when archiving results, but cannot change that shared
relaxation implementation.
