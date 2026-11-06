# Third 100-seed extension

This isolated campaign compares 22 published methods and two experimental
algorithm families, each in its original and hosted-move variant. The resulting
26 methods are not 26 independent research innovations. The candidate
methods combine established concepts; no literature novelty or performance
advantage is claimed.

The four new systems are Cu13, elemental Al8 with a variable cell, Pd7 on a
frozen MgO(001) slab, and Au4 initialized in a frozen all-silica CHA framework.
They were manually selected and are not representative random samples. The
surface and zeolite hosts are reused from the development set, so performance
on these systems is not a completely independent test of generalization to
unseen hosts. The source of the original CHA CIF is not documented.

A separate, predeclared pilot used BH and SurrBH with five seeds per system.
All 40 saved best structures passed the project's energy, guard, distance,
connectivity and applicable volume checks and reached the pilot reference
window. This suggests easy systems but does not prove a global minimum or
physical reliability. Several reference atoms and two zeolite cluster
centers lie outside their initial sampling regions. Those regions guide
initialization and are not confinement constraints. Pilot observations and
the fixed campaign design are in `pilot_audit.json` and `protocol.json`.

The evaluation uses 100 seeds per system and method, fixed defaults and a
shared ionic-step budget per system. GOFEE and SurrBH have a 3,600-second
per-seed cap; an unsuccessful timeout is charged its full budget. The
reference energy includes guard-free saved structures from the separate pilot.
Raw pilot and development records must therefore be retained with the
evaluation records. Scores from different system sets are not directly
comparable. NEP89 is not DFT; zero guard energy and geometric checks do not
prove that a structure is physically correct or globally optimal.

`freeze_manifest.json` contains the NEP model, benchmark, algorithm and shared
code hashes, plus SHA-256 digests of the full benchmark specifications,
substrates, protocol and pilot audit. Run `python scripts/research_freeze.py`
with `GOPT_BENCH_DIR` set to this directory before or after a campaign to
verify that the full sealed configuration is unchanged.
