# Adaptive diversity archive search

`gopt/algorithms/diversity_v2.py` is the hosted-move candidate;
`gopt/algorithms/diversity_search.py` is its original version. They are
registered only in this isolated research copy. The method combines ideas
from quality-diversity and novelty search;
it does not assert literature novelty or reproduce MAP-Elites. Unlike a
single accepted basin-hopping walk or the existing tabu walk's energy-binned
visit memory, it maintains a bounded **population** of relaxed structures.

The descriptor combines equally weighted blocks of sorted species-pair
minimum-image distances, sorted nearest-host-species distances, and, for
variable cells, volume and reduced lattice lengths. A candidate within an
adaptive neighbourhood competes with its closest archive member on energy.
Outside it, candidates fill empty capacity or displace one member of the
most crowded pair if the candidate is more isolated. The best archive energy
is protected, and a new global archive best is always admitted. Parents are
sampled using energy rank, local isolation and inverse use count. The move
radius increases when few trials enter the archive and decreases otherwise;
random restarts, atom displacement, unlike swaps, relocation, cell strains,
and hosted rigid translations provide complementary moves. The only energy evaluations are through
`ctx.relax`; its `BudgetExhausted` is deliberately not caught.

## Pilot investigation and hosted move

The supplied fixed three-seed, 5,000-step pilot reported 0/3 hits on the
hosted development system, versus 2/3 for BH and 3/3 for memetic; all five
other development systems were 3/3 for this candidate. The three stored
candidate results were successful runs, with 166–233 completed relaxations,
no bad evaluations or frozen-position corrections, and final adaptive step
sizes 0.850, 0.307 and 0.187 Å. The original source has no rigid cluster
translation. Its independent random atomic displacements have RMS
centre-of-mass motion `step * sqrt(3/(5*n_search))` before any overlap repair;
with eight atoms this is roughly 0.27 times `step`. The single-species
"swap" branch exchanges nothing and uses only one quarter of `step`.
Relocating one atom can alter shape but is not a coherent site hop.
Additionally, the archive descriptor stores only nearest host distances per
host species, not full registry: the unit tests construct two inequivalent
sites with identical descriptors. Thus reduced coherent site exploration
and descriptor aliasing are plausible candidate-level explanations, **not
proven causes**: these aggregate pilot logs do not record site visits or move
outcomes, and three seeds cannot establish a performance difference.

The new generic move selects a rigid translation with probability
`p_host_shift=0.2` for non-restart proposals on fixed-cell surface or zeolite
problems having a host. Direction is uniformly sampled in the plane of the
first two cell vectors for surfaces, or in 3D for pores. Its length is
uniformly 0.5–2.0 times the mean search-atom covalent radius, independently of
the adaptive atom-displacement step. Up to 16 proposals are screened with
periodic minimum-image distances against host overlaps; if the starting
cluster is within the generic host contact threshold, its translation must
remain in contact. Pore centres must remain in the declared generation region.
If none is valid, the existing move is used. Only search atoms are translated;
the host and cell are unchanged. Screening is geometric and performs no energy
calls. `host_shift_attempts` and `host_shifts` expose attempt and feasibility
counts, not success rates or evidence of improved search.

Fixed defaults are `capacity=16`, `step=0.6`, `strain=0.05`,
`p_random=0.12`, `p_swap=0.15`, `p_relocate=0.15`, `adapt_every=8`, and
`p_host_shift=0.2`. These apply uniformly, not per benchmark.
`gopt/algorithms/diversity_search.py` is the original pilot candidate;
for a controlled ablation, setting `p_host_shift=0` also reproduces its random
trajectory (tested across the four categories). On the subsequent 10-seed
surface development ablation, the original and revised versions hit 3/10 and
4/10 times, respectively; this small comparison does not establish a gain.

These are heuristic design choices, not measured performance gains. The
distance descriptor is not a unique structural fingerprint: homometric
structures and distinct host sites can collide, and reduced-cell lengths do
not completely describe a periodic lattice. Adaptive spacing can coalesce
similar but distinct structures. Unconverged non-aborted relaxations are
eligible, as in the project's baseline algorithms. A short relaxation cap is
used only in the smoke tests, not in the algorithm. The shared `Context`
variable-cell relaxation represents atomic positions in a deforming frame;
for a hypothetical variable-cell system with frozen host atoms it restores
their Cartesian coordinates during evaluation but its cell force does not
correct for that constraint. This candidate restores frozen coordinates before
every global move and on archive insertion, but inherits that shared local
relaxation limitation. All existing variable-cell benchmark systems have no
frozen host. No benchmark-specific parameters or reference energies are used.

## Validation

From the project root:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$PWD" GOPT_BENCH_DIR="$PWD/benches-dev" \
  python -m unittest discover -s dev_history/next_algorithms/diversity -p 'test_*.py' -v
```

Sixteen deterministic tests cover archive quality/novelty/elite policy and
selection, descriptor invariances, aliasing and environment/cell sensitivity,
rigid translations including a tilted slab, host overlap and contact
rejection, pore-region rejection, fallback, exact original ablation,
all four problem classes, fixed coordinates, invalid results, exact budget
termination with both mock and actual `Context`, and variable-cell moves.
Before the hosted-move revision, an isolated true-NEP smoke used `Context(budget=12)`, seed 42, and
`p.max_relax_steps=1` on `cl_Pt13`, `su_Pt8_MgO100`, `ze_Pt4_CHA`, and
`bu_Si8`: each consumed exactly 12 ionic steps, completed six relaxations,
had finite best energy and zero frozen-position corrections. This is only
an API/geometry smoke of the old candidate, not an efficiency or convergence
assessment of either version.

Both variants are available in this isolated research copy; the published
22-method registry is unchanged.
