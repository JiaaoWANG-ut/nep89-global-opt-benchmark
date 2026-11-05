# Extended 100-seed campaign

This is a separate extension to the published 45-system, 50-seed benchmark;
it does not alter that benchmark or its published ranking.

The four candidate systems cover a gas-phase Ni19 cluster, Au10 on rutile
TiO2(110), Pd6 in MFI, and variable-cell Ga2O3 (Ga4O6). They reuse earlier
candidate definitions and fixed hosts. Earlier BH/RS pilots supplied the
budgets; their lowest matching structures passed the guard, energy,
connectivity and volume checks. Pilot results are not the 100-seed experiment,
and they do not prove a global minimum or eliminate NEP89 artefacts.

The campaign `ext100` uses all 22 frozen algorithms, seeds 0 through 99,
and each system's fixed step budget. GOFEE and SurrBH retain the 3600-second
per-seed wall-time cap. A timeout without a hit is charged the full step
budget as in the original study. The expansion queue is separate from the
original farm queue and is stopped when its tasks finish.

```bash
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1
export GOPT_BENCH_DIR="$PWD/benchmarks_extended100"
export FARM_QUEUE=/pscratch/sd/j/jiaao/gopt/farm-queue-extended100
python farm/submit.py --status
python farm/wait.py 'ext100-*' --timeout 60
```

Results are written to `results/ext100/`. Check all 8,800 records and
their hashes, errors, timeouts, convergence, and reference structures before
interpreting or publishing a ranking. In particular, repeat the guard/wall
and geometry screening on the new best structures. The existing report text
describes the original campaign and must not be reused unchanged for this one.
`scripts/extended100_report.py` audits the full seed inventory, checks hashes,
validates reference structures, screens guard-active hits and produces a
separate report in `site_extended100/`. It fails rather than reporting an
incomplete campaign. `farm/extended100_report.sbatch` runs it on a compute
node; review the output and any guard-active structures before publication.
After a successful report, the report job also runs
`farm/extended100_archive.sbatch` to create and verify a CFS archive of the
new raw records, which otherwise live on purgeable pscratch.
The Slurm allocation is limited to 48 hours; if work remains at its end,
requeue stale tasks and submit `farm/extended100.sbatch` again.
