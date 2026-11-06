# Second 100-seed extension

This separate extension adds Al13, Pt8 on CeO2(111), Au8 in the BEA zeolite,
and variable-cell Al2O3 (Al4O6). These definitions reuse candidate structures
from the earlier pilots. The Pt8 and Al4O6 systems had been excluded from the
original main study as relatively easy; they are not a difficulty-selected
or representative sample of all global-optimization problems.

The campaign `ext100b` runs all 22 frozen algorithms on each system with 100
independent seeds. Each system has one fixed ionic-step budget shared by the
algorithms. GOFEE and SurrBH retain the 3600-second per-seed wall-time limit;
an unsuccessful timeout is charged its full budget. The queue is isolated
from both earlier campaigns. NEP89 is not DFT and guard-free energies do not
prove that a structure is physically correct or globally optimal.

```bash
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1
export GOPT_BENCH_DIR="$PWD/benchmarks_extended100b"
export FARM_QUEUE=/pscratch/sd/j/jiaao/gopt/farm-queue-extended100b
python farm/submit.py --status
python farm/wait.py 'ext100b-*' --timeout 60
```

Results are in `results/ext100b/`. After completion,
`scripts/extended100_report.py --campaign ext100b --out site_extended100b
--release-url https://github.com/JiaaoWANG-ut/nep89-global-opt-benchmark/releases/tag/v1.2`
audits the complete 8,800-run inventory, checks frozen hashes and parameters,
validates reference structures and screens guard-active candidate hits. The
separate report must be reviewed before publication. A dependent report job
also creates a checksummed archive on CFS, because pscratch may be purged.
After the generated report has passed its audit, run
`python scripts/annotate_extended100b.py` to add the second batch's
system-selection and model caveats to both HTML pages. This presentation-only
step does not alter the frozen report generator, metrics, or raw records.
