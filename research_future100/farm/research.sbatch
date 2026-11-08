#!/bin/bash
#SBATCH -J gopt-next
#SBATCH -A m4517
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 2
#SBATCH -t 04:00:00
#SBATCH --signal=USR1@600

set -euo pipefail
: "${GOPT_RESEARCH_ROOT:?Set GOPT_RESEARCH_ROOT}"
: "${GOPT_BENCH_DIR:?Set GOPT_BENCH_DIR}"
: "${FARM_QUEUE:?Set FARM_QUEUE}"
: "${FARM_PREFIX:?Set FARM_PREFIX}"
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$GOPT_RESEARCH_ROOT" FARM_CORES=128 FARM_POLL=3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$GOPT_RESEARCH_ROOT"

test ! -e "$FARM_QUEUE/control/DRAIN"
test ! -e "$FARM_QUEUE/control/STOP"
python -c 'import os; from farm.wait import scan; s=scan(os.environ["FARM_PREFIX"] + "-*"); assert s["pending"], "No pending research tasks"'
srun -N "$SLURM_NNODES" --ntasks-per-node=1 -c 256 --cpu-bind=none \
    python -u farm/node_manager.py &
manager=$!
while kill -0 "$manager" 2>/dev/null; do
    if python -c 'import os; from farm.wait import scan; s=scan(os.environ["FARM_PREFIX"] + "-*"); raise SystemExit(bool(s["pending"] or s["running"]))'; then
        touch "$FARM_QUEUE/control/DRAIN"
        break
    fi
    sleep 30
done
wait "$manager"
python farm/wait.py "$FARM_PREFIX-*" --timeout 1
