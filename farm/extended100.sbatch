#!/bin/bash
#SBATCH -J gopt-ext100
#SBATCH -A m4517
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 5
#SBATCH -t 48:00:00
#SBATCH --signal=USR1@600
#SBATCH -o /global/cfs/cdirs/m4517/jiaao/test/global-opt-game/farm/ext100-%j.out

set -euo pipefail
ROOT=/global/cfs/cdirs/m4517/jiaao/test/global-opt-game
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$ROOT" GOPT_BENCH_DIR="${GOPT_BENCH_DIR:-$ROOT/benchmarks_extended100}"
export FARM_QUEUE="${FARM_QUEUE:-/pscratch/sd/j/jiaao/gopt/farm-queue-extended100}"
export FARM_PREFIX="${FARM_PREFIX:-ext100}"
export FARM_CORES=128 FARM_POLL=3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"

python -c 'import os; from farm.wait import scan; s=scan(os.environ["FARM_PREFIX"] + "-*"); assert any(s.values()), "No extension tasks found"'
rm -f "$FARM_QUEUE/control/DRAIN"
python farm/requeue_stale.py 300
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST start $(date)"
srun -N "$SLURM_NNODES" --ntasks-per-node=1 -c 256 --cpu-bind=none \
    python -u farm/node_manager.py &
manager=$!

while kill -0 "$manager" 2>/dev/null; do
    if python -c 'import os; from farm.wait import scan; s=scan(os.environ["FARM_PREFIX"] + "-*"); raise SystemExit(bool(s["pending"] or s["running"]))'; then
        touch "$FARM_QUEUE/control/DRAIN"
        break
    fi
    sleep 60
done
wait "$manager"
python farm/wait.py "$FARM_PREFIX-*" --timeout 1
echo "job $SLURM_JOB_ID finished $(date)"
