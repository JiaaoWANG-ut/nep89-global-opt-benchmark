#!/bin/bash
#SBATCH -J gopt-farm
#SBATCH -A m4517
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 5
#SBATCH -t 48:00:00
#SBATCH --signal=USR1@600
#SBATCH -o /global/cfs/cdirs/m4517/jiaao/test/global-opt-game/farm/slurm-%j.out
#
# Long-lived task farm: one node_manager.py per node, 128 single-core slots per node.
# Add work at any time with farm/submit.py. If farm/queue/control/KEEPALIVE holds a
# positive integer N, the job queues its own successor (afterany) 15 minutes after
# start and writes N-1 back, so the farm renews itself N more times.
# To stop for good: rm farm/queue/control/KEEPALIVE; scancel -n gopt-farm

ROOT=/global/cfs/cdirs/m4517/jiaao/test/global-opt-game
FARM=$ROOT/farm
module load python/3.12-26.1.0 >/dev/null 2>&1
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export FARM_QUEUE=$FARM/queue FARM_CORES=128 FARM_POLL=3
rm -f $FARM_QUEUE/control/DRAIN $FARM_QUEUE/control/STOP

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST start $(date)"
python $FARM/requeue_stale.py 300

(
  sleep 900
  K=$FARM_QUEUE/control/KEEPALIVE
  if [ -f $K ]; then
    n=$(tr -dc '0-9' < $K); n=${n:-0}
    if [ "$n" -gt 0 ]; then
      echo $((n - 1)) > $K
      # premium allows only 5 submitted jobs per user; fall back to the regular QOS
      sbatch --dependency=afterany:$SLURM_JOB_ID $FARM/farm.sbatch ||
        sbatch -q regular --dependency=afterany:$SLURM_JOB_ID $FARM/farm.sbatch
    fi
  fi
) &

srun -N $SLURM_NNODES --ntasks-per-node=1 -c 256 --cpu-bind=none python -u $FARM/node_manager.py
echo "job $SLURM_JOB_ID end $(date)"
