#!/bin/bash
#SBATCH -J gopt-next-geometry
#SBATCH -A m4517
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:45:00

set -euo pipefail
: "${GOPT_RESEARCH_ROOT:?Set GOPT_RESEARCH_ROOT}"
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$GOPT_RESEARCH_ROOT" GOPT_BENCH_DIR="$GOPT_RESEARCH_ROOT/benches-eval"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$GOPT_RESEARCH_ROOT"
python -u dev_history/audit_candidate_structures.py --jobs 32 \
    --out site_next100/candidate_geometry_audit.json
