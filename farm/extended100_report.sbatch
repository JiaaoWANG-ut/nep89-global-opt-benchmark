#!/bin/bash
#SBATCH -J gopt-ext100-report
#SBATCH -A m4517
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o /global/cfs/cdirs/m4517/jiaao/test/global-opt-game/farm/ext100-report-%j.out

set -euo pipefail
ROOT=/global/cfs/cdirs/m4517/jiaao/test/global-opt-game
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$ROOT" GOPT_BENCH_DIR="${GOPT_BENCH_DIR:-$ROOT/benchmarks_extended100}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RUN_CAMPAIGN="${RUN_CAMPAIGN:-ext100}"
export REPORT_DIR="${REPORT_DIR:-$ROOT/site_extended100}"
export ARCHIVE_BASENAME="${ARCHIVE_BASENAME:-gopt_ext100_results}"
RELEASE_URL="${RELEASE_URL:-https://github.com/JiaaoWANG-ut/nep89-global-opt-benchmark/releases/tag/v1.1}"
cd "$ROOT"
python -u scripts/extended100_report.py --campaign "$RUN_CAMPAIGN" --out "$REPORT_DIR" \
    --release-url "$RELEASE_URL"
bash farm/extended100_archive.sbatch
