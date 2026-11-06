#!/bin/bash
#SBATCH -J gopt-next-report
#SBATCH -A m4517
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 02:00:00

set -euo pipefail
: "${GOPT_RESEARCH_ROOT:?Set GOPT_RESEARCH_ROOT}"
: "${GOPT_BENCH_DIR:?Set GOPT_BENCH_DIR}"
: "${FARM_QUEUE:?Set FARM_QUEUE}"
: "${FARM_PREFIX:?Set FARM_PREFIX}"
: "${REPORT_DIR:?Set REPORT_DIR}"
: "${ARCHIVE_DIR:?Set ARCHIVE_DIR}"
: "${RESULTS_PARENT:?Set RESULTS_PARENT}"
: "${RELEASE_URL:?Set RELEASE_URL}"
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$GOPT_RESEARCH_ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$GOPT_RESEARCH_ROOT"
python farm/wait.py "$FARM_PREFIX-*" --timeout 1
python scripts/research_freeze.py
python scripts/extended100_report.py --campaign next100 --out "$REPORT_DIR" \
    --release-url "$RELEASE_URL"
test -f "$REPORT_DIR/summary.json"
test ! -e "$ARCHIVE_DIR/gopt_next100_results.tar.gz"
test ! -e "$ARCHIVE_DIR/gopt_next100_results.sha256"
tmp=$(mktemp "$ARCHIVE_DIR/.gopt-next100-archive.XXXXXXXX")
trap 'rm -f "$tmp"' EXIT
tar --sort=name --owner=0 --group=0 --numeric-owner -C "$RESULTS_PARENT" -cf - \
    results/next100 results/nextsystemspilot results/nextdev results/nextvalidation \
    results/nextablation | pigz -6 -p 4 > "$tmp"
mv "$tmp" "$ARCHIVE_DIR/gopt_next100_results.tar.gz"
(cd "$ARCHIVE_DIR" && sha256sum gopt_next100_results.tar.gz > gopt_next100_results.sha256)
(cd "$ARCHIVE_DIR" && sha256sum -c gopt_next100_results.sha256)
python scripts/audit_final_refs.py --report "$REPORT_DIR" --out "$REPORT_DIR/reference_forces.json"
