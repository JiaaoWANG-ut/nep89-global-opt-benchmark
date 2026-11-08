#!/bin/bash
#SBATCH -J future100-report
#SBATCH -A m4517
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 03:00:00

set -euo pipefail
: "${GOPT_RESEARCH_ROOT:?Set GOPT_RESEARCH_ROOT}"
: "${FARM_QUEUE:?Set FARM_QUEUE}"
source /global/common/software/m4517/jiaao/envs/gopt/bin/activate
export PYTHONPATH="$GOPT_RESEARCH_ROOT" GOPT_BENCH_DIR="$GOPT_RESEARCH_ROOT/benches-eval"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$GOPT_RESEARCH_ROOT"

python farm/wait.py 'future100-*' --timeout 1
python scripts/research_freeze.py

archive_dir=/global/cfs/cdirs/m4517/jiaao/test/global-opt-game/archive
archive="$archive_dir/gopt_future100_results.tar.gz"
checksum="$archive_dir/gopt_future100_results.sha256"
if [[ -e "$archive" || -e "$checksum" ]]; then
    test -f "$archive" && test -f "$checksum"
    (cd "$archive_dir" && sha256sum -c gopt_future100_results.sha256)
else
    tmp=$(mktemp "$archive_dir/.gopt-future100-archive.XXXXXXXX")
    trap 'rm -f "$tmp"' EXIT
    tar --sort=name --owner=0 --group=0 --numeric-owner \
        -C "$GOPT_RESEARCH_ROOT" -cf - results/future100 results/future-systempilot \
        | pigz -6 -p 4 > "$tmp"
    mv "$tmp" "$archive"
    (cd "$archive_dir" && sha256sum gopt_future100_results.tar.gz > gopt_future100_results.sha256)
    trap - EXIT
fi
tar -tzf "$archive" | awk '
    /\/s[0-9]+\.json$/ {
        n++
        if ($0 ~ /^results\/future100\//) formal++
        if ($0 ~ /^results\/future-systempilot\//) pilot++
    }
    END {
        print "Archived formal:", formal, "pilot:", pilot, "total:", n
        if (formal != 15500 || pilot != 50 || n != 15550) exit 1
    }'

python scripts/extended100_report.py --campaign future100 --seed-first 100 \
    --reference-campaign future-systempilot --out site_future100 \
    --release-url https://github.com/JiaaoWANG-ut/nep89-global-opt-benchmark/releases/tag/v1.4
python dev_history/verify_future_report.py --report site_future100 \
    --out site_future100/independent_verification.json
python scripts/audit_final_refs.py --report site_future100 --out site_future100/reference_forces.json
python dev_history/audit_candidate_structures.py --report site_future100 --campaign future100 \
    --reference-campaign future-systempilot --jobs 32 --out site_future100/candidate_geometry_audit.json
python dev_history/compare_ablations.py
python dev_history/publish_future.py
echo "Automated report and physical audits completed; publication still requires review."
