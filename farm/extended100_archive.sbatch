#!/bin/bash
#SBATCH -J gopt-ext100-archive
#SBATCH -A m4517
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -o /global/cfs/cdirs/m4517/jiaao/test/global-opt-game/farm/ext100-archive-%j.out

set -euo pipefail
ROOT=/global/cfs/cdirs/m4517/jiaao/test/global-opt-game
ARCHIVE="$ROOT/archive"
OUT="$ARCHIVE/gopt_ext100_results.tar.gz"
CHECKSUM="$ARCHIVE/gopt_ext100_results.sha256"
test -f "$ROOT/site_extended100/summary.json"
test ! -e "$OUT"
test ! -e "$CHECKSUM"
tmp=$(mktemp "$ARCHIVE/.gopt-ext100-archive.XXXXXXXX")
trap 'rm -f "$tmp"' EXIT
tar --sort=name --owner=0 --group=0 --numeric-owner \
    -C /pscratch/sd/j/jiaao/gopt -cf - results/ext100 | pigz -6 -p 4 > "$tmp"
mv "$tmp" "$OUT"
(cd "$ARCHIVE" && sha256sum gopt_ext100_results.tar.gz > gopt_ext100_results.sha256)
(cd "$ARCHIVE" && sha256sum -c gopt_ext100_results.sha256)
echo "Archived $OUT"
