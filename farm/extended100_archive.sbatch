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
RUN_CAMPAIGN="${RUN_CAMPAIGN:-ext100}"
REPORT_DIR="${REPORT_DIR:-$ROOT/site_extended100}"
ARCHIVE_BASENAME="${ARCHIVE_BASENAME:-gopt_ext100_results}"
[[ "$RUN_CAMPAIGN" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$ARCHIVE_BASENAME" =~ ^[a-zA-Z0-9_-]+$ ]]
OUT="$ARCHIVE/$ARCHIVE_BASENAME.tar.gz"
CHECKSUM="$ARCHIVE/$ARCHIVE_BASENAME.sha256"
test -f "$REPORT_DIR/summary.json"
test ! -e "$OUT"
test ! -e "$CHECKSUM"
tmp=$(mktemp "$ARCHIVE/.gopt-ext100-archive.XXXXXXXX")
trap 'rm -f "$tmp"' EXIT
tar --sort=name --owner=0 --group=0 --numeric-owner \
    -C /pscratch/sd/j/jiaao/gopt -cf - "results/$RUN_CAMPAIGN" | pigz -6 -p 4 > "$tmp"
mv "$tmp" "$OUT"
(cd "$ARCHIVE" && sha256sum "$ARCHIVE_BASENAME.tar.gz" > "$ARCHIVE_BASENAME.sha256")
(cd "$ARCHIVE" && sha256sum -c "$ARCHIVE_BASENAME.sha256")
echo "Archived $OUT"
