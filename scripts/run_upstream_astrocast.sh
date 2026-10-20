#!/usr/bin/env bash
# AstroCaST v0.4.2 official event detector in a SEPARATE Python 3.9/3.10 env.
# Usage: bash scripts/run_upstream_astrocast.sh input_strided.h5 output.roi
set -euo pipefail
if (( $# != 2 )); then
  echo 'Usage: bash scripts/run_upstream_astrocast.sh INPUT_H5 OUTPUT.roi' >&2
  exit 2
fi
INPUT="$(realpath "$1")"
OUTPUT="$2"
case "$OUTPUT" in *.roi) ;; *) echo 'Output must have suffix .roi' >&2; exit 2;; esac
if [[ ! -f "$INPUT" ]]; then echo "No input file: $INPUT" >&2;exit 2;fi
if [[ -e "$OUTPUT" ]];then echo "Output exists; refusing overwrite: $OUTPUT" >&2; exit 2;fi
if ! command -v astrocast >/dev/null 2>&1; then
  echo 'astrocast CLI not found. Activate dedicated Python 3.9/3.10 astrocast environment first.' >&2
  exit 2
fi
OUTPUT="$(realpath -m "$OUTPUT")"
mkdir -p "$(dirname "$OUTPUT")"
LOG="${OUTPUT%.roi}.run.log"
META="${OUTPUT%.roi}.run_metadata.txt"
{
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "input=$INPUT"
  echo "dataset=dff/ch0"
  echo "output=$OUTPUT"
  echo 'upstream_algorithm=astrocast 0.4.2 detect-events (defaults except flags below)'
  echo "command=astrocast detect-events --loc dff/ch0 --output-path OUTPUT --parallel False --on-disk True INPUT"
  sha256sum "$INPUT"
  python -c 'from importlib.metadata import version;print("astrocast_version="+version("astrocast"))'
  python --version
} > "$META"
astrocast detect-events \
  --loc dff/ch0 \
  --output-path "$OUTPUT" \
  --parallel False --on-disk True \
  "$INPUT" 2>&1 | tee "$LOG"
if [[ ! -f "$OUTPUT/event_map.tiff" ]]; then
  echo "No event_map.tiff found in $OUTPUT. Check $LOG; detection output format/settings." >&2
  exit 3
fi
printf 'output_roi=%s\n' "$OUTPUT" | tee -a "$META"
sha256sum "$OUTPUT/event_map.tiff" >> "$META"
