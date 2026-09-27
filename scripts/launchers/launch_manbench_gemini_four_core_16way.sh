#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT/scripts/launchers/environment.sh"
OUTPUT="${2:-results/formal/manbench_balanced_return_gemini3flash_four_core_16way_full_20260908_01}"
CONFIG="configs/variants/gemini_3_flash/manbench_balanced_return_gemini3flash_four_core_16way_formal_full.json"
SHARD_COUNT=16

cd "$ROOT"
source scripts/launchers/environment.sh
if [[ -e "$OUTPUT" ]]; then
  printf 'refusing to overwrite existing output: %s\n' "$OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/shards"

for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_tag="$(printf '%02d' "$shard")"
  nohup bash -lc '
    set -euo pipefail
    cd "$1"
    set -a
    source scripts/launchers/environment.sh
    set +a
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src:.
    export TMPDIR=/dev/shm
    for round in 1 2 3 4 5 6; do
      python -u scripts/experiments/run_manbench_return_batch.py \
        --config "$2" \
        --output-dir "$3" \
        --shard-count "$4" \
        --shard-index "$5" \
        --resume
      summary="$3/shards/shard-$(printf "%02d" "$5").json"
      failed="$(jq -r ".failed // 1" "$summary")"
      printf "round=%s shard=%s unresolved=%s\n" "$round" "$5" "$failed"
      if [[ "$failed" -eq 0 ]]; then
        printf "complete shard=%s round=%s\n" "$5" "$round"
        exit 0
      fi
      sleep 5
    done
    printf "exhausted shard=%s rounds=6\n" "$5" >&2
    exit 2
  ' _ "$ROOT" "$CONFIG" "$OUTPUT" "$SHARD_COUNT" "$shard" \
    >"$OUTPUT/logs/shard-${shard_tag}.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$OUTPUT/pids/shard-${shard_tag}.pid"
  printf 'started shard=%s pid=%s\n' "$shard" "$pid"
done
