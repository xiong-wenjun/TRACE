#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT/scripts/launchers/environment.sh"
OUTPUT="${2:-results/formal/manbench_balanced_return_gemini3flash_four_core_full_20260908_01}"
CONFIG="configs/models/gemini_3_flash/manbench.json"

cd "$ROOT"
source scripts/launchers/environment.sh
if [[ -e "$OUTPUT" ]]; then
  printf 'refusing to overwrite existing output: %s\n' "$OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/shards"

for shard in 0 1 2 3; do
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
        --shard-count 4 \
        --shard-index "$4" \
        --resume
      summary="$3/shards/shard-$(printf "%02d" "$4").json"
      failed="$(jq -r ".failed // 1" "$summary")"
      printf "round=%s shard=%s unresolved=%s\n" "$round" "$4" "$failed"
      if [[ "$failed" -eq 0 ]]; then
        printf "complete shard=%s round=%s\n" "$4" "$round"
        exit 0
      fi
      sleep 5
    done
    printf "exhausted shard=%s rounds=6\n" "$4" >&2
    exit 2
  ' _ "$ROOT" "$CONFIG" "$OUTPUT" "$shard" \
    >"$OUTPUT/logs/shard-${shard_tag}.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$OUTPUT/pids/shard-${shard_tag}.pid"
  printf 'started shard=%s pid=%s\n' "$shard" "$pid"
done
