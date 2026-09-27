#!/usr/bin/env bash
set -euo pipefail

ROOT="$1"
source "$ROOT/scripts/launchers/environment.sh"
CONFIG="$2"
OUTPUT="$3"
SHARD_COUNT="$4"
SHARD_INDEX="$5"
MANIFEST="$6"

cd "$ROOT"
source scripts/launchers/environment.sh
set -a
source scripts/launchers/environment.sh
set +a
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src:.
export TMPDIR=/dev/shm

mapfile -t ids < "$MANIFEST"
episode_args=()
for episode_id in "${ids[@]}"; do
  episode_args+=(--episode-id "$episode_id")
done

for round in 1 2 3 4 5 6; do
  python -u scripts/experiments/run_manbench_return_batch.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --shard-count "$SHARD_COUNT" \
    --shard-index "$SHARD_INDEX" \
    --resume \
    "${episode_args[@]}"
  summary="$OUTPUT/shards/shard-$(printf '%02d' "$SHARD_INDEX").json"
  failed="$(jq -r '.failed // 1' "$summary")"
  printf 'round=%s shard=%s unresolved=%s\n' "$round" "$SHARD_INDEX" "$failed"
  if [[ "$failed" -eq 0 ]]; then
    printf 'complete shard=%s round=%s\n' "$SHARD_INDEX" "$round"
    exit 0
  fi
  sleep 5
done

printf 'exhausted shard=%s rounds=6\n' "$SHARD_INDEX" >&2
exit 2
