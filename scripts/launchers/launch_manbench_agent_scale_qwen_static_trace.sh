#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/launchers/environment.sh"
CONFIG="configs/variants/qwen35_122b_a10b/manbench_agent_scale_qwen_static_trace.json"
OUTPUT="${MANBENCH_AGENT_SCALE_OUTPUT:-results/formal/manbench_agent_scale_qwen_static_trace}"
SHARD_COUNT="${MANBENCH_SHARD_COUNT:-8}"

cd "$PROJECT_ROOT"
source scripts/launchers/environment.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="src:."
export TMPDIR=/dev/shm
mkdir -p "$OUTPUT/logs" "$OUTPUT/pids"

for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  label="$(printf '%02d' "$shard")"
  pid_path="$OUTPUT/pids/shard-$label.pid"
  log_path="$OUTPUT/logs/shard-$label.log"
  if [[ -s "$pid_path" ]]; then
    existing_pid="$(<"$pid_path")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      printf 'shard=%s already_running pid=%s\n' "$label" "$existing_pid"
      continue
    fi
  fi
  nohup python3 scripts/experiments/run_manbench_return_batch.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --shard-count "$SHARD_COUNT" \
    --shard-index "$shard" \
    --resume \
    >"$log_path" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$pid_path"
  printf 'shard=%s started pid=%s\n' "$label" "$pid"
done

printf 'matrix=independent_majority_x_4_active_agent_counts output=%s shards=%s arms=static,trace D=1\n' \
  "$OUTPUT" "$SHARD_COUNT"
