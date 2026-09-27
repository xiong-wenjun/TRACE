#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/launchers/environment.sh"
STAMP="20260818_01"
ACTOR_BASE="${TRACE_ACTOR_BASE_URL:-https://dashscope-us.aliyuncs.com/compatible-mode/v1}"
ACTOR_MODEL="${TRACE_ACTOR_MODEL:-Qwen3.5-122B-A10B}"
EMBEDDING_BASE="${TRACE_EMBEDDING_BASE_URL:-https://dashscope-us.aliyuncs.com/compatible-mode/v1}"
EMBEDDING_MODEL="${TRACE_EMBEDDING_MODEL:-Qwen3-Embedding-8B}"
JUDGE_BASE="${TRACE_JUDGE_BASE_URL:-https://api.openai.com/v1}"
JUDGE_MODEL="${TRACE_JUDGE_MODEL:-gpt-5.6-terra}"
ARMS="temporal_lww,kmu_op"

cd "$ROOT"
source scripts/launchers/environment.sh
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"

launch_departure() {
  local label="$1"
  local fraction="$2"
  local output="results/formal/memora_session_depart${label}_backend_a_qwen_terra_temporal_lww_kmu_op_${STAMP}"
  local log="$output/pipeline.log"
  local pid_file="$output/pipeline.pid"
  mkdir -p "$output"

  if [[ -s "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(<"$pid_file")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      printf 'departure=%s already_running pid=%s\n' "$label" "$existing_pid"
      return
    fi
  fi

  nohup bash -lc '
    set -euo pipefail
    cd "$1"
    source scripts/launchers/environment.sh
    : "${DASHSCOPE_API_KEY:?missing DASHSCOPE_API_KEY}"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="src:." TMPDIR=/dev/shm

    python3 scripts/experiments/generate_memora_mas_return.py \
      --manifest configs/shared/memora_return_confirmation_manifest.json \
      --output-dir "$2" \
      --base-url "$3" \
      --model "$4" \
      --api-key-env TRACE_ACTOR_API_KEY \
      --embedding-base-url "$5" \
      --embedding-model "$6" \
      --embedding-dimensions 1024 \
      --arms "$7" \
      --state-mode predicted \
      --seed 731 \
      --actor-max-tokens 2048 \
      --cupmem-max-tokens 4096 \
      --max-actor-view-chars 16000 \
      --timeout 600 \
      --retries 5 \
      --resume \
      --predicted-departure-fraction "$8" \
      --predicted-predeparture-top-k 10 \
      --predicted-absence-top-k 14 \
      --predicted-predeparture-recency-k 2 \
      --predicted-absence-recency-k 4 \
      --predicted-max-sessions-per-batch 6 \
      --predicted-minimum-confidence 0.45 \
      --predicted-base-selected-facts 12 \
      --predicted-max-selected-facts 16 \
      --predicted-max-tokens 4096 \
      --predicted-structured-output-mode prompt_json

    python3 scripts/experiments/score_memora_generations.py \
      --manifest configs/shared/memora_return_confirmation_manifest.json \
      --input-dir "$2" \
      --output-dir "$2/terra_scored" \
      --judge-base-url "$9" \
      --judge-model "${10}" \
      --judge-api-key-env OPENAI_API_KEY \
      --arms "$7" \
      --seed 731 \
      --judge-max-tokens 256 \
      --timeout 600 \
      --retries 5 \
      --embedding-base-url "$5" \
      --embedding-model "$6" \
      --embedding-dimensions 1024 \
      --resume
  ' _ "$ROOT" "$output" "$ACTOR_BASE" "$ACTOR_MODEL" \
    "$EMBEDDING_BASE" "$EMBEDDING_MODEL" "$ARMS" "$fraction" \
    "$JUDGE_BASE" "$JUDGE_MODEL" >>"$log" 2>&1 < /dev/null &
  local experiment_pid=$!
  printf '%s\n' "$experiment_pid" >"$pid_file"
  printf 'departure=%s started pid=%s output=%s\n' \
    "$label" "$experiment_pid" "$output"
}

# The two arms remain paired inside each departure job so they share exactly
# the same predicted extraction and frozen post-absence checkpoint.
launch_departure 25 0.25
launch_departure 50 0.50
launch_departure 75 0.75

unset OPENAI_API_KEY
