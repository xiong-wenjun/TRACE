#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/launchers/environment.sh"
ACTOR_BASE="${TRACE_ACTOR_BASE_URL:-https://api.deepseek.com}"
ACTOR_MODEL="${TRACE_ACTOR_MODEL:-deepseek-v4-flash-0731}"
JUDGE_BASE="${TRACE_JUDGE_BASE_URL:-https://api.openai.com/v1}"
JUDGE_MODEL="${TRACE_JUDGE_MODEL:-gpt-5.6-terra}"
ARMS="static_no_churn,restore_old,reset,trace"
MAX_RETRY_ROUNDS="${MEMORA_MAX_RETRY_ROUNDS:-3}"

cd "$ROOT"
source scripts/launchers/environment.sh
: "${DEEPSEEK_API_KEY:?missing DEEPSEEK_API_KEY}"
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="src:."
export TMPDIR=/dev/shm

failure_count() {
  local output="$1"
  find "$output/generations" -maxdepth 1 -type f -name '*.failure.json' 2>/dev/null | wc -l
}

retry_departure() {
  local label="$1"
  local fraction="$2"
  local output="results/formal/memora_session_depart${label}_deepseekv4flash0731_terra_four_arm_20260817_01"
  local main_pid
  main_pid="$(<"$output/run.pid")"

  while kill -0 "$main_pid" 2>/dev/null; do
    sleep 30
  done

  local round remaining
  remaining="$(failure_count "$output")"
  printf 'departure=%s main_complete initial_failures=%s\n' "$label" "$remaining"
  for ((round = 1; round <= MAX_RETRY_ROUNDS && remaining > 0; round++)); do
    printf 'departure=%s retry_round=%s failures_before=%s\n' "$label" "$round" "$remaining"
    python3 scripts/experiments/generate_memora_mas_return.py \
      --manifest configs/shared/memora_return_confirmation_manifest.json \
      --output-dir "$output" \
      --base-url "$ACTOR_BASE" \
      --model "$ACTOR_MODEL" \
      --api-key-env DEEPSEEK_API_KEY \
      --embedding-base-url https://dashscope-us.aliyuncs.com/compatible-mode/v1 \
      --embedding-model Qwen3-Embedding-8B \
      --embedding-dimensions 1024 \
      --arms "$ARMS" \
      --state-mode predicted \
      --seed 731 \
      --actor-max-tokens 2048 \
      --max-actor-view-chars 16000 \
      --timeout 180 \
      --retries 5 \
      --resume \
      --predicted-departure-fraction "$fraction" \
      --predicted-predeparture-top-k 10 \
      --predicted-absence-top-k 14 \
      --predicted-predeparture-recency-k 2 \
      --predicted-absence-recency-k 4 \
      --predicted-max-sessions-per-batch 6 \
      --predicted-minimum-confidence 0.45 \
      --predicted-base-selected-facts 12 \
      --predicted-max-selected-facts 16 \
      --predicted-max-tokens 4096
    remaining="$(failure_count "$output")"
    printf 'departure=%s retry_round=%s failures_after=%s\n' "$label" "$round" "$remaining"
  done

  python3 scripts/experiments/score_memora_generations.py \
    --manifest configs/shared/memora_return_confirmation_manifest.json \
    --input-dir "$output" \
    --output-dir "$output/terra_scored" \
    --judge-base-url "$JUDGE_BASE" \
    --judge-model "$JUDGE_MODEL" \
    --judge-api-key-env OPENAI_API_KEY \
    --arms "$ARMS" \
    --seed 731 \
    --judge-max-tokens 256 \
    --timeout 600 \
    --retries 5 \
    --embedding-base-url https://dashscope-us.aliyuncs.com/compatible-mode/v1 \
    --embedding-model Qwen3-Embedding-8B \
    --embedding-dimensions 1024 \
    --resume
  printf 'departure=%s retry_complete remaining_failures=%s\n' "$label" "$(failure_count "$output")"
}

launch_watcher() {
  local label="$1"
  local fraction="$2"
  local output="results/formal/memora_session_depart${label}_deepseekv4flash0731_terra_four_arm_20260817_01"
  local log="$output/retry.log"
  local pid_file="$output/retry.pid"
  if [[ -s "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(<"$pid_file")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      printf 'departure=%s retry_watcher_already_running pid=%s\n' "$label" "$existing_pid"
      return
    fi
  fi
  nohup bash -lc 'retry_departure "$1" "$2"' _ "$label" "$fraction" >>"$log" 2>&1 < /dev/null &
  local watcher_pid=$!
  printf '%s\n' "$watcher_pid" >"$pid_file"
  printf 'departure=%s retry_watcher_started pid=%s\n' "$label" "$watcher_pid"
}

export -f failure_count retry_departure
export ROOT ACTOR_BASE ACTOR_MODEL JUDGE_BASE JUDGE_MODEL ARMS MAX_RETRY_ROUNDS
launch_watcher 25 0.25
launch_watcher 50 0.50
launch_watcher 75 0.75

unset DEEPSEEK_API_KEY OPENAI_API_KEY OPENAI_API_KEY
