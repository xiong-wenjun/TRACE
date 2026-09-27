#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/launchers/environment.sh"
STAMP="20260817_01"
ACTOR_BASE="${TRACE_ACTOR_BASE_URL:-https://api.anthropic.com/v1}"
ACTOR_MODEL="${TRACE_ACTOR_MODEL:-claude-sonnet-5}"
JUDGE_BASE="${TRACE_JUDGE_BASE_URL:-https://api.openai.com/v1}"
JUDGE_MODEL="${TRACE_JUDGE_MODEL:-gpt-5.6-terra}"
ARMS="static_no_churn,restore_old,reset,trace"

cd "$ROOT"
source scripts/launchers/environment.sh
: "${ANTHROPIC_API_KEY:?missing ANTHROPIC_API_KEY}"
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="src:."
export TMPDIR=/dev/shm
export TRACE_GENERATION_LOCK_PATH=/dev/shm/trace_sonnet5_actor.lock

launch_departure() {
  local label="$1"
  local fraction="$2"
  local output="results/formal/memora_session_depart${label}_sonnet5_terra_four_arm_${STAMP}"
  local log="$output/run.log"
  local pid_file="$output/run.pid"
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
    source scripts/launchers/environment.sh
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="src:." TMPDIR=/dev/shm
    export TRACE_GENERATION_LOCK_PATH=/dev/shm/trace_sonnet5_actor.lock
    python3 scripts/experiments/generate_memora_mas_return.py \
      --manifest configs/shared/memora_return_confirmation_manifest.json \
      --output-dir "$2" \
      --base-url "$3" \
      --model "$4" \
      --api-key-env ANTHROPIC_API_KEY \
      --embedding-base-url https://dashscope-us.aliyuncs.com/compatible-mode/v1 \
      --embedding-model Qwen3-Embedding-8B \
      --embedding-dimensions 1024 \
      --arms "$5" \
      --state-mode predicted \
      --seed 731 \
      --actor-max-tokens 2048 \
      --max-actor-view-chars 16000 \
      --timeout 600 \
      --retries 5 \
      --resume \
      --predicted-departure-fraction "$6" \
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
      --judge-base-url "$7" \
      --judge-model "$8" \
      --judge-api-key-env OPENAI_API_KEY \
      --arms "$5" \
      --seed 731 \
      --judge-max-tokens 256 \
      --timeout 600 \
      --retries 5 \
      --embedding-base-url https://dashscope-us.aliyuncs.com/compatible-mode/v1 \
      --embedding-model Qwen3-Embedding-8B \
      --embedding-dimensions 1024 \
      --resume
  ' _ "$ROOT" "$output" "$ACTOR_BASE" "$ACTOR_MODEL" "$ARMS" "$fraction" "$JUDGE_BASE" "$JUDGE_MODEL" \
    >>"$log" 2>&1 < /dev/null &
  local experiment_pid=$!
  printf '%s\n' "$experiment_pid" >"$pid_file"
  printf 'departure=%s started pid=%s output=%s\n' "$label" "$experiment_pid" "$output"
}

launch_departure 25 0.25
launch_departure 50 0.50
launch_departure 75 0.75

unset ANTHROPIC_API_KEY OPENAI_API_KEY OPENAI_API_KEY
