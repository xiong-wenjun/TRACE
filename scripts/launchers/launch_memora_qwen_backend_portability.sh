#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/launchers/environment.sh"
STAMP="20260830_02"
ACTOR_BASE="${TRACE_ACTOR_BASE_URL:-https://dashscope-us.aliyuncs.com/compatible-mode/v1}"
ACTOR_MODEL="${TRACE_ACTOR_MODEL:-Qwen3.5-122B-A10B}"
EMBEDDING_BASE="${TRACE_EMBEDDING_BASE_URL:-https://dashscope-us.aliyuncs.com/compatible-mode/v1}"
EMBEDDING_MODEL="${TRACE_EMBEDDING_MODEL:-Qwen3-Embedding-8B}"
MEMOBASE_BASE="${TRACE_MEMOBASE_BASE_URL:-http://127.0.0.1:18019/api/v1}"
JUDGE_BASE="${TRACE_JUDGE_BASE_URL:-https://api.openai.com/v1}"
JUDGE_MODEL="${TRACE_JUDGE_MODEL:-gpt-5.6-terra}"
ARMS="static_no_churn,trace"
SHARDS=2
MAX_GENERATION_ROUNDS=3
MEM0_DEPS="${TRACE_MEM0_DEPS:-}"

cd "$ROOT"
source scripts/launchers/environment.sh
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"
: "${TRACE_MEMOBASE_API_KEY:?missing TRACE_MEMOBASE_API_KEY}"
launch_backend() {
  local backend="$1"
  local output="results/formal/memora_session_depart50_backend_a_qwen_terra_${backend}_static_trace_${STAMP}"
  local log="$output/pipeline.log"
  local pid_file="$output/pipeline.pid"
  mkdir -p "$output"

  if [[ "$backend" == "memobase" ]]; then
    curl -fsS --max-time 15 \
      -H "Authorization: Bearer $TRACE_MEMOBASE_API_KEY" \
      "$MEMOBASE_BASE/healthcheck" >/dev/null
  fi

  if [[ -s "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(<"$pid_file")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      printf 'backend=%s already_running pid=%s\n' "$backend" "$existing_pid"
      return
    fi
  fi

  nohup bash -lc '
    set -euo pipefail
    root="$1"
    output="$2"
    backend="$3"
    actor_base="$4"
    actor_model="$5"
    embedding_base="$6"
    embedding_model="$7"
    arms="$8"
    memobase_base="$9"
    judge_base="${10}"
    judge_model="${11}"
    shards="${12}"
    max_rounds="${13}"
    mem0_deps="${14}"
    cd "$root"
    source scripts/launchers/environment.sh
    source scripts/launchers/environment.sh
    : "${DASHSCOPE_API_KEY:?missing DASHSCOPE_API_KEY}"
    export TRACE_MEMOBASE_API_KEY
    export MEM0_TELEMETRY=false
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/dev/shm
    export PYTHONPATH="$mem0_deps:src:."

    run_generation_shard() {
      local shard="$1"
      python3 scripts/experiments/generate_memora_mas_return.py \
        --manifest configs/shared/memora_return_confirmation_manifest.json \
        --output-dir "$output" \
        --base-url "$actor_base" \
        --model "$actor_model" \
        --api-key-env TRACE_ACTOR_API_KEY \
        --embedding-base-url "$embedding_base" \
        --embedding-model "$embedding_model" \
        --embedding-dimensions 1024 \
        --memory-backend "$backend" \
        --memory-backend-embedding-dimensions 4096 \
        --memory-backend-storage-dir "$output/native_backend_state" \
        --memobase-base-url "$memobase_base" \
        --memobase-api-key-env TRACE_MEMOBASE_API_KEY \
        --arms "$arms" \
        --state-mode predicted \
        --seed 731 \
        --actor-max-tokens 2048 \
        --max-actor-view-chars 16000 \
        --timeout 600 \
        --retries 5 \
        --resume \
        --shard-count "$shards" \
        --shard-index "$shard" \
        --predicted-departure-fraction 0.50 \
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
    }

    for ((round=1; round<=max_rounds; round++)); do
      pids=()
      for ((shard=0; shard<shards; shard++)); do
        run_generation_shard "$shard" &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do
        wait "$pid" || true
      done
      count="$(find "$output/generations" -maxdepth 1 -type f -name "*.json" ! -name "*.failure.json" | wc -l)"
      printf "backend=%s generation_round=%s completed=%s/137\n" \
        "$backend" "$round" "$count"
      [[ "$count" -eq 137 ]] && break
    done

    count="$(find "$output/generations" -maxdepth 1 -type f -name "*.json" ! -name "*.failure.json" | wc -l)"
    if [[ "$count" -ne 137 ]]; then
      printf "backend=%s generation_incomplete=%s/137\n" "$backend" "$count"
      exit 2
    fi

    judge_pids=()
    for ((shard=0; shard<shards; shard++)); do
      python3 scripts/experiments/score_memora_generations.py \
        --manifest configs/shared/memora_return_confirmation_manifest.json \
        --input-dir "$output" \
        --output-dir "$output/terra_scored" \
        --judge-base-url "$judge_base" \
        --judge-model "$judge_model" \
        --judge-api-key-env OPENAI_API_KEY \
        --arms "$arms" \
        --seed 731 \
        --judge-max-tokens 256 \
        --timeout 600 \
        --retries 5 \
        --embedding-base-url "$embedding_base" \
        --embedding-model "$embedding_model" \
        --embedding-dimensions 1024 \
        --shard-count "$shards" \
        --shard-index "$shard" \
        --resume &
      judge_pids+=("$!")
    done
    for pid in "${judge_pids[@]}"; do
      wait "$pid"
    done
  ' _ "$ROOT" "$output" "$backend" "$ACTOR_BASE" "$ACTOR_MODEL" \
    "$EMBEDDING_BASE" "$EMBEDDING_MODEL" "$ARMS" "$MEMOBASE_BASE" \
    "$JUDGE_BASE" "$JUDGE_MODEL" "$SHARDS" "$MAX_GENERATION_ROUNDS" \
    "$MEM0_DEPS" >>"$log" 2>&1 < /dev/null &
  local experiment_pid=$!
  printf '%s\n' "$experiment_pid" >"$pid_file"
  printf 'backend=%s started pid=%s output=%s\n' \
    "$backend" "$experiment_pid" "$output"
}

launch_backend mem0
launch_backend memobase

unset OPENAI_API_KEY TRACE_MEMOBASE_API_KEY
