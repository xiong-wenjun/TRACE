#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/launchers/environment.sh"
STAMP="20260826_01"
ACTOR_LOCK="/dev/shm/cupmem_full_qwen_actor.lock"
JUDGE_LOCK="/dev/shm/cupmem_full_terra_judge.lock"

cd "$ROOT"
source scripts/launchers/environment.sh
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"

launch() {
  local name="$1"
  local output="$2"
  local command="$3"
  mkdir -p "$output"
  local pid_file="$output/pipeline.pid"
  if [[ -s "$pid_file" ]]; then
    local prior
    prior="$(<"$pid_file")"
    if kill -0 "$prior" 2>/dev/null; then
      printf '%s already_running pid=%s\n' "$name" "$prior"
      return
    fi
  fi
  nohup bash -lc '
    set -u -o pipefail
    cd "$1" || exit 1
    source scripts/launchers/environment.sh
    : "${DASHSCOPE_API_KEY:?missing DASHSCOPE_API_KEY}"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src:.
    export TMPDIR=/dev/shm
    for attempt in 1 2 3 4 5; do
      if bash -lc "$2"; then
        exit 0
      fi
      printf "retry attempt=%s command_sha256=%s\n" \
        "$attempt" "$(printf "%s" "$2" | sha256sum | cut -d" " -f1)"
      sleep 5
    done
    exit 2
  ' _ "$ROOT" "$command" >>"$output/pipeline.log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$pid_file"
  printf '%s started pid=%s output=%s\n' "$name" "$pid" "$output"
}

for label in 25 50 75; do
  case "$label" in
    25) fraction=0.25 ;;
    50) fraction=0.50 ;;
    75) fraction=0.75 ;;
  esac
  output="results/formal/memora_session_depart${label}_backend_a_qwen_terra_cupmem_full_${STAMP}"
  command="python3 scripts/experiments/run_cupmem_memora.py \
    --manifest configs/shared/memora_return_confirmation_manifest.json \
    --output-dir $output \
    --shared-cache-dir results/cache/cupmem_qwen_memora_contract \
    --departure-fraction $fraction \
    --actor-request-lock-path $ACTOR_LOCK \
    --judge-request-lock-path $JUDGE_LOCK \
    --resume"
  launch "Memora-${label}" "$output" "$command"
done

for bucket in early middle late; do
  config="configs/variants/qwen35_122b_a10b/stale_type2_official_backend_a_qwen_terra_natural_${bucket}_memstrata_memtx_dependency.json"
  output="results/formal/stale_type2_backend_a_qwen_terra_cupmem_full_natural_${bucket}_${STAMP}"
  command="TRACE_GENERATION_REQUEST_LOCK=$ACTOR_LOCK \
    python3 scripts/experiments/run_cupmem_stale.py \
    --config $config --output-dir $output --resume"
  launch "STALE-${bucket}" "$output" "$command"
done

manbench_source="results/formal/manbench_balanced_return_backend_qwen_memstrata_cupmem_memtx_full_20260825_01"
manbench_output="results/formal/manbench_balanced_return_backend_qwen_cupmem_full_${STAMP}"
manbench_command="python3 scripts/experiments/run_cupmem_manbench.py \
  --source-run $manbench_source \
  --output-dir $manbench_output \
  --shared-cache-dir results/cache/cupmem_qwen_manbench_contract \
  --actor-request-lock-path $ACTOR_LOCK \
  --resume"
launch "ManBench-Return" "$manbench_output" "$manbench_command"

unset OPENAI_API_KEY
