#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT/scripts/launchers/environment.sh"
STAMP="${2:-20260826_01}"
cd "$ROOT"
source scripts/launchers/environment.sh

source scripts/launchers/environment.sh
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"

launch_bucket() {
  local bucket="$1"
  local config="$2"
  local source_results="$3"
  local output="$4"
  mkdir -p "$output/logs" "$output/pids"

  nohup bash -lc '
    set -u -o pipefail
    cd "$1" || exit 1
    config="$2"
    output="$3"
    source_results="$4"
    bucket="$5"
    source scripts/launchers/environment.sh
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src:.
    export TMPDIR=/dev/shm

    mapfile -t uids < <(
      find "$source_results" -maxdepth 1 -type f -name "*.json" \
        -printf "%f\n" | sed "s/\.json$//" | sort
    )
    uid_args=()
    for uid in "${uids[@]}"; do
      uid_args+=(--uid "$uid")
    done
    printf "bucket=%s uid_count=%s arms=memstrata,memtx\n" \
      "$bucket" "${#uids[@]}"

    for seed_offset in 0 1 2 3 4 5; do
      extra=()
      if [[ "$seed_offset" -gt 0 ]]; then
        extra=(--retry-seed-offset "$seed_offset")
      fi
      if python3 scripts/experiments/run_stale_type2_mas_return.py \
        --config "$config" \
        --output-dir "$output" \
        --workers 1 \
        --allow-anchor-fallback \
        --actor-request-lock-path "/dev/shm/trace_qwen_stale_two_${bucket}_actor.lock" \
        --judge-request-lock-path "/dev/shm/trace_terra_stale_two_${bucket}_judge.lock" \
        "${uid_args[@]}" "${extra[@]}"; then
        exit 0
      fi
      printf "retry bucket=%s seed_offset=%s\n" "$bucket" "$seed_offset"
      sleep 5
    done
    exit 2
  ' _ "$ROOT" "$config" "$output" "$source_results" "$bucket" \
    >"$output/logs/pipeline.log" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$output/pids/pipeline.pid"
  printf '%s started pid=%s output=%s\n' "$bucket" "$pid" "$output"
}

for bucket in early middle late; do
  config="configs/variants/qwen35_122b_a10b/stale_type2_official_backend_a_qwen_terra_natural_${bucket}_memstrata_memtx_dependency.json"
  source_results="results/formal/stale_type2_backend_a_qwen_terra_six_lifecycle_arm_natural_${bucket}_20260819_bounded_v3_01/results"
  output="results/formal/stale_type2_backend_a_qwen_terra_memstrata_memtx_natural_${bucket}_dependency_${STAMP}"
  launch_bucket "$bucket" "$config" "$source_results" "$output"
done

unset OPENAI_API_KEY
