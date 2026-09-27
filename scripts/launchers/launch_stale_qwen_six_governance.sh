#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT/scripts/launchers/environment.sh"
STAMP="${2:-20260828_01}"
cd "$ROOT"
source scripts/launchers/environment.sh

source scripts/launchers/environment.sh
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"

for bucket in early middle late; do
  config="configs/variants/qwen35_122b_a10b/stale_type2_official_backend_a_qwen_terra_natural_${bucket}_six_governance_dependency.json"
  output="results/formal/stale_type2_qwen_terra_six_governance_natural_${bucket}_dependency_${STAMP}"
  mkdir -p "$output/logs" "$output/pids"
  nohup bash -lc '
    set -euo pipefail
    cd "$1"
    source scripts/launchers/environment.sh
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src:.
    export TMPDIR=/dev/shm
    for round in 1 2 3 4 5 6; do
      extra=()
      if [[ "$round" -gt 1 ]]; then
        extra=(--retry-seed-offset "$((round - 1))")
      fi
      python -u scripts/experiments/run_stale_type2_mas_return.py \
        --config "$2" \
        --output-dir "$3" \
        --workers 1 \
        --actor-request-lock-path "/dev/shm/qwen_stale_six_${4}.lock" \
        --judge-request-lock-path "/dev/shm/qwen_stale_six_terra.lock" \
        "${extra[@]}" || true
      unresolved="$(comm -23 \
        <(find "$3/failures" -maxdepth 1 -type f -name "*.json" -printf "%f\n" 2>/dev/null | sort) \
        <(find "$3/results" -maxdepth 1 -type f -name "*.json" -printf "%f\n" 2>/dev/null | sort) \
        | wc -l)"
      expected="$(jq -r ".dataset.records" "$2")"
      completed="$(find "$3/results" -maxdepth 1 -type f -name "*.json" | wc -l)"
      printf "round=%s bucket=%s completed=%s expected=%s unresolved=%s\n" \
        "$round" "$4" "$completed" "$expected" "$unresolved"
      if [[ "$completed" -eq "$expected" && "$unresolved" -eq 0 ]]; then
        exit 0
      fi
      sleep 5
    done
    exit 2
  ' _ "$ROOT" "$config" "$output" "$bucket" \
    >"$output/logs/run.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$output/pids/pipeline.pid"
  printf 'started bucket=%s pid=%s output=%s\n' "$bucket" "$pid" "$output"
done

unset OPENAI_API_KEY
