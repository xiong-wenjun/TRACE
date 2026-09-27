#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT/scripts/launchers/environment.sh"
STAMP="${2:-20260827_01}"
cd "$ROOT"
source scripts/launchers/environment.sh

for bucket in early middle late; do
  config="configs/variants/gemini_3_flash/stale_type2_official_gemini3flash_terra_natural_${bucket}_four_core_dependency.json"
  if [[ "$bucket" == "early" ]]; then config="configs/models/gemini_3_flash/stale_type2.json"; fi
  output="results/formal/stale_type2_gemini3flash_terra_four_core_natural_${bucket}_dependency_${STAMP}"
  mkdir -p "$output/logs" "$output/pids"
  nohup bash -lc '
    set -euo pipefail
    cd "$1"
    set -a
    source scripts/launchers/environment.sh
    source scripts/launchers/environment.sh
    set +a
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src:.
    export TMPDIR=/dev/shm
    for round in 1 2 3 4 5 6; do
      python -u scripts/experiments/run_stale_type2_mas_return.py \
        --config "$2" \
        --output-dir "$3" \
        --workers 1 \
        --actor-request-lock-path "/dev/shm/gemini_stale_four_${4}.lock" \
        --judge-request-lock-path "/dev/shm/gemini_stale_four_terra.lock"
      unresolved="$(comm -23 \
        <(find "$3/failures" -maxdepth 1 -type f -name "*.json" -printf "%f\n" 2>/dev/null | sort) \
        <(find "$3/results" -maxdepth 1 -type f -name "*.json" -printf "%f\n" 2>/dev/null | sort) \
        | wc -l)"
      printf "round=%s bucket=%s unresolved=%s\n" "$round" "$4" "$unresolved"
      if [[ "$unresolved" -eq 0 ]]; then
        expected="$(jq -r ".dataset.records" "$2")"
        completed="$(find "$3/results" -maxdepth 1 -type f -name "*.json" | wc -l)"
        if [[ "$completed" -eq "$expected" ]]; then
          exit 0
        fi
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
