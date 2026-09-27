#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT/scripts/launchers/environment.sh"
STAMP="${2:-20260901_01}"
CONFIG="configs/variants/qwen35_122b_a10b/stale_type2_official_backend_a_qwen_terra_trace_ablation200.json"
OUTPUT="results/formal/stale_type2_qwen_terra_trace_ablation200_${STAMP}"
cd "$ROOT"
source scripts/launchers/environment.sh

source scripts/launchers/environment.sh
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"

mkdir -p "$OUTPUT/cache" "$OUTPUT/logs" "$OUTPUT/pids"
sources=(
  results/formal/stale_type2_qwen_terra_six_governance_natural_early_dependency_v7_20260828_01
  results/formal/stale_type2_qwen_terra_six_governance_natural_middle_dependency_v7_20260828_01
  results/formal/stale_type2_qwen_terra_six_governance_natural_late_dependency_v7_20260828_01
  results/formal/stale_type2_qwen_terra_six_governance_natural_early_dependency_v7_uid85_transport_repair_20260829_01
)
for source in "${sources[@]}"; do
  [[ -d "$source/cache" ]] || continue
  while IFS= read -r uid; do
    [[ -n "$uid" ]] || continue
    destination="$OUTPUT/cache/$uid"
    mkdir -p "$destination"
    for stage in extraction_v6_bounded governance_v3 governance_v4; do
      if [[ -d "$source/cache/$uid/$stage" && ! -e "$destination/$stage" ]]; then
        cp -a "$source/cache/$uid/$stage" "$destination/$stage"
      fi
    done
  done < <(find "$source/cache" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
done

cached_uids="$(find "$OUTPUT/cache" -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "$cached_uids" -ne 200 ]]; then
  echo "expected 200 frozen cache UIDs, found $cached_uids" >&2
  exit 3
fi

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
      --workers 8 \
      "${extra[@]}" || true
    completed="$(find "$3/results" -maxdepth 1 -type f -name "*.json" 2>/dev/null | wc -l)"
    unresolved="$(comm -23 \
      <(find "$3/failures" -maxdepth 1 -type f -name "*.json" -printf "%f\n" 2>/dev/null | sort) \
      <(find "$3/results" -maxdepth 1 -type f -name "*.json" -printf "%f\n" 2>/dev/null | sort) \
      | wc -l)"
    printf "round=%s completed=%s expected=200 unresolved=%s\n" \
      "$round" "$completed" "$unresolved"
    if [[ "$completed" -eq 200 && "$unresolved" -eq 0 ]]; then
      exit 0
    fi
    sleep 5
  done
  exit 2
' _ "$ROOT" "$CONFIG" "$OUTPUT" >"$OUTPUT/logs/run.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$OUTPUT/pids/pipeline.pid"
printf 'started pid=%s output=%s cached_uids=%s\n' "$pid" "$OUTPUT" "$cached_uids"
unset OPENAI_API_KEY
