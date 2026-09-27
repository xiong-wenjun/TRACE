#!/usr/bin/env bash
set -u

repo=${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$repo/scripts/launchers/environment.sh"
output=${2:-results/formal/stale_type2_qwen_terra_stratified60_cupmem_full_context_seed20260826_01}
rounds=${3:-4}

cd "$repo" || exit 1
source scripts/launchers/environment.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=src:.
set -a
source scripts/launchers/environment.sh
set +a

manifest=configs/shared/stale_type2_stratified_subset_seed20260826_n60.json
config_backend_a=configs/variants/qwen35_122b_a10b/stale_type2_official_backend_a_qwen_terra_stratified60_cupmem_full_context.json
config_backend_b=configs/variants/qwen35_122b_a10b/stale_type2_official_backend_b_qwen_terra_stratified60_cupmem_full_context.json

mkdir -p "$output/logs" "$output/pids"

mapfile -t backend_a_uids < <(
  python -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for bucket in ("early", "middle", "late"):
    for uid in value["buckets"][bucket]["uids"][::2]:
        print(uid)
' "$manifest"
)
mapfile -t backend_b_uids < <(
  python -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for bucket in ("early", "middle", "late"):
    for uid in value["buckets"][bucket]["uids"][1::2]:
        print(uid)
' "$manifest"
)

if [ "${#backend_a_uids[@]}" -ne 30 ] || [ "${#backend_b_uids[@]}" -ne 30 ]; then
  echo "expected a deterministic 30/30 stratified split" >&2
  exit 2
fi

run_lane() {
  local config=$1
  local lane=$2
  local round=$3
  shift 3
  local uid_args=()
  local uid
  for uid in "$@"; do
    uid_args+=(--uid "$uid")
  done
  python -u scripts/experiments/run_cupmem_stale.py \
    --config "$config" \
    --output-dir "$output" \
    --expected-total-records 60 \
    --resume \
    "${uid_args[@]}" \
    >>"$output/logs/${lane}-round-${round}.log" 2>&1
}

for round in $(seq 1 "$rounds"); do
  complete=$(find "$output/results" -maxdepth 1 -type f -name '*.json.gz' 2>/dev/null | wc -l)
  printf 'round=%s complete=%s/60\n' "$round" "$complete" \
    >>"$output/logs/supervisor.log"
  if [ "$complete" -ge 60 ]; then
    exit 0
  fi

  run_lane "$config_backend_a" backend_a "$round" "${backend_a_uids[@]}" &
  pid1=$!
  echo "$pid1" >"$output/pids/backend_a.pid"
  run_lane "$config_backend_b" backend_b "$round" "${backend_b_uids[@]}" &
  pid2=$!
  echo "$pid2" >"$output/pids/backend_b.pid"

  wait "$pid1" || true
  wait "$pid2" || true
done

complete=$(find "$output/results" -maxdepth 1 -type f -name '*.json.gz' 2>/dev/null | wc -l)
[ "$complete" -ge 60 ]
