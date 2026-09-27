#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/launchers/environment.sh"
cd "$ROOT"
source scripts/launchers/environment.sh

set -a
source scripts/launchers/environment.sh
set +a
: "${OPENAI_API_KEY:?missing OPENAI_API_KEY}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/dev/shm

bucket_manifest="configs/shared/stale_type2_natural_departure_buckets.json"
run_stamp="20260814_v5_01"

launch_bucket() {
  local bucket="$1"
  local config="configs/variants/qwen35_122b_a10b/stale_type2_official_backend_a_qwen_terra_natural_${bucket}_four_arm.json"
  local output="results/formal/stale_type2_backend_a_qwen_terra_four_arm_natural_${bucket}_${run_stamp}"
  local log="$output/run.log"
  local pid_file="$output/run.pid"
  local -a uids
  local -a command

  mkdir -p "$output"
  if [[ -s "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(<"$pid_file")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      printf '%s already running pid=%s\n' "$bucket" "$existing_pid"
      return
    fi
  fi

  mapfile -t uids < <(
    python3 - "$bucket_manifest" "$bucket" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for row in manifest["buckets"][sys.argv[2]]["records"]:
    print(row["uid"])
PY
  )
  if [[ "${#uids[@]}" -eq 0 ]]; then
    printf 'bucket %s has no records\n' "$bucket" >&2
    return 1
  fi

  command=(
    python3 scripts/experiments/run_stale_type2_mas_return.py
    --config "$config"
    --output-dir "$output"
    --workers 1
  )
  local uid
  for uid in "${uids[@]}"; do
    command+=(--uid "$uid")
  done

  nohup "${command[@]}" >>"$log" 2>&1 < /dev/null &
  local experiment_pid=$!
  printf '%s\n' "$experiment_pid" >"$pid_file"
  printf '%s started pid=%s records=%s output=%s\n' \
    "$bucket" "$experiment_pid" "${#uids[@]}" "$output"
}

launch_bucket early
launch_bucket middle
launch_bucket late

unset OPENAI_API_KEY OPENAI_API_KEY
