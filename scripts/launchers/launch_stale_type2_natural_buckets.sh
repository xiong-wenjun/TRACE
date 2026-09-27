#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/launchers/environment.sh"
cd "$ROOT"
source scripts/launchers/environment.sh

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  read -rsp 'Terra Judge API key: ' OPENAI_API_KEY
  printf '\n'
  export OPENAI_API_KEY
fi
trap 'unset OPENAI_API_KEY' EXIT

python - <<'PY'
import json
import os
import urllib.request

request = urllib.request.Request(
    (os.environ.get("TRACE_JUDGE_BASE_URL") or "https://api.openai.com/v1").rstrip("/") + "/models",
    headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
)
with urllib.request.urlopen(request, timeout=30) as response:
    models = json.load(response)
model_ids = {str(item.get("id")) for item in models.get("data", [])}
if (os.environ.get("TRACE_JUDGE_MODEL") or "gpt-5.6-terra") not in model_ids:
    raise SystemExit("Configured judge model is unavailable from the endpoint")
print("Judge health check passed.")
PY

manifest="configs/shared/stale_type2_natural_departure_buckets.json"

launch_bucket() {
  local bucket="$1"
  local config="configs/variants/qwen35_122b_a10b/stale_type2_official_backend_a_qwen_terra_natural_${bucket}.json"
  local output="results/formal/stale_type2_official_backend_a_qwen_terra_natural_${bucket}"
  local log="$output/run.log"
  local pid_file="$output/run.pid"
  local -a uids
  local -a command

  mkdir -p "$output"
  if [[ -s "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(<"$pid_file")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      printf '%s already running with pid=%s\n' "$bucket" "$existing_pid"
      return
    fi
  fi

  mapfile -t uids < <(
    python - "$manifest" "$bucket" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for row in manifest["buckets"][sys.argv[2]]["records"]:
    print(row["uid"])
PY
  )
  if [[ "${#uids[@]}" -eq 0 ]]; then
    printf 'bucket %s contains no UIDs\n' "$bucket" >&2
    return 1
  fi

  command=(
    python scripts/experiments/run_stale_type2_mas_return.py
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
  printf '%s started pid=%s records=%s log=%s\n' \
    "$bucket" "$experiment_pid" "${#uids[@]}" "$log"
}

launch_bucket early
launch_bucket middle
launch_bucket late

unset OPENAI_API_KEY
trap - EXIT
