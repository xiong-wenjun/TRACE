#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/launchers/environment.sh"
CONFIG="configs/variants/qwen35_122b_a10b/manbench_return_backend_qwen_formal_full.json"
OUTPUT="results/formal/manbench_return_backend_qwen_four_arm_full_20260814_01"
SHARD_COUNT="${MANBENCH_SHARD_COUNT:-8}"

cd "$PROJECT_ROOT"
source scripts/launchers/environment.sh
mkdir -p "$OUTPUT/logs" "$OUTPUT/pids"

export TMPDIR=/dev/shm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="src:."

for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  shard_label="$(printf '%02d' "$shard")"
  pid_path="$OUTPUT/pids/shard-$shard_label.pid"
  log_path="$OUTPUT/logs/shard-$shard_label.log"
  if [[ -s "$pid_path" ]]; then
    existing_pid="$(<"$pid_path")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      printf 'shard=%s already_running pid=%s\n' "$shard_label" "$existing_pid"
      continue
    fi
  fi
  nohup python3 scripts/experiments/run_manbench_return_batch.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --shard-count "$SHARD_COUNT" \
    --shard-index "$shard" \
    --resume \
    >"$log_path" 2>&1 < /dev/null &
  experiment_pid=$!
  printf '%s\n' "$experiment_pid" >"$pid_path"
  printf 'shard=%s started pid=%s\n' "$shard_label" "$experiment_pid"
done

python3 - "$CONFIG" "$OUTPUT" "$SHARD_COUNT" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
output = Path(sys.argv[2])
shard_count = int(sys.argv[3])
config = json.loads(config_path.read_text())
pids = {}
for path in sorted((output / "pids").glob("shard-*.pid")):
    pids[path.stem] = int(path.read_text().strip())
manifest = {
    "schema_version": "manbench_return_formal_launch_v1",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "config_path": str(config_path),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "dataset_sha256": config["dataset"]["dataset_sha256"],
    "expected_examples": config["dataset"]["expected_examples"],
    "model": config["model"]["served_model_id"],
    "fork_arms": config["execution"]["fork_arms"],
    "eligibility_contract": config["execution"]["eligibility_contract"],
    "gold_anchor_injection": config["execution"]["gold_anchor_injection"],
    "shard_count": shard_count,
    "pids": pids,
}
(output / "launch_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY
