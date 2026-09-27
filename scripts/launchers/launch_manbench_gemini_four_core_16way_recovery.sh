#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT/scripts/launchers/environment.sh"
SOURCE="${2:-results/formal/manbench_balanced_return_gemini3flash_four_core_16way_full_20260908_01}"
OUTPUT="${3:-results/formal/manbench_gemini3flash_four_core_16way_transport_recovery_all_tokens1024_20260909_01}"
CONFIG="configs/variants/gemini_3_flash/manbench_balanced_return_gemini3flash_four_core_16way_formal_recovery_all_tokens.json"
SOURCE_CONFIG="configs/variants/gemini_3_flash/manbench_balanced_return_gemini3flash_four_core_16way_formal_full.json"
WORKER="scripts/launchers/run_manbench_gemini_recovery_shard.sh"
SHARD_COUNT=16
EXPECTED_COUNT=162

cd "$ROOT"
source scripts/launchers/environment.sh
if [[ -e "$OUTPUT" ]]; then
  printf 'refusing to overwrite existing output: %s\n' "$OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT/logs" "$OUTPUT/pids" "$OUTPUT/shards"

manifest="$OUTPUT/source_failure_episode_ids.txt"
while IFS= read -r path; do
  relative="${path#"$SOURCE/failures/"}"
  task="${relative%/*}"
  index="${relative##*/}"
  printf '%s:%s\n' "$task" "${index%.json}"
done < <(find "$SOURCE/failures" -mindepth 2 -maxdepth 2 -type f -name '*.json' | LC_ALL=C sort) > "$manifest"

target_count="$(wc -l < "$manifest" | tr -d ' ')"
if [[ "$target_count" -ne "$EXPECTED_COUNT" ]]; then
  printf 'expected %s frozen failure IDs, found %s\n' "$EXPECTED_COUNT" "$target_count" >&2
  exit 2
fi
sha256sum "$manifest" > "$OUTPUT/source_failure_episode_ids.sha256"
sha256sum "$CONFIG" > "$OUTPUT/recovery_config.sha256"
sha256sum "$SOURCE_CONFIG" > "$OUTPUT/source_config.sha256"
printf '%s\n' "$SOURCE" > "$OUTPUT/source_result_root.txt"

for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_tag="$(printf '%02d' "$shard")"
  nohup bash "$WORKER" "$ROOT" "$CONFIG" "$OUTPUT" "$SHARD_COUNT" "$shard" "$manifest" \
    > "$OUTPUT/logs/shard-${shard_tag}.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$OUTPUT/pids/shard-${shard_tag}.pid"
  printf 'started shard=%s pid=%s\n' "$shard" "$pid"
done
