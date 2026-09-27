#!/usr/bin/env bash
# Source from the repository root. Never enable shell tracing with credentials.
set +x
if [[ "${TRACE_ENV_LOADED:-0}" != "1" ]]; then
  _trace_env_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  if [[ -f "$_trace_env_root/.env" ]]; then
    set -a
    source "$_trace_env_root/.env"
    set +a
  fi
  unset _trace_env_root
  export TRACE_ENV_LOADED=1
fi
# CLI Qwen and embedding aliases; JSON configs select a provider explicitly.
export TRACE_ACTOR_API_KEY="${TRACE_ACTOR_API_KEY:-${DASHSCOPE_API_KEY:-}}"
export TRACE_EMBEDDING_API_KEY="${TRACE_EMBEDDING_API_KEY:-${DASHSCOPE_API_KEY:-}}"
