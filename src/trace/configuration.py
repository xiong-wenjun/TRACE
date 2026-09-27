"""Explicit, non-secret environment overrides for anonymous release configs."""
from __future__ import annotations

from copy import deepcopy
import os
from typing import Any, Mapping


def resolve_provider_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy endpoint/model/key-NAME overrides without reading credential values.

    Empty overrides preserve the recorded configuration. CLI overrides can be
    applied after this step. No model alias is silently upgraded or substituted.
    """
    result = deepcopy(dict(config))

    def override(section: Any, role: str) -> None:
        if not isinstance(section, dict):
            return
        for suffix, field in (("BASE_URL", "api_base"), ("MODEL", "served_model_id"),
                              ("API_KEY_ENV", "api_key_env")):
            value = os.environ.get(f"TRACE_{role}_{suffix}", "").strip()
            if value:
                section[field] = value

    for name, role in (("model", "ACTOR"), ("judge", "JUDGE"), ("embedding", "EMBEDDING")):
        override(result.get(name), role)
    backend = result.get("memory_backend")
    if isinstance(backend, dict):
        override(backend.get("embedding"), "EMBEDDING")
    return result
