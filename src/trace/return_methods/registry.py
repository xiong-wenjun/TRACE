"""Single method catalog used by runners, reports, and paper tables.

Every method owns its identity in its implementation module.  This registry
only validates and indexes those specifications, preventing display names,
artifact keys, and implementation-fidelity labels from drifting apart.
"""

from __future__ import annotations

from .base import ReturnMethodSpec
from .governance import TRACE_METHOD
from .lifecycle import KMU_OP_METHOD, TEMPORAL_LWW_METHOD
from .policies import (
    CHECKPOINT_REPLAY_METHOD,
    RESET_METHOD,
    RESTORE_METHOD,
    STATIC_METHOD,
)
from .prompt_defense import (
    COGNITIVE_ANCHORING_ARM,
    COGNITIVE_ANCHORING_METHOD,
    NO_DEFENSE_ARM,
    NO_DEFENSE_METHOD,
    SOURCE_SCRUTINY_ARM,
    SOURCE_SCRUTINY_METHOD,
    PromptDefenseSpec,
    cognitive_anchoring_defense,
    source_scrutiny_defense,
)
from .semantic_state import CUPMEM_ADAPTER_METHOD, CUPMEM_METHOD
from .temporal import MEMSTRATA_METHOD
from .transactional import MEMTX_METHOD


# Compatibility name retained for existing imports.
ReturnMethodMetadata = ReturnMethodSpec

CORE_CONTROL_METHODS = (STATIC_METHOD, RESTORE_METHOD, RESET_METHOD)
PUBLISHED_GOVERNANCE_METHODS = (MEMSTRATA_METHOD, MEMTX_METHOD, TRACE_METHOD)
DIAGNOSTIC_METHODS = (
    CHECKPOINT_REPLAY_METHOD,
    TEMPORAL_LWW_METHOD,
    KMU_OP_METHOD,
    CUPMEM_ADAPTER_METHOD,
)
FULL_MEMORY_SYSTEM_METHODS = (CUPMEM_METHOD,)
ANSWER_TIME_CONTROL_METHODS = (
    NO_DEFENSE_METHOD,
    COGNITIVE_ANCHORING_METHOD,
    SOURCE_SCRUTINY_METHOD,
)

PUBLISHED_GOVERNANCE_ARMS = tuple(
    method.method for method in (MEMSTRATA_METHOD, MEMTX_METHOD)
)
DIAGNOSTIC_ADAPTER_ARMS = (CUPMEM_ADAPTER_METHOD.method,)
FULL_MEMORY_SYSTEM_ARMS = tuple(
    method.method for method in FULL_MEMORY_SYSTEM_METHODS
)
LIFECYCLE_POLICY_ARMS = tuple(
    method.method
    for method in (
        TEMPORAL_LWW_METHOD,
        KMU_OP_METHOD,
        MEMSTRATA_METHOD,
        MEMTX_METHOD,
    )
)
PROMPT_DEFENSE_ARMS = (COGNITIVE_ANCHORING_ARM, SOURCE_SCRUTINY_ARM)
ANSWER_TIME_CONTROL_ARMS = (NO_DEFENSE_ARM, *PROMPT_DEFENSE_ARMS)

_CANONICAL_METHODS = (
    *CORE_CONTROL_METHODS,
    *PUBLISHED_GOVERNANCE_METHODS,
    *DIAGNOSTIC_METHODS,
    *FULL_MEMORY_SYSTEM_METHODS,
    *ANSWER_TIME_CONTROL_METHODS,
)


def _build_registry() -> dict[str, ReturnMethodSpec]:
    registry: dict[str, ReturnMethodSpec] = {}
    for spec in _CANONICAL_METHODS:
        for key in (spec.method, *spec.aliases):
            existing = registry.get(key)
            if existing is not None and existing is not spec:
                raise ValueError(
                    f"RETURN method key {key!r} is claimed by both "
                    f"{existing.method!r} and {spec.method!r}"
                )
            registry[key] = spec
    return registry


RETURN_METHOD_REGISTRY = _build_registry()


def get_return_method(method: str) -> ReturnMethodSpec:
    """Resolve a canonical method or a frozen historical artifact alias."""

    try:
        return RETURN_METHOD_REGISTRY[method]
    except KeyError as error:
        raise KeyError(f"unknown RETURN method: {method}") from error


def canonical_method_name(method: str) -> str:
    """Return the canonical artifact key without changing old files on disk."""

    metadata = RETURN_METHOD_REGISTRY.get(method)
    return metadata.method if metadata is not None else method


def method_display_name(method: str) -> str:
    """Return the single paper-facing name for an artifact method key."""

    metadata = RETURN_METHOD_REGISTRY.get(method)
    return metadata.display_name if metadata is not None else method


def prompt_defense_for(method: str) -> PromptDefenseSpec:
    canonical = canonical_method_name(method)
    if canonical == COGNITIVE_ANCHORING_ARM:
        return cognitive_anchoring_defense()
    if canonical == SOURCE_SCRUTINY_ARM:
        return source_scrutiny_defense()
    raise KeyError(f"not a registered prompt defense: {method}")


__all__ = [
    "ANSWER_TIME_CONTROL_ARMS",
    "ANSWER_TIME_CONTROL_METHODS",
    "CORE_CONTROL_METHODS",
    "DIAGNOSTIC_ADAPTER_ARMS",
    "DIAGNOSTIC_METHODS",
    "FULL_MEMORY_SYSTEM_ARMS",
    "FULL_MEMORY_SYSTEM_METHODS",
    "LIFECYCLE_POLICY_ARMS",
    "PROMPT_DEFENSE_ARMS",
    "PUBLISHED_GOVERNANCE_ARMS",
    "PUBLISHED_GOVERNANCE_METHODS",
    "RETURN_METHOD_REGISTRY",
    "ReturnMethodMetadata",
    "canonical_method_name",
    "get_return_method",
    "method_display_name",
    "prompt_defense_for",
]
