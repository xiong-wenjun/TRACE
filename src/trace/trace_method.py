"""Canonical naming for Temporal Return Admission under Context Evolution.

All runtime code and newly serialized artifacts use ``trace``. Historical
``review`` and ``latch`` spellings are accepted only at deserialization
boundaries so frozen artifacts remain readable without leaking legacy names
into current contracts.
"""

from __future__ import annotations

from typing import Mapping, TypeVar

from .return_methods.governance.trace import TRACE_ARM, TRACE_DISPLAY_NAME


LEGACY_REVIEW_ARM = "review"
LEGACY_LATCH_ARM = "latch"

_LEGACY_ARMS = frozenset((LEGACY_REVIEW_ARM, LEGACY_LATCH_ARM))
_LEGACY_MODE_PREFIXES = {
    LEGACY_LATCH_ARM: TRACE_ARM,
    LEGACY_LATCH_ARM + "_without_fallback": TRACE_ARM + "_without_fallback",
    LEGACY_LATCH_ARM + "_without_provenance": TRACE_ARM + "_without_provenance",
    LEGACY_LATCH_ARM + "_without_freshness": TRACE_ARM + "_without_freshness",
}
_T = TypeVar("_T")


def canonical_return_arm(value: object) -> str:
    """Return the canonical public arm identifier."""

    arm = str(value).strip()
    return TRACE_ARM if arm in _LEGACY_ARMS else arm


def internal_return_arm(value: object) -> str:
    """Return the canonical runtime arm identifier.

    The function name is retained for source compatibility with experiment
    launchers. Unlike the historical implementation, it never emits a legacy
    compiler key.
    """

    return canonical_return_arm(value)


def canonical_trace_mode(value: object) -> str:
    """Canonicalize TRACE ablation names from frozen historical artifacts."""

    mode = str(value).strip()
    return _LEGACY_MODE_PREFIXES.get(mode, mode)


def canonicalize_arm_mapping(
    values: Mapping[str, _T],
) -> dict[str, _T]:
    """Canonicalize arm keys and reject ambiguous legacy mixtures."""

    result: dict[str, _T] = {}
    for raw_arm, item in values.items():
        arm = canonical_return_arm(raw_arm)
        if arm in result:
            raise ValueError(
                "arm mapping contains multiple TRACE or legacy entries"
            )
        result[arm] = item
    return result


__all__ = [
    "LEGACY_LATCH_ARM",
    "LEGACY_REVIEW_ARM",
    "TRACE_ARM",
    "TRACE_DISPLAY_NAME",
    "canonical_return_arm",
    "canonical_trace_mode",
    "canonicalize_arm_mapping",
    "internal_return_arm",
]
