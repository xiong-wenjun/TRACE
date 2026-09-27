"""Diagnostic CUPMem-inspired adapter for benchmark-neutral candidates.

The official CUPMem prototype is specialized to STALE's profile schema and
uses a multi-stage writer (delta extraction, update resolution, indirect
invalidation, stale linking) plus constrained readout.  This adapter preserves
that method boundary in a generic MAS: an actor model performs query-independent
write-side adjudication, only ACTIVE rows enter the current view, and STALE,
REPLACED, or UNKNOWN rows remain audit-only.

It is intentionally identified as ``mechanism_adapted`` rather than an exact
drop-in execution of the benchmark-specific reference repository. Formal
CUPMem results use :mod:`trace.return_methods.semantic_state.cupmem`.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from ..base import (
    ReturnCandidate,
    ReturnMethodCompilation,
    ReturnMethodSpec,
    parse_json_object,
    validate_candidates,
)
from ..lifecycle.temporal_lww import router_return_candidates
from ...router_return_protocol import RouterPostAbsenceCheckpoint


CUPMEM_ADAPTER_ARM = "cupmem"
# Historical import retained while old artifacts and launchers are active.
CUPMEM_ARM = CUPMEM_ADAPTER_ARM
CUPMEM_STATUSES = frozenset(("ACTIVE", "STALE", "REPLACED", "UNKNOWN"))
CUPMEM_PAPER_URL = "https://arxiv.org/abs/2605.06527"
CUPMEM_REFERENCE_URL = "https://github.com/icedreamc/STALE"
CUPMEM_IMPLEMENTATION_KIND = "mechanism_adapted"
CUPMEM_ADAPTER_DISPLAY_NAME = "CUPMem-Adapted (diagnostic)"
CUPMEM_ADAPTER_METHOD = ReturnMethodSpec(
    method=CUPMEM_ADAPTER_ARM,
    display_name=CUPMEM_ADAPTER_DISPLAY_NAME,
    category="diagnostic_semantic_state_adapter",
    produces_memory_view=True,
    paper_url=CUPMEM_PAPER_URL,
    reference_url=CUPMEM_REFERENCE_URL,
    implementation_kind=CUPMEM_IMPLEMENTATION_KIND,
)


def cupmem_adjudication_prompt(
    candidates: Sequence[ReturnCandidate],
    *,
    max_text_chars: int = 1600,
) -> str:
    """Build an ID-safe, query-independent CUPMem write-side request."""

    ordered = validate_candidates(candidates)
    index_by_id = {
        candidate.candidate_id: index
        for index, candidate in enumerate(ordered)
    }
    rows = []
    for index, candidate in enumerate(ordered):
        record = candidate.record(max_text_chars=max_text_chars)
        record.pop("candidate_id", None)
        record["supersedes_candidate_index"] = (
            index_by_id[candidate.supersedes_candidate_id]
            if candidate.supersedes_candidate_id is not None
            else None
        )
        record.pop("supersedes_candidate_id", None)
        record["derived_from_candidate_indices"] = [
            index_by_id[candidate_id]
            for candidate_id in candidate.derived_from_candidate_ids
        ]
        record.pop("derived_from_candidate_ids", None)
        record["candidate_index"] = index
        rows.append(record)
    return (
        "You are the query-independent write-side adjudicator for a CUPMem "
        "memory. The task agent is returning after an absence. Treat each "
        "row as an observed state delta in a typed semantic slot. Detect both "
        "same-slot replacement and cross-slot causal invalidation. Classify "
        "every row as ACTIVE, STALE, REPLACED, or UNKNOWN. ACTIVE means safe "
        "as current grounding; STALE is historical; REPLACED has a newer "
        "settled successor; UNKNOWN means current state is unresolved. Only "
        "ACTIVE rows will be exposed at readout. Repetition by multiple agents "
        "is not independent evidence. No downstream question is available. "
        "Return exactly one JSON object with key decisions and exactly one row "
        "per candidate. Each row must contain candidate_index, status, and "
        "replacement_candidate_index (integer or null). Do not copy opaque "
        "IDs, give reasons, use markdown, or add keys.\n\n"
        + json.dumps(
            {
                "adapter": "cupmem_return_v2",
                "candidates": rows,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _candidate_id(
    raw: Mapping[str, object],
    ordered: Sequence[ReturnCandidate],
) -> str:
    if "candidate_index" in raw:
        index = raw.get("candidate_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("candidate_index must be an integer")
        if not 0 <= index < len(ordered):
            raise ValueError(f"candidate_index outside local range: {index}")
        return ordered[index].candidate_id
    # Compatibility with frozen v1 caches and fixtures.
    return str(
        raw.get("candidate_id")
        or raw.get("item_id")
        or raw.get("memory_id")
        or ""
    ).strip()


def _replacement_id(
    raw: Mapping[str, object],
    ordered: Sequence[ReturnCandidate],
) -> str | None:
    if "replacement_candidate_index" in raw:
        index = raw.get("replacement_candidate_index")
        if index is None:
            return None
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("replacement_candidate_index must be an integer or null")
        if not 0 <= index < len(ordered):
            raise ValueError(
                f"replacement_candidate_index outside local range: {index}"
            )
        return ordered[index].candidate_id
    value = raw.get("replacement_candidate_id")
    if value is None:
        value = raw.get("replacement_item_id")
    if value is None:
        value = raw.get("replacement_memory_id")
    return str(value or "").strip() or None


def compile_cupmem_adjudication(
    candidates: Sequence[ReturnCandidate],
    model_output: str,
) -> ReturnMethodCompilation:
    """Validate one CUPMem adjudication and fail closed on malformed output."""

    ordered = validate_candidates(candidates)
    candidate_ids = tuple(item.candidate_id for item in ordered)
    known = set(candidate_ids)
    candidate_by_id = {item.candidate_id: item for item in ordered}
    candidate_index_by_id = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    parsed: dict[str, dict[str, object]] = {}
    parse_error: str | None = None
    try:
        payload = parse_json_object(model_output)
        rows = payload.get("decisions")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError("CUPMem decisions must be an array")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise TypeError("every CUPMem decision must be an object")
            candidate_id = _candidate_id(raw, ordered)
            if candidate_id not in known or candidate_id in parsed:
                raise ValueError(
                    f"unknown or duplicate CUPMem candidate: {candidate_id}"
                )
            status = str(raw.get("status") or "").strip().upper()
            if status == "UNKNOWN_CURRENT":
                status = "UNKNOWN"
            if status not in CUPMEM_STATUSES:
                raise ValueError(f"unsupported CUPMem status: {status}")
            replacement_id = _replacement_id(raw, ordered)
            if replacement_id is not None and replacement_id not in known:
                raise ValueError(
                    f"unknown CUPMem replacement: {replacement_id}"
                )
            if replacement_id == candidate_id:
                raise ValueError("a CUPMem row cannot replace itself")
            if status == "REPLACED" and replacement_id is None:
                raise ValueError("REPLACED requires a replacement candidate")
            if (
                replacement_id is not None
                and candidate_by_id[replacement_id].logical_time
                < candidate_by_id[candidate_id].logical_time
            ):
                raise ValueError("CUPMem replacement must not precede its target")
            parsed[candidate_id] = {
                "candidate_id": candidate_id,
                "candidate_index": candidate_index_by_id[candidate_id],
                "status": status,
                "replacement_candidate_id": replacement_id,
                "implementation_kind": CUPMEM_IMPLEMENTATION_KIND,
            }
        missing = known - set(parsed)
        if missing:
            raise ValueError(
                "CUPMem omitted candidates: " + ",".join(sorted(missing))
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parse_error = f"{type(error).__name__}: {error}"
        parsed = {}

    decisions = tuple(
        parsed.get(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "status": "UNKNOWN",
                "replacement_candidate_id": None,
                "implementation_kind": CUPMEM_IMPLEMENTATION_KIND,
                "fallback": "fail_closed_invalid_adjudication",
            },
        )
        for index, candidate_id in enumerate(candidate_ids)
    )
    return ReturnMethodCompilation(
        method=CUPMEM_ADAPTER_ARM,
        candidate_ids=candidate_ids,
        selected_ids=tuple(
            candidate_id
            for candidate_id in candidate_ids
            if parsed.get(candidate_id, {}).get("status") == "ACTIVE"
        ),
        decisions=decisions,
        parse_error=parse_error,
    )


def router_cupmem_adjudication_prompt(
    checkpoint: RouterPostAbsenceCheckpoint,
) -> str:
    return cupmem_adjudication_prompt(router_return_candidates(checkpoint))


def compile_router_cupmem(
    checkpoint: RouterPostAbsenceCheckpoint,
    model_output: str,
) -> ReturnMethodCompilation:
    return compile_cupmem_adjudication(
        router_return_candidates(checkpoint), model_output
    )


__all__ = [
    "CUPMEM_ARM",
    "CUPMEM_ADAPTER_ARM",
    "CUPMEM_ADAPTER_DISPLAY_NAME",
    "CUPMEM_ADAPTER_METHOD",
    "CUPMEM_IMPLEMENTATION_KIND",
    "CUPMEM_PAPER_URL",
    "CUPMEM_REFERENCE_URL",
    "CUPMEM_STATUSES",
    "compile_cupmem_adjudication",
    "compile_router_cupmem",
    "cupmem_adjudication_prompt",
    "router_cupmem_adjudication_prompt",
]
