"""Operation-based memory-update baseline adapted for agent re-entry."""

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
from .temporal_lww import router_return_candidates
from ...router_return_protocol import RouterPostAbsenceCheckpoint


KMU_OP_ARM = "kmu_op"
KMU_OPERATIONS = frozenset(("PASS", "REPLACE", "APPEND", "DELETE"))
KMU_OP_METHOD = ReturnMethodSpec(
    method=KMU_OP_ARM,
    display_name="KMU-Op",
    category="diagnostic_memory_update",
    produces_memory_view=True,
)


def _indexed_candidate_record(
    candidate_index: int,
    candidate: ReturnCandidate,
) -> dict[str, object]:
    """Project a candidate without exposing an opaque internal identifier."""

    return {
        "candidate_index": candidate_index,
        "state_key": candidate.state_key,
        "text": candidate.text,
        "logical_time": candidate.logical_time,
        "version": candidate.version,
        "phase": candidate.phase,
        "operation": candidate.operation,
        "source_id": candidate.source_id,
    }


def kmu_operation_prompt(candidates: Sequence[ReturnCandidate]) -> str:
    """Ask for query-independent PASS/REPLACE/APPEND/DELETE updates."""

    ordered = validate_candidates(candidates)
    predeparture = [
        item.record() for item in ordered if item.phase == "predeparture"
    ]
    absence = [item.record() for item in ordered if item.phase == "absence"]
    return (
        "You implement the operation-based memory update baseline for an "
        "agent returning after an absence. Replay each absence candidate in "
        "logical_time order against the predeparture memory. For every "
        "absence candidate choose exactly one operation: PASS when it is "
        "redundant or should not change memory; REPLACE when it supersedes "
        "one existing candidate; APPEND when it adds an independent valid "
        "fact; DELETE when it invalidates one existing candidate without "
        "adding a replacement. Use only the supplied content, state keys, "
        "versions, sources, and chronology. Repetition is not proof and no "
        "downstream question is available. Return exactly one JSON object "
        "with key decisions. Include one row per absence candidate with only "
        "candidate_id, operation, and target_candidate_id (string or null). "
        "REPLACE and DELETE require a target; PASS and APPEND require null. "
        "Do not output explanations, labels such as stale/valid, markdown, or "
        "extra keys.\n\n"
        + json.dumps(
            {
                "predeparture_memory": predeparture,
                "absence_candidates": absence,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def indexed_kmu_operation_prompt(
    candidates: Sequence[ReturnCandidate],
) -> str:
    """Render the same KMU replay contract with local integer references.

    Indices are a transport adapter only.  The evidence, chronological replay,
    and PASS/REPLACE/APPEND/DELETE decision rule are identical to the regular
    KMU operation prompt.
    """

    ordered = validate_candidates(candidates)
    indexed = [
        _indexed_candidate_record(index, candidate)
        for index, candidate in enumerate(ordered)
    ]
    predeparture = [
        row for row in indexed if row["phase"] == "predeparture"
    ]
    absence = [row for row in indexed if row["phase"] == "absence"]
    return (
        "You implement the operation-based memory update baseline for an "
        "agent returning after an absence. Replay each absence candidate in "
        "logical_time order against the predeparture memory. For every "
        "absence candidate choose exactly one operation: PASS when it is "
        "redundant or should not change memory; REPLACE when it supersedes "
        "one existing candidate; APPEND when it adds an independent valid "
        "fact; DELETE when it invalidates one existing candidate without "
        "adding a replacement. Use only the supplied content, state keys, "
        "versions, sources, and chronology. Repetition is not proof and no "
        "downstream question is available. Return exactly one JSON object "
        "with key decisions. Include one row per absence candidate with only "
        "candidate_index, operation, and target_candidate_index (integer or "
        "null). REPLACE and DELETE require the index of a different supplied "
        "candidate as target; PASS and APPEND require null. Candidate indices "
        "are local references, not memory evidence. Do not output opaque "
        "identifiers, explanations, labels such as stale/valid, markdown, or "
        "extra keys.\n\n"
        + json.dumps(
            {
                "predeparture_memory": predeparture,
                "absence_candidates": absence,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def compile_kmu_operations(
    candidates: Sequence[ReturnCandidate],
    model_output: str,
) -> ReturnMethodCompilation:
    """Validate and replay KMU operations; malformed output fails closed."""

    ordered = validate_candidates(candidates)
    candidate_ids = tuple(item.candidate_id for item in ordered)
    known = set(candidate_ids)
    incoming = tuple(item for item in ordered if item.phase == "absence")
    incoming_ids = tuple(item.candidate_id for item in incoming)
    parsed: dict[str, dict[str, object]] = {}
    parse_error: str | None = None
    try:
        payload = parse_json_object(model_output)
        rows = payload.get("decisions")
        if isinstance(rows, Mapping):
            rows = [
                {"candidate_id": str(candidate_id), **dict(value)}
                for candidate_id, value in rows.items()
                if isinstance(value, Mapping)
            ]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError("KMU decisions must be an array or ID-keyed object")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise TypeError("every KMU decision must be an object")
            candidate_id = str(raw.get("candidate_id") or "").strip()
            if candidate_id not in incoming_ids or candidate_id in parsed:
                raise ValueError(
                    f"unknown or duplicate KMU candidate: {candidate_id}"
                )
            operation = str(raw.get("operation") or "").strip().upper()
            if operation not in KMU_OPERATIONS:
                raise ValueError(f"unsupported KMU operation: {operation}")
            target = str(raw.get("target_candidate_id") or "").strip() or None
            if operation in {"REPLACE", "DELETE"}:
                if target is None or target not in known or target == candidate_id:
                    raise ValueError(
                        f"KMU {operation} requires a different known target"
                    )
            elif target is not None:
                raise ValueError(f"KMU {operation} requires a null target")
            parsed[candidate_id] = {
                "candidate_id": candidate_id,
                "operation": operation,
                "target_candidate_id": target,
            }
        missing = set(incoming_ids) - set(parsed)
        if missing:
            raise ValueError(
                "KMU omitted absence candidates: " + ",".join(sorted(missing))
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parse_error = f"{type(error).__name__}: {error}"
        parsed = {}

    if parse_error is not None:
        return ReturnMethodCompilation(
            method=KMU_OP_ARM,
            candidate_ids=candidate_ids,
            selected_ids=(),
            decisions=(),
            parse_error=parse_error,
        )

    active = {
        item.candidate_id for item in ordered if item.phase == "predeparture"
    }
    replay: list[Mapping[str, object]] = []
    for item in incoming:
        decision = parsed[item.candidate_id]
        operation = str(decision["operation"])
        target = decision["target_candidate_id"]
        before = sorted(active)
        if operation == "PASS":
            pass
        elif operation == "APPEND":
            active.add(item.candidate_id)
        elif operation == "REPLACE":
            assert isinstance(target, str)
            active.discard(target)
            active.add(item.candidate_id)
        elif operation == "DELETE":
            assert isinstance(target, str)
            active.discard(target)
        replay.append(
            {
                **decision,
                "active_before": before,
                "active_after": sorted(active),
            }
        )
    return ReturnMethodCompilation(
        method=KMU_OP_ARM,
        candidate_ids=candidate_ids,
        selected_ids=tuple(
            item.candidate_id for item in ordered if item.candidate_id in active
        ),
        decisions=tuple(replay),
    )


def compile_indexed_kmu_operations(
    candidates: Sequence[ReturnCandidate],
    model_output: str,
) -> ReturnMethodCompilation:
    """Map local integer references to IDs, then run the native KMU compiler."""

    ordered = validate_candidates(candidates)
    candidate_ids = tuple(item.candidate_id for item in ordered)
    incoming_indices = {
        index for index, item in enumerate(ordered) if item.phase == "absence"
    }

    def parse_index(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0 or value >= len(ordered):
            raise ValueError(f"{name} is outside the local candidate range")
        return value

    try:
        payload = parse_json_object(model_output)
        rows = payload.get("decisions")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError("KMU decisions must be an array")
        mapped_rows: list[dict[str, object]] = []
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise TypeError("every KMU decision must be an object")
            candidate_index = parse_index(
                raw.get("candidate_index"), "candidate_index"
            )
            if candidate_index not in incoming_indices:
                raise ValueError(
                    "candidate_index must reference an absence candidate"
                )
            raw_target = raw.get("target_candidate_index")
            target_candidate_id: str | None
            if raw_target is None:
                target_candidate_id = None
            else:
                target_index = parse_index(
                    raw_target, "target_candidate_index"
                )
                target_candidate_id = candidate_ids[target_index]
            mapped_rows.append(
                {
                    "candidate_id": candidate_ids[candidate_index],
                    "operation": raw.get("operation"),
                    "target_candidate_id": target_candidate_id,
                }
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return ReturnMethodCompilation(
            method=KMU_OP_ARM,
            candidate_ids=candidate_ids,
            selected_ids=(),
            decisions=(),
            parse_error=f"{type(error).__name__}: {error}",
        )

    return compile_kmu_operations(
        ordered,
        json.dumps({"decisions": mapped_rows}, separators=(",", ":")),
    )


def compile_router_kmu_operations(
    checkpoint: RouterPostAbsenceCheckpoint,
    model_output: str,
) -> ReturnMethodCompilation:
    return compile_kmu_operations(
        router_return_candidates(checkpoint),
        model_output,
    )


__all__ = [
    "KMU_OPERATIONS",
    "KMU_OP_ARM",
    "compile_indexed_kmu_operations",
    "compile_kmu_operations",
    "compile_router_kmu_operations",
    "KMU_OP_METHOD",
    "indexed_kmu_operation_prompt",
    "kmu_operation_prompt",
]
