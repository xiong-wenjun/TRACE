"""Timestamp/version last-write-wins policy for returning agents."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Mapping, Sequence

from ..base import (
    ReturnCandidate,
    ReturnMethodCompilation,
    ReturnMethodSpec,
    validate_candidates,
)
from ...router_return_protocol import (
    RouterLedgerEvent,
    RouterPostAbsenceCheckpoint,
    RouterWorkstateItem,
)


TEMPORAL_LWW_ARM = "temporal_lww"
TEMPORAL_LWW_METHOD = ReturnMethodSpec(
    method=TEMPORAL_LWW_ARM,
    display_name="Temporal-LWW",
    category="diagnostic_temporal_policy",
    produces_memory_view=True,
)


def _normalized_key(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def compile_temporal_lww(
    candidates: Sequence[ReturnCandidate],
    *,
    invalidated_candidate_ids: Sequence[str] = (),
) -> ReturnMethodCompilation:
    """Keep the latest non-delete candidate for each semantic state key.

    This policy deliberately ignores provenance quality and source consensus.
    Its only authority is actor-visible temporal order/version, making it a
    clean temporal-policy baseline rather than a weaker copy of TRACE.
    """

    ordered = validate_candidates(candidates)
    invalidated = set(invalidated_candidate_ids)
    unknown = invalidated - {item.candidate_id for item in ordered}
    if unknown:
        raise ValueError(
            "Temporal-LWW invalidated unknown candidates: "
            + ",".join(sorted(unknown))
        )
    latest: dict[str, ReturnCandidate] = {}
    for item in ordered:
        key = _normalized_key(item.state_key)
        previous = latest.get(key)
        if previous is None or (
            item.logical_time,
            item.version,
            item.candidate_id,
        ) >= (
            previous.logical_time,
            previous.version,
            previous.candidate_id,
        ):
            latest[key] = item
    selected_set = {
        item.candidate_id
        for item in latest.values()
        if item.operation != "delete" and item.candidate_id not in invalidated
    }
    decisions = []
    for item in ordered:
        winner = latest[_normalized_key(item.state_key)]
        if item.candidate_id in invalidated:
            action = "drop_invalidated_by_lifecycle_event"
        elif winner.candidate_id != item.candidate_id:
            action = "drop_older_write"
        elif item.operation == "delete":
            action = "apply_delete_tombstone"
        else:
            action = "select_latest_write"
        decisions.append(
            {
                "candidate_id": item.candidate_id,
                "state_key": item.state_key,
                "logical_time": item.logical_time,
                "version": item.version,
                "operation": item.operation,
                "action": action,
                "winning_candidate_id": winner.candidate_id,
            }
        )
    return ReturnMethodCompilation(
        method=TEMPORAL_LWW_ARM,
        candidate_ids=tuple(item.candidate_id for item in ordered),
        selected_ids=tuple(
            item.candidate_id
            for item in ordered
            if item.candidate_id in selected_set
        ),
        decisions=tuple(decisions),
    )


def _candidate_union(
    checkpoint: RouterPostAbsenceCheckpoint,
) -> tuple[RouterWorkstateItem, ...]:
    rows = {item.item_id: item for item in checkpoint.departure.workstate_items}
    rows.update(
        {item.item_id: item for item in checkpoint.current_workstate_items}
    )
    return tuple(rows[item_id] for item_id in sorted(rows))


def _structured_text(item: RouterWorkstateItem) -> Mapping[str, object]:
    try:
        value = json.loads(item.text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _lineage_keys(
    items: Sequence[RouterWorkstateItem],
) -> dict[str, str]:
    parent = {
        item.item_id: item.supersedes_item_id
        for item in items
        if item.supersedes_item_id is not None
    }

    def root(item_id: str) -> str:
        seen: set[str] = set()
        current = item_id
        while current in parent and parent[current] not in seen:
            seen.add(current)
            current = str(parent[current])
        return current

    involved = set(parent) | {str(value) for value in parent.values()}
    return {item_id: f"lineage:{root(item_id)}" for item_id in involved}


def router_return_candidates(
    checkpoint: RouterPostAbsenceCheckpoint,
) -> tuple[ReturnCandidate, ...]:
    """Normalize one frozen Router checkpoint without reading oracle fields."""

    items = _candidate_union(checkpoint)
    departure_ids = {
        item.item_id for item in checkpoint.departure.workstate_items
    }
    created_at: dict[str, int] = {}
    for entry in checkpoint.absence_delta.ledger_entries:
        if entry.event is RouterLedgerEvent.WORKSTATE_CREATED:
            item_id = str(entry.payload.get("item_id") or "")
            if item_id:
                created_at[item_id] = entry.logical_time
        elif entry.event is RouterLedgerEvent.WORKSTATE_SUPERSEDED:
            item_id = str(entry.payload.get("replacement_item_id") or "")
            if item_id:
                created_at[item_id] = entry.logical_time
        elif entry.event is RouterLedgerEvent.WORKSTATE_DISPUTED:
            item_id = str(entry.payload.get("challenger_item_id") or "")
            if item_id:
                created_at[item_id] = entry.logical_time
    lineages = _lineage_keys(items)
    known_item_ids = {item.item_id for item in items}
    result = []
    for item in items:
        structured = _structured_text(item)
        operation = str(
            structured.get("predicted_change_kind")
            or structured.get("operation")
            or ""
        ).strip().casefold()
        if operation not in {"add", "update", "delete"}:
            if "deletion tombstone" in item.text.casefold():
                operation = "delete"
            elif item.supersedes_item_id is not None:
                operation = "update"
            else:
                operation = "add"
        semantic_key = str(structured.get("state_key") or "").strip()
        if item.item_id in lineages:
            state_key = lineages[item.item_id]
        elif semantic_key:
            state_key = "semantic:" + _normalized_key(semantic_key)
        elif item.dependency_ids:
            state_key = "dependencies:" + "|".join(
                sorted(_normalized_key(value) for value in item.dependency_ids)
            )
        else:
            state_key = "obligations:" + "|".join(
                sorted(_normalized_key(value) for value in item.obligation_ids)
            )
        logical_time = created_at.get(
            item.item_id,
            (
                checkpoint.departure.departure_ledger_sequence
                if item.item_id in departure_ids
                else checkpoint.absence_delta.relevant_change_count
            ),
        )
        raw_confidence = structured.get("confidence")
        try:
            confidence = (
                float(raw_confidence) if raw_confidence is not None else None
            )
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            confidence = None
        semantic_value = str(
            structured.get("semantic_value")
            or structured.get("proposed_value")
            or structured.get("predicted_value")
            or structured.get("statement")
            or structured.get("value")
            or item.text
        ).strip()
        entity = str(
            structured.get("entity") or structured.get("subject") or ""
        ).strip() or None
        attribute = str(
            structured.get("attribute")
            or structured.get("relation")
            or semantic_key
            or ""
        ).strip() or None
        result.append(
            ReturnCandidate(
                candidate_id=item.item_id,
                state_key=state_key,
                text=item.text,
                logical_time=logical_time,
                version=item.version,
                phase=(
                    "predeparture"
                    if item.item_id in departure_ids
                    else "absence"
                ),
                operation=operation,
                source_id=item.writer_principal_id,
                supersedes_candidate_id=item.supersedes_item_id,
                semantic_value=semantic_value,
                entity=entity,
                attribute=attribute,
                confidence=confidence,
                valid_from=logical_time,
                derived_from_candidate_ids=tuple(
                    dependency_id
                    for dependency_id in item.dependency_ids
                    if dependency_id in known_item_ids
                ),
                provenance_status=item.provenance_status,
                provenance_source_ids=item.provenance_source_ids,
                provenance_receipt_sha256s=item.provenance_receipt_sha256s,
            )
        )
    return validate_candidates(result)


def compile_router_temporal_lww(
    checkpoint: RouterPostAbsenceCheckpoint,
) -> ReturnMethodCompilation:
    """Compile Temporal-LWW from a shared Router checkpoint."""

    candidates = router_return_candidates(checkpoint)
    invalidated = tuple(
        dict.fromkeys(
            checkpoint.absence_delta.superseded_item_ids
            + checkpoint.absence_delta.revoked_item_ids
        )
    )
    compilation = compile_temporal_lww(
        candidates,
        invalidated_candidate_ids=invalidated,
    )
    return replace(compilation, method=TEMPORAL_LWW_ARM)


__all__ = [
    "TEMPORAL_LWW_ARM",
    "compile_router_temporal_lww",
    "compile_temporal_lww",
    "TEMPORAL_LWW_METHOD",
    "router_return_candidates",
]
