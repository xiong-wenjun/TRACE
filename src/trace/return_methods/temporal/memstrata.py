"""MemStrata-style assertion ledger for returning agents.

This is a mechanism-level reimplementation of the temporal layer described in
MemStrata, not an import of the product's daemon.  The public open-core release
does not expose the paper's complete ``temporal_v6`` ingestion path.  The
adapter therefore implements the auditable pieces required by the paper:

* normalized ``(entity, attribute)`` assertion keys;
* exact semantic duplicate suppression;
* retain-then-supersede for a newer value on the same assertion key; and
* a bitemporal ledger recording validity and transaction order.

Only actor-visible ``ReturnCandidate`` fields are consumed.  Benchmark labels,
questions, and evaluator rubrics are deliberately unavailable here.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Sequence

from ..base import (
    ReturnCandidate,
    ReturnMethodCompilation,
    ReturnMethodSpec,
    validate_candidates,
)
from ..lifecycle.temporal_lww import router_return_candidates
from ...router_return_protocol import RouterPostAbsenceCheckpoint


MEMSTRATA_ARM = "memstrata"
MEMSTRATA_PAPER_URL = "https://arxiv.org/abs/2606.26511"
MEMSTRATA_IMPLEMENTATION_KIND = "paper_mechanism_reimplementation"
MEMSTRATA_METHOD = ReturnMethodSpec(
    method=MEMSTRATA_ARM,
    display_name="MemStrata",
    category="temporal_freshness",
    produces_memory_view=True,
    paper_url=MEMSTRATA_PAPER_URL,
    implementation_kind=MEMSTRATA_IMPLEMENTATION_KIND,
)


def _normalize(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _assertion_key(candidate: ReturnCandidate) -> str:
    entity = _normalize(candidate.entity)
    attribute = _normalize(candidate.attribute)
    if entity and attribute:
        return f"{entity}::{attribute}"
    return _normalize(candidate.state_key)


def _semantic_fingerprint(candidate: ReturnCandidate) -> str:
    value = candidate.semantic_value or candidate.text
    # Punctuation and whitespace differences must not turn an exact assertion
    # duplicate into a new temporal version.
    return re.sub(r"[^\w]+", " ", _normalize(value)).strip()


def compile_memstrata(
    candidates: Sequence[ReturnCandidate],
    *,
    invalidated_candidate_ids: Sequence[str] = (),
) -> ReturnMethodCompilation:
    """Compile a current-only view while retaining a bitemporal audit ledger."""

    ordered = validate_candidates(candidates)
    known = {item.candidate_id for item in ordered}
    invalidated = set(invalidated_candidate_ids)
    unknown = invalidated - known
    if unknown:
        raise ValueError(
            "MemStrata invalidated unknown candidates: "
            + ",".join(sorted(unknown))
        )

    active_by_key: dict[str, str] = {}
    by_id = {item.candidate_id: item for item in ordered}
    ledger: dict[str, dict[str, object]] = {}

    def close(candidate_id: str, *, at: int, superseded_by: str) -> None:
        row = ledger[candidate_id]
        row["status"] = "SUPERSEDED"
        row["valid_to"] = at
        row["superseded_by_candidate_id"] = superseded_by

    for transaction_time, candidate in enumerate(ordered, start=1):
        assertion_key = _assertion_key(candidate)
        valid_from = (
            candidate.valid_from
            if candidate.valid_from is not None
            else candidate.logical_time
        )
        row: dict[str, object] = {
            "candidate_id": candidate.candidate_id,
            "assertion_key": assertion_key,
            "semantic_fingerprint": _semantic_fingerprint(candidate),
            "transaction_time": transaction_time,
            "valid_from": valid_from,
            "valid_to": candidate.valid_to,
            "status": "ACTIVE",
            "action": "store_novel_assertion",
            "duplicate_of_candidate_id": None,
            "superseded_by_candidate_id": None,
            "implementation_kind": MEMSTRATA_IMPLEMENTATION_KIND,
        }
        ledger[candidate.candidate_id] = row

        explicit_target = candidate.supersedes_candidate_id
        if explicit_target is not None and explicit_target in ledger:
            close(
                explicit_target,
                at=valid_from,
                superseded_by=candidate.candidate_id,
            )
            explicit_key = _assertion_key(by_id[explicit_target])
            if active_by_key.get(explicit_key) == explicit_target:
                active_by_key.pop(explicit_key, None)

        current_id = active_by_key.get(assertion_key)
        if candidate.operation == "delete":
            if current_id is not None:
                close(
                    current_id,
                    at=valid_from,
                    superseded_by=candidate.candidate_id,
                )
            row["status"] = "TOMBSTONE"
            row["action"] = "close_assertion_with_tombstone"
            row["valid_to"] = valid_from
            active_by_key.pop(assertion_key, None)
            continue

        if current_id is None:
            active_by_key[assertion_key] = candidate.candidate_id
            if explicit_target is not None:
                row["action"] = "store_explicit_successor"
            continue

        current = by_id[current_id]
        if _semantic_fingerprint(current) == _semantic_fingerprint(candidate):
            row["status"] = "DUPLICATE"
            row["action"] = "suppress_exact_semantic_duplicate"
            row["duplicate_of_candidate_id"] = current_id
            row["valid_to"] = valid_from
            continue

        close(
            current_id,
            at=valid_from,
            superseded_by=candidate.candidate_id,
        )
        row["action"] = "supersede_same_assertion_key"
        active_by_key[assertion_key] = candidate.candidate_id

    final_logical_time = max(
        (item.logical_time for item in ordered), default=0
    )
    for candidate_id in invalidated:
        row = ledger[candidate_id]
        if row["status"] == "ACTIVE":
            row["status"] = "INVALIDATED"
            row["action"] = "close_from_actor_visible_lifecycle_event"
            row["valid_to"] = final_logical_time
        key = _assertion_key(by_id[candidate_id])
        if active_by_key.get(key) == candidate_id:
            active_by_key.pop(key, None)

    for key, candidate_id in tuple(active_by_key.items()):
        candidate = by_id[candidate_id]
        if (
            candidate.valid_to is not None
            and candidate.valid_to < final_logical_time
        ):
            ledger[candidate_id]["status"] = "EXPIRED"
            ledger[candidate_id]["action"] = "exclude_expired_assertion"
            active_by_key.pop(key, None)

    selected = set(active_by_key.values())
    return ReturnMethodCompilation(
        method=MEMSTRATA_ARM,
        candidate_ids=tuple(item.candidate_id for item in ordered),
        selected_ids=tuple(
            item.candidate_id
            for item in ordered
            if item.candidate_id in selected
        ),
        decisions=tuple(ledger[item.candidate_id] for item in ordered),
    )


def compile_router_memstrata(
    checkpoint: RouterPostAbsenceCheckpoint,
) -> ReturnMethodCompilation:
    """Compile MemStrata from the shared frozen Router checkpoint."""

    invalidated = tuple(
        dict.fromkeys(
            checkpoint.absence_delta.superseded_item_ids
            + checkpoint.absence_delta.revoked_item_ids
        )
    )
    compilation = compile_memstrata(
        router_return_candidates(checkpoint),
        invalidated_candidate_ids=invalidated,
    )
    return replace(compilation, method=MEMSTRATA_ARM)


__all__ = [
    "MEMSTRATA_ARM",
    "MEMSTRATA_IMPLEMENTATION_KIND",
    "MEMSTRATA_METHOD",
    "MEMSTRATA_PAPER_URL",
    "compile_memstrata",
    "compile_router_memstrata",
]
