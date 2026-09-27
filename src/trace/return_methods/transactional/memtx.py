"""MemTX transaction-state adapter for shared RETURN memory.

The implementation follows the released MemTX manager, state machine,
conflict detector, and dependency graph at the pinned reference commit below.
It maps benchmark-neutral ``ReturnCandidate`` rows to transactions, applies
evidence/validity/dependency validation, resolves same-slot conflicts by source
authority, quarantines unresolved equal-authority cross-agent conflicts, and
exposes only committed records at readout.

No network or model call is made here.  Source authority is an explicit frozen
input; when absent, every task agent receives the same authority.  This avoids
silently granting the returning agent special status.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
import re
from typing import Mapping, Sequence

from ..base import (
    ReturnCandidate,
    ReturnMethodCompilation,
    ReturnMethodSpec,
    validate_candidates,
)
from ..lifecycle.temporal_lww import router_return_candidates
from ...router_return_protocol import RouterPostAbsenceCheckpoint


MEMTX_ARM = "memtx"
MEMTX_PAPER_URL = "https://arxiv.org/abs/2607.23929"
MEMTX_REFERENCE_URL = "https://github.com/lxy1134/MEMTX_"
MEMTX_REFERENCE_COMMIT = "4e1124ccd5fd384857463020e3497aa73ec93be6"
MEMTX_IMPLEMENTATION_KIND = "official_code_mechanism_reimplementation"
MEMTX_METHOD = ReturnMethodSpec(
    method=MEMTX_ARM,
    display_name="MemTX",
    category="shared_memory_transaction",
    produces_memory_view=True,
    paper_url=MEMTX_PAPER_URL,
    reference_url=MEMTX_REFERENCE_URL,
    implementation_kind=MEMTX_IMPLEMENTATION_KIND,
    reference_commit=MEMTX_REFERENCE_COMMIT,
)


class MemTxState(str, Enum):
    RAW = "RAW"
    TENTATIVE = "TENTATIVE"
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"
    ACTION_SAFE = "ACTION_SAFE"
    QUARANTINED = "QUARANTINED"
    SUPERSEDED = "SUPERSEDED"
    REVOKED = "REVOKED"


class MemTxIsolation(str, Enum):
    RAW = "raw"
    COMMITTED = "committed"
    VERIFIED = "verified"
    CAUSALLY_STABLE = "causally-stable"
    ACTION_SAFE = "action-safe"


def _normalize(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _slot(candidate: ReturnCandidate) -> tuple[str, str]:
    entity = _normalize(candidate.entity) or "shared_return_state"
    attribute = _normalize(candidate.attribute) or _normalize(candidate.state_key)
    return entity, attribute


def _value(candidate: ReturnCandidate) -> str:
    value = candidate.semantic_value or candidate.text
    return re.sub(r"[^\w]+", " ", _normalize(value)).strip()


def compile_memtx(
    candidates: Sequence[ReturnCandidate],
    *,
    source_authority: Mapping[str, float] | None = None,
    isolation: MemTxIsolation = MemTxIsolation.COMMITTED,
    minimum_confidence: float = 0.6,
    authority_override_threshold: float = 0.9,
    default_confidence: float = 0.75,
    default_source_authority: float = 0.5,
) -> ReturnMethodCompilation:
    """Execute a serializable transaction per actor-visible candidate."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between zero and one")
    if not 0.0 <= authority_override_threshold <= 1.0:
        raise ValueError(
            "authority_override_threshold must be between zero and one"
        )
    if not 0.0 <= default_confidence <= 1.0:
        raise ValueError("default_confidence must be between zero and one")
    if not 0.0 <= default_source_authority <= 1.0:
        raise ValueError("default_source_authority must be between zero and one")

    ordered = validate_candidates(candidates)
    by_id = {item.candidate_id: item for item in ordered}
    authority = dict(source_authority or {})
    for source_id, score in authority.items():
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"invalid source authority for {source_id}")
    final_logical_time = max(
        (item.logical_time for item in ordered), default=0
    )
    state: dict[str, MemTxState] = {}
    decisions: dict[str, dict[str, object]] = {}
    active_by_slot: dict[tuple[str, str], set[str]] = {}
    dependents: dict[str, set[str]] = {}

    def source_score(candidate: ReturnCandidate) -> float:
        return float(authority.get(candidate.source_id, default_source_authority))

    def remove_active(candidate_id: str) -> None:
        slot = _slot(by_id[candidate_id])
        active_by_slot.setdefault(slot, set()).discard(candidate_id)

    def transition(candidate_id: str, target: MemTxState, reason: str) -> None:
        state[candidate_id] = target
        decisions[candidate_id]["state"] = target.value
        decisions[candidate_id]["resolution"] = reason
        decisions[candidate_id]["transitions"].append(target.value)
        if target not in {MemTxState.COMMITTED, MemTxState.ACTION_SAFE}:
            decisions[candidate_id]["read_visible"] = False

    def cascade_revoke(root_id: str) -> None:
        stack = list(dependents.get(root_id, ()))
        seen: set[str] = set()
        while stack:
            child_id = stack.pop()
            if child_id in seen or child_id not in state:
                continue
            seen.add(child_id)
            if state[child_id] in {
                MemTxState.COMMITTED,
                MemTxState.ACTION_SAFE,
                MemTxState.VALIDATED,
            }:
                remove_active(child_id)
                transition(
                    child_id,
                    MemTxState.REVOKED,
                    f"dependency_invalidated:{root_id}",
                )
            stack.extend(dependents.get(child_id, ()))

    for tx_index, candidate in enumerate(ordered, start=1):
        candidate_id = candidate.candidate_id
        slot = _slot(candidate)
        confidence = (
            candidate.confidence
            if candidate.confidence is not None
            else default_confidence
        )
        score = source_score(candidate)
        snapshot_ids = tuple(
            item.candidate_id
            for item in ordered
            if state.get(item.candidate_id)
            in {MemTxState.COMMITTED, MemTxState.ACTION_SAFE}
        )
        decisions[candidate_id] = {
            "candidate_id": candidate_id,
            "transaction_id": f"return-tx-{tx_index:04d}",
            "snapshot_committed_ids": snapshot_ids,
            "entity": slot[0],
            "attribute": slot[1],
            "source_id": candidate.source_id,
            "source_authority": score,
            "confidence": confidence,
            "state": MemTxState.RAW.value,
            "resolution": "staged",
            "conflict_with_candidate_ids": [],
            "transitions": [
                MemTxState.RAW.value,
                MemTxState.TENTATIVE.value,
            ],
            "implementation_kind": MEMTX_IMPLEMENTATION_KIND,
            "reference_commit": MEMTX_REFERENCE_COMMIT,
        }
        state[candidate_id] = MemTxState.TENTATIVE
        for parent_id in candidate.derived_from_candidate_ids:
            dependents.setdefault(parent_id, set()).add(candidate_id)

        if confidence < minimum_confidence and score < authority_override_threshold:
            transition(
                candidate_id,
                MemTxState.QUARANTINED,
                "abort_insufficient_evidence",
            )
            continue
        if (
            candidate.valid_from is not None
            and candidate.valid_from > final_logical_time
        ) or (
            candidate.valid_to is not None
            and candidate.valid_to < final_logical_time
        ):
            transition(
                candidate_id,
                MemTxState.QUARANTINED,
                "abort_outside_validity_interval",
            )
            continue
        unstable_dependencies = tuple(
            parent_id
            for parent_id in candidate.derived_from_candidate_ids
            if state.get(parent_id)
            not in {MemTxState.COMMITTED, MemTxState.ACTION_SAFE}
        )
        if unstable_dependencies:
            decisions[candidate_id]["unstable_dependency_ids"] = (
                unstable_dependencies
            )
            transition(
                candidate_id,
                MemTxState.QUARANTINED,
                "abort_unstable_dependency",
            )
            continue

        transition(candidate_id, MemTxState.VALIDATED, "validated")
        current_ids = set(active_by_slot.get(slot, set()))
        explicit_target = candidate.supersedes_candidate_id

        if candidate.operation == "delete":
            targets = current_ids
            if explicit_target is not None:
                targets = targets | {explicit_target}
            for target_id in targets:
                if target_id not in state:
                    continue
                remove_active(target_id)
                transition(
                    target_id,
                    MemTxState.REVOKED,
                    f"revoked_by_tombstone:{candidate_id}",
                )
                cascade_revoke(target_id)
            transition(
                candidate_id,
                MemTxState.COMMITTED,
                "commit_tombstone",
            )
            decisions[candidate_id]["read_visible"] = False
            continue

        if explicit_target is not None and explicit_target in current_ids:
            remove_active(explicit_target)
            transition(
                explicit_target,
                MemTxState.SUPERSEDED,
                f"explicitly_superseded_by:{candidate_id}",
            )
            cascade_revoke(explicit_target)
            current_ids.discard(explicit_target)

        if not current_ids:
            transition(candidate_id, MemTxState.COMMITTED, "commit_no_conflict")
            active_by_slot.setdefault(slot, set()).add(candidate_id)
            decisions[candidate_id]["read_visible"] = True
            continue

        same_value_ids = {
            current_id
            for current_id in current_ids
            if _value(by_id[current_id]) == _value(candidate)
        }
        if same_value_ids:
            transition(
                candidate_id,
                MemTxState.SUPERSEDED,
                "idempotent_duplicate_write",
            )
            decisions[candidate_id]["duplicate_of_candidate_ids"] = tuple(
                sorted(same_value_ids)
            )
            decisions[candidate_id]["read_visible"] = False
            continue

        decisions[candidate_id]["conflict_with_candidate_ids"] = tuple(
            sorted(current_ids)
        )
        highest_existing = max(
            source_score(by_id[current_id]) for current_id in current_ids
        )
        same_source = all(
            by_id[current_id].source_id == candidate.source_id
            for current_id in current_ids
        )
        if same_source or score > highest_existing:
            resolution = (
                "supersede_same_source"
                if same_source
                else "supersede_lower_authority"
            )
            for current_id in current_ids:
                remove_active(current_id)
                transition(
                    current_id,
                    MemTxState.SUPERSEDED,
                    f"{resolution}:{candidate_id}",
                )
                cascade_revoke(current_id)
            transition(candidate_id, MemTxState.COMMITTED, resolution)
            active_by_slot.setdefault(slot, set()).add(candidate_id)
            decisions[candidate_id]["read_visible"] = True
        elif score < highest_existing:
            transition(
                candidate_id,
                MemTxState.QUARANTINED,
                "abort_lower_source_authority",
            )
            decisions[candidate_id]["read_visible"] = False
        else:
            # MemTX's ASK_USER branch has no privileged user intervention in
            # the automated benchmark.  Both versions remain quarantined.
            for current_id in current_ids:
                remove_active(current_id)
                transition(
                    current_id,
                    MemTxState.QUARANTINED,
                    f"equal_authority_conflict_with:{candidate_id}",
                )
                cascade_revoke(current_id)
            transition(
                candidate_id,
                MemTxState.QUARANTINED,
                "equal_authority_cross_source_conflict_requires_user",
            )
            decisions[candidate_id]["read_visible"] = False

    visible_states = {
        MemTxState.COMMITTED,
        MemTxState.ACTION_SAFE,
    }
    if isolation is MemTxIsolation.RAW:
        visible_states = set(MemTxState)
    elif isolation is MemTxIsolation.VERIFIED:
        visible_states = {
            MemTxState.VALIDATED,
            MemTxState.COMMITTED,
            MemTxState.ACTION_SAFE,
        }
    elif isolation is MemTxIsolation.ACTION_SAFE:
        visible_states = {MemTxState.ACTION_SAFE}
    # causally-stable and committed both expose committed records here because
    # unstable descendants have already been quarantined or cascade-revoked.
    selected = {
        candidate_id
        for ids in active_by_slot.values()
        for candidate_id in ids
        if state.get(candidate_id) in visible_states
    }
    return ReturnMethodCompilation(
        method=MEMTX_ARM,
        candidate_ids=tuple(item.candidate_id for item in ordered),
        selected_ids=tuple(
            item.candidate_id
            for item in ordered
            if item.candidate_id in selected
        ),
        decisions=tuple(decisions[item.candidate_id] for item in ordered),
    )


def compile_router_memtx(
    checkpoint: RouterPostAbsenceCheckpoint,
    *,
    source_authority: Mapping[str, float] | None = None,
) -> ReturnMethodCompilation:
    compilation = compile_memtx(
        router_return_candidates(checkpoint),
        source_authority=source_authority,
    )
    return replace(compilation, method=MEMTX_ARM)


__all__ = [
    "MEMTX_ARM",
    "MEMTX_IMPLEMENTATION_KIND",
    "MEMTX_METHOD",
    "MEMTX_PAPER_URL",
    "MEMTX_REFERENCE_COMMIT",
    "MEMTX_REFERENCE_URL",
    "MemTxIsolation",
    "MemTxState",
    "compile_memtx",
    "compile_router_memtx",
]
