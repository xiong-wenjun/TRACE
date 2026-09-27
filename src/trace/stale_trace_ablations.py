"""Outcome-blind TRACE component ablations for the STALE Type-II adapter.

The adapter reuses one frozen extraction and one TRACE candidate scan.  Each
ablation below changes exactly one governance decision after that shared scan;
it never reads STALE annotations, queries, answers, or Judge rubrics.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .stale_type2_methods import (
    CHECKPOINT_REPLAY_ARM,
    STALE_TRACE_ABLATION_ARMS,
    SESSION_OBSERVATION_KEY_PREFIX,
    TRACE_WITHOUT_FALLBACK_ARM,
    TRACE_WITHOUT_FRESHNESS_ARM,
    TRACE_WITHOUT_PROVENANCE_ARM,
    VALIDITY_FILTER_ONLY_ARM,
    StaleArmSelection,
    StaleMemoryItem,
)


TRACE_INVALIDATOR_EXPANSION_ARMS = frozenset(
    {
        "trace",
        TRACE_WITHOUT_PROVENANCE_ARM,
        TRACE_WITHOUT_FALLBACK_ARM,
    }
)
TRACE_LIFECYCLE_RESOLUTION_ARMS = TRACE_INVALIDATOR_EXPANSION_ARMS


def _clone_selection(
    selection: StaleArmSelection,
    *,
    arm: str,
) -> StaleArmSelection:
    return StaleArmSelection(
        arm=arm,
        candidate_item_ids=selection.candidate_item_ids,
        selected_item_ids=selection.selected_item_ids,
        decisions=selection.decisions,
        parse_error=selection.parse_error,
    )


def _selection_from_unverified_edges(
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
    departure_after_session: int,
    proposed_edges: Sequence[Mapping[str, object]],
) -> StaleArmSelection:
    """Compile high-recall proposals without provenance-bound verification."""

    observations = tuple(
        item
        for item in items
        if item.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX)
    )
    observation_by_id = {item.memory_id: item for item in observations}
    targets = tuple(
        item
        for item in observations
        if item.agent_id == returning_agent_id
        and item.session_index <= departure_after_session
    )
    target_ids = {item.memory_id for item in targets}
    candidates: dict[str, set[str]] = {
        item.memory_id: set() for item in targets
    }
    for edge in proposed_edges:
        target_id = str(edge.get("target_memory_id") or "").strip()
        invalidator_id = str(edge.get("invalidated_by_memory_id") or "").strip()
        if target_id not in target_ids or invalidator_id not in observation_by_id:
            raise ValueError("unverified TRACE edge references an unknown memory")
        if (
            observation_by_id[invalidator_id].session_index
            <= observation_by_id[target_id].session_index
        ):
            raise ValueError("unverified TRACE edge must point forward in time")
        candidates[target_id].add(invalidator_id)

    observation_decisions: dict[str, Mapping[str, object]] = {}
    for item in observations:
        linked = sorted(
            candidates.get(item.memory_id, ()),
            key=lambda memory_id: (
                observation_by_id[memory_id].session_index,
                memory_id,
            ),
        )
        invalidator_id = linked[-1] if linked else None
        observation_decisions[item.memory_id] = {
            "memory_id": item.memory_id,
            "status": "SUPERSEDED" if invalidator_id is not None else "ADMIT",
            "invalidated_by_memory_id": invalidator_id,
            "provenance_verification_bypassed": invalidator_id is not None,
        }

    observations_by_receipt: dict[str, list[StaleMemoryItem]] = {}
    for item in observations:
        observations_by_receipt.setdefault(item.source_session_sha256, []).append(item)
    decisions: list[Mapping[str, object]] = []
    for item in items:
        direct = observation_decisions.get(item.memory_id)
        if direct is not None:
            decisions.append(direct)
            continue
        inherited = next(
            (
                observation_decisions[source.memory_id]
                for source in observations_by_receipt.get(
                    item.source_session_sha256, ()
                )
                if observation_decisions[source.memory_id]["status"]
                == "SUPERSEDED"
            ),
            None,
        )
        decisions.append(
            {
                "memory_id": item.memory_id,
                "status": "SUPERSEDED" if inherited is not None else "ADMIT",
                "invalidated_by_memory_id": (
                    inherited["invalidated_by_memory_id"]
                    if inherited is not None
                    else None
                ),
                "provenance_verification_bypassed": inherited is not None,
            }
        )
    return StaleArmSelection(
        arm=TRACE_WITHOUT_PROVENANCE_ARM,
        candidate_item_ids=tuple(item.memory_id for item in items),
        selected_item_ids=tuple(
            str(row["memory_id"])
            for row in decisions
            if row["status"] == "ADMIT"
        ),
        decisions=tuple(decisions),
    )


def compile_stale_trace_ablations(
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
    departure_after_session: int,
    base_selections: Mapping[str, StaleArmSelection],
    trace_selection: StaleArmSelection,
    proposed_edges: Sequence[Mapping[str, object]],
) -> tuple[dict[str, StaleArmSelection], Mapping[str, object]]:
    """Derive the four preregistered arms from one frozen candidate pool."""

    all_ids = tuple(item.memory_id for item in items)
    without_freshness = StaleArmSelection(
        arm=TRACE_WITHOUT_FRESHNESS_ARM,
        candidate_item_ids=all_ids,
        selected_item_ids=all_ids,
        decisions=tuple(
            {
                "memory_id": item.memory_id,
                "status": "ADMIT",
                "invalidated_by_memory_id": None,
                "freshness_check_bypassed": True,
            }
            for item in items
        ),
    )
    without_provenance = _selection_from_unverified_edges(
        items,
        returning_agent_id=returning_agent_id,
        departure_after_session=departure_after_session,
        proposed_edges=proposed_edges,
    )
    without_fallback = _clone_selection(
        trace_selection,
        arm=TRACE_WITHOUT_FALLBACK_ARM,
    )
    validity = _clone_selection(
        base_selections[CHECKPOINT_REPLAY_ARM],
        arm=VALIDITY_FILTER_ONLY_ARM,
    )
    selections = {
        TRACE_WITHOUT_FRESHNESS_ARM: without_freshness,
        TRACE_WITHOUT_PROVENANCE_ARM: without_provenance,
        TRACE_WITHOUT_FALLBACK_ARM: without_fallback,
        VALIDITY_FILTER_ONLY_ARM: validity,
    }
    audit = {
        "schema_version": "stale_trace_component_ablation_v1",
        "candidate_pool_shared": True,
        "query_visible": False,
        "oracle_fields_visible": False,
        "proposed_edge_count": len(proposed_edges),
        "interventions": {
            TRACE_WITHOUT_FRESHNESS_ARM: "bypass_supersession_filter",
            TRACE_WITHOUT_PROVENANCE_ARM: (
                "accept_candidate_edges_without_pairwise_grounded_verification"
            ),
            TRACE_WITHOUT_FALLBACK_ARM: "disable_incomplete_view_fallback",
            VALIDITY_FILTER_ONLY_ARM: (
                "exact_key_add_update_delete_filter_without_cross_key_dependencies"
            ),
        },
        "fallback_activated_in_complete_episode": False,
        "fallback_note": (
            "Official STALE episodes have complete session coverage; this arm is "
            "expected to match TRACE unless an incomplete safe view is encountered."
        ),
    }
    return selections, audit


__all__ = [
    "STALE_TRACE_ABLATION_ARMS",
    "TRACE_INVALIDATOR_EXPANSION_ARMS",
    "TRACE_LIFECYCLE_RESOLUTION_ARMS",
    "TRACE_WITHOUT_FALLBACK_ARM",
    "TRACE_WITHOUT_FRESHNESS_ARM",
    "TRACE_WITHOUT_PROVENANCE_ARM",
    "VALIDITY_FILTER_ONLY_ARM",
    "compile_stale_trace_ablations",
]
