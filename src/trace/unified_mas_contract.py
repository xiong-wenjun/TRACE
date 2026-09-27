"""Shared five-task-agent execution contract for MAS RETURN benchmarks.

Benchmark adapters retain their native departure boundary and scoring rules.
This module freezes the topology and lifecycle invariants that must not vary
between Memora, STALE, and ManBench.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .router_return_protocol import canonical_sha256


UNIFIED_MAS_EXECUTION_CONTRACT = "five_task_agent_single_return_v1"
PARAMETERIZED_MAS_EXECUTION_CONTRACT = "parameterized_single_return_v1"
RETURNING_AGENT_ID = "agent1"
ACTIVE_AGENT_IDS = ("agent2", "agent3", "agent4", "agent5")
TASK_AGENT_IDS = (RETURNING_AGENT_ID, *ACTIVE_AGENT_IDS)
TASK_AGENT_COUNT = len(TASK_AGENT_IDS)
DEPARTING_AGENT_COUNT = 1
ACTIVE_AGENT_COUNT = len(ACTIVE_AGENT_IDS)
LIFECYCLE_SCHEDULER_ID = "lifecycle_scheduler"
JUDGE_COMPONENT_ID = "judge"
INTRA_EPISODE_EXECUTION_MODE = "deterministic_serial"


def active_agent_ids_for_count(active_agent_count: int) -> tuple[str, ...]:
    """Return deterministic active-agent IDs for one returning agent.

    ``N`` in the ManBench provenance-stress protocol is the number of agents
    that remain active during the absence.  The returning agent is separate,
    so the task roster contains ``N + D`` agents with ``D=1``.
    """

    count = int(active_agent_count)
    if count < 1:
        raise ValueError("active_agent_count must be positive")
    return tuple(f"agent{index}" for index in range(2, count + 2))


def task_agent_ids_for_active_count(
    active_agent_count: int,
    *,
    returning_agent_id: str = RETURNING_AGENT_ID,
) -> tuple[str, ...]:
    returning = str(returning_agent_id).strip()
    if returning != RETURNING_AGENT_ID:
        raise ValueError("the unified protocol reserves agent1 as the returner")
    return (returning, *active_agent_ids_for_count(active_agent_count))


def _normalize_agent_ids(
    active_agent_ids: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(str(item).strip() for item in active_agent_ids)
    if not normalized or any(not item for item in normalized):
        raise ValueError("active_agent_ids must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("active_agent_ids must be unique")
    if RETURNING_AGENT_ID in normalized:
        raise ValueError("active_agent_ids cannot contain agent1")
    expected = active_agent_ids_for_count(len(normalized))
    if normalized != expected:
        raise ValueError("active_agent_ids must follow agent2..agentN order")
    return normalized


def validate_unified_mas_config(value: Mapping[str, object]) -> None:
    """Reject configuration drift from the paper-facing task-agent roster."""

    expected: dict[str, object] = {
        "execution_contract": UNIFIED_MAS_EXECUTION_CONTRACT,
        "task_agent_count": TASK_AGENT_COUNT,
        "returning_agent_id": RETURNING_AGENT_ID,
        "active_agent_ids": list(ACTIVE_AGENT_IDS),
        "intra_episode_execution_mode": INTRA_EPISODE_EXECUTION_MODE,
    }
    if dict(value) != expected:
        raise ValueError("config drifted from unified MAS contract")


def deterministic_session_assignment(
    ordered_session_ids: Sequence[int],
    *,
    departure_after_session: int,
    active_agent_ids: Sequence[str] = ACTIVE_AGENT_IDS,
) -> dict[str, tuple[int, ...]]:
    """Assign one chronological prefix to agent1 and the suffix round-robin."""

    session_ids = tuple(int(item) for item in ordered_session_ids)
    if not session_ids:
        raise ValueError("session assignment requires at least one session")
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("session ids must be unique")
    if tuple(sorted(session_ids)) != session_ids:
        raise ValueError("session ids must be chronologically ordered")
    if departure_after_session not in session_ids:
        raise ValueError("departure marker must identify an observed session")
    cut = session_ids.index(departure_after_session)
    active_ids = _normalize_agent_ids(active_agent_ids)
    task_ids = (RETURNING_AGENT_ID, *active_ids)
    assignments: dict[str, list[int]] = {agent_id: [] for agent_id in task_ids}
    assignments[RETURNING_AGENT_ID].extend(session_ids[: cut + 1])
    for offset, session_id in enumerate(session_ids[cut + 1 :]):
        assignments[active_ids[offset % len(active_ids)]].append(session_id)
    return {
        agent_id: tuple(assignments[agent_id]) for agent_id in task_ids
    }


def validate_session_assignment(
    assignments: Mapping[str, Sequence[int]],
    *,
    ordered_session_ids: Sequence[int],
    departure_after_session: int,
    active_agent_ids: Sequence[str] = ACTIVE_AGENT_IDS,
) -> None:
    """Reject topology drift, duplicate ownership, or a non-prefix returner."""

    task_ids = (RETURNING_AGENT_ID, *_normalize_agent_ids(active_agent_ids))
    if set(assignments) != set(task_ids):
        raise ValueError("session assignment roster does not match active agents")
    expected = deterministic_session_assignment(
        ordered_session_ids,
        departure_after_session=departure_after_session,
        active_agent_ids=active_agent_ids,
    )
    normalized = {
        agent_id: tuple(int(item) for item in assignments[agent_id])
        for agent_id in task_ids
    }
    if normalized != expected:
        raise ValueError(
            "session assignment violates prefix/round-robin lifecycle policy"
        )


def unified_mas_architecture_record(
    *,
    active_agent_ids: Sequence[str] = ACTIVE_AGENT_IDS,
) -> dict[str, object]:
    """Return the benchmark-neutral topology without counting system services."""

    active_ids = _normalize_agent_ids(active_agent_ids)
    task_ids = (RETURNING_AGENT_ID, *active_ids)
    parameterized = active_ids != ACTIVE_AGENT_IDS
    return {
        "schema_version": "unified_mas_architecture_v1",
        "execution_contract": (
            PARAMETERIZED_MAS_EXECUTION_CONTRACT
            if parameterized
            else UNIFIED_MAS_EXECUTION_CONTRACT
        ),
        "task_agent_count": len(task_ids),
        "departing_agent_count": DEPARTING_AGENT_COUNT,
        "active_agent_count_during_absence": len(active_ids),
        "task_agent_ids": list(task_ids),
        "returning_agent_id": RETURNING_AGENT_ID,
        "active_agent_ids": list(active_ids),
        "final_answer_agent_id": RETURNING_AGENT_ID,
        "aggregator_agent_enabled": False,
        "session_assignment_policy": (
            "chronological_prefix_to_agent1_then_deterministic_round_robin"
        ),
        "absence_execution": {
            "logical_independence": True,
            "active_agents_share_private_context": False,
            "intra_episode_execution_mode": INTRA_EPISODE_EXECUTION_MODE,
            "physical_parallel_model_calls": False,
            "synchronization_barrier_before_checkpoint": True,
            "episode_level_parallelism_configurable": True,
        },
        "lifecycle_scheduler": {
            "component_id": LIFECYCLE_SCHEDULER_ID,
            "counted_as_agent": False,
            "semantic_inference": False,
            "model_calls": 0,
            "responsibilities": [
                "assign_sessions",
                "record_departure_and_return",
                "freeze_post_absence_checkpoint",
            ],
        },
        "judge": {
            "component_id": JUDGE_COMPONENT_ID,
            "counted_as_agent": False,
            "evaluator_only": True,
            "may_access_hidden_rubric": True,
            "may_write_actor_memory": False,
        },
    }


def unified_mas_episode_record(
    *,
    benchmark: str,
    episode_id: str,
    assignments: Mapping[str, Sequence[int]],
    ordered_session_ids: Sequence[int],
    departure_after_session: int,
    return_after_session: int,
    checkpoint_sha256: str | None = None,
    active_agent_ids: Sequence[str] = ACTIVE_AGENT_IDS,
) -> dict[str, object]:
    """Bind one benchmark episode to the shared topology and lifecycle."""

    validate_session_assignment(
        assignments,
        ordered_session_ids=ordered_session_ids,
        departure_after_session=departure_after_session,
        active_agent_ids=active_agent_ids,
    )
    session_ids = tuple(int(item) for item in ordered_session_ids)
    if return_after_session != session_ids[-1]:
        raise ValueError("return must follow the last benchmark session")
    active_ids = _normalize_agent_ids(active_agent_ids)
    task_ids = (RETURNING_AGENT_ID, *active_ids)
    record: dict[str, object] = {
        "schema_version": "unified_mas_return_episode_v1",
        "benchmark": str(benchmark),
        "episode_id": str(episode_id),
        "architecture": unified_mas_architecture_record(
            active_agent_ids=active_ids
        ),
        "departure_after_session": int(departure_after_session),
        "return_after_session": int(return_after_session),
        "return_before_final_task": True,
        "agent_session_ids": {
            agent_id: [int(item) for item in assignments[agent_id]]
            for agent_id in task_ids
        },
        "all_sessions_assigned_exactly_once": True,
        "single_frozen_checkpoint_all_methods": True,
        "checkpoint_sha256": checkpoint_sha256,
        "judge_context_isolated": True,
    }
    record["receipt_sha256"] = canonical_sha256(record)
    return record


__all__ = [
    "ACTIVE_AGENT_COUNT",
    "ACTIVE_AGENT_IDS",
    "DEPARTING_AGENT_COUNT",
    "JUDGE_COMPONENT_ID",
    "INTRA_EPISODE_EXECUTION_MODE",
    "PARAMETERIZED_MAS_EXECUTION_CONTRACT",
    "LIFECYCLE_SCHEDULER_ID",
    "RETURNING_AGENT_ID",
    "TASK_AGENT_COUNT",
    "TASK_AGENT_IDS",
    "UNIFIED_MAS_EXECUTION_CONTRACT",
    "active_agent_ids_for_count",
    "task_agent_ids_for_active_count",
    "deterministic_session_assignment",
    "unified_mas_architecture_record",
    "unified_mas_episode_record",
    "validate_unified_mas_config",
    "validate_session_assignment",
]
