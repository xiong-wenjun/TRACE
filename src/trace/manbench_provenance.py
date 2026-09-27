"""Controlled provenance stress protocol for ManBench-Return.

The stress layer changes only the evidence topology around the same frozen
two-candidate Return episode.  It never changes the ManBench question, gold
answer, lifecycle assignment, or actor-visible prompt.  ``N`` denotes the
number of active agents during the one-agent absence; the returning agent is
``agent1`` and ``D=1`` is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

from .router_return_protocol import canonical_sha256
from .unified_mas_contract import (
    active_agent_ids_for_count,
    task_agent_ids_for_active_count,
)


PROVENANCE_STRESS_SCENARIOS = (
    "independent_majority",
    "replicated_majority",
    "missing_receipt",
    "conflicting_receipt",
)
PROVENANCE_STRESS_AGENT_COUNTS = (4, 8, 16, 32)
PROVENANCE_STRESS_DEPARTING_AGENT_COUNT = 1


def _digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProvenanceStressSpec:
    """One cell in the provenance × active-agent-size experiment matrix."""

    scenario: str
    active_agent_count: int
    departing_agent_count: int = PROVENANCE_STRESS_DEPARTING_AGENT_COUNT

    def __post_init__(self) -> None:
        if self.scenario not in PROVENANCE_STRESS_SCENARIOS:
            raise ValueError(f"unknown provenance stress scenario: {self.scenario}")
        if self.active_agent_count not in PROVENANCE_STRESS_AGENT_COUNTS:
            raise ValueError(
                "active_agent_count must be one of "
                + ", ".join(str(item) for item in PROVENANCE_STRESS_AGENT_COUNTS)
            )
        if self.departing_agent_count != 1:
            raise ValueError("provenance stress fixes departing_agent_count=1")

    @property
    def active_agent_ids(self) -> tuple[str, ...]:
        return active_agent_ids_for_count(self.active_agent_count)

    @property
    def task_agent_ids(self) -> tuple[str, ...]:
        return task_agent_ids_for_active_count(self.active_agent_count)

    @property
    def case_id(self) -> str:
        return f"{self.scenario}:n{self.active_agent_count}:d1"

    def episode_id(self, base_episode_id: str) -> str:
        return f"{base_episode_id}|{self.case_id}"

    def record(self) -> dict[str, object]:
        return {
            "protocol": "manbench_provenance_stress",
            "case_id": self.case_id,
            "scenario": self.scenario,
            "n_definition": "active_agents_during_absence",
            "active_agent_count": self.active_agent_count,
            "departing_agent_count": self.departing_agent_count,
            "active_agent_ids": list(self.active_agent_ids),
            "task_agent_ids": list(self.task_agent_ids),
            "actor_visible": False,
        }


def stress_specs(
    scenarios: Sequence[str] = PROVENANCE_STRESS_SCENARIOS,
    active_agent_counts: Sequence[int] = PROVENANCE_STRESS_AGENT_COUNTS,
) -> tuple[ProvenanceStressSpec, ...]:
    """Validate and deterministically enumerate the requested matrix."""

    result = tuple(
        ProvenanceStressSpec(
            scenario=str(scenario),
            active_agent_count=int(active_agent_count),
        )
        for scenario in scenarios
        for active_agent_count in active_agent_counts
    )
    if not result:
        raise ValueError("provenance stress matrix cannot be empty")
    if len({item.case_id for item in result}) != len(result):
        raise ValueError("provenance stress cells must be unique")
    return result


def _receipt(
    *,
    episode_id: str,
    candidate_id: str,
    source_id: str,
    replica_index: int,
    payload: str,
) -> str:
    return canonical_sha256(
        {
            "protocol": "manbench_provenance_stress",
            "episode_id": episode_id,
            "candidate_id": candidate_id,
            "source_id": source_id,
            "replica_index": replica_index,
            "payload_sha256": _digest(payload),
        }
    )


def build_challenger_provenance(
    spec: ProvenanceStressSpec,
    *,
    episode_id: str,
    candidate_id: str = "answer-challenger",
    payload: str,
) -> dict[str, object]:
    """Create hidden, auditable provenance metadata for one challenger.

    The four cases differ only in evidence quality.  They deliberately keep
    the same active-agent vote and candidate text, so a provenance-aware
    method is tested against a matched candidate pool rather than an easier
    question.
    """

    active_ids = spec.active_agent_ids
    if spec.scenario == "independent_majority":
        source_ids = tuple(f"source:{agent_id}" for agent_id in active_ids)
        receipts = tuple(
            _receipt(
                episode_id=episode_id,
                candidate_id=candidate_id,
                source_id=source_id,
                replica_index=index,
                payload=payload,
            )
            for index, source_id in enumerate(source_ids)
        )
        status = "valid"
    elif spec.scenario == "replicated_majority":
        source_ids = tuple("source:agent2" for _ in active_ids)
        receipt = _receipt(
            episode_id=episode_id,
            candidate_id=candidate_id,
            source_id="source:agent2",
            replica_index=0,
            payload=payload,
        )
        receipts = tuple(receipt for _ in active_ids)
        status = "replicated"
    elif spec.scenario == "missing_receipt":
        source_ids = tuple(f"source:{agent_id}" for agent_id in active_ids)
        receipts = ()
        status = "missing"
    else:
        source_ids = tuple(f"source:{agent_id}" for agent_id in active_ids)
        first = _receipt(
            episode_id=episode_id,
            candidate_id=candidate_id,
            source_id=source_ids[0],
            replica_index=0,
            payload=payload,
        )
        # The second digest is well-formed but binds a different payload,
        # modelling a receipt conflict rather than malformed transport.
        second = _receipt(
            episode_id=episode_id,
            candidate_id=candidate_id,
            source_id=source_ids[-1],
            replica_index=1,
            payload=payload + " [conflicting binding]",
        )
        receipts = (first, second)
        status = "conflicting"
    return {
        "candidate_id": candidate_id,
        "status": status,
        "source_ids": list(source_ids),
        "receipt_sha256s": list(receipts),
        "support_count": len(active_ids),
        "independent_source_count": len(set(source_ids)),
        "actor_visible": False,
    }


def validate_stress_metadata(
    value: Mapping[str, object],
    *,
    expected_spec: ProvenanceStressSpec | None = None,
) -> None:
    """Fail closed on incomplete provenance stress records."""

    scenario = str(value.get("scenario") or "")
    count = int(value.get("active_agent_count") or 0)
    spec = ProvenanceStressSpec(
        scenario=scenario,
        active_agent_count=count,
        departing_agent_count=int(value.get("departing_agent_count") or 0),
    )
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("provenance stress metadata does not match requested cell")
    if value.get("actor_visible") is not False:
        raise ValueError("provenance stress labels must be hidden from actors")
    if list(value.get("active_agent_ids") or ()) != list(spec.active_agent_ids):
        raise ValueError("provenance stress active-agent roster mismatch")
    if list(value.get("task_agent_ids") or ()) != list(spec.task_agent_ids):
        raise ValueError("provenance stress task-agent roster mismatch")


__all__ = [
    "PROVENANCE_STRESS_AGENT_COUNTS",
    "PROVENANCE_STRESS_DEPARTING_AGENT_COUNT",
    "PROVENANCE_STRESS_SCENARIOS",
    "ProvenanceStressSpec",
    "build_challenger_provenance",
    "stress_specs",
    "validate_stress_metadata",
]
