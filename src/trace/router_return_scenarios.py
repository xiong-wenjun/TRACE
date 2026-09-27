"""Benchmark-neutral builders for real prework, absence work, and RETURN."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Mapping, Sequence

from .router_return_protocol import (
    RouterObligation,
    RouterReturnProtocol,
    RouterTaskDAG,
    RouterTaskNode,
    RouterWorkstateItem,
    RouterWorkstateStatus,
    canonical_sha256,
)


@dataclass(frozen=True)
class RouterScenarioIds:
    prefix_node_id: str
    absence_node_id: str
    synthesis_node_id: str
    obligation_id: str
    prefix_dependency_id: str
    absence_dependency_id: str
    old_item_id: str

    def record(self) -> dict[str, str]:
        return {
            "prefix_node_id": self.prefix_node_id,
            "absence_node_id": self.absence_node_id,
            "synthesis_node_id": self.synthesis_node_id,
            "obligation_id": self.obligation_id,
            "prefix_dependency_id": self.prefix_dependency_id,
            "absence_dependency_id": self.absence_dependency_id,
            "old_item_id": self.old_item_id,
        }


@dataclass(frozen=True)
class MultiWorkerRouterScenarioIds:
    """Stable identifiers for one returning and four absence workers."""

    prefix_node_id: str
    synthesis_node_id: str
    obligation_id: str
    prefix_dependency_id: str
    old_item_id: str
    returning_principal_id: str
    absence_principal_ids: tuple[str, ...]
    absence_node_ids: tuple[str, ...]
    absence_dependency_ids: tuple[str, ...]
    aggregator_principal_id: str | None = None
    evaluator_principal_id: str | None = None
    aggregation_node_id: str | None = None
    evaluation_node_id: str | None = None
    predeparture_dependency_ids: tuple[str, ...] = ()
    critical_absence_principal_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.absence_principal_ids) != 4:
            raise ValueError("the five-worker protocol requires four absence workers")
        if not (
            len(self.absence_principal_ids)
            == len(self.absence_node_ids)
            == len(self.absence_dependency_ids)
        ):
            raise ValueError("absence worker identifiers are misaligned")
        if len(set(self.absence_principal_ids)) != 4:
            raise ValueError("absence workers must be distinct")
        if self.predeparture_dependency_ids:
            if self.prefix_dependency_id not in set(
                self.predeparture_dependency_ids
            ):
                raise ValueError(
                    "legacy prefix dependency must be part of the "
                    "predeparture dependency set"
                )
        if set(self.critical_absence_principal_ids) - set(
            self.absence_principal_ids
        ):
            raise ValueError(
                "critical absence workers must be absence principals"
            )

    @property
    def effective_predeparture_dependency_ids(self) -> tuple[str, ...]:
        return self.predeparture_dependency_ids or (
            self.prefix_dependency_id,
        )

    @property
    def effective_critical_absence_principal_ids(self) -> tuple[str, ...]:
        return self.critical_absence_principal_ids or (
            self.absence_principal_ids
        )

    @property
    def critical_absence_dependency_ids(self) -> tuple[str, ...]:
        return tuple(
            self.for_worker(principal_id)[1]
            for principal_id in self.effective_critical_absence_principal_ids
        )

    def for_worker(self, principal_id: str) -> tuple[str, str]:
        try:
            index = self.absence_principal_ids.index(principal_id)
        except ValueError as error:
            raise KeyError(f"unknown absence worker: {principal_id}") from error
        return (
            self.absence_node_ids[index],
            self.absence_dependency_ids[index],
        )

    def record(self) -> dict[str, object]:
        return {
            "prefix_node_id": self.prefix_node_id,
            "synthesis_node_id": self.synthesis_node_id,
            "obligation_id": self.obligation_id,
            "prefix_dependency_id": self.prefix_dependency_id,
            "predeparture_dependency_ids": list(
                self.effective_predeparture_dependency_ids
            ),
            "old_item_id": self.old_item_id,
            "returning_principal_id": self.returning_principal_id,
            "absence_workers": [
                {
                    "principal_id": principal_id,
                    "node_id": self.absence_node_ids[index],
                    "dependency_id": self.absence_dependency_ids[index],
                }
                for index, principal_id in enumerate(
                    self.absence_principal_ids
                )
            ],
            "critical_absence_principal_ids": list(
                self.effective_critical_absence_principal_ids
            ),
            "critical_absence_dependency_ids": list(
                self.critical_absence_dependency_ids
            ),
            "aggregator_principal_id": self.aggregator_principal_id,
            "evaluator_principal_id": self.evaluator_principal_id,
            "aggregation_node_id": self.aggregation_node_id,
            "evaluation_node_id": self.evaluation_node_id,
        }


@dataclass(frozen=True)
class HorizonRouterScenarioIds:
    """Stable identifiers for a variable-length sequential absence."""

    prefix_node_id: str
    synthesis_node_id: str
    obligation_id: str
    prefix_dependency_id: str
    old_item_id: str
    returning_principal_id: str
    absence_principal_ids: tuple[str, ...]
    absence_node_ids: tuple[str, ...]
    absence_dependency_ids: tuple[str, ...]
    critical_absence_task_indices: tuple[int, ...]
    aggregator_principal_id: str | None = None
    evaluator_principal_id: str | None = None
    aggregation_node_id: str | None = None
    evaluation_node_id: str | None = None
    predeparture_dependency_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        task_count = len(self.absence_principal_ids)
        if task_count < 1:
            raise ValueError("absence horizon must contain at least one task")
        if not (
            task_count
            == len(self.absence_node_ids)
            == len(self.absence_dependency_ids)
        ):
            raise ValueError("absence horizon identifiers are misaligned")
        if any(
            index < 0 or index >= task_count
            for index in self.critical_absence_task_indices
        ):
            raise ValueError("critical absence task index is out of range")
        if self.predeparture_dependency_ids:
            if self.prefix_dependency_id not in set(
                self.predeparture_dependency_ids
            ):
                raise ValueError(
                    "legacy prefix dependency must be part of the "
                    "predeparture dependency set"
                )

    @property
    def effective_predeparture_dependency_ids(self) -> tuple[str, ...]:
        """Return the calibrated dependency set or the legacy singleton."""

        return self.predeparture_dependency_ids or (
            self.prefix_dependency_id,
        )

    def for_task(self, task_index: int) -> tuple[str, str, str]:
        if task_index < 0 or task_index >= len(self.absence_node_ids):
            raise IndexError("absence task index is out of range")
        return (
            self.absence_principal_ids[task_index],
            self.absence_node_ids[task_index],
            self.absence_dependency_ids[task_index],
        )

    def record(self) -> dict[str, object]:
        return {
            "prefix_node_id": self.prefix_node_id,
            "synthesis_node_id": self.synthesis_node_id,
            "obligation_id": self.obligation_id,
            "prefix_dependency_id": self.prefix_dependency_id,
            "predeparture_dependency_ids": list(
                self.effective_predeparture_dependency_ids
            ),
            "old_item_id": self.old_item_id,
            "returning_principal_id": self.returning_principal_id,
            "absence_horizon_tasks": len(self.absence_node_ids),
            "critical_absence_task_indices": list(
                self.critical_absence_task_indices
            ),
            "absence_tasks": [
                {
                    "task_index": index,
                    "principal_id": principal_id,
                    "node_id": self.absence_node_ids[index],
                    "dependency_id": self.absence_dependency_ids[index],
                    "critical": (
                        index in self.critical_absence_task_indices
                    ),
                }
                for index, principal_id in enumerate(
                    self.absence_principal_ids
                )
            ],
            "aggregator_principal_id": self.aggregator_principal_id,
            "evaluator_principal_id": self.evaluator_principal_id,
            "aggregation_node_id": self.aggregation_node_id,
            "evaluation_node_id": self.evaluation_node_id,
        }


@dataclass(frozen=True)
class AbsenceReview:
    relation: str
    current_fact: str
    audit_note: str

    def __post_init__(self) -> None:
        if self.relation not in {"retain", "supersede", "revoke", "extend"}:
            raise ValueError("unsupported absence relation")
        if not self.current_fact.strip():
            raise ValueError("absence review requires a current fact")

    def record(self) -> dict[str, str]:
        return {
            "relation": self.relation,
            "current_fact": self.current_fact,
            "audit_note": self.audit_note,
        }


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}:{canonical_sha256(list(parts))[:20]}"


def build_router_return_scenario(
    *,
    task_id: str,
    task_description: str,
    members: Sequence[tuple[str, str]],
    returning_principal_id: str,
    absence_principal_id: str,
    prefix_description: str,
    absence_description: str,
    synthesis_description: str,
) -> tuple[RouterReturnProtocol, RouterScenarioIds]:
    """Build the shared three-phase DAG used by all benchmark adapters."""

    if returning_principal_id == absence_principal_id:
        raise ValueError("absence work must be performed by another agent")
    member_ids = {principal_id for principal_id, _ in members}
    if {returning_principal_id, absence_principal_id} - member_ids:
        raise ValueError("scenario principals must be registered members")
    protocol = RouterReturnProtocol(
        task_id=task_id, task_description=task_description
    )
    for principal_id, role_id in members:
        protocol.add_member(principal_id, role_id)
    ids = RouterScenarioIds(
        prefix_node_id=_stable_id("node-prefix", task_id),
        absence_node_id=_stable_id("node-absence", task_id),
        synthesis_node_id=_stable_id("node-synthesis", task_id),
        obligation_id=_stable_id(
            "obligation", task_id, returning_principal_id
        ),
        prefix_dependency_id=_stable_id(
            "dep-prefix", task_id, returning_principal_id
        ),
        absence_dependency_id=_stable_id(
            "dep-absence", task_id, returning_principal_id
        ),
        old_item_id=_stable_id(
            "workstate-old", task_id, returning_principal_id
        ),
    )
    obligation = RouterObligation(
        obligation_id=ids.obligation_id,
        description=(
            "Complete the current task using both validated pre-absence work "
            "and relevant work produced while the role was absent."
        ),
        owner_principal_id=returning_principal_id,
        required_dependency_ids=(
            ids.prefix_dependency_id,
            ids.absence_dependency_id,
        ),
        critical_dependency_ids=(
            ids.prefix_dependency_id,
            ids.absence_dependency_id,
        ),
    )
    dag = RouterTaskDAG(
        task_id=task_id,
        task_description=task_description,
        nodes=(
            RouterTaskNode(
                node_id=ids.prefix_node_id,
                description=prefix_description,
                assigned_principal_id=returning_principal_id,
                obligation_ids=(ids.obligation_id,),
            ),
            RouterTaskNode(
                node_id=ids.absence_node_id,
                description=absence_description,
                assigned_principal_id=absence_principal_id,
                depends_on=(ids.prefix_node_id,),
                obligation_ids=(ids.obligation_id,),
            ),
            RouterTaskNode(
                node_id=ids.synthesis_node_id,
                description=synthesis_description,
                assigned_principal_id=returning_principal_id,
                depends_on=(ids.absence_node_id,),
                obligation_ids=(ids.obligation_id,),
            ),
        ),
    )
    protocol.install_plan(dag, (obligation,))
    return protocol, ids


def build_five_worker_router_return_scenario(
    *,
    task_id: str,
    task_description: str,
    workers: Sequence[tuple[str, str]],
    returning_principal_id: str,
    prefix_description: str,
    absence_descriptions: Mapping[str, str],
    synthesis_description: str,
    predeparture_critical_dependency_count: int = 1,
    critical_absence_principal_ids: Sequence[str] | None = None,
    aggregator_principal_id: str | None = None,
    aggregator_role_id: str = "role:aggregator",
    aggregation_description: str = "Aggregate the final answer.",
    evaluator_principal_id: str | None = None,
    evaluator_role_id: str = "role:evaluator",
    evaluation_description: str = "Evaluate the final answer.",
) -> tuple[RouterReturnProtocol, MultiWorkerRouterScenarioIds]:
    """Build a Router DAG with one returning and four concurrent workers."""

    if len(workers) != 5:
        raise ValueError("the multi-worker protocol requires exactly five workers")
    worker_ids = tuple(principal_id for principal_id, _ in workers)
    if len(set(worker_ids)) != 5:
        raise ValueError("five-worker principals must be unique")
    if returning_principal_id not in worker_ids:
        raise ValueError("returning principal is not one of the five workers")
    absence_principal_ids = tuple(
        principal_id
        for principal_id in worker_ids
        if principal_id != returning_principal_id
    )
    if set(absence_descriptions) != set(absence_principal_ids):
        raise ValueError(
            "absence descriptions must cover all four non-returning workers"
        )
    if predeparture_critical_dependency_count < 1:
        raise ValueError(
            "predeparture_critical_dependency_count must be positive"
        )
    critical_absence_ids = tuple(
        dict.fromkeys(
            str(principal_id)
            for principal_id in (
                critical_absence_principal_ids
                if critical_absence_principal_ids is not None
                else absence_principal_ids
            )
        )
    )
    if not critical_absence_ids:
        raise ValueError("at least one absence worker must be critical")
    if set(critical_absence_ids) - set(absence_principal_ids):
        raise ValueError(
            "critical absence workers must be non-returning workers"
        )
    if (aggregator_principal_id is None) != (evaluator_principal_id is None):
        raise ValueError(
            "aggregator and evaluator must either both be present or absent"
        )
    extra_ids = tuple(
        principal_id
        for principal_id in (
            aggregator_principal_id,
            evaluator_principal_id,
        )
        if principal_id is not None
    )
    if set(extra_ids).intersection(worker_ids) or len(set(extra_ids)) != len(
        extra_ids
    ):
        raise ValueError("aggregator/evaluator identities must be distinct")
    protocol = RouterReturnProtocol(
        task_id=task_id,
        task_description=task_description,
    )
    for principal_id, role_id in workers:
        protocol.add_member(principal_id, role_id)
    legacy_prefix_dependency_id = _stable_id(
        "dep-prefix", task_id, returning_principal_id
    )
    predeparture_dependency_ids = (
        legacy_prefix_dependency_id,
        *(
            _stable_id(
                "dep-prefix",
                task_id,
                returning_principal_id,
                str(index),
            )
            for index in range(
                1, predeparture_critical_dependency_count
            )
        ),
    )
    ids = MultiWorkerRouterScenarioIds(
        prefix_node_id=_stable_id("node-prefix", task_id),
        synthesis_node_id=_stable_id("node-synthesis", task_id),
        obligation_id=_stable_id(
            "obligation", task_id, returning_principal_id
        ),
        prefix_dependency_id=legacy_prefix_dependency_id,
        old_item_id=_stable_id(
            "workstate-old", task_id, returning_principal_id
        ),
        returning_principal_id=returning_principal_id,
        absence_principal_ids=absence_principal_ids,
        absence_node_ids=tuple(
            _stable_id("node-absence", task_id, principal_id)
            for principal_id in absence_principal_ids
        ),
        absence_dependency_ids=tuple(
            _stable_id("dep-absence", task_id, principal_id)
            for principal_id in absence_principal_ids
        ),
        aggregator_principal_id=aggregator_principal_id,
        evaluator_principal_id=evaluator_principal_id,
        aggregation_node_id=(
            _stable_id("node-aggregate", task_id)
            if aggregator_principal_id is not None
            else None
        ),
        evaluation_node_id=(
            _stable_id("node-evaluate", task_id)
            if evaluator_principal_id is not None
            else None
        ),
        predeparture_dependency_ids=predeparture_dependency_ids,
        critical_absence_principal_ids=critical_absence_ids,
    )
    if aggregator_principal_id is not None:
        protocol.add_member(
            aggregator_principal_id, aggregator_role_id
        )
        protocol.add_member(evaluator_principal_id, evaluator_role_id)  # type: ignore[arg-type]
    required = (
        ids.effective_predeparture_dependency_ids
        + ids.critical_absence_dependency_ids
    )
    obligation = RouterObligation(
        obligation_id=ids.obligation_id,
        description=(
            "Complete the current task using validated pre-absence work and "
            "all obligation-relevant work produced by the four workers while "
            "the returning role was absent."
        ),
        owner_principal_id=returning_principal_id,
        required_dependency_ids=required,
        critical_dependency_ids=required,
    )
    absence_nodes = tuple(
        RouterTaskNode(
            node_id=ids.absence_node_ids[index],
            description=absence_descriptions[principal_id],
            assigned_principal_id=principal_id,
            depends_on=(ids.prefix_node_id,),
            obligation_ids=(ids.obligation_id,),
        )
        for index, principal_id in enumerate(absence_principal_ids)
    )
    terminal_nodes: tuple[RouterTaskNode, ...] = ()
    if aggregator_principal_id is not None:
        if ids.aggregation_node_id is None or ids.evaluation_node_id is None:
            raise RuntimeError("terminal node identifiers are unavailable")
        terminal_nodes = (
            RouterTaskNode(
                node_id=ids.aggregation_node_id,
                description=aggregation_description,
                assigned_principal_id=aggregator_principal_id,
                depends_on=(ids.synthesis_node_id,),
                obligation_ids=(ids.obligation_id,),
            ),
            RouterTaskNode(
                node_id=ids.evaluation_node_id,
                description=evaluation_description,
                assigned_principal_id=evaluator_principal_id,  # type: ignore[arg-type]
                depends_on=(ids.aggregation_node_id,),
                obligation_ids=(ids.obligation_id,),
            ),
        )
    dag = RouterTaskDAG(
        task_id=task_id,
        task_description=task_description,
        nodes=(
            RouterTaskNode(
                node_id=ids.prefix_node_id,
                description=prefix_description,
                assigned_principal_id=returning_principal_id,
                obligation_ids=(ids.obligation_id,),
            ),
            *absence_nodes,
            RouterTaskNode(
                node_id=ids.synthesis_node_id,
                description=synthesis_description,
                assigned_principal_id=returning_principal_id,
                depends_on=ids.absence_node_ids,
                obligation_ids=(ids.obligation_id,),
            ),
            *terminal_nodes,
        ),
    )
    protocol.install_plan(dag, (obligation,))
    return protocol, ids


def build_horizon_router_return_scenario(
    *,
    task_id: str,
    task_description: str,
    workers: Sequence[tuple[str, str]],
    returning_principal_id: str,
    prefix_description: str,
    absence_tasks: Sequence[tuple[str, str]],
    synthesis_description: str,
    critical_absence_task_indices: Sequence[int],
    predeparture_critical_dependency_count: int = 1,
    aggregator_principal_id: str | None = None,
    aggregator_role_id: str = "role:aggregator",
    aggregation_description: str = "Aggregate the final answer.",
    evaluator_principal_id: str | None = None,
    evaluator_role_id: str = "role:evaluator",
    evaluation_description: str = "Evaluate the final answer.",
) -> tuple[RouterReturnProtocol, HorizonRouterScenarioIds]:
    """Build a five-worker DAG with a sequential H-task absence window."""

    if len(workers) != 5:
        raise ValueError("the horizon protocol requires exactly five workers")
    worker_ids = tuple(principal_id for principal_id, _ in workers)
    if len(set(worker_ids)) != 5:
        raise ValueError("five-worker principals must be unique")
    if returning_principal_id not in worker_ids:
        raise ValueError("returning principal is not one of the five workers")
    nonreturning_ids = set(worker_ids) - {returning_principal_id}
    if not absence_tasks:
        raise ValueError("absence horizon must contain at least one task")
    absence_principal_ids = tuple(
        str(principal_id) for principal_id, _ in absence_tasks
    )
    if set(absence_principal_ids) - nonreturning_ids:
        raise ValueError(
            "absence tasks must be assigned to non-returning workers"
        )
    absence_descriptions = tuple(
        str(description).strip() for _, description in absence_tasks
    )
    if any(not value for value in absence_descriptions):
        raise ValueError("absence task descriptions must be non-empty")
    if (aggregator_principal_id is None) != (evaluator_principal_id is None):
        raise ValueError(
            "aggregator and evaluator must either both be present or absent"
        )
    extra_ids = tuple(
        principal_id
        for principal_id in (
            aggregator_principal_id,
            evaluator_principal_id,
        )
        if principal_id is not None
    )
    if set(extra_ids).intersection(worker_ids) or len(set(extra_ids)) != len(
        extra_ids
    ):
        raise ValueError("aggregator/evaluator identities must be distinct")
    critical_indices = tuple(
        dict.fromkeys(int(index) for index in critical_absence_task_indices)
    )
    if predeparture_critical_dependency_count < 1:
        raise ValueError(
            "predeparture_critical_dependency_count must be positive"
        )
    protocol = RouterReturnProtocol(
        task_id=task_id,
        task_description=task_description,
    )
    for principal_id, role_id in workers:
        protocol.add_member(principal_id, role_id)
    absence_node_ids = tuple(
        _stable_id("node-absence", task_id, str(index), principal_id)
        for index, principal_id in enumerate(absence_principal_ids)
    )
    absence_dependency_ids = tuple(
        _stable_id("dep-absence", task_id, str(index), principal_id)
        for index, principal_id in enumerate(absence_principal_ids)
    )
    legacy_prefix_dependency_id = _stable_id(
        "dep-prefix", task_id, returning_principal_id
    )
    predeparture_dependency_ids = (
        legacy_prefix_dependency_id,
        *(
            _stable_id(
                "dep-prefix",
                task_id,
                returning_principal_id,
                str(index),
            )
            for index in range(
                1, predeparture_critical_dependency_count
            )
        ),
    )
    ids = HorizonRouterScenarioIds(
        prefix_node_id=_stable_id("node-prefix", task_id),
        synthesis_node_id=_stable_id("node-synthesis", task_id),
        obligation_id=_stable_id(
            "obligation", task_id, returning_principal_id
        ),
        prefix_dependency_id=legacy_prefix_dependency_id,
        old_item_id=_stable_id(
            "workstate-old", task_id, returning_principal_id
        ),
        returning_principal_id=returning_principal_id,
        absence_principal_ids=absence_principal_ids,
        absence_node_ids=absence_node_ids,
        absence_dependency_ids=absence_dependency_ids,
        critical_absence_task_indices=critical_indices,
        aggregator_principal_id=aggregator_principal_id,
        evaluator_principal_id=evaluator_principal_id,
        aggregation_node_id=(
            _stable_id("node-aggregate", task_id)
            if aggregator_principal_id is not None
            else None
        ),
        evaluation_node_id=(
            _stable_id("node-evaluate", task_id)
            if evaluator_principal_id is not None
            else None
        ),
        predeparture_dependency_ids=predeparture_dependency_ids,
    )
    if aggregator_principal_id is not None:
        protocol.add_member(
            aggregator_principal_id, aggregator_role_id
        )
        protocol.add_member(evaluator_principal_id, evaluator_role_id)  # type: ignore[arg-type]
    critical_dependencies = (
        *ids.effective_predeparture_dependency_ids,
        *(
            ids.absence_dependency_ids[index]
            for index in critical_indices
        ),
    )
    obligation = RouterObligation(
        obligation_id=ids.obligation_id,
        description=(
            "Complete the current task using valid pre-departure work and "
            "the critical task updates produced during the absence horizon."
        ),
        owner_principal_id=returning_principal_id,
        required_dependency_ids=critical_dependencies,
        critical_dependency_ids=critical_dependencies,
    )
    absence_nodes: list[RouterTaskNode] = []
    previous_node_id = ids.prefix_node_id
    for index, principal_id in enumerate(absence_principal_ids):
        node_id = ids.absence_node_ids[index]
        absence_nodes.append(
            RouterTaskNode(
                node_id=node_id,
                description=absence_descriptions[index],
                assigned_principal_id=principal_id,
                depends_on=(previous_node_id,),
                obligation_ids=(ids.obligation_id,),
            )
        )
        previous_node_id = node_id
    terminal_nodes: tuple[RouterTaskNode, ...] = ()
    if aggregator_principal_id is not None:
        if ids.aggregation_node_id is None or ids.evaluation_node_id is None:
            raise RuntimeError("terminal node identifiers are unavailable")
        terminal_nodes = (
            RouterTaskNode(
                node_id=ids.aggregation_node_id,
                description=aggregation_description,
                assigned_principal_id=aggregator_principal_id,
                depends_on=(ids.synthesis_node_id,),
                obligation_ids=(ids.obligation_id,),
            ),
            RouterTaskNode(
                node_id=ids.evaluation_node_id,
                description=evaluation_description,
                assigned_principal_id=evaluator_principal_id,  # type: ignore[arg-type]
                depends_on=(ids.aggregation_node_id,),
                obligation_ids=(ids.obligation_id,),
            ),
        )
    dag = RouterTaskDAG(
        task_id=task_id,
        task_description=task_description,
        nodes=(
            RouterTaskNode(
                node_id=ids.prefix_node_id,
                description=prefix_description,
                assigned_principal_id=returning_principal_id,
                obligation_ids=(ids.obligation_id,),
            ),
            *absence_nodes,
            RouterTaskNode(
                node_id=ids.synthesis_node_id,
                description=synthesis_description,
                assigned_principal_id=returning_principal_id,
                depends_on=(ids.absence_node_ids[-1],),
                obligation_ids=(ids.obligation_id,),
            ),
            *terminal_nodes,
        ),
    )
    protocol.install_plan(dag, (obligation,))
    return protocol, ids


def record_preabsence_work(
    protocol: RouterReturnProtocol,
    ids: RouterScenarioIds,
    *,
    returning_principal_id: str,
    result: str,
) -> RouterWorkstateItem:
    result = result.strip()
    if not result:
        raise ValueError("pre-absence work result must be non-empty")
    protocol.start_node(ids.prefix_node_id, returning_principal_id)
    protocol.complete_node(
        ids.prefix_node_id, returning_principal_id, result
    )
    item = protocol.add_workstate_item(
        RouterWorkstateItem(
            item_id=ids.old_item_id,
            text=result,
            owner_principal_id=returning_principal_id,
            writer_principal_id=returning_principal_id,
            dependency_ids=tuple(
                getattr(
                    ids,
                    "effective_predeparture_dependency_ids",
                    (ids.prefix_dependency_id,),
                )
            ),
            obligation_ids=(ids.obligation_id,),
        )
    )
    protocol.depart(returning_principal_id)
    return item


def parse_absence_review(value: object) -> AbsenceReview:
    """Parse a reviewer relation, retaining a safe additive fallback."""

    if isinstance(value, Mapping):
        parsed = dict(value)
    else:
        text = str(value or "").strip()
        candidate = text
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, count=1)
            candidate = re.sub(r"\s*```$", "", candidate, count=1)
        if "{" in candidate and "}" in candidate:
            candidate = candidate[
                candidate.find("{") : candidate.rfind("}") + 1
            ]
        try:
            loaded = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            loaded = None
        parsed = dict(loaded) if isinstance(loaded, Mapping) else {
            "relation": "extend",
            "current_fact": text,
            "audit_note": "unstructured reviewer output; treated as additive",
        }
    relation = str(parsed.get("relation") or "extend").strip().lower()
    if relation not in {"retain", "supersede", "revoke", "extend"}:
        relation = "extend"
    current_fact = str(
        parsed.get("current_fact")
        or parsed.get("fact")
        or parsed.get("result")
        or ""
    ).strip()
    if not current_fact:
        current_fact = "The absence reviewer produced no admissible update."
        relation = "revoke"
    return AbsenceReview(
        relation=relation,
        current_fact=current_fact,
        audit_note=str(parsed.get("audit_note") or "").strip(),
    )


def absence_review_prompt(
    *,
    task_description: str,
    old_workstate: str,
    absence_assignment: str,
) -> str:
    return (
        "The assigned role has now left the team. Continue the task as the "
        "independent absence reviewer. Audit the public pre-absence work and "
        "return one JSON object with relation (retain, supersede, revoke, or "
        "extend), current_fact, and audit_note. 'supersede' means the old fact "
        "is no longer current; 'revoke' means it must not be reused; 'extend' "
        "adds a new dependency without invalidating it. Do not produce the "
        "final answer.\n\n"
        f"Overall task:\n{task_description}\n\n"
        f"Assignment:\n{absence_assignment}\n\n"
        f"Public pre-absence work:\n{old_workstate}"
    )


def record_absence_work(
    protocol: RouterReturnProtocol,
    ids: RouterScenarioIds,
    *,
    absence_principal_id: str,
    returning_principal_id: str,
    review: AbsenceReview,
) -> RouterWorkstateItem | None:
    protocol.start_node(ids.absence_node_id, absence_principal_id)
    protocol.complete_node(
        ids.absence_node_id,
        absence_principal_id,
        json.dumps(review.record(), ensure_ascii=False, sort_keys=True),
    )
    replacement: RouterWorkstateItem | None
    if review.relation == "revoke":
        protocol.revoke_workstate_item(
            actor_principal_id=absence_principal_id,
            item_id=ids.old_item_id,
            reason=review.audit_note or review.current_fact,
        )
        replacement = protocol.add_workstate_item(
            RouterWorkstateItem(
                item_id=_stable_id(
                    "workstate-revocation", protocol.task_id, review.current_fact
                ),
                text=review.current_fact,
                owner_principal_id=returning_principal_id,
                writer_principal_id=absence_principal_id,
                dependency_ids=(ids.absence_dependency_id,),
                obligation_ids=(ids.obligation_id,),
                version=2,
            )
        )
    elif review.relation == "supersede":
        replacement = protocol.supersede_workstate_item(
            actor_principal_id=absence_principal_id,
            old_item_id=ids.old_item_id,
            replacement_item=RouterWorkstateItem(
                item_id=_stable_id(
                    "workstate-current", protocol.task_id, review.current_fact
                ),
                text=review.current_fact,
                owner_principal_id=returning_principal_id,
                writer_principal_id=absence_principal_id,
                dependency_ids=(
                    ids.prefix_dependency_id,
                    ids.absence_dependency_id,
                ),
                obligation_ids=(ids.obligation_id,),
                version=2,
                supersedes_item_id=ids.old_item_id,
            ),
        )
    else:
        replacement = protocol.add_workstate_item(
            RouterWorkstateItem(
                item_id=_stable_id(
                    "workstate-delta", protocol.task_id, review.current_fact
                ),
                text=review.current_fact,
                owner_principal_id=returning_principal_id,
                writer_principal_id=absence_principal_id,
                dependency_ids=(ids.absence_dependency_id,),
                obligation_ids=(ids.obligation_id,),
                version=2,
            )
        )
    return replacement


def record_multiworker_absence_work(
    protocol: RouterReturnProtocol,
    ids: MultiWorkerRouterScenarioIds,
    *,
    absence_principal_id: str,
    review: AbsenceReview,
) -> RouterWorkstateItem:
    """Commit one absence worker's independent result to Router state."""

    node_id, dependency_id = ids.for_worker(absence_principal_id)
    protocol.start_node(node_id, absence_principal_id)
    protocol.complete_node(
        node_id,
        absence_principal_id,
        json.dumps(review.record(), ensure_ascii=False, sort_keys=True),
    )
    old = protocol.workstate[ids.old_item_id]
    old_is_active = old.status is RouterWorkstateStatus.ACTIVE
    if review.relation == "revoke" and old_is_active:
        protocol.revoke_workstate_item(
            actor_principal_id=absence_principal_id,
            item_id=ids.old_item_id,
            reason=review.audit_note or review.current_fact,
        )
    if review.relation == "supersede" and old_is_active:
        return protocol.supersede_workstate_item(
            actor_principal_id=absence_principal_id,
            old_item_id=ids.old_item_id,
            replacement_item=RouterWorkstateItem(
                item_id=_stable_id(
                    "workstate-current",
                    protocol.task_id,
                    absence_principal_id,
                    review.current_fact,
                ),
                text=review.current_fact,
                owner_principal_id=ids.returning_principal_id,
                writer_principal_id=absence_principal_id,
                dependency_ids=(
                    *ids.effective_predeparture_dependency_ids,
                    dependency_id,
                ),
                obligation_ids=(ids.obligation_id,),
                version=2,
                supersedes_item_id=ids.old_item_id,
            ),
        )
    return protocol.add_workstate_item(
        RouterWorkstateItem(
            item_id=_stable_id(
                (
                    "workstate-revocation"
                    if review.relation == "revoke"
                    else "workstate-delta"
                ),
                protocol.task_id,
                absence_principal_id,
                review.current_fact,
            ),
            text=review.current_fact,
            owner_principal_id=ids.returning_principal_id,
            writer_principal_id=absence_principal_id,
            dependency_ids=(dependency_id,),
            obligation_ids=(ids.obligation_id,),
            version=2,
        )
    )


def record_horizon_absence_work(
    protocol: RouterReturnProtocol,
    ids: HorizonRouterScenarioIds,
    *,
    task_index: int,
    absence_principal_id: str,
    review: AbsenceReview,
) -> RouterWorkstateItem:
    """Commit one ordered absence task and its version relation."""

    expected_principal_id, node_id, dependency_id = ids.for_task(
        task_index
    )
    if absence_principal_id != expected_principal_id:
        raise ValueError("absence task was executed by the wrong principal")
    protocol.start_node(node_id, absence_principal_id)
    protocol.complete_node(
        node_id,
        absence_principal_id,
        json.dumps(review.record(), ensure_ascii=False, sort_keys=True),
    )
    old = protocol.workstate[ids.old_item_id]
    old_is_active = old.status is RouterWorkstateStatus.ACTIVE
    if review.relation == "revoke" and old_is_active:
        protocol.revoke_workstate_item(
            actor_principal_id=absence_principal_id,
            item_id=ids.old_item_id,
            reason=review.audit_note or review.current_fact,
        )
    if review.relation == "supersede" and old_is_active:
        return protocol.supersede_workstate_item(
            actor_principal_id=absence_principal_id,
            old_item_id=ids.old_item_id,
            replacement_item=RouterWorkstateItem(
                item_id=_stable_id(
                    "workstate-current",
                    protocol.task_id,
                    str(task_index),
                    absence_principal_id,
                    review.current_fact,
                ),
                text=review.current_fact,
                owner_principal_id=ids.returning_principal_id,
                writer_principal_id=absence_principal_id,
                dependency_ids=(
                    *ids.effective_predeparture_dependency_ids,
                    dependency_id,
                ),
                obligation_ids=(ids.obligation_id,),
                version=2,
                supersedes_item_id=ids.old_item_id,
            ),
        )
    return protocol.add_workstate_item(
        RouterWorkstateItem(
            item_id=_stable_id(
                (
                    "workstate-revocation"
                    if review.relation == "revoke"
                    else "workstate-delta"
                ),
                protocol.task_id,
                str(task_index),
                absence_principal_id,
                review.current_fact,
            ),
            text=review.current_fact,
            owner_principal_id=ids.returning_principal_id,
            writer_principal_id=absence_principal_id,
            dependency_ids=(dependency_id,),
            obligation_ids=(ids.obligation_id,),
            version=task_index + 2,
        )
    )


__all__ = [
    "AbsenceReview",
    "HorizonRouterScenarioIds",
    "MultiWorkerRouterScenarioIds",
    "RouterScenarioIds",
    "absence_review_prompt",
    "build_five_worker_router_return_scenario",
    "build_horizon_router_return_scenario",
    "build_router_return_scenario",
    "parse_absence_review",
    "record_absence_work",
    "record_horizon_absence_work",
    "record_multiworker_absence_work",
    "record_preabsence_work",
]
