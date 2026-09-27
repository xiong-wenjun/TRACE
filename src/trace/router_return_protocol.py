"""Router-owned execution state for controlled agent RETURN experiments.

The protocol separates three concerns that were previously mixed inside
benchmark runners:

* a router decomposes one task into a dependency DAG;
* the router owns membership, obligations, workstate, and an append-only ledger;
* all RETURN methods are materialized from one immutable post-absence
  checkpoint.

The module deliberately contains no model or benchmark dependency.  A runner
may use any planner/worker implementation, but it must report state changes to
this protocol before it can claim a real RETURN event.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


ROUTER_RETURN_PROTOCOL = "router_return_protocol_v1"
ROUTER_TASK_DAG_SCHEMA = "router_return_task_dag_v1"
ROUTER_CHECKPOINT_SCHEMA = "router_return_post_absence_checkpoint_v2"


def canonical_json(value: object) -> bytes:
    """Encode protocol records deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _required_text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _unique_text(values: Iterable[object], field_name: str) -> tuple[str, ...]:
    result = tuple(_required_text(value, field_name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} values must be unique")
    return result


class RouterNodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class RouterMembershipStatus(str, Enum):
    ACTIVE = "active"
    ABSENT = "absent"
    RETURNING = "returning"
    ADMITTED = "admitted"
    BLOCKED = "blocked"


class RouterObligationStatus(str, Enum):
    OPEN = "open"
    SATISFIED = "satisfied"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class RouterWorkstateStatus(str, Enum):
    ACTIVE = "active"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


class RouterDisputeStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class RouterLedgerEvent(str, Enum):
    TASK_DECOMPOSED = "task_decomposed"
    MEMBER_ADDED = "member_added"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    OBLIGATION_OPENED = "obligation_opened"
    OBLIGATION_UPDATED = "obligation_updated"
    WORKSTATE_CREATED = "workstate_created"
    WORKSTATE_DISPUTED = "workstate_disputed"
    WORKSTATE_DISPUTE_RESOLVED = "workstate_dispute_resolved"
    WORKSTATE_SUPERSEDED = "workstate_superseded"
    WORKSTATE_REVOKED = "workstate_revoked"
    MEMBER_DEPARTED = "member_departed"
    RETURN_STARTED = "return_started"
    RETURN_ADMITTED = "return_admitted"
    RETURN_BLOCKED = "return_blocked"
    CHECKPOINT_FROZEN = "checkpoint_frozen"


class RouterReturnArm(str, Enum):
    """Counterfactual RETURN views from one post-absence checkpoint."""

    STATIC = "static_no_churn"
    STATIC_COMPACT = "static_compact"
    COMPACT_ONLY = "compact_only"
    RESET = "reset"
    RESET_REBRIEF = "reset_rebrief"
    RESTORE_OLD = "restore_old"
    VALIDITY_FILTER_ONLY = "validity_filter_only"
    FULL_MERGE = "full_merge"
    TRACE = "trace"

    @classmethod
    def _missing_(cls, value: object) -> "RouterReturnArm | None":
        # Keep legacy artifact parsing at the enum boundary; runtime records
        # and all new forks remain canonically named TRACE.
        from .trace_method import canonical_return_arm

        canonical = canonical_return_arm(value)
        if canonical != str(value).strip():
            return cls(canonical)
        return None


@dataclass(frozen=True)
class RouterTaskNode:
    node_id: str
    description: str
    assigned_principal_id: str
    depends_on: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    status: RouterNodeStatus = RouterNodeStatus.PENDING
    result_summary: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.node_id, "node_id")
        _required_text(self.description, "description")
        _required_text(self.assigned_principal_id, "assigned_principal_id")
        _unique_text(self.depends_on, "depends_on")
        _unique_text(self.obligation_ids, "obligation_ids")
        if self.node_id in self.depends_on:
            raise ValueError("a task node cannot depend on itself")
        if not isinstance(self.status, RouterNodeStatus):
            raise TypeError("status must be RouterNodeStatus")
        if self.status is RouterNodeStatus.COMPLETED and not str(
            self.result_summary or ""
        ).strip():
            raise ValueError("a completed node requires a result summary")

    def record(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "description": self.description,
            "assigned_principal_id": self.assigned_principal_id,
            "depends_on": list(self.depends_on),
            "obligation_ids": list(self.obligation_ids),
            "status": self.status.value,
            "result_summary": self.result_summary,
        }


@dataclass(frozen=True)
class RouterTaskDAG:
    task_id: str
    task_description: str
    nodes: tuple[RouterTaskNode, ...]
    schema_version: str = ROUTER_TASK_DAG_SCHEMA

    def __post_init__(self) -> None:
        _required_text(self.task_id, "task_id")
        _required_text(self.task_description, "task_description")
        if not self.nodes:
            raise ValueError("a router plan requires at least one task node")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("router task node ids must be unique")
        known = set(node_ids)
        for node in self.nodes:
            unknown = set(node.depends_on) - known
            if unknown:
                raise ValueError(
                    f"node {node.node_id} has unknown dependencies: "
                    + ",".join(sorted(unknown))
                )
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        dependencies = {
            node.node_id: set(node.depends_on) for node in self.nodes
        }
        resolved: set[str] = set()
        while len(resolved) < len(dependencies):
            ready = {
                node_id
                for node_id, required in dependencies.items()
                if node_id not in resolved and required.issubset(resolved)
            }
            if not ready:
                raise ValueError("router task graph contains a dependency cycle")
            resolved.update(ready)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.record())

    def ready_node_ids(self) -> tuple[str, ...]:
        completed = {
            node.node_id
            for node in self.nodes
            if node.status is RouterNodeStatus.COMPLETED
        }
        return tuple(
            node.node_id
            for node in self.nodes
            if node.status is RouterNodeStatus.PENDING
            and set(node.depends_on).issubset(completed)
        )

    def record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "nodes": [node.record() for node in self.nodes],
        }


@dataclass(frozen=True)
class RouterMembership:
    principal_id: str
    role_id: str
    epoch: int = 0
    status: RouterMembershipStatus = RouterMembershipStatus.ACTIVE

    def __post_init__(self) -> None:
        _required_text(self.principal_id, "principal_id")
        _required_text(self.role_id, "role_id")
        if self.epoch < 0:
            raise ValueError("membership epoch must be non-negative")
        if not isinstance(self.status, RouterMembershipStatus):
            raise TypeError("status must be RouterMembershipStatus")

    def record(self) -> dict[str, object]:
        return {
            "principal_id": self.principal_id,
            "role_id": self.role_id,
            "epoch": self.epoch,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class RouterObligation:
    obligation_id: str
    description: str
    owner_principal_id: str
    required_dependency_ids: tuple[str, ...]
    critical_dependency_ids: tuple[str, ...] = ()
    status: RouterObligationStatus = RouterObligationStatus.OPEN

    def __post_init__(self) -> None:
        _required_text(self.obligation_id, "obligation_id")
        _required_text(self.description, "description")
        _required_text(self.owner_principal_id, "owner_principal_id")
        required = _unique_text(
            self.required_dependency_ids, "required_dependency_ids"
        )
        critical = _unique_text(
            self.critical_dependency_ids, "critical_dependency_ids"
        )
        if not required:
            raise ValueError("an obligation requires at least one dependency")
        if not set(critical).issubset(required):
            raise ValueError("critical dependencies must also be required")
        if not isinstance(self.status, RouterObligationStatus):
            raise TypeError("status must be RouterObligationStatus")

    def record(self) -> dict[str, object]:
        return {
            "obligation_id": self.obligation_id,
            "description": self.description,
            "owner_principal_id": self.owner_principal_id,
            "required_dependency_ids": list(self.required_dependency_ids),
            "critical_dependency_ids": list(self.critical_dependency_ids),
            "status": self.status.value,
        }


@dataclass(frozen=True)
class RouterWorkstateItem:
    """A versioned workstate item stored by the Router, not an LLM message."""

    item_id: str
    text: str
    owner_principal_id: str
    writer_principal_id: str
    dependency_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    version: int = 1
    status: RouterWorkstateStatus = RouterWorkstateStatus.ACTIVE
    supersedes_item_id: str | None = None
    team_public: bool = True
    provenance_sha256: str = ""
    provenance_status: str = "legacy"
    provenance_source_ids: tuple[str, ...] = ()
    provenance_receipt_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.item_id, "item_id")
        _required_text(self.text, "text")
        _required_text(self.owner_principal_id, "owner_principal_id")
        _required_text(self.writer_principal_id, "writer_principal_id")
        _unique_text(self.dependency_ids, "dependency_ids")
        _unique_text(self.obligation_ids, "obligation_ids")
        if self.version < 1:
            raise ValueError("workstate version must be positive")
        if not isinstance(self.status, RouterWorkstateStatus):
            raise TypeError("status must be RouterWorkstateStatus")
        if self.supersedes_item_id is not None:
            _required_text(self.supersedes_item_id, "supersedes_item_id")
            if self.supersedes_item_id == self.item_id:
                raise ValueError("a workstate item cannot supersede itself")
        if self.provenance_sha256:
            if len(self.provenance_sha256) != 64:
                raise ValueError("provenance_sha256 must be SHA-256")
            bytes.fromhex(self.provenance_sha256)
        if self.provenance_status not in {
            "legacy",
            "valid",
            "replicated",
            "missing",
            "conflicting",
        }:
            raise ValueError(
                "unsupported provenance_status: "
                + str(self.provenance_status)
            )
        for source_id in self.provenance_source_ids:
            _required_text(source_id, "provenance_source_id")
        for receipt in self.provenance_receipt_sha256s:
            if len(str(receipt)) != 64:
                raise ValueError("provenance receipt must be SHA-256")
            bytes.fromhex(str(receipt))
        if self.provenance_status == "missing" and self.provenance_receipt_sha256s:
            raise ValueError("missing provenance cannot carry receipts")
        if self.provenance_status == "valid" and not self.provenance_receipt_sha256s:
            raise ValueError("valid provenance requires at least one receipt")

    def with_default_provenance(self) -> "RouterWorkstateItem":
        if self.provenance_sha256:
            return self
        return replace(
            self,
            provenance_sha256=canonical_sha256(
                {
                    "item_id": self.item_id,
                    "text": self.text,
                    "owner_principal_id": self.owner_principal_id,
                    "writer_principal_id": self.writer_principal_id,
                    "dependency_ids": list(self.dependency_ids),
                    "obligation_ids": list(self.obligation_ids),
                    "version": self.version,
                    "supersedes_item_id": self.supersedes_item_id,
                    "provenance_status": self.provenance_status,
                    "provenance_source_ids": list(self.provenance_source_ids),
                    "provenance_receipt_sha256s": list(
                        self.provenance_receipt_sha256s
                    ),
                }
            ),
        )

    def record(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "owner_principal_id": self.owner_principal_id,
            "writer_principal_id": self.writer_principal_id,
            "dependency_ids": list(self.dependency_ids),
            "obligation_ids": list(self.obligation_ids),
            "version": self.version,
            "status": self.status.value,
            "supersedes_item_id": self.supersedes_item_id,
            "team_public": self.team_public,
            "provenance_sha256": self.provenance_sha256,
            "provenance_status": self.provenance_status,
            "provenance_source_ids": list(self.provenance_source_ids),
            "provenance_receipt_sha256s": list(
                self.provenance_receipt_sha256s
            ),
        }


@dataclass(frozen=True)
class RouterWorkstateDispute:
    """A conflicting update that cannot mutate current truth until verified."""

    dispute_id: str
    existing_item_id: str
    challenger_item_id: str
    opened_by_principal_id: str
    reason: str
    opened_ledger_sequence: int
    status: RouterDisputeStatus = RouterDisputeStatus.OPEN
    accepted_item_id: str | None = None
    rejected_item_id: str | None = None
    verifier_principal_id: str | None = None
    verification_evidence_sha256: str | None = None
    resolution_reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.dispute_id, "dispute_id")
        _required_text(self.existing_item_id, "existing_item_id")
        _required_text(self.challenger_item_id, "challenger_item_id")
        _required_text(self.opened_by_principal_id, "opened_by_principal_id")
        _required_text(self.reason, "reason")
        if self.existing_item_id == self.challenger_item_id:
            raise ValueError("a dispute requires two different workstate items")
        if self.opened_ledger_sequence < 0:
            raise ValueError("opened_ledger_sequence must be non-negative")
        if not isinstance(self.status, RouterDisputeStatus):
            raise TypeError("status must be RouterDisputeStatus")
        if self.status is RouterDisputeStatus.RESOLVED:
            required = {
                "accepted_item_id": self.accepted_item_id,
                "rejected_item_id": self.rejected_item_id,
                "verifier_principal_id": self.verifier_principal_id,
                "verification_evidence_sha256": (
                    self.verification_evidence_sha256
                ),
                "resolution_reason": self.resolution_reason,
            }
            missing = [key for key, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "resolved dispute is missing: " + ",".join(missing)
                )
        if self.verification_evidence_sha256 is not None:
            if len(self.verification_evidence_sha256) != 64:
                raise ValueError(
                    "verification_evidence_sha256 must be SHA-256"
                )
            bytes.fromhex(self.verification_evidence_sha256)

    def record(self) -> dict[str, object]:
        return {
            "dispute_id": self.dispute_id,
            "existing_item_id": self.existing_item_id,
            "challenger_item_id": self.challenger_item_id,
            "opened_by_principal_id": self.opened_by_principal_id,
            "reason": self.reason,
            "opened_ledger_sequence": self.opened_ledger_sequence,
            "status": self.status.value,
            "accepted_item_id": self.accepted_item_id,
            "rejected_item_id": self.rejected_item_id,
            "verifier_principal_id": self.verifier_principal_id,
            "verification_evidence_sha256": (
                self.verification_evidence_sha256
            ),
            "resolution_reason": self.resolution_reason,
        }


@dataclass(frozen=True)
class RouterLedgerEntry:
    sequence: int
    event: RouterLedgerEvent
    actor_principal_id: str
    logical_time: int
    payload: Mapping[str, object]
    affected_principal_ids: tuple[str, ...] = ()
    affected_obligation_ids: tuple[str, ...] = ()
    affected_dependency_ids: tuple[str, ...] = ()
    previous_entry_sha256: str | None = None
    entry_sha256: str = ""

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.logical_time < 0:
            raise ValueError("ledger sequence/time must be non-negative")
        if not isinstance(self.event, RouterLedgerEvent):
            raise TypeError("event must be RouterLedgerEvent")
        _required_text(self.actor_principal_id, "actor_principal_id")
        _unique_text(self.affected_principal_ids, "affected_principal_ids")
        _unique_text(self.affected_obligation_ids, "affected_obligation_ids")
        _unique_text(self.affected_dependency_ids, "affected_dependency_ids")
        if self.previous_entry_sha256 is not None:
            if len(self.previous_entry_sha256) != 64:
                raise ValueError("previous ledger digest must be SHA-256")
            bytes.fromhex(self.previous_entry_sha256)
        if self.entry_sha256:
            if self.entry_sha256 != canonical_sha256(self.unsigned_record()):
                raise ValueError("ledger entry digest mismatch")

    def unsigned_record(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "event": self.event.value,
            "actor_principal_id": self.actor_principal_id,
            "logical_time": self.logical_time,
            "payload": dict(self.payload),
            "affected_principal_ids": list(self.affected_principal_ids),
            "affected_obligation_ids": list(self.affected_obligation_ids),
            "affected_dependency_ids": list(self.affected_dependency_ids),
            "previous_entry_sha256": self.previous_entry_sha256,
        }

    def record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "entry_sha256": self.entry_sha256}


@dataclass(frozen=True)
class RouterDepartureCheckpoint:
    task_id: str
    principal_id: str
    source_epoch: int
    departure_ledger_sequence: int
    completed_node_ids: tuple[str, ...]
    open_obligation_ids: tuple[str, ...]
    workstate_items: tuple[RouterWorkstateItem, ...]
    dag_digest: str
    ledger_head_sha256: str
    checkpoint_sha256: str = ""

    def unsigned_record(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "principal_id": self.principal_id,
            "source_epoch": self.source_epoch,
            "departure_ledger_sequence": self.departure_ledger_sequence,
            "completed_node_ids": list(self.completed_node_ids),
            "open_obligation_ids": list(self.open_obligation_ids),
            "workstate_items": [item.record() for item in self.workstate_items],
            "dag_digest": self.dag_digest,
            "ledger_head_sha256": self.ledger_head_sha256,
        }

    def record(self) -> dict[str, object]:
        return {
            **self.unsigned_record(),
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True)
class RouterAbsenceDelta:
    principal_id: str
    ledger_entries: tuple[RouterLedgerEntry, ...]
    created_item_ids: tuple[str, ...]
    superseded_item_ids: tuple[str, ...]
    revoked_item_ids: tuple[str, ...]
    changed_obligation_ids: tuple[str, ...]
    relevant_entry_sequences: tuple[int, ...]
    nonowner_action_count: int
    disputed_item_ids: tuple[str, ...] = ()
    resolved_dispute_ids: tuple[str, ...] = ()

    @property
    def relevant_change_count(self) -> int:
        return len(self.relevant_entry_sequences)

    @property
    def qualified(self) -> bool:
        return self.nonowner_action_count > 0 and self.relevant_change_count > 0

    def record(self) -> dict[str, object]:
        return {
            "principal_id": self.principal_id,
            "ledger_entries": [entry.record() for entry in self.ledger_entries],
            "created_item_ids": list(self.created_item_ids),
            "superseded_item_ids": list(self.superseded_item_ids),
            "revoked_item_ids": list(self.revoked_item_ids),
            "changed_obligation_ids": list(self.changed_obligation_ids),
            "disputed_item_ids": list(self.disputed_item_ids),
            "resolved_dispute_ids": list(self.resolved_dispute_ids),
            "relevant_entry_sequences": list(self.relevant_entry_sequences),
            "nonowner_action_count": self.nonowner_action_count,
            "relevant_change_count": self.relevant_change_count,
            "qualified": self.qualified,
        }


@dataclass(frozen=True)
class RouterPostAbsenceCheckpoint:
    task_id: str
    returning_principal_id: str
    target_epoch: int
    departure: RouterDepartureCheckpoint
    absence_delta: RouterAbsenceDelta
    dag: RouterTaskDAG
    memberships: tuple[RouterMembership, ...]
    obligations: tuple[RouterObligation, ...]
    current_workstate_items: tuple[RouterWorkstateItem, ...]
    disputes: tuple[RouterWorkstateDispute, ...]
    ledger_head_sha256: str
    checkpoint_sha256: str = ""
    schema_version: str = ROUTER_CHECKPOINT_SCHEMA

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "returning_principal_id": self.returning_principal_id,
            "target_epoch": self.target_epoch,
            "departure": self.departure.record(),
            "absence_delta": self.absence_delta.record(),
            "dag": self.dag.record(),
            "memberships": [item.record() for item in self.memberships],
            "obligations": [item.record() for item in self.obligations],
            "current_workstate_items": [
                item.record() for item in self.current_workstate_items
            ],
            "disputes": [item.record() for item in self.disputes],
            "ledger_head_sha256": self.ledger_head_sha256,
        }

    def record(self) -> dict[str, object]:
        return {
            **self.unsigned_record(),
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True)
class RouterReturnFork:
    arm: RouterReturnArm
    checkpoint_sha256: str
    returning_principal_id: str
    inherited_item_ids: tuple[str, ...]
    candidate_item_ids: tuple[str, ...]
    compiler_required: bool = False
    blocked: bool = False
    block_reason: str | None = None

    def record(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "checkpoint_sha256": self.checkpoint_sha256,
            "returning_principal_id": self.returning_principal_id,
            "inherited_item_ids": list(self.inherited_item_ids),
            "candidate_item_ids": list(self.candidate_item_ids),
            "compiler_required": self.compiler_required,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }


def router_decomposition_prompt(
    *,
    task_description: str,
    members: Sequence[RouterMembership],
    returning_principal_id: str,
) -> str:
    """Return the strict plan contract used by model-backed Routers."""

    member_rows = [
        {
            "principal_id": member.principal_id,
            "role_id": member.role_id,
        }
        for member in members
    ]
    return (
        "Decompose the task into a small dependency DAG for a multi-agent "
        "team. Return one JSON object with key 'subtasks'. Each subtask must "
        "contain node_id, description, assigned_principal_id, depends_on, "
        "obligation_id, obligation_description, required_dependency_ids, and "
        "critical_dependency_ids. At least one early node must be assigned to "
        f"{returning_principal_id}; independent or downstream work must remain "
        "for other agents during its absence. Do not include an answer or a "
        "complete action sequence.\n\n"
        f"Task:\n{task_description}\n\n"
        f"Members:\n{json.dumps(member_rows, ensure_ascii=False, sort_keys=True)}"
    )


def parse_router_plan(
    *,
    task_id: str,
    task_description: str,
    value: Mapping[str, object],
) -> tuple[RouterTaskDAG, tuple[RouterObligation, ...]]:
    """Parse a strict Router plan without accepting hidden workflow fields."""

    raw_subtasks = value.get("subtasks")
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        raise ValueError("router plan requires a non-empty subtasks list")
    nodes: list[RouterTaskNode] = []
    obligations: list[RouterObligation] = []
    for raw in raw_subtasks:
        if not isinstance(raw, Mapping):
            raise ValueError("every router subtask must be an object")
        forbidden = {
            "answer",
            "gold",
            "workflow",
            "action_sequence",
            "expert_plan",
        }.intersection(str(key) for key in raw)
        if forbidden:
            raise ValueError(
                "router plan contains workflow-oracle fields: "
                + ",".join(sorted(forbidden))
            )
        obligation_id = _required_text(
            raw.get("obligation_id"), "obligation_id"
        )
        principal_id = _required_text(
            raw.get("assigned_principal_id"), "assigned_principal_id"
        )
        nodes.append(
            RouterTaskNode(
                node_id=_required_text(raw.get("node_id"), "node_id"),
                description=_required_text(
                    raw.get("description"), "description"
                ),
                assigned_principal_id=principal_id,
                depends_on=_unique_text(
                    raw.get("depends_on") or (), "depends_on"
                ),
                obligation_ids=(obligation_id,),
            )
        )
        obligations.append(
            RouterObligation(
                obligation_id=obligation_id,
                description=_required_text(
                    raw.get("obligation_description"),
                    "obligation_description",
                ),
                owner_principal_id=principal_id,
                required_dependency_ids=_unique_text(
                    raw.get("required_dependency_ids") or (),
                    "required_dependency_ids",
                ),
                critical_dependency_ids=_unique_text(
                    raw.get("critical_dependency_ids") or (),
                    "critical_dependency_ids",
                ),
            )
        )
    return (
        RouterTaskDAG(
            task_id=task_id,
            task_description=task_description,
            nodes=tuple(nodes),
        ),
        tuple(obligations),
    )


@dataclass
class RouterReturnProtocol:
    """Mutable Router state whose checkpoints and ledger are immutable records."""

    task_id: str
    task_description: str
    dag: RouterTaskDAG | None = None
    memberships: dict[str, RouterMembership] = field(default_factory=dict)
    obligations: dict[str, RouterObligation] = field(default_factory=dict)
    workstate: dict[str, RouterWorkstateItem] = field(default_factory=dict)
    disputes: dict[str, RouterWorkstateDispute] = field(default_factory=dict)
    ledger: list[RouterLedgerEntry] = field(default_factory=list)
    departures: dict[str, RouterDepartureCheckpoint] = field(
        default_factory=dict
    )
    frozen_checkpoints: dict[str, RouterPostAbsenceCheckpoint] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.task_id = _required_text(self.task_id, "task_id")
        self.task_description = _required_text(
            self.task_description, "task_description"
        )

    @property
    def ledger_head_sha256(self) -> str:
        return self.ledger[-1].entry_sha256 if self.ledger else "0" * 64

    def _append(
        self,
        event: RouterLedgerEvent,
        actor_principal_id: str,
        payload: Mapping[str, object],
        *,
        affected_principal_ids: Iterable[str] = (),
        affected_obligation_ids: Iterable[str] = (),
        affected_dependency_ids: Iterable[str] = (),
    ) -> RouterLedgerEntry:
        entry = RouterLedgerEntry(
            sequence=len(self.ledger),
            event=event,
            actor_principal_id=_required_text(
                actor_principal_id, "actor_principal_id"
            ),
            logical_time=len(self.ledger),
            payload=dict(payload),
            affected_principal_ids=_unique_text(
                affected_principal_ids, "affected_principal_ids"
            ),
            affected_obligation_ids=_unique_text(
                affected_obligation_ids, "affected_obligation_ids"
            ),
            affected_dependency_ids=_unique_text(
                affected_dependency_ids, "affected_dependency_ids"
            ),
            previous_entry_sha256=(
                self.ledger[-1].entry_sha256 if self.ledger else None
            ),
        )
        entry = replace(
            entry, entry_sha256=canonical_sha256(entry.unsigned_record())
        )
        self.ledger.append(entry)
        return entry

    def add_member(
        self, principal_id: str, role_id: str
    ) -> RouterMembership:
        principal_id = _required_text(principal_id, "principal_id")
        if principal_id in self.memberships:
            raise ValueError(f"member already exists: {principal_id}")
        member = RouterMembership(principal_id=principal_id, role_id=role_id)
        self.memberships[principal_id] = member
        self._append(
            RouterLedgerEvent.MEMBER_ADDED,
            "router",
            member.record(),
            affected_principal_ids=(principal_id,),
        )
        return member

    def install_plan(
        self,
        dag: RouterTaskDAG,
        obligations: Sequence[RouterObligation],
    ) -> None:
        if self.dag is not None:
            raise RuntimeError("router plan is already installed")
        if dag.task_id != self.task_id:
            raise ValueError("router plan task_id mismatch")
        assigned = {node.assigned_principal_id for node in dag.nodes}
        unknown_members = assigned - set(self.memberships)
        if unknown_members:
            raise ValueError(
                "router plan assigns unknown members: "
                + ",".join(sorted(unknown_members))
            )
        obligation_map = {
            obligation.obligation_id: obligation
            for obligation in obligations
        }
        if len(obligation_map) != len(obligations):
            raise ValueError("router obligation ids must be unique")
        referenced = {
            obligation_id
            for node in dag.nodes
            for obligation_id in node.obligation_ids
        }
        if referenced - set(obligation_map):
            raise ValueError("task DAG references unknown obligations")
        self.dag = dag
        self.obligations = obligation_map
        self._append(
            RouterLedgerEvent.TASK_DECOMPOSED,
            "router",
            {
                "dag_digest": dag.digest,
                "node_ids": [node.node_id for node in dag.nodes],
            },
            affected_principal_ids=sorted(assigned),
            affected_obligation_ids=sorted(obligation_map),
        )
        for obligation in obligations:
            self._append(
                RouterLedgerEvent.OBLIGATION_OPENED,
                "router",
                obligation.record(),
                affected_principal_ids=(obligation.owner_principal_id,),
                affected_obligation_ids=(obligation.obligation_id,),
                affected_dependency_ids=obligation.required_dependency_ids,
            )

    def _member(self, principal_id: str) -> RouterMembership:
        try:
            return self.memberships[principal_id]
        except KeyError as error:
            raise KeyError(f"unknown Router member: {principal_id}") from error

    def _assert_can_act(self, principal_id: str) -> None:
        status = self._member(principal_id).status
        if status not in {
            RouterMembershipStatus.ACTIVE,
            RouterMembershipStatus.ADMITTED,
        }:
            raise PermissionError(
                f"member {principal_id} cannot act while {status.value}"
            )

    def _replace_node(self, node_id: str, **changes: object) -> RouterTaskNode:
        if self.dag is None:
            raise RuntimeError("router plan is not installed")
        selected: RouterTaskNode | None = None
        nodes: list[RouterTaskNode] = []
        for node in self.dag.nodes:
            if node.node_id == node_id:
                selected = replace(node, **changes)
                nodes.append(selected)
            else:
                nodes.append(node)
        if selected is None:
            raise KeyError(f"unknown Router task node: {node_id}")
        self.dag = replace(self.dag, nodes=tuple(nodes))
        return selected

    def start_node(self, node_id: str, principal_id: str) -> None:
        self._assert_can_act(principal_id)
        if self.dag is None:
            raise RuntimeError("router plan is not installed")
        node = next(
            (item for item in self.dag.nodes if item.node_id == node_id),
            None,
        )
        if node is None:
            raise KeyError(f"unknown Router task node: {node_id}")
        if node.assigned_principal_id != principal_id:
            raise PermissionError("only the assigned member may start a node")
        if node.status is not RouterNodeStatus.PENDING:
            raise RuntimeError("only a pending node may be started")
        if node_id not in self.dag.ready_node_ids():
            raise RuntimeError("task node dependencies are not complete")
        self._replace_node(node_id, status=RouterNodeStatus.RUNNING)
        self._append(
            RouterLedgerEvent.NODE_STARTED,
            principal_id,
            {"node_id": node_id},
            affected_principal_ids=(principal_id,),
            affected_obligation_ids=node.obligation_ids,
        )

    def complete_node(
        self,
        node_id: str,
        principal_id: str,
        result_summary: str,
    ) -> None:
        self._assert_can_act(principal_id)
        if self.dag is None:
            raise RuntimeError("router plan is not installed")
        node = next(
            (item for item in self.dag.nodes if item.node_id == node_id),
            None,
        )
        if node is None:
            raise KeyError(f"unknown Router task node: {node_id}")
        if node.assigned_principal_id != principal_id:
            raise PermissionError("only the assigned member may complete a node")
        if node.status is not RouterNodeStatus.RUNNING:
            raise RuntimeError("only a running node may be completed")
        result_summary = _required_text(result_summary, "result_summary")
        self._replace_node(
            node_id,
            status=RouterNodeStatus.COMPLETED,
            result_summary=result_summary,
        )
        self._append(
            RouterLedgerEvent.NODE_COMPLETED,
            principal_id,
            {"node_id": node_id, "result_sha256": canonical_sha256(result_summary)},
            affected_principal_ids=(principal_id,),
            affected_obligation_ids=node.obligation_ids,
        )

    def add_workstate_item(
        self, item: RouterWorkstateItem
    ) -> RouterWorkstateItem:
        self._assert_can_act(item.writer_principal_id)
        if item.item_id in self.workstate:
            raise ValueError(f"workstate item already exists: {item.item_id}")
        if item.owner_principal_id not in self.memberships:
            raise ValueError("workstate owner is not a Router member")
        unknown_obligations = set(item.obligation_ids) - set(self.obligations)
        if unknown_obligations:
            raise ValueError(
                "workstate item references unknown obligations: "
                + ",".join(sorted(unknown_obligations))
            )
        item = item.with_default_provenance()
        self.workstate[item.item_id] = item
        self._append(
            RouterLedgerEvent.WORKSTATE_CREATED,
            item.writer_principal_id,
            {
                "item_id": item.item_id,
                "owner_principal_id": item.owner_principal_id,
                "status": item.status.value,
                "provenance_sha256": item.provenance_sha256,
            },
            affected_principal_ids=(item.owner_principal_id,),
            affected_obligation_ids=item.obligation_ids,
            affected_dependency_ids=item.dependency_ids,
        )
        return item

    def open_workstate_dispute(
        self,
        *,
        dispute_id: str,
        actor_principal_id: str,
        existing_item_id: str,
        challenger_item: RouterWorkstateItem,
        reason: str,
    ) -> RouterWorkstateDispute:
        """Quarantine both sides of a conflict until independent verification."""

        self._assert_can_act(actor_principal_id)
        dispute_id = _required_text(dispute_id, "dispute_id")
        reason = _required_text(reason, "reason")
        if dispute_id in self.disputes:
            raise ValueError(f"workstate dispute already exists: {dispute_id}")
        try:
            existing = self.workstate[existing_item_id]
        except KeyError as error:
            raise KeyError(
                f"unknown workstate item: {existing_item_id}"
            ) from error
        if existing.status is not RouterWorkstateStatus.ACTIVE:
            raise RuntimeError("only an active workstate item may be disputed")
        if any(
            item.status is RouterDisputeStatus.OPEN
            and item.existing_item_id == existing_item_id
            for item in self.disputes.values()
        ):
            raise RuntimeError("workstate item already has an open dispute")
        if challenger_item.writer_principal_id != actor_principal_id:
            raise PermissionError("challenger writer does not match actor")
        if challenger_item.owner_principal_id != existing.owner_principal_id:
            raise ValueError("challenger must preserve the workstate owner")
        if challenger_item.supersedes_item_id != existing_item_id:
            raise ValueError("challenger must bind the existing item")
        if challenger_item.version <= existing.version:
            raise ValueError("challenger version must increase")
        if challenger_item.status is not RouterWorkstateStatus.ACTIVE:
            raise ValueError("new challenger must start as active input")

        challenger = self.add_workstate_item(
            replace(challenger_item, status=RouterWorkstateStatus.DISPUTED)
        )
        self.workstate[existing_item_id] = replace(
            existing, status=RouterWorkstateStatus.DISPUTED
        )
        dispute = RouterWorkstateDispute(
            dispute_id=dispute_id,
            existing_item_id=existing_item_id,
            challenger_item_id=challenger.item_id,
            opened_by_principal_id=actor_principal_id,
            reason=reason,
            opened_ledger_sequence=len(self.ledger),
        )
        self.disputes[dispute_id] = dispute
        self._append(
            RouterLedgerEvent.WORKSTATE_DISPUTED,
            actor_principal_id,
            dispute.record(),
            affected_principal_ids=(existing.owner_principal_id,),
            affected_obligation_ids=tuple(
                sorted(
                    set(existing.obligation_ids)
                    | set(challenger.obligation_ids)
                )
            ),
            affected_dependency_ids=tuple(
                sorted(
                    set(existing.dependency_ids)
                    | set(challenger.dependency_ids)
                )
            ),
        )
        return dispute

    def resolve_workstate_dispute(
        self,
        *,
        dispute_id: str,
        verifier_principal_id: str,
        accepted_item_id: str,
        verification_evidence_sha256: str,
        reason: str,
    ) -> RouterWorkstateDispute:
        """Resolve a conflict using evidence from an independent verifier."""

        try:
            dispute = self.disputes[dispute_id]
        except KeyError as error:
            raise KeyError(f"unknown workstate dispute: {dispute_id}") from error
        if dispute.status is not RouterDisputeStatus.OPEN:
            raise RuntimeError("workstate dispute is already resolved")
        verifier_principal_id = _required_text(
            verifier_principal_id, "verifier_principal_id"
        )
        if verifier_principal_id != "router":
            self._assert_can_act(verifier_principal_id)
        if verifier_principal_id == dispute.opened_by_principal_id:
            raise PermissionError("a challenger cannot verify its own update")
        candidates = {
            dispute.existing_item_id,
            dispute.challenger_item_id,
        }
        if accepted_item_id not in candidates:
            raise ValueError("accepted_item_id is not part of the dispute")
        verification_evidence_sha256 = _required_text(
            verification_evidence_sha256,
            "verification_evidence_sha256",
        )
        if len(verification_evidence_sha256) != 64:
            raise ValueError("verification evidence must be SHA-256")
        bytes.fromhex(verification_evidence_sha256)
        reason = _required_text(reason, "reason")

        rejected_item_id = next(
            item_id for item_id in candidates if item_id != accepted_item_id
        )
        existing = self.workstate[dispute.existing_item_id]
        challenger = self.workstate[dispute.challenger_item_id]
        if (
            existing.status is not RouterWorkstateStatus.DISPUTED
            or challenger.status is not RouterWorkstateStatus.DISPUTED
        ):
            raise RuntimeError("disputed workstate items changed before resolution")
        if accepted_item_id == dispute.existing_item_id:
            self.workstate[existing.item_id] = replace(
                existing, status=RouterWorkstateStatus.ACTIVE
            )
            self.workstate[challenger.item_id] = replace(
                challenger, status=RouterWorkstateStatus.REVOKED
            )
        else:
            self.workstate[existing.item_id] = replace(
                existing, status=RouterWorkstateStatus.SUPERSEDED
            )
            self.workstate[challenger.item_id] = replace(
                challenger, status=RouterWorkstateStatus.ACTIVE
            )
        resolved = replace(
            dispute,
            status=RouterDisputeStatus.RESOLVED,
            accepted_item_id=accepted_item_id,
            rejected_item_id=rejected_item_id,
            verifier_principal_id=verifier_principal_id,
            verification_evidence_sha256=verification_evidence_sha256,
            resolution_reason=reason,
        )
        self.disputes[dispute_id] = resolved
        self._append(
            RouterLedgerEvent.WORKSTATE_DISPUTE_RESOLVED,
            verifier_principal_id,
            resolved.record(),
            affected_principal_ids=(existing.owner_principal_id,),
            affected_obligation_ids=tuple(
                sorted(
                    set(existing.obligation_ids)
                    | set(challenger.obligation_ids)
                )
            ),
            affected_dependency_ids=tuple(
                sorted(
                    set(existing.dependency_ids)
                    | set(challenger.dependency_ids)
                )
            ),
        )
        return resolved

    def supersede_workstate_item(
        self,
        *,
        actor_principal_id: str,
        old_item_id: str,
        replacement_item: RouterWorkstateItem,
    ) -> RouterWorkstateItem:
        self._assert_can_act(actor_principal_id)
        try:
            old = self.workstate[old_item_id]
        except KeyError as error:
            raise KeyError(f"unknown workstate item: {old_item_id}") from error
        if old.status is not RouterWorkstateStatus.ACTIVE:
            raise RuntimeError("only an active workstate item can be superseded")
        if replacement_item.writer_principal_id != actor_principal_id:
            raise PermissionError("replacement writer does not match actor")
        if replacement_item.supersedes_item_id != old_item_id:
            raise ValueError("replacement must bind supersedes_item_id")
        if replacement_item.version <= old.version:
            raise ValueError("replacement version must increase")
        self.workstate[old_item_id] = replace(
            old, status=RouterWorkstateStatus.SUPERSEDED
        )
        replacement = self.add_workstate_item(replacement_item)
        self._append(
            RouterLedgerEvent.WORKSTATE_SUPERSEDED,
            actor_principal_id,
            {
                "old_item_id": old_item_id,
                "replacement_item_id": replacement.item_id,
            },
            affected_principal_ids=(old.owner_principal_id,),
            affected_obligation_ids=tuple(
                sorted(set(old.obligation_ids) | set(replacement.obligation_ids))
            ),
            affected_dependency_ids=tuple(
                sorted(set(old.dependency_ids) | set(replacement.dependency_ids))
            ),
        )
        return replacement

    def revoke_workstate_item(
        self,
        *,
        actor_principal_id: str,
        item_id: str,
        reason: str,
    ) -> None:
        self._assert_can_act(actor_principal_id)
        try:
            item = self.workstate[item_id]
        except KeyError as error:
            raise KeyError(f"unknown workstate item: {item_id}") from error
        if item.status is not RouterWorkstateStatus.ACTIVE:
            raise RuntimeError("only an active workstate item can be revoked")
        self.workstate[item_id] = replace(
            item, status=RouterWorkstateStatus.REVOKED
        )
        self._append(
            RouterLedgerEvent.WORKSTATE_REVOKED,
            actor_principal_id,
            {"item_id": item_id, "reason": _required_text(reason, "reason")},
            affected_principal_ids=(item.owner_principal_id,),
            affected_obligation_ids=item.obligation_ids,
            affected_dependency_ids=item.dependency_ids,
        )

    def update_obligation(
        self,
        *,
        actor_principal_id: str,
        obligation_id: str,
        status: RouterObligationStatus,
        description: str | None = None,
    ) -> RouterObligation:
        self._assert_can_act(actor_principal_id)
        try:
            prior = self.obligations[obligation_id]
        except KeyError as error:
            raise KeyError(f"unknown obligation: {obligation_id}") from error
        updated = replace(
            prior,
            status=status,
            description=(
                _required_text(description, "description")
                if description is not None
                else prior.description
            ),
        )
        self.obligations[obligation_id] = updated
        self._append(
            RouterLedgerEvent.OBLIGATION_UPDATED,
            actor_principal_id,
            {
                "obligation_id": obligation_id,
                "old_status": prior.status.value,
                "new_status": updated.status.value,
                "description_sha256": canonical_sha256(updated.description),
            },
            affected_principal_ids=(prior.owner_principal_id,),
            affected_obligation_ids=(obligation_id,),
            affected_dependency_ids=prior.required_dependency_ids,
        )
        return updated

    def refine_obligation_dependencies(
        self,
        *,
        actor_principal_id: str,
        obligation_id: str,
        required_dependency_ids: Iterable[object],
        critical_dependency_ids: Iterable[object],
        reason: str,
    ) -> RouterObligation:
        """Replace a live obligation's predicted dependency frontier.

        Dynamic RETURN adapters initially open an obligation before all
        absence-period evidence exists.  Once the Router has observed that
        evidence, it may refine the dependency frontier without rebuilding the
        task or mutating the immutable departure snapshot.
        """

        if actor_principal_id != "router":
            self._assert_can_act(actor_principal_id)
        try:
            prior = self.obligations[obligation_id]
        except KeyError as error:
            raise KeyError(f"unknown obligation: {obligation_id}") from error
        required = _unique_text(
            required_dependency_ids, "required_dependency_ids"
        )
        critical = _unique_text(
            critical_dependency_ids, "critical_dependency_ids"
        )
        if not required:
            raise ValueError("required_dependency_ids must be non-empty")
        if not set(critical).issubset(required):
            raise ValueError(
                "critical_dependency_ids must be a subset of required dependencies"
            )
        updated = replace(
            prior,
            required_dependency_ids=required,
            critical_dependency_ids=critical,
        )
        self.obligations[obligation_id] = updated
        affected = tuple(
            dict.fromkeys(
                (
                    *prior.required_dependency_ids,
                    *prior.critical_dependency_ids,
                    *required,
                    *critical,
                )
            )
        )
        self._append(
            RouterLedgerEvent.OBLIGATION_UPDATED,
            actor_principal_id,
            {
                "obligation_id": obligation_id,
                "old_required_dependency_ids": list(
                    prior.required_dependency_ids
                ),
                "new_required_dependency_ids": list(required),
                "old_critical_dependency_ids": list(
                    prior.critical_dependency_ids
                ),
                "new_critical_dependency_ids": list(critical),
                "reason": _required_text(reason, "reason"),
            },
            affected_principal_ids=(prior.owner_principal_id,),
            affected_obligation_ids=(obligation_id,),
            affected_dependency_ids=affected,
        )
        return updated

    def depart(self, principal_id: str) -> RouterDepartureCheckpoint:
        self._assert_can_act(principal_id)
        if self.dag is None:
            raise RuntimeError("router plan is not installed")
        if principal_id in self.departures:
            raise RuntimeError("member already has an open absence")
        completed = tuple(
            node.node_id
            for node in self.dag.nodes
            if node.assigned_principal_id == principal_id
            and node.status is RouterNodeStatus.COMPLETED
        )
        owned_items = tuple(
            item
            for item in self.workstate.values()
            if item.owner_principal_id == principal_id
        )
        if not completed or not owned_items:
            raise RuntimeError(
                "RETURN experiment requires completed pre-absence work and "
                "a materialized workstate item"
            )
        member = self._member(principal_id)
        self.memberships[principal_id] = replace(
            member,
            epoch=member.epoch + 1,
            status=RouterMembershipStatus.ABSENT,
        )
        entry = self._append(
            RouterLedgerEvent.MEMBER_DEPARTED,
            "router",
            {
                "principal_id": principal_id,
                "source_epoch": member.epoch,
                "target_epoch": member.epoch + 1,
            },
            affected_principal_ids=(principal_id,),
            affected_obligation_ids=tuple(
                sorted(
                    obligation.obligation_id
                    for obligation in self.obligations.values()
                    if obligation.owner_principal_id == principal_id
                    and obligation.status is RouterObligationStatus.OPEN
                )
            ),
        )
        unsigned = {
            "task_id": self.task_id,
            "principal_id": principal_id,
            "source_epoch": member.epoch,
            "departure_ledger_sequence": entry.sequence,
            "completed_node_ids": list(completed),
            "open_obligation_ids": sorted(
                obligation.obligation_id
                for obligation in self.obligations.values()
                if obligation.owner_principal_id == principal_id
                and obligation.status is RouterObligationStatus.OPEN
            ),
            "workstate_items": [item.record() for item in owned_items],
            "dag_digest": self.dag.digest,
            "ledger_head_sha256": self.ledger_head_sha256,
        }
        checkpoint = RouterDepartureCheckpoint(
            task_id=self.task_id,
            principal_id=principal_id,
            source_epoch=member.epoch,
            departure_ledger_sequence=entry.sequence,
            completed_node_ids=completed,
            open_obligation_ids=tuple(unsigned["open_obligation_ids"]),
            workstate_items=owned_items,
            dag_digest=self.dag.digest,
            ledger_head_sha256=self.ledger_head_sha256,
            checkpoint_sha256=canonical_sha256(unsigned),
        )
        self.departures[principal_id] = checkpoint
        return checkpoint

    def build_absence_delta(
        self, principal_id: str
    ) -> RouterAbsenceDelta:
        try:
            departure = self.departures[principal_id]
        except KeyError as error:
            raise RuntimeError("member has no open absence") from error
        entries = tuple(
            entry
            for entry in self.ledger
            if entry.sequence > departure.departure_ledger_sequence
            and entry.event
            not in {
                RouterLedgerEvent.RETURN_STARTED,
                RouterLedgerEvent.RETURN_ADMITTED,
                RouterLedgerEvent.RETURN_BLOCKED,
                RouterLedgerEvent.CHECKPOINT_FROZEN,
            }
        )
        departure_items = {
            item.item_id: item for item in departure.workstate_items
        }
        relevant_obligations = set(departure.open_obligation_ids)
        relevant_dependencies = {
            dependency_id
            for obligation_id in relevant_obligations
            for dependency_id in self.obligations[
                obligation_id
            ].required_dependency_ids
        }
        relevant_sequences: list[int] = []
        for entry in entries:
            payload_ids = {
                str(entry.payload.get("item_id") or ""),
                str(entry.payload.get("old_item_id") or ""),
                str(entry.payload.get("replacement_item_id") or ""),
                str(entry.payload.get("existing_item_id") or ""),
                str(entry.payload.get("challenger_item_id") or ""),
                str(entry.payload.get("accepted_item_id") or ""),
                str(entry.payload.get("rejected_item_id") or ""),
            }
            if (
                principal_id in entry.affected_principal_ids
                or relevant_obligations.intersection(
                    entry.affected_obligation_ids
                )
                or relevant_dependencies.intersection(
                    entry.affected_dependency_ids
                )
                or set(departure_items).intersection(payload_ids)
            ):
                relevant_sequences.append(entry.sequence)
        created = tuple(
            str(entry.payload["item_id"])
            for entry in entries
            if entry.event is RouterLedgerEvent.WORKSTATE_CREATED
            and entry.payload.get("item_id")
        )
        superseded = tuple(
            str(entry.payload["old_item_id"])
            for entry in entries
            if entry.event is RouterLedgerEvent.WORKSTATE_SUPERSEDED
            and entry.payload.get("old_item_id")
        )
        revoked = tuple(
            str(entry.payload["item_id"])
            for entry in entries
            if entry.event is RouterLedgerEvent.WORKSTATE_REVOKED
            and entry.payload.get("item_id")
        )
        disputed = tuple(
            str(item_id)
            for entry in entries
            if entry.event is RouterLedgerEvent.WORKSTATE_DISPUTED
            for item_id in (
                entry.payload.get("existing_item_id"),
                entry.payload.get("challenger_item_id"),
            )
            if item_id
        )
        resolved_disputes = tuple(
            str(entry.payload["dispute_id"])
            for entry in entries
            if entry.event
            is RouterLedgerEvent.WORKSTATE_DISPUTE_RESOLVED
            and entry.payload.get("dispute_id")
        )
        changed_obligations = tuple(
            sorted(
                {
                    obligation_id
                    for entry in entries
                    if entry.event
                    in {
                        RouterLedgerEvent.OBLIGATION_OPENED,
                        RouterLedgerEvent.OBLIGATION_UPDATED,
                    }
                    for obligation_id in entry.affected_obligation_ids
                }
            )
        )
        nonowner_actions = sum(
            entry.actor_principal_id not in {"router", principal_id}
            and entry.event
            in {
                RouterLedgerEvent.NODE_STARTED,
                RouterLedgerEvent.NODE_COMPLETED,
                RouterLedgerEvent.WORKSTATE_CREATED,
                RouterLedgerEvent.WORKSTATE_DISPUTED,
                RouterLedgerEvent.WORKSTATE_DISPUTE_RESOLVED,
                RouterLedgerEvent.WORKSTATE_SUPERSEDED,
                RouterLedgerEvent.WORKSTATE_REVOKED,
                RouterLedgerEvent.OBLIGATION_UPDATED,
            }
            for entry in entries
        )
        return RouterAbsenceDelta(
            principal_id=principal_id,
            ledger_entries=entries,
            created_item_ids=created,
            superseded_item_ids=superseded,
            revoked_item_ids=revoked,
            changed_obligation_ids=changed_obligations,
            relevant_entry_sequences=tuple(relevant_sequences),
            nonowner_action_count=nonowner_actions,
            disputed_item_ids=tuple(dict.fromkeys(disputed)),
            resolved_dispute_ids=tuple(dict.fromkeys(resolved_disputes)),
        )

    def freeze_post_absence_checkpoint(
        self,
        principal_id: str,
        *,
        require_relevant_delta: bool = True,
    ) -> RouterPostAbsenceCheckpoint:
        if self.dag is None:
            raise RuntimeError("router plan is not installed")
        member = self._member(principal_id)
        if member.status is not RouterMembershipStatus.ABSENT:
            raise RuntimeError("post-absence checkpoint requires an absent member")
        departure = self.departures[principal_id]
        delta = self.build_absence_delta(principal_id)
        if delta.nonowner_action_count < 1:
            raise RuntimeError(
                "post-absence checkpoint requires work by another agent"
            )
        if require_relevant_delta and delta.relevant_change_count < 1:
            raise RuntimeError(
                "post-absence checkpoint requires a semantically relevant "
                "absence delta"
            )
        unsigned = {
            "schema_version": ROUTER_CHECKPOINT_SCHEMA,
            "task_id": self.task_id,
            "returning_principal_id": principal_id,
            "target_epoch": member.epoch,
            "departure": departure.record(),
            "absence_delta": delta.record(),
            "dag": self.dag.record(),
            "memberships": [
                item.record()
                for item in sorted(
                    self.memberships.values(),
                    key=lambda value: value.principal_id,
                )
            ],
            "obligations": [
                item.record()
                for item in sorted(
                    self.obligations.values(),
                    key=lambda value: value.obligation_id,
                )
            ],
            "current_workstate_items": [
                item.record()
                for item in sorted(
                    self.workstate.values(),
                    key=lambda value: value.item_id,
                )
            ],
            "disputes": [
                item.record()
                for item in sorted(
                    self.disputes.values(),
                    key=lambda value: value.dispute_id,
                )
            ],
            "ledger_head_sha256": self.ledger_head_sha256,
        }
        checkpoint = RouterPostAbsenceCheckpoint(
            task_id=self.task_id,
            returning_principal_id=principal_id,
            target_epoch=member.epoch,
            departure=departure,
            absence_delta=delta,
            dag=self.dag,
            memberships=tuple(
                sorted(
                    self.memberships.values(),
                    key=lambda value: value.principal_id,
                )
            ),
            obligations=tuple(
                sorted(
                    self.obligations.values(),
                    key=lambda value: value.obligation_id,
                )
            ),
            current_workstate_items=tuple(
                sorted(
                    self.workstate.values(),
                    key=lambda value: value.item_id,
                )
            ),
            disputes=tuple(
                sorted(
                    self.disputes.values(),
                    key=lambda value: value.dispute_id,
                )
            ),
            ledger_head_sha256=self.ledger_head_sha256,
            checkpoint_sha256=canonical_sha256(unsigned),
        )
        self.frozen_checkpoints[principal_id] = checkpoint
        self._append(
            RouterLedgerEvent.CHECKPOINT_FROZEN,
            "router",
            {
                "principal_id": principal_id,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "absence_delta_relevant_changes": delta.relevant_change_count,
            },
            affected_principal_ids=(principal_id,),
            affected_obligation_ids=departure.open_obligation_ids,
        )
        return checkpoint

    def fork_return_methods(
        self,
        checkpoint: RouterPostAbsenceCheckpoint,
        *,
        trace_selected_item_ids: Sequence[str] | None = None,
        trace_block_reason: str | None = None,
    ) -> tuple[RouterReturnFork, ...]:
        """Materialize raw and compiler-backed views without mutation."""

        if (
            self.frozen_checkpoints.get(
                checkpoint.returning_principal_id
            )
            != checkpoint
        ):
            raise ValueError("checkpoint was not frozen by this Router")
        old_ids = tuple(
            item.item_id for item in checkpoint.departure.workstate_items
        )
        current_active_ids = tuple(
            item.item_id
            for item in checkpoint.current_workstate_items
            if item.status is RouterWorkstateStatus.ACTIVE
        )
        full_merge_ids = tuple(
            dict.fromkeys(
                old_ids
                + tuple(
                    item.item_id
                    for item in checkpoint.current_workstate_items
                    if item.item_id not in old_ids
                )
            )
        )
        candidate_ids = full_merge_ids
        if trace_selected_item_ids is None:
            trace_ids: tuple[str, ...] = ()
            compiler_required = True
        else:
            trace_ids = _unique_text(
                trace_selected_item_ids, "trace_selected_item_ids"
            )
            unknown = set(trace_ids) - set(candidate_ids)
            if unknown:
                raise ValueError(
                    "TRACE selected items outside frozen candidates: "
                    + ",".join(sorted(unknown))
                )
            compiler_required = False
        common = {
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "returning_principal_id": checkpoint.returning_principal_id,
            "candidate_item_ids": candidate_ids,
        }
        return (
            RouterReturnFork(
                arm=RouterReturnArm.STATIC,
                inherited_item_ids=current_active_ids,
                **common,
            ),
            RouterReturnFork(
                arm=RouterReturnArm.STATIC_COMPACT,
                inherited_item_ids=(),
                compiler_required=True,
                **common,
            ),
            RouterReturnFork(
                arm=RouterReturnArm.COMPACT_ONLY,
                inherited_item_ids=(),
                compiler_required=True,
                **common,
            ),
            RouterReturnFork(
                arm=RouterReturnArm.RESET,
                inherited_item_ids=(),
                **common,
            ),
            RouterReturnFork(
                arm=RouterReturnArm.RESET_REBRIEF,
                inherited_item_ids=(),
                compiler_required=True,
                **common,
            ),
            RouterReturnFork(
                arm=RouterReturnArm.RESTORE_OLD,
                inherited_item_ids=old_ids,
                **common,
            ),
            RouterReturnFork(
                arm=RouterReturnArm.VALIDITY_FILTER_ONLY,
                inherited_item_ids=(),
                compiler_required=True,
                **common,
            ),
            RouterReturnFork(
                arm=RouterReturnArm.FULL_MERGE,
                inherited_item_ids=full_merge_ids,
                **common,
            ),
            RouterReturnFork(
                arm=RouterReturnArm.TRACE,
                inherited_item_ids=trace_ids,
                compiler_required=compiler_required,
                blocked=trace_block_reason is not None,
                block_reason=trace_block_reason,
                **common,
            ),
        )

    def start_return(self, principal_id: str) -> None:
        member = self._member(principal_id)
        if member.status is not RouterMembershipStatus.ABSENT:
            raise RuntimeError("only an absent member may start RETURN")
        if principal_id not in self.frozen_checkpoints:
            raise RuntimeError("RETURN requires a frozen post-absence checkpoint")
        self.memberships[principal_id] = replace(
            member, status=RouterMembershipStatus.RETURNING
        )
        self._append(
            RouterLedgerEvent.RETURN_STARTED,
            "router",
            {
                "principal_id": principal_id,
                "checkpoint_sha256": self.frozen_checkpoints[
                    principal_id
                ].checkpoint_sha256,
            },
            affected_principal_ids=(principal_id,),
        )

    def finish_return(
        self,
        principal_id: str,
        *,
        admitted: bool,
        reason: str,
    ) -> None:
        member = self._member(principal_id)
        if member.status is not RouterMembershipStatus.RETURNING:
            raise RuntimeError("member is not currently returning")
        status = (
            RouterMembershipStatus.ADMITTED
            if admitted
            else RouterMembershipStatus.BLOCKED
        )
        self.memberships[principal_id] = replace(member, status=status)
        self._append(
            (
                RouterLedgerEvent.RETURN_ADMITTED
                if admitted
                else RouterLedgerEvent.RETURN_BLOCKED
            ),
            "router",
            {
                "principal_id": principal_id,
                "reason": _required_text(reason, "reason"),
            },
            affected_principal_ids=(principal_id,),
        )

    def record(self) -> dict[str, object]:
        return {
            "schema_version": ROUTER_RETURN_PROTOCOL,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "dag": self.dag.record() if self.dag else None,
            "memberships": [
                item.record()
                for item in sorted(
                    self.memberships.values(),
                    key=lambda value: value.principal_id,
                )
            ],
            "obligations": [
                item.record()
                for item in sorted(
                    self.obligations.values(),
                    key=lambda value: value.obligation_id,
                )
            ],
            "workstate": [
                item.record()
                for item in sorted(
                    self.workstate.values(),
                    key=lambda value: value.item_id,
                )
            ],
            "disputes": [
                item.record()
                for item in sorted(
                    self.disputes.values(),
                    key=lambda value: value.dispute_id,
                )
            ],
            "ledger": [entry.record() for entry in self.ledger],
            "departure_checkpoints": {
                key: value.record()
                for key, value in sorted(self.departures.items())
            },
            "post_absence_checkpoints": {
                key: value.record()
                for key, value in sorted(self.frozen_checkpoints.items())
            },
        }


__all__ = [
    "ROUTER_CHECKPOINT_SCHEMA",
    "ROUTER_RETURN_PROTOCOL",
    "ROUTER_TASK_DAG_SCHEMA",
    "RouterAbsenceDelta",
    "RouterDepartureCheckpoint",
    "RouterDisputeStatus",
    "RouterLedgerEntry",
    "RouterLedgerEvent",
    "RouterMembership",
    "RouterMembershipStatus",
    "RouterNodeStatus",
    "RouterObligation",
    "RouterObligationStatus",
    "RouterPostAbsenceCheckpoint",
    "RouterReturnArm",
    "RouterReturnFork",
    "RouterReturnProtocol",
    "RouterTaskDAG",
    "RouterTaskNode",
    "RouterWorkstateItem",
    "RouterWorkstateDispute",
    "RouterWorkstateStatus",
    "canonical_json",
    "canonical_sha256",
    "parse_router_plan",
    "router_decomposition_prompt",
]
