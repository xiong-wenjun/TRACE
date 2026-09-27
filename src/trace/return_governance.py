"""Item-level memory governance for an authenticated agent RETURN.

RETURN is the only externally visible event in the current protocol. It is
implemented as one absence cycle:

1. remove the authenticated workload instance from its role;
2. allow the team and task to evolve while the role is absent;
3. add a fresh workload instance for the same principal back to that role.

The objects in this module carry *role workstate*, not a workflow or an action
plan.  Safety eligibility is evaluated per item.  Minimal disclosure is a
set-level optimization: after filtering, an obligation-aware selector chooses
the lowest-cost bounded view that covers the current role's declared
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import hmac
import json
from typing import Iterable, Mapping, Sequence


RETURN_VIEW_SCHEMA = "return_memory_view_v2"
RETURN_ADMISSION_SCHEMA = "return_memory_view_admission_v2"
RETURN_COMPILATION_SCHEMA = "return_memory_compilation_v1"
RETURN_SELECTION_SCHEMA = "return_memory_selection_v1"
RETURN_MEMORY_SOURCE = "verified_signed_return_memory_item"
MAX_RETURN_ITEM_CHARS = 4096


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be non-empty normalized text")
    return value


def _digest(value: object, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or text != text.lower():
        raise ValueError(f"{field} must be a lowercase SHA-256")
    try:
        bytes.fromhex(text)
    except ValueError as error:
        raise ValueError(f"{field} must be a lowercase SHA-256") from error
    return text


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def stable_obligation_id(task_id: str, role_id: str, obligation_key: str) -> str:
    """Derive a stable, non-semantic identifier for one open obligation."""

    return "obl_" + canonical_sha256(
        {
            "task_id": _text(task_id, "task_id"),
            "role_id": _text(role_id, "role_id"),
            "obligation_key": _text(obligation_key, "obligation_key"),
        }
    )[:24]


def stable_dependency_id(obligation_id: str, dependency_key: str) -> str:
    """Derive a stable identifier for one declarative obligation dependency."""

    return "dep_" + canonical_sha256(
        {
            "obligation_id": _text(obligation_id, "obligation_id"),
            "dependency_key": _text(dependency_key, "dependency_key"),
        }
    )[:24]


class ReturnMemoryKind(str, Enum):
    """The only workstate item kinds admissible after RETURN."""

    VERIFIED_FACT = "verified_fact"
    PROGRESS_FRONTIER = "progress_frontier"
    OPEN_OBLIGATION = "open_obligation"
    ROLE_CONSTRAINT = "role_constraint"
    INTERFACE_CONSTRAINT = "interface_constraint"


class ReturnMemoryScope(str, Enum):
    ROLE = "role"
    TEAM_PUBLIC = "team_public"


class ReturnIncompleteAction(str, Enum):
    """Fail-closed behavior when no safe sufficient view fits the budget."""

    RESET = "reset"
    BLOCK = "blocked"


class ReturnSelectionStatus(str, Enum):
    ISSUED = "issued"
    RESET_FALLBACK = "reset_fallback"
    BLOCKED = "blocked"


class ReturnAblationMode(str, Enum):
    """Frozen method variants used by RETURN ablation panels.

    ``TRACE`` remains the production/default path.  The other values are
    explicit experimental controls; weakened controls are recorded inside the
    signed view so that they cannot be confused with a production admission.
    """

    TRACE = "trace"
    FULL_RESTORE = "full_restore"
    VALIDITY_FILTER_ONLY = "validity_filter_only"
    TRACE_WITHOUT_FALLBACK = "trace_without_fallback"
    TRACE_WITHOUT_PROVENANCE = "trace_without_provenance"
    TRACE_WITHOUT_FRESHNESS = "trace_without_freshness"
    STATIC_COMPACT = "static_compact"
    RESET_REBRIEF = "reset_rebrief"

    @classmethod
    def _missing_(cls, value: object) -> "ReturnAblationMode | None":
        # Historical ablation values are normalized only when old records are
        # parsed. New compilations always serialize the TRACE spelling.
        from .trace_method import canonical_trace_mode

        canonical = canonical_trace_mode(value)
        if canonical != str(value).strip():
            return cls(canonical)
        return None


_FORBIDDEN_ORACLE_KEYS = frozenset(
    {
        "action_sequence",
        "actions",
        "complete_workflow",
        "expert_plan",
        "next_action",
        "plan_steps",
        "workflow",
        "workflow_steps",
    }
)


def _forbidden_oracle_paths(
    value: object,
    *,
    path: str = "attributes",
) -> tuple[str, ...]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key) in _FORBIDDEN_ORACLE_KEYS:
                paths.append(child)
            paths.extend(_forbidden_oracle_paths(item, path=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(
                _forbidden_oracle_paths(item, path=f"{path}[{index}]")
            )
    return tuple(paths)


@dataclass(frozen=True)
class ReturnObligationSpec:
    """A stable obligation and the declarative evidence it requires."""

    obligation_id: str
    required_dependency_ids: tuple[str, ...]
    critical_dependency_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.obligation_id, "obligation_id")
        if not self.required_dependency_ids:
            raise ValueError("an open obligation requires at least one dependency")
        for field, values in (
            ("required_dependency_id", self.required_dependency_ids),
            ("critical_dependency_id", self.critical_dependency_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field}s must be unique")
            for value in values:
                _text(value, field)
        if not set(self.critical_dependency_ids).issubset(
            self.required_dependency_ids
        ):
            raise ValueError("critical dependencies must also be required")

    def record(self) -> dict[str, object]:
        return {
            "obligation_id": self.obligation_id,
            "required_dependency_ids": list(self.required_dependency_ids),
            "critical_dependency_ids": list(self.critical_dependency_ids),
        }


@dataclass(frozen=True)
class ReturnProvenanceReceipt:
    """Authority-approved binding between an item and graph references.

    The outer signed return view authenticates this receipt.  The receipt
    digest prevents a valid source digest from being reused for a different
    item or for different obligation/dependency edges.
    """

    item_id: str
    source_sha256: str
    bound_reference_ids: tuple[str, ...]
    issuer_id: str
    issued_epoch: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        _text(self.item_id, "receipt.item_id")
        _digest(self.source_sha256, "receipt.source_sha256")
        _text(self.issuer_id, "receipt.issuer_id")
        if self.issued_epoch < 0:
            raise ValueError("receipt issued_epoch must be non-negative")
        if len(self.bound_reference_ids) != len(set(self.bound_reference_ids)):
            raise ValueError("receipt reference ids must be unique")
        for value in self.bound_reference_ids:
            _text(value, "receipt.bound_reference_id")
        _digest(self.receipt_sha256, "receipt.receipt_sha256")
        if self.receipt_sha256 != canonical_sha256(self.unsigned_record()):
            raise ValueError("provenance receipt digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        item_id: str,
        source_sha256: str,
        bound_reference_ids: Sequence[str],
        issuer_id: str,
        issued_epoch: int,
    ) -> "ReturnProvenanceReceipt":
        record = {
            "item_id": _text(item_id, "receipt.item_id"),
            "source_sha256": _digest(
                source_sha256, "receipt.source_sha256"
            ),
            "bound_reference_ids": sorted(
                {
                    _text(value, "receipt.bound_reference_id")
                    for value in bound_reference_ids
                }
            ),
            "issuer_id": _text(issuer_id, "receipt.issuer_id"),
            "issued_epoch": issued_epoch,
        }
        if issued_epoch < 0:
            raise ValueError("receipt issued_epoch must be non-negative")
        return cls(
            item_id=str(record["item_id"]),
            source_sha256=str(record["source_sha256"]),
            bound_reference_ids=tuple(record["bound_reference_ids"]),
            issuer_id=str(record["issuer_id"]),
            issued_epoch=issued_epoch,
            receipt_sha256=canonical_sha256(record),
        )

    def unsigned_record(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "source_sha256": self.source_sha256,
            "bound_reference_ids": list(self.bound_reference_ids),
            "issuer_id": self.issuer_id,
            "issued_epoch": self.issued_epoch,
        }

    def record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class ReturnMemoryItem:
    """One bounded, non-private workstate statement."""

    item_id: str
    kind: ReturnMemoryKind
    text: str
    task_id: str
    role_id: str
    writer_principal_id: str
    source_epoch: int
    valid_from_epoch: int
    valid_until_epoch: int
    purposes: tuple[str, ...]
    provenance_sha256: tuple[str, ...]
    scope: ReturnMemoryScope = ReturnMemoryScope.ROLE
    superseded_by: str | None = None
    obligation_id: str | None = None
    supports: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    provenance_receipts: tuple[ReturnProvenanceReceipt, ...] = ()
    attributes: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for field in (
            "item_id",
            "text",
            "task_id",
            "role_id",
            "writer_principal_id",
        ):
            _text(getattr(self, field), field)
        if not isinstance(self.kind, ReturnMemoryKind):
            raise TypeError("kind must be ReturnMemoryKind")
        if len(self.text) > MAX_RETURN_ITEM_CHARS:
            raise ValueError(
                "one return memory item exceeds the atomic disclosure limit"
            )
        if not isinstance(self.scope, ReturnMemoryScope):
            raise TypeError("scope must be ReturnMemoryScope")
        if min(self.source_epoch, self.valid_from_epoch, self.valid_until_epoch) < 0:
            raise ValueError("epochs must be non-negative")
        if self.valid_from_epoch > self.valid_until_epoch:
            raise ValueError("validity interval is empty")
        if not self.purposes:
            raise ValueError("at least one purpose is required")
        for purpose in self.purposes:
            _text(purpose, "purpose")
        if not self.provenance_sha256:
            raise ValueError("at least one provenance digest is required")
        for index, digest in enumerate(self.provenance_sha256):
            _digest(digest, f"provenance_sha256[{index}]")
        if self.superseded_by is not None:
            _text(self.superseded_by, "superseded_by")
        for field, values in (
            ("supports", self.supports),
            ("depends_on", self.depends_on),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field} references must be unique")
            for value in values:
                _text(value, f"{field} reference")
        if self.kind is ReturnMemoryKind.OPEN_OBLIGATION:
            if self.obligation_id is None:
                raise ValueError(
                    "open_obligation items require a stable obligation_id"
                )
            _text(self.obligation_id, "obligation_id")
        elif self.obligation_id is not None:
            raise ValueError(
                "only open_obligation items may declare obligation_id"
            )
        if (
            self.kind is ReturnMemoryKind.PROGRESS_FRONTIER
            and not self.depends_on
        ):
            raise ValueError(
                "progress_frontier must point to an open obligation"
            )
        for receipt in self.provenance_receipts:
            if not isinstance(receipt, ReturnProvenanceReceipt):
                raise TypeError(
                    "provenance_receipts must contain ReturnProvenanceReceipt"
                )
        attributes = dict(self.attributes or {})
        forbidden = sorted(_forbidden_oracle_paths(attributes))
        if forbidden:
            raise ValueError(
                "return memory items cannot encode a workflow oracle: "
                + ",".join(forbidden)
            )

    def record(self, *, include_text: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "item_id": self.item_id,
            "kind": self.kind.value,
            "task_id": self.task_id,
            "role_id": self.role_id,
            "writer_principal_id": self.writer_principal_id,
            "source_epoch": self.source_epoch,
            "valid_from_epoch": self.valid_from_epoch,
            "valid_until_epoch": self.valid_until_epoch,
            "purposes": list(self.purposes),
            "provenance_sha256": list(self.provenance_sha256),
            "scope": self.scope.value,
            "superseded_by": self.superseded_by,
            "obligation_id": self.obligation_id,
            "supports": list(self.supports),
            "depends_on": list(self.depends_on),
            "provenance_receipts": [
                receipt.record() for receipt in self.provenance_receipts
            ],
            "attributes": dict(self.attributes or {}),
            "content_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
        }
        if include_text:
            record["text"] = self.text
        return record

    def memory_metadata(
        self,
        *,
        view_id: str,
        signature_verified: bool,
    ) -> dict[str, object]:
        """Metadata used by retrieval-time validity checks."""

        return {
            "return_view_id": view_id,
            "return_item_id": self.item_id,
            "return_memory_kind": self.kind.value,
            "return_role_id": self.role_id,
            "return_purposes": list(self.purposes),
            "return_valid_from_epoch": self.valid_from_epoch,
            "return_valid_until_epoch": self.valid_until_epoch,
            "return_provenance_sha256": list(self.provenance_sha256),
            "return_scope": self.scope.value,
            "return_superseded_by": self.superseded_by,
            "return_obligation_id": self.obligation_id,
            "return_supports": list(self.supports),
            "return_depends_on": list(self.depends_on),
            "return_provenance_receipts": [
                receipt.record() for receipt in self.provenance_receipts
            ],
            "return_item_checks": {
                "authorization": True,
                "freshness": True,
                "applicability": True,
                "provenance": True,
                "individual_budget_eligible": True,
            },
            "signature_verified": bool(signature_verified),
        }

    @property
    def graph_reference_ids(self) -> frozenset[str]:
        values = set(self.supports) | set(self.depends_on)
        if self.obligation_id is not None:
            values.add(self.obligation_id)
        return frozenset(values)

    @property
    def disclosure_cost(self) -> int:
        """Deterministic disclosure cost used by the greedy selector."""

        return max(1, len(self.text))


@dataclass(frozen=True)
class ReturnReadmissionContext:
    return_event_id: str
    absence_id: str
    task_id: str
    role_id: str
    principal_id: str
    target_epoch: int
    iteration: int
    purpose: str = "task_execution"
    max_items: int = 8
    max_chars: int = 4096
    allowed_kinds: frozenset[ReturnMemoryKind] = frozenset(ReturnMemoryKind)
    obligations: tuple[ReturnObligationSpec, ...] = ()
    minimum_obligation_coverage: float = 1.0
    minimum_critical_dependency_recall: float = 1.0
    incomplete_action: ReturnIncompleteAction = ReturnIncompleteAction.RESET

    def __post_init__(self) -> None:
        for field in (
            "return_event_id",
            "absence_id",
            "task_id",
            "role_id",
            "principal_id",
            "purpose",
        ):
            _text(getattr(self, field), field)
        if self.target_epoch < 1 or self.iteration < 0:
            raise ValueError("target_epoch must be positive and iteration non-negative")
        if self.max_items < 1 or self.max_chars < 1:
            raise ValueError("minimal-disclosure budgets must be positive")
        if not self.allowed_kinds:
            raise ValueError("at least one return memory kind must be allowed")
        if not 0.0 <= self.minimum_obligation_coverage <= 1.0:
            raise ValueError("minimum_obligation_coverage must be within [0, 1]")
        if not 0.0 <= self.minimum_critical_dependency_recall <= 1.0:
            raise ValueError(
                "minimum_critical_dependency_recall must be within [0, 1]"
            )
        if not isinstance(self.incomplete_action, ReturnIncompleteAction):
            raise TypeError("incomplete_action must be ReturnIncompleteAction")
        obligation_ids = [item.obligation_id for item in self.obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("return obligation ids must be unique")
        if any(
            not isinstance(item, ReturnObligationSpec)
            for item in self.obligations
        ):
            raise TypeError("obligations must contain ReturnObligationSpec")


@dataclass(frozen=True)
class ReturnItemDecision:
    item_id: str
    admitted: bool
    authorization: bool
    freshness: bool
    applicability: bool
    provenance: bool
    individual_budget_eligible: bool
    reasons: tuple[str, ...]
    bypassed_checks: tuple[str, ...] = ()

    def record(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "admitted": self.admitted,
            "checks": {
                "authorization": self.authorization,
                "freshness": self.freshness,
                "applicability": self.applicability,
                "provenance": self.provenance,
                "individual_budget_eligible": (
                    self.individual_budget_eligible
                ),
            },
            "reasons": list(self.reasons),
            "bypassed_checks": list(self.bypassed_checks),
        }


@dataclass(frozen=True)
class ReturnObligationGraph:
    """Auditable bipartite graph from obligations to eligible memory items."""

    obligations: tuple[ReturnObligationSpec, ...]
    edges: tuple[tuple[str, str, str], ...]

    @classmethod
    def build(
        cls,
        obligations: Sequence[ReturnObligationSpec],
        items: Sequence[ReturnMemoryItem],
    ) -> "ReturnObligationGraph":
        edges: list[tuple[str, str, str]] = []
        for obligation in obligations:
            required = set(obligation.required_dependency_ids)
            for item in items:
                for dependency_id in sorted(required.intersection(item.supports)):
                    edges.append(
                        (obligation.obligation_id, item.item_id, dependency_id)
                    )
        return cls(
            obligations=tuple(obligations),
            edges=tuple(sorted(edges)),
        )

    @property
    def required_dependency_ids(self) -> frozenset[str]:
        return frozenset(
            dependency_id
            for obligation in self.obligations
            for dependency_id in obligation.required_dependency_ids
        )

    @property
    def critical_dependency_ids(self) -> frozenset[str]:
        return frozenset(
            dependency_id
            for obligation in self.obligations
            for dependency_id in obligation.critical_dependency_ids
        )

    def record(self) -> dict[str, object]:
        return {
            "schema_version": "return_obligation_memory_graph_v1",
            "obligations": [item.record() for item in self.obligations],
            "edges": [
                {
                    "obligation_id": obligation_id,
                    "item_id": item_id,
                    "dependency_id": dependency_id,
                }
                for obligation_id, item_id, dependency_id in self.edges
            ],
        }


@dataclass(frozen=True)
class ReturnSelectionReport:
    status: ReturnSelectionStatus
    algorithm: str
    eligible_item_ids: tuple[str, ...]
    selected_item_ids: tuple[str, ...]
    required_obligation_ids: tuple[str, ...]
    required_dependency_ids: tuple[str, ...]
    covered_dependency_ids: tuple[str, ...]
    critical_dependency_ids: tuple[str, ...]
    covered_critical_dependency_ids: tuple[str, ...]
    obligation_coverage: float
    critical_dependency_recall: float
    disclosure_cost: int
    disclosure_items: int
    disclosure_chars: int
    max_items: int
    max_chars: int
    graph: ReturnObligationGraph
    fallback_reason: str | None = None
    schema_version: str = RETURN_SELECTION_SCHEMA

    def record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "algorithm": self.algorithm,
            "eligible_item_ids": list(self.eligible_item_ids),
            "selected_item_ids": list(self.selected_item_ids),
            "required_obligation_ids": list(self.required_obligation_ids),
            "required_dependency_ids": list(self.required_dependency_ids),
            "covered_dependency_ids": list(self.covered_dependency_ids),
            "critical_dependency_ids": list(self.critical_dependency_ids),
            "covered_critical_dependency_ids": list(
                self.covered_critical_dependency_ids
            ),
            "obligation_coverage": self.obligation_coverage,
            "critical_dependency_recall": self.critical_dependency_recall,
            "disclosure_cost": self.disclosure_cost,
            "disclosure_items": self.disclosure_items,
            "disclosure_chars": self.disclosure_chars,
            "max_items": self.max_items,
            "max_chars": self.max_chars,
            "graph": self.graph.record(),
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class SignedReturnMemoryView:
    view_id: str
    return_event_id: str
    absence_id: str
    task_id: str
    role_id: str
    returning_principal_id: str
    source_epoch: int
    target_epoch: int
    issued_iteration: int
    expires_after_iteration: int
    purpose: str
    items: tuple[ReturnMemoryItem, ...]
    item_digest: str
    signer_id: str
    signature: str
    selection: ReturnSelectionReport
    ablation_mode: ReturnAblationMode = ReturnAblationMode.TRACE
    schema_version: str = RETURN_VIEW_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "view_id",
            "return_event_id",
            "absence_id",
            "task_id",
            "role_id",
            "returning_principal_id",
            "purpose",
            "signer_id",
        ):
            _text(getattr(self, field), field)
        if self.source_epoch < 0 or self.target_epoch <= self.source_epoch:
            raise ValueError("return view must cross a newer epoch")
        if self.expires_after_iteration < self.issued_iteration:
            raise ValueError("return view expires before it is issued")
        if not self.items:
            raise ValueError("return view must contain at least one item")
        if self.selection.status is not ReturnSelectionStatus.ISSUED:
            raise ValueError("a signed return view requires issued selection")
        if not isinstance(self.ablation_mode, ReturnAblationMode):
            raise TypeError("ablation_mode must be ReturnAblationMode")
        if tuple(item.item_id for item in self.items) != (
            self.selection.selected_item_ids
        ):
            raise ValueError("selected item ids do not match signed view order")
        expected = canonical_sha256(
            [item.record(include_text=True) for item in self.items]
        )
        if self.item_digest != expected:
            raise ValueError("return view item digest mismatch")
        _digest(self.item_digest, "item_digest")
        _digest(self.signature, "signature")

    def unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "view_id": self.view_id,
            "return_event_id": self.return_event_id,
            "absence_id": self.absence_id,
            "task_id": self.task_id,
            "role_id": self.role_id,
            "returning_principal_id": self.returning_principal_id,
            "source_epoch": self.source_epoch,
            "target_epoch": self.target_epoch,
            "issued_iteration": self.issued_iteration,
            "expires_after_iteration": self.expires_after_iteration,
            "purpose": self.purpose,
            "items": [item.record(include_text=True) for item in self.items],
            "item_digest": self.item_digest,
            "signer_id": self.signer_id,
            "selection": self.selection.record(),
            "ablation_mode": self.ablation_mode.value,
            "contains_workflow_or_action_sequence": False,
            "private_owner_memory_reads": 0,
        }

    def record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "signature": self.signature}


@dataclass(frozen=True)
class ReturnViewAdmission:
    accepted: bool
    view_id: str
    return_event_id: str
    absence_id: str
    admitted_epoch: int
    admitted_iteration: int
    item_decisions: tuple[ReturnItemDecision, ...]
    reason: str | None = None
    schema_version: str = RETURN_ADMISSION_SCHEMA

    def record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "accepted": self.accepted,
            "view_id": self.view_id,
            "return_event_id": self.return_event_id,
            "absence_id": self.absence_id,
            "admitted_epoch": self.admitted_epoch,
            "admitted_iteration": self.admitted_iteration,
            "item_decisions": [item.record() for item in self.item_decisions],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReturnCompilation:
    """Outcome of safe filtering and obligation-aware selection."""

    status: ReturnSelectionStatus
    view: SignedReturnMemoryView | None
    item_decisions: tuple[ReturnItemDecision, ...]
    selection: ReturnSelectionReport
    schema_version: str = RETURN_COMPILATION_SCHEMA

    def __post_init__(self) -> None:
        if (self.status is ReturnSelectionStatus.ISSUED) != (
            self.view is not None
        ):
            raise ValueError("only an issued compilation may contain a view")
        if self.selection.status is not self.status:
            raise ValueError("compilation and selection status disagree")

    def record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "selection": self.selection.record(),
            "item_decisions": [
                decision.record() for decision in self.item_decisions
            ],
            "return_memory_view": (
                self.view.record() if self.view is not None else None
            ),
        }


class ReturnMemoryGovernor:
    """Compile and authenticate the minimum valid role-workstate view."""

    def __init__(self, signer_id: str, key: bytes) -> None:
        self.signer_id = _text(signer_id, "signer_id")
        if not isinstance(key, bytes) or len(key) < 32:
            raise ValueError("signing key must contain at least 32 bytes")
        self._key = key

    def evaluate_item(
        self,
        item: ReturnMemoryItem,
        context: ReturnReadmissionContext,
        *,
        chars_already_admitted: int = 0,
        items_already_admitted: int = 0,
        mode: ReturnAblationMode = ReturnAblationMode.TRACE,
    ) -> ReturnItemDecision:
        if not isinstance(mode, ReturnAblationMode):
            mode = ReturnAblationMode(mode)
        authorization = (
            item.scope is ReturnMemoryScope.TEAM_PUBLIC
            or (
                item.scope is ReturnMemoryScope.ROLE
                and item.role_id == context.role_id
            )
        )
        freshness = (
            item.source_epoch < context.target_epoch
            and item.valid_from_epoch <= context.target_epoch <= item.valid_until_epoch
            and item.superseded_by is None
        )
        applicability = (
            item.task_id == context.task_id
            and item.role_id == context.role_id
            and context.purpose in item.purposes
            and item.kind in context.allowed_kinds
        )
        reference_ids = item.graph_reference_ids
        if reference_ids:
            provenance = any(
                receipt.item_id == item.item_id
                and receipt.issuer_id == self.signer_id
                and receipt.receipt_sha256 in item.provenance_sha256
                and reference_ids.issubset(receipt.bound_reference_ids)
                for receipt in item.provenance_receipts
            )
        else:
            # Legacy independent facts remain readable, but active
            # obligation-aware experiments attach bound receipts.
            provenance = bool(item.provenance_sha256)
        individual_budget_eligible = (
            items_already_admitted < context.max_items
            and chars_already_admitted + len(item.text) <= context.max_chars
        )
        enforced = {
            "authorization": mode is not ReturnAblationMode.FULL_RESTORE,
            "freshness": mode
            not in {
                ReturnAblationMode.FULL_RESTORE,
                ReturnAblationMode.TRACE_WITHOUT_FRESHNESS,
            },
            "applicability": mode is not ReturnAblationMode.FULL_RESTORE,
            "provenance": mode
            not in {
                ReturnAblationMode.FULL_RESTORE,
                ReturnAblationMode.TRACE_WITHOUT_PROVENANCE,
            },
            "individual_budget_eligible": (
                mode is not ReturnAblationMode.FULL_RESTORE
            ),
        }
        checks = {
            "authorization": authorization,
            "freshness": freshness,
            "applicability": applicability,
            "provenance": provenance,
            "individual_budget_eligible": individual_budget_eligible,
        }
        reasons: list[str] = []
        bypassed: list[str] = []
        if not authorization and enforced["authorization"]:
            reasons.append("authorization_denied")
        elif not authorization:
            bypassed.append("authorization")
        if not freshness and enforced["freshness"]:
            reasons.append("stale_or_superseded")
        elif not freshness:
            bypassed.append("freshness")
        if not applicability and enforced["applicability"]:
            reasons.append("not_applicable")
        elif not applicability:
            bypassed.append("applicability")
        if not provenance and enforced["provenance"]:
            reasons.append("provenance_missing")
        elif not provenance:
            bypassed.append("provenance")
        if (
            not individual_budget_eligible
            and enforced["individual_budget_eligible"]
        ):
            reasons.append("individual_disclosure_budget_exceeded")
        elif not individual_budget_eligible:
            bypassed.append("individual_budget_eligible")
        return ReturnItemDecision(
            item_id=item.item_id,
            admitted=all(
                checks[name] or not required
                for name, required in enforced.items()
            ),
            authorization=authorization,
            freshness=freshness,
            applicability=applicability,
            provenance=provenance,
            individual_budget_eligible=individual_budget_eligible,
            reasons=tuple(reasons),
            bypassed_checks=tuple(bypassed),
        )

    @staticmethod
    def _selection_metrics(
        items: Sequence[ReturnMemoryItem],
        context: ReturnReadmissionContext,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        float,
        float,
    ]:
        covered = set().union(*(set(item.supports) for item in items))
        required = {
            dependency_id
            for obligation in context.obligations
            for dependency_id in obligation.required_dependency_ids
        }
        critical = {
            dependency_id
            for obligation in context.obligations
            for dependency_id in obligation.critical_dependency_ids
        }
        completed_obligations = sum(
            set(obligation.required_dependency_ids).issubset(covered)
            for obligation in context.obligations
        )
        obligation_coverage = (
            completed_obligations / len(context.obligations)
            if context.obligations
            else 1.0
        )
        critical_recall = (
            len(covered.intersection(critical)) / len(critical)
            if critical
            else 1.0
        )
        return (
            tuple(sorted(covered.intersection(required))),
            tuple(sorted(covered.intersection(critical))),
            obligation_coverage,
            critical_recall,
        )

    @staticmethod
    def _required_structure_missing(
        items: Sequence[ReturnMemoryItem],
        context: ReturnReadmissionContext,
    ) -> str | None:
        required_obligations = {
            obligation.obligation_id for obligation in context.obligations
        }
        declared = {
            item.obligation_id
            for item in items
            if item.kind is ReturnMemoryKind.OPEN_OBLIGATION
        }
        if not required_obligations.issubset(declared):
            return "missing_open_obligation_item"
        frontier_references = {
            reference
            for item in items
            if item.kind is ReturnMemoryKind.PROGRESS_FRONTIER
            for reference in item.depends_on
        }
        if not required_obligations.issubset(frontier_references):
            return "missing_progress_frontier_reference"
        return None

    def _selection_report(
        self,
        *,
        status: ReturnSelectionStatus,
        eligible: Sequence[ReturnMemoryItem],
        selected: Sequence[ReturnMemoryItem],
        context: ReturnReadmissionContext,
        graph: ReturnObligationGraph,
        fallback_reason: str | None,
        algorithm: str = "cost_aware_obligation_greedy_v1",
    ) -> ReturnSelectionReport:
        (
            covered,
            covered_critical,
            obligation_coverage,
            critical_recall,
        ) = self._selection_metrics(selected, context)
        required = sorted(graph.required_dependency_ids)
        critical = sorted(graph.critical_dependency_ids)
        return ReturnSelectionReport(
            status=status,
            algorithm=algorithm,
            eligible_item_ids=tuple(item.item_id for item in eligible),
            selected_item_ids=tuple(item.item_id for item in selected),
            required_obligation_ids=tuple(
                obligation.obligation_id for obligation in context.obligations
            ),
            required_dependency_ids=tuple(required),
            covered_dependency_ids=covered,
            critical_dependency_ids=tuple(critical),
            covered_critical_dependency_ids=covered_critical,
            obligation_coverage=obligation_coverage,
            critical_dependency_recall=critical_recall,
            disclosure_cost=sum(item.disclosure_cost for item in selected),
            disclosure_items=len(selected),
            disclosure_chars=sum(len(item.text) for item in selected),
            max_items=context.max_items,
            max_chars=context.max_chars,
            graph=graph,
            fallback_reason=fallback_reason,
        )

    def _select_items(
        self,
        eligible: Sequence[ReturnMemoryItem],
        context: ReturnReadmissionContext,
        *,
        issue_incomplete: bool = False,
    ) -> tuple[
        tuple[ReturnMemoryItem, ...],
        ReturnSelectionStatus,
        str | None,
        ReturnObligationGraph,
    ]:
        graph = ReturnObligationGraph.build(context.obligations, eligible)
        if not eligible:
            status = (
                ReturnSelectionStatus.RESET_FALLBACK
                if context.incomplete_action is ReturnIncompleteAction.RESET
                else ReturnSelectionStatus.BLOCKED
            )
            return (), status, "no_safe_candidate", graph

        selected: list[ReturnMemoryItem] = []
        selected_ids: set[str] = set()
        chars = 0

        def add(item: ReturnMemoryItem) -> bool:
            nonlocal chars
            if item.item_id in selected_ids:
                return True
            if (
                len(selected) >= context.max_items
                or chars + len(item.text) > context.max_chars
            ):
                return False
            selected.append(item)
            selected_ids.add(item.item_id)
            chars += len(item.text)
            return True

        if not context.obligations:
            for item in eligible:
                add(item)
            return (
                tuple(selected),
                ReturnSelectionStatus.ISSUED,
                None,
                graph,
            )

        required_obligation_ids = {
            obligation.obligation_id for obligation in context.obligations
        }
        mandatory = [
            item
            for item in eligible
            if (
                item.kind is ReturnMemoryKind.OPEN_OBLIGATION
                and item.obligation_id in required_obligation_ids
            )
            or (
                item.kind is ReturnMemoryKind.PROGRESS_FRONTIER
                and bool(required_obligation_ids.intersection(item.depends_on))
            )
        ]
        structural_failure = self._required_structure_missing(
            mandatory, context
        )
        if structural_failure is None:
            for item in mandatory:
                if not add(item):
                    structural_failure = "mandatory_workstate_exceeds_budget"
                    break

        required_dependencies = set(graph.required_dependency_ids)
        critical_dependencies = set(graph.critical_dependency_ids)
        covered = set().union(*(set(item.supports) for item in selected))
        remaining = [
            item
            for item in eligible
            if item.item_id not in selected_ids
        ]
        while required_dependencies - covered:
            ranked: list[
                tuple[float, int, int, str, ReturnMemoryItem]
            ] = []
            for item in remaining:
                newly_covered = (
                    set(item.supports)
                    .intersection(required_dependencies)
                    .difference(covered)
                )
                gain = sum(
                    2 if dependency in critical_dependencies else 1
                    for dependency in newly_covered
                )
                if gain <= 0:
                    continue
                if (
                    len(selected) >= context.max_items
                    or chars + len(item.text) > context.max_chars
                ):
                    continue
                ranked.append(
                    (
                        gain / item.disclosure_cost,
                        gain,
                        -item.disclosure_cost,
                        item.item_id,
                        item,
                    )
                )
            if not ranked:
                break
            # Descending utility/cost and gain, then lower cost and stable id.
            best = sorted(
                ranked,
                key=lambda row: (-row[0], -row[1], -row[2], row[3]),
            )[0][4]
            add(best)
            covered.update(best.supports)
            remaining = [
                item for item in remaining if item.item_id != best.item_id
            ]

        (
            _,
            _,
            obligation_coverage,
            critical_recall,
        ) = self._selection_metrics(selected, context)
        failure = structural_failure
        if (
            failure is None
            and obligation_coverage < context.minimum_obligation_coverage
        ):
            failure = "insufficient_obligation_coverage"
        if (
            failure is None
            and critical_recall
            < context.minimum_critical_dependency_recall
        ):
            failure = "insufficient_critical_dependency_recall"
        if failure is not None:
            if issue_incomplete and selected:
                return (
                    tuple(selected),
                    ReturnSelectionStatus.ISSUED,
                    f"incomplete_view_issued:{failure}",
                    graph,
                )
            status = (
                ReturnSelectionStatus.RESET_FALLBACK
                if context.incomplete_action is ReturnIncompleteAction.RESET
                else ReturnSelectionStatus.BLOCKED
            )
            return (), status, failure, graph
        return tuple(selected), ReturnSelectionStatus.ISSUED, None, graph

    @staticmethod
    def _select_fixed_order(
        eligible: Sequence[ReturnMemoryItem],
        context: ReturnReadmissionContext,
    ) -> tuple[
        tuple[ReturnMemoryItem, ...],
        ReturnSelectionStatus,
        str | None,
        ReturnObligationGraph,
    ]:
        """Take filtered items in frozen order without obligation utility."""

        graph = ReturnObligationGraph.build(context.obligations, eligible)
        selected: list[ReturnMemoryItem] = []
        chars = 0
        for item in eligible:
            if len(selected) >= context.max_items:
                break
            if chars + len(item.text) > context.max_chars:
                continue
            selected.append(item)
            chars += len(item.text)
        if selected:
            return (
                tuple(selected),
                ReturnSelectionStatus.ISSUED,
                None,
                graph,
            )
        status = (
            ReturnSelectionStatus.RESET_FALLBACK
            if context.incomplete_action is ReturnIncompleteAction.RESET
            else ReturnSelectionStatus.BLOCKED
        )
        return (), status, "no_safe_candidate", graph

    @staticmethod
    def _select_full_restore(
        candidates: Sequence[ReturnMemoryItem],
        context: ReturnReadmissionContext,
    ) -> tuple[
        tuple[ReturnMemoryItem, ...],
        ReturnSelectionStatus,
        str | None,
        ReturnObligationGraph,
    ]:
        """Restore every old workstate item, deliberately bypassing controls."""

        graph = ReturnObligationGraph.build(context.obligations, candidates)
        if candidates:
            return (
                tuple(candidates),
                ReturnSelectionStatus.ISSUED,
                None,
                graph,
            )
        status = (
            ReturnSelectionStatus.RESET_FALLBACK
            if context.incomplete_action is ReturnIncompleteAction.RESET
            else ReturnSelectionStatus.BLOCKED
        )
        return (), status, "no_restore_candidate", graph

    def compile(
        self,
        items: Iterable[ReturnMemoryItem],
        context: ReturnReadmissionContext,
        *,
        source_epoch: int,
        expires_after_iteration: int,
        mode: ReturnAblationMode = ReturnAblationMode.TRACE,
    ) -> ReturnCompilation:
        if not isinstance(mode, ReturnAblationMode):
            mode = ReturnAblationMode(mode)
        candidates = sorted(
            tuple(items),
            key=lambda item: (
                tuple(ReturnMemoryKind).index(item.kind),
                item.item_id,
            ),
        )
        base_decisions: list[ReturnItemDecision] = []
        eligible: list[ReturnMemoryItem] = []
        for item in candidates:
            decision = self.evaluate_item(item, context, mode=mode)
            base_decisions.append(decision)
            if decision.admitted:
                eligible.append(item)

        if mode is ReturnAblationMode.FULL_RESTORE:
            selected, status, fallback_reason, graph = (
                self._select_full_restore(candidates, context)
            )
            algorithm = "full_restore_all_workstate_v1"
        elif mode in {
            ReturnAblationMode.VALIDITY_FILTER_ONLY,
            ReturnAblationMode.RESET_REBRIEF,
        }:
            selected, status, fallback_reason, graph = (
                self._select_fixed_order(eligible, context)
            )
            algorithm = (
                "reset_rebrief_current_state_fixed_order_v1"
                if mode is ReturnAblationMode.RESET_REBRIEF
                else "validity_filter_fixed_order_v1"
            )
        else:
            selected, status, fallback_reason, graph = self._select_items(
                eligible,
                context,
                issue_incomplete=(
                    mode is ReturnAblationMode.TRACE_WITHOUT_FALLBACK
                ),
            )
            algorithm = (
                "static_compact_obligation_greedy_v1"
                if mode is ReturnAblationMode.STATIC_COMPACT
                else (
                    "cost_aware_obligation_greedy_without_fallback_v1"
                    if mode
                    is ReturnAblationMode.TRACE_WITHOUT_FALLBACK
                    else "cost_aware_obligation_greedy_v1"
                )
            )
        selected_ids = {item.item_id for item in selected}
        decisions: list[ReturnItemDecision] = []
        for decision in base_decisions:
            if decision.item_id in selected_ids:
                decisions.append(replace(decision, reasons=()))
            elif decision.admitted:
                nonselection_reason = (
                    "not_selected_by_fixed_budget"
                    if mode
                    in {
                        ReturnAblationMode.VALIDITY_FILTER_ONLY,
                        ReturnAblationMode.RESET_REBRIEF,
                    }
                    else "not_selected_by_obligation_optimizer"
                )
                decisions.append(
                    replace(
                        decision,
                        admitted=False,
                        reasons=(nonselection_reason,),
                    )
                )
            else:
                decisions.append(decision)
        selection = self._selection_report(
            status=status,
            eligible=eligible,
            selected=selected,
            context=context,
            graph=graph,
            fallback_reason=fallback_reason,
            algorithm=algorithm,
        )
        if status is not ReturnSelectionStatus.ISSUED:
            return ReturnCompilation(
                status=status,
                view=None,
                item_decisions=tuple(decisions),
                selection=selection,
            )

        digest = canonical_sha256(
            [item.record(include_text=True) for item in selected]
        )
        view_id = "return_view_" + hmac.new(
            self._key,
            (
                f"{context.return_event_id}:{context.absence_id}:"
                f"{mode.value}:{digest}"
            ).encode(),
            hashlib.sha256,
        ).hexdigest()[:24]
        placeholder = SignedReturnMemoryView(
            view_id=view_id,
            return_event_id=context.return_event_id,
            absence_id=context.absence_id,
            task_id=context.task_id,
            role_id=context.role_id,
            returning_principal_id=context.principal_id,
            source_epoch=source_epoch,
            target_epoch=context.target_epoch,
            issued_iteration=context.iteration,
            expires_after_iteration=expires_after_iteration,
            purpose=context.purpose,
            items=tuple(selected),
            item_digest=digest,
            signer_id=self.signer_id,
            signature="0" * 64,
            selection=selection,
            ablation_mode=mode,
        )
        signature = hmac.new(
            self._key,
            json.dumps(
                placeholder.unsigned_record(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        view = replace(placeholder, signature=signature)
        return ReturnCompilation(
            status=status,
            view=view,
            item_decisions=tuple(decisions),
            selection=selection,
        )

    def issue(
        self,
        items: Iterable[ReturnMemoryItem],
        context: ReturnReadmissionContext,
        *,
        source_epoch: int,
        expires_after_iteration: int,
        mode: ReturnAblationMode = ReturnAblationMode.TRACE,
    ) -> tuple[SignedReturnMemoryView, tuple[ReturnItemDecision, ...]]:
        """Backward-compatible strict API for callers that require a view."""

        compilation = self.compile(
            items,
            context,
            source_epoch=source_epoch,
            expires_after_iteration=expires_after_iteration,
            mode=mode,
        )
        if compilation.view is None:
            raise PermissionError(
                "return workstate compilation did not issue a view: "
                f"{compilation.selection.fallback_reason}"
            )
        return compilation.view, compilation.item_decisions

    def verify(self, view: SignedReturnMemoryView) -> bool:
        expected = hmac.new(
            self._key,
            json.dumps(
                view.unsigned_record(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(view.signature, expected)

    def admit(
        self,
        view: SignedReturnMemoryView,
        context: ReturnReadmissionContext,
    ) -> ReturnViewAdmission:
        mode = view.ablation_mode
        reason: str | None = None
        if not self.verify(view):
            reason = "signature_invalid"
        elif view.return_event_id != context.return_event_id:
            reason = "return_event_mismatch"
        elif view.absence_id != context.absence_id:
            reason = "absence_mismatch"
        elif view.task_id != context.task_id or view.role_id != context.role_id:
            reason = "applicability_mismatch"
        elif view.returning_principal_id != context.principal_id:
            reason = "principal_mismatch"
        elif view.target_epoch != context.target_epoch:
            reason = "epoch_mismatch"
        elif context.iteration > view.expires_after_iteration:
            reason = "view_expired"
        decisions_list: list[ReturnItemDecision] = []
        chars = 0
        for item in view.items:
            decision = self.evaluate_item(
                item,
                context,
                chars_already_admitted=chars,
                items_already_admitted=len(decisions_list),
                mode=mode,
            )
            decisions_list.append(decision)
            if decision.admitted:
                chars += len(item.text)
        decisions = tuple(decisions_list)
        if reason is None and not all(item.admitted for item in decisions):
            reason = "item_revalidation_failed"
        (
            covered,
            covered_critical,
            obligation_coverage,
            critical_recall,
        ) = self._selection_metrics(view.items, context)
        expected_obligations = tuple(
            obligation.obligation_id for obligation in context.obligations
        )
        selection = view.selection
        coverage_required = mode not in {
            ReturnAblationMode.FULL_RESTORE,
            ReturnAblationMode.VALIDITY_FILTER_ONLY,
            ReturnAblationMode.RESET_REBRIEF,
            ReturnAblationMode.TRACE_WITHOUT_FALLBACK,
        }
        selection_matches = (
            selection.status is ReturnSelectionStatus.ISSUED
            and selection.selected_item_ids
            == tuple(item.item_id for item in view.items)
            and selection.required_obligation_ids == expected_obligations
            and selection.covered_dependency_ids == covered
            and selection.covered_critical_dependency_ids
            == covered_critical
            and abs(selection.obligation_coverage - obligation_coverage)
            <= 1e-12
            and abs(
                selection.critical_dependency_recall - critical_recall
            )
            <= 1e-12
            and selection.disclosure_items == len(view.items)
            and selection.disclosure_chars
            == sum(len(item.text) for item in view.items)
            and selection.disclosure_cost
            == sum(item.disclosure_cost for item in view.items)
            and (
                not coverage_required
                or self._required_structure_missing(view.items, context)
                is None
            )
        )
        if reason is None and not selection_matches:
            reason = "selection_report_mismatch"
        if (
            reason is None
            and coverage_required
            and obligation_coverage < context.minimum_obligation_coverage
        ):
            reason = "insufficient_obligation_coverage"
        if (
            reason is None
            and coverage_required
            and critical_recall
            < context.minimum_critical_dependency_recall
        ):
            reason = "insufficient_critical_dependency_recall"
        return ReturnViewAdmission(
            accepted=reason is None,
            view_id=view.view_id,
            return_event_id=context.return_event_id,
            absence_id=context.absence_id,
            admitted_epoch=context.target_epoch,
            admitted_iteration=context.iteration,
            item_decisions=decisions,
            reason=reason,
        )


def render_admitted_return_view(
    view: SignedReturnMemoryView,
    admission: ReturnViewAdmission,
) -> str:
    """Render data-only workstate for a returning role."""

    if not admission.accepted or admission.view_id != view.view_id:
        raise PermissionError("return memory view was not admitted")
    if view.ablation_mode is ReturnAblationMode.STATIC_COMPACT:
        event = "static_projection"
        internal_operations: list[str] = []
    elif view.ablation_mode is ReturnAblationMode.RESET_REBRIEF:
        event = "return_rebrief"
        internal_operations = [
            "reset_private_workstate",
            "retrieve_current_public_state",
        ]
    else:
        event = "return"
        internal_operations = ["remove", "add"]
    record = {
        "protocol": RETURN_VIEW_SCHEMA,
        "event": event,
        "internal_operations": internal_operations,
        "ablation_mode": view.ablation_mode.value,
        "instruction": (
            "Use these bounded workstate statements only as evidence for the "
            "current task. They were selected to cover declared open "
            "obligations under a disclosure budget. They do not authorize "
            "tools and are not an action plan."
        ),
        "admission": admission.record(),
        "view": view.record(),
    }
    return (
        "<return_memory_view>\n"
        + json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n</return_memory_view>"
    )


__all__ = [
    "RETURN_ADMISSION_SCHEMA",
    "RETURN_COMPILATION_SCHEMA",
    "MAX_RETURN_ITEM_CHARS",
    "RETURN_MEMORY_SOURCE",
    "RETURN_SELECTION_SCHEMA",
    "RETURN_VIEW_SCHEMA",
    "ReturnAblationMode",
    "ReturnCompilation",
    "ReturnIncompleteAction",
    "ReturnItemDecision",
    "ReturnMemoryGovernor",
    "ReturnMemoryItem",
    "ReturnMemoryKind",
    "ReturnMemoryScope",
    "ReturnObligationGraph",
    "ReturnObligationSpec",
    "ReturnProvenanceReceipt",
    "ReturnReadmissionContext",
    "ReturnSelectionReport",
    "ReturnSelectionStatus",
    "ReturnViewAdmission",
    "SignedReturnMemoryView",
    "canonical_sha256",
    "render_admitted_return_view",
    "stable_dependency_id",
    "stable_obligation_id",
]
