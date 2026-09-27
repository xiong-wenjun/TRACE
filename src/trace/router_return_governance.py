"""Bridge a frozen Router checkpoint into the TRACE compiler."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .return_governance import (
    ReturnAblationMode,
    ReturnCompilation,
    ReturnIncompleteAction,
    ReturnMemoryGovernor,
    ReturnMemoryItem,
    ReturnMemoryKind,
    ReturnMemoryScope,
    ReturnObligationSpec,
    ReturnProvenanceReceipt,
    ReturnReadmissionContext,
    ReturnViewAdmission,
    canonical_sha256,
)
from .router_return_protocol import (
    RouterObligationStatus,
    RouterPostAbsenceCheckpoint,
    RouterReturnArm,
    RouterReturnFork,
    RouterReturnProtocol,
    RouterWorkstateItem,
    RouterWorkstateStatus,
)


def _verified_dispute_items(
    checkpoint: RouterPostAbsenceCheckpoint,
    trusted_provenance_sha256: Sequence[str],
) -> dict[str, tuple[bool, str, str]]:
    """Map uniquely receipt-verified dispute items to resolution metadata."""

    trusted = set(trusted_provenance_sha256)
    if not trusted:
        return {}
    current = {
        item.item_id: item for item in checkpoint.current_workstate_items
    }
    resolutions: dict[str, tuple[bool, str, str]] = {}
    for dispute in checkpoint.disputes:
        if dispute.status.value != "open":
            continue
        existing = current.get(dispute.existing_item_id)
        challenger = current.get(dispute.challenger_item_id)
        if existing is None or challenger is None:
            continue
        verified = [
            item
            for item in (existing, challenger)
            if item.provenance_sha256 in trusted
        ]
        if len(verified) != 1:
            continue
        accepted = verified[0]
        rejected = challenger if accepted.item_id == existing.item_id else existing
        resolutions[accepted.item_id] = (
            True,
            dispute.dispute_id,
            rejected.item_id,
        )
        resolutions[rejected.item_id] = (
            False,
            dispute.dispute_id,
            accepted.item_id,
        )
    return resolutions


@dataclass(frozen=True)
class RouterTraceCompilation:
    """TRACE output and matched counterfactual forks from one checkpoint."""

    context: ReturnReadmissionContext
    compilation: ReturnCompilation
    admission: ReturnViewAdmission | None
    forks: tuple[RouterReturnFork, ...]
    router_candidate_item_ids: tuple[str, ...]
    selected_router_item_ids: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "schema_version": "router_trace_compilation_v1",
            "checkpoint_sha256": self.forks[0].checkpoint_sha256,
            "context": {
                "return_event_id": self.context.return_event_id,
                "absence_id": self.context.absence_id,
                "task_id": self.context.task_id,
                "role_id": self.context.role_id,
                "principal_id": self.context.principal_id,
                "target_epoch": self.context.target_epoch,
                "purpose": self.context.purpose,
                "max_items": self.context.max_items,
                "max_chars": self.context.max_chars,
            },
            "router_candidate_item_ids": list(
                self.router_candidate_item_ids
            ),
            "selected_router_item_ids": list(
                self.selected_router_item_ids
            ),
            "compilation": self.compilation.record(),
            "admission": self.admission.record() if self.admission else None,
            "forks": [fork.record() for fork in self.forks],
        }


def _replacement_map(
    checkpoint: RouterPostAbsenceCheckpoint,
) -> dict[str, str]:
    return {
        item.supersedes_item_id: item.item_id
        for item in checkpoint.current_workstate_items
        if item.supersedes_item_id is not None
        and item.status is RouterWorkstateStatus.ACTIVE
    }


def _relevant_router_items(
    checkpoint: RouterPostAbsenceCheckpoint,
    *,
    candidate_scope: str,
) -> tuple[RouterWorkstateItem, ...]:
    obligation_ids = {
        obligation.obligation_id
        for obligation in checkpoint.obligations
        if obligation.owner_principal_id
        == checkpoint.returning_principal_id
        and obligation.status
        in {
            RouterObligationStatus.OPEN,
            RouterObligationStatus.BLOCKED,
        }
    }
    current_relevant = {
        item.item_id: item
        for item in checkpoint.current_workstate_items
        if item.status is RouterWorkstateStatus.ACTIVE
        and (
            item.owner_principal_id == checkpoint.returning_principal_id
            or obligation_ids.intersection(item.obligation_ids)
        )
    }
    if candidate_scope == "current_active":
        return tuple(
            current_relevant[item_id]
            for item_id in sorted(current_relevant)
        )
    if candidate_scope != "return_candidates":
        raise ValueError(
            "candidate_scope must be return_candidates or current_active"
        )
    combined: dict[str, RouterWorkstateItem] = {
        item.item_id: item for item in checkpoint.departure.workstate_items
    }
    combined.update(
        {
            item.item_id: item
            for item in checkpoint.current_workstate_items
            if item.owner_principal_id == checkpoint.returning_principal_id
            or obligation_ids.intersection(item.obligation_ids)
        }
    )
    current_by_id = {
        item.item_id: item for item in checkpoint.current_workstate_items
    }
    return tuple(
        current_by_id.get(item_id, item)
        for item_id, item in sorted(combined.items())
    )


def compile_router_trace(
    protocol: RouterReturnProtocol,
    checkpoint: RouterPostAbsenceCheckpoint,
    governor: ReturnMemoryGovernor,
    *,
    purpose: str,
    max_items: int = 8,
    max_chars: int = 4096,
    iteration: int = 1,
    expires_after_iteration: int = 2,
    mode: ReturnAblationMode = ReturnAblationMode.TRACE,
    incomplete_action: ReturnIncompleteAction = ReturnIncompleteAction.RESET,
    candidate_scope: str = "return_candidates",
    trusted_provenance_sha256: Sequence[str] = (),
    static_include_disputed_challengers: bool = False,
    require_independent_provenance: bool = False,
) -> RouterTraceCompilation:
    """Compile authenticated workstate and bind it to all matched arms."""

    membership = next(
        (
            item
            for item in checkpoint.memberships
            if item.principal_id == checkpoint.returning_principal_id
        ),
        None,
    )
    if membership is None:
        raise ValueError("returning principal is absent from checkpoint membership")
    obligation_rows = tuple(
        obligation
        for obligation in checkpoint.obligations
        if obligation.owner_principal_id
        == checkpoint.returning_principal_id
        and obligation.status
        in {
            RouterObligationStatus.OPEN,
            RouterObligationStatus.BLOCKED,
        }
    )
    context = ReturnReadmissionContext(
        return_event_id=(
            f"return:{checkpoint.task_id}:"
            f"{checkpoint.returning_principal_id}"
        ),
        absence_id=(
            f"absence:{checkpoint.task_id}:"
            f"{checkpoint.returning_principal_id}"
        ),
        task_id=checkpoint.task_id,
        role_id=membership.role_id,
        principal_id=checkpoint.returning_principal_id,
        target_epoch=checkpoint.target_epoch,
        iteration=iteration,
        purpose=purpose,
        max_items=max_items,
        max_chars=max_chars,
        obligations=tuple(
            ReturnObligationSpec(
                obligation_id=obligation.obligation_id,
                required_dependency_ids=obligation.required_dependency_ids,
                critical_dependency_ids=obligation.critical_dependency_ids,
            )
            for obligation in obligation_rows
        ),
        incomplete_action=incomplete_action,
    )
    items: list[ReturnMemoryItem] = []

    def add_structure(
        *,
        item_id: str,
        kind: ReturnMemoryKind,
        text: str,
        obligation_id: str,
        depends_on: tuple[str, ...] = (),
    ) -> None:
        source = canonical_sha256(
            {
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "item_id": item_id,
                "text": text,
                "obligation_id": obligation_id,
            }
        )
        references = tuple(
            dict.fromkeys((obligation_id,) + depends_on)
        )
        receipt = ReturnProvenanceReceipt.create(
            item_id=item_id,
            source_sha256=source,
            bound_reference_ids=references,
            issuer_id=governor.signer_id,
            issued_epoch=checkpoint.departure.source_epoch,
        )
        items.append(
            ReturnMemoryItem(
                item_id=item_id,
                kind=kind,
                text=text,
                task_id=checkpoint.task_id,
                role_id=membership.role_id,
                writer_principal_id="router",
                source_epoch=checkpoint.departure.source_epoch,
                valid_from_epoch=checkpoint.target_epoch,
                valid_until_epoch=checkpoint.target_epoch + 1,
                purposes=(purpose,),
                provenance_sha256=(receipt.receipt_sha256,),
                scope=ReturnMemoryScope.ROLE,
                obligation_id=(
                    obligation_id
                    if kind is ReturnMemoryKind.OPEN_OBLIGATION
                    else None
                ),
                depends_on=depends_on,
                provenance_receipts=(receipt,),
                attributes={
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "router_structure": True,
                },
            )
        )

    for obligation in obligation_rows:
        add_structure(
            item_id=f"router:{obligation.obligation_id}:open",
            kind=ReturnMemoryKind.OPEN_OBLIGATION,
            text=obligation.description,
            obligation_id=obligation.obligation_id,
        )
        add_structure(
            item_id=f"router:{obligation.obligation_id}:frontier",
            kind=ReturnMemoryKind.PROGRESS_FRONTIER,
            text=(
                "Resume only after the listed critical dependencies have "
                "been safely reconciled."
            ),
            obligation_id=obligation.obligation_id,
            depends_on=(obligation.obligation_id,),
        )

    replacements = _replacement_map(checkpoint)
    verified_disputes = _verified_dispute_items(
        checkpoint,
        trusted_provenance_sha256,
    )
    router_items = _relevant_router_items(
        checkpoint,
        candidate_scope=candidate_scope,
    )
    for item in router_items:
        source = item.provenance_sha256 or canonical_sha256(item.record())
        is_challenger = item.supersedes_item_id is not None
        independent_sources = len(set(item.provenance_source_ids))
        provenance_eligible = item.provenance_status == "legacy"
        if item.provenance_status == "valid":
            provenance_eligible = bool(item.provenance_receipt_sha256s)
        if (
            require_independent_provenance
            and is_challenger
            and mode is not ReturnAblationMode.TRACE_WITHOUT_PROVENANCE
        ):
            provenance_eligible = (
                item.provenance_status == "valid"
                and independent_sources >= 2
                and len(item.provenance_receipt_sha256s) >= 2
            )
        receipt = (
            ReturnProvenanceReceipt.create(
                item_id=item.item_id,
                source_sha256=source,
                bound_reference_ids=tuple(
                    dict.fromkeys(item.dependency_ids + item.obligation_ids)
                ),
                issuer_id=governor.signer_id,
                issued_epoch=checkpoint.departure.source_epoch,
            )
            if provenance_eligible
            else None
        )
        superseded_by = replacements.get(item.item_id)
        text = item.text
        if item.status is RouterWorkstateStatus.REVOKED:
            superseded_by = f"revoked:{item.item_id}"
        elif item.status is RouterWorkstateStatus.SUPERSEDED:
            superseded_by = superseded_by or f"superseded:{item.item_id}"
        elif item.status is RouterWorkstateStatus.DISPUTED:
            resolution = verified_disputes.get(item.item_id)
            if resolution is None:
                superseded_by = f"disputed:{item.item_id}"
            elif resolution[0]:
                superseded_by = None
                case_id = resolution[1].rsplit(":dispute:", 1)[-1]
                text = (
                    "[Evidence-verified dispute resolution: "
                    f"case_id={case_id}; retained_state_id="
                    f"{item.item_id}; rejected_state_id={resolution[2]}; "
                    f"receipt_sha256={item.provenance_sha256}]\n{item.text}"
                )
            else:
                superseded_by = f"dispute_rejected:{resolution[2]}"
        provenance_digest = (
            receipt.receipt_sha256
            if receipt is not None
            else canonical_sha256(
                {
                    "unverified_item_id": item.item_id,
                    "source": source,
                    "provenance_status": item.provenance_status,
                }
            )
        )
        items.append(
            ReturnMemoryItem(
                item_id=item.item_id,
                kind=ReturnMemoryKind.VERIFIED_FACT,
                text=text,
                task_id=checkpoint.task_id,
                role_id=membership.role_id,
                writer_principal_id=item.writer_principal_id,
                source_epoch=checkpoint.departure.source_epoch,
                valid_from_epoch=checkpoint.target_epoch,
                valid_until_epoch=checkpoint.target_epoch + 1,
                purposes=(purpose,),
                provenance_sha256=(provenance_digest,),
                scope=(
                    ReturnMemoryScope.TEAM_PUBLIC
                    if item.team_public
                    else ReturnMemoryScope.ROLE
                ),
                superseded_by=superseded_by,
                supports=item.dependency_ids,
                depends_on=item.obligation_ids,
                provenance_receipts=((receipt,) if receipt is not None else ()),
                attributes={
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "router_version": item.version,
                    "provenance_status": item.provenance_status,
                    "provenance_source_count": independent_sources,
                    "provenance_receipt_count": len(
                        item.provenance_receipt_sha256s
                    ),
                    "provenance_eligible": provenance_eligible,
                    "router_status": (
                        "active_after_evidence_verification"
                        if verified_disputes.get(item.item_id, (False, "", ""))[0]
                        else item.status.value
                    ),
                },
            )
        )
    compilation = governor.compile(
        items,
        context,
        source_epoch=checkpoint.departure.source_epoch,
        expires_after_iteration=expires_after_iteration,
        mode=mode,
    )
    admission: ReturnViewAdmission | None = None
    if compilation.view is not None:
        admission = governor.admit(compilation.view, context)
        if not admission.accepted:
            raise PermissionError(
                "Router TRACE view failed admission: " + admission.reason
            )
    router_candidate_ids = tuple(item.item_id for item in router_items)
    selected_router_ids = tuple(
        item.item_id
        for item in (compilation.view.items if compilation.view else ())
        if item.item_id in set(router_candidate_ids)
    )
    forks = protocol.fork_return_methods(
        checkpoint,
        trace_selected_item_ids=(
            selected_router_ids if compilation.view is not None else ()
        ),
        trace_block_reason=(
            None
            if compilation.view is not None
            else (
                compilation.selection.fallback_reason
                or compilation.status.value
            )
        ),
    )
    if static_include_disputed_challengers:
        active_ids = tuple(
            item.item_id
            for item in checkpoint.current_workstate_items
            if item.status is RouterWorkstateStatus.ACTIVE
        )
        challenger_ids = tuple(
            dispute.challenger_item_id
            for dispute in checkpoint.disputes
            if dispute.status.value == "open"
        )
        static_ids = tuple(dict.fromkeys((*challenger_ids, *active_ids)))
        forks = tuple(
            replace(fork, inherited_item_ids=static_ids)
            if fork.arm is RouterReturnArm.STATIC
            else fork
            for fork in forks
        )
    return RouterTraceCompilation(
        context=context,
        compilation=compilation,
        admission=admission,
        forks=forks,
        router_candidate_item_ids=router_candidate_ids,
        selected_router_item_ids=selected_router_ids,
    )


def render_router_fork(
    fork: RouterReturnFork,
    checkpoint: RouterPostAbsenceCheckpoint,
) -> str:
    """Render inherited workstate only; the task/prompt remains arm-invariant."""

    by_id: dict[str, Mapping[str, object]] = {}
    for item in checkpoint.departure.workstate_items:
        by_id[item.item_id] = item.record()
    for item in checkpoint.current_workstate_items:
        by_id[item.item_id] = item.record()
    return json_dumps(
        {
            "checkpoint_sha256": fork.checkpoint_sha256,
            "workstate_items": [
                by_id[item_id]
                for item_id in fork.inherited_item_ids
                if item_id in by_id
            ],
        }
    )


def json_dumps(value: object) -> str:
    import json

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


__all__ = [
    "RouterTraceCompilation",
    "compile_router_trace",
    "render_router_fork",
]
