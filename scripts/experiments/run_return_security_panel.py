#!/usr/bin/env python3
"""Run the preregisterable deterministic TRACE security-negative panel.

The panel contains no model calls.  Each row is a fully specified invalid
RETURN attempt or weakened-method negative control.  TRACE is expected to
fail closed; ablations are expected to expose the precise safety property
that they remove.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trace.return_governance import (  # noqa: E402
    ReturnAblationMode,
    ReturnCompilation,
    ReturnMemoryGovernor,
    ReturnMemoryItem,
    ReturnMemoryKind,
    ReturnMemoryScope,
    ReturnObligationSpec,
    ReturnProvenanceReceipt,
    ReturnReadmissionContext,
    ReturnSelectionStatus,
    SignedReturnMemoryView,
    canonical_sha256,
    stable_dependency_id,
    stable_obligation_id,
)


SCHEMA = "trace_return_security_panel_v1"
METHODS = (
    ReturnAblationMode.TRACE,
    ReturnAblationMode.FULL_RESTORE,
    ReturnAblationMode.TRACE_WITHOUT_FALLBACK,
    ReturnAblationMode.TRACE_WITHOUT_PROVENANCE,
    ReturnAblationMode.TRACE_WITHOUT_FRESHNESS,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()


def write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()


def context(**changes: object) -> ReturnReadmissionContext:
    obligation_id = stable_obligation_id(
        "security-task", "role:researcher", "finish-open-analysis"
    )
    dependency_id = stable_dependency_id(obligation_id, "verified-critical-fact")
    values: dict[str, object] = {
        "return_event_id": "return:security:1",
        "absence_id": "absence:security:1",
        "task_id": "security-task",
        "role_id": "role:researcher",
        "principal_id": "alice",
        "target_epoch": 2,
        "iteration": 2,
        "purpose": "continue_analysis",
        "max_items": 3,
        "max_chars": 512,
        "obligations": (
            ReturnObligationSpec(
                obligation_id=obligation_id,
                required_dependency_ids=(dependency_id,),
                critical_dependency_ids=(dependency_id,),
            ),
        ),
    }
    values.update(changes)
    return ReturnReadmissionContext(**values)  # type: ignore[arg-type]


def signed_item(
    governor: ReturnMemoryGovernor,
    *,
    item_id: str,
    kind: ReturnMemoryKind,
    text: str,
    obligation_id: str | None = None,
    supports: tuple[str, ...] = (),
    depends_on: tuple[str, ...] = (),
    role_id: str = "role:researcher",
    source_epoch: int = 0,
    valid_from_epoch: int = 2,
    valid_until_epoch: int = 3,
    superseded_by: str | None = None,
) -> ReturnMemoryItem:
    references = tuple(
        sorted(
            set(supports)
            | set(depends_on)
            | ({obligation_id} if obligation_id else set())
        )
    )
    receipt = ReturnProvenanceReceipt.create(
        item_id=item_id,
        source_sha256=canonical_sha256(
            {"fixture": item_id, "text_sha256": canonical_sha256(text)}
        ),
        bound_reference_ids=references,
        issuer_id=governor.signer_id,
        issued_epoch=0,
    )
    return ReturnMemoryItem(
        item_id=item_id,
        kind=kind,
        text=text,
        task_id="security-task",
        role_id=role_id,
        writer_principal_id="alice",
        source_epoch=source_epoch,
        valid_from_epoch=valid_from_epoch,
        valid_until_epoch=valid_until_epoch,
        purposes=("continue_analysis",),
        provenance_sha256=(receipt.receipt_sha256,),
        scope=ReturnMemoryScope.ROLE,
        superseded_by=superseded_by,
        obligation_id=obligation_id,
        supports=supports,
        depends_on=depends_on,
        provenance_receipts=(receipt,),
    )


def fixture(
    governor: ReturnMemoryGovernor,
) -> tuple[
    ReturnReadmissionContext,
    tuple[ReturnMemoryItem, ReturnMemoryItem],
    ReturnMemoryItem,
]:
    readmission = context()
    obligation = readmission.obligations[0]
    structure = (
        signed_item(
            governor,
            item_id="open-obligation",
            kind=ReturnMemoryKind.OPEN_OBLIGATION,
            text="Complete the open analysis.",
            obligation_id=obligation.obligation_id,
        ),
        signed_item(
            governor,
            item_id="progress-frontier",
            kind=ReturnMemoryKind.PROGRESS_FRONTIER,
            text="Evidence collection is complete; analysis remains open.",
            depends_on=(obligation.obligation_id,),
        ),
    )
    fact = signed_item(
        governor,
        item_id="critical-fact",
        kind=ReturnMemoryKind.VERIFIED_FACT,
        text="The verified critical value is 17.056.",
        supports=(obligation.required_dependency_ids[0],),
    )
    return readmission, structure, fact


def compile_view(
    governor: ReturnMemoryGovernor,
    items: Iterable[ReturnMemoryItem],
    readmission: ReturnReadmissionContext,
    *,
    mode: ReturnAblationMode = ReturnAblationMode.TRACE,
    expires_after_iteration: int = 3,
) -> ReturnCompilation:
    return governor.compile(
        tuple(items),
        readmission,
        source_epoch=0,
        expires_after_iteration=expires_after_iteration,
        mode=mode,
    )


def admission_row(
    *,
    case_id: str,
    category: str,
    method: ReturnAblationMode,
    governor: ReturnMemoryGovernor,
    view: SignedReturnMemoryView,
    readmission: ReturnReadmissionContext,
    expected_accepted: bool,
    invalid_attempt: bool = True,
    replay_attempt: bool = False,
) -> dict[str, Any]:
    admission = governor.admit(view, readmission)
    passed = admission.accepted is expected_accepted
    return {
        "schema_version": SCHEMA,
        "case_id": case_id,
        "category": category,
        "method": method.value,
        "operation": "admit_view",
        "expected": {
            "accepted": expected_accepted,
            "fail_closed": not expected_accepted,
        },
        "observed": {
            "accepted": admission.accepted,
            "reason": admission.reason,
            "view_id": admission.view_id,
        },
        "invalid_attempt": invalid_attempt,
        "unauthorized_candidate": False,
        "unauthorized_disclosed": False,
        "stale_candidate": False,
        "stale_reused": False,
        "replay_attempt": replay_attempt,
        "replay_accepted": replay_attempt and admission.accepted,
        "fail_closed_expected": not expected_accepted,
        "fail_closed_observed": not admission.accepted,
        "passed": passed,
    }


def compilation_row(
    *,
    case_id: str,
    category: str,
    method: ReturnAblationMode,
    compilation: ReturnCompilation,
    expected_status: ReturnSelectionStatus,
    invalid_item_id: str | None = None,
    unauthorized_candidate: bool = False,
    stale_candidate: bool = False,
    fail_closed_expected: bool,
) -> dict[str, Any]:
    selected = set(compilation.selection.selected_item_ids)
    invalid_selected = (
        invalid_item_id is not None and invalid_item_id in selected
    )
    expected_invalid_selected = (
        None
        if invalid_item_id is None
        else not fail_closed_expected
    )
    fail_closed_observed = compilation.status in {
        ReturnSelectionStatus.RESET_FALLBACK,
        ReturnSelectionStatus.BLOCKED,
    }
    return {
        "schema_version": SCHEMA,
        "case_id": case_id,
        "category": category,
        "method": method.value,
        "operation": "compile_view",
        "expected": {
            "status": expected_status.value,
            "fail_closed": fail_closed_expected,
        },
        "observed": {
            "status": compilation.status.value,
            "fallback_reason": compilation.selection.fallback_reason,
            "selected_item_ids": sorted(selected),
            "obligation_coverage": compilation.selection.obligation_coverage,
            "critical_dependency_recall": (
                compilation.selection.critical_dependency_recall
            ),
            "disclosure_cost": compilation.selection.disclosure_cost,
            "item_decisions": [
                decision.record() for decision in compilation.item_decisions
            ],
            "invalid_item_selected": invalid_selected,
        },
        "expected_invalid_item_selected": expected_invalid_selected,
        "invalid_attempt": True,
        "unauthorized_candidate": unauthorized_candidate,
        "unauthorized_disclosed": unauthorized_candidate and invalid_selected,
        "stale_candidate": stale_candidate,
        "stale_reused": stale_candidate and invalid_selected,
        "replay_attempt": False,
        "replay_accepted": False,
        "fail_closed_expected": fail_closed_expected,
        "fail_closed_observed": fail_closed_observed,
        "passed": (
            compilation.status is expected_status
            and fail_closed_observed is fail_closed_expected
            and (
                expected_invalid_selected is None
                or invalid_selected is expected_invalid_selected
            )
        ),
    }


def run_panel() -> list[dict[str, Any]]:
    governor = ReturnMemoryGovernor("trace-security-authority", b"s" * 32)
    readmission, structure, fact = fixture(governor)
    baseline = compile_view(governor, (*structure, fact), readmission)
    if baseline.view is None:
        raise RuntimeError("valid baseline view was not issued")
    view = baseline.view
    rows: list[dict[str, Any]] = []

    rows.append(
        admission_row(
            case_id="wrong_principal",
            category="invalid_admission",
            method=ReturnAblationMode.TRACE,
            governor=governor,
            view=view,
            readmission=replace(readmission, principal_id="mallory"),
            expected_accepted=False,
        )
    )
    rows.append(
        admission_row(
            case_id="old_epoch",
            category="invalid_admission",
            method=ReturnAblationMode.TRACE,
            governor=governor,
            view=view,
            readmission=replace(readmission, target_epoch=3),
            expected_accepted=False,
        )
    )
    rows.append(
        admission_row(
            case_id="expired_view",
            category="invalid_admission",
            method=ReturnAblationMode.TRACE,
            governor=governor,
            view=view,
            readmission=replace(readmission, iteration=4),
            expected_accepted=False,
        )
    )
    rows.append(
        admission_row(
            case_id="forged_signature",
            category="invalid_admission",
            method=ReturnAblationMode.TRACE,
            governor=governor,
            view=replace(view, signature="0" * 64),
            readmission=readmission,
            expected_accepted=False,
        )
    )
    rows.append(
        admission_row(
            case_id="replayed_return_view",
            category="replay",
            method=ReturnAblationMode.TRACE,
            governor=governor,
            view=view,
            readmission=replace(
                readmission,
                return_event_id="return:security:2",
                absence_id="absence:security:2",
            ),
            expected_accepted=False,
            replay_attempt=True,
        )
    )

    dependency_id = readmission.obligations[0].required_dependency_ids[0]
    cross_role = signed_item(
        governor,
        item_id="cross-role-critical",
        kind=ReturnMemoryKind.VERIFIED_FACT,
        text="Private planner-only critical value.",
        role_id="role:planner",
        supports=(dependency_id,),
    )
    stale = signed_item(
        governor,
        item_id="stale-critical",
        kind=ReturnMemoryKind.VERIFIED_FACT,
        text="A revoked old critical value.",
        supports=(dependency_id,),
        valid_from_epoch=0,
        valid_until_epoch=1,
        superseded_by="critical-fact",
    )
    missing_receipt = ReturnMemoryItem(
        item_id="unproven-critical",
        kind=ReturnMemoryKind.VERIFIED_FACT,
        text="An unproven critical value.",
        task_id=readmission.task_id,
        role_id=readmission.role_id,
        writer_principal_id="mallory",
        source_epoch=0,
        valid_from_epoch=2,
        valid_until_epoch=3,
        purposes=(readmission.purpose,),
        provenance_sha256=(hashlib.sha256(b"unbound").hexdigest(),),
        scope=ReturnMemoryScope.ROLE,
        supports=(dependency_id,),
        provenance_receipts=(),
    )

    rows.append(
        compilation_row(
            case_id="cross_role_scope",
            category="unauthorized_disclosure",
            method=ReturnAblationMode.TRACE,
            compilation=compile_view(
                governor, (*structure, cross_role), readmission
            ),
            expected_status=ReturnSelectionStatus.RESET_FALLBACK,
            invalid_item_id=cross_role.item_id,
            unauthorized_candidate=True,
            fail_closed_expected=True,
        )
    )
    rows.append(
        compilation_row(
            case_id="stale_or_superseded_item",
            category="stale_reuse",
            method=ReturnAblationMode.TRACE,
            compilation=compile_view(governor, (*structure, stale), readmission),
            expected_status=ReturnSelectionStatus.RESET_FALLBACK,
            invalid_item_id=stale.item_id,
            stale_candidate=True,
            fail_closed_expected=True,
        )
    )
    rows.append(
        compilation_row(
            case_id="missing_critical_dependency",
            category="fail_closed",
            method=ReturnAblationMode.TRACE,
            compilation=compile_view(governor, structure, readmission),
            expected_status=ReturnSelectionStatus.RESET_FALLBACK,
            fail_closed_expected=True,
        )
    )
    rows.append(
        compilation_row(
            case_id="disclosure_budget_insufficient",
            category="fail_closed",
            method=ReturnAblationMode.TRACE,
            compilation=compile_view(
                governor,
                (*structure, fact),
                replace(readmission, max_items=2),
            ),
            expected_status=ReturnSelectionStatus.RESET_FALLBACK,
            fail_closed_expected=True,
        )
    )

    rows.append(
        compilation_row(
            case_id="full_restore_cross_role",
            category="unsafe_control",
            method=ReturnAblationMode.FULL_RESTORE,
            compilation=compile_view(
                governor,
                (*structure, fact, cross_role),
                readmission,
                mode=ReturnAblationMode.FULL_RESTORE,
            ),
            expected_status=ReturnSelectionStatus.ISSUED,
            invalid_item_id=cross_role.item_id,
            unauthorized_candidate=True,
            fail_closed_expected=False,
        )
    )
    rows.append(
        compilation_row(
            case_id="without_provenance",
            category="unsafe_control",
            method=ReturnAblationMode.TRACE_WITHOUT_PROVENANCE,
            compilation=compile_view(
                governor,
                (*structure, missing_receipt),
                readmission,
                mode=ReturnAblationMode.TRACE_WITHOUT_PROVENANCE,
            ),
            expected_status=ReturnSelectionStatus.ISSUED,
            invalid_item_id=missing_receipt.item_id,
            fail_closed_expected=False,
        )
    )
    rows.append(
        compilation_row(
            case_id="without_freshness",
            category="unsafe_control",
            method=ReturnAblationMode.TRACE_WITHOUT_FRESHNESS,
            compilation=compile_view(
                governor,
                (*structure, stale),
                readmission,
                mode=ReturnAblationMode.TRACE_WITHOUT_FRESHNESS,
            ),
            expected_status=ReturnSelectionStatus.ISSUED,
            invalid_item_id=stale.item_id,
            stale_candidate=True,
            fail_closed_expected=False,
        )
    )
    rows.append(
        compilation_row(
            case_id="without_fallback",
            category="unsafe_control",
            method=ReturnAblationMode.TRACE_WITHOUT_FALLBACK,
            compilation=compile_view(
                governor,
                structure,
                readmission,
                mode=ReturnAblationMode.TRACE_WITHOUT_FALLBACK,
            ),
            expected_status=ReturnSelectionStatus.ISSUED,
            fail_closed_expected=False,
        )
    )
    return rows


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    trace = [
        row for row in rows if row["method"] == ReturnAblationMode.TRACE.value
    ]
    invalid = [row for row in trace if row["invalid_attempt"]]
    unauthorized = [row for row in trace if row["unauthorized_candidate"]]
    stale = [row for row in trace if row["stale_candidate"]]
    replay = [row for row in trace if row["replay_attempt"]]
    fail_closed = [row for row in trace if row["fail_closed_expected"]]
    by_method: dict[str, Any] = {}
    for method in (item.value for item in METHODS):
        selected = [row for row in rows if row["method"] == method]
        by_method[method] = {
            "cases": len(selected),
            "passed_expected_behavior": sum(bool(row["passed"]) for row in selected),
            "invalid_admissions": sum(
                bool(row["observed"].get("accepted"))
                for row in selected
                if row["operation"] == "admit_view"
                and row["invalid_attempt"]
            ),
            "unauthorized_disclosures": sum(
                bool(row["unauthorized_disclosed"]) for row in selected
            ),
            "stale_reuses": sum(bool(row["stale_reused"]) for row in selected),
            "replay_acceptances": sum(
                bool(row["replay_accepted"]) for row in selected
            ),
            "fail_closed_correct": sum(
                bool(row["fail_closed_observed"])
                for row in selected
                if row["fail_closed_expected"]
            ),
        }
    return {
        "schema_version": SCHEMA,
        "completed_at_utc": utc_now(),
        "cases": len(rows),
        "all_expected_behaviors_observed": all(
            bool(row["passed"]) for row in rows
        ),
        "trace_metrics": {
            "invalid_admission_rate": rate(
                sum(
                    bool(row["observed"].get("accepted"))
                    for row in invalid
                    if row["operation"] == "admit_view"
                ),
                sum(row["operation"] == "admit_view" for row in invalid),
            ),
            "unauthorized_disclosure_rate": rate(
                sum(bool(row["unauthorized_disclosed"]) for row in unauthorized),
                len(unauthorized),
            ),
            "stale_reuse_rate": rate(
                sum(bool(row["stale_reused"]) for row in stale),
                len(stale),
            ),
            "replay_acceptance_rate": rate(
                sum(bool(row["replay_accepted"]) for row in replay),
                len(replay),
            ),
            "fail_closed_correctness": rate(
                sum(bool(row["fail_closed_observed"]) for row in fail_closed),
                len(fail_closed),
            ),
        },
        "by_method": by_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("security panel output must be empty")
    source_paths = (
        Path("src/trace/return_governance.py"),
        Path("scripts/experiments/run_return_security_panel.py"),
    )
    frozen_plan = {
        "schema_version": SCHEMA,
        "created_at_utc": utc_now(),
        "deterministic": True,
        "model_calls": 0,
        "methods": [method.value for method in METHODS],
        "case_ids": [
            "wrong_principal",
            "old_epoch",
            "expired_view",
            "forged_signature",
            "replayed_return_view",
            "cross_role_scope",
            "stale_or_superseded_item",
            "missing_critical_dependency",
            "disclosure_budget_insufficient",
            "full_restore_cross_role",
            "without_provenance",
            "without_freshness",
            "without_fallback",
        ],
        "source_hashes": {
            str(path): sha256(ROOT / path) for path in source_paths
        },
    }
    write_json_once(output / "frozen_plan.json", frozen_plan)
    rows = run_panel()
    write_jsonl_once(output / "records.jsonl", rows)
    summary = summarize(rows)
    write_json_once(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
