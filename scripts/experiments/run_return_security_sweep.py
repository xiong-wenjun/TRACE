#!/usr/bin/env python3
"""Run a deterministic parameter sweep over TRACE RETURN security checks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import run_return_security_panel as base  # noqa: E402


SCHEMA = "trace_return_security_sweep_v1"
FAMILIES = (
    "wrong_principal",
    "epoch_mismatch",
    "expired_view",
    "forged_signature",
    "replayed_view",
    "cross_role_scope",
    "stale_item",
    "missing_provenance",
    "missing_dependency",
    "insufficient_budget",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_once(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def unproven_item(
    *,
    item_id: str,
    readmission: base.ReturnReadmissionContext,
    dependency_id: str,
) -> base.ReturnMemoryItem:
    return base.ReturnMemoryItem(
        item_id=item_id,
        kind=base.ReturnMemoryKind.VERIFIED_FACT,
        text=f"Unproven critical value for {item_id}.",
        task_id=readmission.task_id,
        role_id=readmission.role_id,
        writer_principal_id=f"untrusted:{item_id}",
        source_epoch=0,
        valid_from_epoch=2,
        valid_until_epoch=3,
        purposes=(readmission.purpose,),
        provenance_sha256=(hashlib.sha256(item_id.encode()).hexdigest(),),
        scope=base.ReturnMemoryScope.ROLE,
        supports=(dependency_id,),
        provenance_receipts=(),
    )


def run_sweep(variants: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(variants):
        governor = base.ReturnMemoryGovernor(
            f"trace-security-sweep-authority-{index}",
            hashlib.sha256(f"sweep-key-{index}".encode()).digest(),
        )
        readmission, structure, fact = base.fixture(governor)
        baseline = base.compile_view(
            governor, (*structure, fact), readmission
        )
        if baseline.view is None:
            raise RuntimeError("valid sweep baseline view was not issued")
        view = baseline.view
        suffix = f"{index:04d}"
        rows.extend(
            (
                base.admission_row(
                    case_id=f"wrong_principal:{suffix}",
                    category="wrong_principal",
                    method=base.ReturnAblationMode.TRACE,
                    governor=governor,
                    view=view,
                    readmission=replace(
                        readmission, principal_id=f"mallory-{suffix}"
                    ),
                    expected_accepted=False,
                ),
                base.admission_row(
                    case_id=f"epoch_mismatch:{suffix}",
                    category="epoch_mismatch",
                    method=base.ReturnAblationMode.TRACE,
                    governor=governor,
                    view=view,
                    readmission=replace(
                        readmission, target_epoch=3 + index
                    ),
                    expected_accepted=False,
                ),
                base.admission_row(
                    case_id=f"expired_view:{suffix}",
                    category="expired_view",
                    method=base.ReturnAblationMode.TRACE,
                    governor=governor,
                    view=view,
                    readmission=replace(
                        readmission, iteration=4 + index
                    ),
                    expected_accepted=False,
                ),
                base.admission_row(
                    case_id=f"forged_signature:{suffix}",
                    category="forged_signature",
                    method=base.ReturnAblationMode.TRACE,
                    governor=governor,
                    view=replace(
                        view,
                        signature=hashlib.sha256(
                            f"forged-{index}".encode()
                        ).hexdigest(),
                    ),
                    readmission=readmission,
                    expected_accepted=False,
                ),
                base.admission_row(
                    case_id=f"replayed_view:{suffix}",
                    category="replayed_view",
                    method=base.ReturnAblationMode.TRACE,
                    governor=governor,
                    view=view,
                    readmission=replace(
                        readmission,
                        return_event_id=f"return:replay:{suffix}",
                        absence_id=f"absence:replay:{suffix}",
                    ),
                    expected_accepted=False,
                    replay_attempt=True,
                ),
            )
        )
        dependency_id = (
            readmission.obligations[0].required_dependency_ids[0]
        )
        cross_role = base.signed_item(
            governor,
            item_id=f"cross-role-critical-{suffix}",
            kind=base.ReturnMemoryKind.VERIFIED_FACT,
            text=f"Cross-role secret {suffix}.",
            role_id=f"role:planner:{suffix}",
            supports=(dependency_id,),
        )
        stale = base.signed_item(
            governor,
            item_id=f"stale-critical-{suffix}",
            kind=base.ReturnMemoryKind.VERIFIED_FACT,
            text=f"Superseded value {suffix}.",
            supports=(dependency_id,),
            valid_from_epoch=0,
            valid_until_epoch=1,
            superseded_by=f"current-critical-{suffix}",
        )
        unproven = unproven_item(
            item_id=f"unproven-critical-{suffix}",
            readmission=readmission,
            dependency_id=dependency_id,
        )
        rows.extend(
            (
                base.compilation_row(
                    case_id=f"cross_role_scope:{suffix}",
                    category="cross_role_scope",
                    method=base.ReturnAblationMode.TRACE,
                    compilation=base.compile_view(
                        governor, (*structure, cross_role), readmission
                    ),
                    expected_status=base.ReturnSelectionStatus.RESET_FALLBACK,
                    invalid_item_id=cross_role.item_id,
                    unauthorized_candidate=True,
                    fail_closed_expected=True,
                ),
                base.compilation_row(
                    case_id=f"stale_item:{suffix}",
                    category="stale_item",
                    method=base.ReturnAblationMode.TRACE,
                    compilation=base.compile_view(
                        governor, (*structure, stale), readmission
                    ),
                    expected_status=base.ReturnSelectionStatus.RESET_FALLBACK,
                    invalid_item_id=stale.item_id,
                    stale_candidate=True,
                    fail_closed_expected=True,
                ),
                base.compilation_row(
                    case_id=f"missing_provenance:{suffix}",
                    category="missing_provenance",
                    method=base.ReturnAblationMode.TRACE,
                    compilation=base.compile_view(
                        governor, (*structure, unproven), readmission
                    ),
                    expected_status=base.ReturnSelectionStatus.RESET_FALLBACK,
                    invalid_item_id=unproven.item_id,
                    fail_closed_expected=True,
                ),
                base.compilation_row(
                    case_id=f"missing_dependency:{suffix}",
                    category="missing_dependency",
                    method=base.ReturnAblationMode.TRACE,
                    compilation=base.compile_view(
                        governor, structure, readmission
                    ),
                    expected_status=base.ReturnSelectionStatus.RESET_FALLBACK,
                    fail_closed_expected=True,
                ),
                base.compilation_row(
                    case_id=f"insufficient_budget:{suffix}",
                    category="insufficient_budget",
                    method=base.ReturnAblationMode.TRACE,
                    compilation=base.compile_view(
                        governor,
                        (*structure, fact),
                        replace(readmission, max_items=2),
                    ),
                    expected_status=base.ReturnSelectionStatus.RESET_FALLBACK,
                    fail_closed_expected=True,
                ),
            )
        )
        rows.extend(
            (
                base.compilation_row(
                    case_id=f"full_restore_cross_role:{suffix}",
                    category="unsafe_control_full_restore",
                    method=base.ReturnAblationMode.FULL_RESTORE,
                    compilation=base.compile_view(
                        governor,
                        (*structure, fact, cross_role),
                        readmission,
                        mode=base.ReturnAblationMode.FULL_RESTORE,
                    ),
                    expected_status=base.ReturnSelectionStatus.ISSUED,
                    invalid_item_id=cross_role.item_id,
                    unauthorized_candidate=True,
                    fail_closed_expected=False,
                ),
                base.compilation_row(
                    case_id=f"without_provenance:{suffix}",
                    category="unsafe_control_without_provenance",
                    method=base.ReturnAblationMode.TRACE_WITHOUT_PROVENANCE,
                    compilation=base.compile_view(
                        governor,
                        (*structure, unproven),
                        readmission,
                        mode=base.ReturnAblationMode.TRACE_WITHOUT_PROVENANCE,
                    ),
                    expected_status=base.ReturnSelectionStatus.ISSUED,
                    invalid_item_id=unproven.item_id,
                    fail_closed_expected=False,
                ),
                base.compilation_row(
                    case_id=f"without_freshness:{suffix}",
                    category="unsafe_control_without_freshness",
                    method=base.ReturnAblationMode.TRACE_WITHOUT_FRESHNESS,
                    compilation=base.compile_view(
                        governor,
                        (*structure, stale),
                        readmission,
                        mode=base.ReturnAblationMode.TRACE_WITHOUT_FRESHNESS,
                    ),
                    expected_status=base.ReturnSelectionStatus.ISSUED,
                    invalid_item_id=stale.item_id,
                    stale_candidate=True,
                    fail_closed_expected=False,
                ),
                base.compilation_row(
                    case_id=f"without_fallback:{suffix}",
                    category="unsafe_control_without_fallback",
                    method=base.ReturnAblationMode.TRACE_WITHOUT_FALLBACK,
                    compilation=base.compile_view(
                        governor,
                        structure,
                        readmission,
                        mode=base.ReturnAblationMode.TRACE_WITHOUT_FALLBACK,
                    ),
                    expected_status=base.ReturnSelectionStatus.ISSUED,
                    fail_closed_expected=False,
                ),
            )
        )
    return rows


def safety_violation(row: Mapping[str, Any]) -> bool:
    if row["operation"] == "admit_view":
        return bool(row["observed"].get("accepted"))
    if row.get("unauthorized_candidate"):
        return bool(row.get("unauthorized_disclosed"))
    if row.get("stale_candidate"):
        return bool(row.get("stale_reused"))
    if row.get("expected_invalid_item_selected") is False:
        return bool(row["observed"].get("invalid_item_selected"))
    return not bool(row.get("fail_closed_observed"))


def wilson_upper(errors: int, trials: int, z: float = 1.6448536269514722) -> float:
    if trials <= 0:
        raise ValueError("Wilson interval requires trials")
    proportion = errors / trials
    denominator = 1 + z * z / trials
    center = proportion + z * z / (2 * trials)
    radius = z * math.sqrt(
        proportion * (1 - proportion) / trials
        + z * z / (4 * trials * trials)
    )
    return min(1.0, (center + radius) / denominator)


def summarize(rows: list[Mapping[str, Any]], variants: int) -> dict[str, Any]:
    trace = [
        row
        for row in rows
        if row["method"] == base.ReturnAblationMode.TRACE.value
    ]
    by_family: dict[str, Any] = {}
    for family in FAMILIES:
        selected = [row for row in trace if row["category"] == family]
        errors = sum(safety_violation(row) for row in selected)
        by_family[family] = {
            "trials": len(selected),
            "violations": errors,
            "violation_rate": errors / len(selected),
            "one_sided_95_wilson_upper": wilson_upper(
                errors, len(selected)
            ),
            "all_expected_behaviors_observed": all(
                bool(row["passed"]) for row in selected
            ),
        }
    controls = [
        row
        for row in rows
        if row["method"] != base.ReturnAblationMode.TRACE.value
    ]
    return {
        "schema_version": SCHEMA,
        "completed_at_utc": utc_now(),
        "variants_per_family": variants,
        "trace_trials": len(trace),
        "trace_violations": sum(safety_violation(row) for row in trace),
        "all_trace_expected_behaviors_observed": all(
            bool(row["passed"]) for row in trace
        ),
        "unsafe_control_trials": len(controls),
        "all_unsafe_controls_exposed_expected_weakness": all(
            bool(row["passed"]) for row in controls
        ),
        "by_family": by_family,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants-per-family", type=int, default=50)
    args = parser.parse_args()
    if args.variants_per_family < 1:
        raise ValueError("variants-per-family must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = run_sweep(args.variants_per_family)
    summary = summarize(rows, args.variants_per_family)
    frozen = {
        "schema_version": SCHEMA,
        "created_at_utc": utc_now(),
        "deterministic": True,
        "model_calls": 0,
        "variants_per_family": args.variants_per_family,
        "families": list(FAMILIES),
        "source_sha256": {
            "run_return_security_sweep.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "run_return_security_panel.py": hashlib.sha256(
                (SCRIPT_DIR / "run_return_security_panel.py").read_bytes()
            ).hexdigest(),
            "return_governance.py": hashlib.sha256(
                (ROOT / "src/trace/return_governance.py").read_bytes()
            ).hexdigest(),
        },
    }
    write_once(
        output / "frozen_plan.json",
        json.dumps(frozen, indent=2, ensure_ascii=False, sort_keys=True).encode()
        + b"\n",
    )
    write_once(
        output / "records.jsonl",
        b"".join(canonical_bytes(row) + b"\n" for row in rows),
    )
    write_once(
        output / "summary.json",
        json.dumps(
            summary, indent=2, ensure_ascii=False, sort_keys=True
        ).encode()
        + b"\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
