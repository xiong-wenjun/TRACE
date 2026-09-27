#!/usr/bin/env python3
"""Report paired, persona-period clustered Memora RETURN statistics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace.memora_statistics import (  # noqa: E402
    holm_adjust,
    paired_cluster_bootstrap,
)
from trace.trace_method import (  # noqa: E402
    TRACE_ARM,
    TRACE_DISPLAY_NAME,
    canonicalize_arm_mapping,
)
from trace.router_return_protocol import RouterReturnArm  # noqa: E402


SCHEMA = "memora_return_clustered_statistics_v1"
METRICS = (
    "fama",
    "memory_presence_accuracy",
    "forgetting_absence_accuracy",
    "overall_accuracy",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rows(panel: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    episode_dir = panel / "episodes" if (panel / "episodes").is_dir() else panel
    paths = sorted(episode_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    selected_paths: list[Path] = []
    for path in paths:
        if path.name.endswith(".failure.json"):
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") not in {
            "memora_mas_return_run_v1",
            "memora_mas_return_run_v2",
            "memora_mas_return_run_v3",
        }:
            continue
        episode = value.get("episode")
        raw_arms = value.get("arms")
        if not isinstance(episode, Mapping) or not isinstance(
            raw_arms, Mapping
        ):
            continue
        arms = canonicalize_arm_mapping(raw_arms)
        if not arms:
            continue
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "cluster_id": episode["cluster_id"],
                "period": episode["period"],
                "persona": episode["persona"],
                "task_type": episode["task_type"],
                "question_id": episode["question_id"],
                "arms": arms,
                "state_mode": value.get("state_mode", "oracle"),
                "fork_audit": (
                    value.get("fork_bundle", {}).get("audit", {})
                    if isinstance(value.get("fork_bundle"), Mapping)
                    else {}
                ),
                "prediction_quality": value.get(
                    "prediction_quality_evaluator_only"
                ),
            }
        )
        selected_paths.append(path)
    return rows, selected_paths


def _score(
    row: Mapping[str, object], arm: str, metric: str
) -> float:
    arms = row["arms"]
    assert isinstance(arms, Mapping)
    arm_row = arms[arm]
    assert isinstance(arm_row, Mapping)
    score = arm_row["score"]
    assert isinstance(score, Mapping)
    return float(score[metric])


def _complete(
    rows: Sequence[Mapping[str, object]], arms: Sequence[str]
) -> list[Mapping[str, object]]:
    result = []
    for row in rows:
        arm_rows = row.get("arms")
        if not isinstance(arm_rows, Mapping):
            continue
        if not all(
            arm in arm_rows
            and isinstance(arm_rows[arm], Mapping)
            and isinstance(arm_rows[arm].get("score"), Mapping)
            for arm in arms
        ):
            continue
        result.append(row)
    return result


def _mean(
    rows: Sequence[Mapping[str, object]], arm: str, metric: str
) -> float:
    return sum(_score(row, arm, metric) for row in rows) / len(rows)


def _strata(
    rows: Sequence[Mapping[str, object]],
    *,
    arms: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in ("period", "task_type"):
        groups: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            groups.setdefault(str(row[field]), []).append(row)
        result[field] = {
            name: {
                "questions": len(group),
                "clusters": len(
                    {str(item["cluster_id"]) for item in group}
                ),
                "means": {
                    arm: {
                        metric: _mean(group, arm, metric)
                        for metric in METRICS
                    }
                    for arm in arms
                },
            }
            for name, group in sorted(groups.items())
        }
    return result


def _comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    treatment: str,
    comparator: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    clusters = {str(row["cluster_id"]) for row in rows}
    if len(clusters) < 2:
        return {
            "calculable": False,
            "reason": "fewer_than_two_persona_period_clusters",
            "paired_questions": len(rows),
            "clusters": len(clusters),
        }
    return {
        "calculable": True,
        **paired_cluster_bootstrap(
            rows,
            treatment=treatment,
            comparator=comparator,
            metric=metric,
            replicates=replicates,
            seed=seed,
        ),
    }


def _predicted_diagnostics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    predicted = [
        row for row in rows if row.get("state_mode") == "predicted"
    ]
    if not predicted:
        return None
    metric_names = (
        "fact_extraction_precision",
        "fact_extraction_recall",
        "dependency_session_precision",
        "dependency_session_recall",
        "update_delete_session_precision",
        "update_delete_session_recall",
    )
    means: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for name in metric_names:
        values = [
            float(quality[name])
            for row in predicted
            for quality in (row.get("prediction_quality"),)
            if isinstance(quality, Mapping)
            and quality.get(name) is not None
        ]
        means[name] = sum(values) / len(values) if values else None
        counts[name] = len(values)
    eligible = 0
    stale_detected = 0
    degraded = 0
    continuity_fallback = 0
    for row in predicted:
        audit = row.get("fork_audit")
        if not isinstance(audit, Mapping):
            continue
        eligible += int(bool(audit.get("event_eligible")))
        degraded += int(bool(audit.get("prediction_degraded")))
        continuity_fallback += int(
            bool(audit.get("deterministic_predeparture_fallback_used"))
        )
        stale_detected += int(
            int(audit.get("predicted_stale_count") or 0) > 0
        )
    return {
        "questions": len(predicted),
        "event_eligible_questions": eligible,
        "event_eligible_rate": eligible / len(predicted),
        "prediction_degraded_questions": degraded,
        "prediction_degraded_rate": degraded / len(predicted),
        "continuity_fallback_questions": continuity_fallback,
        "continuity_fallback_rate": continuity_fallback / len(predicted),
        "questions_with_predicted_stale_state": stale_detected,
        "questions_with_predicted_stale_state_rate": (
            stale_detected / len(predicted)
        ),
        "evaluator_only_quality_means": means,
        "metric_observation_counts": counts,
        "official_evidence_consumed_at_runtime": False,
    }


def _divergence_witness_subset(
    rows: Sequence[Mapping[str, object]],
    *,
    arms: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    eligible = [
        row
        for row in rows
        if isinstance(row.get("fork_audit"), Mapping)
        and bool(row["fork_audit"].get("event_eligible"))
    ]
    if not eligible:
        return {
            "questions": 0,
            "calculable": False,
            "reason": "no_predicted_divergence_witness",
        }
    means = {
        arm: {metric: _mean(eligible, arm, metric) for metric in METRICS}
        for arm in arms
    }
    static_gap = (
        means[RouterReturnArm.STATIC.value]["fama"]
        - means[RouterReturnArm.RESTORE_OLD.value]["fama"]
    )
    trace_gain = (
        means[TRACE_ARM]["fama"]
        - means[RouterReturnArm.RESTORE_OLD.value]["fama"]
    )
    return {
        "questions": len(eligible),
        "clusters": len({str(row["cluster_id"]) for row in eligible}),
        "calculable": True,
        "means": means,
        "trace_vs_restore_old_fama": _comparison(
            eligible,
            treatment=TRACE_ARM,
            comparator=RouterReturnArm.RESTORE_OLD.value,
            metric="fama",
            replicates=replicates,
            seed=seed,
        ),
        "trace_vs_reset_fama": _comparison(
            eligible,
            treatment=TRACE_ARM,
            comparator=RouterReturnArm.RESET.value,
            metric="fama",
            replicates=replicates,
            seed=seed + 1,
        ),
        "recovery_ratio": (
            trace_gain / static_gap if static_gap > 0.0 else None
        ),
        "selection_rule": (
            "actor-visible predicted V/X/N gate frozen before official scoring"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument(
        "--noninferiority-margin",
        type=float,
        default=-0.05,
        help="Preregistered TRACE-minus-Static FAMA margin.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, paths = _load_rows(args.panel.resolve())
    arms = (
        RouterReturnArm.STATIC.value,
        RouterReturnArm.RESET.value,
        RouterReturnArm.RESTORE_OLD.value,
        TRACE_ARM,
    )
    complete = _complete(rows, arms)
    if not complete:
        raise RuntimeError("no complete four-arm Memora episodes")
    means = {
        arm: {metric: _mean(complete, arm, metric) for metric in METRICS}
        for arm in arms
    }
    comparisons: dict[str, object] = {}
    specs = (
        (
            "trace_vs_restore_old_fama",
            TRACE_ARM,
            RouterReturnArm.RESTORE_OLD.value,
            "fama",
        ),
        (
            "trace_vs_reset_fama",
            TRACE_ARM,
            RouterReturnArm.RESET.value,
            "fama",
        ),
        (
            "trace_vs_reset_memory_presence",
            TRACE_ARM,
            RouterReturnArm.RESET.value,
            "memory_presence_accuracy",
        ),
        (
            "trace_vs_restore_old_forgetting",
            TRACE_ARM,
            RouterReturnArm.RESTORE_OLD.value,
            "forgetting_absence_accuracy",
        ),
        (
            "trace_vs_static_fama",
            TRACE_ARM,
            RouterReturnArm.STATIC.value,
            "fama",
        ),
    )
    for index, (name, treatment, comparator, metric) in enumerate(specs):
        comparisons[name] = _comparison(
            complete,
            treatment=treatment,
            comparator=comparator,
            metric=metric,
            replicates=args.bootstrap_replicates,
            seed=args.seed + index,
        )
    primary_names = (
        "trace_vs_restore_old_fama",
        "trace_vs_reset_fama",
    )
    primary_p = {
        name: float(comparisons[name]["significance"]["p_value"])
        for name in primary_names
        if comparisons[name].get("calculable")
    }
    adjusted = holm_adjust(primary_p) if primary_p else {}
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p_value"] = value
    static_restore_gap = (
        means[RouterReturnArm.STATIC.value]["fama"]
        - means[RouterReturnArm.RESTORE_OLD.value]["fama"]
    )
    trace_restore_gain = (
        means[TRACE_ARM]["fama"]
        - means[RouterReturnArm.RESTORE_OLD.value]["fama"]
    )
    recovery_ratio = (
        trace_restore_gain / static_restore_gap
        if static_restore_gap > 0.0
        else None
    )
    trace_static = comparisons["trace_vs_static_fama"]
    noninferiority = {
        "margin_trace_minus_static": args.noninferiority_margin,
        "calculable": bool(trace_static.get("calculable")),
        "one_sided_95_lower_bound": (
            trace_static.get("one_sided_95_lower_bound")
            if trace_static.get("calculable")
            else None
        ),
        "noninferior": (
            float(trace_static["one_sided_95_lower_bound"])
            > args.noninferiority_margin
            if trace_static.get("calculable")
            else None
        ),
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "panel": str(args.panel.resolve()),
        "source_files": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
        "complete_paired_questions": len(complete),
        "persona_period_clusters": len(
            {str(row["cluster_id"]) for row in complete}
        ),
        "personas": len({str(row["persona"]) for row in complete}),
        "arms": list(arms),
        "method_display_names": {
            RouterReturnArm.STATIC.value: "Static",
            RouterReturnArm.RESET.value: "Reset",
            RouterReturnArm.RESTORE_OLD.value: "Restore",
            TRACE_ARM: (
                f"{TRACE_DISPLAY_NAME}-Predicted"
                if all(
                    row.get("state_mode") == "predicted" for row in complete
                )
                else f"{TRACE_DISPLAY_NAME}-Oracle"
            ),
        },
        "state_modes": sorted(
            {str(row.get("state_mode") or "oracle") for row in complete}
        ),
        "means": means,
        "comparisons": comparisons,
        "primary_holm_family": list(primary_names),
        "recovery_ratio": {
            "formula": (
                "(FAMA_TRACE-FAMA_RestoreOld)/"
                "(FAMA_Static-FAMA_RestoreOld)"
            ),
            "denominator_policy": (
                "defined_only_when_static_minus_restore_old_is_positive"
            ),
            "static_minus_restore_old": static_restore_gap,
            "trace_minus_restore_old": trace_restore_gain,
            "value": recovery_ratio,
            "calculable": recovery_ratio is not None
            and math.isfinite(recovery_ratio),
        },
        "noninferiority_trace_vs_static": noninferiority,
        "stratified": _strata(complete, arms=arms),
        "predicted_diagnostics": _predicted_diagnostics(complete),
        "divergence_witness_subset": _divergence_witness_subset(
            complete,
            arms=arms,
            replicates=args.bootstrap_replicates,
            seed=args.seed + 100,
        ),
        "independence_warning": (
            "Questions are not treated as iid; inference resamples paired "
            "persona-period clusters."
        ),
    }
    output = args.output or (
        args.panel.resolve() / "memora_clustered_statistics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
