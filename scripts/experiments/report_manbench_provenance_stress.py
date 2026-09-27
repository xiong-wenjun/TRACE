#!/usr/bin/env python3
"""Aggregate the ManBench provenance-stress matrix without mixing main runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trace.manbench_provenance import validate_stress_metadata


METRICS = (
    "final_answer_accuracy",
    "valid_information_availability",
    "invalid_information_rejection",
    "wrong_state_admission_rate",
)


def _ratio(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _new_totals(arms: tuple[str, ...]) -> dict[str, dict[str, dict[str, int]]]:
    return {
        arm: {
            metric: {"numerator": 0, "denominator": 0}
            for metric in METRICS
        }
        for arm in arms
    }


def _ratios(
    totals: dict[str, dict[str, dict[str, int]]],
) -> dict[str, dict[str, dict[str, int | float]]]:
    return {
        arm: {
            metric: _ratio(
                rows["numerator"],
                rows["denominator"],
            )
            for metric, rows in metrics.items()
        }
        for arm, metrics in totals.items()
    }


def _stress_paths(output_dir: Path) -> list[Path]:
    return sorted(
        (output_dir / "episodes" / "provenance_stress").glob(
            "*/*/*.json"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = _stress_paths(args.output_dir)
    if not paths:
        raise ValueError(
            "no provenance-stress episodes found: " + str(args.output_dir)
        )
    first = json.loads(paths[0].read_text(encoding="utf-8"))
    raw_arms = first.get("evaluated_arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("first stress artifact has no evaluated arms")
    arms = tuple(str(item) for item in raw_arms)
    if len(arms) != len(set(arms)):
        raise ValueError("stress evaluated arms must be unique")
    groups: dict[str, dict[str, Any]] = {}
    lifecycle_groups: dict[str, dict[str, Any]] = {}
    total_completed = 0
    total_eligible = 0
    for path in paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        raw_stress = artifact.get("provenance_stress")
        if not isinstance(raw_stress, dict):
            raise ValueError("stress artifact lacks provenance metadata: " + str(path))
        validate_stress_metadata(raw_stress)
        if tuple(str(item) for item in artifact.get("evaluated_arms", ())) != arms:
            raise ValueError("stress evaluated-arm mismatch: " + str(path))
        metrics = artifact.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(arms):
            raise ValueError("stress metrics/arms mismatch: " + str(path))
        case_id = str(raw_stress["case_id"])
        lifecycle = str(
            artifact.get("lifecycle_scenario", {}).get("scenario")
            if isinstance(artifact.get("lifecycle_scenario"), dict)
            else "unknown"
        )
        for group_key, group_store in (
            (case_id, groups),
            (f"{case_id}|{lifecycle}", lifecycle_groups),
        ):
            if group_key not in group_store:
                group_store[group_key] = {
                    "scenario": raw_stress["scenario"],
                    "active_agent_count": int(
                        raw_stress["active_agent_count"]
                    ),
                    "lifecycle_scenario": lifecycle,
                    "completed_episodes": 0,
                    "eligible_episodes": 0,
                    "totals": _new_totals(arms),
                }
            group = group_store[group_key]
            group["completed_episodes"] += 1
            eligible = artifact.get("return_evaluation_eligible") is True
            group["eligible_episodes"] += int(eligible)
            total_completed += int(group_store is groups)
            total_eligible += int(eligible and group_store is groups)
            for arm in arms:
                for metric in METRICS:
                    row = metrics[arm].get(metric)
                    if not isinstance(row, dict):
                        raise ValueError(f"missing {arm}/{metric}: {path}")
                    group["totals"][arm][metric]["numerator"] += int(
                        row.get("numerator", 0)
                    )
                    group["totals"][arm][metric]["denominator"] += int(
                        row.get("denominator", 0)
                    )
    for group in groups.values():
        group["results"] = _ratios(group.pop("totals"))
    for group in lifecycle_groups.values():
        group["results"] = _ratios(group.pop("totals"))
    report = {
        "schema_version": "manbench_provenance_stress_statistics",
        "completed_episodes": total_completed,
        "eligible_episodes": total_eligible,
        "evaluated_arms": list(arms),
        "cell_results": groups,
        "lifecycle_cell_results": lifecycle_groups,
        "grouping": {
            "provenance_scenario": "four matched evidence profiles",
            "n_definition": "active_agents_during_absence",
            "departing_agent_count": 1,
        },
    }
    path = args.output_dir / "provenance_stress_statistics.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
