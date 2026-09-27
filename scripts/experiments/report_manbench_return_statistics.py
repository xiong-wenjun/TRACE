#!/usr/bin/env python3
"""Aggregate balanced Old-Valid/Old-Stale ManBench Return metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace.return_methods import (
    ANSWER_TIME_CONTROL_ARMS,
    method_display_name,
)


METRICS = (
    "final_answer_accuracy",
    "valid_information_availability",
    "invalid_information_rejection",
    "wrong_state_admission_rate",
)
SCENARIOS = ("old_valid", "old_stale")


def _evaluation_arms(
    artifact: dict[str, object], path: Path
) -> tuple[str, ...]:
    raw_arms = artifact.get("evaluated_arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError(f"missing evaluated arms: {path}")
    arms = tuple(str(arm) for arm in raw_arms)
    if len(arms) != len(set(arms)):
        raise ValueError(f"duplicate evaluated arms: {path}")
    raw_metrics = artifact.get("metrics")
    if not isinstance(raw_metrics, dict) or set(raw_metrics) != set(arms):
        raise ValueError(f"evaluated arms/metrics mismatch: {path}")
    return arms


def _new_totals(
    arms: tuple[str, ...],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        arm: {
            metric: {"numerator": 0, "denominator": 0}
            for metric in METRICS
        }
        for arm in arms
    }


def _ratios(
    totals: dict[str, dict[str, dict[str, int]]],
    arms: tuple[str, ...],
) -> dict[str, dict[str, dict[str, int | float]]]:
    results: dict[str, dict[str, dict[str, int | float]]] = {}
    for arm in arms:
        results[arm] = {}
        for metric in METRICS:
            numerator = totals[arm][metric]["numerator"]
            denominator = totals[arm][metric]["denominator"]
            results[arm][metric] = {
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator if denominator else 0.0,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact_paths = sorted((args.output_dir / "episodes").glob("*/*.json"))
    if not artifact_paths:
        raise ValueError(f"no ManBench Return episodes found: {args.output_dir}")
    first_artifact = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
    arms = _evaluation_arms(first_artifact, artifact_paths[0])
    totals = _new_totals(arms)
    scenario_totals = {
        scenario: _new_totals(arms) for scenario in SCENARIOS
    }
    scenario_counts = {scenario: 0 for scenario in SCENARIOS}
    eligible_scenario_counts = {scenario: 0 for scenario in SCENARIOS}
    task_counts: dict[str, int] = {}
    eligible_task_counts: dict[str, int] = {}
    exclusion_counts: dict[str, int] = {}
    eligible_episodes = 0
    baseline_correct_episodes = 0
    for path in artifact_paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("schema_version") != "manbench_balanced_return_episode_v4":
            raise ValueError(f"unexpected episode schema: {path}")
        if tuple(artifact.get("primary_metrics") or ()) != METRICS:
            raise ValueError(f"unexpected metric contract: {path}")
        if _evaluation_arms(artifact, path) != arms:
            raise ValueError(f"inconsistent evaluated arms: {path}")
        if artifact.get("gold_anchor_injected") is not False:
            raise ValueError(f"gold-derived anchor is forbidden: {path}")
        eligible = artifact.get("return_evaluation_eligible")
        if not isinstance(eligible, bool):
            raise ValueError(f"missing baseline eligibility: {path}")
        baseline = artifact.get("baseline_reality")
        if not isinstance(baseline, dict):
            raise ValueError(f"missing baseline reality receipt: {path}")
        if baseline.get("gold_label_visible_to_actor") is not False:
            raise ValueError(f"baseline actor saw the gold label: {path}")
        if baseline.get("evaluator_correct") is True:
            baseline_correct_episodes += 1
        lifecycle = artifact.get("lifecycle_scenario")
        if not isinstance(lifecycle, dict):
            raise ValueError(f"missing lifecycle scenario: {path}")
        scenario = str(lifecycle.get("scenario"))
        if scenario not in SCENARIOS:
            raise ValueError(f"unexpected lifecycle scenario: {path}")
        if lifecycle.get("gold_visible_to_actor") is not False:
            raise ValueError(f"scenario exposed gold to actor: {path}")
        scenario_counts[scenario] += 1
        task_name = str(artifact["example"]["task_name"])
        task_counts[task_name] = task_counts.get(task_name, 0) + 1
        if eligible:
            eligible_episodes += 1
            eligible_scenario_counts[scenario] += 1
            eligible_task_counts[task_name] = (
                eligible_task_counts.get(task_name, 0) + 1
            )
            mas = artifact.get("mas_execution")
            if not isinstance(mas, dict):
                raise ValueError(f"missing unified MAS receipt: {path}")
            architecture = mas.get("architecture")
            if not isinstance(architecture, dict) or (
                architecture.get("execution_contract")
                != "five_task_agent_single_return_v1"
            ):
                raise ValueError(f"unexpected MAS contract: {path}")
            anchor = artifact.get("departure_anchor")
            if not isinstance(anchor, dict) or (
                anchor.get("source") != "agent1_baseline_response"
                or anchor.get("gold_injected") is not False
            ):
                raise ValueError(f"invalid departure anchor: {path}")
            candidate_pool = artifact.get("candidate_pool")
            if not isinstance(candidate_pool, dict) or (
                candidate_pool.get("same_frozen_pool_for_all_arms") is not True
                or candidate_pool.get("item_ids")
                != ["answer-old", "answer-challenger"]
            ):
                raise ValueError(f"arms did not share one candidate pool: {path}")
            truth = artifact.get("ground_truth_state_ids")
            if not isinstance(truth, dict) or (
                truth.get("visible_to_actors") is not False
            ):
                raise ValueError(f"invalid hidden state labels: {path}")
            if scenario == "old_stale":
                revision = artifact.get("task_revision")
                update = artifact.get("active_group_update")
                if not isinstance(revision, dict) or (
                    revision.get("occurred") is not True
                    or revision.get("gold_derived") is not False
                ):
                    raise ValueError(f"invalid task revision: {path}")
                if not isinstance(update, dict) or (
                    update.get("evaluator_current_answer_correct") is not True
                    or update.get("gold_label_visible_to_active_agents") is not False
                ):
                    raise ValueError(f"invalid active update: {path}")
        else:
            if artifact.get("mas_execution") is not None:
                raise ValueError(f"excluded question executed Return: {path}")
            reason = str(artifact.get("eligibility_reason"))
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
        taxonomy = artifact.get("method_taxonomy")
        answer_time_prompt_contract = (
            taxonomy.get("answer_time_prompt_contract")
            if isinstance(taxonomy, dict)
            else None
        )
        for arm in arms:
            for metric in METRICS:
                row = artifact["metrics"][arm][metric]
                is_answer_time_admission_metric = (
                    arm in ANSWER_TIME_CONTROL_ARMS
                    and metric != "final_answer_accuracy"
                    and answer_time_prompt_contract
                    == "same_raw_candidate_pool_neutral_base_prompt_v1"
                )
                expected_denominator = (
                    1
                    if eligible and not is_answer_time_admission_metric
                    else 0
                )
                if int(row["denominator"]) != expected_denominator:
                    raise ValueError(
                        f"eligibility denominator mismatch for {arm}/{metric}: "
                        f"{path}"
                    )
                totals[arm][metric]["numerator"] += int(row["numerator"])
                totals[arm][metric]["denominator"] += int(row["denominator"])
                scenario_totals[scenario][arm][metric]["numerator"] += int(
                    row["numerator"]
                )
                scenario_totals[scenario][arm][metric]["denominator"] += int(
                    row["denominator"]
                )
        if eligible:
            for arm in arms:
                iir = artifact["metrics"][arm]["invalid_information_rejection"]
                wsar = artifact["metrics"][arm]["wrong_state_admission_rate"]
                if int(iir["denominator"]) == 0:
                    continue
                if int(iir["numerator"]) + int(wsar["numerator"]) != 1:
                    raise ValueError(f"IIR/WSAR complement violated: {path}")
    results = _ratios(totals, arms)
    results_by_scenario = {
        scenario: _ratios(scenario_totals[scenario], arms)
        for scenario in SCENARIOS
    }
    balanced_macro_results: dict[str, dict[str, dict[str, object]]] = {}
    for arm in arms:
        balanced_macro_results[arm] = {}
        for metric in METRICS:
            values = [
                float(results_by_scenario[scenario][arm][metric]["value"])
                for scenario in SCENARIOS
                if int(
                    results_by_scenario[scenario][arm][metric]["denominator"]
                )
                > 0
            ]
            balanced_macro_results[arm][metric] = {
                "scenario_values": {
                    scenario: results_by_scenario[scenario][arm][metric]["value"]
                    for scenario in SCENARIOS
                },
                "value": sum(values) / len(values) if values else 0.0,
                "scenario_count": len(values),
            }
    failures = len(list((args.output_dir / "failures").glob("*/*.json")))
    completed_episodes = len(artifact_paths)
    report = {
        "schema_version": "manbench_balanced_return_statistics_v3",
        "eligibility_contract": (
            "departure_anchor_correct_and_old_stale_active_update_correct"
        ),
        "completed_episodes": completed_episodes,
        "eligible_return_episodes": eligible_episodes,
        "excluded_episodes": completed_episodes - eligible_episodes,
        "baseline_accuracy": {
            "numerator": baseline_correct_episodes,
            "denominator": completed_episodes,
        },
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "failure_files": failures,
        "task_counts": dict(sorted(task_counts.items())),
        "eligible_task_counts": dict(sorted(eligible_task_counts.items())),
        "scenario_counts": scenario_counts,
        "eligible_scenario_counts": eligible_scenario_counts,
        "evaluated_arms": list(arms),
        "method_display_names": {
            arm: method_display_name(arm) for arm in arms
        },
        "primary_metrics": list(METRICS),
        "micro_results": results,
        "results_by_scenario": results_by_scenario,
        "balanced_macro_results": balanced_macro_results,
    }
    report["baseline_accuracy"]["value"] = (
        report["baseline_accuracy"]["numerator"] / completed_episodes
        if completed_episodes
        else 0.0
    )
    path = args.output_dir / "statistics.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
