#!/usr/bin/env python3
"""Report Memora annotator agreement and GPT Judge calibration."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_annotations(
    path: Path,
) -> tuple[dict[str, bool], dict[str, Mapping[str, str]]]:
    labels: dict[str, bool] = {}
    rows: dict[str, Mapping[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = str(row["item_id"])
            if item_id in labels:
                raise ValueError(f"duplicate item ID in {path}: {item_id}")
            answer = str(row["human_answer_yes_or_no"]).strip().lower()
            if answer not in {"yes", "no"}:
                raise ValueError(
                    f"{path}:{item_id} must have a yes/no human answer"
                )
            labels[item_id] = answer == "yes"
            rows[item_id] = row
    if not labels:
        raise ValueError(f"{path} contains no annotations")
    return labels, rows


def _cohen_kappa(left: Sequence[bool], right: Sequence[bool]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa requires non-empty paired labels")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_yes = sum(left) / len(left)
    right_yes = sum(right) / len(right)
    expected = left_yes * right_yes + (1 - left_yes) * (1 - right_yes)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _classification(
    predicted: Sequence[bool],
    actual: Sequence[bool],
) -> dict[str, float | int]:
    if len(predicted) != len(actual) or not actual:
        raise ValueError("classification metrics require paired labels")
    true_positive = sum(p and a for p, a in zip(predicted, actual))
    false_positive = sum(p and not a for p, a in zip(predicted, actual))
    false_negative = sum(not p and a for p, a in zip(predicted, actual))
    true_negative = sum(not p and not a for p, a in zip(predicted, actual))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "n": len(actual),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "accuracy": (true_positive + true_negative) / len(actual),
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "predicted_yes_rate": sum(predicted) / len(predicted),
        "human_yes_rate": sum(actual) / len(actual),
    }


def _report_subset(
    item_ids: Sequence[str],
    *,
    left: Mapping[str, bool],
    right: Mapping[str, bool],
    key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    left_values = [left[item_id] for item_id in item_ids]
    right_values = [right[item_id] for item_id in item_ids]
    agreed_ids = [
        item_id
        for item_id in item_ids
        if left[item_id] == right[item_id]
    ]
    consensus = [left[item_id] for item_id in agreed_ids]
    predicted = [
        str(key[item_id]["machine_answer"]).lower() == "yes"
        for item_id in agreed_ids
    ]
    return {
        "annotator_agreement": {
            "n": len(item_ids),
            "raw_agreement": len(agreed_ids) / len(item_ids),
            "cohen_kappa": _cohen_kappa(left_values, right_values),
            "annotator_a_yes_rate": sum(left_values) / len(left_values),
            "annotator_b_yes_rate": sum(right_values) / len(right_values),
            "disagreement_count": len(item_ids) - len(agreed_ids),
        },
        "judge_vs_agreed_human": {
            "consensus_coverage": len(agreed_ids) / len(item_ids),
            **_classification(predicted, consensus),
        },
    }


def build_report(
    annotator_a: Path,
    annotator_b: Path,
    key_path: Path,
) -> tuple[dict[str, Any], list[str], Mapping[str, Mapping[str, str]]]:
    left, public_rows = _load_annotations(annotator_a)
    right, right_public_rows = _load_annotations(annotator_b)
    key_payload = json.loads(key_path.read_text(encoding="utf-8"))
    key = key_payload["items"]
    if set(left) != set(right) or set(left) != set(key):
        raise ValueError("annotator and coordinator item IDs differ")
    item_ids = sorted(left)
    disagreements = [
        item_id for item_id in item_ids if left[item_id] != right[item_id]
    ]
    strata: dict[str, Any] = {}
    for field in (
        "evaluation_type",
        "arm",
        "model_family",
        "departure_fraction",
    ):
        values = sorted({str(key[item_id][field]) for item_id in item_ids})
        strata[field] = {
            value: _report_subset(
                [
                    item_id
                    for item_id in item_ids
                    if str(key[item_id][field]) == value
                ],
                left=left,
                right=right,
                key=key,
            )
            for value in values
        }
    report = {
        "schema_version": "memora_human_judge_calibration_v1",
        "positive_class": "yes",
        "consensus_policy": (
            "Use only A/B agreements; disagreements require adjudication."
        ),
        "overall": _report_subset(
            item_ids,
            left=left,
            right=right,
            key=key,
        ),
        "sensitivity_judge_vs_individual_annotator": {
            "annotator_a": _classification(
                [
                    str(key[item_id]["machine_answer"]).lower() == "yes"
                    for item_id in item_ids
                ],
                [left[item_id] for item_id in item_ids],
            ),
            "annotator_b": _classification(
                [
                    str(key[item_id]["machine_answer"]).lower() == "yes"
                    for item_id in item_ids
                ],
                [right[item_id] for item_id in item_ids],
            ),
        },
        "stratified": strata,
        "disagreement_item_ids": disagreements,
        "annotator_confidence": {
            "annotator_a": dict(
                sorted(
                    Counter(
                        str(public_rows[item_id]["confidence_1_to_3"])
                        for item_id in item_ids
                    ).items()
                )
            ),
            "annotator_b": dict(
                sorted(
                    Counter(
                        str(
                            right_public_rows[item_id][
                                "confidence_1_to_3"
                            ]
                        )
                        for item_id in item_ids
                    ).items()
                )
            ),
        },
    }
    return report, disagreements, public_rows


def _write_adjudication(
    path: Path,
    item_ids: Sequence[str],
    public_rows: Mapping[str, Mapping[str, str]],
) -> None:
    fields = (
        "item_id",
        "user_question",
        "candidate_response",
        "rubric_question",
        "adjudicated_answer_yes_or_no",
        "confidence_1_to_3",
        "notes",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item_id in item_ids:
            source = public_rows[item_id]
            writer.writerow(
                {
                    "item_id": item_id,
                    "user_question": source["user_question"],
                    "candidate_response": source["candidate_response"],
                    "rubric_question": source["rubric_question"],
                    "adjudicated_answer_yes_or_no": "",
                    "confidence_1_to_3": "",
                    "notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adjudication-output", type=Path)
    args = parser.parse_args()
    report, disagreements, public_rows = build_report(
        args.annotator_a,
        args.annotator_b,
        args.key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if args.adjudication_output:
        args.adjudication_output.parent.mkdir(parents=True, exist_ok=True)
        _write_adjudication(
            args.adjudication_output,
            disagreements,
            public_rows,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
