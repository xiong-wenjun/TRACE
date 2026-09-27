#!/usr/bin/env python3
"""Report human agreement and automatic-score calibration."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


def _load_annotations(
    path: Path,
    *,
    binary_labels: Sequence[str] = (),
    ordinal_labels: Sequence[str] = (),
) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for value in csv.DictReader(handle):
                item_id = str(value["item_id"])
                if item_id in rows:
                    raise ValueError(
                        f"duplicate annotation item: {item_id}"
                    )
                labels: dict[str, Any] = {}
                for label in binary_labels:
                    raw = str(value.get(label, "")).strip().lower()
                    if raw not in {"true", "false", "yes", "no", "1", "0"}:
                        raise ValueError(
                            f"{item_id}:{label} must be yes/no or true/false"
                        )
                    labels[label] = raw in {"true", "yes", "1"}
                for label in ordinal_labels:
                    raw = str(value.get(label, "")).strip()
                    try:
                        labels[label] = int(raw)
                    except ValueError as error:
                        raise ValueError(
                            f"{item_id}:{label} must be an integer from 1 to 5"
                        ) from error
                rows[item_id] = labels
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            item_id = str(value["item_id"])
            if item_id in rows:
                raise ValueError(f"duplicate annotation item: {item_id}")
            labels = value.get("labels")
            if not isinstance(labels, Mapping):
                raise ValueError(f"missing labels for {item_id}")
            rows[item_id] = labels
    return rows


def _binary_label(
    labels: Mapping[str, Any],
    label: str,
    item_id: str,
) -> bool:
    value = labels.get(label)
    if not isinstance(value, bool):
        raise ValueError(
            f"{item_id}:{label} must be a JSON boolean"
        )
    return value


def _ordinal_label(
    labels: Mapping[str, Any],
    label: str,
    item_id: str,
) -> int:
    value = labels.get(label)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > 5
    ):
        raise ValueError(
            f"{item_id}:{label} must be an integer from 1 to 5"
        )
    return value


def _cohen_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa requires non-empty paired labels")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts).union(right_counts)
    expected = sum(
        left_counts[label] * right_counts[label] for label in labels
    ) / (len(left) ** 2)
    return (
        (observed - expected) / (1.0 - expected)
        if expected < 1.0
        else 1.0
    )


def _weighted_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("weighted kappa requires non-empty paired labels")
    minimum = min((*left, *right))
    maximum = max((*left, *right))
    span = max(1, maximum - minimum)
    observed = statistics.fmean(
        1.0 - ((a - b) / span) ** 2 for a, b in zip(left, right)
    )
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (
            1.0 - ((a - b) / span) ** 2
        )
        * left_counts[a]
        * right_counts[b]
        for a in left_counts
        for b in right_counts
    ) / (len(left) ** 2)
    return (
        (observed - expected) / (1.0 - expected)
        if expected < 1.0
        else 1.0
    )


def _binary_metrics(
    predicted: Sequence[bool],
    actual: Sequence[bool],
) -> dict[str, float | int]:
    true_positive = sum(p and a for p, a in zip(predicted, actual))
    false_positive = sum(p and not a for p, a in zip(predicted, actual))
    false_negative = sum(not p and a for p, a in zip(predicted, actual))
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
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while (
            end < len(ordered)
            and values[ordered[end]] == values[ordered[index]]
        ):
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for position in ordered[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left_ranks)
        * sum((b - right_mean) ** 2 for b in right_ranks)
    )
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    key = json.loads(args.key.read_text(encoding="utf-8"))
    left = _load_annotations(
        args.annotator_a,
        binary_labels=key["binary_labels"],
        ordinal_labels=key["ordinal_labels"],
    )
    right = _load_annotations(
        args.annotator_b,
        binary_labels=key["binary_labels"],
        ordinal_labels=key["ordinal_labels"],
    )
    if set(left) != set(right):
        raise ValueError("annotators must label the same item IDs")
    if set(left) != set(key["items"]):
        raise ValueError("annotation IDs do not match the coordinator key")

    binary: dict[str, Any] = {}
    for label in key["binary_labels"]:
        left_labels = [
            _binary_label(left[item], label, item) for item in left
        ]
        right_labels = [
            _binary_label(right[item], label, item) for item in left
        ]
        left_values = [int(value) for value in left_labels]
        right_values = [int(value) for value in right_labels]
        agreed_ids = [
            item
            for item, left_value, right_value in zip(
                left,
                left_labels,
                right_labels,
            )
            if left_value == right_value
        ]
        consensus = [
            _binary_label(left[item], label, item) for item in agreed_ids
        ]
        predicted = [
            bool(key["items"][item]["machine_binary"][label])
            for item in agreed_ids
        ]
        binary[label] = {
            "cohen_kappa": _cohen_kappa(left_values, right_values),
            "consensus_coverage": len(agreed_ids) / len(left),
            "automatic_vs_human_consensus": _binary_metrics(
                predicted,
                consensus,
            ),
        }

    ordinal: dict[str, Any] = {}
    for label in key["ordinal_labels"]:
        left_values = [
            _ordinal_label(left[item], label, item) for item in left
        ]
        right_values = [
            _ordinal_label(right[item], label, item) for item in left
        ]
        consensus = [
            (a + b) / 2.0 for a, b in zip(left_values, right_values)
        ]
        if all(
            label in key["items"][item].get(
                "judge_ordinal_1_to_5", {}
            )
            for item in left
        ):
            judge = [
                float(
                    key["items"][item]["judge_ordinal_1_to_5"][label]
                )
                for item in left
            ]
            calibration = {
                "spearman": _spearman(judge, consensus),
                "mean_absolute_error": statistics.fmean(
                    abs(a - b) for a, b in zip(judge, consensus)
                ),
            }
        else:
            calibration = None
        ordinal[label] = {
            "quadratic_weighted_cohen_kappa": _weighted_kappa(
                left_values,
                right_values,
            ),
            "judge_calibration": calibration,
        }

    payload = {
        "schema_version": "human_judge_calibration_v1",
        "item_count": len(left),
        "binary_labels": binary,
        "ordinal_labels": ordinal,
    }
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
