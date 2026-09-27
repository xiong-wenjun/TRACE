#!/usr/bin/env python3
"""Summarize and plot matched TRACE/Compact-Only Memora Pareto sweeps."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


METRICS = (
    "fama",
    "memory_presence_accuracy",
    "forgetting_absence_accuracy",
    "overall_accuracy",
)
METHODS = ("compact_only", "trace")


def _settings(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = spec.get("settings")
    if not isinstance(values, list) or not values:
        raise ValueError("Pareto spec must contain settings")
    result = [dict(value) for value in values if isinstance(value, Mapping)]
    names = [str(value.get("name") or "") for value in result]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("Pareto setting names must be unique and non-empty")
    return result


def _dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    keys = ("memory_presence_accuracy", "forgetting_absence_accuracy")
    return all(left[key] >= right[key] for key in keys) and any(
        left[key] > right[key] for key in keys
    )


def _nondominated(points: Sequence[Mapping[str, Any]], method: str) -> set[str]:
    result: set[str] = set()
    for point in points:
        values = point["means"][method]
        if not any(
            other is not point
            and _dominates(other["means"][method], values)
            for other in points
        ):
            result.add(str(point["name"]))
    return result


def _selected_settings(
    points: Sequence[Mapping[str, Any]],
    *,
    default_setting: str,
    limit: int,
) -> list[str]:
    frontier_names = _nondominated(points, "trace")
    frontier = [point for point in points if point["name"] in frontier_names]
    selected: list[str] = []

    def add(name: str) -> None:
        if name and name not in selected and len(selected) < limit:
            selected.append(name)

    add(default_setting)
    if frontier:
        add(
            str(
                max(
                    frontier,
                    key=lambda point: point["means"]["trace"][
                        "memory_presence_accuracy"
                    ],
                )["name"]
            )
        )
        add(
            str(
                max(
                    frontier,
                    key=lambda point: point["means"]["trace"][
                        "forgetting_absence_accuracy"
                    ],
                )["name"]
            )
        )
        presence = [
            point["means"]["trace"]["memory_presence_accuracy"]
            for point in frontier
        ]
        forgetting = [
            point["means"]["trace"]["forgetting_absence_accuracy"]
            for point in frontier
        ]
        presence_range = max(presence) - min(presence)
        forgetting_range = max(forgetting) - min(forgetting)

        def knee(point: Mapping[str, Any]) -> float:
            values = point["means"]["trace"]
            normalized_presence = (
                (values["memory_presence_accuracy"] - min(presence))
                / presence_range
                if presence_range
                else 1.0
            )
            normalized_forgetting = (
                (values["forgetting_absence_accuracy"] - min(forgetting))
                / forgetting_range
                if forgetting_range
                else 1.0
            )
            return normalized_presence + normalized_forgetting

        add(str(max(frontier, key=knee)["name"]))
    return selected


def _svg(points: Sequence[Mapping[str, Any]], output: Path, split: str) -> None:
    width, height = 900, 590
    left, right, top, bottom = 92, 40, 62, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [
        point["means"][method][metric]
        for point in points
        for method in METHODS
        for metric in (
            "memory_presence_accuracy",
            "forgetting_absence_accuracy",
        )
    ]
    minimum = max(0.0, min(all_values) - 0.05)
    maximum = min(1.0, max(all_values) + 0.05)
    if maximum - minimum < 0.1:
        maximum = min(1.0, minimum + 0.1)

    def x(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * plot_width

    def y(value: float) -> float:
        return top + (maximum - value) / (maximum - minimum) * plot_height

    colors = {"trace": "#2457d6", "compact_only": "#777777"}
    labels = {"trace": "TRACE", "compact_only": "Compact-Only"}
    rows = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}'
        '.axis{stroke:#263247;stroke-width:1.4}.grid{stroke:#dce1e9;stroke-width:1}'
        '.label{font-size:13px}.title{font-size:20px;font-weight:700}'
        '.legend{font-size:14px;font-weight:600}</style>',
        f'<text class="title" x="{width / 2}" y="31" text-anchor="middle">'
        f'Memora {split.title()} Presence–Forgetting Pareto Sweep</text>',
    ]
    for index in range(6):
        value = minimum + index * (maximum - minimum) / 5
        px, py = x(value), y(value)
        rows.append(
            f'<line class="grid" x1="{px:.1f}" y1="{top}" '
            f'x2="{px:.1f}" y2="{height - bottom}"/>'
        )
        rows.append(
            f'<line class="grid" x1="{left}" y1="{py:.1f}" '
            f'x2="{width - right}" y2="{py:.1f}"/>'
        )
        rows.append(
            f'<text class="label" x="{px:.1f}" y="{height - bottom + 23}" '
            f'text-anchor="middle">{value:.2f}</text>'
        )
        rows.append(
            f'<text class="label" x="{left - 13}" y="{py + 4:.1f}" '
            f'text-anchor="end">{value:.2f}</text>'
        )
    rows.extend(
        [
            f'<line class="axis" x1="{left}" y1="{height-bottom}" '
            f'x2="{width-right}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{height-bottom}"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 26}" '
            'text-anchor="middle" font-size="16">Forgetting ↑</text>',
            f'<text transform="translate(25 {top + plot_height / 2}) rotate(-90)" '
            'text-anchor="middle" font-size="16">Memory Presence ↑</text>',
        ]
    )
    budget_points = [
        point for point in points if "budget" in point.get("families", [])
    ]
    for method in METHODS:
        ordered = sorted(
            budget_points,
            key=lambda point: (
                int(point["max_selected_facts"]),
                int(point["base_selected_facts"]),
            ),
        )
        coordinates = " ".join(
            f'{x(point["means"][method]["forgetting_absence_accuracy"]):.1f},'
            f'{y(point["means"][method]["memory_presence_accuracy"]):.1f}'
            for point in ordered
        )
        rows.append(
            f'<polyline points="{coordinates}" fill="none" '
            f'stroke="{colors[method]}" stroke-width="2.3"/>'
        )
        for point in ordered:
            values = point["means"][method]
            px = x(values["forgetting_absence_accuracy"])
            py = y(values["memory_presence_accuracy"])
            rows.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.3" '
                f'fill="{colors[method]}"/>'
            )
    for index, method in enumerate(METHODS):
        legend_x = width - 250
        legend_y = 30 + index * 24
        rows.append(
            f'<circle cx="{legend_x}" cy="{legend_y}" r="5" '
            f'fill="{colors[method]}"/>'
        )
        rows.append(
            f'<text class="legend" x="{legend_x + 13}" y="{legend_y + 5}">'
            f'{labels[method]}</text>'
        )
    rows.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path)
    parser.add_argument("--selection-limit", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    settings = _settings(spec)
    requested = settings
    selected_input = args.results_root / "dev" / "selected_settings.json"
    if args.split == "test" and selected_input.is_file():
        selected_names = set(
            json.loads(selected_input.read_text(encoding="utf-8"))[
                "selected_settings"
            ]
        )
        requested = [
            setting for setting in settings if setting["name"] in selected_names
        ]
    points: list[dict[str, Any]] = []
    for setting in requested:
        statistics_path = (
            args.results_root
            / args.split
            / str(setting["name"])
            / "sol_scored"
            / "statistics.json"
        )
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        means = statistics.get("means")
        if not isinstance(means, Mapping) or not all(
            method in means for method in METHODS
        ):
            raise ValueError(f"missing matched means: {statistics_path}")
        point = dict(setting)
        point["means"] = {
            method: {
                metric: float(means[method][metric])
                for metric in METRICS
            }
            for method in METHODS
        }
        point["trace_minus_compact"] = {
            metric: point["means"]["trace"][metric]
            - point["means"]["compact_only"][metric]
            for metric in METRICS
        }
        point["paired_comparisons"] = statistics.get(
            "paired_comparisons", {}
        )
        point["questions"] = int(statistics["complete_paired_questions"])
        point["clusters"] = int(statistics["persona_period_clusters"])
        points.append(point)
    for method in METHODS:
        names = _nondominated(points, method)
        for point in points:
            point.setdefault("pareto_nondominated", {})[method] = (
                point["name"] in names
            )
    report = {
        "schema_version": "memora_presence_forgetting_pareto_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "spec": str(args.spec.resolve()),
        "results_root": str(args.results_root.resolve()),
        "points": points,
        "axes": {
            "x": "forgetting_absence_accuracy",
            "y": "memory_presence_accuracy",
            "direction": "higher_is_better_for_both",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "setting",
                "family",
                "base_selected_facts",
                "max_selected_facts",
                "minimum_confidence",
                "method",
                *METRICS,
                "pareto_nondominated",
            ]
        )
        for point in points:
            for method in METHODS:
                writer.writerow(
                    [
                        args.split,
                        point["name"],
                        "+".join(point.get("families", [])),
                        point["base_selected_facts"],
                        point["max_selected_facts"],
                        point["minimum_confidence"],
                        method,
                        *(point["means"][method][metric] for metric in METRICS),
                        point["pareto_nondominated"][method],
                    ]
                )
    _svg(points, args.svg_output, args.split)
    if args.selected_output is not None:
        selected = _selected_settings(
            points,
            default_setting=str(spec["default_setting"]),
            limit=args.selection_limit,
        )
        value = {
            "schema_version": "memora_pareto_selected_settings_v1",
            "selection_split": args.split,
            "selection_rule": (
                "default_plus_trace_presence_extreme_forgetting_extreme_knee"
            ),
            "selection_limit": args.selection_limit,
            "selected_settings": selected,
        }
        args.selected_output.parent.mkdir(parents=True, exist_ok=True)
        args.selected_output.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
