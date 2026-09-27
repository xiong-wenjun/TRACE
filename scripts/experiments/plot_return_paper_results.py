#!/usr/bin/env python3
"""Render paper-ready RETURN performance, security, and cost figures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_panel(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("panel must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path).resolve()


def save_figure(figure: Any, output: Path) -> list[str]:
    paths = []
    for suffix in (".pdf", ".png"):
        path = output.with_suffix(suffix)
        figure.savefig(
            path,
            bbox_inches="tight",
            dpi=300 if suffix == ".png" else None,
        )
        paths.append(str(path))
    plt.close(figure)
    return paths


def performance_figure(
    panels: list[tuple[str, dict[str, Any]]],
    output: Path,
) -> list[str]:
    labels = [label for label, _ in panels]
    effects = []
    lower = []
    upper = []
    for _, report in panels:
        stats = report["paired_statistics"]
        effect = float(stats["trace_minus_restore_old"])
        interval = stats["bootstrap_95_ci"]["trace_minus_restore_old"]
        effects.append(100 * effect)
        lower.append(100 * (effect - float(interval[0])))
        upper.append(100 * (float(interval[1]) - effect))
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    positions = list(range(len(labels)))
    axis.errorbar(
        effects,
        positions,
        xerr=[lower, upper],
        fmt="o",
        color="#1f4e79",
        ecolor="#6f8fae",
        capsize=4,
        linewidth=1.5,
        markersize=6,
    )
    axis.axvline(0, color="#333333", linewidth=1, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("TRACE − Restore (percentage points, paired 95% CI)")
    axis.set_title("End-task effect of trusted RETURN reconciliation")
    axis.grid(axis="x", alpha=0.2)
    for position, effect in zip(positions, effects):
        axis.annotate(
            f"{effect:+.2f}",
            (effect, position),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    return save_figure(figure, output)


def security_figure(report: dict[str, Any], output: Path) -> list[str]:
    families = list(report["by_family"])
    upper = [
        100
        * float(report["by_family"][family]["one_sided_95_wilson_upper"])
        for family in families
    ]
    labels = [family.replace("_", " ") for family in families]
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    positions = list(range(len(labels)))
    axis.bar(
        positions,
        upper,
        color="#8fb9a8",
        edgecolor="#315c4c",
        linewidth=0.8,
    )
    axis.scatter(
        positions,
        [0] * len(positions),
        color="#9b1c31",
        marker="x",
        s=35,
        label="observed violation rate (0/100)",
        zorder=3,
    )
    axis.set_xticks(positions, labels, rotation=32, ha="right")
    axis.set_ylabel("Violation rate (%)")
    axis.set_title(
        "TRACE security mutation sweep: observed rate and one-sided 95% upper bound"
    )
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="upper right")
    return save_figure(figure, output)


def cost_figure(report: dict[str, Any], output: Path) -> list[str]:
    rows = report["results"]
    candidates = [int(row["candidate_items"]) for row in rows]
    p95 = [float(row["latency_ms"]["p95"]) for row in rows]
    storage_kib = [
        float(row["serialized_compilation_bytes"]) / 1024 for row in rows
    ]
    figure, latency_axis = plt.subplots(figsize=(6.8, 3.9))
    storage_axis = latency_axis.twinx()
    latency_axis.plot(
        candidates,
        p95,
        marker="o",
        color="#1f4e79",
        linewidth=2,
        label="compiler p95 latency",
    )
    storage_axis.plot(
        candidates,
        storage_kib,
        marker="s",
        color="#b65d2e",
        linewidth=2,
        label="serialized compilation",
    )
    latency_axis.set_xscale("log", base=2)
    latency_axis.set_xlabel("Candidate memory items (log₂ scale)")
    latency_axis.set_ylabel("p95 compilation latency (ms)", color="#1f4e79")
    storage_axis.set_ylabel("Serialized compilation (KiB)", color="#b65d2e")
    latency_axis.set_title("Model-independent TRACE RETURN cost")
    latency_axis.grid(alpha=0.2)
    lines = latency_axis.lines + storage_axis.lines
    latency_axis.legend(
        lines,
        [line.get_label() for line in lines],
        frameon=False,
        loc="upper left",
    )
    return save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        action="append",
        required=True,
        help="Repeat LABEL=statistics.json in desired plot order.",
    )
    parser.add_argument("--security", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    panel_paths = [parse_panel(value) for value in args.panel]
    panels = [(label, load(path)) for label, path in panel_paths]
    security_path = args.security.resolve()
    cost_path = args.cost.resolve()
    generated = {
        "performance": performance_figure(
            panels, output / "return_performance_effects"
        ),
        "security": security_figure(
            load(security_path), output / "return_security_sweep"
        ),
        "cost": cost_figure(
            load(cost_path), output / "return_compiler_cost"
        ),
    }
    inputs = [path for _, path in panel_paths] + [
        security_path,
        cost_path,
    ]
    manifest = {
        "schema_version": "return_paper_figures_v1",
        "generated_at": utc_now(),
        "generated_files": generated,
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    descriptor = os.open(
        output / "manifest.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
