#!/usr/bin/env python3
"""Generate publication-ready STALE Qwen accuracy/cost figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


METHODS = [
    {
        "name": "Static",
        "group": "Basic policy",
        "overall": 33.3333,
        "via": 45.0000,
        "iir": 21.6667,
        "ls": 13.3333,
        "calls": 47.8000,
        "tokens": 218_227.3333,
        "n": 60,
    },
    {
        "name": "Restore",
        "group": "Basic policy",
        "overall": 5.5556,
        "via": 13.3333,
        "iir": 0.0000,
        "ls": 0.0000,
        "calls": 46.9667,
        "tokens": 208_266.5667,
        "n": 60,
    },
    {
        "name": "Reset",
        "group": "Basic policy",
        "overall": 12.2222,
        "via": 11.6667,
        "iir": 1.6667,
        "ls": 1.6667,
        "calls": 46.9500,
        "tokens": 204_858.8000,
        "n": 60,
    },
    {
        "name": "MemStrata",
        "group": "Governance baseline",
        "overall": 31.1111,
        "via": 35.0000,
        "iir": 20.0000,
        "ls": 13.3333,
        "calls": 47.8500,
        "tokens": 216_169.1667,
        "n": 60,
    },
    {
        "name": "MemTX",
        "group": "Governance baseline",
        "overall": 30.0000,
        "via": 33.3333,
        "iir": 20.0000,
        "ls": 13.3333,
        "calls": 47.8000,
        "tokens": 216_005.7167,
        "n": 60,
    },
    {
        "name": "CUPMem",
        "group": "Specialist system",
        "overall": 81.9209,
        "via": 76.2712,
        "iir": 83.0508,
        "ls": 64.4068,
        "calls": 655.3220,
        "tokens": 1_927_417.4237,
        "n": 59,
    },
    {
        "name": "TRACE",
        "group": "TRACE (Ours)",
        "overall": 58.8889,
        "via": 60.0000,
        "iir": 53.3333,
        "ls": 40.0000,
        "calls": 162.6333,
        "tokens": 499_658.6167,
        "n": 60,
    },
]

COLORS = {
    "Basic policy": "#969696",
    "Governance baseline": "#4C78A8",
    "TRACE (Ours)": "#F28E2B",
    "Specialist system": "#2CA02C",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def bubble_size(overall: float) -> float:
    return 180.0 + 13.5 * overall


def token_tick(value: float, _: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    return f"{value / 1_000:g}K"


def pareto_frontier(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    frontier = []
    best_y = -float("inf")
    for row in sorted(rows, key=lambda item: (item["tokens"], -item["ls"])):
        if row["ls"] > best_y:
            frontier.append(row)
            best_y = row["ls"]
    return frontier


def draw_pareto(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([200_000, 500_000, 1_000_000, 2_000_000]))
    ax.xaxis.set_major_formatter(FuncFormatter(token_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.grid(True, which="minor", axis="x", color="#ECECEC", linewidth=0.5, alpha=0.65)
    ax.set_axisbelow(True)

    frontier = pareto_frontier(METHODS)
    ax.plot(
        [row["tokens"] for row in frontier],
        [row["ls"] for row in frontier],
        linestyle=(0, (4, 3)),
        color="#444444",
        linewidth=1.4,
        zorder=1,
        label="Pareto frontier",
    )

    for row in METHODS:
        ax.scatter(
            row["tokens"],
            row["ls"],
            s=bubble_size(row["overall"]),
            color=COLORS[row["group"]],
            edgecolor="white",
            linewidth=1.3,
            alpha=0.92,
            zorder=3,
        )

    offsets = {
        "Static": (68, 30),
        "Restore": (65, 10),
        "Reset": (-38, 22),
        "MemStrata": (75, -18),
        "MemTX": (-45, 30),
        "TRACE": (12, 13),
        "CUPMem": (-72, 14),
    }
    for row in METHODS:
        label = row["name"]
        weight = "bold" if row["name"] in {"TRACE", "CUPMem"} else "normal"
        ax.annotate(
            label,
            (row["tokens"], row["ls"]),
            xytext=offsets[row["name"]],
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9.3,
            fontweight=weight,
            color="#202020",
            arrowprops={"arrowstyle": "-", "color": "#A0A0A0", "lw": 0.65},
            zorder=4,
        )

    ax.set_xlim(175_000, 2_600_000)
    ax.set_ylim(-6, 75)
    ax.set_xlabel("Actor tokens per episode (log scale)")
    ax.set_ylabel("Lifecycle Success (LS, %)")

    group_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS[group],
            markeredgecolor="white",
            markersize=8,
            label=group,
        )
        for group in ["Basic policy", "Governance baseline", "TRACE (Ours)", "Specialist system"]
    ]
    frontier_handle = Line2D(
        [0], [0], color="#444444", linestyle=(0, (4, 3)), linewidth=1.4, label="Pareto frontier"
    )
    first_legend = ax.legend(
        handles=group_handles + [frontier_handle],
        loc="upper left",
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
    )
    ax.add_artist(first_legend)

    size_handles = [
        ax.scatter([], [], s=bubble_size(value), color="#B8B8B8", edgecolor="white", label=f"{value}%")
        for value in [20, 50, 80]
    ]
    ax.legend(
        handles=size_handles,
        title="Overall (bubble size)",
        loc="lower right",
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        labelspacing=1.2,
    )

    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99))
    fig.savefig(output_dir / "qwen_stale60_tokens_ls_pareto.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "qwen_stale60_tokens_ls_pareto.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def draw_radar(output_dir: Path) -> None:
    selected_names = ["Reset", "MemTX", "TRACE", "CUPMem"]
    selected = [row for row in METHODS if row["name"] in selected_names]
    selected.sort(key=lambda row: selected_names.index(row["name"]))

    min_tokens = min(row["tokens"] for row in METHODS)
    min_calls = min(row["calls"] for row in METHODS)
    axes = ["Overall", "VIA", "IIR", "LS", "Token\nEfficiency", "Call\nEfficiency"]
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7.4, 6.1), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(axes, fontsize=10.2)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.2, color="#666666")
    ax.set_rlabel_position(24)
    ax.grid(color="#D0D0D0", linewidth=0.75)
    ax.spines["polar"].set_color("#B8B8B8")

    for row in selected:
        values = [
            row["overall"] / 100,
            row["via"] / 100,
            row["iir"] / 100,
            row["ls"] / 100,
            min_tokens / row["tokens"],
            min_calls / row["calls"],
        ]
        closed_values = values + values[:1]
        label = row["name"]
        color = COLORS[row["group"]]
        linewidth = 2.7 if row["name"] == "TRACE" else 2.0
        zorder = 5 if row["name"] == "TRACE" else 3
        ax.plot(
            closed_angles,
            closed_values,
            color=color,
            linewidth=linewidth,
            marker="o",
            markersize=4.2,
            label=label,
            zorder=zorder,
        )
        ax.fill(closed_angles, closed_values, color=color, alpha=0.075, zorder=zorder - 1)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        fontsize=8.5,
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
    )

    fig.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.18)
    fig.savefig(output_dir / "qwen_stale60_effectiveness_efficiency_radar.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "qwen_stale60_effectiveness_efficiency_radar.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def write_data(output_dir: Path) -> None:
    min_tokens = min(row["tokens"] for row in METHODS)
    min_calls = min(row["calls"] for row in METHODS)
    fieldnames = [
        "method",
        "group",
        "episodes",
        "overall_pct",
        "via_pct",
        "iir_pct",
        "ls_pct",
        "calls_per_episode",
        "tokens_per_episode",
        "token_efficiency",
        "call_efficiency",
    ]
    data_path = output_dir / "qwen_stale60_cost_figure_data.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in METHODS:
            writer.writerow(
                {
                    "method": row["name"],
                    "group": row["group"],
                    "episodes": row["n"],
                    "overall_pct": row["overall"],
                    "via_pct": row["via"],
                    "iir_pct": row["iir"],
                    "ls_pct": row["ls"],
                    "calls_per_episode": row["calls"],
                    "tokens_per_episode": row["tokens"],
                    "token_efficiency": min_tokens / row["tokens"],
                    "call_efficiency": min_calls / row["calls"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    write_data(args.output_dir)
    draw_pareto(args.output_dir)
    draw_radar(args.output_dir)


if __name__ == "__main__":
    main()
