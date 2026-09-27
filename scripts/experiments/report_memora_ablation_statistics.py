#!/usr/bin/env python3
"""Report paired clustered statistics for matched Memora ablation arms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence

from report_memora_return_statistics import (
    METRICS,
    _comparison,
    _complete,
    _load_rows,
    _mean,
    _strata,
)

from trace.memora_statistics import holm_adjust
from trace.router_return_protocol import RouterReturnArm
from trace.trace_method import TRACE_ARM


SCHEMA = "memora_return_ablation_statistics_v1"
DEFAULT_ARMS = (
    RouterReturnArm.STATIC_COMPACT.value,
    RouterReturnArm.VALIDITY_FILTER_ONLY.value,
    TRACE_ARM,
)


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(result) < 2:
        raise argparse.ArgumentTypeError("arms must contain at least two values")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arms", type=_csv, default=DEFAULT_ARMS)
    parser.add_argument("--treatment", default=TRACE_ARM)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=731)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = tuple(args.arms)
    if len(arms) != len(set(arms)):
        raise ValueError("arms must be unique")
    if args.treatment not in arms:
        raise ValueError("treatment must be included in arms")
    rows, paths = _load_rows(args.panel.resolve())
    complete = _complete(rows, arms)
    if not complete:
        raise RuntimeError("no complete matched Memora ablation episodes")
    means = {
        arm: {metric: _mean(complete, arm, metric) for metric in METRICS}
        for arm in arms
    }
    comparisons: dict[str, object] = {}
    fama_p_values: dict[str, float] = {}
    comparison_index = 0
    for comparator in arms:
        if comparator == args.treatment:
            continue
        for metric in METRICS:
            name = f"{args.treatment}_vs_{comparator}_{metric}"
            comparison = _comparison(
                complete,
                treatment=args.treatment,
                comparator=comparator,
                metric=metric,
                replicates=args.bootstrap_replicates,
                seed=args.seed + comparison_index,
            )
            comparisons[name] = comparison
            comparison_index += 1
            if metric == "fama" and comparison.get("calculable"):
                fama_p_values[name] = float(
                    comparison["significance"]["p_value"]
                )
    for name, adjusted in holm_adjust(fama_p_values).items():
        comparisons[name]["holm_adjusted_p_value"] = adjusted
    report = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel": str(args.panel.resolve()),
        "complete_paired_questions": len(complete),
        "persona_period_clusters": len(
            {str(row["cluster_id"]) for row in complete}
        ),
        "arms": list(arms),
        "treatment": args.treatment,
        "metrics": list(METRICS),
        "means": means,
        "paired_comparisons": comparisons,
        "stratified": _strata(complete, arms=arms),
        "source_files": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
    }
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
