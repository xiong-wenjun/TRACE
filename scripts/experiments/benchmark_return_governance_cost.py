#!/usr/bin/env python3
"""Benchmark model-independent TRACE RETURN compilation cost."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean, median
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import run_return_security_panel as security  # noqa: E402


SCHEMA = "trace_return_compiler_cost_benchmark_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def candidates(
    governor: security.ReturnMemoryGovernor,
    candidate_count: int,
) -> tuple[
    security.ReturnReadmissionContext,
    tuple[security.ReturnMemoryItem, ...],
]:
    readmission, structure, fact = security.fixture(governor)
    distractions = tuple(
        security.signed_item(
            governor,
            item_id=f"irrelevant-{index:05d}",
            kind=security.ReturnMemoryKind.VERIFIED_FACT,
            text=(
                f"Outcome-blind unrelated observation {index}: "
                + "x" * 96
            ),
        )
        for index in range(max(0, candidate_count - 3))
    )
    return (
        replace(
            readmission,
            max_items=max(8, candidate_count),
            max_chars=max(4096, candidate_count * 256),
        ),
        (*structure, fact, *distractions),
    )


def benchmark_size(
    *,
    candidate_count: int,
    repeats: int,
) -> dict[str, Any]:
    governor = security.ReturnMemoryGovernor(
        f"trace-cost-{candidate_count}",
        hashlib.sha256(f"cost-key-{candidate_count}".encode()).digest(),
    )
    readmission, items = candidates(governor, candidate_count)
    warmup_repeats = min(20, repeats)
    for _ in range(warmup_repeats):
        warmup = security.compile_view(governor, items, readmission)
        if warmup.status is not security.ReturnSelectionStatus.ISSUED:
            raise RuntimeError("cost benchmark warm-up did not issue a view")
    latencies_ms: list[float] = []
    last = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        compilation = security.compile_view(
            governor, items, readmission
        )
        latencies_ms.append(
            (time.perf_counter_ns() - started) / 1_000_000
        )
        last = compilation
    assert last is not None
    if last.status is not security.ReturnSelectionStatus.ISSUED:
        raise RuntimeError("cost benchmark failed to issue a valid view")
    record = last.record()
    return {
        "candidate_items": candidate_count,
        "warmup_repeats": warmup_repeats,
        "repeats": repeats,
        "latency_ms": {
            "mean": mean(latencies_ms),
            "median": median(latencies_ms),
            "p95": percentile(latencies_ms, 0.95),
            "max": max(latencies_ms),
        },
        "selected_items": len(last.selection.selected_item_ids),
        "disclosure_chars": last.selection.disclosure_chars,
        "obligation_coverage": last.selection.obligation_coverage,
        "critical_dependency_recall": (
            last.selection.critical_dependency_recall
        ),
        "serialized_compilation_bytes": len(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--candidate-counts", default="8,32,128,512"
    )
    args = parser.parse_args()
    counts = tuple(
        int(value.strip())
        for value in args.candidate_counts.split(",")
        if value.strip()
    )
    if args.repeats < 1 or not counts or any(value < 3 for value in counts):
        raise ValueError("invalid repeat or candidate-count configuration")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": SCHEMA,
        "generated_at": utc_now(),
        "model_calls": 0,
        "candidate_counts": list(counts),
        "repeats_per_size": args.repeats,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "results": [
            benchmark_size(candidate_count=count, repeats=args.repeats)
            for count in counts
        ],
        "source_sha256": {
            "benchmark_return_governance_cost.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "return_governance.py": hashlib.sha256(
                (ROOT / "src/trace/return_governance.py").read_bytes()
            ).hexdigest(),
        },
    }
    descriptor = os.open(
        output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
