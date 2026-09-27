#!/usr/bin/env python3
"""Materialize a frozen, outcome-blind Memora Agent RETURN manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace.memora_return import (  # noqa: E402
    build_memora_manifest,
    discover_memora_timelines,
    manifest_summary,
)


DEFAULT_DATA_ROOT = (
    ROOT / "data" / "Memora" / "data"
)


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return result


def _fractions(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in _csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "departure fractions must be numeric"
        ) from error
    if any(not 0.0 < item < 1.0 for item in result):
        raise argparse.ArgumentTypeError(
            "departure fractions must lie strictly in (0, 1)"
        )
    return result


def _git_revision(data_root: Path) -> str:
    for candidate in (data_root, *data_root.parents):
        if not (candidate / ".git").exists():
            continue
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--periods",
        type=_csv,
        default=("weekly", "monthly", "quarterly"),
    )
    parser.add_argument("--personas", type=_csv)
    parser.add_argument(
        "--task-types",
        type=_csv,
        default=("remembering", "recommending"),
    )
    parser.add_argument(
        "--departure-fractions",
        type=_fractions,
        default=(0.25, 0.5, 0.75),
        help=(
            "Frozen priority order. The first cut that realizes V/X/N is "
            "used; model outcomes are never consulted."
        ),
    )
    parser.add_argument("--minimum-absence-sessions", type=int, default=1)
    parser.add_argument("--source-revision")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.resolve()
    timelines = discover_memora_timelines(
        data_root,
        periods=args.periods,
        personas=args.personas,
    )
    source_revision = args.source_revision or _git_revision(data_root)
    manifest = build_memora_manifest(
        timelines,
        source_revision=source_revision,
        departure_fractions=args.departure_fractions,
        minimum_absence_sessions=args.minimum_absence_sessions,
        task_types=args.task_types,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "data_root": str(data_root),
                "source_revision": source_revision,
                **manifest_summary(manifest),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
