#!/usr/bin/env python3
"""Materialize a metadata-only MAS Return sidecar for official STALE Type-II."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace.stale_type2_return import (  # noqa: E402
    STALE_TYPE2_COUNT,
    build_mas_return_episode,
    build_sidecar_manifest,
    leakage_audit,
    load_official_stale_type2,
)


DEFAULT_DATASET = (
    ROOT
    / "data"
    / "stale"
    / "T1_T2_400_FULL.json"
)
DEFAULT_OUTPUT = ROOT / "configs" / "shared/stale_type2_official_mas_return.json"


def _write_json_once(path: Path, value: object) -> None:
    """Create an immutable receipt without overwriting an existing manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the official STALE artifact and create an external MAS "
            "lifecycle sidecar. The source records are never rewritten."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--agent-count", type=int, default=5)
    args = parser.parse_args()

    records = load_official_stale_type2(args.dataset)
    audits = tuple(
        leakage_audit(
            record,
            build_mas_return_episode(record, agent_count=args.agent_count),
        )
        for record in records
    )
    failed = [audit["uid"] for audit in audits if not audit["passed"]]
    if failed:
        raise RuntimeError(f"actor annotation leakage detected: {failed[:5]}")

    manifest = build_sidecar_manifest(
        records,
        source_path=args.dataset,
        agent_count=args.agent_count,
    )
    _write_json_once(args.output.resolve(), manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "type2_records": len(records),
                "expected_type2_records": STALE_TYPE2_COUNT,
                "agent_count": args.agent_count,
                "official_records_modified": False,
                "ground_truth_fields_actor_visible": False,
                "leakage_audits_passed": len(audits),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
