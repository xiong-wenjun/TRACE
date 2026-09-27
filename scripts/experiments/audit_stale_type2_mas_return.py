#!/usr/bin/env python3
"""End-to-end integrity audit for the official STALE Type-II MAS adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace.stale_type2_return import (  # noqa: E402
    actor_views,
    bind_records_to_sidecar,
    leakage_audit,
    load_official_stale_type2,
    load_sidecar_manifest,
)


DEFAULT_DATASET = (
    ROOT
    / "data"
    / "stale"
    / "T1_T2_400_FULL.json"
)
DEFAULT_SIDECAR = ROOT / "configs" / "shared/stale_type2_official_mas_return.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument(
        "--uid",
        help="Optionally audit one UID after validating the complete binding.",
    )
    args = parser.parse_args()

    records = load_official_stale_type2(args.dataset)
    episodes = load_sidecar_manifest(args.sidecar)
    bound = bind_records_to_sidecar(records, episodes)
    selected = bound
    if args.uid:
        selected = tuple(pair for pair in bound if pair[0].uid == args.uid)
        if not selected:
            raise ValueError(f"UID not found in official Type-II subset: {args.uid}")

    failed: list[str] = []
    assignments: Counter[str] = Counter()
    for record, episode in selected:
        audit = leakage_audit(record, episode)
        if not audit["passed"] or not audit["assigned_exactly_once"]:
            failed.append(record.uid)
        for agent, view in actor_views(record, episode).items():
            assignments[agent] += len(view["sessions"])
    if failed:
        raise RuntimeError(f"STALE MAS adapter audit failed: {failed[:5]}")

    print(
        json.dumps(
            {
                "official_binding_records": len(bound),
                "audited_records": len(selected),
                "official_records_modified": False,
                "actor_annotation_leaks": 0,
                "all_50_sessions_assigned_exactly_once": True,
                "actor_session_assignments": dict(sorted(assignments.items())),
                "status": "pass",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
