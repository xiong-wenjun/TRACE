#!/usr/bin/env python3
"""Create outcome-blind cluster-level Memora dev/test manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be one JSON object")
    expected = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if expected and expected != _canonical_sha256(unsigned):
        raise ValueError("parent manifest digest mismatch")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("manifest must contain episodes")
    return value


def _cluster_split(
    episodes: Sequence[Mapping[str, object]],
    *,
    seed: int,
) -> tuple[set[str], set[str]]:
    clusters_by_persona: dict[str, dict[str, str]] = {}
    for episode in episodes:
        persona = str(episode["persona"])
        period = str(episode["period"])
        cluster_id = str(episode["cluster_id"])
        current = clusters_by_persona.setdefault(persona, {})
        if period in current and current[period] != cluster_id:
            raise ValueError("persona-period maps to multiple clusters")
        current[period] = cluster_id
    periods = sorted(
        {period for values in clusters_by_persona.values() for period in values}
    )
    if len(periods) < 2:
        raise ValueError("cluster split requires at least two periods")
    personas = sorted(clusters_by_persona)
    generator = random.Random(seed)
    generator.shuffle(personas)
    generator.shuffle(periods)
    dev: set[str] = set()
    for index, persona in enumerate(personas):
        available = clusters_by_persona[persona]
        preferred = periods[index % len(periods)]
        if preferred not in available:
            preferred = sorted(
                available,
                key=lambda period: hashlib.sha256(
                    f"{seed}:{persona}:{period}".encode()
                ).hexdigest(),
            )[0]
        dev.add(available[preferred])
    all_clusters = {
        str(episode["cluster_id"])
        for episode in episodes
    }
    test = all_clusters - dev
    if not test or dev & test:
        raise ValueError("invalid cluster partition")
    return dev, test


def _subset_manifest(
    parent: Mapping[str, Any],
    *,
    cluster_ids: set[str],
    split_name: str,
    seed: int,
    parent_sha256: str,
) -> dict[str, Any]:
    episodes = [
        episode
        for episode in parent["episodes"]
        if str(episode["cluster_id"]) in cluster_ids
    ]
    result = dict(parent)
    result.pop("manifest_sha256", None)
    result["episodes"] = episodes
    result["eligible_cluster_count"] = len(cluster_ids)
    result["eligible_episode_count"] = len(episodes)
    result["timeline_count"] = len(cluster_ids)
    result["parent_manifest_sha256"] = parent_sha256
    result["split"] = {
        "schema_version": "memora_pareto_cluster_split_v1",
        "name": split_name,
        "seed": seed,
        "unit": "persona_period_cluster",
        "cluster_ids": sorted(cluster_ids),
        "outcome_blind": True,
    }
    result["manifest_sha256"] = _canonical_sha256(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=731)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parent = _validated_manifest(args.manifest)
    parent_sha256 = str(parent.get("manifest_sha256") or "")
    if not parent_sha256:
        unsigned = dict(parent)
        unsigned.pop("manifest_sha256", None)
        parent_sha256 = _canonical_sha256(unsigned)
    episodes = [
        episode
        for episode in parent["episodes"]
        if isinstance(episode, Mapping)
    ]
    dev_clusters, test_clusters = _cluster_split(episodes, seed=args.seed)
    outputs = (
        (
            args.dev_output,
            _subset_manifest(
                parent,
                cluster_ids=dev_clusters,
                split_name="dev",
                seed=args.seed,
                parent_sha256=parent_sha256,
            ),
        ),
        (
            args.test_output,
            _subset_manifest(
                parent,
                cluster_ids=test_clusters,
                split_name="test",
                seed=args.seed,
                parent_sha256=parent_sha256,
            ),
        ),
    )
    for path, value in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "schema_version": "memora_pareto_split_receipt_v1",
                "seed": args.seed,
                "dev_clusters": len(dev_clusters),
                "dev_episodes": len(outputs[0][1]["episodes"]),
                "test_clusters": len(test_clusters),
                "test_episodes": len(outputs[1][1]["episodes"]),
                "dev_output": str(args.dev_output),
                "test_output": str(args.test_output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
