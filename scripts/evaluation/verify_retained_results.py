#!/usr/bin/env python3
"""Verify retained experiment directories against the frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "results" / "formal" / "RETENTION_MANIFEST.json"


def _tree_inventory(path: Path) -> dict[str, int | str]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    size_bytes = 0
    for item in files:
        content = item.read_bytes()
        relative = item.relative_to(path).as_posix()
        digest.update(
            f"{hashlib.sha256(content).hexdigest()}  {relative}\n".encode(
                "utf-8"
            )
        )
        size_bytes += len(content)
    return {
        "file_count": len(files),
        "size_bytes": size_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def verify_manifest(
    manifest_path: Path,
    *,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "return_experiment_retention_manifest_v1"
    ):
        raise ValueError("unsupported retention manifest schema")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("retention manifest must contain assets")
    rows: list[dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, Mapping):
            raise ValueError("retention asset must be an object")
        asset_id = str(asset["asset_id"])
        path = root / str(asset["path"])
        if not path.is_dir():
            raise FileNotFoundError(f"{asset_id}: missing directory {path}")
        observed = _tree_inventory(path)
        expected = {
            field: asset[field]
            for field in ("file_count", "size_bytes", "tree_sha256")
        }
        if observed != expected:
            raise ValueError(
                f"{asset_id}: retained tree mismatch; "
                f"expected={expected}, observed={observed}"
            )
        rows.append(
            {
                "asset_id": asset_id,
                "path": str(asset["path"]),
                **observed,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    rows = verify_manifest(args.manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "verified_assets": len(rows),
                "assets": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
