"""Source receipts for native memories; these receipts never generate text.

Mem0 and Memobase do not expose a complete formation read set through their
public APIs. Changed records therefore inherit a conservative union of the
namespace's sources. This is an upper bound on dependency, not a causal label.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import errno
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from .base import BackendMemory


def source_ids(metadata: Mapping[str, object]) -> tuple[str, ...]:
    values = metadata.get("source_workstate_ids", ())
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("source_workstate_ids must be a sequence of IDs")
    single = metadata.get("source_workstate_id")
    return tuple(dict.fromkeys(str(v).strip() for v in (*values, single)
                               if v is not None and str(v).strip()))


def checked_entries(entries: Sequence[BackendMemory]) -> tuple[BackendMemory, ...]:
    result = tuple(entries)
    ids = set()
    for entry in result:
        if not entry.external_id or not entry.text.strip():
            raise RuntimeError("native backend returned an empty ID or text")
        if entry.external_id in ids:
            raise RuntimeError("native backend returned duplicate memory IDs")
        if not math.isfinite(entry.score):
            raise RuntimeError("native backend returned a non-finite score")
        ids.add(entry.external_id)
    return result


def write_receipt_json(path: Path, payload: Mapping[str, object]) -> None:
    """Durable single-writer receipt; tolerate known shared-filesystem rename limits.

    A crash during the narrow direct-write fallback may leave invalid JSON;
    readers reject it rather than reconstructing or assuming complete provenance.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    encoded = json.dumps(payload, sort_keys=True)
    def write(destination):
        with destination.open("w") as f:
            os.chmod(destination, 0o600)
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
    try:
        write(temp)
        try:
            os.replace(temp, path)
        except OSError as error:
            if error.errno not in {errno.EPERM, errno.EXDEV}:
                raise
            write(path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


@dataclass
class SourceLedger:
    storage_dir: Path | None = None
    _states: dict[str, dict] = field(default_factory=dict)

    def _path(self, namespace: str) -> Path | None:
        if self.storage_dir is None:
            return None
        return self.storage_dir / (hashlib.sha256(namespace.encode()).hexdigest() + ".json")

    def _state(self, namespace: str) -> dict:
        if namespace not in self._states:
            path = self._path(namespace)
            state = json.loads(path.read_text()) if path and path.exists() else {
                "schema_version": "native_source_ledger_v1", "failed": False, "entries": {}
            }
            if state.get("schema_version") != "native_source_ledger_v1":
                raise RuntimeError("unsupported native source ledger")
            self._states[namespace] = state
        return self._states[namespace]

    def _save(self, namespace: str) -> None:
        path = self._path(namespace)
        if path is not None:
            write_receipt_json(path, self._state(namespace))

    def check(self, namespace: str) -> None:
        if self._state(namespace)["failed"]:
            raise RuntimeError("native namespace has an incomplete operation; reset before reuse")

    def fail(self, namespace: str) -> None:
        self._state(namespace)["failed"] = True
        self._save(namespace)

    def reset(self, namespace: str) -> None:
        self._states.pop(namespace, None)
        path = self._path(namespace)
        if path:
            path.unlink(missing_ok=True)

    def bind(self, namespace: str, entry: BackendMemory, sources: Sequence[str],
             *, complete: bool, metadata: Mapping[str, object] | None = None) -> None:
        self.check(namespace)
        self._state(namespace)["entries"][entry.external_id] = {
            "sources": sorted(set(sources)), "complete": complete,
            "text_sha256": hashlib.sha256(entry.text.encode()).hexdigest(),
            "metadata": dict(metadata or {}),
        }
        self._save(namespace)

    def decorate(self, namespace: str, entry: BackendMemory) -> BackendMemory:
        self.check(namespace)
        receipt = self._state(namespace)["entries"].get(entry.external_id)
        if receipt is None:
            return replace(entry, source_workstate_ids=(), metadata={
                **dict(entry.metadata), "provenance_complete": False,
            })
        if receipt["text_sha256"] != hashlib.sha256(entry.text.encode()).hexdigest():
            raise RuntimeError("native memory changed outside the adapter source ledger")
        return replace(entry, source_workstate_ids=tuple(receipt["sources"]), metadata={
            **dict(receipt["metadata"]), **dict(entry.metadata),
            "source_workstate_ids": list(receipt["sources"]),
            "provenance_complete": bool(receipt["complete"]),
        })

    def observe(self, namespace: str, before: Sequence[BackendMemory],
                after: Sequence[BackendMemory], metadata: Mapping[str, object]) -> tuple[str, ...]:
        """Bind changed output to the full available input scope conservatively."""
        incoming = set(source_ids(metadata))
        state = self._state(namespace)
        # A buffered chat may form a profile only on a later write. Track every
        # successful input, even if this write materializes no native records.
        dependencies = incoming | set(state.get("formation_scope", [])) | {
            s for e in before for s in e.source_workstate_ids}
        complete = bool(incoming) and state.get("scope_complete", True) and all(
            e.metadata.get("provenance_complete", bool(e.source_workstate_ids)) for e in before
        )
        state["formation_scope"] = sorted(dependencies)
        state["scope_complete"] = bool(complete)
        self._save(namespace)
        old = {e.external_id: e for e in before}
        changed = []
        for entry in checked_entries(after):
            previous = old.get(entry.external_id)
            if previous is not None and previous.text == entry.text and (
                previous.metadata.get("native_revision") == entry.metadata.get("native_revision")
            ):
                continue
            self.bind(namespace, entry, dependencies, complete=complete, metadata={
                **dict(metadata), "lineage_policy": "conservative_formation_scope",
            })
            changed.append(entry.external_id)
        return tuple(changed)


def verify_import(source: Sequence[BackendMemory], imported: Sequence[BackendMemory]) -> None:
    from collections import Counter

    def signature(entry: BackendMemory) -> tuple:
        return (entry.text, tuple(sorted(entry.source_workstate_ids)),
                bool(entry.metadata.get("provenance_complete", bool(entry.source_workstate_ids))))
    checked_entries(imported)
    if Counter(map(signature, source)) != Counter(map(signature, imported)):
        raise RuntimeError("native checkpoint import changed text, lineage, or coverage")
