"""A-MEM formation and retrieval under TRACE's native backend contract."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .base import BackendMemory
from .lineage import SourceLedger, checked_entries, verify_import, write_receipt_json
from ..router_return_protocol import canonical_sha256
from ..vendor.amem_official.memory_system import MemoryNote


AMEM_UPSTREAM_REVISION = "ceffb860f0712bbae97b184d440df62bc910ca8d"


def _render(note: Mapping[str, Any]) -> str:
    # Use the native note fields, including evolved context. The lineage
    # sidecar is never inserted into an actor's prompt.
    return (f"{note['content']}\nContext: {note.get('context', '')}\n"
            f"Keywords: {json.dumps(note.get('keywords', []), ensure_ascii=False)}\n"
            f"Tags: {json.dumps(note.get('tags', []), ensure_ascii=False)}")


@dataclass
class AMemDriver:
    system_factory: Callable[[str], Any] = field(repr=False)
    ledger: SourceLedger = field(default_factory=SourceLedger)
    state_dir: Path | None = None
    backend_id: str = "amem_ceffb860_adapter_v1"
    upstream_revision: str = AMEM_UPSTREAM_REVISION
    _systems: dict[str, Any] = field(default_factory=dict, repr=False)
    _audit: list[dict[str, object]] = field(default_factory=list)
    runtime_receipt: Callable[[], dict] = field(default=lambda: {}, repr=False)

    @classmethod
    def build(cls, *, storage_dir: Path, **kwargs) -> "AMemDriver":
        from .amem_runtime import build_system_factory

        factory, receipt = build_system_factory(storage_dir=storage_dir, **kwargs)
        return cls(system_factory=factory, ledger=SourceLedger(storage_dir / "source_receipts"),
                   state_dir=storage_dir / "notes", runtime_receipt=receipt)

    def _path(self, namespace: str) -> Path | None:
        if self.state_dir is None:
            return None
        return self.state_dir / (hashlib.sha256(namespace.encode()).hexdigest() + ".json")

    def _system(self, namespace: str):
        self.ledger.check(namespace)
        if namespace not in self._systems:
            system = self.system_factory(namespace)
            path = self._path(namespace)
            if path and path.exists():
                saved = json.loads(path.read_text())
                if saved.get("upstream_revision") != self.upstream_revision:
                    raise RuntimeError("A-MEM persisted state revision mismatch")
                system.memories = {row["id"]: MemoryNote(**row) for row in saved["notes"]}
                system.evo_cnt = saved["evo_cnt"]
            self._systems[namespace] = system
        return self._systems[namespace]

    def _persist(self, namespace: str) -> None:
        path = self._path(namespace)
        if path is None:
            return
        system = self._systems[namespace]
        write_receipt_json(path, {"upstream_revision": self.upstream_revision,
                                  "evo_cnt": system.evo_cnt,
                                  "notes": [vars(n) for n in system.memories.values()]})

    def reset_namespace(self, namespace: str) -> None:
        self.ledger.reset(namespace)
        self._systems.pop(namespace, None)
        path = self._path(namespace)
        if path:
            path.unlink(missing_ok=True)
        system = self._system(namespace)
        system.memories.clear()
        system.evo_cnt = 0
        system.retriever.reset_namespace()
        self._persist(namespace)
        self._audit.append({"operation": "reset", "namespace": namespace})

    def _raw_list(self, namespace: str) -> tuple[BackendMemory, ...]:
        system = self._system(namespace)
        index = system.retriever.snapshot()
        if set(index) != set(system.memories):
            raise RuntimeError("A-MEM note/index snapshot coverage mismatch")
        return checked_entries(tuple(BackendMemory(
            external_id=note_id, text=_render(vars(note)), metadata={
                "amem_note": deepcopy(vars(note)), "amem_index": deepcopy(index[note_id]),
                "native_revision": canonical_sha256(vars(note)),
                "amem_evo_cnt": system.evo_cnt,
            }) for note_id, note in system.memories.items()))

    def list(self, namespace: str) -> tuple[BackendMemory, ...]:
        return tuple(self.ledger.decorate(namespace, e) for e in self._raw_list(namespace))

    def write(self, namespace: str, text: str,
              metadata: Mapping[str, object]) -> tuple[BackendMemory, ...]:
        before = self.list(namespace)
        system = self._system(namespace)
        try:
            # The library exposes analysis separately from add_note. Call its
            # own analysis, then its native linking/evolution implementation.
            attributes = system.analyze_content(text)
            if not isinstance(attributes, dict) or not isinstance(attributes.get("context"), str):
                raise RuntimeError("A-MEM returned malformed note analysis")
            for field_name in ("keywords", "tags"):
                values = attributes.get(field_name)
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                    raise RuntimeError("A-MEM returned malformed note attributes")
            system.add_note(text, keywords=attributes["keywords"],
                            context=attributes["context"], tags=attributes["tags"])
            after = self._raw_list(namespace)
            changed = self.ledger.observe(namespace, before, after, metadata)
            self._persist(namespace)
            entries = tuple(self.ledger.decorate(namespace, e) for e in after if e.external_id in changed)
        except Exception:
            self.ledger.fail(namespace)
            raise
        self._audit.append({"operation": "native_note_formation", "namespace": namespace,
                            "changed_note_count": len(entries)})
        return entries

    def import_entries(self, namespace: str,
                       entries: Sequence[BackendMemory]) -> tuple[BackendMemory, ...]:
        system = self._system(namespace)
        if system.memories:
            raise RuntimeError("A-MEM checkpoint import requires an empty branch")
        checked_entries(entries)
        retained_ids = {e.external_id for e in entries}
        try:
            for source in entries:
                fields = deepcopy(source.metadata["amem_note"])
                if fields["id"] != source.external_id or _render(fields) != source.text:
                    raise RuntimeError("A-MEM checkpoint note identity/content mismatch")
                fields["links"] = [i for i in fields.get("links", []) if i in retained_ids]
                note = MemoryNote(**fields)
                system.memories[note.id] = note
                index_row = deepcopy(source.metadata["amem_index"])
                # A link cannot resurrect a record excluded from this branch.
                index_row["metadata"]["links"] = json.dumps(fields["links"])
                system.retriever.restore(note.id, index_row)
                self.ledger.bind(namespace, source, source.source_workstate_ids,
                                 complete=bool(source.metadata.get("provenance_complete", False)),
                                 metadata={"native_checkpoint_import": True,
                                           "source_native_memory_id": source.external_id,
                                           "lineage_policy": source.metadata.get("lineage_policy")})
            system.evo_cnt = max((int(e.metadata.get("amem_evo_cnt", 0)) for e in entries), default=0)
            self._persist(namespace)
            imported = self.list(namespace)
            verify_import(entries, imported)
        except Exception:
            self.ledger.fail(namespace)
            raise
        self._audit.append({"operation": "checkpoint_import", "namespace": namespace,
                            "source_count": len(entries), "native_inference_repeated": False,
                            "embeddings_copied": True})
        return imported

    def search(self, namespace: str, query: str, *, limit: int) -> tuple[BackendMemory, ...]:
        if limit < 1:
            raise ValueError("search limit must be positive")
        system = self._system(namespace)
        known = {e.external_id: e for e in self.list(namespace)}
        rows = system.search_agentic(query, k=limit)
        entries = []
        for row in rows:
            entry = known.get(str(row.get("id", "")))
            if entry is None:
                raise RuntimeError("A-MEM retrieval referenced a record outside the namespace")
            distance = float(row.get("score", 0.0))
            if not math.isfinite(distance):
                raise RuntimeError("A-MEM returned a non-finite distance")
            # Upstream returns distances (smaller is better). Preserve native
            # order; normalize solely for the common high-is-better interface.
            entries.append(replace(entry, score=1.0 / (1.0 + max(0.0, distance))))
        self._audit.append({"operation": "native_search", "namespace": namespace,
                            "result_count": len(entries)})
        return checked_entries(entries)

    def record(self) -> dict[str, object]:
        return {"schema_version": "amem_driver_v1", "backend_id": self.backend_id,
                "upstream_revision": self.upstream_revision,
                "native_formation": True, "native_retrieval": True,
                "branch_import_repeats_inference": False,
                "lineage_policy": "conservative_formation_scope",
                "operation_counts": dict(Counter(e["operation"] for e in self._audit)),
                "runtime": self.runtime_receipt()}
