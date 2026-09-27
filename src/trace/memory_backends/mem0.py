"""Official Mem0 OSS adapter for RETURN memory experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
from pathlib import Path
from collections import Counter
from typing import Any, Mapping, Sequence

from .base import BackendMemory
from .lineage import SourceLedger, checked_entries, source_ids, verify_import


MEM0_UPSTREAM_REVISION = "19cb89aff472325c707f64b2f34ae6afdbf7faf7"
MEM0_REQUIRED_VERSION = "2.0.19"


def _rows(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        if "results" not in value:
            raise RuntimeError("Mem0 response is missing results")
        value = value["results"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError("Mem0 results must be an array")
    if any(not isinstance(row, Mapping) for row in value):
        raise RuntimeError("Mem0 returned a malformed result row")
    return tuple(value)


def _source_ids(metadata: Mapping[str, object]) -> tuple[str, ...]:
    return source_ids(metadata)


def _entry(row: Mapping[str, object]) -> BackendMemory:
    nested = row.get("metadata")
    metadata = dict(nested) if isinstance(nested, Mapping) else {}
    for key in (
        "shadow_memory_id",
        "source_workstate_id",
        "source_workstate_ids",
        "principal_id",
        "task_id",
        "intent",
        "branch",
    ):
        if key in row and key not in metadata:
            metadata[key] = row[key]
    metadata["native_revision"] = row.get("updated_at") or row.get("created_at")
    return BackendMemory(
        external_id=str(row.get("id") or row.get("memory_id") or ""),
        text=str(row.get("memory") or row.get("text") or "").strip(),
        score=float(row.get("score", 1.0) or 0.0),
        source_workstate_ids=_source_ids(metadata),
        metadata=metadata,
    )


@dataclass
class Mem0Driver:
    """Thin adapter over ``mem0.Memory`` without reimplementing Mem0 logic."""

    memory: Any
    backend_id: str = "mem0_oss_v2_0_19"
    upstream_revision: str = MEM0_UPSTREAM_REVISION
    optional_spacy_model_available: bool = True
    _audit: list[dict[str, object]] = field(default_factory=list)
    ledger: SourceLedger = field(default_factory=SourceLedger)
    snapshot_limit: int = 1000

    @classmethod
    def build(
        cls,
        *,
        storage_dir: Path,
        generation_base_url: str,
        generation_model: str,
        generation_api_key: str,
        embedding_base_url: str,
        embedding_model: str,
        embedding_api_key: str,
        embedding_dimensions: int,
    ) -> "Mem0Driver":
        try:
            installed = importlib.metadata.version("mem0ai")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "Mem0 backend requires optional dependency "
                f"mem0ai=={MEM0_REQUIRED_VERSION}"
            ) from error
        if installed != MEM0_REQUIRED_VERSION:
            raise RuntimeError(
                "Mem0 version mismatch: expected "
                f"{MEM0_REQUIRED_VERSION}, found {installed}"
            )
        from mem0 import Memory

        # Mem0 treats spaCy NLP as optional and falls back to its built-in
        # lexical path when the model cannot be loaded.  On an offline worker,
        # mark that optional model unavailable up front so Mem0 does not try a
        # network installation during every process start.
        optional_spacy_model_available = True
        try:
            import spacy
            from mem0.utils import spacy_models

            optional_spacy_model_available = spacy.util.is_package(
                "en_core_web_sm"
            )
            if not optional_spacy_model_available:
                spacy_models._load_failed_full = True
                spacy_models._load_failed_lemma = True
        except ImportError:
            optional_spacy_model_available = False

        storage_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "version": "v1.1",
            "history_db_path": str(storage_dir / "history.db"),
            "custom_instructions": (
                "Extract concise, evidence-grounded state memories useful "
                "for the stated task. Preserve updates and deletions as "
                "current facts. Do not infer benchmark labels or answers."
            ),
            "llm": {
                "provider": "openai",
                "config": {
                    "model": generation_model,
                    "api_key": generation_api_key,
                    "openai_base_url": generation_base_url,
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "is_reasoning_model": False,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding_model,
                    "api_key": embedding_api_key,
                    "openai_base_url": embedding_base_url,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "trace_mem0",
                    "embedding_model_dims": embedding_dimensions,
                    "path": str(storage_dir / "qdrant"),
                    "on_disk": True,
                },
            },
        }
        return cls(
            memory=Memory.from_config(config),
            optional_spacy_model_available=optional_spacy_model_available,
            ledger=SourceLedger(storage_dir / "source_receipts"),
        )

    def reset_namespace(self, namespace: str) -> None:
        try:
            self.memory.delete_all(run_id=namespace)
        except Exception as error:
            # A never-created namespace is already reset.  Other failures are
            # surfaced so backend outages cannot silently change an arm.
            if getattr(error, "status_code", None) != 404:
                raise
        self.ledger.reset(namespace)
        self._audit.append({"operation": "reset", "namespace": namespace})

    def write(
        self,
        namespace: str,
        text: str,
        metadata: Mapping[str, object],
    ) -> tuple[BackendMemory, ...]:
        before = self.list(namespace)
        try:
            result = self.memory.add(
                messages=[{"role": "user", "content": text}],
                run_id=namespace, metadata=dict(metadata), infer=True,
            )
            _rows(result)  # ADD/UPDATE/DELETE events are not a full checkpoint.
            changed = self.ledger.observe(namespace, before, self._raw_list(namespace), metadata)
            entries = tuple(e for e in self.list(namespace) if e.external_id in changed)
        except Exception:
            self.ledger.fail(namespace)
            raise
        self._audit.append(
            {
                "operation": "native_add",
                "namespace": namespace,
                "result_count": len(entries),
            }
        )
        return entries

    def _raw_list(self, namespace: str) -> tuple[BackendMemory, ...]:
        result = self.memory.get_all(
            filters={"run_id": namespace},
            top_k=self.snapshot_limit + 1,
        )
        rows = _rows(result)
        if len(rows) > self.snapshot_limit:
            raise RuntimeError("Mem0 snapshot exceeds adapter capacity; refusing truncation")
        return checked_entries(tuple(_entry(row) for row in rows))

    def list(self, namespace: str) -> tuple[BackendMemory, ...]:
        self.ledger.check(namespace)
        return tuple(self.ledger.decorate(namespace, e) for e in self._raw_list(namespace))

    def import_entries(
        self,
        namespace: str,
        entries: Sequence[BackendMemory],
    ) -> tuple[BackendMemory, ...]:
        self.ledger.check(namespace)
        checked_entries(entries)
        try:
            imported: list[BackendMemory] = []
            for source in entries:
                metadata = {
                    **dict(source.metadata),
                    "source_workstate_ids": list(source.source_workstate_ids),
                    "native_checkpoint_import": True,
                    "source_native_memory_id": source.external_id,
                }
                result = self.memory.add(
                    messages=[{"role": "user", "content": source.text}],
                    run_id=namespace,
                    metadata=metadata,
                    infer=False,
                )
                created = checked_entries(tuple(_entry(row) for row in _rows(result)))
                if len(created) != 1 or created[0].text != source.text:
                    self.ledger.fail(namespace)
                    raise RuntimeError("Mem0 direct import did not preserve one source record")
                self.ledger.bind(namespace, created[0], source.source_workstate_ids,
                                 complete=bool(source.metadata.get("provenance_complete", bool(source.source_workstate_ids))),
                                 metadata=metadata)
                imported.append(self.ledger.decorate(namespace, created[0]))
            verify_import(entries, self.list(namespace))
            self._audit.append(
                {
                    "operation": "checkpoint_import",
                    "namespace": namespace,
                    "source_count": len(entries),
                    "result_count": len(imported),
                    "native_inference_repeated": False,
                }
            )
            return tuple(imported)
        except Exception:
            self.ledger.fail(namespace)
            raise

    def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int,
    ) -> tuple[BackendMemory, ...]:
        self.ledger.check(namespace)
        if limit < 1:
            raise ValueError("search limit must be positive")
        result = self.memory.search(
            query,
            filters={"run_id": namespace},
            top_k=max(1, limit),
            threshold=0.0,
            rerank=False,
        )
        entries = checked_entries(tuple(
            self.ledger.decorate(namespace, _entry(row)) for row in _rows(result)
        ))
        self._audit.append(
            {
                "operation": "native_search",
                "namespace": namespace,
                "limit": limit,
                "result_count": len(entries),
            }
        )
        return entries

    def record(self) -> dict[str, object]:
        counts = Counter(str(item["operation"]) for item in self._audit)
        return {
            "schema_version": "mem0_driver_v2",
            "adapter_revision": "source_complete_snapshots_v2",
            "lineage_policy": "conservative_formation_scope",
            "backend_id": self.backend_id,
            "package": f"mem0ai=={MEM0_REQUIRED_VERSION}",
            "upstream_revision": self.upstream_revision,
            "native_formation": True,
            "native_retrieval": True,
            "branch_import_repeats_inference": False,
            "optional_spacy_model_available": (
                self.optional_spacy_model_available
            ),
            "operation_counts": dict(sorted(counts.items())),
        }


__all__ = ["Mem0Driver"]
