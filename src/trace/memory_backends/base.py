"""Backend-neutral contracts for RETURN memory portability experiments.

The governance method decides which workstate IDs are eligible.  A backend
adapter owns only memory formation, storage, and retrieval.  Keeping these
surfaces separate prevents a native memory system from silently becoming a
second lifecycle governor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
import hashlib
from typing import Mapping, Protocol, Sequence

from ..mas_return_pipeline import AgentMemoryArm, EpisodeLongTermMemory, RETURNING_WORKER_ID
from ..private_episodic_memory import (
    PrivateEpisodicMemory,
    PrivateMemoryHit,
    PrivateMemoryItem,
    PrivateMemoryRetrieval,
)
from ..router_return_protocol import canonical_sha256


@dataclass(frozen=True)
class BackendMemory:
    """One normalized memory returned by an external native backend."""

    external_id: str
    text: str
    score: float = 1.0
    source_workstate_ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        return {
            "external_id": self.external_id,
            "text_sha256": hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest(),
            "score": self.score,
            "source_workstate_ids": list(self.source_workstate_ids),
            "metadata": dict(self.metadata),
        }


class NativeMemoryDriver(Protocol):
    """Smallest native surface needed by the benchmark adapter."""

    backend_id: str
    upstream_revision: str

    def reset_namespace(self, namespace: str) -> None:
        ...

    def write(
        self,
        namespace: str,
        text: str,
        metadata: Mapping[str, object],
    ) -> tuple[BackendMemory, ...]:
        ...

    def list(self, namespace: str) -> tuple[BackendMemory, ...]:
        ...

    def import_entries(
        self,
        namespace: str,
        entries: Sequence[BackendMemory],
    ) -> tuple[BackendMemory, ...]:
        ...

    def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int,
    ) -> tuple[BackendMemory, ...]:
        ...

    def record(self) -> dict[str, object]:
        ...


def _namespace(
    backend_id: str,
    episode_id: str,
    principal_id: str,
    branch: str | None,
) -> str:
    payload = {
        "backend": backend_id,
        "episode": episode_id,
        "principal": principal_id,
        "branch": branch or "source",
    }
    return "trace:" + canonical_sha256(payload)[:40]


@dataclass
class ExternalEpisodeLongTermMemory:
    """Episode memory runtime backed by a native memory driver.

    A shadow ``EpisodeLongTermMemory`` retains identity, epoch, provenance,
    and immutable branch receipts.  Semantic formation and retrieval are
    delegated to the native backend.  The shadow never contributes retrieval
    text, so it cannot change backend outcomes.
    """

    shadow: EpisodeLongTermMemory
    driver: NativeMemoryDriver
    _bank_namespaces: dict[int, str] = field(default_factory=dict)
    _events: list[dict[str, object]] = field(default_factory=list)
    _initialized: set[str] = field(default_factory=set)
    _quarantined: set[str] = field(default_factory=set)
    _admitted: dict[str, frozenset[str]] = field(default_factory=dict)
    _checkpoints: dict[str, tuple[BackendMemory, ...]] = field(default_factory=dict)

    def _initialize(self, namespace: str) -> None:
        if namespace not in self._initialized:
            self.driver.reset_namespace(namespace)
            self._initialized.add(namespace)

    @property
    def episode_id(self) -> str:
        return self.shadow.episode_id

    def bank(self, principal_id: str) -> PrivateEpisodicMemory:
        bank = self.shadow.bank(principal_id)
        self._bank_namespaces.setdefault(
            id(bank),
            _namespace(
                self.driver.backend_id,
                self.episode_id,
                principal_id,
                None,
            ),
        )
        self._initialize(self._bank_namespaces[id(bank)])
        return bank

    def _bank_namespace(
        self,
        bank: PrivateEpisodicMemory,
        *,
        principal_id: str,
        branch: str | None,
    ) -> str:
        namespace = self._bank_namespaces.get(id(bank))
        if namespace is not None:
            return namespace
        namespace = _namespace(
            self.driver.backend_id,
            self.episode_id,
            principal_id,
            branch,
        )
        self._bank_namespaces[id(bank)] = namespace
        self._initialize(namespace)
        return namespace

    def remember(
        self,
        *,
        principal_id: str,
        task_id: str,
        intent: str,
        experience: str,
        obligation_ids: Sequence[str] = (),
        dependency_ids: Sequence[str] = (),
        source_workstate_id: str | None = None,
        attributes: Mapping[str, object] | None = None,
        bank: PrivateEpisodicMemory | None = None,
        branch: str | None = None,
    ) -> PrivateMemoryItem:
        target = bank or self.bank(principal_id)
        namespace = self._bank_namespace(target, principal_id=principal_id, branch=branch)
        if namespace in self._quarantined or namespace in self._checkpoints:
            raise RuntimeError("cannot write to quarantined or frozen source memory")
        if namespace in self._admitted and source_workstate_id is not None:
            if source_workstate_id not in self._admitted[namespace]:
                raise PermissionError("return write references a non-admitted source")
        item = self.shadow.remember(
            principal_id=principal_id,
            task_id=task_id,
            intent=intent,
            experience=experience,
            obligation_ids=obligation_ids,
            dependency_ids=dependency_ids,
            source_workstate_id=source_workstate_id,
            attributes=attributes,
            bank=target,
            branch=branch,
        )
        namespace = self._bank_namespace(
            target,
            principal_id=principal_id,
            branch=branch,
        )
        native = self.driver.write(
            namespace,
            experience,
            {
                "shadow_memory_id": item.memory_id,
                "principal_id": principal_id,
                "task_id": task_id,
                "intent": intent,
                "source_workstate_id": source_workstate_id,
                "branch": branch,
                "attributes": dict(attributes or {}),
            },
        )
        self._events.append(
            {
                "event": "native_memory_written",
                "principal_id": principal_id,
                "branch": branch,
                "namespace_sha256": canonical_sha256(namespace),
                "shadow_memory_id": item.memory_id,
                "native_memory_ids": [entry.external_id for entry in native],
                "source_workstate_id": source_workstate_id,
            }
        )
        return item

    @staticmethod
    def _entry_item(
        entry: BackendMemory,
        *,
        bank: PrivateEpisodicMemory,
        intent: str,
    ) -> PrivateMemoryItem:
        source_id = (
            entry.source_workstate_ids[0]
            if entry.source_workstate_ids
            else None
        )
        template = next(
            (
                item
                for item in bank.items.values()
                if item.memory_id
                == str(entry.metadata.get("shadow_memory_id") or "")
                or (
                    source_id is not None
                    and item.source_workstate_id == source_id
                )
            ),
            None,
        )
        if template is not None:
            return replace(
                template,
                memory_id=(
                    f"external:{entry.external_id}"
                ),
                experience=entry.text,
                embedding=(),
                embedding_model_id=None,
                provenance_sha256="",
                attributes={
                    **dict(template.attributes),
                    "native_backend_memory_id": entry.external_id,
                    "source_workstate_ids": list(entry.source_workstate_ids),
                },
            ).with_default_provenance()
        return PrivateMemoryItem(
            memory_id=f"external:{entry.external_id}",
            owner_principal_id=bank.owner_principal_id,
            owner_instance_id=bank.owner_instance_id,
            role_id=bank.role_id,
            task_id="native-backend-retrieval",
            intent=intent,
            experience=entry.text,
            created_epoch=bank.epoch,
            valid_from_epoch=bank.epoch,
            source_workstate_id=source_id,
            attributes={
                "native_backend_memory_id": entry.external_id,
                "source_workstate_ids": list(entry.source_workstate_ids),
            },
        ).with_default_provenance()

    def recall(
        self,
        *,
        principal_id: str,
        intent: str,
        k1: int = 5,
        k2: int = 3,
        similarity_threshold: float = 0.0,
        bank: PrivateEpisodicMemory | None = None,
        branch: str | None = None,
    ) -> PrivateMemoryRetrieval:
        from .lineage import checked_entries

        if k1 < 1 or k2 < 1 or k2 > k1:
            raise ValueError("retrieval requires 1 <= k2 <= k1")
        target = bank or self.bank(principal_id)
        if target.owner_principal_id != principal_id:
            raise PermissionError("memory recall principal mismatch")
        namespace = self._bank_namespace(
            target,
            principal_id=principal_id,
            branch=branch,
        )
        if namespace in self._quarantined:
            raise PermissionError("quarantined native memory is not readable before return")
        admissible = checked_entries(self.driver.list(namespace))
        allowed = self._admitted.get(namespace)
        if allowed is not None:
            admissible = tuple(e for e in admissible if self._eligible(e, allowed))
        known = {e.external_id: e for e in admissible}
        entries = checked_entries(self.driver.search(namespace, intent, limit=k1))
        for entry in entries:
            canonical = known.get(entry.external_id)
            if canonical is None or entry.text != canonical.text or (
                set(entry.source_workstate_ids) != set(canonical.source_workstate_ids)
            ):
                raise RuntimeError("native search escaped its admitted snapshot")
        selected_entries = tuple(
            entry
            for entry in entries
            if entry.score >= similarity_threshold
        )[: max(1, k2)]
        hits = tuple(
            PrivateMemoryHit(
                memory_id=f"external:{entry.external_id}",
                similarity=float(entry.score),
                utility=0.5,
                composite_score=float(entry.score),
                item=self._entry_item(entry, bank=target, intent=intent),
            )
            for entry in selected_entries
        )
        retrieval = PrivateMemoryRetrieval(
            requesting_principal_id=principal_id,
            intent=intent,
            epoch=target.epoch,
            k1=k1,
            k2=k2,
            similarity_threshold=similarity_threshold,
            utility_weight=0.0,
            similarity_backend=self.driver.backend_id,
            query_embedding_sha256=None,
            admissible_item_ids=tuple(
                f"external:{entry.external_id}" for entry in admissible
            ),
            recalled_item_ids=tuple(
                f"external:{entry.external_id}" for entry in entries
            ),
            selected=hits,
        )
        self._events.append(
            {
                "event": "native_memory_recalled",
                "principal_id": principal_id,
                "branch": branch,
                "namespace_sha256": canonical_sha256(namespace),
                "selected_memory_ids": [hit.memory_id for hit in hits],
            }
        )
        return retrieval

    def quarantine_principal(
        self,
        principal_id: str,
        *,
        bank: PrivateEpisodicMemory | None = None,
        branch: str | None = None,
    ) -> str:
        target = bank or self.bank(principal_id)
        namespace = self._bank_namespace(target, principal_id=principal_id, branch=branch)
        digest = self.shadow.quarantine_principal(
            principal_id,
            bank=bank,
            branch=branch,
        )
        self._quarantined.add(namespace)
        return digest

    def quarantine_returning_worker(self) -> str:
        return self.quarantine_principal(RETURNING_WORKER_ID)

    @staticmethod
    def _eligible(entry: BackendMemory, admitted: frozenset[str]) -> bool:
        return bool(entry.source_workstate_ids) and (
            set(entry.source_workstate_ids) <= admitted
        ) and bool(entry.metadata.get("provenance_complete", True))

    def fork_return_arm(
        self,
        *,
        arm: str,
        admitted_workstate_ids: Sequence[str],
    ) -> AgentMemoryArm:
        from .lineage import checked_entries, verify_import

        if arm in self.shadow.branches:
            raise ValueError(f"return branch already exists: {arm}")
        branch = self.shadow.fork_return_arm(
            arm=arm,
            admitted_workstate_ids=admitted_workstate_ids,
        )
        source_namespace = _namespace(
            self.driver.backend_id,
            self.episode_id,
            branch.returning_worker.owner_principal_id,
            None,
        )
        branch_namespace = _namespace(
            self.driver.backend_id,
            self.episode_id,
            branch.returning_worker.owner_principal_id,
            arm,
        )
        admitted = frozenset(admitted_workstate_ids)
        current = tuple(sorted(checked_entries(self.driver.list(source_namespace)),
                               key=lambda entry: entry.external_id))
        if source_namespace not in self._checkpoints:
            self._checkpoints[source_namespace] = deepcopy(current)
        source_entries = self._checkpoints[source_namespace]
        source_digest = canonical_sha256([e.record() for e in source_entries])
        if canonical_sha256([e.record() for e in current]) != source_digest:
            raise RuntimeError("native source checkpoint changed between return branches")
        selected = tuple(
            entry
            for entry in source_entries
            if self._eligible(entry, admitted)
        )
        self.driver.reset_namespace(branch_namespace)
        imported = self.driver.import_entries(branch_namespace, selected)
        verify_import(selected, imported)
        self._initialized.add(branch_namespace)
        self._admitted[branch_namespace] = admitted
        self._bank_namespaces[id(branch.returning_worker)] = branch_namespace
        self._events.append(
            {
                "event": "native_return_branch_forked",
                "arm": arm,
                "source_native_checkpoint_sha256": source_digest,
                "source_admission_policy": "all_dependencies_required",
                "source_namespace_sha256": canonical_sha256(
                    source_namespace
                ),
                "branch_namespace_sha256": canonical_sha256(
                    branch_namespace
                ),
                "admitted_workstate_ids": sorted(admitted),
                "source_native_memory_ids": [
                    entry.external_id for entry in selected
                ],
                "branch_native_memory_ids": [
                    entry.external_id for entry in imported
                ],
            }
        )
        return branch

    def record(
        self,
        *,
        include_private_experience: bool = True,
    ) -> dict[str, object]:
        return {
            "schema_version": "external_episode_memory_v1",
            "episode_id": self.episode_id,
            "backend_id": self.driver.backend_id,
            "upstream_revision": self.driver.upstream_revision,
            "governance_backend_separation": True,
            "utility_learning_enabled": False,
            "native_events": list(self._events),
            "native_driver": self.driver.record(),
            "lifecycle_shadow": self.shadow.record(
                include_private_experience=include_private_experience
            ),
        }


__all__ = [
    "BackendMemory",
    "ExternalEpisodeLongTermMemory",
    "NativeMemoryDriver",
]
