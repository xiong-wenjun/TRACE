"""Principal-isolated episodic memory with MemRL-style utility learning.

This module deliberately separates two concerns:

* private memory decides which of one principal's *admissible* experiences are
  useful for the current intent;
* Router/TRACE governance decides which historical items are admissible after
  a RETURN event.

Utility must never override identity, epoch, status, or scope checks.  The
retrieval order is therefore: hard admission filters -> semantic recall ->
utility-aware reranking.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import re
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence


_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required(value: str, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _required_verbatim(value: object, name: str) -> str:
    text = "" if value is None else str(value)
    if not text.strip():
        raise ValueError(f"{name} must be non-empty")
    return text


def _unique(values: Iterable[str], name: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(_required(value, name) for value in values))
    return result


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in _TOKEN.findall(text))


def _cosine(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _vector_cosine(
    left: Sequence[float], right: Sequence[float]
) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    norm_a = math.sqrt(sum(value * value for value in left))
    norm_b = math.sqrt(sum(value * value for value in right))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _zscore(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    mean = fmean(values)
    deviation = pstdev(values)
    if deviation <= 1e-12:
        return tuple(0.0 for _ in values)
    return tuple((value - mean) / deviation for value in values)


class PrivateMemoryStatus(str, Enum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


@dataclass(frozen=True)
class PrivateMemoryItem:
    """One Intent-Experience-Utility triplet in a principal namespace."""

    memory_id: str
    owner_principal_id: str
    owner_instance_id: str
    role_id: str
    task_id: str
    intent: str
    experience: str
    embedding: tuple[float, ...] = ()
    embedding_model_id: str | None = None
    utility: float = 0.5
    reward_count: int = 0
    access_count: int = 0
    created_epoch: int = 1
    valid_from_epoch: int = 1
    valid_until_epoch: int | None = None
    status: PrivateMemoryStatus = PrivateMemoryStatus.ACTIVE
    obligation_ids: tuple[str, ...] = ()
    dependency_ids: tuple[str, ...] = ()
    source_workstate_id: str | None = None
    provenance_sha256: str = ""
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.memory_id, "memory_id")
        _required(self.owner_principal_id, "owner_principal_id")
        _required(self.owner_instance_id, "owner_instance_id")
        _required(self.role_id, "role_id")
        _required(self.task_id, "task_id")
        _required(self.intent, "intent")
        _required(self.experience, "experience")
        if any(not math.isfinite(float(value)) for value in self.embedding):
            raise ValueError("embedding values must be finite")
        if self.embedding and not self.embedding_model_id:
            raise ValueError(
                "embedding_model_id is required when embedding is present"
            )
        if self.embedding_model_id is not None:
            _required(self.embedding_model_id, "embedding_model_id")
        _unique(self.obligation_ids, "obligation_ids")
        _unique(self.dependency_ids, "dependency_ids")
        if not 0.0 <= float(self.utility) <= 1.0:
            raise ValueError("utility must be in [0, 1]")
        if self.reward_count < 0 or self.access_count < 0:
            raise ValueError("memory counters must be non-negative")
        if self.created_epoch < 1 or self.valid_from_epoch < 1:
            raise ValueError("memory epochs must be positive")
        if (
            self.valid_until_epoch is not None
            and self.valid_until_epoch < self.valid_from_epoch
        ):
            raise ValueError("valid_until_epoch precedes valid_from_epoch")
        if not isinstance(self.status, PrivateMemoryStatus):
            raise TypeError("status must be PrivateMemoryStatus")
        if self.source_workstate_id is not None:
            _required(self.source_workstate_id, "source_workstate_id")
        if self.provenance_sha256:
            if len(self.provenance_sha256) != 64:
                raise ValueError("provenance_sha256 must be SHA-256")
            bytes.fromhex(self.provenance_sha256)

    def with_default_provenance(self) -> "PrivateMemoryItem":
        if self.provenance_sha256:
            return self
        return replace(
            self,
            provenance_sha256=canonical_sha256(
                {
                    "memory_id": self.memory_id,
                    "owner_principal_id": self.owner_principal_id,
                    "owner_instance_id": self.owner_instance_id,
                    "role_id": self.role_id,
                    "task_id": self.task_id,
                    "intent": self.intent,
                    "experience": self.experience,
                    "embedding_sha256": canonical_sha256(
                        list(self.embedding)
                    ),
                    "embedding_model_id": self.embedding_model_id,
                    "created_epoch": self.created_epoch,
                    "obligation_ids": list(self.obligation_ids),
                    "dependency_ids": list(self.dependency_ids),
                    "source_workstate_id": self.source_workstate_id,
                }
            ),
        )

    def is_admissible(
        self,
        *,
        requesting_principal_id: str,
        epoch: int,
    ) -> bool:
        return (
            requesting_principal_id == self.owner_principal_id
            and self.status is PrivateMemoryStatus.ACTIVE
            and self.valid_from_epoch <= epoch
            and (
                self.valid_until_epoch is None
                or epoch <= self.valid_until_epoch
            )
        )

    def record(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "owner_principal_id": self.owner_principal_id,
            "owner_instance_id": self.owner_instance_id,
            "role_id": self.role_id,
            "task_id": self.task_id,
            "intent": self.intent,
            "experience": self.experience,
            "embedding": list(self.embedding),
            "embedding_sha256": canonical_sha256(list(self.embedding)),
            "embedding_model_id": self.embedding_model_id,
            "utility": self.utility,
            "reward_count": self.reward_count,
            "access_count": self.access_count,
            "created_epoch": self.created_epoch,
            "valid_from_epoch": self.valid_from_epoch,
            "valid_until_epoch": self.valid_until_epoch,
            "status": self.status.value,
            "obligation_ids": list(self.obligation_ids),
            "dependency_ids": list(self.dependency_ids),
            "source_workstate_id": self.source_workstate_id,
            "provenance_sha256": self.provenance_sha256,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_record(
        cls, value: Mapping[str, object]
    ) -> "PrivateMemoryItem":
        raw_attributes = value.get("attributes")
        if raw_attributes is None:
            attributes: Mapping[str, object] = {}
        elif isinstance(raw_attributes, Mapping):
            attributes = dict(raw_attributes)
        else:
            raise ValueError("private memory attributes must be an object")
        raw_embedding = value.get("embedding") or ()
        if not isinstance(raw_embedding, Sequence) or isinstance(
            raw_embedding, (str, bytes)
        ):
            raise ValueError("private memory embedding must be a sequence")
        item = cls(
            memory_id=_required(value.get("memory_id"), "memory_id"),
            owner_principal_id=_required(
                value.get("owner_principal_id"),
                "owner_principal_id",
            ),
            owner_instance_id=_required(
                value.get("owner_instance_id"),
                "owner_instance_id",
            ),
            role_id=_required(value.get("role_id"), "role_id"),
            task_id=_required_verbatim(value.get("task_id"), "task_id"),
            intent=_required_verbatim(value.get("intent"), "intent"),
            experience=_required_verbatim(
                value.get("experience"), "experience"
            ),
            embedding=tuple(float(item) for item in raw_embedding),
            embedding_model_id=(
                str(value["embedding_model_id"])
                if value.get("embedding_model_id") is not None
                else None
            ),
            utility=float(value.get("utility", 0.5)),
            reward_count=int(value.get("reward_count", 0)),
            access_count=int(value.get("access_count", 0)),
            created_epoch=int(value.get("created_epoch", 1)),
            valid_from_epoch=int(value.get("valid_from_epoch", 1)),
            valid_until_epoch=(
                int(value["valid_until_epoch"])
                if value.get("valid_until_epoch") is not None
                else None
            ),
            status=PrivateMemoryStatus(
                str(value.get("status") or PrivateMemoryStatus.ACTIVE.value)
            ),
            obligation_ids=tuple(
                str(item) for item in value.get("obligation_ids") or ()
            ),
            dependency_ids=tuple(
                str(item) for item in value.get("dependency_ids") or ()
            ),
            source_workstate_id=(
                str(value["source_workstate_id"])
                if value.get("source_workstate_id") is not None
                else None
            ),
            provenance_sha256=str(
                value.get("provenance_sha256") or ""
            ),
            attributes=attributes,
        )
        expected_embedding = value.get("embedding_sha256")
        if (
            expected_embedding is not None
            and str(expected_embedding)
            != canonical_sha256(list(item.embedding))
        ):
            raise ValueError("private memory embedding digest mismatch")
        return item


@dataclass(frozen=True)
class PrivateMemoryHit:
    memory_id: str
    similarity: float
    utility: float
    composite_score: float
    item: PrivateMemoryItem

    def record(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "similarity": self.similarity,
            "utility": self.utility,
            "composite_score": self.composite_score,
            "item": self.item.record(),
        }


@dataclass(frozen=True)
class PrivateMemoryRetrieval:
    requesting_principal_id: str
    intent: str
    epoch: int
    k1: int
    k2: int
    similarity_threshold: float
    utility_weight: float
    similarity_backend: str
    query_embedding_sha256: str | None
    admissible_item_ids: tuple[str, ...]
    recalled_item_ids: tuple[str, ...]
    selected: tuple[PrivateMemoryHit, ...]

    def render(
        self, max_chars: int = 6000, *, include_utility: bool = True
    ) -> str:
        rows: list[str] = []
        used = 0
        for index, hit in enumerate(self.selected, 1):
            metadata = (
                f"utility={hit.utility:.3f}; "
                if include_utility
                else ""
            )
            row = (
                f"[Private memory {index}; {metadata}"
                f"similarity={hit.similarity:.3f}]\n"
                f"{hit.item.experience}"
            )
            if rows and used + len(row) > max_chars:
                break
            rows.append(row)
            used += len(row)
        return "\n\n".join(rows)

    def record(self) -> dict[str, object]:
        return {
            "requesting_principal_id": self.requesting_principal_id,
            "intent": self.intent,
            "epoch": self.epoch,
            "k1": self.k1,
            "k2": self.k2,
            "similarity_threshold": self.similarity_threshold,
            "utility_weight": self.utility_weight,
            "similarity_backend": self.similarity_backend,
            "query_embedding_sha256": self.query_embedding_sha256,
            "admissible_item_ids": list(self.admissible_item_ids),
            "recalled_item_ids": list(self.recalled_item_ids),
            "selected": [hit.record() for hit in self.selected],
        }


@dataclass
class PrivateEpisodicMemory:
    """An isolated, serializable memory bank for one principal."""

    owner_principal_id: str
    owner_instance_id: str
    role_id: str
    epoch: int = 1
    items: dict[str, PrivateMemoryItem] = field(default_factory=dict)
    retrieval_log: list[PrivateMemoryRetrieval] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.owner_principal_id = _required(
            self.owner_principal_id, "owner_principal_id"
        )
        self.owner_instance_id = _required(
            self.owner_instance_id, "owner_instance_id"
        )
        self.role_id = _required(self.role_id, "role_id")
        if self.epoch < 1:
            raise ValueError("epoch must be positive")

    def _authorize(self, requesting_principal_id: str) -> None:
        if requesting_principal_id != self.owner_principal_id:
            raise PermissionError(
                "private memory namespace is accessible only to its principal"
            )

    def remember(
        self,
        *,
        requesting_principal_id: str,
        task_id: str,
        intent: str,
        experience: str,
        embedding: Sequence[float] = (),
        embedding_model_id: str | None = None,
        memory_id: str | None = None,
        initial_utility: float = 0.5,
        obligation_ids: Sequence[str] = (),
        dependency_ids: Sequence[str] = (),
        source_workstate_id: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> PrivateMemoryItem:
        self._authorize(requesting_principal_id)
        identifier = memory_id or (
            "pmem:"
            + canonical_sha256(
                {
                    "principal": self.owner_principal_id,
                    "instance": self.owner_instance_id,
                    "task": task_id,
                    "intent": intent,
                    "experience": experience,
                    "ordinal": len(self.items),
                }
            )[:24]
        )
        if identifier in self.items:
            raise ValueError(f"private memory already exists: {identifier}")
        item = PrivateMemoryItem(
            memory_id=identifier,
            owner_principal_id=self.owner_principal_id,
            owner_instance_id=self.owner_instance_id,
            role_id=self.role_id,
            task_id=task_id,
            intent=intent,
            experience=experience,
            embedding=tuple(float(value) for value in embedding),
            embedding_model_id=embedding_model_id,
            utility=initial_utility,
            created_epoch=self.epoch,
            valid_from_epoch=self.epoch,
            obligation_ids=_unique(obligation_ids, "obligation_ids"),
            dependency_ids=_unique(dependency_ids, "dependency_ids"),
            source_workstate_id=source_workstate_id,
            attributes=dict(attributes or {}),
        ).with_default_provenance()
        self.items[identifier] = item
        return item

    def retrieve(
        self,
        *,
        requesting_principal_id: str,
        intent: str,
        k1: int = 5,
        k2: int = 3,
        similarity_threshold: float = 0.0,
        utility_weight: float = 0.5,
        epoch: int | None = None,
        query_embedding: Sequence[float] = (),
        embedding_model_id: str | None = None,
    ) -> PrivateMemoryRetrieval:
        self._authorize(requesting_principal_id)
        if k1 < 1 or k2 < 1 or k2 > k1:
            raise ValueError("retrieval requires 1 <= k2 <= k1")
        if not 0.0 <= utility_weight <= 1.0:
            raise ValueError("utility_weight must be in [0, 1]")
        query_vector = tuple(float(value) for value in query_embedding)
        if any(not math.isfinite(value) for value in query_vector):
            raise ValueError("query embedding values must be finite")
        if query_vector and not embedding_model_id:
            raise ValueError(
                "embedding_model_id is required for vector retrieval"
            )
        target_epoch = self.epoch if epoch is None else epoch
        admissible = tuple(
            item
            for item in self.items.values()
            if item.is_admissible(
                requesting_principal_id=requesting_principal_id,
                epoch=target_epoch,
            )
        )
        def similarity(item: PrivateMemoryItem) -> float:
            if query_vector and item.embedding:
                if item.embedding_model_id != embedding_model_id:
                    return -1.0
                return _vector_cosine(query_vector, item.embedding)
            return _cosine(intent, f"{item.intent}\n{item.experience}")

        scored = tuple((item, similarity(item)) for item in admissible)
        recalled = tuple(
            sorted(
                (
                    (item, similarity)
                    for item, similarity in scored
                    if similarity >= similarity_threshold
                ),
                key=lambda pair: (-pair[1], pair[0].memory_id),
            )[:k1]
        )
        similarities = tuple(value for _, value in recalled)
        utilities = tuple(item.utility for item, _ in recalled)
        similarity_z = _zscore(similarities)
        utility_z = _zscore(utilities)
        hits = [
            PrivateMemoryHit(
                memory_id=item.memory_id,
                similarity=similarity,
                utility=item.utility,
                composite_score=(
                    (1.0 - utility_weight) * similarity_z[index]
                    + utility_weight * utility_z[index]
                ),
                item=item,
            )
            for index, (item, similarity) in enumerate(recalled)
        ]
        selected = tuple(
            sorted(
                hits,
                key=lambda hit: (
                    -hit.composite_score,
                    -hit.similarity,
                    hit.memory_id,
                ),
            )[:k2]
        )
        for hit in selected:
            self.items[hit.memory_id] = replace(
                self.items[hit.memory_id],
                access_count=self.items[hit.memory_id].access_count + 1,
            )
        result = PrivateMemoryRetrieval(
            requesting_principal_id=requesting_principal_id,
            intent=_required(intent, "intent"),
            epoch=target_epoch,
            k1=k1,
            k2=k2,
            similarity_threshold=similarity_threshold,
            utility_weight=utility_weight,
            similarity_backend=(
                "dense_embedding_cosine"
                if query_vector
                else "token_cosine"
            ),
            query_embedding_sha256=(
                canonical_sha256(list(query_vector))
                if query_vector
                else None
            ),
            admissible_item_ids=tuple(
                sorted(item.memory_id for item in admissible)
            ),
            recalled_item_ids=tuple(item.memory_id for item, _ in recalled),
            selected=selected,
        )
        self.retrieval_log.append(result)
        return result

    def apply_feedback(
        self,
        *,
        requesting_principal_id: str,
        memory_ids: Sequence[str],
        reward: float,
        alpha: float = 0.2,
    ) -> tuple[PrivateMemoryItem, ...]:
        """Apply MemRL's Monte-Carlo-style Q update to used memories."""

        self._authorize(requesting_principal_id)
        if not 0.0 <= reward <= 1.0:
            raise ValueError("reward must be in [0, 1]")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        updated: list[PrivateMemoryItem] = []
        for memory_id in _unique(memory_ids, "memory_ids"):
            try:
                item = self.items[memory_id]
            except KeyError as error:
                raise KeyError(f"unknown private memory: {memory_id}") from error
            value = item.utility + alpha * (reward - item.utility)
            item = replace(
                item,
                utility=min(1.0, max(0.0, value)),
                reward_count=item.reward_count + 1,
            )
            self.items[memory_id] = item
            updated.append(item)
        return tuple(updated)

    def quarantine_for_departure(
        self, *, requesting_principal_id: str
    ) -> str:
        self._authorize(requesting_principal_id)
        for memory_id, item in tuple(self.items.items()):
            if item.status is PrivateMemoryStatus.ACTIVE:
                self.items[memory_id] = replace(
                    item,
                    status=PrivateMemoryStatus.QUARANTINED,
                    valid_until_epoch=self.epoch,
                )
        return self.snapshot_sha256

    def fork_return_instance(
        self,
        *,
        requesting_principal_id: str,
        new_instance_id: str,
        target_epoch: int,
        admitted_memory_ids: Sequence[str],
    ) -> "PrivateEpisodicMemory":
        """Create a fresh instance containing only explicitly admitted items."""

        self._authorize(requesting_principal_id)
        if target_epoch <= self.epoch:
            raise ValueError("RETURN target_epoch must advance")
        selected = _unique(admitted_memory_ids, "admitted_memory_ids")
        unknown = set(selected) - set(self.items)
        if unknown:
            raise KeyError(
                "unknown admitted private memories: "
                + ",".join(sorted(unknown))
            )
        bank = PrivateEpisodicMemory(
            owner_principal_id=self.owner_principal_id,
            owner_instance_id=_required(new_instance_id, "new_instance_id"),
            role_id=self.role_id,
            epoch=target_epoch,
        )
        for memory_id in selected:
            source = self.items[memory_id]
            bank.items[memory_id] = replace(
                source,
                owner_instance_id=bank.owner_instance_id,
                created_epoch=target_epoch,
                valid_from_epoch=target_epoch,
                valid_until_epoch=None,
                status=PrivateMemoryStatus.ACTIVE,
                attributes={
                    **dict(source.attributes),
                    "readmitted_from_instance": source.owner_instance_id,
                    "readmitted_from_provenance": source.provenance_sha256,
                },
            )
        return bank

    def clone_branch(
        self,
        *,
        requesting_principal_id: str,
        new_instance_id: str,
    ) -> "PrivateEpisodicMemory":
        """Clone one principal's current snapshot for a counterfactual arm."""

        self._authorize(requesting_principal_id)
        bank = PrivateEpisodicMemory(
            owner_principal_id=self.owner_principal_id,
            owner_instance_id=_required(new_instance_id, "new_instance_id"),
            role_id=self.role_id,
            epoch=self.epoch,
        )
        bank.items = {
            memory_id: replace(
                item,
                owner_instance_id=bank.owner_instance_id,
                attributes={
                    **dict(item.attributes),
                    "branched_from_instance": self.owner_instance_id,
                },
            )
            for memory_id, item in self.items.items()
        }
        return bank

    @property
    def snapshot_sha256(self) -> str:
        return canonical_sha256(
            {
                "owner_principal_id": self.owner_principal_id,
                "owner_instance_id": self.owner_instance_id,
                "role_id": self.role_id,
                "epoch": self.epoch,
                "items": [
                    self.items[memory_id].record()
                    for memory_id in sorted(self.items)
                ],
            }
        )

    def record(self, *, include_experience: bool = True) -> dict[str, object]:
        rows = [
            self.items[memory_id].record() for memory_id in sorted(self.items)
        ]
        if not include_experience:
            rows = [
                {
                    **row,
                    "experience": "[redacted private experience]",
                    "embedding": [],
                }
                for row in rows
            ]
        return {
            "schema_version": "private_episodic_memory_v2",
            "owner_principal_id": self.owner_principal_id,
            "owner_instance_id": self.owner_instance_id,
            "role_id": self.role_id,
            "epoch": self.epoch,
            "snapshot_sha256": self.snapshot_sha256,
            "items": rows,
            "retrieval_count": len(self.retrieval_log),
        }

    @classmethod
    def from_record(
        cls, value: Mapping[str, object]
    ) -> "PrivateEpisodicMemory":
        raw_items = value.get("items")
        if not isinstance(raw_items, Sequence) or isinstance(
            raw_items, (str, bytes)
        ):
            raise ValueError("private memory record requires an items list")
        bank = cls(
            owner_principal_id=_required(
                value.get("owner_principal_id"),
                "owner_principal_id",
            ),
            owner_instance_id=_required(
                value.get("owner_instance_id"),
                "owner_instance_id",
            ),
            role_id=_required(value.get("role_id"), "role_id"),
            epoch=int(value.get("epoch", 1)),
        )
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                raise ValueError("private memory item must be an object")
            item = PrivateMemoryItem.from_record(raw_item)
            if item.owner_principal_id != bank.owner_principal_id:
                raise ValueError("private memory item principal mismatch")
            if item.owner_instance_id != bank.owner_instance_id:
                raise ValueError("private memory item instance mismatch")
            if item.memory_id in bank.items:
                raise ValueError(
                    f"duplicate private memory item: {item.memory_id}"
                )
            bank.items[item.memory_id] = item
        expected = str(value.get("snapshot_sha256") or "")
        if expected and expected != bank.snapshot_sha256:
            raise ValueError("private memory snapshot digest mismatch")
        return bank


@dataclass
class PrincipalMemoryRegistry:
    """Holds isolated Router and Worker banks without cross-principal reads."""

    banks: dict[str, PrivateEpisodicMemory] = field(default_factory=dict)

    def register(
        self,
        *,
        principal_id: str,
        instance_id: str,
        role_id: str,
        epoch: int = 1,
    ) -> PrivateEpisodicMemory:
        if principal_id in self.banks:
            raise ValueError(f"principal memory already registered: {principal_id}")
        bank = PrivateEpisodicMemory(
            owner_principal_id=principal_id,
            owner_instance_id=instance_id,
            role_id=role_id,
            epoch=epoch,
        )
        self.banks[principal_id] = bank
        return bank

    def owned(
        self, principal_id: str, *, requesting_principal_id: str
    ) -> PrivateEpisodicMemory:
        if principal_id != requesting_principal_id:
            raise PermissionError("registry forbids cross-principal private reads")
        try:
            return self.banks[principal_id]
        except KeyError as error:
            raise KeyError(f"unknown memory principal: {principal_id}") from error

    def replace_owned(
        self,
        principal_id: str,
        bank: PrivateEpisodicMemory,
        *,
        requesting_principal_id: str,
    ) -> None:
        if principal_id != requesting_principal_id:
            raise PermissionError("registry forbids cross-principal replacement")
        if bank.owner_principal_id != principal_id:
            raise ValueError("replacement bank principal mismatch")
        self.banks[principal_id] = bank

    def record(self, *, include_private_experience: bool = False) -> dict[str, object]:
        return {
            "schema_version": "principal_memory_registry_v2",
            "principals": {
                principal_id: bank.record(
                    include_experience=include_private_experience
                )
                for principal_id, bank in sorted(self.banks.items())
            },
        }

    @classmethod
    def from_record(
        cls, value: Mapping[str, object]
    ) -> "PrincipalMemoryRegistry":
        raw_principals = value.get("principals")
        if not isinstance(raw_principals, Mapping):
            raise ValueError(
                "principal memory registry requires a principals object"
            )
        registry = cls()
        for principal_id, raw_bank in raw_principals.items():
            if not isinstance(raw_bank, Mapping):
                raise ValueError("principal memory bank must be an object")
            bank = PrivateEpisodicMemory.from_record(raw_bank)
            if bank.owner_principal_id != str(principal_id):
                raise ValueError("principal memory registry key mismatch")
            registry.banks[str(principal_id)] = bank
        return registry
