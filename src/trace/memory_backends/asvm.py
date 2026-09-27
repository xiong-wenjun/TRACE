"""Agent-Scoped Versioned Memory (ASVM) backend.

ASVM is the paper-facing backend name for the existing episode-scoped,
principal-isolated memory implementation.  It owns storage, checkpointing,
branching, and retrieval; TRACE and the comparison methods only decide which
shared workstate IDs are eligible for a return branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Mapping, Sequence

from ..mas_return_pipeline import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MAX_CHARS,
    EpisodeLongTermMemory,
    MemoryEmbeddingProvider,
)
from ..private_episodic_memory import PrivateMemoryRetrieval
from ..router_return_protocol import canonical_sha256
from ..unified_mas_contract import RETURNING_AGENT_ID


def _required(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(_required(value, name) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique")
    return normalized


@dataclass(frozen=True)
class AsvmReturnView:
    """One branch-local memory view retrieved after governance."""

    arm: str
    source_checkpoint_sha256: str
    admitted_workstate_ids: tuple[str, ...]
    retrieved_source_workstate_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    texts: tuple[str, ...]
    retrieval: PrivateMemoryRetrieval

    def record(self) -> dict[str, object]:
        retrieval_record = {
            "requesting_principal_id": self.retrieval.requesting_principal_id,
            "epoch": self.retrieval.epoch,
            "k1": self.retrieval.k1,
            "k2": self.retrieval.k2,
            "similarity_threshold": self.retrieval.similarity_threshold,
            "utility_weight": self.retrieval.utility_weight,
            "similarity_backend": self.retrieval.similarity_backend,
            "query_embedding_sha256": (
                self.retrieval.query_embedding_sha256
            ),
            "admissible_item_ids": list(
                self.retrieval.admissible_item_ids
            ),
            "recalled_item_ids": list(self.retrieval.recalled_item_ids),
            "selected": [
                {
                    "memory_id": hit.memory_id,
                    "similarity": hit.similarity,
                    "utility": hit.utility,
                    "composite_score": hit.composite_score,
                    "source_workstate_id": hit.item.source_workstate_id,
                    "provenance_sha256": hit.item.provenance_sha256,
                }
                for hit in self.retrieval.selected
            ],
        }
        return {
            "schema_version": "asvm_return_view_v1",
            "arm": self.arm,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "admitted_workstate_ids": list(self.admitted_workstate_ids),
            "retrieved_source_workstate_ids": list(
                self.retrieved_source_workstate_ids
            ),
            "retrieved_memory_ids": list(self.retrieved_memory_ids),
            "text_sha256": [
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                for text in self.texts
            ],
            "retrieval": retrieval_record,
            "direct_candidate_prompt_injection": False,
        }


@dataclass
class AgentScopedVersionedMemoryBackend:
    """Complete ASVM lifecycle around an ``EpisodeLongTermMemory`` runtime."""

    runtime: EpisodeLongTermMemory
    returning_principal_id: str = RETURNING_AGENT_ID
    _source_checkpoint_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _write_audit: list[dict[str, object]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _return_views: dict[str, AsvmReturnView] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def build(
        cls,
        episode_id: str,
        *,
        worker_principal_ids: Sequence[str] | None = None,
        embedding_provider: MemoryEmbeddingProvider | None = None,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        embedding_max_chars: int = DEFAULT_EMBEDDING_MAX_CHARS,
    ) -> "AgentScopedVersionedMemoryBackend":
        return cls(
            runtime=EpisodeLongTermMemory.build(
                episode_id,
                worker_principal_ids=worker_principal_ids,
                embedding_provider=embedding_provider,
                embedding_dimensions=embedding_dimensions,
                embedding_max_chars=embedding_max_chars,
            )
        )

    @property
    def episode_id(self) -> str:
        return self.runtime.episode_id

    @property
    def source_checkpoint_sha256(self) -> str | None:
        return self._source_checkpoint_sha256

    def write(
        self,
        *,
        principal_id: str,
        task_id: str,
        intent: str,
        experience: str,
        source_workstate_id: str,
        phase: str,
        receipt_sha256: str | None = None,
    ) -> str:
        """Write one actor-produced memory before the source checkpoint."""

        if self._source_checkpoint_sha256 is not None:
            raise RuntimeError("ASVM source checkpoint is already frozen")
        attributes: dict[str, object] = {
            "phase": _required(phase, "phase"),
            "benchmark_oracle_visible": False,
        }
        if receipt_sha256 is not None:
            attributes["source_receipt_sha256"] = _required(
                receipt_sha256,
                "receipt_sha256",
            )
        item = self.runtime.remember(
            principal_id=_required(principal_id, "principal_id"),
            task_id=_required(task_id, "task_id"),
            intent=_required(intent, "intent"),
            experience=_required(experience, "experience"),
            source_workstate_id=_required(
                source_workstate_id,
                "source_workstate_id",
            ),
            attributes=attributes,
        )
        self._write_audit.append(
            {
                "principal_id": item.owner_principal_id,
                "memory_id": item.memory_id,
                "source_workstate_id": item.source_workstate_id,
                "phase": attributes["phase"],
                "provenance_sha256": item.provenance_sha256,
            }
        )
        return item.memory_id

    def quarantine_returning_agent(self) -> str:
        if self._source_checkpoint_sha256 is not None:
            raise RuntimeError("ASVM source checkpoint is already frozen")
        return self.runtime.quarantine_principal(
            self.returning_principal_id
        )

    def freeze_source_checkpoint(self) -> str:
        """Freeze the common pre-treatment source for all method branches."""

        if self._source_checkpoint_sha256 is not None:
            return self._source_checkpoint_sha256
        self._source_checkpoint_sha256 = canonical_sha256(
            self.runtime.record(include_private_experience=True)
        )
        return self._source_checkpoint_sha256

    def fork_return_view(
        self,
        *,
        arm: str,
        admitted_workstate_ids: Sequence[str],
        workstate_texts: Mapping[str, str],
        task_id: str,
        intent: str,
    ) -> AsvmReturnView:
        """Fork agent1, materialize admitted shared state, and retrieve it."""

        checkpoint_sha256 = self._source_checkpoint_sha256
        if checkpoint_sha256 is None:
            raise RuntimeError("ASVM source checkpoint must be frozen first")
        arm_id = _required(arm, "arm")
        if arm_id in self._return_views:
            raise ValueError(f"ASVM return arm already exists: {arm_id}")
        admitted = _unique(admitted_workstate_ids, "admitted_workstate_ids")
        unknown = set(admitted) - set(workstate_texts)
        if unknown:
            raise KeyError(
                "ASVM cannot materialize unknown workstate IDs: "
                + ",".join(sorted(unknown))
            )
        branch = self.runtime.fork_return_arm(
            arm=arm_id,
            admitted_workstate_ids=admitted,
        )
        bank = branch.returning_worker
        existing_sources = {
            item.source_workstate_id
            for item in bank.items.values()
            if item.source_workstate_id is not None
        }
        for workstate_id in admitted:
            if workstate_id in existing_sources:
                continue
            self.runtime.remember(
                principal_id=self.returning_principal_id,
                task_id=_required(task_id, "task_id"),
                intent=_required(intent, "intent"),
                experience=_required(
                    workstate_texts[workstate_id],
                    f"workstate_texts[{workstate_id}]",
                ),
                source_workstate_id=workstate_id,
                attributes={
                    "phase": "return_readmission",
                    "readmitted_from_shared_workstate": True,
                    "benchmark_oracle_visible": False,
                },
                bank=bank,
                branch=arm_id,
            )
        retrieval_width = max(1, len(bank.items))
        retrieval = self.runtime.recall(
            principal_id=self.returning_principal_id,
            intent=intent,
            k1=retrieval_width,
            k2=retrieval_width,
            similarity_threshold=-1.0,
            bank=bank,
            branch=arm_id,
        )
        retrieved_sources = tuple(
            str(hit.item.source_workstate_id)
            for hit in retrieval.selected
            if hit.item.source_workstate_id is not None
        )
        if set(retrieved_sources) != set(admitted):
            raise RuntimeError(
                "ASVM retrieval did not preserve the governed admission set"
            )
        view = AsvmReturnView(
            arm=arm_id,
            source_checkpoint_sha256=checkpoint_sha256,
            admitted_workstate_ids=admitted,
            retrieved_source_workstate_ids=retrieved_sources,
            retrieved_memory_ids=tuple(
                hit.memory_id for hit in retrieval.selected
            ),
            texts=tuple(hit.item.experience for hit in retrieval.selected),
            retrieval=retrieval,
        )
        self._return_views[arm_id] = view
        return view

    def record(self) -> dict[str, object]:
        semantic_backend = (
            "dense_embedding_cosine"
            if self.runtime.embedding_model_id is not None
            else "token_cosine"
        )
        return {
            "schema_version": "asvm_backend_v1",
            "backend_name": "Agent-Scoped Versioned Memory",
            "backend_acronym": "ASVM",
            "backend_id": (
                "asvm_vector_v1"
                if semantic_backend == "dense_embedding_cosine"
                else "asvm_lexical_v1"
            ),
            "fully_instantiated": True,
            "episode_id": self.runtime.episode_id,
            "returning_principal_id": self.returning_principal_id,
            "semantic_backend": semantic_backend,
            "embedding_model_id": self.runtime.embedding_model_id,
            "source_checkpoint_sha256": self._source_checkpoint_sha256,
            "same_source_checkpoint_for_all_arms": True,
            "direct_candidate_prompt_injection": False,
            "write_audit": list(self._write_audit),
            "return_views": {
                arm: view.record()
                for arm, view in sorted(self._return_views.items())
            },
            "runtime": self.runtime.record(
                include_private_experience=False
            ),
        }


__all__ = [
    "AgentScopedVersionedMemoryBackend",
    "AsvmReturnView",
]
