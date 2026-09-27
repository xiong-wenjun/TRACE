"""Lifecycle implementation for Agent RETURN experiments.

The paper-facing topology is the five-task-agent contract from
``unified_mas_contract``. Router, Evaluator, and historical Aggregator
namespaces below are internal coordination/audit services, not task agents.
The benchmark adapter owns tools, prompts, and scoring. This module owns the
lifecycle and principal-isolation machinery:

* agent1 completes material prefix work before it departs;
* agent2..agent5 complete distinct work while agent1 is absent;
* one immutable post-absence checkpoint feeds every counterfactual arm;
* RETURN happens after the checkpoint and before downstream inference;
* every principal has an isolated private-memory namespace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, Sequence

from .eight_agent_pipeline import (
    AGGREGATOR_ID,
    ALL_AGENT_IDS,
    EVALUATOR_ID,
    ROUTER_ID,
    WORKER_IDS,
    validate_eight_agent_contract,
)
from .private_episodic_memory import (
    PrincipalMemoryRegistry,
    PrivateEpisodicMemory,
    PrivateMemoryItem,
    PrivateMemoryRetrieval,
)
from .return_governance import ReturnMemoryGovernor
from .router_return_governance import (
    RouterTraceCompilation,
    compile_router_trace,
)
from .router_return_protocol import (
    RouterPostAbsenceCheckpoint,
    RouterReturnArm,
    RouterReturnProtocol,
    RouterWorkstateItem,
    canonical_sha256,
)
from .router_return_scenarios import (
    AbsenceReview,
    HorizonRouterScenarioIds,
    MultiWorkerRouterScenarioIds,
    build_five_worker_router_return_scenario,
    build_horizon_router_return_scenario,
    record_horizon_absence_work,
    record_multiworker_absence_work,
)
from .unified_mas_contract import (
    ACTIVE_AGENT_IDS as UNIFIED_ACTIVE_AGENT_IDS,
    RETURNING_AGENT_ID as UNIFIED_RETURNING_AGENT_ID,
    TASK_AGENT_IDS as UNIFIED_TASK_AGENT_IDS,
    unified_mas_architecture_record,
)


RETURNING_WORKER_ID = WORKER_IDS[0]
ABSENCE_WORKER_IDS = WORKER_IDS[1:]
MAS_RETURN_EXECUTION_CONTRACT = (
    "router_5worker_aggregator_evaluator_return_v1"
)
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_EMBEDDING_MAX_CHARS = 24_000


class MemoryEmbeddingProvider(Protocol):
    """Minimal embedding surface required by private long-term memory."""

    model_id: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def eight_agent_architecture_record(
    *, absence_parallel_seconds: float | None = None
) -> dict[str, object]:
    """Return the legacy internal-service receipt, not paper topology."""

    record: dict[str, object] = {
        "execution_contract": MAS_RETURN_EXECUTION_CONTRACT,
        "router_count": 1,
        "worker_count": 5,
        "active_worker_count": 5,
        "absence_worker_count": 4,
        "aggregator_count": 1,
        "evaluator_count": 1,
        "total_agent_count": 8,
        "all_agent_ids": list(ALL_AGENT_IDS),
        "returning_worker_id": RETURNING_WORKER_ID,
        "absence_worker_ids": list(ABSENCE_WORKER_IDS),
        "parallel_absence_execution": False,
        "intra_episode_execution_mode": "deterministic_serial",
        "utility_learning_enabled": False,
        "value_learner": None,
        "ground_truth_reader": EVALUATOR_ID,
        "workers_generate_final_answer": False,
        "aggregator_may_access_ground_truth": False,
    }
    if absence_parallel_seconds is not None:
        record["absence_parallel_seconds"] = float(
            absence_parallel_seconds
        )
    return record


def unified_five_task_agent_architecture_record() -> dict[str, object]:
    """Return the paper-facing topology shared by all Return benchmarks."""

    if tuple(WORKER_IDS) != tuple(UNIFIED_TASK_AGENT_IDS):
        raise RuntimeError("worker roster drifted from the unified MAS contract")
    if RETURNING_WORKER_ID != UNIFIED_RETURNING_AGENT_ID:
        raise RuntimeError("returning Worker drifted from the unified contract")
    if tuple(ABSENCE_WORKER_IDS) != tuple(UNIFIED_ACTIVE_AGENT_IDS):
        raise RuntimeError("absence Worker roster drifted from unified contract")
    return unified_mas_architecture_record()


def _required(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


@dataclass(frozen=True)
class FiveWorkerMasDecomposition:
    """One fan-out/fan-in task plan with fixed RETURN role identities."""

    prefix_assignment: str
    absence_assignments: Mapping[str, str]
    synthesis_assignment: str
    aggregator_assignment: str
    evaluator_assignment: str
    fallback_used: bool = False

    def __post_init__(self) -> None:
        _required(self.prefix_assignment, "prefix_assignment")
        _required(self.synthesis_assignment, "synthesis_assignment")
        _required(self.aggregator_assignment, "aggregator_assignment")
        _required(self.evaluator_assignment, "evaluator_assignment")
        if set(self.absence_assignments) != set(ABSENCE_WORKER_IDS):
            raise ValueError(
                "absence assignments must cover agent2..agent5 exactly once"
            )
        for worker_id, assignment in self.absence_assignments.items():
            _required(assignment, f"absence assignment for {worker_id}")

    def record(self) -> dict[str, object]:
        return {
            "schema_version": "five_worker_mas_decomposition_v1",
            "prefix_worker_id": RETURNING_WORKER_ID,
            "prefix_assignment": self.prefix_assignment,
            "absence_assignments": {
                worker_id: self.absence_assignments[worker_id]
                for worker_id in ABSENCE_WORKER_IDS
            },
            "synthesis_assignment": self.synthesis_assignment,
            "aggregator_assignment": self.aggregator_assignment,
            "evaluator_assignment": self.evaluator_assignment,
            "fallback_used": self.fallback_used,
        }


def parse_five_worker_decomposition(
    value: Mapping[str, object],
    *,
    fallback: FiveWorkerMasDecomposition,
) -> FiveWorkerMasDecomposition:
    """Parse a Router plan while rejecting answer/workflow oracle fields."""

    forbidden = {
        "answer",
        "gold",
        "reference_answer",
        "workflow",
        "action_sequence",
        "expert_plan",
    }.intersection(str(key) for key in value)
    if forbidden:
        raise ValueError(
            "Router decomposition contains oracle fields: "
            + ",".join(sorted(forbidden))
        )
    raw_absence = value.get("absence_assignments")
    absence: dict[str, str] = {}
    if isinstance(raw_absence, Mapping):
        absence = {
            str(worker_id): str(assignment or "").strip()
            for worker_id, assignment in raw_absence.items()
        }
    elif isinstance(raw_absence, Sequence) and not isinstance(
        raw_absence, (str, bytes)
    ):
        for row in raw_absence:
            if not isinstance(row, Mapping):
                continue
            absence[str(row.get("worker_id") or "")] = str(
                row.get("assignment") or ""
            ).strip()

    fallback_used = False

    def assignment_text(
        raw: object,
        *,
        default: str,
        preferred_key: str | None = None,
    ) -> str:
        nonlocal fallback_used
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        if isinstance(raw, Mapping):
            if preferred_key is not None:
                preferred = str(raw.get(preferred_key) or "").strip()
                if preferred:
                    return preferred
            values = [
                str(item or "").strip()
                for item in raw.values()
                if str(item or "").strip()
            ]
            if len(values) == 1:
                return values[0]
        fallback_used = True
        return default

    try:
        return FiveWorkerMasDecomposition(
            prefix_assignment=assignment_text(
                value.get("prefix_assignment"),
                default=fallback.prefix_assignment,
                preferred_key=RETURNING_WORKER_ID,
            ),
            absence_assignments=absence,
            synthesis_assignment=assignment_text(
                value.get("synthesis_assignment"),
                default=fallback.synthesis_assignment,
                preferred_key=RETURNING_WORKER_ID,
            ),
            aggregator_assignment=assignment_text(
                value.get("aggregator_assignment"),
                default=fallback.aggregator_assignment,
                preferred_key=AGGREGATOR_ID,
            ),
            evaluator_assignment=assignment_text(
                value.get("evaluator_assignment"),
                default=fallback.evaluator_assignment,
                preferred_key=EVALUATOR_ID,
            ),
            fallback_used=fallback_used,
        )
    except (TypeError, ValueError):
        return FiveWorkerMasDecomposition(
            prefix_assignment=fallback.prefix_assignment,
            absence_assignments=dict(fallback.absence_assignments),
            synthesis_assignment=fallback.synthesis_assignment,
            aggregator_assignment=fallback.aggregator_assignment,
            evaluator_assignment=fallback.evaluator_assignment,
            fallback_used=True,
        )


def build_private_memory_registry(
    worker_principal_ids: Sequence[str],
    *,
    instance_namespace: str = "initial",
) -> PrincipalMemoryRegistry:
    """Create isolated memory banks for one dynamic worker roster."""

    namespace = _required(instance_namespace, "instance_namespace")
    worker_ids = tuple(
        _required(principal_id, "worker principal_id")
        for principal_id in worker_principal_ids
    )
    if not worker_ids:
        raise ValueError("worker_principal_ids must be non-empty")
    if len(worker_ids) != len(set(worker_ids)):
        raise ValueError("worker_principal_ids must be unique")
    reserved_ids = {ROUTER_ID, AGGREGATOR_ID, EVALUATOR_ID}
    if reserved_ids.intersection(worker_ids):
        raise ValueError("worker_principal_ids contain a reserved principal")
    registry = PrincipalMemoryRegistry()
    principal_ids = (
        ROUTER_ID,
        *worker_ids,
        AGGREGATOR_ID,
        EVALUATOR_ID,
    )
    for principal_id in principal_ids:
        if principal_id == ROUTER_ID:
            role_id = "role:router"
        elif principal_id == AGGREGATOR_ID:
            role_id = "role:aggregator"
        elif principal_id == EVALUATOR_ID:
            role_id = "role:evaluator"
        else:
            role_id = f"role:{principal_id}"
        registry.register(
            principal_id=principal_id,
            instance_id=f"{principal_id}:{namespace}",
            role_id=role_id,
        )
    return registry


def build_eight_agent_private_memory_registry(
    *, instance_namespace: str = "initial"
) -> PrincipalMemoryRegistry:
    """Create exactly eight principal-isolated memory banks."""

    validate_eight_agent_contract()
    return build_private_memory_registry(
        WORKER_IDS,
        instance_namespace=instance_namespace,
    )


@dataclass(frozen=True)
class AgentMemoryArm:
    """Private-memory branches for one paired RETURN treatment arm."""

    arm: str
    returning_worker: PrivateEpisodicMemory
    aggregator: PrivateEpisodicMemory
    evaluator: PrivateEpisodicMemory

    def record(
        self, *, include_private_experience: bool = True
    ) -> dict[str, object]:
        return {
            "arm": self.arm,
            "returning_worker": self.returning_worker.record(
                include_experience=include_private_experience
            ),
            "aggregator": self.aggregator.record(
                include_experience=include_private_experience
            ),
            "evaluator": self.evaluator.record(
                include_experience=include_private_experience
            ),
        }

    @classmethod
    def from_record(
        cls, value: Mapping[str, object]
    ) -> "AgentMemoryArm":
        banks: dict[str, PrivateEpisodicMemory] = {}
        for key in ("returning_worker", "aggregator", "evaluator"):
            raw_bank = value.get(key)
            if not isinstance(raw_bank, Mapping):
                raise ValueError(f"memory arm requires {key} bank")
            banks[key] = PrivateEpisodicMemory.from_record(raw_bank)
        if banks["returning_worker"].owner_principal_id != RETURNING_WORKER_ID:
            raise ValueError("memory arm returning Worker principal mismatch")
        if banks["aggregator"].owner_principal_id != AGGREGATOR_ID:
            raise ValueError("memory arm Aggregator principal mismatch")
        if banks["evaluator"].owner_principal_id != EVALUATOR_ID:
            raise ValueError("memory arm Evaluator principal mismatch")
        return cls(
            arm=_required(value.get("arm"), "arm"),
            returning_worker=banks["returning_worker"],
            aggregator=banks["aggregator"],
            evaluator=banks["evaluator"],
        )


@dataclass
class EpisodeLongTermMemory:
    """Principal-scoped memory that persists for one longitudinal episode."""

    episode_id: str
    registry: PrincipalMemoryRegistry
    embedding_provider: MemoryEmbeddingProvider | None = field(
        default=None,
        repr=False,
    )
    embedding_model_id: str | None = None
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    embedding_max_chars: int = DEFAULT_EMBEDDING_MAX_CHARS
    events: list[dict[str, object]] = field(default_factory=list)
    branches: dict[str, AgentMemoryArm] = field(default_factory=dict)
    _embedding_cache: dict[str, tuple[float, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be positive")
        if self.embedding_max_chars < 1:
            raise ValueError("embedding_max_chars must be positive")
        if self.embedding_provider is not None:
            provider_model = _required(
                self.embedding_provider.model_id,
                "embedding provider model_id",
            )
            if (
                self.embedding_model_id is not None
                and self.embedding_model_id != provider_model
            ):
                raise ValueError("embedding provider/model mismatch")
            self.embedding_model_id = provider_model

    @classmethod
    def build(
        cls,
        episode_id: str,
        *,
        worker_principal_ids: Sequence[str] | None = None,
        embedding_provider: MemoryEmbeddingProvider | None = None,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        embedding_max_chars: int = DEFAULT_EMBEDDING_MAX_CHARS,
    ) -> "EpisodeLongTermMemory":
        identifier = _required(episode_id, "episode_id")
        if worker_principal_ids is None:
            registry = build_eight_agent_private_memory_registry(
                instance_namespace=f"{identifier}:epoch1"
            )
        else:
            registry = build_private_memory_registry(
                worker_principal_ids,
                instance_namespace=f"{identifier}:epoch1",
            )
        return cls(
            episode_id=identifier,
            registry=registry,
            embedding_provider=embedding_provider,
            embedding_dimensions=embedding_dimensions,
            embedding_max_chars=embedding_max_chars,
        )

    def _embedding_text(self, intent: str, experience: str) -> str:
        prefix = f"Intent:\n{intent}\n\nExperience:\n"
        available = max(0, self.embedding_max_chars - len(prefix))
        return prefix + experience[:available]

    def _embed(self, text: str) -> tuple[float, ...]:
        if self.embedding_provider is None:
            if self.embedding_model_id is not None:
                raise RuntimeError(
                    "embedding-backed memory was restored without attaching "
                    "its embedding provider"
                )
            return ()
        cache_key = canonical_sha256(
            {
                "model": self.embedding_model_id,
                "dimensions": self.embedding_dimensions,
                "text": text,
            }
        )
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        rows = self.embedding_provider.embed((text,))
        if len(rows) != 1 or not rows[0]:
            raise RuntimeError("embedding provider returned no vector")
        vector = tuple(float(value) for value in rows[0])
        if any(not math.isfinite(value) for value in vector):
            raise RuntimeError("embedding provider returned non-finite values")
        if len(vector) < self.embedding_dimensions:
            raise RuntimeError(
                "embedding vector is smaller than configured dimensions"
            )
        vector = vector[: self.embedding_dimensions]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            raise RuntimeError("embedding provider returned a zero vector")
        normalized = tuple(value / norm for value in vector)
        self._embedding_cache[cache_key] = normalized
        return normalized

    def bank(self, principal_id: str) -> PrivateEpisodicMemory:
        return self.registry.owned(
            principal_id,
            requesting_principal_id=principal_id,
        )

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
        target = bank or self.bank(principal_id)
        if target.owner_principal_id != principal_id:
            raise PermissionError("memory recall principal mismatch")
        query_embedding = self._embed(
            self._embedding_text(intent, "")
        )
        retrieval = target.retrieve(
            requesting_principal_id=principal_id,
            intent=intent,
            k1=k1,
            k2=k2,
            similarity_threshold=similarity_threshold,
            utility_weight=0.0,
            query_embedding=query_embedding,
            embedding_model_id=(
                self.embedding_model_id if query_embedding else None
            ),
        )
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "event": "private_memory_recalled",
                "principal_id": principal_id,
                "instance_id": target.owner_instance_id,
                "branch": branch,
                "intent_sha256": canonical_sha256(intent),
                "admissible_memory_ids": list(
                    retrieval.admissible_item_ids
                ),
                "recalled_memory_ids": list(retrieval.recalled_item_ids),
                "selected_memory_ids": [
                    item.memory_id for item in retrieval.selected
                ],
                "similarity_backend": retrieval.similarity_backend,
                "embedding_model_id": self.embedding_model_id,
                "embedding_dimensions": (
                    self.embedding_dimensions
                    if query_embedding
                    else None
                ),
                "utility_learning_enabled": False,
            }
        )
        return retrieval

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
        if target.owner_principal_id != principal_id:
            raise PermissionError("memory write principal mismatch")
        embedding = self._embed(
            self._embedding_text(intent, experience)
        )
        item = target.remember(
            requesting_principal_id=principal_id,
            task_id=task_id,
            intent=intent,
            experience=experience,
            embedding=embedding,
            embedding_model_id=(
                self.embedding_model_id if embedding else None
            ),
            initial_utility=0.5,
            obligation_ids=obligation_ids,
            dependency_ids=dependency_ids,
            source_workstate_id=source_workstate_id,
            attributes={
                **dict(attributes or {}),
                "utility_learning_enabled": False,
            },
        )
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "event": "private_memory_written",
                "principal_id": principal_id,
                "instance_id": target.owner_instance_id,
                "branch": branch,
                "memory_id": item.memory_id,
                "task_id": task_id,
                "source_workstate_id": source_workstate_id,
                "provenance_sha256": item.provenance_sha256,
                "embedding_model_id": item.embedding_model_id,
                "embedding_dimensions": len(item.embedding),
                "utility_learning_enabled": False,
            }
        )
        return item

    def quarantine_principal(
        self,
        principal_id: str,
        *,
        bank: PrivateEpisodicMemory | None = None,
        branch: str | None = None,
    ) -> str:
        principal = _required(principal_id, "principal_id")
        target = bank or self.bank(principal)
        if target.owner_principal_id != principal:
            raise PermissionError("memory quarantine principal mismatch")
        digest = target.quarantine_for_departure(
            requesting_principal_id=principal
        )
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "event": "private_memory_quarantined",
                "principal_id": principal,
                "instance_id": target.owner_instance_id,
                "branch": branch,
                "snapshot_sha256": digest,
                "memory_ids": sorted(target.items),
            }
        )
        return digest

    def quarantine_returning_worker(self) -> str:
        return self.quarantine_principal(RETURNING_WORKER_ID)

    def fork_principal_return(
        self,
        *,
        principal_id: str,
        branch_id: str,
        admitted_workstate_ids: Sequence[str],
        source_bank: PrivateEpisodicMemory | None = None,
    ) -> PrivateEpisodicMemory:
        principal = _required(principal_id, "principal_id")
        branch = _required(branch_id, "branch_id")
        source = source_bank or self.bank(principal)
        if source.owner_principal_id != principal:
            raise PermissionError("memory return principal mismatch")
        admitted_sources = set(admitted_workstate_ids)
        admitted_memory_ids = tuple(
            memory_id
            for memory_id, item in sorted(source.items.items())
            if item.source_workstate_id in admitted_sources
        )
        returned = source.fork_return_instance(
            requesting_principal_id=principal,
            new_instance_id=(
                f"{principal}:{self.episode_id}:{branch}:return"
            ),
            target_epoch=source.epoch + 1,
            admitted_memory_ids=admitted_memory_ids,
        )
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "event": "private_memory_return_forked",
                "principal_id": principal,
                "branch": branch,
                "source_snapshot_sha256": source.snapshot_sha256,
                "admitted_workstate_ids": sorted(admitted_sources),
                "admitted_memory_ids": list(admitted_memory_ids),
                "rejected_memory_ids": sorted(
                    set(source.items) - set(admitted_memory_ids)
                ),
                "returned_snapshot_sha256": returned.snapshot_sha256,
            }
        )
        return returned

    def fork_return_arm(
        self,
        *,
        arm: str,
        admitted_workstate_ids: Sequence[str],
    ) -> AgentMemoryArm:
        arm_id = _required(arm, "arm")
        if arm_id in self.branches:
            raise ValueError(f"private memory arm already exists: {arm_id}")
        returning_worker = self.fork_principal_return(
            principal_id=RETURNING_WORKER_ID,
            branch_id=arm_id,
            admitted_workstate_ids=admitted_workstate_ids,
        )
        aggregator = self.bank(AGGREGATOR_ID).clone_branch(
            requesting_principal_id=AGGREGATOR_ID,
            new_instance_id=(
                f"{AGGREGATOR_ID}:{self.episode_id}:{arm_id}"
            ),
        )
        evaluator = self.bank(EVALUATOR_ID).clone_branch(
            requesting_principal_id=EVALUATOR_ID,
            new_instance_id=(
                f"{EVALUATOR_ID}:{self.episode_id}:{arm_id}"
            ),
        )
        branch = AgentMemoryArm(
            arm=arm_id,
            returning_worker=returning_worker,
            aggregator=aggregator,
            evaluator=evaluator,
        )
        self.branches[arm_id] = branch
        return branch

    def record(
        self, *, include_private_experience: bool = True
    ) -> dict[str, object]:
        return {
            "schema_version": "episode_long_term_memory_v2",
            "episode_id": self.episode_id,
            "scope": "one_longitudinal_episode",
            "cross_topic_persistence": False,
            "semantic_backend": (
                "dense_embedding_cosine"
                if self.embedding_model_id is not None
                else "token_cosine"
            ),
            "embedding_model_id": self.embedding_model_id,
            "embedding_dimensions": (
                self.embedding_dimensions
                if self.embedding_model_id is not None
                else None
            ),
            "embedding_max_chars": self.embedding_max_chars,
            "utility_learning_enabled": False,
            "registry": self.registry.record(
                include_private_experience=include_private_experience
            ),
            "branches": {
                arm: branch.record(
                    include_private_experience=include_private_experience
                )
                for arm, branch in sorted(self.branches.items())
            },
            "events": list(self.events),
        }

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, object],
        *,
        embedding_provider: MemoryEmbeddingProvider | None = None,
    ) -> "EpisodeLongTermMemory":
        raw_registry = value.get("registry")
        if not isinstance(raw_registry, Mapping):
            raise ValueError("episode memory requires a registry")
        runtime = cls(
            episode_id=_required(value.get("episode_id"), "episode_id"),
            registry=PrincipalMemoryRegistry.from_record(raw_registry),
            embedding_provider=embedding_provider,
            embedding_model_id=(
                str(value["embedding_model_id"])
                if value.get("embedding_model_id") is not None
                else None
            ),
            embedding_dimensions=int(
                value.get(
                    "embedding_dimensions",
                    DEFAULT_EMBEDDING_DIMENSIONS,
                )
                or DEFAULT_EMBEDDING_DIMENSIONS
            ),
            embedding_max_chars=int(
                value.get(
                    "embedding_max_chars",
                    DEFAULT_EMBEDDING_MAX_CHARS,
                )
            ),
        )
        raw_branches = value.get("branches") or {}
        if not isinstance(raw_branches, Mapping):
            raise ValueError("episode memory branches must be an object")
        for arm, raw_branch in raw_branches.items():
            if not isinstance(raw_branch, Mapping):
                raise ValueError("episode memory branch must be an object")
            branch = AgentMemoryArm.from_record(raw_branch)
            if branch.arm != str(arm):
                raise ValueError("episode memory branch key mismatch")
            runtime.branches[str(arm)] = branch
        raw_events = value.get("events") or ()
        if not isinstance(raw_events, Sequence) or isinstance(
            raw_events, (str, bytes)
        ):
            raise ValueError("episode memory events must be a list")
        runtime.events = [
            dict(item)
            for item in raw_events
            if isinstance(item, Mapping)
        ]
        if len(runtime.events) != len(raw_events):
            raise ValueError("episode memory event must be an object")
        return runtime


def audit_five_worker_return_divergence(
    *,
    target_cdt: float,
    ids: MultiWorkerRouterScenarioIds,
    checkpoint: RouterPostAbsenceCheckpoint,
    trace_compilation: RouterTraceCompilation,
    workstates: Mapping[str, str],
    absence_workstate_ids: Mapping[str, str],
) -> dict[str, object]:
    """Audit a four-dependency RETURN fork before arm inference.

    The audit is outcome-blind: it only checks event structure, version
    relations, inherited item identities, selector coverage, and leakage.
    """

    critical_dependencies = tuple(
        ids.effective_predeparture_dependency_ids
    ) + tuple(ids.critical_absence_dependency_ids)
    changed_dependencies = tuple(ids.critical_absence_dependency_ids)
    actual_cdt = (
        len(changed_dependencies) / len(critical_dependencies)
        if critical_dependencies
        else 0.0
    )
    fork_by_arm = {fork.arm.value: fork for fork in trace_compilation.forks}
    restore_ids = set(
        fork_by_arm[
            RouterReturnArm.RESTORE_OLD.value
        ].inherited_item_ids
    )
    trace_ids = set(
        fork_by_arm[RouterReturnArm.TRACE.value].inherited_item_ids
    )
    absence_ids = set(absence_workstate_ids.values())
    departure_ids = {
        item.item_id for item in checkpoint.departure.workstate_items
    }
    current_by_id = {
        item.item_id: item for item in checkpoint.current_workstate_items
    }
    stale_old_id = ids.old_item_id
    replacement_ids = {
        item.item_id
        for item in checkpoint.current_workstate_items
        if item.supersedes_item_id == stale_old_id
    }
    old_status = getattr(
        current_by_id.get(stale_old_id), "status", None
    )
    stale_conflict = bool(
        stale_old_id in departure_ids
        and old_status is not None
        and old_status.value == "superseded"
        and any(
            current_by_id[item_id].status.value == "active"
            for item_id in replacement_ids
        )
    )
    selection = trace_compilation.compilation.selection.record()
    recall = float(
        selection.get("critical_dependency_recall") or 0.0
    )
    view_different = (
        restore_ids != trace_ids
        and workstates[RouterReturnArm.RESTORE_OLD.value]
        != workstates[RouterReturnArm.TRACE.value]
    )
    bypass = bool(restore_ids.intersection(absence_ids))
    receipt: dict[str, object] = {
        "schema_version": "return_divergence_audit_v1",
        "target_cdt": target_cdt,
        "actual_cdt": actual_cdt,
        "cdt_error": abs(actual_cdt - target_cdt),
        "critical_dependency_total": len(critical_dependencies),
        "changed_critical_dependency_count": len(changed_dependencies),
        "critical_dependency_ids": list(critical_dependencies),
        "changed_critical_dependency_ids": list(changed_dependencies),
        "restore_trace_view_different": view_different,
        "restore_old_item_ids": sorted(restore_ids),
        "trace_item_ids": sorted(trace_ids),
        "restore_old_bypass_item_ids": sorted(
            restore_ids.intersection(absence_ids)
        ),
        "bypass_leakage_detected": bypass,
        "stale_old_item_id": stale_old_id,
        "stale_replacement_item_ids": sorted(replacement_ids),
        "stale_conflict_present": stale_conflict,
        "restore_old_inherits_stale": stale_old_id in restore_ids,
        "trace_rejects_stale": stale_old_id not in trace_ids,
        "trace_critical_dependency_recall": recall,
    }
    receipt["event_eligible"] = bool(
        receipt["cdt_error"] <= 1e-12
        and view_different
        and stale_conflict
        and not bypass
        and recall >= 0.999999
        and receipt["restore_old_inherits_stale"]
        and receipt["trace_rejects_stale"]
    )
    return receipt


@dataclass
class FiveWorkerReturnController:
    """Enforce the shared prefix → absence → checkpoint → RETURN timeline."""

    protocol: RouterReturnProtocol
    ids: MultiWorkerRouterScenarioIds
    decomposition: FiveWorkerMasDecomposition
    prefix_recorded: bool = False
    absence_workers_recorded: set[str] = field(default_factory=set)
    checkpoint: RouterPostAbsenceCheckpoint | None = None
    trace_compilation: RouterTraceCompilation | None = None

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        task_description: str,
        decomposition: FiveWorkerMasDecomposition,
        predeparture_critical_dependency_count: int = 1,
        critical_absence_principal_ids: Sequence[str] | None = None,
    ) -> "FiveWorkerReturnController":
        workers = tuple(
            (worker_id, f"role:{worker_id}") for worker_id in WORKER_IDS
        )
        protocol, ids = build_five_worker_router_return_scenario(
            task_id=task_id,
            task_description=task_description,
            workers=workers,
            returning_principal_id=RETURNING_WORKER_ID,
            prefix_description=decomposition.prefix_assignment,
            absence_descriptions=decomposition.absence_assignments,
            synthesis_description=decomposition.synthesis_assignment,
            predeparture_critical_dependency_count=(
                predeparture_critical_dependency_count
            ),
            critical_absence_principal_ids=(
                critical_absence_principal_ids
            ),
            aggregator_principal_id=AGGREGATOR_ID,
            aggregation_description=decomposition.aggregator_assignment,
            evaluator_principal_id=EVALUATOR_ID,
            evaluation_description=decomposition.evaluator_assignment,
        )
        return cls(
            protocol=protocol,
            ids=ids,
            decomposition=decomposition,
        )

    def record_prefix(self, result: str) -> RouterWorkstateItem:
        if self.prefix_recorded:
            raise RuntimeError("prefix work is already recorded")
        result = _required(result, "prefix result")
        self.protocol.start_node(
            self.ids.prefix_node_id, RETURNING_WORKER_ID
        )
        self.protocol.complete_node(
            self.ids.prefix_node_id, RETURNING_WORKER_ID, result
        )
        item = self.protocol.add_workstate_item(
            RouterWorkstateItem(
                item_id=self.ids.old_item_id,
                text=result,
                owner_principal_id=RETURNING_WORKER_ID,
                writer_principal_id=RETURNING_WORKER_ID,
                dependency_ids=(
                    self.ids.effective_predeparture_dependency_ids
                ),
                obligation_ids=(self.ids.obligation_id,),
            )
        )
        self.protocol.depart(RETURNING_WORKER_ID)
        self.prefix_recorded = True
        return item

    def record_absence(
        self, worker_id: str, review: AbsenceReview
    ) -> RouterWorkstateItem:
        if not self.prefix_recorded:
            raise RuntimeError("agent1 must depart before absence work")
        if worker_id not in ABSENCE_WORKER_IDS:
            raise ValueError("absence work must be assigned to agent2..agent5")
        if worker_id in self.absence_workers_recorded:
            raise RuntimeError(f"absence work already recorded: {worker_id}")
        item = record_multiworker_absence_work(
            self.protocol,
            self.ids,
            absence_principal_id=worker_id,
            review=review,
        )
        self.absence_workers_recorded.add(worker_id)
        return item

    def checkpoint_and_return(
        self,
        *,
        governor: ReturnMemoryGovernor,
        purpose: str,
        max_items: int,
        max_chars: int,
        expires_after_iteration: int = 2,
    ) -> tuple[RouterPostAbsenceCheckpoint, RouterTraceCompilation]:
        if self.absence_workers_recorded != set(ABSENCE_WORKER_IDS):
            missing = set(ABSENCE_WORKER_IDS) - self.absence_workers_recorded
            raise RuntimeError(
                "cannot freeze before all absence Workers complete: "
                + ",".join(sorted(missing))
            )
        checkpoint = self.protocol.freeze_post_absence_checkpoint(
            RETURNING_WORKER_ID
        )
        trace_compilation = compile_router_trace(
            self.protocol,
            checkpoint,
            governor,
            purpose=purpose,
            max_items=max_items,
            max_chars=max_chars,
            expires_after_iteration=expires_after_iteration,
        )
        self.protocol.start_return(RETURNING_WORKER_ID)
        self.protocol.finish_return(
            RETURNING_WORKER_ID,
            admitted=True,
            reason=(
                "fresh agent1 instance admitted after the immutable "
                "post-absence checkpoint; arm policy controls inherited state"
            ),
        )
        self.checkpoint = checkpoint
        self.trace_compilation = trace_compilation
        return checkpoint, trace_compilation

    def timeline_record(self) -> dict[str, object]:
        ledger = self.protocol.record()["ledger"]
        event_trace = [
            row
            for row in ledger
            if row["event"]
            in {
                "member_departed",
                "checkpoint_frozen",
                "return_started",
                "return_admitted",
            }
        ]
        return {
            "schema_version": "five_worker_mas_return_timeline_v1",
            "returning_worker_id": RETURNING_WORKER_ID,
            "absence_worker_ids": list(ABSENCE_WORKER_IDS),
            "events": event_trace,
            "prefix_recorded": self.prefix_recorded,
            "absence_workers_recorded": sorted(
                self.absence_workers_recorded
            ),
            "return_divergence_contract": {
                "critical_dependency_total": (
                    len(
                        self.ids.effective_predeparture_dependency_ids
                    )
                    + len(
                        self.ids.critical_absence_dependency_ids
                    )
                ),
                "changed_critical_dependency_count": len(
                    self.ids.critical_absence_dependency_ids
                ),
                "actual_cdt": (
                    len(self.ids.critical_absence_dependency_ids)
                    / (
                        len(
                            self.ids.effective_predeparture_dependency_ids
                        )
                        + len(
                            self.ids.critical_absence_dependency_ids
                        )
                    )
                ),
            },
            "checkpoint_sha256": (
                self.checkpoint.checkpoint_sha256
                if self.checkpoint is not None
                else None
            ),
        }


@dataclass
class HorizonReturnController:
    """Enforce prefix → H sequential absence tasks → checkpoint → RETURN."""

    protocol: RouterReturnProtocol
    ids: HorizonRouterScenarioIds
    decomposition: FiveWorkerMasDecomposition
    prefix_recorded: bool = False
    absence_tasks_recorded: set[int] = field(default_factory=set)
    checkpoint: RouterPostAbsenceCheckpoint | None = None
    trace_compilation: RouterTraceCompilation | None = None
    controlled_divergence: bool = False
    forced_stale_conflict: bool = False
    stale_old_item_id: str | None = None
    stale_replacement_item_id: str | None = None

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        task_description: str,
        decomposition: FiveWorkerMasDecomposition,
        absence_horizon_tasks: int,
        critical_absence_tasks: int = 3,
        critical_dependency_total: int | None = None,
        predeparture_critical_dependencies: int = 1,
        force_stale_conflict: bool = False,
    ) -> "HorizonReturnController":
        if absence_horizon_tasks < 1:
            raise ValueError("absence_horizon_tasks must be positive")
        if predeparture_critical_dependencies < 1:
            raise ValueError(
                "predeparture_critical_dependencies must be positive"
            )
        if (
            critical_dependency_total is not None
            and predeparture_critical_dependencies != 1
        ):
            raise ValueError(
                "legacy critical_dependency_total cannot be combined with "
                "predeparture_critical_dependencies"
            )
        workers = tuple(
            (worker_id, f"role:{worker_id}") for worker_id in WORKER_IDS
        )
        schedule = tuple(
            (
                ABSENCE_WORKER_IDS[index % len(ABSENCE_WORKER_IDS)],
                (
                    decomposition.absence_assignments[
                        ABSENCE_WORKER_IDS[index % len(ABSENCE_WORKER_IDS)]
                    ]
                    + f"\nAbsence task {index + 1}/"
                    f"{absence_horizon_tasks}; process the next team-state "
                    "update and preserve version relations."
                ),
            )
            for index in range(absence_horizon_tasks)
        )
        critical_count = min(
            max(1, critical_absence_tasks), absence_horizon_tasks
        )
        if (
            critical_dependency_total is not None
            and critical_dependency_total <= critical_count
        ):
            raise ValueError(
                "critical_dependency_total must leave at least one "
                "valid predeparture dependency"
            )
        predeparture_critical_count = (
            critical_dependency_total - critical_count
            if critical_dependency_total is not None
            else predeparture_critical_dependencies
        )
        critical_indices = tuple(
            range(absence_horizon_tasks - critical_count, absence_horizon_tasks)
        )
        protocol, ids = build_horizon_router_return_scenario(
            task_id=task_id,
            task_description=task_description,
            workers=workers,
            returning_principal_id=RETURNING_WORKER_ID,
            prefix_description=decomposition.prefix_assignment,
            absence_tasks=schedule,
            synthesis_description=decomposition.synthesis_assignment,
            critical_absence_task_indices=critical_indices,
            predeparture_critical_dependency_count=(
                predeparture_critical_count
            ),
            aggregator_principal_id=AGGREGATOR_ID,
            aggregation_description=decomposition.aggregator_assignment,
            evaluator_principal_id=EVALUATOR_ID,
            evaluation_description=decomposition.evaluator_assignment,
        )
        return cls(
            protocol=protocol,
            ids=ids,
            decomposition=decomposition,
            controlled_divergence=(
                critical_dependency_total is not None
            ),
            forced_stale_conflict=force_stale_conflict,
        )

    def record_prefix(self, result: str) -> RouterWorkstateItem:
        return self.record_prefix_items((result,))[0]

    def record_prefix_items(
        self, results: Sequence[str]
    ) -> tuple[RouterWorkstateItem, ...]:
        """Record atomic pre-departure items under distinct dependencies.

        Long-horizon benchmarks may have several independently necessary
        prefix results.  Combining them into one text blob can violate the
        governance layer's atomic-disclosure limit and hides dependency-level
        coverage.  This method preserves one item and dependency per result.
        """

        if self.prefix_recorded:
            raise RuntimeError("prefix work is already recorded")
        normalized = tuple(
            _required(result, "prefix result") for result in results
        )
        dependency_ids = self.ids.effective_predeparture_dependency_ids
        if len(normalized) == 1:
            dependency_groups = (dependency_ids,)
        elif len(normalized) == len(dependency_ids):
            dependency_groups = tuple(
                (dependency_id,) for dependency_id in dependency_ids
            )
        else:
            raise ValueError(
                "prefix item count must match predeparture critical "
                "dependency count"
            )
        self.protocol.start_node(
            self.ids.prefix_node_id, RETURNING_WORKER_ID
        )
        self.protocol.complete_node(
            self.ids.prefix_node_id,
            RETURNING_WORKER_ID,
            "\n".join(
                f"prefix_item_{index + 1}_sha256="
                f"{canonical_sha256(result)}"
                for index, result in enumerate(normalized)
            ),
        )
        items = tuple(
            self.protocol.add_workstate_item(
                RouterWorkstateItem(
                    item_id=(
                        self.ids.old_item_id
                        if index == 0
                        else f"{self.ids.old_item_id}:part-{index + 1}"
                    ),
                    text=result,
                    owner_principal_id=RETURNING_WORKER_ID,
                    writer_principal_id=RETURNING_WORKER_ID,
                    dependency_ids=dependency_groups[index],
                    obligation_ids=(self.ids.obligation_id,),
                )
            )
            for index, result in enumerate(normalized)
        )
        if self.forced_stale_conflict:
            if len(normalized) != 1:
                raise ValueError(
                    "forced stale conflict requires one prefix item"
                )
            self.stale_old_item_id = (
                f"{self.ids.old_item_id}:stale-candidate"
            )
            self.protocol.add_workstate_item(
                RouterWorkstateItem(
                    item_id=self.stale_old_item_id,
                    text=(
                        "Pre-departure candidate pending revalidation:\n"
                        + normalized[0]
                    ),
                    owner_principal_id=RETURNING_WORKER_ID,
                    writer_principal_id=RETURNING_WORKER_ID,
                    dependency_ids=(
                        f"{self.ids.prefix_dependency_id}:stale-candidate",
                    ),
                    obligation_ids=(self.ids.obligation_id,),
                )
            )
        self.protocol.depart(RETURNING_WORKER_ID)
        self.prefix_recorded = True
        return items

    def record_absence_task(
        self,
        task_index: int,
        worker_id: str,
        review: AbsenceReview,
    ) -> RouterWorkstateItem:
        if not self.prefix_recorded:
            raise RuntimeError("agent1 must depart before absence work")
        if task_index in self.absence_tasks_recorded:
            raise RuntimeError(
                f"absence task already recorded: {task_index}"
            )
        protocol_review = (
            AbsenceReview(
                relation="retain",
                current_fact=review.current_fact,
                audit_note=(
                    review.audit_note
                    + " Controlled CDT calibration keeps required "
                    "predeparture dependencies valid; stale-state testing "
                    "uses a separate deterministic version chain."
                ).strip(),
            )
            if self.controlled_divergence
            else review
        )
        item = record_horizon_absence_work(
            self.protocol,
            self.ids,
            task_index=task_index,
            absence_principal_id=worker_id,
            review=protocol_review,
        )
        if (
            task_index == 0
            and self.forced_stale_conflict
            and self.stale_old_item_id is not None
        ):
            self.stale_replacement_item_id = (
                f"{self.stale_old_item_id}:current"
            )
            self.protocol.supersede_workstate_item(
                actor_principal_id=worker_id,
                old_item_id=self.stale_old_item_id,
                replacement_item=RouterWorkstateItem(
                    item_id=self.stale_replacement_item_id,
                    text=(
                        "Current absence-period replacement:\n"
                        + review.current_fact
                    ),
                    owner_principal_id=RETURNING_WORKER_ID,
                    writer_principal_id=worker_id,
                    dependency_ids=(
                        f"{self.ids.prefix_dependency_id}:stale-candidate",
                    ),
                    obligation_ids=(self.ids.obligation_id,),
                    version=2,
                    supersedes_item_id=self.stale_old_item_id,
                ),
            )
        self.absence_tasks_recorded.add(task_index)
        return item

    def checkpoint_and_return(
        self,
        *,
        governor: ReturnMemoryGovernor,
        purpose: str,
        max_items: int,
        max_chars: int,
        expires_after_iteration: int = 2,
    ) -> tuple[RouterPostAbsenceCheckpoint, RouterTraceCompilation]:
        expected = set(range(len(self.ids.absence_node_ids)))
        if self.absence_tasks_recorded != expected:
            missing = expected - self.absence_tasks_recorded
            raise RuntimeError(
                "cannot freeze before all absence tasks complete: "
                + ",".join(str(index) for index in sorted(missing))
            )
        checkpoint = self.protocol.freeze_post_absence_checkpoint(
            RETURNING_WORKER_ID
        )
        trace_compilation = compile_router_trace(
            self.protocol,
            checkpoint,
            governor,
            purpose=purpose,
            max_items=max_items,
            max_chars=max_chars,
            expires_after_iteration=expires_after_iteration,
        )
        self.protocol.start_return(RETURNING_WORKER_ID)
        self.protocol.finish_return(
            RETURNING_WORKER_ID,
            admitted=True,
            reason=(
                "fresh agent1 instance admitted after the immutable "
                "H-task post-absence checkpoint"
            ),
        )
        self.checkpoint = checkpoint
        self.trace_compilation = trace_compilation
        return checkpoint, trace_compilation

    def timeline_record(self) -> dict[str, object]:
        ledger = self.protocol.record()["ledger"]
        event_trace = [
            row
            for row in ledger
            if row["event"]
            in {
                "member_departed",
                "checkpoint_frozen",
                "return_started",
                "return_admitted",
            }
        ]
        return {
            "schema_version": "horizon_mas_return_timeline_v1",
            "returning_worker_id": RETURNING_WORKER_ID,
            "absence_horizon_tasks": len(self.ids.absence_node_ids),
            "absence_task_ledger_writer_schedule": list(
                self.ids.absence_principal_ids
            ),
            "critical_absence_task_indices": list(
                self.ids.critical_absence_task_indices
            ),
            "return_divergence": {
                "critical_dependency_total": len(
                    self.ids.effective_predeparture_dependency_ids
                )
                + len(self.ids.critical_absence_task_indices),
                "valid_predeparture_dependency_count": len(
                    self.ids.effective_predeparture_dependency_ids
                ),
                "absence_critical_dependency_count": len(
                    self.ids.critical_absence_task_indices
                ),
                "actual_cdt": (
                    len(self.ids.critical_absence_task_indices)
                    / (
                        len(
                            self.ids.effective_predeparture_dependency_ids
                        )
                        + len(self.ids.critical_absence_task_indices)
                    )
                ),
                "forced_stale_conflict": self.forced_stale_conflict,
                "stale_old_item_id": self.stale_old_item_id,
                "stale_replacement_item_id": (
                    self.stale_replacement_item_id
                ),
            },
            "events": event_trace,
            "prefix_recorded": self.prefix_recorded,
            "absence_tasks_recorded": sorted(
                self.absence_tasks_recorded
            ),
            "checkpoint_sha256": (
                self.checkpoint.checkpoint_sha256
                if self.checkpoint is not None
                else None
            ),
        }


__all__ = [
    "ABSENCE_WORKER_IDS",
    "AgentMemoryArm",
    "EpisodeLongTermMemory",
    "FiveWorkerMasDecomposition",
    "FiveWorkerReturnController",
    "HorizonReturnController",
    "MAS_RETURN_EXECUTION_CONTRACT",
    "RETURNING_WORKER_ID",
    "build_eight_agent_private_memory_registry",
    "eight_agent_architecture_record",
    "unified_five_task_agent_architecture_record",
    "parse_five_worker_decomposition",
]
