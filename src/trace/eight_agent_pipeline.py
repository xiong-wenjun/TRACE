"""Role-separated Router/Worker/Aggregator/Evaluator execution contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .private_episodic_memory import (
    PrivateEpisodicMemory,
    PrivateMemoryRetrieval,
    canonical_sha256,
)


ROUTER_ID = "router"
WORKER_IDS = ("agent1", "agent2", "agent3", "agent4", "agent5")
AGGREGATOR_ID = "aggregator"
EVALUATOR_ID = "evaluator"
ALL_AGENT_IDS = (ROUTER_ID,) + WORKER_IDS + (
    AGGREGATOR_ID,
    EVALUATOR_ID,
)


@dataclass(frozen=True)
class AgentRoleContract:
    principal_id: str
    role: str
    private_memory_kind: str
    learns_value: bool
    may_access_ground_truth: bool
    may_generate_final_answer: bool

    def record(self) -> dict[str, object]:
        return {
            "principal_id": self.principal_id,
            "role": self.role,
            "private_memory_kind": self.private_memory_kind,
            "learns_value": self.learns_value,
            "may_access_ground_truth": self.may_access_ground_truth,
            "may_generate_final_answer": self.may_generate_final_answer,
        }


EIGHT_AGENT_CONTRACT = (
    AgentRoleContract(
        principal_id=ROUTER_ID,
        role="router",
        private_memory_kind="coordination_episodic_memory",
        learns_value=False,
        may_access_ground_truth=False,
        may_generate_final_answer=False,
    ),
    *tuple(
        AgentRoleContract(
            principal_id=worker_id,
            role="worker",
            private_memory_kind="non_valued_expert_episodic_memory",
            learns_value=False,
            may_access_ground_truth=False,
            may_generate_final_answer=False,
        )
        for worker_id in WORKER_IDS
    ),
    AgentRoleContract(
        principal_id=AGGREGATOR_ID,
        role="aggregator",
        private_memory_kind="non_valued_aggregation_episodic_memory",
        learns_value=False,
        may_access_ground_truth=False,
        may_generate_final_answer=True,
    ),
    AgentRoleContract(
        principal_id=EVALUATOR_ID,
        role="evaluator",
        private_memory_kind="private_evaluation_audit_log",
        learns_value=False,
        may_access_ground_truth=True,
        may_generate_final_answer=False,
    ),
)


def validate_eight_agent_contract() -> None:
    if len(EIGHT_AGENT_CONTRACT) != 8:
        raise RuntimeError("the execution contract must contain eight agents")
    if {contract.principal_id for contract in EIGHT_AGENT_CONTRACT} != set(
        ALL_AGENT_IDS
    ):
        raise RuntimeError("the eight-agent identities are inconsistent")
    learners = [
        contract.principal_id
        for contract in EIGHT_AGENT_CONTRACT
        if contract.learns_value
    ]
    if learners:
        raise RuntimeError("utility learning is disabled for this study")
    ground_truth_readers = [
        contract.principal_id
        for contract in EIGHT_AGENT_CONTRACT
        if contract.may_access_ground_truth
    ]
    if ground_truth_readers != [EVALUATOR_ID]:
        raise RuntimeError("only the Evaluator may access ground truth")


@dataclass(frozen=True)
class AggregationRequest:
    task_id: str
    arm: str
    task_description: str
    output_format: str
    router_workstate: str
    returning_worker_report: str


@dataclass(frozen=True)
class AggregationReceipt:
    task_id: str
    arm: str
    aggregator_principal_id: str
    answer: str
    answer_sha256: str
    prompt_sha256: str
    memory_retrieval: PrivateMemoryRetrieval
    stored_memory_id: str
    ground_truth_visible: bool = False

    def record(self, *, include_answer: bool = True) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "arm": self.arm,
            "aggregator_principal_id": self.aggregator_principal_id,
            "answer": self.answer if include_answer else "[redacted]",
            "answer_sha256": self.answer_sha256,
            "prompt_sha256": self.prompt_sha256,
            "selected_private_memory_ids": [
                hit.memory_id for hit in self.memory_retrieval.selected
            ],
            "stored_memory_id": self.stored_memory_id,
            "ground_truth_visible": self.ground_truth_visible,
        }


class AggregatorAgent:
    """Generate the final answer without access to ground truth."""

    def __init__(
        self,
        memory: PrivateEpisodicMemory,
        *,
        profile: str,
    ) -> None:
        if memory.owner_principal_id != AGGREGATOR_ID:
            raise ValueError("Aggregator memory principal mismatch")
        self.memory = memory
        self.profile = profile

    def aggregate(
        self,
        request: AggregationRequest,
        *,
        generate: Callable[[str], str],
        memory_config: Mapping[str, object],
    ) -> AggregationReceipt:
        retrieval = self.memory.retrieve(
            requesting_principal_id=AGGREGATOR_ID,
            intent=(
                f"{request.task_description}\n"
                f"{request.output_format}\naggregate final answer"
            ),
            k1=int(memory_config["k1"]),
            k2=int(memory_config["k2"]),
            similarity_threshold=float(
                memory_config["similarity_threshold"]
            ),
            utility_weight=0.0,
        )
        memory_context = retrieval.render(
            int(memory_config["max_context_chars"]),
            include_utility=False,
        )
        prompt = (
            f"You are the independent Aggregator. Profile:\n{self.profile}\n\n"
            f"Overall task:\n{request.task_description}\n\n"
            f"Required output format:\n{request.output_format}\n\n"
            "Aggregate the final answer from the policy-visible team "
            "workstate and the returning Worker's specialist report. Resolve "
            "conflicts in favor of current, provenance-backed state. Do not "
            "invent missing evidence. You have no access to ground truth.\n\n"
            "Policy-visible Router workstate:\n"
            f"{request.router_workstate or '[empty]'}\n\n"
            "Returning Worker report:\n"
            f"{request.returning_worker_report or '[empty]'}\n\n"
            "Relevant aggregation experiences from only your private "
            "namespace:\n"
            f"{memory_context or '[none]'}"
        )
        answer = str(generate(prompt) or "").strip()
        if not answer:
            raise RuntimeError("Aggregator produced an empty answer")
        memory = self.memory.remember(
            requesting_principal_id=AGGREGATOR_ID,
            task_id=request.task_id,
            intent=(
                f"Aggregate final answer for {request.task_description}"
            ),
            experience=answer,
            initial_utility=0.5,
            attributes={
                "kind": "aggregation_experience",
                "arm": request.arm,
                "value_updated": False,
            },
        )
        return AggregationReceipt(
            task_id=request.task_id,
            arm=request.arm,
            aggregator_principal_id=AGGREGATOR_ID,
            answer=answer,
            answer_sha256=canonical_sha256(answer),
            prompt_sha256=canonical_sha256(prompt),
            memory_retrieval=retrieval,
            stored_memory_id=memory.memory_id,
        )


@dataclass(frozen=True)
class EvaluationReceipt:
    task_id: str
    arm: str
    evaluator_principal_id: str
    answer_sha256: str
    score: float
    dimensions: Mapping[str, float]
    scorer: str
    ground_truth_visible_to_evaluator: bool = True
    ground_truth_disclosed_to_other_agents: bool = False

    def record(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "arm": self.arm,
            "evaluator_principal_id": self.evaluator_principal_id,
            "answer_sha256": self.answer_sha256,
            "score": self.score,
            "dimensions": dict(self.dimensions),
            "scorer": self.scorer,
            "ground_truth_visible_to_evaluator": (
                self.ground_truth_visible_to_evaluator
            ),
            "ground_truth_disclosed_to_other_agents": (
                self.ground_truth_disclosed_to_other_agents
            ),
        }


class EvaluatorAgent:
    """The only role allowed to invoke the benchmark scorer."""

    principal_id = EVALUATOR_ID

    def evaluate(
        self,
        *,
        task_id: str,
        arm: str,
        answer: str,
        score: Callable[[str], tuple[float, Mapping[str, float]]],
        scorer: str,
    ) -> EvaluationReceipt:
        value, dimensions = score(answer)
        normalized = float(value)
        if not 0.0 <= normalized <= 1.0:
            raise ValueError("Evaluator score must be in [0, 1]")
        return EvaluationReceipt(
            task_id=task_id,
            arm=arm,
            evaluator_principal_id=self.principal_id,
            answer_sha256=canonical_sha256(answer),
            score=normalized,
            dimensions=dict(dimensions),
            scorer=scorer,
        )


validate_eight_agent_contract()


__all__ = [
    "AGGREGATOR_ID",
    "ALL_AGENT_IDS",
    "AggregatorAgent",
    "AggregationReceipt",
    "AggregationRequest",
    "EIGHT_AGENT_CONTRACT",
    "EVALUATOR_ID",
    "EvaluationReceipt",
    "EvaluatorAgent",
    "ROUTER_ID",
    "WORKER_IDS",
    "validate_eight_agent_contract",
]
