"""Memora benchmark adapter for controlled long-horizon Agent RETURN.

The upstream benchmark contains privileged mutation annotations and evaluation
rubrics.  This module deliberately separates those fields from the records
that may be rendered into an actor prompt:

* loaders retain the official annotations for materialization and evaluation;
* a divergence manifest stores only session identifiers and digests;
* actor views are reconstructed from raw conversation text;
* all RETURN arms fork from one immutable Router checkpoint.

The controlled panel evaluates state readmission, not mutation extraction.
Consequently, official annotations may be used offline to select an eligible
episode and construct version relations, but are never exposed to task agents.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .eight_agent_pipeline import WORKER_IDS
from .return_governance import ReturnMemoryGovernor
from .router_return_governance import (
    RouterTraceCompilation,
    compile_router_trace,
    render_router_fork,
)
from .router_return_protocol import (
    RouterObligation,
    RouterPostAbsenceCheckpoint,
    RouterReturnArm,
    RouterReturnFork,
    RouterReturnProtocol,
    RouterTaskDAG,
    RouterTaskNode,
    RouterWorkstateItem,
    canonical_sha256,
)
from .unified_mas_contract import (
    deterministic_session_assignment,
    unified_mas_episode_record,
)


MEMORA_RETURN_MANIFEST_SCHEMA = "memora_mas_return_manifest_v1"
MEMORA_RETURN_EPISODE_SCHEMA = "memora_mas_return_episode_v1"
MEMORA_DIVERGENCE_SCHEMA = "memora_divergence_witness_v1"
MEMORA_FAMA_SCHEMA = "memora_fama_score_v1"
MEMORA_ACTOR_VIEW_SCHEMA = "memora_actor_return_view_v1"

PRIVILEGED_SESSION_FIELDS = frozenset(
    {"operation", "operation_details", "share_memory"}
)
PRIVILEGED_QUESTION_FIELDS = frozenset(
    {"memory_evidence", "forgetting_evidence", "evaluation"}
)


def _required(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _unique_ints(values: Iterable[object]) -> tuple[int, ...]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        number = int(value)
        if number < 1:
            raise ValueError("session ids must be positive")
        if number not in seen:
            result.append(number)
            seen.add(number)
    return tuple(result)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _slug(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    return text.strip("-") or "unknown"


@dataclass(frozen=True)
class MemoraTurn:
    turn: int
    speaker: str
    message: str
    share_memory: bool | None = None

    def actor_record(self) -> dict[str, object]:
        """Return only fields observable in the original conversation."""

        return {
            "turn": self.turn,
            "speaker": self.speaker,
            "message": self.message,
        }


@dataclass(frozen=True)
class MemoraSession:
    session_id: int
    session_type: str
    operation: str | None
    operation_details: Mapping[str, object]
    date: str
    persona: str
    conversation: tuple[MemoraTurn, ...]
    source_path: str
    source_sha256: str

    @property
    def conversation_text(self) -> str:
        return "\n".join(
            f"{turn.speaker}: {turn.message}" for turn in self.conversation
        )

    @property
    def effective_operation(self) -> str | None:
        """Resolve converted/fallback operations in the released data."""

        actual = str(
            self.operation_details.get("actual_operation") or ""
        ).strip()
        if actual in {"add", "update", "delete"}:
            return actual
        converted = str(
            self.operation_details.get("operation_converted") or ""
        ).strip()
        if "_to_" in converted:
            candidate = converted.rsplit("_to_", 1)[-1]
            if candidate in {"add", "update", "delete"}:
                return candidate
        if self.operation in {"add", "update", "delete"}:
            return self.operation
        return None

    def actor_record(self) -> dict[str, object]:
        """Return a prompt-safe record with all benchmark labels removed."""

        return {
            "session_id": self.session_id,
            "date": self.date,
            "persona": self.persona,
            "conversation": [
                turn.actor_record() for turn in self.conversation
            ],
            "source_sha256": self.source_sha256,
        }

    def actor_text(self) -> str:
        return json.dumps(
            self.actor_record(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class MemoraRubricItem:
    evaluation_question_id: str
    evaluation_question: str
    expected_answer: str
    evaluation_type: str

    def __post_init__(self) -> None:
        if self.expected_answer not in {"yes", "no"}:
            raise ValueError("Memora rubric expected_answer must be yes/no")
        if self.evaluation_type not in {
            "memory_presence",
            "forgetting_absence",
        }:
            raise ValueError("unknown Memora evaluation_type")

    def record(self) -> dict[str, object]:
        return {
            "evaluation_question_id": self.evaluation_question_id,
            "evaluation_question": self.evaluation_question,
            "expected_answer": self.expected_answer,
            "evaluation_type": self.evaluation_type,
        }


@dataclass(frozen=True)
class MemoraQuestion:
    question_id: str
    task_type: str
    question: str
    question_date: str
    memory_evidence: Mapping[str, object]
    forgetting_evidence: Mapping[str, object]
    rubric: tuple[MemoraRubricItem, ...]

    def actor_record(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "task_type": self.task_type,
            "question": self.question,
            "question_date": self.question_date,
        }

    def evaluator_record(self) -> dict[str, object]:
        return {
            **self.actor_record(),
            "memory_evidence": dict(self.memory_evidence),
            "forgetting_evidence": dict(self.forgetting_evidence),
            "evaluation": {
                "evaluation_questions": [
                    item.record() for item in self.rubric
                ]
            },
        }


@dataclass(frozen=True)
class MemoraTimeline:
    period: str
    persona: str
    sessions: tuple[MemoraSession, ...]
    questions: tuple[MemoraQuestion, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        session_ids = tuple(item.session_id for item in self.sessions)
        if session_ids != tuple(sorted(session_ids)):
            raise ValueError("Memora sessions must be chronological")
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("Memora session ids must be unique")

    @property
    def cluster_id(self) -> str:
        return f"{self.period}:{self.persona}"

    @property
    def by_session_id(self) -> dict[int, MemoraSession]:
        return {item.session_id: item for item in self.sessions}

    @property
    def by_question_id(self) -> dict[str, MemoraQuestion]:
        return {item.question_id: item for item in self.questions}


@dataclass(frozen=True)
class MemoraStaleLink:
    mutation_session_id: int
    old_source_session_id: int
    effective_operation: str
    forgotten_value_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "mutation_session_id": self.mutation_session_id,
            "old_source_session_id": self.old_source_session_id,
            "effective_operation": self.effective_operation,
            "forgotten_value_sha256": self.forgotten_value_sha256,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "MemoraStaleLink":
        return cls(
            mutation_session_id=int(value["mutation_session_id"]),
            old_source_session_id=int(value["old_source_session_id"]),
            effective_operation=_required(
                value["effective_operation"], "effective_operation"
            ),
            forgotten_value_sha256=_required(
                value["forgotten_value_sha256"],
                "forgotten_value_sha256",
            ),
        )


@dataclass(frozen=True)
class MemoraDivergenceWitness:
    departure_session_id: int
    return_session_id: int
    departure_fraction: float
    valid_old_session_ids: tuple[int, ...]
    required_absence_session_ids: tuple[int, ...]
    stale_links: tuple[MemoraStaleLink, ...]
    actual_absence_sessions: int
    controlled_event_oracle: bool
    actor_ground_truth_exposed: bool
    eligible: bool
    reasons: tuple[str, ...] = ()

    def record(self) -> dict[str, object]:
        record = {
            "schema_version": MEMORA_DIVERGENCE_SCHEMA,
            "departure_session_id": self.departure_session_id,
            "return_session_id": self.return_session_id,
            "departure_fraction": self.departure_fraction,
            "valid_old_session_ids": list(self.valid_old_session_ids),
            "required_absence_session_ids": list(
                self.required_absence_session_ids
            ),
            "stale_links": [item.record() for item in self.stale_links],
            "actual_absence_sessions": self.actual_absence_sessions,
            "controlled_event_oracle": self.controlled_event_oracle,
            "actor_ground_truth_exposed": self.actor_ground_truth_exposed,
            "has_valid_old": bool(self.valid_old_session_ids),
            "has_required_absence": bool(
                self.required_absence_session_ids
            ),
            "has_stale_invalidation": bool(self.stale_links),
            "eligible": self.eligible,
            "reasons": list(self.reasons),
        }
        record["witness_sha256"] = canonical_sha256(record)
        return record

    @classmethod
    def from_record(
        cls, value: Mapping[str, object]
    ) -> "MemoraDivergenceWitness":
        witness = cls(
            departure_session_id=int(value["departure_session_id"]),
            return_session_id=int(value["return_session_id"]),
            departure_fraction=float(value["departure_fraction"]),
            valid_old_session_ids=_unique_ints(
                value.get("valid_old_session_ids") or ()
            ),
            required_absence_session_ids=_unique_ints(
                value.get("required_absence_session_ids") or ()
            ),
            stale_links=tuple(
                MemoraStaleLink.from_record(item)
                for item in value.get("stale_links") or ()
                if isinstance(item, Mapping)
            ),
            actual_absence_sessions=int(
                value.get("actual_absence_sessions") or 0
            ),
            controlled_event_oracle=bool(
                value.get("controlled_event_oracle", True)
            ),
            actor_ground_truth_exposed=bool(
                value.get("actor_ground_truth_exposed", False)
            ),
            eligible=bool(value.get("eligible", False)),
            reasons=tuple(str(item) for item in value.get("reasons") or ()),
        )
        expected = value.get("witness_sha256")
        if expected and witness.record()["witness_sha256"] != expected:
            raise ValueError("Memora divergence witness digest mismatch")
        return witness


@dataclass(frozen=True)
class MemoraReturnEpisode:
    episode_id: str
    period: str
    persona: str
    task_type: str
    question_id: str
    timeline_sha256: str
    witness: MemoraDivergenceWitness
    source_revision: str

    @property
    def cluster_id(self) -> str:
        return f"{self.period}:{self.persona}"

    def record(self) -> dict[str, object]:
        record = {
            "schema_version": MEMORA_RETURN_EPISODE_SCHEMA,
            "episode_id": self.episode_id,
            "cluster_id": self.cluster_id,
            "period": self.period,
            "persona": self.persona,
            "task_type": self.task_type,
            "question_id": self.question_id,
            "timeline_sha256": self.timeline_sha256,
            "witness": self.witness.record(),
            "source_revision": self.source_revision,
        }
        record["episode_sha256"] = canonical_sha256(record)
        return record

    @classmethod
    def from_record(
        cls, value: Mapping[str, object]
    ) -> "MemoraReturnEpisode":
        witness = value.get("witness")
        if not isinstance(witness, Mapping):
            raise ValueError("Memora episode requires a witness")
        episode = cls(
            episode_id=_required(value["episode_id"], "episode_id"),
            period=_required(value["period"], "period"),
            persona=_required(value["persona"], "persona"),
            task_type=_required(value["task_type"], "task_type"),
            question_id=_required(value["question_id"], "question_id"),
            timeline_sha256=_required(
                value["timeline_sha256"], "timeline_sha256"
            ),
            witness=MemoraDivergenceWitness.from_record(witness),
            source_revision=_required(
                value.get("source_revision") or "unknown",
                "source_revision",
            ),
        )
        expected = value.get("episode_sha256")
        if expected and episode.record()["episode_sha256"] != expected:
            raise ValueError("Memora episode digest mismatch")
        return episode


def memora_unified_mas_episode_record(
    timeline: MemoraTimeline,
    episode: MemoraReturnEpisode,
    *,
    departure_session_id: int | None = None,
    return_session_id: int | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    """Bind a Memora departure cut to the shared five-agent topology."""

    if timeline.period != episode.period or timeline.persona != episode.persona:
        raise ValueError("episode/timeline identity mismatch")
    departure_id = (
        episode.witness.departure_session_id
        if departure_session_id is None
        else int(departure_session_id)
    )
    return_id = (
        episode.witness.return_session_id
        if return_session_id is None
        else int(return_session_id)
    )
    session_ids = tuple(
        session.session_id
        for session in timeline.sessions
        if session.session_id <= return_id
    )
    assignments = deterministic_session_assignment(
        session_ids,
        departure_after_session=departure_id,
    )
    return unified_mas_episode_record(
        benchmark="Memora-MAS-Return",
        episode_id=episode.episode_id,
        assignments=assignments,
        ordered_session_ids=session_ids,
        departure_after_session=departure_id,
        return_after_session=return_id,
        checkpoint_sha256=checkpoint_sha256,
    )


@dataclass(frozen=True)
class MemoraReturnForkBundle:
    episode: MemoraReturnEpisode
    protocol: RouterReturnProtocol
    checkpoint: RouterPostAbsenceCheckpoint
    trace_compilation: RouterTraceCompilation
    forks: tuple[RouterReturnFork, ...]
    actor_views: Mapping[str, str]
    valid_old_item_ids: tuple[str, ...]
    required_absence_item_ids: tuple[str, ...]
    stale_old_item_ids: tuple[str, ...]
    mutation_item_ids: tuple[str, ...]
    audit: Mapping[str, object]

    def record(self) -> dict[str, object]:
        return {
            "schema_version": "memora_return_fork_bundle_v2",
            "episode_id": self.episode.episode_id,
            "checkpoint_sha256": self.checkpoint.checkpoint_sha256,
            "forks": [item.record() for item in self.forks],
            "valid_old_item_ids": list(self.valid_old_item_ids),
            "required_absence_item_ids": list(
                self.required_absence_item_ids
            ),
            "stale_old_item_ids": list(self.stale_old_item_ids),
            "mutation_item_ids": list(self.mutation_item_ids),
            "trace_compilation": self.trace_compilation.record(),
            "audit": dict(self.audit),
        }


@dataclass(frozen=True)
class MemoraRubricJudgment:
    evaluation_question_id: str
    predicted_answer: str

    def __post_init__(self) -> None:
        if self.predicted_answer not in {"yes", "no"}:
            raise ValueError("rubric judgment must be yes/no")


@dataclass(frozen=True)
class MemoraFamaScore:
    total_evaluations: int
    correct_evaluations: int
    overall_accuracy: float
    memory_presence_total: int
    memory_presence_correct: int
    memory_presence_accuracy: float
    forgetting_absence_total: int
    forgetting_absence_correct: int
    forgetting_absence_accuracy: float
    forgetting_weight: float
    fama: float
    item_results: tuple[Mapping[str, object], ...]

    def record(self) -> dict[str, object]:
        return {
            "schema_version": MEMORA_FAMA_SCHEMA,
            "total_evaluations": self.total_evaluations,
            "correct_evaluations": self.correct_evaluations,
            "overall_accuracy": self.overall_accuracy,
            "memory_presence_total": self.memory_presence_total,
            "memory_presence_correct": self.memory_presence_correct,
            "memory_presence_accuracy": self.memory_presence_accuracy,
            "forgetting_absence_total": self.forgetting_absence_total,
            "forgetting_absence_correct": self.forgetting_absence_correct,
            "forgetting_absence_accuracy": self.forgetting_absence_accuracy,
            "forgetting_weight": self.forgetting_weight,
            "fama": self.fama,
            "item_results": [dict(item) for item in self.item_results],
        }


def _load_turn(value: Mapping[str, object]) -> MemoraTurn:
    return MemoraTurn(
        turn=int(value.get("turn") or 0),
        speaker=_required(value.get("speaker"), "speaker"),
        message=_required(value.get("message"), "message"),
        share_memory=(
            bool(value["share_memory"])
            if "share_memory" in value
            else None
        ),
    )


def _load_session(path: Path) -> MemoraSession:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Memora session must be an object: {path}")
    conversation = value.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError(f"Memora session has no conversation: {path}")
    details = value.get("operation_details")
    if not isinstance(details, Mapping):
        details = {}
    operation = value.get("operation")
    return MemoraSession(
        session_id=int(value["session_id"]),
        session_type=str(value.get("session_type") or "unknown"),
        operation=(str(operation) if operation is not None else None),
        operation_details=dict(details),
        date=str(value.get("date") or ""),
        persona=_required(value.get("persona"), "persona"),
        conversation=tuple(
            _load_turn(item)
            for item in conversation
            if isinstance(item, Mapping)
        ),
        source_path=str(path),
        source_sha256=_file_sha256(path),
    )


def _load_questions(path: Path) -> tuple[MemoraQuestion, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = value.get("questions")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Memora evaluation file is malformed: {path}")
    result: list[MemoraQuestion] = []
    for task_type, rows in raw.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            evaluation = row.get("evaluation")
            if not isinstance(evaluation, Mapping):
                raise ValueError("Memora question is missing evaluation")
            rubric_rows = evaluation.get("evaluation_questions")
            if not isinstance(rubric_rows, list) or not rubric_rows:
                raise ValueError("Memora question has an empty rubric")
            memory_evidence = row.get("memory_evidence")
            forgetting_evidence = row.get("forgetting_evidence")
            result.append(
                MemoraQuestion(
                    question_id=_required(
                        row.get("question_id"), "question_id"
                    ),
                    task_type=str(task_type),
                    question=_required(row.get("question"), "question"),
                    question_date=str(row.get("question_date") or ""),
                    memory_evidence=(
                        dict(memory_evidence)
                        if isinstance(memory_evidence, Mapping)
                        else {}
                    ),
                    forgetting_evidence=(
                        dict(forgetting_evidence)
                        if isinstance(forgetting_evidence, Mapping)
                        else {}
                    ),
                    rubric=tuple(
                        MemoraRubricItem(
                            evaluation_question_id=_required(
                                item.get("evaluation_question_id"),
                                "evaluation_question_id",
                            ),
                            evaluation_question=_required(
                                item.get("evaluation_question"),
                                "evaluation_question",
                            ),
                            expected_answer=str(
                                item.get("expected_answer") or ""
                            ).strip().lower(),
                            evaluation_type=_required(
                                item.get("evaluation_type"),
                                "evaluation_type",
                            ),
                        )
                        for item in rubric_rows
                        if isinstance(item, Mapping)
                    ),
                )
            )
    return tuple(result)


def load_memora_timeline(
    data_root: Path, *, period: str, persona: str
) -> MemoraTimeline:
    """Load one persona-period as a persistent benchmark workspace."""

    root = data_root / period / persona
    conversation_dir = root / "conversations"
    question_path = root / f"evaluation_questions_{persona}.json"
    if not conversation_dir.is_dir():
        raise FileNotFoundError(f"missing Memora conversations: {root}")
    if not question_path.is_file():
        raise FileNotFoundError(f"missing Memora questions: {question_path}")
    sessions = tuple(
        sorted(
            (
                _load_session(path)
                for path in conversation_dir.glob("session_*.json")
            ),
            key=lambda item: item.session_id,
        )
    )
    if not sessions:
        raise ValueError(f"Memora timeline is empty: {root}")
    timeline_source = {
        "period": period,
        "persona": persona,
        "sessions": [
            {
                "session_id": item.session_id,
                "source_sha256": item.source_sha256,
            }
            for item in sessions
        ],
        "questions_sha256": _file_sha256(question_path),
    }
    return MemoraTimeline(
        period=period,
        persona=persona,
        sessions=sessions,
        questions=_load_questions(question_path),
        source_sha256=canonical_sha256(timeline_source),
    )


def discover_memora_timelines(
    data_root: Path,
    *,
    periods: Sequence[str] = ("weekly", "monthly", "quarterly"),
    personas: Sequence[str] | None = None,
) -> tuple[MemoraTimeline, ...]:
    result: list[MemoraTimeline] = []
    requested = set(personas or ())
    for period in periods:
        period_root = data_root / period
        if not period_root.is_dir():
            raise FileNotFoundError(f"missing Memora period: {period_root}")
        for persona_root in sorted(
            item for item in period_root.iterdir() if item.is_dir()
        ):
            if requested and persona_root.name not in requested:
                continue
            if not (persona_root / "conversations").is_dir():
                continue
            result.append(
                load_memora_timeline(
                    data_root,
                    period=period,
                    persona=persona_root.name,
                )
            )
    return tuple(result)


def evidence_session_ids(value: object) -> tuple[int, ...]:
    result: list[int] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            session_id = item.get("session_id")
            if isinstance(session_id, int) and session_id > 0:
                result.append(session_id)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(sorted(set(result)))


def forgetting_value_sessions(value: object) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            session_id = item.get("session_id")
            forgotten = item.get("value")
            if (
                isinstance(session_id, int)
                and session_id > 0
                and forgotten is not None
                and str(forgotten).strip()
            ):
                result.append((str(forgotten).strip(), session_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    deduplicated: dict[tuple[str, int], None] = {}
    for item in result:
        deduplicated[item] = None
    return tuple(deduplicated)


def _old_source_for_value(
    timeline: MemoraTimeline,
    *,
    value: str,
    mutation_session_id: int,
    departure_session_id: int,
) -> int | None:
    needle = _normalized_text(value)
    if not needle:
        return None
    candidates = [
        session.session_id
        for session in timeline.sessions
        if session.session_id <= departure_session_id
        and session.session_id < mutation_session_id
        and needle in _normalized_text(session.conversation_text)
    ]
    return max(candidates) if candidates else None


def build_memora_divergence_witness(
    timeline: MemoraTimeline,
    question: MemoraQuestion,
    *,
    departure_fractions: Sequence[float] = (0.5,),
    minimum_absence_sessions: int = 1,
) -> MemoraDivergenceWitness:
    """Select the first preregistered cut that realizes V, X, and N.

    The function is outcome-blind.  It uses official evidence only to
    materialize a controlled causal episode; no model response is available
    at this stage.  Fractions are tried in the supplied, frozen order.
    """

    if minimum_absence_sessions < 1:
        raise ValueError("minimum_absence_sessions must be positive")
    fractions = tuple(float(value) for value in departure_fractions)
    if not fractions or any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in fractions
    ):
        raise ValueError("departure fractions must lie strictly in (0, 1)")
    valid_ids = evidence_session_ids(question.memory_evidence)
    forgotten = forgetting_value_sessions(question.forgetting_evidence)
    session_by_id = timeline.by_session_id
    return_session_id = max(session_by_id)
    last_reasons: tuple[str, ...] = ()
    for fraction in fractions:
        departure = max(
            1,
            min(
                return_session_id - 1,
                int(round(return_session_id * fraction)),
            ),
        )
        valid_old = tuple(
            session_id for session_id in valid_ids if session_id <= departure
        )
        required_absence = tuple(
            session_id
            for session_id in valid_ids
            if departure < session_id <= return_session_id
        )
        stale_links: list[MemoraStaleLink] = []
        for value, mutation_session_id in forgotten:
            if not departure < mutation_session_id <= return_session_id:
                continue
            mutation = session_by_id.get(mutation_session_id)
            if mutation is None or mutation.effective_operation not in {
                "update",
                "delete",
            }:
                continue
            old_source = _old_source_for_value(
                timeline,
                value=value,
                mutation_session_id=mutation_session_id,
                departure_session_id=departure,
            )
            if old_source is None:
                continue
            stale_links.append(
                MemoraStaleLink(
                    mutation_session_id=mutation_session_id,
                    old_source_session_id=old_source,
                    effective_operation=mutation.effective_operation,
                    forgotten_value_sha256=hashlib.sha256(
                        value.encode("utf-8")
                    ).hexdigest(),
                )
            )
        stale_links = list(
            {
                (
                    item.mutation_session_id,
                    item.old_source_session_id,
                    item.forgotten_value_sha256,
                ): item
                for item in stale_links
            }.values()
        )
        absence = return_session_id - departure
        reasons: list[str] = []
        if not valid_old:
            reasons.append("no_valid_predeparture_evidence")
        if not required_absence:
            reasons.append("no_required_absence_evidence")
        if not stale_links:
            reasons.append("no_traceable_absence_invalidation")
        if absence < minimum_absence_sessions:
            reasons.append("absence_horizon_too_short")
        if question.task_type == "reasoning" and not any(
            item.evaluation_type == "forgetting_absence"
            for item in question.rubric
        ):
            reasons.append("reasoning_question_has_no_forgetting_target")
        if not reasons:
            return MemoraDivergenceWitness(
                departure_session_id=departure,
                return_session_id=return_session_id,
                departure_fraction=fraction,
                valid_old_session_ids=valid_old,
                required_absence_session_ids=required_absence,
                stale_links=tuple(stale_links),
                actual_absence_sessions=absence,
                controlled_event_oracle=True,
                actor_ground_truth_exposed=False,
                eligible=True,
            )
        last_reasons = tuple(reasons)
    return MemoraDivergenceWitness(
        departure_session_id=max(
            1,
            int(round(return_session_id * fractions[0])),
        ),
        return_session_id=return_session_id,
        departure_fraction=fractions[0],
        valid_old_session_ids=(),
        required_absence_session_ids=(),
        stale_links=(),
        actual_absence_sessions=max(
            0, return_session_id - int(round(return_session_id * fractions[0]))
        ),
        controlled_event_oracle=True,
        actor_ground_truth_exposed=False,
        eligible=False,
        reasons=last_reasons or ("no_eligible_departure_cut",),
    )


def materialize_memora_return_episodes(
    timelines: Sequence[MemoraTimeline],
    *,
    source_revision: str,
    departure_fractions: Sequence[float] = (0.5,),
    minimum_absence_sessions: int = 1,
    task_types: Sequence[str] = ("remembering", "recommending"),
) -> tuple[
    tuple[MemoraReturnEpisode, ...],
    tuple[Mapping[str, object], ...],
]:
    """Create eligible episodes and auditable exclusion receipts."""

    allowed = set(task_types)
    episodes: list[MemoraReturnEpisode] = []
    excluded: list[Mapping[str, object]] = []
    for timeline in timelines:
        for question in timeline.questions:
            if question.task_type not in allowed:
                excluded.append(
                    {
                        "cluster_id": timeline.cluster_id,
                        "question_id": question.question_id,
                        "task_type": question.task_type,
                        "reasons": ["task_type_not_selected"],
                    }
                )
                continue
            witness = build_memora_divergence_witness(
                timeline,
                question,
                departure_fractions=departure_fractions,
                minimum_absence_sessions=minimum_absence_sessions,
            )
            if not witness.eligible:
                excluded.append(
                    {
                        "cluster_id": timeline.cluster_id,
                        "question_id": question.question_id,
                        "task_type": question.task_type,
                        "witness": witness.record(),
                        "reasons": list(witness.reasons),
                    }
                )
                continue
            episode_id = (
                f"memora:{timeline.period}:{timeline.persona}:"
                f"{question.question_id}"
            )
            episodes.append(
                MemoraReturnEpisode(
                    episode_id=episode_id,
                    period=timeline.period,
                    persona=timeline.persona,
                    task_type=question.task_type,
                    question_id=question.question_id,
                    timeline_sha256=timeline.source_sha256,
                    witness=witness,
                    source_revision=source_revision,
                )
            )
    return tuple(episodes), tuple(excluded)


def build_memora_manifest(
    timelines: Sequence[MemoraTimeline],
    *,
    source_revision: str,
    departure_fractions: Sequence[float] = (0.5,),
    minimum_absence_sessions: int = 1,
    task_types: Sequence[str] = ("remembering", "recommending"),
) -> dict[str, object]:
    episodes, excluded = materialize_memora_return_episodes(
        timelines,
        source_revision=source_revision,
        departure_fractions=departure_fractions,
        minimum_absence_sessions=minimum_absence_sessions,
        task_types=task_types,
    )
    clusters = sorted({item.cluster_id for item in episodes})
    manifest: dict[str, object] = {
        "schema_version": MEMORA_RETURN_MANIFEST_SCHEMA,
        "source_revision": source_revision,
        "departure_policy": {
            "kind": "first_eligible_preregistered_fraction",
            "fractions": [float(value) for value in departure_fractions],
            "minimum_absence_sessions": minimum_absence_sessions,
            "outcome_blind": True,
            "controlled_event_oracle": True,
        },
        "task_types": list(task_types),
        "timeline_count": len(timelines),
        "eligible_cluster_count": len(clusters),
        "eligible_episode_count": len(episodes),
        "excluded_question_count": len(excluded),
        "episodes": [item.record() for item in episodes],
        "excluded": [dict(item) for item in excluded],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def load_memora_manifest(path: Path) -> tuple[MemoraReturnEpisode, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != MEMORA_RETURN_MANIFEST_SCHEMA:
        raise ValueError("unknown Memora RETURN manifest schema")
    expected = value.get("manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    if expected and canonical_sha256(unsigned) != expected:
        raise ValueError("Memora RETURN manifest digest mismatch")
    rows = value.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("Memora manifest episodes must be a list")
    return tuple(
        MemoraReturnEpisode.from_record(item)
        for item in rows
        if isinstance(item, Mapping)
    )


def _session_item_text(
    session: MemoraSession, *, label: str
) -> str:
    # The released conversations may contain very long assistant turns.  A
    # workstate capsule carries the raw user evidence span plus provenance,
    # not an entire transcript or a generated summary.  Speaker identity is
    # observable; ``share_memory`` and operation labels are not consulted.
    user_turns = tuple(
        turn
        for turn in session.conversation
        if "user" in turn.speaker.casefold()
    )
    selected = user_turns or session.conversation
    span = "\n".join(
        f"{turn.speaker}: {turn.message}" for turn in selected
    )
    max_span_chars = 3000
    if len(span) > max_span_chars:
        half = (max_span_chars - 80) // 2
        span = (
            span[:half]
            + "\n...[raw evidence span clipped at a fixed boundary]...\n"
            + span[-half:]
        )
    provenance = json.dumps(
        {
            "session_id": session.session_id,
            "date": session.date,
            "persona": session.persona,
            "source_sha256": session.source_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{label}. This text is reconstructed only from the original "
        "conversation and date; benchmark operation/evidence labels are "
        "not actor-visible.\n"
        f"<provenance>{provenance}</provenance>\n"
        f"<raw_evidence_span>{span}</raw_evidence_span>"
    )


def _make_task_dag(
    episode: MemoraReturnEpisode,
    *,
    obligation_id: str,
) -> RouterTaskDAG:
    prefix_id = "memora:prefix"
    absence_nodes = tuple(
        RouterTaskNode(
            node_id=f"memora:absence:{worker_id}",
            description=(
                "Process raw absence-period conversations under the "
                f"{worker_id} specialist contract."
            ),
            assigned_principal_id=worker_id,
            depends_on=(prefix_id,),
            obligation_ids=(obligation_id,),
        )
        for worker_id in WORKER_IDS[1:]
    )
    absence_ids = tuple(item.node_id for item in absence_nodes)
    return RouterTaskDAG(
        task_id=episode.episode_id,
        task_description=(
            "Answer one Memora question after a real absence-period state "
            "divergence without using evaluator-only annotations."
        ),
        nodes=(
            RouterTaskNode(
                node_id=prefix_id,
                description=(
                    "Agent1 materializes valid and potentially stale "
                    "predeparture workstate."
                ),
                assigned_principal_id=WORKER_IDS[0],
                obligation_ids=(obligation_id,),
            ),
            *absence_nodes,
            RouterTaskNode(
                node_id="memora:return:synthesis",
                description=(
                    "Agent1 answers from only its arm-visible readmitted state."
                ),
                assigned_principal_id=WORKER_IDS[0],
                depends_on=absence_ids,
                obligation_ids=(obligation_id,),
            ),
        ),
    )


def build_memora_return_forks(
    timeline: MemoraTimeline,
    episode: MemoraReturnEpisode,
    *,
    governor: ReturnMemoryGovernor,
    purpose: str = "answer_memora_question",
    max_chars: int = 120_000,
) -> MemoraReturnForkBundle:
    """Reconstruct all RETURN arms from one immutable checkpoint."""

    if not episode.witness.eligible:
        raise ValueError("cannot run an ineligible Memora RETURN episode")
    if timeline.period != episode.period or timeline.persona != episode.persona:
        raise ValueError("episode/timeline identity mismatch")
    if timeline.source_sha256 != episode.timeline_sha256:
        raise ValueError("episode/timeline source digest mismatch")
    question = timeline.by_question_id.get(episode.question_id)
    if question is None:
        raise KeyError(f"unknown Memora question: {episode.question_id}")
    session_by_id = timeline.by_session_id
    obligation_id = f"obligation:{_slug(episode.question_id)}"
    valid_dependencies = tuple(
        f"valid:{session_id}"
        for session_id in episode.witness.valid_old_session_ids
    )
    absence_dependencies = tuple(
        f"absence:{session_id}"
        for session_id in episode.witness.required_absence_session_ids
    )
    mutation_dependencies = tuple(
        f"mutation:{item.mutation_session_id}:{index}"
        for index, item in enumerate(episode.witness.stale_links)
    )
    required_dependencies = tuple(
        dict.fromkeys(
            valid_dependencies
            + absence_dependencies
            + mutation_dependencies
        )
    )
    protocol = RouterReturnProtocol(
        task_id=episode.episode_id,
        task_description=question.question,
    )
    for principal_id in WORKER_IDS:
        protocol.add_member(principal_id, f"role:{principal_id}")
    protocol.install_plan(
        _make_task_dag(episode, obligation_id=obligation_id),
        (
            RouterObligation(
                obligation_id=obligation_id,
                description=(
                    "Answer the Memora question using current valid state and "
                    "excluding absence-period invalidations."
                ),
                owner_principal_id=WORKER_IDS[0],
                required_dependency_ids=required_dependencies,
                critical_dependency_ids=required_dependencies,
            ),
        ),
    )
    protocol.start_node("memora:prefix", WORKER_IDS[0])
    protocol.complete_node(
        "memora:prefix",
        WORKER_IDS[0],
        (
            "Materialized controlled predeparture state from raw "
            "conversation spans only."
        ),
    )
    valid_old_item_ids: list[str] = []
    for session_id in episode.witness.valid_old_session_ids:
        session = session_by_id[session_id]
        item_id = f"memora:valid-old:{session_id}"
        protocol.add_workstate_item(
            RouterWorkstateItem(
                item_id=item_id,
                text=_session_item_text(
                    session, label="Valid predeparture evidence"
                ),
                owner_principal_id=WORKER_IDS[0],
                writer_principal_id=WORKER_IDS[0],
                dependency_ids=(f"valid:{session_id}",),
                obligation_ids=(obligation_id,),
            )
        )
        valid_old_item_ids.append(item_id)
    stale_old_item_ids: list[str] = []
    for index, link in enumerate(episode.witness.stale_links):
        source = session_by_id[link.old_source_session_id]
        item_id = (
            f"memora:stale-old:{link.old_source_session_id}:"
            f"{link.mutation_session_id}:{index}"
        )
        protocol.add_workstate_item(
            RouterWorkstateItem(
                item_id=item_id,
                text=_session_item_text(
                    source, label="Predeparture state pending revalidation"
                ),
                owner_principal_id=WORKER_IDS[0],
                writer_principal_id=WORKER_IDS[0],
                dependency_ids=(
                    f"stale:{link.mutation_session_id}:{index}",
                ),
                obligation_ids=(obligation_id,),
            )
        )
        stale_old_item_ids.append(item_id)
    protocol.depart(WORKER_IDS[0])

    for worker_id in WORKER_IDS[1:]:
        node_id = f"memora:absence:{worker_id}"
        protocol.start_node(node_id, worker_id)
        protocol.complete_node(
            node_id,
            worker_id,
            (
                "Processed assigned absence-period raw conversation spans "
                "and exported only minimal public workstate."
            ),
        )

    required_absence_item_ids: list[str] = []
    for index, session_id in enumerate(
        episode.witness.required_absence_session_ids
    ):
        writer = WORKER_IDS[1:][index % len(WORKER_IDS[1:])]
        session = session_by_id[session_id]
        item_id = f"memora:absence-current:{session_id}"
        protocol.add_workstate_item(
            RouterWorkstateItem(
                item_id=item_id,
                text=_session_item_text(
                    session, label="Current absence-period evidence"
                ),
                owner_principal_id=WORKER_IDS[0],
                writer_principal_id=writer,
                dependency_ids=(f"absence:{session_id}",),
                obligation_ids=(obligation_id,),
            )
        )
        required_absence_item_ids.append(item_id)

    mutation_item_ids: list[str] = []
    for index, (link, stale_item_id) in enumerate(
        zip(episode.witness.stale_links, stale_old_item_ids)
    ):
        writer = WORKER_IDS[1:][index % len(WORKER_IDS[1:])]
        mutation = session_by_id[link.mutation_session_id]
        mutation_item_id = (
            f"memora:mutation:{link.mutation_session_id}:{index}"
        )
        replacement = RouterWorkstateItem(
            item_id=mutation_item_id,
            text=_session_item_text(
                mutation,
                label=(
                    "Current superseding state"
                    if link.effective_operation == "update"
                    else "Current deletion tombstone"
                ),
            ),
            owner_principal_id=WORKER_IDS[0],
            writer_principal_id=writer,
            dependency_ids=(
                f"mutation:{link.mutation_session_id}:{index}",
            ),
            obligation_ids=(obligation_id,),
            version=2,
            supersedes_item_id=(
                stale_item_id
                if link.effective_operation == "update"
                else None
            ),
        )
        if link.effective_operation == "update":
            protocol.supersede_workstate_item(
                actor_principal_id=writer,
                old_item_id=stale_item_id,
                replacement_item=replacement,
            )
        else:
            protocol.revoke_workstate_item(
                actor_principal_id=writer,
                item_id=stale_item_id,
                reason=(
                    "Official offline materialization links this raw "
                    "conversation to a deletion; the label is not rendered."
                ),
            )
            protocol.add_workstate_item(replace(replacement, version=1))
        mutation_item_ids.append(mutation_item_id)

    checkpoint = protocol.freeze_post_absence_checkpoint(WORKER_IDS[0])
    data_item_count = (
        len(valid_old_item_ids)
        + len(required_absence_item_ids)
        + len(mutation_item_ids)
    )
    trace_compilation = compile_router_trace(
        protocol,
        checkpoint,
        governor,
        purpose=purpose,
        max_items=max(4, data_item_count + 2),
        max_chars=max_chars,
    )
    protocol.start_return(WORKER_IDS[0])
    protocol.finish_return(
        WORKER_IDS[0],
        admitted=trace_compilation.compilation.view is not None,
        reason=(
            "Memora controlled RETURN readmission compiled from one frozen "
            "post-absence checkpoint."
        ),
    )
    forks = trace_compilation.forks
    actor_views = {
        fork.arm.value: render_router_fork(fork, checkpoint)
        for fork in forks
    }
    fork_by_arm = {item.arm: item for item in forks}
    valid_old = set(valid_old_item_ids)
    required_absence = set(required_absence_item_ids)
    stale_old = set(stale_old_item_ids)
    mutation = set(mutation_item_ids)
    static_ids = set(
        fork_by_arm[RouterReturnArm.STATIC].inherited_item_ids
    )
    reset_ids = set(fork_by_arm[RouterReturnArm.RESET].inherited_item_ids)
    restore_ids = set(
        fork_by_arm[RouterReturnArm.RESTORE_OLD].inherited_item_ids
    )
    trace_ids = set(
        fork_by_arm[RouterReturnArm.TRACE].inherited_item_ids
    )
    audit = {
        "schema_version": "memora_return_fork_audit_v1",
        "same_checkpoint_all_arms": len(
            {item.checkpoint_sha256 for item in forks}
        )
        == 1,
        "static_has_valid_old": valid_old.issubset(static_ids),
        "static_has_required_absence": required_absence.issubset(
            static_ids
        ),
        "static_rejects_stale": stale_old.isdisjoint(static_ids),
        "reset_is_empty": not reset_ids,
        "restore_has_valid_old": valid_old.issubset(restore_ids),
        "restore_misses_required_absence": required_absence.isdisjoint(
            restore_ids
        ),
        "restore_inherits_stale": stale_old.issubset(restore_ids),
        "trace_has_valid_old": valid_old.issubset(trace_ids),
        "trace_has_required_absence": required_absence.issubset(
            trace_ids
        ),
        "trace_has_mutation_receipts": mutation.issubset(trace_ids),
        "trace_rejects_stale": stale_old.isdisjoint(trace_ids),
        "trace_critical_dependency_recall": (
            trace_compilation.compilation.selection.critical_dependency_recall
        ),
        "actor_ground_truth_exposed": False,
        "controlled_event_oracle": True,
    }
    audit["eligible"] = all(
        bool(audit[key])
        for key in (
            "same_checkpoint_all_arms",
            "static_has_valid_old",
            "static_has_required_absence",
            "static_rejects_stale",
            "reset_is_empty",
            "restore_has_valid_old",
            "restore_misses_required_absence",
            "restore_inherits_stale",
            "trace_has_valid_old",
            "trace_has_required_absence",
            "trace_has_mutation_receipts",
            "trace_rejects_stale",
        )
    ) and float(audit["trace_critical_dependency_recall"]) >= 0.999999
    if not audit["eligible"]:
        raise RuntimeError(
            "Memora RETURN fork failed divergence audit: "
            + json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
    return MemoraReturnForkBundle(
        episode=episode,
        protocol=protocol,
        checkpoint=checkpoint,
        trace_compilation=trace_compilation,
        forks=forks,
        actor_views=actor_views,
        valid_old_item_ids=tuple(valid_old_item_ids),
        required_absence_item_ids=tuple(required_absence_item_ids),
        stale_old_item_ids=tuple(stale_old_item_ids),
        mutation_item_ids=tuple(mutation_item_ids),
        audit=audit,
    )


def build_returning_worker_prompt(
    *,
    question: MemoraQuestion,
    arm: str,
    actor_view: str,
) -> str:
    """Build a prompt that contains no Memora evaluator-only fields."""

    safe_question = json.dumps(
        question.actor_record(),
        ensure_ascii=False,
        sort_keys=True,
    )
    prompt = (
        "You are agent1 returning to a long-running multi-agent workspace. "
        "Use only the policy-visible Return View below. Identify the current "
        "valid facts needed to answer the user's question. Do not recover "
        "facts that are absent from the view and do not treat superseded or "
        "revoked state as current. Answer the final task directly and do not "
        "mention treatment arm names.\n\n"
        f"<question>\n{safe_question}\n</question>\n\n"
        f"<return_view>\n{actor_view}\n</return_view>"
    )
    forbidden = (
        "memory_evidence",
        "forgetting_evidence",
        "evaluation_question_id",
        "\"operation_details\"",
        "\"share_memory\"",
    )
    if any(value in prompt for value in forbidden):
        raise RuntimeError("evaluator-only Memora label entered actor prompt")
    return prompt


def normalize_yes_no(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^[`*\s]+|[`*\s]+$", "", text)
    match = re.search(r"\b(yes|no)\b", text)
    if match is None:
        raise ValueError(f"judge response is not yes/no: {value!r}")
    return match.group(1)


def score_memora_fama(
    question: MemoraQuestion,
    judgments: Sequence[MemoraRubricJudgment],
) -> MemoraFamaScore:
    """Compute the official per-question FAMA decomposition."""

    predicted = {
        item.evaluation_question_id: item.predicted_answer
        for item in judgments
    }
    if len(predicted) != len(judgments):
        raise ValueError("duplicate Memora rubric judgment id")
    expected_ids = {
        item.evaluation_question_id for item in question.rubric
    }
    if set(predicted) != expected_ids:
        missing = expected_ids - set(predicted)
        extra = set(predicted) - expected_ids
        raise ValueError(
            "incomplete Memora judgments; missing="
            + ",".join(sorted(missing))
            + " extra="
            + ",".join(sorted(extra))
        )
    item_results: list[Mapping[str, object]] = []
    totals = {"memory_presence": 0, "forgetting_absence": 0}
    correct = {"memory_presence": 0, "forgetting_absence": 0}
    for item in question.rubric:
        actual = predicted[item.evaluation_question_id]
        is_correct = actual == item.expected_answer
        totals[item.evaluation_type] += 1
        correct[item.evaluation_type] += int(is_correct)
        item_results.append(
            {
                **item.record(),
                "predicted_answer": actual,
                "correct": is_correct,
            }
        )
    presence_total = totals["memory_presence"]
    forgetting_total = totals["forgetting_absence"]
    if presence_total < 1:
        raise ValueError("FAMA requires at least one memory_presence item")
    presence_accuracy = correct["memory_presence"] / presence_total
    forgetting_accuracy = (
        correct["forgetting_absence"] / forgetting_total
        if forgetting_total
        else 1.0
    )
    total = presence_total + forgetting_total
    weight = forgetting_total / total
    fama = max(
        0.0,
        presence_accuracy - weight * (1.0 - forgetting_accuracy),
    )
    return MemoraFamaScore(
        total_evaluations=total,
        correct_evaluations=sum(correct.values()),
        overall_accuracy=sum(correct.values()) / total,
        memory_presence_total=presence_total,
        memory_presence_correct=correct["memory_presence"],
        memory_presence_accuracy=presence_accuracy,
        forgetting_absence_total=forgetting_total,
        forgetting_absence_correct=correct["forgetting_absence"],
        forgetting_absence_accuracy=forgetting_accuracy,
        forgetting_weight=weight,
        fama=fama,
        item_results=tuple(item_results),
    )


def manifest_summary(value: Mapping[str, object]) -> dict[str, object]:
    episodes = [
        item
        for item in value.get("episodes") or ()
        if isinstance(item, Mapping)
    ]
    by_period: dict[str, int] = {}
    by_task: dict[str, int] = {}
    clusters: set[str] = set()
    for item in episodes:
        period = str(item.get("period") or "unknown")
        task = str(item.get("task_type") or "unknown")
        by_period[period] = by_period.get(period, 0) + 1
        by_task[task] = by_task.get(task, 0) + 1
        clusters.add(str(item.get("cluster_id") or "unknown"))
    return {
        "eligible_episodes": len(episodes),
        "eligible_clusters": len(clusters),
        "by_period": by_period,
        "by_task_type": by_task,
        "excluded_questions": len(value.get("excluded") or ()),
    }


__all__ = [
    "MEMORA_ACTOR_VIEW_SCHEMA",
    "MEMORA_DIVERGENCE_SCHEMA",
    "MEMORA_FAMA_SCHEMA",
    "MEMORA_RETURN_EPISODE_SCHEMA",
    "MEMORA_RETURN_MANIFEST_SCHEMA",
    "MemoraDivergenceWitness",
    "MemoraFamaScore",
    "MemoraQuestion",
    "MemoraReturnEpisode",
    "MemoraReturnForkBundle",
    "MemoraRubricItem",
    "MemoraRubricJudgment",
    "MemoraSession",
    "MemoraStaleLink",
    "MemoraTimeline",
    "build_memora_divergence_witness",
    "build_memora_manifest",
    "build_memora_return_forks",
    "build_returning_worker_prompt",
    "discover_memora_timelines",
    "evidence_session_ids",
    "forgetting_value_sessions",
    "load_memora_manifest",
    "load_memora_timeline",
    "manifest_summary",
    "materialize_memora_return_episodes",
    "memora_unified_mas_episode_record",
    "normalize_yes_no",
    "score_memora_fama",
]
