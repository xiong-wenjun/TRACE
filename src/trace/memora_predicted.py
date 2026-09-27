"""Outcome-blind Memora adapter for end-to-end RESUME evaluation.

This module is intentionally unable to consume Memora ``memory_evidence``,
``forgetting_evidence``, operation labels, or operation details.  It receives
only actor-visible questions, raw conversation records, and model predictions.
Official annotations may be used later by an isolated evaluator to score the
frozen prediction trace.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Iterable, Mapping, Sequence

from .eight_agent_pipeline import WORKER_IDS
from .memora_return import MemoraReturnEpisode, MemoraTimeline
from .return_governance import ReturnAblationMode, ReturnMemoryGovernor
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


PREDICTED_TRACE_SCHEMA = "memora_resume_predicted_trace_v1"
PREDICTED_BUNDLE_SCHEMA = "memora_resume_predicted_bundle_v2"
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_RETRIEVAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "could",
        "current",
        "do",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "recommending",
        "suggest",
        "tell",
        "the",
        "to",
        "user",
        "what",
        "which",
        "with",
        "would",
        "you",
        "your",
    }
)
_FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "memory_evidence",
        "forgetting_evidence",
        "evaluation",
        "expected_answer",
        "operation",
        "operation_details",
        "share_memory",
    }
)


def _required(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN.findall(str(value or "")))


def _normalized_key(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.:-]+", "-", str(value or "").casefold())
    text = text.strip("-.:")
    if not text:
        raise ValueError("predicted state_key must be non-empty")
    return text[:160]


def _safe_json(value: object) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if any(f'"{key}"' in text for key in _FORBIDDEN_RUNTIME_KEYS):
        raise RuntimeError("privileged Memora field entered predicted runtime")
    return text


def _cosine_tokens(left: object, right: object) -> float:
    a: dict[str, int] = {}
    b: dict[str, int] = {}
    for token in _tokens(left):
        a[token] = a.get(token, 0) + 1
    for token in _tokens(right):
        b[token] = b.get(token, 0) + 1
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    left_norm = math.sqrt(sum(value * value for value in a.values()))
    right_norm = math.sqrt(sum(value * value for value in b.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _retrieval_content_tokens(value: object) -> frozenset[str]:
    result: set[str] = set()
    for token in _tokens(value):
        if token in _RETRIEVAL_STOPWORDS or len(token) < 2:
            continue
        result.add(token)
        if len(token) > 4 and token.endswith("s"):
            result.add(token[:-1])
    return frozenset(result)


def _session_relevance(query: str, row: Mapping[str, object]) -> float:
    """Prefer a highly relevant turn over generic overlap in a long session."""

    query_tokens = _retrieval_content_tokens(query)
    rows = row.get("conversation")
    messages = tuple(
        str(item.get("message") or "")
        for item in rows or ()
        if isinstance(item, Mapping)
    )
    if not query_tokens or not messages:
        return _cosine_tokens(query, _safe_json(row))
    turn_coverages = [
        len(query_tokens & _retrieval_content_tokens(message))
        / len(query_tokens)
        for message in messages
    ]
    return max(turn_coverages, default=0.0) + 0.05 * _cosine_tokens(
        query, _safe_json(row)
    )


@dataclass(frozen=True)
class PredictedRouterPlan:
    obligation: str
    retrieval_queries: tuple[str, ...]
    raw_completion_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "obligation": self.obligation,
            "retrieval_queries": list(self.retrieval_queries),
            "raw_completion_sha256": self.raw_completion_sha256,
        }


@dataclass(frozen=True)
class PredictedStateFact:
    fact_id: str
    session_id: int
    phase: str
    worker_id: str
    state_key: str
    statement: str
    operation: str
    evidence_quote: str
    source_sha256: str
    confidence: float
    quote_verified: bool

    def __post_init__(self) -> None:
        if self.phase not in {"predeparture", "absence"}:
            raise ValueError("predicted fact phase is invalid")
        if self.operation not in {"add", "update", "delete"}:
            raise ValueError("predicted fact operation is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("predicted fact confidence must be in [0, 1]")

    @property
    def dependency_id(self) -> str:
        return f"predicted-dependency:{self.fact_id}"

    @property
    def item_id(self) -> str:
        return f"memora:predicted:{self.fact_id}"

    def record(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "session_id": self.session_id,
            "phase": self.phase,
            "worker_id": self.worker_id,
            "state_key": self.state_key,
            "statement": self.statement,
            "operation": self.operation,
            "evidence_quote": self.evidence_quote,
            "source_sha256": self.source_sha256,
            "confidence": self.confidence,
            "quote_verified": self.quote_verified,
            "dependency_id": self.dependency_id,
            "item_id": self.item_id,
        }


@dataclass(frozen=True)
class PredictedDependencyPlan:
    selected_fact_ids: tuple[str, ...]
    critical_fact_ids: tuple[str, ...]
    rationale: str
    raw_completion_sha256: str
    deterministic_fallback: bool = False

    def record(self) -> dict[str, object]:
        return {
            "selected_fact_ids": list(self.selected_fact_ids),
            "critical_fact_ids": list(self.critical_fact_ids),
            "rationale": self.rationale,
            "raw_completion_sha256": self.raw_completion_sha256,
            "deterministic_fallback": self.deterministic_fallback,
        }


@dataclass(frozen=True)
class PredictedSelectorBudget:
    """Outcome-blind dynamic budget for obligation-conditioned selection."""

    requested_base: int
    requested_hard_cap: int
    effective_base: int
    current_fact_count: int
    evolved_fact_count: int
    obligation_query_count: int
    candidate_pressure: int
    evolution_pressure: int
    obligation_pressure: int
    growth_extra: int
    effective_budget: int

    def record(self) -> dict[str, object]:
        return {
            "policy": "dynamic_obligation_coverage_v1",
            "requested_base": self.requested_base,
            "requested_hard_cap": self.requested_hard_cap,
            "effective_base": self.effective_base,
            "current_fact_count": self.current_fact_count,
            "evolved_fact_count": self.evolved_fact_count,
            "obligation_query_count": self.obligation_query_count,
            "candidate_pressure": self.candidate_pressure,
            "evolution_pressure": self.evolution_pressure,
            "obligation_pressure": self.obligation_pressure,
            "growth_extra": self.growth_extra,
            "effective_budget": self.effective_budget,
        }


def dynamic_selector_budget(
    current_facts: Sequence[PredictedStateFact],
    *,
    obligation_query_count: int,
    base_selected_facts: int = 12,
    max_selected_facts: int = 16,
) -> PredictedSelectorBudget:
    """Allocate 12--16 slots from visible state complexity.

    An explicitly smaller hard cap remains supported for frozen legacy runs.
    No rubric, expected answer, or evaluator-only evidence enters this rule.
    """

    if base_selected_facts < 1 or max_selected_facts < 1:
        raise ValueError("selector budgets must be positive")
    if obligation_query_count < 0:
        raise ValueError("obligation_query_count must be non-negative")
    effective_base = min(base_selected_facts, max_selected_facts)
    current_fact_count = len(current_facts)
    evolved_fact_count = sum(
        item.phase == "absence" or item.operation in {"update", "delete"}
        for item in current_facts
    )
    candidate_pressure = math.ceil(
        max(0, current_fact_count - effective_base) / 4
    )
    evolution_pressure = math.ceil(evolved_fact_count / 4)
    obligation_pressure = max(0, obligation_query_count - 2)
    growth_extra = min(
        max_selected_facts - effective_base,
        max(
            candidate_pressure,
            evolution_pressure,
            obligation_pressure,
        ),
    )
    effective_budget = min(
        current_fact_count,
        effective_base + growth_extra,
        max_selected_facts,
    )
    return PredictedSelectorBudget(
        requested_base=base_selected_facts,
        requested_hard_cap=max_selected_facts,
        effective_base=effective_base,
        current_fact_count=current_fact_count,
        evolved_fact_count=evolved_fact_count,
        obligation_query_count=obligation_query_count,
        candidate_pressure=candidate_pressure,
        evolution_pressure=evolution_pressure,
        obligation_pressure=obligation_pressure,
        growth_extra=growth_extra,
        effective_budget=effective_budget,
    )


@dataclass(frozen=True)
class PredictedExtractionBatch:
    phase: str
    worker_id: str
    session_ids: tuple[int, ...]
    facts: tuple[PredictedStateFact, ...]
    rejected_rows: int
    raw_completion_sha256: str
    deterministic_fallback: bool = False

    def record(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "worker_id": self.worker_id,
            "session_ids": list(self.session_ids),
            "facts": [item.record() for item in self.facts],
            "rejected_rows": self.rejected_rows,
            "raw_completion_sha256": self.raw_completion_sha256,
            "deterministic_fallback": self.deterministic_fallback,
        }


@dataclass(frozen=True)
class PredictedTrace:
    departure_session_id: int
    return_session_id: int
    departure_fraction: float
    router_plan: PredictedRouterPlan
    predeparture_candidate_session_ids: tuple[int, ...]
    absence_candidate_session_ids: tuple[int, ...]
    extraction_batches: tuple[PredictedExtractionBatch, ...]
    dependency_plan: PredictedDependencyPlan
    selector_budget: PredictedSelectorBudget | None = None

    @property
    def facts(self) -> tuple[PredictedStateFact, ...]:
        by_id: dict[str, PredictedStateFact] = {}
        for batch in self.extraction_batches:
            for fact in batch.facts:
                by_id[fact.fact_id] = fact
        return tuple(sorted(by_id.values(), key=lambda item: (item.session_id, item.fact_id)))

    def record(self) -> dict[str, object]:
        record = {
            "schema_version": PREDICTED_TRACE_SCHEMA,
            "departure_session_id": self.departure_session_id,
            "return_session_id": self.return_session_id,
            "departure_fraction": self.departure_fraction,
            "router_plan": self.router_plan.record(),
            "predeparture_candidate_session_ids": list(
                self.predeparture_candidate_session_ids
            ),
            "absence_candidate_session_ids": list(
                self.absence_candidate_session_ids
            ),
            "extraction_batches": [
                item.record() for item in self.extraction_batches
            ],
            "dependency_plan": self.dependency_plan.record(),
            "selector_budget": (
                self.selector_budget.record()
                if self.selector_budget is not None
                else None
            ),
            "official_evidence_runtime_reads": 0,
            "privileged_session_metadata_runtime_reads": 0,
        }
        record["trace_sha256"] = canonical_sha256(record)
        return record


@dataclass(frozen=True)
class PredictedReturnForkBundle:
    episode: MemoraReturnEpisode
    protocol: RouterReturnProtocol
    checkpoint: RouterPostAbsenceCheckpoint
    trace_compilation: RouterTraceCompilation
    forks: tuple[RouterReturnFork, ...]
    actor_views: Mapping[str, str]
    trace: PredictedTrace
    selected_current_item_ids: tuple[str, ...]
    selected_predeparture_item_ids: tuple[str, ...]
    selected_absence_item_ids: tuple[str, ...]
    stale_old_item_ids: tuple[str, ...]
    mutation_item_ids: tuple[str, ...]
    ablation_compilations: Mapping[str, RouterTraceCompilation]
    audit: Mapping[str, object]

    def record(self) -> dict[str, object]:
        return {
            "schema_version": PREDICTED_BUNDLE_SCHEMA,
            "episode_id": self.episode.episode_id,
            "checkpoint_sha256": self.checkpoint.checkpoint_sha256,
            "forks": [item.record() for item in self.forks],
            "trace": self.trace.record(),
            "selected_current_item_ids": list(self.selected_current_item_ids),
            "selected_predeparture_item_ids": list(
                self.selected_predeparture_item_ids
            ),
            "selected_absence_item_ids": list(
                self.selected_absence_item_ids
            ),
            "stale_old_item_ids": list(self.stale_old_item_ids),
            "mutation_item_ids": list(self.mutation_item_ids),
            "trace_compilation": self.trace_compilation.record(),
            "ablation_compilations": {
                arm: compilation.record()
                for arm, compilation in sorted(
                    self.ablation_compilations.items()
                )
            },
            "audit": dict(self.audit),
        }


def safe_question_record(
    question_id: str,
    task_type: str,
    question: str,
    question_date: str,
) -> dict[str, object]:
    return {
        "question_id": _required(question_id, "question_id"),
        "task_type": _required(task_type, "task_type"),
        "question": _required(question, "question"),
        "question_date": str(question_date or ""),
    }


def actor_session_records(
    timeline: MemoraTimeline,
    *,
    first_session_id: int,
    last_session_id: int,
) -> tuple[dict[str, object], ...]:
    """Copy only actor-visible fields from a timeline interval."""

    return tuple(
        session.actor_record()
        for session in timeline.sessions
        if first_session_id <= session.session_id <= last_session_id
    )


def plan_router_prompt(question_record: Mapping[str, object]) -> str:
    safe = _safe_json(dict(question_record))
    return (
        "You are the Router of a long-running multi-agent workspace. From the "
        "user's current question only, state the unresolved obligation and "
        "write 3-8 short lexical retrieval queries that could locate relevant "
        "raw conversation sessions. Do not invent user facts. Return exactly "
        "one JSON object: {\"obligation\": string, "
        "\"retrieval_queries\": [string, ...]}.\n\n"
        f"<question>{safe}</question>"
    )


def parse_router_plan(text: str) -> PredictedRouterPlan:
    value = extract_json(text)
    if not isinstance(value, Mapping):
        raise ValueError("predicted Router plan must be a JSON object")
    queries = tuple(
        dict.fromkeys(
            _required(item, "retrieval query")
            for item in value.get("retrieval_queries") or ()
        )
    )
    if not queries:
        raise ValueError("predicted Router plan has no retrieval queries")
    return PredictedRouterPlan(
        obligation=_required(value.get("obligation"), "obligation"),
        retrieval_queries=queries[:8],
        raw_completion_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def retrieve_actor_sessions(
    records: Sequence[Mapping[str, object]],
    *,
    question_record: Mapping[str, object],
    retrieval_queries: Sequence[str],
    top_k: int,
    recency_k: int,
) -> tuple[dict[str, object], ...]:
    """Outcome-blind lexical+recency retrieval over raw conversation only."""

    if top_k < 1 or recency_k < 0:
        raise ValueError("invalid predicted retrieval budget")
    query = " ".join(
        (
            str(question_record.get("question") or ""),
            str(question_record.get("task_type") or ""),
            *retrieval_queries,
        )
    )
    ordered = tuple(sorted(records, key=lambda row: int(row["session_id"])))
    scored = sorted(
        ordered,
        key=lambda row: (
            -_session_relevance(query, row),
            -int(row["session_id"]),
        ),
    )
    chosen: dict[int, dict[str, object]] = {}
    for row in scored[:top_k]:
        chosen[int(row["session_id"])] = dict(row)
    for row in ordered[-min(recency_k, len(ordered)) :]:
        chosen[int(row["session_id"])] = dict(row)
    return tuple(chosen[key] for key in sorted(chosen))


def shard_actor_sessions(
    records: Sequence[Mapping[str, object]],
    worker_ids: Sequence[str],
    *,
    max_sessions_per_batch: int,
) -> tuple[tuple[str, tuple[Mapping[str, object], ...]], ...]:
    if not worker_ids:
        raise ValueError("predicted extraction requires at least one worker")
    if max_sessions_per_batch < 1:
        raise ValueError("max_sessions_per_batch must be positive")
    buckets: list[list[Mapping[str, object]]] = [
        [] for _ in range(len(worker_ids))
    ]
    for index, record in enumerate(records):
        buckets[index % len(buckets)].append(record)
    result: list[tuple[str, tuple[Mapping[str, object], ...]]] = []
    for worker_id, bucket in zip(worker_ids, buckets):
        for start in range(0, len(bucket), max_sessions_per_batch):
            result.append(
                (worker_id, tuple(bucket[start : start + max_sessions_per_batch]))
            )
    return tuple(result)


def extraction_prompt(
    *,
    question_record: Mapping[str, object],
    router_plan: PredictedRouterPlan,
    phase: str,
    worker_id: str,
    sessions: Sequence[Mapping[str, object]],
    predeparture_catalog: Sequence[PredictedStateFact] = (),
) -> str:
    if phase not in {"predeparture", "absence"}:
        raise ValueError("invalid predicted extraction phase")
    safe_sessions = [dict(item) for item in sessions]
    catalog = [
        {
            "state_key": item.state_key,
            "statement": item.statement,
            "session_id": item.session_id,
        }
        for item in predeparture_catalog
    ]
    payload = {
        "question": dict(question_record),
        "obligation": router_plan.obligation,
        "phase": phase,
        "worker_id": worker_id,
        "known_predeparture_state": catalog,
        "raw_sessions": safe_sessions,
    }
    return (
        "You are a Worker extracting current workstate from raw user "
        "conversation. Use no outside knowledge. Emit only facts that are "
        "explicitly supported and potentially useful for the Router's "
        "obligation. state_key must identify one specific entity/aspect, not "
        "a broad category: independent preferences or todo items require "
        "different keys. Reuse a prior state_key only when this session "
        "updates or deletes that exact same item. operation must be add, "
        "update, or delete. evidence_quote must copy one complete user "
        "message verbatim from that session. Return exactly "
        "{\"facts\": [{\"session_id\": int, "
        "\"state_key\": string, \"statement\": string, \"operation\": "
        "\"add|update|delete\", \"evidence_quote\": string, "
        "\"confidence\": number}]}. Return an empty facts list if nothing is "
        "relevant. Never output benchmark labels.\n\n"
        f"<actor_visible_input>{_safe_json(payload)}</actor_visible_input>"
    )


def _conversation_text(record: Mapping[str, object]) -> str:
    rows = record.get("conversation")
    if not isinstance(rows, list):
        return ""
    return "\n".join(
        str(item.get("message") or "")
        for item in rows
        if isinstance(item, Mapping)
    )


def _ground_evidence_quote(
    *,
    source: Mapping[str, object],
    quote: str,
    statement: str,
) -> tuple[str, bool]:
    """Bind a model quote to an exact source turn, repairing punctuation drift."""

    conversation = _conversation_text(source)
    normalized_quote = re.sub(r"\W+", " ", quote).casefold().strip()
    normalized_conversation = (
        re.sub(r"\W+", " ", conversation).casefold().strip()
    )
    if normalized_quote and normalized_quote in normalized_conversation:
        return quote, True
    rows = source.get("conversation")
    messages = tuple(
        str(item.get("message") or "").strip()
        for item in rows or ()
        if isinstance(item, Mapping) and str(item.get("message") or "").strip()
    )
    if not messages:
        return quote, False
    ranked = sorted(
        messages,
        key=lambda message: (
            -max(
                _cosine_tokens(quote, message),
                _cosine_tokens(statement, message),
            ),
            -len(message),
        ),
    )
    best = ranked[0]
    score = max(
        _cosine_tokens(quote, best),
        _cosine_tokens(statement, best),
    )
    if score < 0.35:
        return quote, False
    return best, True


def parse_extraction(
    text: str,
    *,
    phase: str,
    worker_id: str,
    sessions: Sequence[Mapping[str, object]],
    minimum_confidence: float,
) -> PredictedExtractionBatch:
    value = extract_json(text)
    rows = value.get("facts") if isinstance(value, Mapping) else value
    if not isinstance(rows, list):
        raise ValueError("predicted extraction is missing facts")
    by_session = {int(item["session_id"]): item for item in sessions}
    accepted: dict[str, PredictedStateFact] = {}
    rejected = 0
    for row in rows:
        if not isinstance(row, Mapping):
            rejected += 1
            continue
        try:
            session_id = int(row["session_id"])
            source = by_session[session_id]
            operation = str(row.get("operation") or "").strip().casefold()
            if operation not in {"add", "update", "delete"}:
                raise ValueError("bad operation")
            confidence = float(row.get("confidence", 0.5))
            if not math.isfinite(confidence) or confidence < minimum_confidence:
                raise ValueError("low confidence")
            state_key = _normalized_key(row.get("state_key"))
            statement = _required(row.get("statement"), "statement")[:1200]
            quote = _required(row.get("evidence_quote"), "evidence_quote")[:500]
            quote, quote_verified = _ground_evidence_quote(
                source=source,
                quote=quote,
                statement=statement,
            )
            if not quote_verified:
                raise ValueError("unverified quote")
            source_sha256 = _required(
                source.get("source_sha256"), "source_sha256"
            )
            fact_id = "pf_" + canonical_sha256(
                {
                    "session_id": session_id,
                    "state_key": state_key,
                    "statement": statement,
                    "operation": operation,
                    "source_sha256": source_sha256,
                }
            )[:24]
            accepted[fact_id] = PredictedStateFact(
                fact_id=fact_id,
                session_id=session_id,
                phase=phase,
                worker_id=worker_id,
                state_key=state_key,
                statement=statement,
                operation=operation,
                evidence_quote=quote,
                source_sha256=source_sha256,
                confidence=min(1.0, max(0.0, confidence)),
                quote_verified=True,
            )
        except (KeyError, TypeError, ValueError):
            rejected += 1
    return PredictedExtractionBatch(
        phase=phase,
        worker_id=worker_id,
        session_ids=tuple(sorted(by_session)),
        facts=tuple(
            sorted(accepted.values(), key=lambda item: (item.session_id, item.fact_id))
        ),
        rejected_rows=rejected,
        raw_completion_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def active_facts_at_departure(
    facts: Sequence[PredictedStateFact],
) -> tuple[PredictedStateFact, ...]:
    current: dict[str, PredictedStateFact] = {}
    for fact in sorted(facts, key=lambda item: (item.session_id, item.fact_id)):
        if fact.operation == "delete":
            current.pop(fact.state_key, None)
        else:
            current[fact.state_key] = fact
    return tuple(sorted(current.values(), key=lambda item: (item.session_id, item.fact_id)))


def dependency_prompt(
    *,
    question_record: Mapping[str, object],
    router_plan: PredictedRouterPlan,
    current_facts: Sequence[PredictedStateFact],
    max_selected_facts: int,
) -> str:
    rows = [
        {
            "fact_id": item.fact_id,
            "session_id": item.session_id,
            "state_key": item.state_key,
            "statement": item.statement,
            "phase": item.phase,
            "predicted_change_kind": item.operation,
        }
        for item in current_facts
    ]
    payload = {
        "question": dict(question_record),
        "obligation": router_plan.obligation,
        "candidate_current_facts": rows,
        "max_selected_facts": max_selected_facts,
    }
    return (
        "You are the Router at agent RETURN. Select every independently "
        "necessary current fact for the unresolved obligation while excluding "
        "irrelevant facts. For recommendations, lists, remaining tasks, or "
        "plural questions, preserve all relevant preferences and constraints "
        "up to the budget; never choose only one illustrative example. Select "
        "only fact_id values from the candidates and do not infer hidden "
        "facts. critical_fact_ids must be a subset of selected_fact_ids. "
        "Return exactly one JSON object: "
        "{\"selected_fact_ids\": [string, ...], \"critical_fact_ids\": "
        "[string, ...], \"rationale\": string}.\n\n"
        f"<actor_visible_input>{_safe_json(payload)}</actor_visible_input>"
    )


def parse_dependency_plan(
    text: str,
    *,
    current_facts: Sequence[PredictedStateFact],
    question_record: Mapping[str, object],
    max_selected_facts: int,
) -> PredictedDependencyPlan:
    value = extract_json(text)
    if not isinstance(value, Mapping):
        raise ValueError("predicted dependency plan must be an object")
    known = {item.fact_id: item for item in current_facts}
    selected = tuple(
        dict.fromkeys(
            str(item)
            for item in value.get("selected_fact_ids") or ()
            if str(item) in known
        )
    )[:max_selected_facts]
    critical = tuple(
        item
        for item in dict.fromkeys(
            str(item) for item in value.get("critical_fact_ids") or ()
        )
        if item in selected
    )
    fallback = False
    if not selected and current_facts:
        query = str(question_record.get("question") or "")
        ranked = sorted(
            current_facts,
            key=lambda item: (
                -_cosine_tokens(query, f"{item.state_key} {item.statement}"),
                -item.session_id,
                item.fact_id,
            ),
        )
        selected = tuple(item.fact_id for item in ranked[:max_selected_facts])
        critical = selected[: min(2, len(selected))]
        fallback = True
    if not selected:
        raise ValueError("no current fact can support predicted obligation")
    if not critical:
        critical = selected[:1]
    return PredictedDependencyPlan(
        selected_fact_ids=selected,
        critical_fact_ids=critical,
        rationale=str(value.get("rationale") or "")[:1200],
        raw_completion_sha256=hashlib.sha256(text.encode()).hexdigest(),
        deterministic_fallback=fallback,
    )


def extract_json(text: str) -> Mapping[str, object] | list[object]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", value):
            try:
                decoded, _ = decoder.raw_decode(value, match.start())
                break
            except json.JSONDecodeError:
                continue
        else:
            raise
    if not isinstance(decoded, (Mapping, list)):
        raise ValueError("predicted model output must be JSON")
    return decoded


def _make_dag(
    *,
    episode_id: str,
    question: str,
    obligation_id: str,
) -> RouterTaskDAG:
    prefix = "memora:predicted:prefix"
    absence_nodes = tuple(
        RouterTaskNode(
            node_id=f"memora:predicted:absence:{worker_id}",
            description="Extract raw absence-period state changes.",
            assigned_principal_id=worker_id,
            depends_on=(prefix,),
            obligation_ids=(obligation_id,),
        )
        for worker_id in WORKER_IDS[1:]
    )
    return RouterTaskDAG(
        task_id=episode_id,
        task_description=question,
        nodes=(
            RouterTaskNode(
                node_id=prefix,
                description="Extract predeparture state from raw sessions.",
                assigned_principal_id=WORKER_IDS[0],
                obligation_ids=(obligation_id,),
            ),
            *absence_nodes,
            RouterTaskNode(
                node_id="memora:predicted:return",
                description="Resume from the admitted predicted workstate.",
                assigned_principal_id=WORKER_IDS[0],
                depends_on=tuple(item.node_id for item in absence_nodes),
                obligation_ids=(obligation_id,),
            ),
        ),
    )


def _item(
    fact: PredictedStateFact,
    *,
    obligation_id: str,
    supersedes_item_id: str | None = None,
    version: int = 1,
) -> RouterWorkstateItem:
    return RouterWorkstateItem(
        item_id=fact.item_id,
        text=_safe_json(
            {
                "session_id": fact.session_id,
                "date_phase": fact.phase,
                "state_key": fact.state_key,
                "statement": fact.statement,
                "predicted_change_kind": fact.operation,
                "evidence_quote": fact.evidence_quote,
                "source_sha256": fact.source_sha256,
                "prediction_confidence": fact.confidence,
            }
        ),
        owner_principal_id=WORKER_IDS[0],
        writer_principal_id=fact.worker_id,
        dependency_ids=(fact.dependency_id,),
        obligation_ids=(obligation_id,),
        version=version,
        supersedes_item_id=supersedes_item_id,
        provenance_sha256=fact.source_sha256,
    )


def _compact_only_item_ids(
    checkpoint: RouterPostAbsenceCheckpoint,
    *,
    candidate_item_ids: Sequence[str],
    question_text: str,
    max_items: int,
) -> tuple[str, ...]:
    """Compact raw candidates without applying RETURN governance."""

    by_id: dict[str, RouterWorkstateItem] = {}
    for item in checkpoint.departure.workstate_items:
        by_id[item.item_id] = item
    for item in checkpoint.current_workstate_items:
        by_id[item.item_id] = item
    candidates = tuple(
        by_id[item_id]
        for item_id in dict.fromkeys(candidate_item_ids)
        if item_id in by_id
    )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -_cosine_tokens(question_text, item.text),
            -item.version,
            item.item_id,
        ),
    )
    return tuple(item.item_id for item in ranked[:max_items])


def build_predicted_return_forks(
    timeline: MemoraTimeline,
    episode: MemoraReturnEpisode,
    *,
    question_record: Mapping[str, object],
    trace: PredictedTrace,
    governor: ReturnMemoryGovernor,
    ablation_arms: Sequence[str] = (),
    purpose: str = "answer_memora_question_predicted",
    max_chars: int = 120_000,
) -> PredictedReturnForkBundle:
    """Build matched arms using only the frozen predicted trace."""

    if timeline.period != episode.period or timeline.persona != episode.persona:
        raise ValueError("episode/timeline identity mismatch")
    if timeline.source_sha256 != episode.timeline_sha256:
        raise ValueError("episode/timeline digest mismatch")
    question_text = _required(question_record.get("question"), "question")
    obligation_id = f"predicted-obligation:{episode.question_id}"
    bootstrap_dependency = f"predicted-bootstrap:{episode.question_id}"
    protocol = RouterReturnProtocol(
        task_id=episode.episode_id,
        task_description=question_text,
    )
    for principal_id in WORKER_IDS:
        protocol.add_member(principal_id, f"role:{principal_id}")
    protocol.install_plan(
        _make_dag(
            episode_id=episode.episode_id,
            question=question_text,
            obligation_id=obligation_id,
        ),
        (
            RouterObligation(
                obligation_id=obligation_id,
                description=trace.router_plan.obligation,
                owner_principal_id=WORKER_IDS[0],
                required_dependency_ids=(bootstrap_dependency,),
                critical_dependency_ids=(bootstrap_dependency,),
            ),
        ),
    )

    pre_facts = [
        item for item in trace.facts if item.phase == "predeparture"
    ]
    active_pre = active_facts_at_departure(pre_facts)
    build_fallback_used = False
    if not active_pre:
        if not pre_facts:
            raise ValueError(
                "predicted predeparture extraction produced no source fact"
            )
        source = max(
            pre_facts, key=lambda item: (item.session_id, item.fact_id)
        )
        fallback_id = "pf_" + canonical_sha256(
            {
                "source_fact_id": source.fact_id,
                "state_key": "workspace-continuity",
                "deterministic_build_fallback": True,
            }
        )[:24]
        active_pre = (
            replace(
                source,
                fact_id=fallback_id,
                state_key=f"workspace-continuity:{source.session_id}",
                statement=source.evidence_quote,
                operation="add",
                confidence=0.0,
            ),
        )
        build_fallback_used = True
    protocol.start_node("memora:predicted:prefix", WORKER_IDS[0])
    protocol.complete_node(
        "memora:predicted:prefix",
        WORKER_IDS[0],
        "Predicted predeparture state extracted from actor-visible sessions.",
    )
    current_by_key: dict[str, PredictedStateFact] = {}
    item_by_fact_id: dict[str, str] = {}
    for fact in active_pre:
        normalized = replace(fact, worker_id=WORKER_IDS[0])
        protocol.add_workstate_item(_item(normalized, obligation_id=obligation_id))
        current_by_key[fact.state_key] = normalized
        item_by_fact_id[fact.fact_id] = fact.item_id
    protocol.depart(WORKER_IDS[0])

    for worker_id in WORKER_IDS[1:]:
        node_id = f"memora:predicted:absence:{worker_id}"
        protocol.start_node(node_id, worker_id)
        protocol.complete_node(
            node_id,
            worker_id,
            "Predicted absence state extracted from actor-visible sessions.",
        )

    stale_old: list[str] = []
    mutation_items: list[str] = []
    absence_facts = sorted(
        (item for item in trace.facts if item.phase == "absence"),
        key=lambda item: (item.session_id, item.fact_id),
    )
    for fact in absence_facts:
        prior = current_by_key.get(fact.state_key)
        prior_item_id = prior.item_id if prior is not None else None
        if fact.operation == "delete":
            if prior_item_id is not None:
                protocol.revoke_workstate_item(
                    actor_principal_id=fact.worker_id,
                    item_id=prior_item_id,
                    reason="Predicted deletion from actor-visible raw session.",
                )
                stale_old.append(prior_item_id)
            protocol.add_workstate_item(
                _item(fact, obligation_id=obligation_id, version=1)
            )
            current_by_key[fact.state_key] = fact
        elif prior_item_id is not None:
            replacement_version = (
                protocol.workstate[prior_item_id].version + 1
            )
            protocol.supersede_workstate_item(
                actor_principal_id=fact.worker_id,
                old_item_id=prior_item_id,
                replacement_item=_item(
                    fact,
                    obligation_id=obligation_id,
                    supersedes_item_id=prior_item_id,
                    version=replacement_version,
                ),
            )
            stale_old.append(prior_item_id)
            current_by_key[fact.state_key] = fact
        else:
            protocol.add_workstate_item(_item(fact, obligation_id=obligation_id))
            current_by_key[fact.state_key] = fact
        item_by_fact_id[fact.fact_id] = fact.item_id
        mutation_items.append(fact.item_id)

    current_fact_ids = {item.fact_id for item in current_by_key.values()}
    selected_fact_ids = tuple(
        item
        for item in trace.dependency_plan.selected_fact_ids
        if item in current_fact_ids
    )
    critical_fact_ids = tuple(
        item
        for item in trace.dependency_plan.critical_fact_ids
        if item in selected_fact_ids
    )
    if not selected_fact_ids:
        raise ValueError("predicted Router selected no currently active fact")
    required_dependencies = tuple(
        next(
            item.dependency_id
            for item in current_by_key.values()
            if item.fact_id == fact_id
        )
        for fact_id in selected_fact_ids
    )
    critical_dependencies = tuple(
        next(
            item.dependency_id
            for item in current_by_key.values()
            if item.fact_id == fact_id
        )
        for fact_id in critical_fact_ids
    )
    protocol.refine_obligation_dependencies(
        actor_principal_id="router",
        obligation_id=obligation_id,
        required_dependency_ids=required_dependencies,
        critical_dependency_ids=critical_dependencies or required_dependencies[:1],
        reason="Question-conditioned dependencies predicted from raw sessions.",
    )

    checkpoint = protocol.freeze_post_absence_checkpoint(WORKER_IDS[0])
    trace_compilation = compile_router_trace(
        protocol,
        checkpoint,
        governor,
        purpose=purpose,
        max_items=max(4, len(required_dependencies) + 2),
        max_chars=max_chars,
    )
    ablation_specs = {
        RouterReturnArm.STATIC_COMPACT.value: (
            RouterReturnArm.STATIC_COMPACT,
            ReturnAblationMode.STATIC_COMPACT,
            "current_active",
        ),
        RouterReturnArm.VALIDITY_FILTER_ONLY.value: (
            RouterReturnArm.VALIDITY_FILTER_ONLY,
            ReturnAblationMode.VALIDITY_FILTER_ONLY,
            "return_candidates",
        ),
    }
    requested_ablation_arms = tuple(
        arm for arm in ablation_arms if arm in ablation_specs
    )
    ablation_compilations: dict[str, RouterTraceCompilation] = {}
    fork_by_kind = {fork.arm: fork for fork in trace_compilation.forks}
    compact_only_audit: dict[str, object] | None = None
    if RouterReturnArm.COMPACT_ONLY.value in ablation_arms:
        compact_only_budget = len(trace_compilation.selected_router_item_ids)
        compact_only_ids = _compact_only_item_ids(
            checkpoint,
            candidate_item_ids=trace_compilation.router_candidate_item_ids,
            question_text=question_text,
            max_items=compact_only_budget,
        )
        fork_by_kind[RouterReturnArm.COMPACT_ONLY] = replace(
            fork_by_kind[RouterReturnArm.COMPACT_ONLY],
            inherited_item_ids=compact_only_ids,
            candidate_item_ids=trace_compilation.router_candidate_item_ids,
            compiler_required=False,
        )
        compact_only_audit = {
            "schema_version": "memora_compact_only_selection_v1",
            "selection_policy": (
                "question_token_cosine_then_version_then_item_id"
            ),
            "candidate_pool": "same_raw_return_candidates_as_trace",
            "candidate_count": len(trace_compilation.router_candidate_item_ids),
            "item_budget": compact_only_budget,
            "selected_item_ids": list(compact_only_ids),
            "freshness_filter": False,
            "supersede_revoke_filter": False,
            "provenance_verification": False,
            "obligation_aware_selection": False,
            "signed_view": False,
            "admission_revalidation": False,
            "fail_closed_fallback": False,
        }
    for arm in requested_ablation_arms:
        arm_kind, mode, candidate_scope = ablation_specs[arm]
        compilation = compile_router_trace(
            protocol,
            checkpoint,
            governor,
            purpose=purpose,
            max_items=max(4, len(required_dependencies) + 2),
            max_chars=max_chars,
            mode=mode,
            candidate_scope=candidate_scope,
        )
        if (
            compilation.compilation.view is None
            or compilation.admission is None
        ):
            raise RuntimeError(f"{arm} failed to compile an admitted view")
        selected_fork = next(
            fork
            for fork in compilation.forks
            if fork.arm is RouterReturnArm.TRACE
        )
        fork_by_kind[arm_kind] = replace(
            selected_fork,
            arm=arm_kind,
            candidate_item_ids=compilation.router_candidate_item_ids,
        )
        ablation_compilations[arm] = compilation
    forks = tuple(fork_by_kind[fork.arm] for fork in trace_compilation.forks)
    protocol.start_return(WORKER_IDS[0])
    protocol.finish_return(
        WORKER_IDS[0],
        admitted=trace_compilation.compilation.view is not None,
        reason="Predicted RESUME view compiled from actor-visible state only.",
    )
    actor_views = {
        fork.arm.value: render_router_fork(fork, checkpoint) for fork in forks
    }
    selected_current_items = tuple(
        item_by_fact_id[item] for item in selected_fact_ids
    )
    selected_pre = tuple(
        item.item_id
        for item in current_by_key.values()
        if item.fact_id in selected_fact_ids and item.phase == "predeparture"
    )
    selected_absence = tuple(
        item.item_id
        for item in current_by_key.values()
        if item.fact_id in selected_fact_ids and item.phase == "absence"
    )
    deterministic_fallback_fact_ids = {
        fact.fact_id
        for batch in trace.extraction_batches
        if batch.deterministic_fallback
        for fact in batch.facts
    }
    if build_fallback_used:
        deterministic_fallback_fact_ids.update(
            item.fact_id for item in active_pre
        )
    prediction_degraded = build_fallback_used or any(
        batch.deterministic_fallback for batch in trace.extraction_batches
    )
    selected_fallback_count = len(
        deterministic_fallback_fact_ids & set(selected_fact_ids)
    )
    fork_by_arm = {item.arm: item for item in forks}
    restore = set(
        fork_by_arm[RouterReturnArm.RESTORE_OLD].inherited_item_ids
    )
    resume = set(fork_by_arm[RouterReturnArm.TRACE].inherited_item_ids)
    static = set(fork_by_arm[RouterReturnArm.STATIC].inherited_item_ids)
    audit = {
        "schema_version": "memora_resume_predicted_audit_v1",
        "mode": "predicted",
        "official_evidence_runtime_reads": 0,
        "privileged_session_metadata_runtime_reads": 0,
        "same_checkpoint_all_arms": len(
            {item.checkpoint_sha256 for item in forks}
        )
        == 1,
        "predicted_fact_count": len(trace.facts),
        "predicted_stale_count": len(set(stale_old)),
        "selected_dependency_count": len(required_dependencies),
        "selector_budget": (
            trace.selector_budget.record()
            if trace.selector_budget is not None
            else None
        ),
        "selector_budget_hit": (
            trace.selector_budget is not None
            and len(required_dependencies)
            >= trace.selector_budget.effective_budget
        ),
        "selector_candidate_coverage": (
            len(required_dependencies)
            / trace.selector_budget.current_fact_count
            if trace.selector_budget is not None
            and trace.selector_budget.current_fact_count
            else 0.0
        ),
        "selected_predeparture_count": len(selected_pre),
        "selected_absence_count": len(selected_absence),
        "deterministic_predeparture_fallback_used": bool(
            deterministic_fallback_fact_ids
        ),
        "prediction_degraded": prediction_degraded,
        "selected_deterministic_fallback_count": selected_fallback_count,
        "static_contains_selected_current": set(selected_current_items).issubset(
            static
        ),
        "restore_contains_selected_predeparture": set(selected_pre).issubset(
            restore
        ),
        "restore_misses_selected_absence": set(selected_absence).isdisjoint(
            restore
        ),
        "resume_contains_selected_current": set(selected_current_items).issubset(
            resume
        ),
        "resume_rejects_predicted_stale": set(stale_old).isdisjoint(resume),
        "restore_resume_view_difference": restore != resume,
        "critical_dependency_recall": (
            trace_compilation.compilation.selection.critical_dependency_recall
        ),
    }
    if compact_only_audit is not None:
        audit["compact_only"] = compact_only_audit
    audit["event_eligible"] = bool(
        audit["same_checkpoint_all_arms"]
        and not audit["prediction_degraded"]
        and int(audit["selected_predeparture_count"]) > 0
        and int(audit["selected_absence_count"]) > 0
        and int(audit["predicted_stale_count"]) > 0
        and audit["resume_contains_selected_current"]
        and audit["resume_rejects_predicted_stale"]
        and audit["restore_resume_view_difference"]
        and float(audit["critical_dependency_recall"]) >= 0.999999
    )
    return PredictedReturnForkBundle(
        episode=episode,
        protocol=protocol,
        checkpoint=checkpoint,
        trace_compilation=trace_compilation,
        forks=forks,
        actor_views=actor_views,
        trace=trace,
        selected_current_item_ids=selected_current_items,
        selected_predeparture_item_ids=selected_pre,
        selected_absence_item_ids=selected_absence,
        stale_old_item_ids=tuple(dict.fromkeys(stale_old)),
        mutation_item_ids=tuple(dict.fromkeys(mutation_items)),
        ablation_compilations=ablation_compilations,
        audit=audit,
    )


__all__ = [
    "PredictedDependencyPlan",
    "PredictedExtractionBatch",
    "PredictedReturnForkBundle",
    "PredictedRouterPlan",
    "PredictedSelectorBudget",
    "PredictedStateFact",
    "PredictedTrace",
    "active_facts_at_departure",
    "actor_session_records",
    "build_predicted_return_forks",
    "dependency_prompt",
    "dynamic_selector_budget",
    "extract_json",
    "extraction_prompt",
    "parse_dependency_plan",
    "parse_extraction",
    "parse_router_plan",
    "plan_router_prompt",
    "retrieve_actor_sessions",
    "safe_question_record",
    "shard_actor_sessions",
]
