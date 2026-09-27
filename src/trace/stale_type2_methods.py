"""Actor-only method layer for the official STALE Type-II MAS adapter.

All extraction and governance interfaces consume only the original actor
sessions plus lifecycle metadata.  Oracle fields never enter these prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

from .return_methods.base import ReturnCandidate
from .return_methods.lifecycle import (
    KMU_OP_ARM,
    TEMPORAL_LWW_ARM,
    compile_indexed_kmu_operations,
    compile_kmu_operations,
    compile_temporal_lww,
    indexed_kmu_operation_prompt,
)
from .return_methods.semantic_state import (
    CUPMEM_ARM,
    compile_cupmem_adjudication,
    cupmem_adjudication_prompt as shared_cupmem_adjudication_prompt,
)
from .return_methods.temporal import MEMSTRATA_ARM, compile_memstrata
from .return_methods.transactional import MEMTX_ARM, compile_memtx
from .stale_type2_return import canonical_sha256


STATIC_ARM = "static_no_churn"
RESET_ARM = "reset"
RESTORE_ARM = "restore_old"
CHECKPOINT_REPLAY_ARM = "checkpoint_replay"
TRACE_ARM = "trace"
TRACE_WITHOUT_FRESHNESS_ARM = "trace_without_freshness"
TRACE_WITHOUT_PROVENANCE_ARM = "trace_without_provenance"
TRACE_WITHOUT_FALLBACK_ARM = "trace_without_fallback"
VALIDITY_FILTER_ONLY_ARM = "validity_filter_only"
STALE_TRACE_ABLATION_ARMS = (
    TRACE_WITHOUT_FRESHNESS_ARM, TRACE_WITHOUT_PROVENANCE_ARM,
    TRACE_WITHOUT_FALLBACK_ARM, VALIDITY_FILTER_ONLY_ARM,
)
STALE_SIX_ARMS = (
    STATIC_ARM,
    RESET_ARM,
    RESTORE_ARM,
    CHECKPOINT_REPLAY_ARM,
    CUPMEM_ARM,
    TRACE_ARM,
)
STALE_ALL_ARMS = (
    *STALE_SIX_ARMS,
    TEMPORAL_LWW_ARM,
    KMU_OP_ARM,
    MEMSTRATA_ARM,
    MEMTX_ARM,
    *STALE_TRACE_ABLATION_ARMS,
)
STALE_FOUR_ARMS = (
    STATIC_ARM,
    RESTORE_ARM,
    RESET_ARM,
    TRACE_ARM,
)

_OPERATIONS = frozenset(("add", "update", "delete"))
_TRACE_STATUSES = frozenset(("ADMIT", "SUPERSEDED", "QUARANTINE"))
SESSION_OBSERVATION_KEY_PREFIX = "session_observation/"
SESSION_TRANSPORT_PAGE_KEY = "_transport_page"


def _required(value: object, name: str, *, max_chars: int | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    if max_chars is not None and len(text) > max_chars:
        raise ValueError(f"{name} exceeds {max_chars} characters")
    return text


def _json_object(text: str) -> Mapping[str, object]:
    candidate = re.sub(r"^```(?:json)?\s*", "", text.strip(), count=1)
    candidate = re.sub(r"\s*```$", "", candidate, count=1)
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
    value = json.loads(candidate)
    if not isinstance(value, Mapping):
        raise TypeError("model output must be a JSON object")
    return value


@dataclass(frozen=True)
class StaleMemoryItem:
    memory_id: str
    agent_id: str
    session_index: int
    timestamp: str
    state_key: str
    statement: str
    operation: str
    evidence_quote: str
    source_session_sha256: str

    def __post_init__(self) -> None:
        _required(self.memory_id, "memory_id")
        _required(self.agent_id, "agent_id")
        if self.session_index < 0:
            raise ValueError("session_index must be non-negative")
        _required(self.timestamp, "timestamp")
        _required(self.state_key, "state_key", max_chars=160)
        _required(self.statement, "statement", max_chars=1200)
        if self.operation not in _OPERATIONS:
            raise ValueError(f"unsupported memory operation: {self.operation}")
        _required(self.evidence_quote, "evidence_quote", max_chars=8000)
        if len(self.source_session_sha256) != 64:
            raise ValueError("source_session_sha256 must be a SHA-256")
        bytes.fromhex(self.source_session_sha256)

    def record(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "session_index": self.session_index,
            "timestamp": self.timestamp,
            "state_key": self.state_key,
            "statement": self.statement,
            "operation": self.operation,
            "evidence_quote": self.evidence_quote,
            "source_session_sha256": self.source_session_sha256,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "StaleMemoryItem":
        return cls(
            memory_id=_required(value.get("memory_id"), "memory_id"),
            agent_id=_required(value.get("agent_id"), "agent_id"),
            session_index=int(value["session_index"]),
            timestamp=_required(value.get("timestamp"), "timestamp"),
            state_key=_required(value.get("state_key"), "state_key"),
            statement=_required(value.get("statement"), "statement"),
            operation=_required(value.get("operation"), "operation").lower(),
            evidence_quote=_required(value.get("evidence_quote"), "evidence_quote"),
            source_session_sha256=_required(
                value.get("source_session_sha256"), "source_session_sha256"
            ),
        )

    def compact_record(self, *, quote_chars: int = 500) -> dict[str, object]:
        quote = self.evidence_quote
        if len(quote) > quote_chars:
            quote = quote[:quote_chars] + "…"
        return {
            "memory_id": self.memory_id,
            "source_agent": self.agent_id,
            "session_index": self.session_index,
            "timestamp": self.timestamp,
            "state_key": self.state_key,
            "statement": self.statement,
            "operation": self.operation,
            "evidence_quote": quote,
        }


@dataclass(frozen=True)
class StaleArmSelection:
    arm: str
    candidate_item_ids: tuple[str, ...]
    selected_item_ids: tuple[str, ...]
    decisions: tuple[Mapping[str, object], ...] = ()
    parse_error: str | None = None

    def __post_init__(self) -> None:
        if self.arm not in STALE_ALL_ARMS:
            raise ValueError(f"unsupported STALE arm: {self.arm}")
        if len(self.candidate_item_ids) != len(set(self.candidate_item_ids)):
            raise ValueError("candidate_item_ids must be unique")
        if len(self.selected_item_ids) != len(set(self.selected_item_ids)):
            raise ValueError("selected_item_ids must be unique")
        if not set(self.selected_item_ids).issubset(self.candidate_item_ids):
            raise ValueError("selected items must be candidates")

    def record(self) -> dict[str, object]:
        return {
            "schema_version": "stale_type2_arm_selection_v1",
            "arm": self.arm,
            "candidate_item_ids": list(self.candidate_item_ids),
            "selected_item_ids": list(self.selected_item_ids),
            "decisions": [dict(item) for item in self.decisions],
            "parse_error": self.parse_error,
        }


def session_observation_items(
    *,
    uid: str,
    agent_id: str,
    session: Mapping[str, object],
) -> tuple[StaleMemoryItem, ...]:
    """Create one provenance-bound, outcome-blind anchor for a session.

    Keeping exactly one raw anchor per official session gives every session the
    same write opportunity and also keeps lifecycle compilation within the
    model context window.  The anchor concatenates all user turns in their
    original order; it is not selected using a query or benchmark annotation.
    """

    session_index = int(session["session_index"])
    timestamp = _required(session.get("timestamp"), "timestamp")
    source_digest = canonical_sha256(session)
    messages = session.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("session messages must be an array")
    visible = [
        str(message.get("content") or "").strip()
        for message in messages
        if isinstance(message, Mapping)
        and str(message.get("role") or "") == "user"
        and str(message.get("content") or "").strip()
    ]
    if not visible:
        visible = [
            str(message.get("content") or "").strip()
            for message in messages
            if isinstance(message, Mapping)
            and str(message.get("content") or "").strip()
        ]
    if not visible:
        raise ValueError(f"session {session_index} has no visible message")
    content = "\n".join(visible)
    quote = content[:8000]
    state_key = f"{SESSION_OBSERVATION_KEY_PREFIX}all_user_turns"
    statement = ("Session observation: " + content)[:1200]
    memory_id = "sto_" + hashlib.sha256(
        json.dumps(
            {
                "uid": uid,
                "agent_id": agent_id,
                "session_index": session_index,
                "source_session_sha256": source_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    return (
        StaleMemoryItem(
            memory_id=memory_id,
            agent_id=agent_id,
            session_index=session_index,
            timestamp=timestamp,
            state_key=state_key,
            statement=statement,
            operation="add",
            evidence_quote=quote,
            source_session_sha256=source_digest,
        ),
    )


def chunk_actor_sessions(
    sessions: Sequence[Mapping[str, object]],
    *,
    max_chars: int,
    max_sessions: int | None = None,
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    """Losslessly group whole sessions under size and count budgets."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if max_sessions is not None and max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    chunks: list[tuple[Mapping[str, object], ...]] = []
    current: list[Mapping[str, object]] = []
    current_chars = 2
    seen: list[int] = []
    for session in sessions:
        encoded = json.dumps(session, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > max_chars:
            raise ValueError(
                f"official session {session.get('session_index')} exceeds chunk budget"
            )
        extra = len(encoded) + (1 if current else 0)
        if current and (
            current_chars + extra > max_chars
            or (max_sessions is not None and len(current) >= max_sessions)
        ):
            chunks.append(tuple(current))
            current = []
            current_chars = 2
        current.append(session)
        current_chars += len(encoded) + (1 if len(current) > 1 else 0)
        seen.append(int(session["session_index"]))
    if current:
        chunks.append(tuple(current))
    reconstructed = [
        int(session["session_index"])
        for chunk in chunks
        for session in chunk
    ]
    if reconstructed != seen:
        raise RuntimeError("actor session chunking changed session order")
    return tuple(chunks)


def paginate_actor_sessions(
    sessions: Sequence[Mapping[str, object]],
    *,
    max_chars: int,
) -> tuple[Mapping[str, object], ...]:
    """Losslessly page oversized actor-visible sessions for transport.

    Official sessions that already fit are returned unchanged.  An oversized
    session is projected into deterministic message fragments carrying the
    same official session index, timestamp, and full-session receipt.  The
    projection is query-independent and contains no benchmark annotations.
    Concatenating fragments by ``message_index`` and ``fragment_index``
    reconstructs every original message content exactly.
    """

    if max_chars < 256:
        raise ValueError("max_chars is too small for session transport metadata")

    def encoded_size(value: Mapping[str, object]) -> int:
        return len(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )

    pages: list[Mapping[str, object]] = []
    for session in sessions:
        if encoded_size(session) <= max_chars:
            pages.append(session)
            continue
        session_index = int(session["session_index"])
        timestamp = _required(session.get("timestamp"), "timestamp")
        messages = session.get("messages")
        if not isinstance(messages, Sequence) or isinstance(
            messages, (str, bytes)
        ):
            raise TypeError("session messages must be an array")
        source_digest = canonical_sha256(session)
        for message_index, raw_message in enumerate(messages):
            if not isinstance(raw_message, Mapping):
                raise TypeError("every session message must be an object")
            role = str(raw_message.get("role") or "")
            content = str(raw_message.get("content") or "")
            cursor = 0
            fragment_index = 0
            emitted_empty = False
            while cursor < len(content) or not emitted_empty:
                def page(fragment: str) -> dict[str, object]:
                    return {
                        "session_index": session_index,
                        "timestamp": timestamp,
                        "messages": [{"role": role, "content": fragment}],
                        SESSION_TRANSPORT_PAGE_KEY: {
                            "schema_version": "stale_session_transport_page_v1",
                            "source_session_sha256": source_digest,
                            "message_index": message_index,
                            "fragment_index": fragment_index,
                        },
                    }

                if not content:
                    candidate = page("")
                    if encoded_size(candidate) > max_chars:
                        raise ValueError(
                            f"session {session_index} transport metadata exceeds chunk budget"
                        )
                    pages.append(candidate)
                    emitted_empty = True
                    break

                low, high = 1, len(content) - cursor
                best = 0
                while low <= high:
                    middle = (low + high) // 2
                    candidate = page(content[cursor : cursor + middle])
                    if encoded_size(candidate) <= max_chars:
                        best = middle
                        low = middle + 1
                    else:
                        high = middle - 1
                if best < 1:
                    raise ValueError(
                        f"session {session_index} transport metadata exceeds chunk budget"
                    )
                pages.append(page(content[cursor : cursor + best]))
                cursor += best
                fragment_index += 1
                emitted_empty = True
    return tuple(pages)


def extraction_prompt(
    *,
    agent_id: str,
    sessions: Sequence[Mapping[str, object]],
    max_items: int,
) -> str:
    if max_items < 1:
        raise ValueError("max_items must be positive")
    payload = {
        "principal_id": agent_id,
        "sessions": list(sessions),
    }
    return (
        "Build a query-independent long-term state-memory report from these "
        "original sessions. Extract only durable user facts, preferences, "
        "constraints, plans, capabilities, and state changes that could affect "
        "future assistance. Preserve indirect changes even when they do not "
        "explicitly contradict an earlier fact. Do not answer a future query, "
        "guess missing facts, or label anything using benchmark terminology. "
        f"Return one JSON object with key memories containing at most {max_items} "
        "rows. Each row must contain session_index (integer), state_key (short "
        "stable semantic slot), statement (self-contained current-or-historical "
        "observation), operation (add, update, or delete), and evidence_quote "
        "copied verbatim from one message in that session. Prefer state changes "
        "over conversational details. Return {\"memories\":[]} if none exist.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def parse_extraction(
    text: str,
    *,
    uid: str,
    agent_id: str,
    sessions: Sequence[Mapping[str, object]],
    max_items: int,
    skip_ungrounded: bool = False,
) -> tuple[StaleMemoryItem, ...]:
    payload = _json_object(text)
    rows = payload.get("memories")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("extraction memories must be an array")
    if len(rows) > max_items:
        raise ValueError("extraction exceeded max_items")
    sessions_by_index: dict[int, list[Mapping[str, object]]] = {}
    for item in sessions:
        sessions_by_index.setdefault(int(item["session_index"]), []).append(item)
    result: list[StaleMemoryItem] = []
    seen: set[tuple[int, str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("every extracted memory must be an object")
        session_index = int(raw["session_index"])
        sources = sessions_by_index.get(session_index, ())
        if not sources:
            raise ValueError(f"unassigned source session: {session_index}")
        state_key = _required(raw.get("state_key"), "state_key", max_chars=160)
        statement = _required(raw.get("statement"), "statement", max_chars=1200)
        operation = _required(raw.get("operation"), "operation").lower()
        if operation not in _OPERATIONS:
            raise ValueError(f"unsupported operation: {operation}")
        quote = _required(raw.get("evidence_quote"), "evidence_quote", max_chars=8000)
        grounded_sources = [
            source
            for source in sources
            if any(
                quote in str(message.get("content") or "")
                for message in source.get("messages", ())
                if isinstance(message, Mapping)
            )
        ]
        if not grounded_sources:
            if skip_ungrounded:
                continue
            raise ValueError(
                f"evidence quote is not verbatim in session {session_index}"
            )
        source = grounded_sources[0]
        dedup = (session_index, state_key.casefold(), statement.casefold())
        if dedup in seen:
            continue
        seen.add(dedup)
        transport = source.get(SESSION_TRANSPORT_PAGE_KEY)
        source_digest = (
            _required(
                transport.get("source_session_sha256"),
                "source_session_sha256",
            )
            if isinstance(transport, Mapping)
            else canonical_sha256(source)
        )
        if len(source_digest) != 64:
            raise ValueError("source_session_sha256 must be a SHA-256")
        bytes.fromhex(source_digest)
        memory_id = "stm_" + hashlib.sha256(
            json.dumps(
                {
                    "uid": uid,
                    "agent_id": agent_id,
                    "session_index": session_index,
                    "state_key": state_key,
                    "statement": statement,
                    "evidence_quote": quote,
                    "source_session_sha256": source_digest,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]
        result.append(
            StaleMemoryItem(
                memory_id=memory_id,
                agent_id=agent_id,
                session_index=session_index,
                timestamp=_required(source.get("timestamp"), "timestamp"),
                state_key=state_key,
                statement=statement,
                operation=operation,
                evidence_quote=quote,
                source_session_sha256=source_digest,
            )
        )
    return tuple(result)


def deterministic_arm_selections(
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
) -> dict[str, StaleArmSelection]:
    ordered = tuple(sorted(items, key=lambda item: (item.session_index, item.memory_id)))
    all_ids = tuple(item.memory_id for item in ordered)
    old_ids = tuple(
        item.memory_id for item in ordered if item.agent_id == returning_agent_id
    )
    replay_state: dict[str, StaleMemoryItem] = {}
    replay_decisions: list[Mapping[str, object]] = []
    for item in ordered:
        key = item.state_key.strip().casefold()
        previous = replay_state.get(key)
        if item.operation == "delete":
            replay_state.pop(key, None)
            action = "delete"
        else:
            replay_state[key] = item
            action = "replace" if previous is not None else "add"
        replay_decisions.append(
            {
                "memory_id": item.memory_id,
                "state_key": item.state_key,
                "action": action,
                "replaced_memory_id": (
                    previous.memory_id if previous is not None else None
                ),
            }
        )
    replay_ids = tuple(
        item.memory_id
        for item in ordered
        if replay_state.get(item.state_key.strip().casefold()) == item
    )
    candidates = stale_return_candidates(
        ordered,
        returning_agent_id=returning_agent_id,
    )
    temporal = compile_temporal_lww(candidates)
    memstrata = compile_memstrata(candidates)
    memtx = compile_memtx(candidates)
    return {
        STATIC_ARM: StaleArmSelection(STATIC_ARM, all_ids, all_ids),
        RESET_ARM: StaleArmSelection(RESET_ARM, all_ids, ()),
        RESTORE_ARM: StaleArmSelection(RESTORE_ARM, all_ids, old_ids),
        CHECKPOINT_REPLAY_ARM: StaleArmSelection(
            CHECKPOINT_REPLAY_ARM,
            all_ids,
            replay_ids,
            decisions=tuple(replay_decisions),
        ),
        TEMPORAL_LWW_ARM: StaleArmSelection(
            TEMPORAL_LWW_ARM,
            all_ids,
            temporal.selected_ids,
            decisions=temporal.decisions,
        ),
        MEMSTRATA_ARM: StaleArmSelection(
            MEMSTRATA_ARM,
            all_ids,
            memstrata.selected_ids,
            decisions=memstrata.decisions,
        ),
        MEMTX_ARM: StaleArmSelection(
            MEMTX_ARM,
            all_ids,
            memtx.selected_ids,
            decisions=memtx.decisions,
        ),
    }


def stale_return_candidates(
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
) -> tuple[ReturnCandidate, ...]:
    """Normalize STALE memory rows for shared lifecycle baselines."""

    result = []
    for item in items:
        state_key = item.state_key
        # Raw session anchors do not claim semantic equivalence. Treating all
        # such anchors as one slot would give LWW an artificial oracle-like
        # collapse to the last session.
        if state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX):
            state_key = f"{state_key}{item.memory_id}"
        result.append(
            ReturnCandidate(
                candidate_id=item.memory_id,
                state_key=state_key,
                text=item.statement + "\nEvidence: " + item.evidence_quote,
                logical_time=item.session_index,
                version=1,
                phase=(
                    "predeparture"
                    if item.agent_id == returning_agent_id
                    else "absence"
                ),
                operation=item.operation,
                source_id=item.agent_id,
                semantic_value=item.statement,
                entity="stale_profile",
                attribute=item.state_key,
                valid_from=item.session_index,
            )
        )
    return tuple(result)


def kmu_prompt(
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
    statement_chars: int = 320,
    evidence_chars: int = 200,
) -> str:
    """Build a bounded, query-independent KMU replay prompt.

    The full provenance records remain in the audit store.  KMU sees a
    deterministic compact projection so a long checkpoint cannot make even a
    single absence candidate exceed the context budget.
    """

    if statement_chars < 1:
        raise ValueError("statement_chars must be positive")
    if evidence_chars < 0:
        raise ValueError("evidence_chars must be non-negative")
    by_id = {item.memory_id: item for item in items}
    candidates = []
    for candidate in stale_return_candidates(
        items,
        returning_agent_id=returning_agent_id,
    ):
        item = by_id[candidate.candidate_id]
        statement = item.statement[:statement_chars]
        evidence = item.evidence_quote[:evidence_chars]
        text = statement
        if evidence:
            text += "\nEvidence excerpt: " + evidence
        candidates.append(
            ReturnCandidate(
                candidate_id=candidate.candidate_id,
                state_key=candidate.state_key,
                text=text,
                logical_time=candidate.logical_time,
                version=candidate.version,
                phase=candidate.phase,
                operation=candidate.operation,
                source_id=candidate.source_id,
            )
        )
    return indexed_kmu_operation_prompt(
        tuple(candidates)
    )


def parse_kmu_selection(
    text: str,
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
) -> StaleArmSelection:
    compilation = compile_kmu_operations(
        stale_return_candidates(
            items,
            returning_agent_id=returning_agent_id,
        ),
        text,
    )
    if compilation.parse_error is not None:
        raise ValueError(compilation.parse_error)
    return StaleArmSelection(
        arm=KMU_OP_ARM,
        candidate_item_ids=compilation.candidate_ids,
        selected_item_ids=compilation.selected_ids,
        decisions=compilation.decisions,
        parse_error=compilation.parse_error,
    )


def parse_indexed_kmu_selection(
    text: str,
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
) -> StaleArmSelection:
    """Compile local indices through KMU's native operation replay rules."""

    compilation = compile_indexed_kmu_operations(
        stale_return_candidates(
            items,
            returning_agent_id=returning_agent_id,
        ),
        text,
    )
    if compilation.parse_error is not None:
        raise ValueError(compilation.parse_error)
    return StaleArmSelection(
        arm=KMU_OP_ARM,
        candidate_item_ids=compilation.candidate_ids,
        selected_item_ids=compilation.selected_ids,
        decisions=compilation.decisions,
        parse_error=compilation.parse_error,
    )


def chunk_kmu_candidates(
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
    max_prompt_chars: int,
    max_absence_items_per_chunk: int = 4,
) -> tuple[tuple[StaleMemoryItem, ...], ...]:
    """Bound KMU calls without dropping or label-selecting candidates.

    Every chunk repeats the returning agent's predeparture memory and carries
    a disjoint, chronological slice of absence candidates.  This preserves
    each absence candidate's opportunity to update the checkpoint while
    preventing a single unbounded request from exceeding the actor context.
    """

    if max_prompt_chars < 1:
        raise ValueError("max_prompt_chars must be positive")
    if max_absence_items_per_chunk < 1:
        raise ValueError("max_absence_items_per_chunk must be positive")
    ordered = tuple(items)

    def order_key(item: StaleMemoryItem) -> tuple[int, str]:
        return item.session_index, item.memory_id
    predeparture = tuple(
        sorted(
            (item for item in ordered if item.agent_id == returning_agent_id),
            key=order_key,
        )
    )
    absence = tuple(
        sorted(
            (item for item in ordered if item.agent_id != returning_agent_id),
            key=order_key,
        )
    )
    if not absence:
        return ()
    if len(kmu_prompt(predeparture, returning_agent_id=returning_agent_id)) > (
        max_prompt_chars
    ):
        raise ValueError("KMU predeparture memory exceeds prompt budget")

    result: list[tuple[StaleMemoryItem, ...]] = []
    batch: list[StaleMemoryItem] = []
    for item in absence:
        candidate = (*predeparture, *batch, item)
        if batch and (
            len(batch) >= max_absence_items_per_chunk
            or len(
                kmu_prompt(candidate, returning_agent_id=returning_agent_id)
            )
            > max_prompt_chars
        ):
            result.append((*predeparture, *batch))
            batch = [item]
            candidate = (*predeparture, item)
        else:
            batch.append(item)
        if (
            len(kmu_prompt(candidate, returning_agent_id=returning_agent_id))
            > max_prompt_chars
        ):
            raise ValueError(
                f"KMU candidate exceeds prompt budget: {item.memory_id}"
            )
    if batch:
        result.append((*predeparture, *batch))
    return tuple(result)


def kmu_chunk_index_contract(
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return local candidate, incoming, and legal non-self target indices."""

    candidate_indices = tuple(range(len(items)))
    absence_indices = tuple(
        index
        for index, item in enumerate(items)
        if item.agent_id != returning_agent_id
    )
    if len(absence_indices) != 1:
        raise ValueError("bounded KMU chunk must contain one absence candidate")
    legal_target_indices = tuple(
        index for index in candidate_indices if index not in absence_indices
    )
    return candidate_indices, absence_indices, legal_target_indices


def merge_kmu_chunk_selections(
    items: Sequence[StaleMemoryItem],
    chunks: Sequence[Sequence[StaleMemoryItem]],
    selections: Sequence[StaleArmSelection],
    *,
    returning_agent_id: str,
) -> StaleArmSelection:
    """Merge bounded KMU replays into one full-pool selection."""

    if len(chunks) != len(selections):
        raise ValueError("KMU chunk/selection count mismatch")
    ordered = tuple(items)
    candidate_ids = tuple(item.memory_id for item in ordered)
    predeparture_ids = {
        item.memory_id
        for item in ordered
        if item.agent_id == returning_agent_id
    }
    absence_ids = set(candidate_ids) - predeparture_ids
    kept_predeparture = set(predeparture_ids)
    admitted_absence: set[str] = set()
    seen_absence: set[str] = set()
    decisions: list[Mapping[str, object]] = []
    # Length equality is validated above; avoiding ``zip(strict=...)`` keeps
    # this pure compiler usable in Python 3.9 audit environments as well.
    for chunk, selection in zip(chunks, selections):
        chunk_ids = tuple(item.memory_id for item in chunk)
        if selection.arm != KMU_OP_ARM:
            raise ValueError("non-KMU selection in KMU merge")
        if set(selection.candidate_item_ids) != set(chunk_ids):
            raise ValueError("KMU selection does not match its chunk")
        chunk_absence = set(chunk_ids) - predeparture_ids
        if seen_absence & chunk_absence:
            raise ValueError("KMU absence candidate appeared in multiple chunks")
        seen_absence.update(chunk_absence)
        selected = set(selection.selected_item_ids)
        kept_predeparture.intersection_update(selected)
        admitted_absence.update(selected & chunk_absence)
        decisions.extend(selection.decisions)
    if seen_absence != absence_ids:
        raise ValueError("KMU chunks did not cover every absence candidate")
    selected_ids = kept_predeparture | admitted_absence
    return StaleArmSelection(
        arm=KMU_OP_ARM,
        candidate_item_ids=candidate_ids,
        selected_item_ids=tuple(
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id in selected_ids
        ),
        decisions=tuple(decisions),
    )


def cupmem_prompt(
    items: Sequence[StaleMemoryItem],
    *,
    max_text_chars: int = 1600,
) -> str:
    """Build a compact, query-independent CUPMem write-side request.

    STALE candidates duplicate the same statement in ``text`` and
    ``semantic_value`` and repeat ``state_key`` in ``attribute``.  Keeping all
    three copies made the prompt grow to hundreds of thousands of characters
    without exposing any additional evidence.  The compact projection retains
    the lifecycle fields, source, state key, and a bounded evidence-bearing
    text while removing only those duplicate fields.  Full records remain in
    the immutable extraction cache.
    """

    if max_text_chars < 1:
        raise ValueError("max_text_chars must be positive")
    candidates = tuple(
        replace(
            candidate,
            semantic_value=None,
            entity=None,
            attribute=None,
        )
        for candidate in stale_return_candidates(
            items,
            returning_agent_id="agent1",
        )
    )
    return shared_cupmem_adjudication_prompt(
        candidates,
        max_text_chars=max_text_chars,
    )


def chunk_cupmem_candidates(
    items: Sequence[StaleMemoryItem],
    *,
    returning_agent_id: str,
    max_prompt_chars: int,
    max_text_chars: int = 160,
    max_candidates_per_chunk: int = 112,
    max_absence_items_per_chunk: int = 96,
) -> tuple[tuple[StaleMemoryItem, ...], ...]:
    """Create bounded, label-blind CUPMem pages with cross-phase coverage.

    Every predeparture candidate is paired with every absence candidate in at
    least one CUPMem call.  Candidates are split only by chronological order
    and prompt size; benchmark labels, downstream queries, and evaluator
    fields are unavailable.  If the predeparture checkpoint itself needs more
    than one page, absence candidates are replayed against every page.  This
    preserves CUPMem's write-side old/new adjudication without issuing one
    unbounded full-history request.
    """

    if max_prompt_chars < 1:
        raise ValueError("max_prompt_chars must be positive")
    if max_text_chars < 1:
        raise ValueError("max_text_chars must be positive")
    if max_candidates_per_chunk < 2:
        raise ValueError("max_candidates_per_chunk must be at least two")
    if max_absence_items_per_chunk < 1:
        raise ValueError("max_absence_items_per_chunk must be positive")

    ordered = tuple(
        sorted(items, key=lambda item: (item.session_index, item.memory_id))
    )
    if not ordered:
        return ()
    predeparture = tuple(
        item for item in ordered if item.agent_id == returning_agent_id
    )
    absence = tuple(
        item for item in ordered if item.agent_id != returning_agent_id
    )

    def prompt_chars(rows: Sequence[StaleMemoryItem]) -> int:
        return len(cupmem_prompt(rows, max_text_chars=max_text_chars))

    def paginate(
        rows: Sequence[StaleMemoryItem],
        *,
        prompt_budget: int,
        item_budget: int,
    ) -> tuple[tuple[StaleMemoryItem, ...], ...]:
        pages: list[tuple[StaleMemoryItem, ...]] = []
        current: list[StaleMemoryItem] = []
        for item in rows:
            candidate = (*current, item)
            if current and (
                len(candidate) > item_budget
                or prompt_chars(candidate) > prompt_budget
            ):
                pages.append(tuple(current))
                current = [item]
            else:
                current.append(item)
            if prompt_chars(current) > prompt_budget:
                raise ValueError(
                    "one compact CUPMem candidate exceeds prompt budget: "
                    + item.memory_id
                )
        if current:
            pages.append(tuple(current))
        return tuple(pages)

    # Reserve one quarter of the budget for at least one absence delta and the
    # structured-output contract.  This also prevents a nearly-full checkpoint
    # page from making the first absence candidate impossible to append.
    predeparture_prompt_budget = max(1, (max_prompt_chars * 3) // 4)
    predeparture_pages = paginate(
        predeparture,
        prompt_budget=predeparture_prompt_budget,
        item_budget=max_candidates_per_chunk - 1,
    )
    if not absence:
        return predeparture_pages
    if not predeparture_pages:
        predeparture_pages = ((),)

    chunks: list[tuple[StaleMemoryItem, ...]] = []
    for predeparture_page in predeparture_pages:
        batch: list[StaleMemoryItem] = []
        for item in absence:
            candidate = (*predeparture_page, *batch, item)
            if batch and (
                len(batch) >= max_absence_items_per_chunk
                or len(candidate) > max_candidates_per_chunk
                or prompt_chars(candidate) > max_prompt_chars
            ):
                chunks.append((*predeparture_page, *batch))
                batch = [item]
                candidate = (*predeparture_page, item)
            else:
                batch.append(item)
            if prompt_chars(candidate) > max_prompt_chars:
                raise ValueError(
                    "one CUPMem cross-phase page exceeds prompt budget: "
                    + item.memory_id
                )
        if batch:
            chunks.append((*predeparture_page, *batch))

    expected_predeparture = {item.memory_id for item in predeparture}
    expected_absence = {item.memory_id for item in absence}
    observed = {item.memory_id for chunk in chunks for item in chunk}
    if observed != expected_predeparture | expected_absence:
        raise RuntimeError("CUPMem chunking changed candidate coverage")
    for old_item in predeparture:
        for new_item in absence:
            if not any(
                old_item in chunk and new_item in chunk for chunk in chunks
            ):
                raise RuntimeError("CUPMem chunking omitted a cross-phase pair")
    if any(
        len(chunk) > max_candidates_per_chunk
        or prompt_chars(chunk) > max_prompt_chars
        for chunk in chunks
    ):
        raise RuntimeError("CUPMem chunking exceeded a configured budget")
    return tuple(chunks)


def merge_cupmem_chunk_selections(
    items: Sequence[StaleMemoryItem],
    chunks: Sequence[Sequence[StaleMemoryItem]],
    selections: Sequence[StaleArmSelection],
) -> StaleArmSelection:
    """Merge repeated bounded CUPMem adjudications conservatively.

    A candidate is current only when every page that observed it classified it
    ACTIVE.  Any REPLACED, STALE, or UNKNOWN outcome therefore remains
    audit-only, matching CUPMem's constrained readout rule.
    """

    if len(chunks) != len(selections):
        raise ValueError("CUPMem chunk/selection count mismatch")
    ordered = tuple(
        sorted(items, key=lambda item: (item.session_index, item.memory_id))
    )
    candidate_ids = tuple(item.memory_id for item in ordered)
    known = set(candidate_ids)
    decisions_by_id: dict[str, list[Mapping[str, object]]] = {
        candidate_id: [] for candidate_id in candidate_ids
    }
    for chunk, selection in zip(chunks, selections):
        chunk_ids = {item.memory_id for item in chunk}
        if selection.arm != CUPMEM_ARM:
            raise ValueError("non-CUPMem selection in CUPMem merge")
        if set(selection.candidate_item_ids) != chunk_ids:
            raise ValueError("CUPMem selection does not match its chunk")
        seen: set[str] = set()
        for raw in selection.decisions:
            candidate_id = str(
                raw.get("candidate_id")
                or raw.get("memory_id")
                or ""
            )
            if candidate_id not in chunk_ids or candidate_id in seen:
                raise ValueError("invalid CUPMem chunk decision coverage")
            seen.add(candidate_id)
            decisions_by_id[candidate_id].append(dict(raw))
        if seen != chunk_ids:
            raise ValueError("CUPMem chunk omitted candidate decisions")
    if set(decisions_by_id) != known or any(
        not rows for rows in decisions_by_id.values()
    ):
        raise ValueError("CUPMem chunks did not cover every candidate")

    priority = {
        "REPLACED": 0,
        "STALE": 1,
        "UNKNOWN": 2,
        "ACTIVE": 3,
    }
    merged: list[Mapping[str, object]] = []
    selected: list[str] = []
    for candidate_id in candidate_ids:
        rows = decisions_by_id[candidate_id]
        statuses = [str(row.get("status") or "").upper() for row in rows]
        if any(status not in priority for status in statuses):
            raise ValueError("unsupported CUPMem status during merge")
        chosen = min(
            rows,
            key=lambda row: (
                priority[str(row.get("status") or "").upper()],
                str(row.get("replacement_candidate_id") or ""),
            ),
        )
        merged.append(dict(chosen))
        if all(status == "ACTIVE" for status in statuses):
            selected.append(candidate_id)
    return StaleArmSelection(
        arm=CUPMEM_ARM,
        candidate_item_ids=candidate_ids,
        selected_item_ids=tuple(selected),
        decisions=tuple(merged),
    )


def trace_prompt(items: Sequence[StaleMemoryItem], *, departure_after_session: int) -> str:
    rows = []
    for item in items:
        row = item.compact_record()
        row.update(
            {
                "observed_phase": (
                    "predeparture"
                    if item.session_index <= departure_after_session
                    else "absence"
                ),
                "source_receipt_sha256": item.source_session_sha256,
            }
        )
        rows.append(row)
    return (
        "Compile the query-independent TRACE return view for an agent returning "
        "after an absence. Treat each candidate as a provenance-bound observation, "
        "not as a vote. Determine cross-key temporal and causal dependencies: a "
        "newer fact may invalidate an older fact indirectly without using the same "
        "state_key or explicitly denying it. Preserve a predeparture item unless "
        "there is grounded later evidence that supersedes or makes it unsafe. "
        "ADMIT only facts safe to use as present state; mark outdated items "
        "SUPERSEDED and unresolved conflicts QUARANTINE. Do not see or anticipate "
        "downstream questions. Return exactly one JSON object with key decisions; "
        "include one row for every memory_id with memory_id, status (ADMIT, "
        "SUPERSEDED, or QUARANTINE), invalidated_by_memory_id (string or null), "
        "and no other keys. Do not return reasons, explanations, or markdown.\n\n"
        + json.dumps(
            {
                "return_boundary": {
                    "departure_after_session": departure_after_session,
                    "return_after_session": 49,
                },
                "candidates": rows,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def trace_dependency_scan_prompt(
    target_items: Sequence[StaleMemoryItem],
    later_items: Sequence[StaleMemoryItem],
    *,
    departure_after_session: int,
) -> str:
    """Build one bounded, outcome-blind dependency edge scan."""

    if not target_items:
        raise ValueError("TRACE dependency scan requires targets")
    if not later_items:
        raise ValueError("TRACE dependency scan requires later observations")
    if len({item.memory_id for item in target_items}) != len(target_items):
        raise ValueError("TRACE dependency targets must be unique")
    if len({item.memory_id for item in later_items}) != len(later_items):
        raise ValueError("TRACE later observations must be unique")

    def dependency_record(item: StaleMemoryItem) -> dict[str, object]:
        return {
            "memory_id": item.memory_id,
            "source_agent": item.agent_id,
            "session_index": item.session_index,
            "timestamp": item.timestamp,
            "observation_text": item.evidence_quote,
            "observed_phase": (
                "predeparture"
                if item.session_index <= departure_after_session
                else "absence"
            ),
            "source_receipt_sha256": item.source_session_sha256,
        }

    return (
        "Scan the supplied target observations against this complete chunk of "
        "later actor-visible observations. Emit an edge only when strong grounded "
        "later evidence makes a target unsafe to use as the user's current state, "
        "including implicit cross-attribute changes. General examples include "
        "environmental relocation, health changes that invalidate routines, "
        "schedule changes, ownership changes, and completed or abandoned plans. "
        "A mere topic change, compatible fact, or weak guess is not an "
        "invalidation. Each edge must link target_memory_id to one strictly-later "
        "invalidated_by_memory_id from this chunk. Return an empty edges array if "
        "there is no grounded invalidation. This is query-independent: do not use "
        "downstream questions or benchmark labels. Emit only edge records.\n\n"
        + json.dumps(
            {
                "return_boundary": {
                    "departure_after_session": departure_after_session,
                    "return_after_session": 49,
                },
                "targets": [dependency_record(item) for item in target_items],
                "later_observations": [dependency_record(item) for item in later_items],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def trace_dependency_candidate_prompt(
    target_item: StaleMemoryItem,
    later_items: Sequence[StaleMemoryItem],
    *,
    departure_after_session: int,
) -> str:
    """Build a high-recall, outcome-blind first-stage dependency scan.

    The model must classify every later observation instead of taking the
    cheap sparse-output path of returning an empty edge list.  A second,
    pairwise verifier decides whether a proposed candidate is strong enough
    to become a lifecycle edge.
    """

    if not later_items:
        raise ValueError("TRACE candidate scan requires later observations")
    if len({item.memory_id for item in later_items}) != len(later_items):
        raise ValueError("TRACE later observations must be unique")
    if any(item.session_index <= target_item.session_index for item in later_items):
        raise ValueError("TRACE candidates must be strictly later than the target")

    def dependency_record(
        item: StaleMemoryItem,
        *,
        local_index: int,
    ) -> dict[str, object]:
        return {
            # Gemini rejects integer-valued JSON-schema enums.  Expose the
            # transport index as a decimal string and map it back to the
            # local observation in the parser; no benchmark identifiers are
            # revealed to the model.
            "local_index": str(local_index),
            "source_agent": item.agent_id,
            "session_index": item.session_index,
            "timestamp": item.timestamp,
            "observation_text": item.evidence_quote,
            "observed_phase": (
                "predeparture"
                if item.session_index <= departure_after_session
                else "absence"
            ),
            "source_receipt_sha256": item.source_session_sha256,
        }

    return (
        "Perform a high-recall, query-independent lifecycle candidate scan. "
        "Compare the single target with EVERY later observation and emit one "
        "decision for every decimal-string later_index. Mark "
        "POSSIBLE_INVALIDATOR whenever "
        "the later observation could make the target unsafe as current user "
        "state. Look beyond shared words and check prerequisite/consequence "
        "relations across attributes. General high-recall cues include: climate, "
        "soil, native plants or animals, pests, architecture, and irrigation as "
        "evidence of geographic or home-environment change; relocation or new "
        "housing as evidence about day-to-day household composition; health or "
        "mobility changes as evidence about routines; and schedule, employment, "
        "ownership, completion, or abandonment changes as evidence about active "
        "plans. A city or state need not be named explicitly. Prefer "
        "POSSIBLE_INVALIDATOR when the relationship is plausible but requires "
        "common-sense reasoning; the next stage will verify it. Mark NO_EVIDENCE "
        "only for clearly compatible or unrelated facts. "
        "Do not use downstream questions or benchmark annotations. Return only "
        "the complete decision records.\n\n"
        + json.dumps(
            {
                "return_boundary": {
                    "departure_after_session": departure_after_session,
                    "return_after_session": 49,
                },
                "target": dependency_record(target_item, local_index=0),
                "later_observations": [
                    dependency_record(item, local_index=index)
                    for index, item in enumerate(later_items)
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def trace_dependency_verification_prompt(
    target_item: StaleMemoryItem,
    candidate_item: StaleMemoryItem,
    *,
    departure_after_session: int,
) -> str:
    """Build one grounded pairwise verification prompt."""

    if candidate_item.session_index <= target_item.session_index:
        raise ValueError("TRACE verifier requires a strictly later candidate")

    def dependency_record(item: StaleMemoryItem, *, role: str) -> dict[str, object]:
        return {
            "role": role,
            "source_agent": item.agent_id,
            "session_index": item.session_index,
            "timestamp": item.timestamp,
            "observation_text": item.evidence_quote,
            "observed_phase": (
                "predeparture"
                if item.session_index <= departure_after_session
                else "absence"
            ),
            "source_receipt_sha256": item.source_session_sha256,
        }

    return (
        "Verify one proposed lifecycle dependency using only the two grounded "
        "observations below. Return INVALIDATES_CURRENT_USE only when the later "
        "observation provides enough evidence that using the target as the "
        "user's present, day-to-day state would be unsafe or materially "
        "misleading. Do NOT require a permanent change or an explicit verbal "
        "contradiction: a substantial temporary relocation can invalidate a "
        "day-to-day co-residence claim, a health restriction can invalidate a "
        "routine, completion or abandonment can invalidate an active plan, and "
        "a newly evidenced climate or ecosystem can invalidate a location- or "
        "weather-specific home state even when no place name is stated. "
        "Apply a counterfactual compatibility test: assume the target is still "
        "the current state, then ask whether the later observation requires a "
        "location, physical environment, capability, schedule, or ownership "
        "condition that cannot reasonably hold at the same time. A repeated "
        "day-to-day activity is evidence about the user's current environment, "
        "not merely a hypothetical topic. If the required conditions are "
        "mutually exclusive, return INVALIDATES_CURRENT_USE. "
        "The older observation may remain historically true while being unsafe "
        "for current assistance. The relationship may be implicit and "
        "cross-attribute, but it must follow from the supplied evidence and "
        "ordinary common sense. Return "
        "DOES_NOT_INVALIDATE for merely related, compatible, ambiguous, or "
        "unrelated facts. Do not use downstream questions or benchmark labels.\n\n"
        + json.dumps(
            {
                "return_boundary": {
                    "departure_after_session": departure_after_session,
                    "return_after_session": 49,
                },
                "target": dependency_record(target_item, role="target"),
                "candidate": dependency_record(candidate_item, role="candidate"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def chunk_trace_candidate_observations(
    target_item: StaleMemoryItem,
    later_items: Sequence[StaleMemoryItem],
    *,
    departure_after_session: int,
    max_prompt_chars: int,
    max_items_per_chunk: int,
) -> tuple[tuple[StaleMemoryItem, ...], ...]:
    """Losslessly bound both candidate count and prompt size for stage one."""

    if max_prompt_chars < 1:
        raise ValueError("max_prompt_chars must be positive")
    if max_items_per_chunk < 1:
        raise ValueError("max_items_per_chunk must be positive")
    chunks: list[tuple[StaleMemoryItem, ...]] = []
    current: list[StaleMemoryItem] = []
    for item in later_items:
        candidate = (*current, item)
        prompt = trace_dependency_candidate_prompt(
            target_item,
            candidate,
            departure_after_session=departure_after_session,
        )
        if current and (
            len(candidate) > max_items_per_chunk or len(prompt) > max_prompt_chars
        ):
            chunks.append(tuple(current))
            current = [item]
            prompt = trace_dependency_candidate_prompt(
                target_item,
                current,
                departure_after_session=departure_after_session,
            )
        else:
            current.append(item)
        if len(prompt) > max_prompt_chars:
            raise ValueError(
                "one TRACE candidate pair exceeds prompt budget: "
                f"target_session={target_item.session_index}, "
                f"later_session={item.session_index}"
            )
    if current:
        chunks.append(tuple(current))
    reconstructed = [item.memory_id for chunk in chunks for item in chunk]
    if reconstructed != [item.memory_id for item in later_items]:
        raise RuntimeError("TRACE candidate chunking changed observation order")
    return tuple(chunks)


def chunk_trace_later_observations(
    target_items: Sequence[StaleMemoryItem],
    later_items: Sequence[StaleMemoryItem],
    *,
    departure_after_session: int,
    max_prompt_chars: int,
) -> tuple[tuple[StaleMemoryItem, ...], ...]:
    """Losslessly split later observations under an exact prompt-size budget."""

    if max_prompt_chars < 1:
        raise ValueError("max_prompt_chars must be positive")
    chunks: list[tuple[StaleMemoryItem, ...]] = []
    current: list[StaleMemoryItem] = []
    seen: list[str] = []
    for item in later_items:
        candidate = (*current, item)
        prompt = trace_dependency_scan_prompt(
            target_items,
            candidate,
            departure_after_session=departure_after_session,
        )
        if current and len(prompt) > max_prompt_chars:
            chunks.append(tuple(current))
            current = [item]
            prompt = trace_dependency_scan_prompt(
                target_items,
                current,
                departure_after_session=departure_after_session,
            )
        else:
            current.append(item)
        if len(prompt) > max_prompt_chars:
            raise ValueError(
                f"one TRACE session observation exceeds prompt budget: "
                f"session={item.session_index}"
            )
        seen.append(item.memory_id)
    if current:
        chunks.append(tuple(current))
    reconstructed = [item.memory_id for chunk in chunks for item in chunk]
    if reconstructed != seen:
        raise RuntimeError("TRACE chunking changed later-observation order")
    return tuple(chunks)


def parse_cupmem_selection(
    text: str,
    items: Sequence[StaleMemoryItem],
) -> StaleArmSelection:
    compilation = compile_cupmem_adjudication(
        stale_return_candidates(items, returning_agent_id="agent1"), text
    )
    return StaleArmSelection(
        arm=CUPMEM_ARM,
        candidate_item_ids=compilation.candidate_ids,
        selected_item_ids=compilation.selected_ids,
        decisions=compilation.decisions,
        parse_error=compilation.parse_error,
    )


def parse_trace_selection(
    text: str,
    items: Sequence[StaleMemoryItem],
) -> StaleArmSelection:
    return _parse_model_selection(
        text,
        items,
        arm=TRACE_ARM,
        statuses=_TRACE_STATUSES,
        selected_status="ADMIT",
        link_key="invalidated_by_memory_id",
    )


def parse_trace_dependency_decisions(
    text: str,
    *,
    target_items: Sequence[StaleMemoryItem],
    observation_items: Sequence[StaleMemoryItem],
) -> tuple[Mapping[str, object], ...]:
    """Parse a complete decision batch and require grounded forward links."""

    by_id = {item.memory_id: item for item in observation_items}
    target_ids = tuple(item.memory_id for item in target_items)
    payload = _json_object(text)
    rows = payload.get("decisions")
    if isinstance(rows, Mapping):
        rows = [
            {"memory_id": str(memory_id), **dict(value)}
            for memory_id, value in rows.items()
            if isinstance(value, Mapping)
        ]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("TRACE dependency decisions must be an array or object")
    parsed: dict[str, Mapping[str, object]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("every TRACE dependency decision must be an object")
        memory_id = _required(raw.get("memory_id"), "memory_id")
        if memory_id not in target_ids or memory_id in parsed:
            raise ValueError(f"unknown or duplicate target memory_id: {memory_id}")
        status = _required(raw.get("status"), "status").upper()
        if status not in _TRACE_STATUSES:
            raise ValueError(f"unsupported TRACE status: {status}")
        linked = str(raw.get("invalidated_by_memory_id") or "").strip() or None
        if linked is not None and linked not in by_id:
            raise ValueError(f"unknown invalidator memory_id: {linked}")
        if status == "SUPERSEDED":
            if linked is None:
                raise ValueError("SUPERSEDED decision requires an invalidator")
            if by_id[linked].session_index <= by_id[memory_id].session_index:
                raise ValueError("TRACE invalidator must be strictly later")
        elif linked is not None:
            raise ValueError(f"{status} decision cannot link an invalidator")
        parsed[memory_id] = {
            "memory_id": memory_id,
            "status": status,
            "invalidated_by_memory_id": linked,
        }
    missing = set(target_ids) - set(parsed)
    if missing:
        raise ValueError("TRACE dependency omitted targets: " + ",".join(sorted(missing)))
    return tuple(parsed[memory_id] for memory_id in target_ids)


def parse_trace_dependency_edges(
    text: str,
    *,
    target_items: Sequence[StaleMemoryItem],
    later_items: Sequence[StaleMemoryItem],
) -> tuple[Mapping[str, object], ...]:
    """Parse grounded forward edges from one bounded dependency scan."""

    targets = {item.memory_id: item for item in target_items}
    later = {item.memory_id: item for item in later_items}
    payload = _json_object(text)
    rows = payload.get("edges")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("TRACE dependency edges must be an array")
    parsed: list[Mapping[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("every TRACE dependency edge must be an object")
        target_id = _required(raw.get("target_memory_id"), "target_memory_id")
        invalidator_id = _required(
            raw.get("invalidated_by_memory_id"), "invalidated_by_memory_id"
        )
        if target_id not in targets:
            raise ValueError(f"unknown target memory_id: {target_id}")
        if invalidator_id not in later:
            raise ValueError(f"unknown invalidator memory_id: {invalidator_id}")
        if later[invalidator_id].session_index <= targets[target_id].session_index:
            # A constrained decoder can still pair two individually valid enum
            # values into a temporally invalid edge.  The edge is not evidence
            # and must never enter the graph, but it should not invalidate the
            # rest of this otherwise auditable scan.
            continue
        pair = (target_id, invalidator_id)
        if pair in seen:
            raise ValueError("duplicate TRACE dependency edge")
        seen.add(pair)
        parsed.append(
            {
                "target_memory_id": target_id,
                "invalidated_by_memory_id": invalidator_id,
            }
        )
    return tuple(parsed)


def parse_trace_dependency_candidates(
    text: str,
    *,
    target_item: StaleMemoryItem,
    later_items: Sequence[StaleMemoryItem],
) -> tuple[str, ...]:
    """Parse a complete high-recall decision for every supplied later item."""

    later = {item.memory_id: item for item in later_items}
    payload = _json_object(text)
    rows = payload.get("decisions")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("TRACE candidate decisions must be an array")
    parsed: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("every TRACE candidate decision must be an object")
        if "later_index" in raw:
            raw_index = raw.get("later_index")
            if isinstance(raw_index, bool):
                raise TypeError("later_index must be a decimal string")
            if isinstance(raw_index, str):
                normalized_index = raw_index.strip()
                if not normalized_index.isdecimal():
                    raise TypeError("later_index must be a decimal string")
                local_index = int(normalized_index)
            elif isinstance(raw_index, int):
                # Accept immutable pre-v5 fixtures and caches while all new
                # forced-tool requests use the provider-compatible string
                # representation.
                local_index = raw_index
            else:
                raise TypeError("later_index must be a decimal string")
            if not 0 <= local_index < len(later_items):
                raise ValueError(f"unknown TRACE later_index: {raw_index}")
            later_id = later_items[local_index].memory_id
        else:
            # Backward-compatible parsing for immutable pre-v4 test fixtures.
            target_id = _required(
                raw.get("target_memory_id"), "target_memory_id"
            )
            later_id = _required(raw.get("later_memory_id"), "later_memory_id")
            if target_id != target_item.memory_id:
                raise ValueError(f"unknown TRACE candidate target: {target_id}")
        status = _required(raw.get("status"), "status").upper()
        if later_id not in later or later_id in parsed:
            raise ValueError(f"unknown or duplicate later_memory_id: {later_id}")
        if later[later_id].session_index <= target_item.session_index:
            raise ValueError("TRACE candidate must be strictly later")
        if status not in {"POSSIBLE_INVALIDATOR", "NO_EVIDENCE"}:
            raise ValueError(f"unsupported TRACE candidate status: {status}")
        parsed[later_id] = status
    missing = set(later) - set(parsed)
    if missing:
        raise ValueError(
            "TRACE candidate scan omitted later observations: "
            + ",".join(sorted(missing))
        )
    return tuple(
        item.memory_id
        for item in later_items
        if parsed[item.memory_id] == "POSSIBLE_INVALIDATOR"
    )


def parse_trace_dependency_verification(
    text: str,
    *,
    target_item: StaleMemoryItem,
    candidate_item: StaleMemoryItem,
) -> bool:
    """Parse one pairwise lifecycle verdict with exact provenance binding."""

    payload = _json_object(text)
    # The v4 verifier is bound to exactly one pair by its prompt and receipt,
    # so the model returns only a verdict.  Accept and validate legacy IDs when
    # reading older fixtures, but do not ask a model to copy opaque hashes.
    target_id = str(payload.get("target_memory_id") or "").strip() or None
    candidate_id = str(payload.get("candidate_memory_id") or "").strip() or None
    verdict = _required(payload.get("verdict"), "verdict").upper()
    if target_id is not None and target_id != target_item.memory_id:
        raise ValueError(f"unknown TRACE verification target: {target_id}")
    if candidate_id is not None and candidate_id != candidate_item.memory_id:
        raise ValueError(f"unknown TRACE verification candidate: {candidate_id}")
    if candidate_item.session_index <= target_item.session_index:
        raise ValueError("TRACE verification candidate must be strictly later")
    if verdict not in {"INVALIDATES_CURRENT_USE", "DOES_NOT_INVALIDATE"}:
        raise ValueError(f"unsupported TRACE verification verdict: {verdict}")
    return verdict == "INVALIDATES_CURRENT_USE"


def compile_trace_dependency_edges(
    target_items: Sequence[StaleMemoryItem],
    observation_items: Sequence[StaleMemoryItem],
    edges: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Deterministically compile scan edges into one decision per target."""

    observations = {item.memory_id: item for item in observation_items}
    candidates: dict[str, set[str]] = {
        item.memory_id: set() for item in target_items
    }
    for edge in edges:
        target_id = _required(edge.get("target_memory_id"), "target_memory_id")
        invalidator_id = _required(
            edge.get("invalidated_by_memory_id"), "invalidated_by_memory_id"
        )
        if target_id not in candidates or invalidator_id not in observations:
            raise ValueError("TRACE edge compilation received an unknown memory_id")
        if observations[invalidator_id].session_index <= observations[target_id].session_index:
            raise ValueError("TRACE edge compilation requires a forward edge")
        candidates[target_id].add(invalidator_id)
    result: list[Mapping[str, object]] = []
    for target in target_items:
        linked = sorted(
            candidates[target.memory_id],
            key=lambda memory_id: (
                observations[memory_id].session_index,
                memory_id,
            ),
        )
        invalidator_id = linked[-1] if linked else None
        result.append(
            {
                "memory_id": target.memory_id,
                "status": "SUPERSEDED" if invalidator_id is not None else "ADMIT",
                "invalidated_by_memory_id": invalidator_id,
            }
        )
    return tuple(result)


def expand_trace_retrieval(
    items: Sequence[StaleMemoryItem],
    initially_retrieved: Sequence[StaleMemoryItem],
    selection: StaleArmSelection,
) -> tuple[StaleMemoryItem, ...]:
    """Replace retrieved stale observations with terminal invalidators."""

    by_id = {item.memory_id: item for item in items}
    selected = set(selection.selected_item_ids)
    links = {
        str(row["memory_id"]): str(row["invalidated_by_memory_id"])
        for row in selection.decisions
        if row.get("status") == "SUPERSEDED"
        and row.get("invalidated_by_memory_id")
    }
    chosen: set[str] = set()
    for item in initially_retrieved:
        memory_id = item.memory_id
        visited: set[str] = set()
        while memory_id in links:
            if memory_id in visited:
                raise ValueError("TRACE invalidator dependency contains a cycle")
            visited.add(memory_id)
            memory_id = links[memory_id]
        if memory_id in selected:
            chosen.add(memory_id)
    return tuple(
        item
        for item in sorted(items, key=lambda row: (row.session_index, row.memory_id))
        if item.memory_id in chosen and item.memory_id in by_id
    )


def _parse_model_selection(
    text: str,
    items: Sequence[StaleMemoryItem],
    *,
    arm: str,
    statuses: frozenset[str],
    selected_status: str,
    link_key: str,
) -> StaleArmSelection:
    known = {item.memory_id for item in items}
    ordered_ids = tuple(item.memory_id for item in items)
    payload = _json_object(text)
    rows = payload.get("decisions")
    if isinstance(rows, Mapping):
        normalized_rows: list[Mapping[str, object]] = []
        for memory_id, value in rows.items():
            if not isinstance(value, Mapping):
                raise TypeError("mapped governance decisions must be objects")
            normalized_rows.append({"memory_id": str(memory_id), **dict(value)})
        rows = normalized_rows
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("governance decisions must be an array or ID-keyed object")
    parsed: dict[str, dict[str, object]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("every governance decision must be an object")
        memory_id = _required(raw.get("memory_id"), "memory_id")
        if memory_id not in known or memory_id in parsed:
            raise ValueError(f"unknown or duplicate memory_id: {memory_id}")
        status = _required(raw.get("status"), "status").upper()
        if status not in statuses:
            raise ValueError(f"unsupported {arm} status: {status}")
        linked = str(raw.get(link_key) or "").strip() or None
        if linked is not None and linked not in known:
            raise ValueError(f"unknown linked memory_id: {linked}")
        parsed[memory_id] = {
            "memory_id": memory_id,
            "status": status,
            link_key: linked,
            "reason": str(raw.get("reason") or "").strip()[:600],
        }
    missing = known - set(parsed)
    if missing:
        raise ValueError("governance omitted candidates: " + ",".join(sorted(missing)))
    selected = tuple(
        memory_id
        for memory_id in ordered_ids
        if parsed[memory_id]["status"] == selected_status
    )
    return StaleArmSelection(
        arm=arm,
        candidate_item_ids=ordered_ids,
        selected_item_ids=selected,
        decisions=tuple(parsed[memory_id] for memory_id in ordered_ids),
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have the same non-zero dimension")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot / (left_norm * right_norm)


def retrieve_for_queries(
    items: Sequence[StaleMemoryItem],
    queries: Sequence[str],
    *,
    item_embeddings: Mapping[str, Sequence[float]],
    query_embeddings: Sequence[Sequence[float]],
    top_k_per_query: int,
    max_union_items: int,
) -> tuple[StaleMemoryItem, ...]:
    if len(queries) != len(query_embeddings):
        raise ValueError("query embedding count mismatch")
    if top_k_per_query < 1 or max_union_items < 1:
        raise ValueError("retrieval budgets must be positive")
    chosen: dict[str, float] = {}
    for query_vector in query_embeddings:
        ranked = sorted(
            (
                (
                    cosine_similarity(item_embeddings[item.memory_id], query_vector),
                    item.session_index,
                    item.memory_id,
                )
                for item in items
            ),
            reverse=True,
        )
        for score, _, memory_id in ranked[:top_k_per_query]:
            chosen[memory_id] = max(chosen.get(memory_id, -2.0), score)
    limited_ids = {
        memory_id
        for memory_id, _ in sorted(
            chosen.items(), key=lambda row: (row[1], row[0]), reverse=True
        )[:max_union_items]
    }
    return tuple(
        item
        for item in sorted(items, key=lambda row: (row.session_index, row.memory_id))
        if item.memory_id in limited_ids
    )


def answer_prompt(
    items: Sequence[StaleMemoryItem],
    queries: Sequence[str],
    *,
    lifecycle_resolutions: Sequence[Mapping[str, object]] = (),
) -> str:
    if len(queries) != 3:
        raise ValueError("official STALE answer requires exactly three queries")
    memory_rows = [item.compact_record(quote_chars=700) for item in items]
    resolution_instruction = (
        " Lifecycle resolutions are provenance-grounded decisions produced "
        "before question time. When a resolution marks a historical target "
        "SUPERSEDED, do not treat that target as current; use its grounded "
        "current invalidator instead."
        if lifecycle_resolutions
        else ""
    )
    return (
        "You are an agent returning to an ongoing user-assistance task. The "
        "following return-memory view was produced before you saw the questions. "
        "Use only this view and the questions; do not assume missing user facts. "
        "Answer each question directly and helpfully. If later evidence changes "
        "an earlier state, respect the currently valid state and resist false "
        "premises. Return exactly one JSON object with string keys "
        "dim1_response, dim2_response, and dim3_response.\n\n"
        + resolution_instruction
        + ("\n\n" if resolution_instruction else "")
        + json.dumps(
            {
                "return_memory": memory_rows,
                "lifecycle_resolutions": [
                    dict(resolution) for resolution in lifecycle_resolutions
                ],
                "questions": {
                    "dim1_query": queries[0],
                    "dim2_query": queries[1],
                    "dim3_query": queries[2],
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def full_history_answer_prompt(
    sessions: Sequence[Mapping[str, object]],
    queries: Sequence[str],
) -> str:
    """Render an uncompressed actor-visible history for diagnostic use only."""

    if len(queries) != 3:
        raise ValueError("official STALE answer requires exactly three queries")
    ordered: list[Mapping[str, object]] = []
    seen: set[int] = set()
    for session in sessions:
        session_index = int(session["session_index"])
        if session_index in seen:
            raise ValueError("full-history diagnostic contains duplicate sessions")
        seen.add(session_index)
        messages = session.get("messages")
        if not isinstance(messages, Sequence) or isinstance(
            messages, (str, bytes)
        ):
            raise TypeError("full-history session messages must be an array")
        ordered.append(
            {
                "session_index": session_index,
                "timestamp": _required(session.get("timestamp"), "timestamp"),
                "messages": [dict(message) for message in messages],
            }
        )
    ordered.sort(key=lambda row: int(row["session_index"]))
    return (
        "Diagnostic full-history condition. You are agent1 returning to the "
        "user-assistance task, and the evaluator has supplied the complete "
        "actor-visible chronological history without benchmark annotations. "
        "Answer each question directly and helpfully. Respect later changes, "
        "reject stale premises, and do not assume missing user facts. Return "
        "exactly one JSON object with string keys dim1_response, dim2_response, "
        "and dim3_response.\n\n"
        + json.dumps(
            {
                "chronological_history": ordered,
                "questions": {
                    "dim1_query": queries[0],
                    "dim2_query": queries[1],
                    "dim3_query": queries[2],
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def parse_answers(text: str) -> dict[str, str]:
    payload = _json_object(text)
    return {
        key: _required(payload.get(key), key, max_chars=8000)
        for key in ("dim1_response", "dim2_response", "dim3_response")
    }


def parse_judge(text: str) -> dict[str, object]:
    payload = _json_object(text)
    result: dict[str, object] = {}
    for index in range(1, 4):
        key = f"dim{index}_eval"
        raw = payload.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Judge omitted {key}")
        passed = raw.get("pass")
        if not isinstance(passed, bool):
            raise ValueError(f"Judge {key}.pass must be boolean")
        result[key] = {
            "reasoning": str(raw.get("reasoning") or "").strip()[:4000],
            "pass": passed,
        }
    return result


def memory_items_sha256(items: Sequence[StaleMemoryItem]) -> str:
    return canonical_sha256([item.record() for item in items])


__all__ = [
    "CHECKPOINT_REPLAY_ARM",
    "CUPMEM_ARM",
    "KMU_OP_ARM",
    "MEMSTRATA_ARM",
    "MEMTX_ARM",
    "RESET_ARM",
    "RESTORE_ARM",
    "STALE_SIX_ARMS",
    "STALE_TRACE_ABLATION_ARMS",
    "TRACE_WITHOUT_FALLBACK_ARM",
    "TRACE_WITHOUT_FRESHNESS_ARM",
    "TRACE_WITHOUT_PROVENANCE_ARM",
    "VALIDITY_FILTER_ONLY_ARM",
    "STALE_ALL_ARMS",
    "STALE_FOUR_ARMS",
    "SESSION_OBSERVATION_KEY_PREFIX",
    "SESSION_TRANSPORT_PAGE_KEY",
    "STATIC_ARM",
    "TEMPORAL_LWW_ARM",
    "TRACE_ARM",
    "StaleArmSelection",
    "StaleMemoryItem",
    "answer_prompt",
    "chunk_actor_sessions",
    "chunk_cupmem_candidates",
    "paginate_actor_sessions",
    "chunk_kmu_candidates",
    "chunk_trace_candidate_observations",
    "chunk_trace_later_observations",
    "compile_trace_dependency_edges",
    "cosine_similarity",
    "cupmem_prompt",
    "deterministic_arm_selections",
    "extraction_prompt",
    "expand_trace_retrieval",
    "full_history_answer_prompt",
    "memory_items_sha256",
    "kmu_prompt",
    "kmu_chunk_index_contract",
    "merge_cupmem_chunk_selections",
    "merge_kmu_chunk_selections",
    "parse_answers",
    "parse_cupmem_selection",
    "parse_extraction",
    "parse_judge",
    "parse_indexed_kmu_selection",
    "parse_kmu_selection",
    "parse_trace_selection",
    "parse_trace_dependency_candidates",
    "parse_trace_dependency_decisions",
    "parse_trace_dependency_edges",
    "parse_trace_dependency_verification",
    "retrieve_for_queries",
    "session_observation_items",
    "stale_return_candidates",
    "trace_dependency_candidate_prompt",
    "trace_dependency_scan_prompt",
    "trace_dependency_verification_prompt",
    "trace_prompt",
]
