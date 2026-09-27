"""Official CUPMem pipeline with benchmark-neutral runtime adapters.

The implementation under :mod:`trace.vendor.cupmem_official` is an
unmodified, commit-pinned copy of the authors' released CUPMem package. This
module only supplies infrastructure adapters that are outside the method:

* an OpenAI-compatible remote embedding backend;
* a filesystem request lock for a shared local actor endpoint; and
* deterministic, label-blind splitting for sessions that exceed an endpoint's
  context budget.

It deliberately does not reuse the lightweight ``cupmem`` return adapter.
That adapter is diagnostic-only; the public method name ``CUPMem`` executes
extraction, delta
resolution, invalidation, stale linking, retrieval, premise verification,
basis recovery, action grounding, and answer composition from the official
pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from trace.providers import OpenAICompatibleEmbeddingClient
from trace.return_methods.base import ReturnMethodSpec
from trace.vendor_support.openai_compat import (
    ContextWindowExceededError,
    OpenAICompat,
    install_openai_compat_if_missing,
)


install_openai_compat_if_missing()

from trace.vendor.cupmem_official.core.config import (
    PipelineThresholds,
    TraceConfig,
)
from trace.vendor.cupmem_official.llm_layer.client import LLMClient
from trace.vendor.cupmem_official.memory.models import (
    BucketTrackRef,
    ProfileItem,
    SessionChunk,
    SessionDelta,
    StaleSupportLink,
    UnknownCurrent,
)
from trace.vendor.cupmem_official.memory.schema import schema_prompt_text
from trace.vendor.cupmem_official.pipeline import (
    CupMemEngine as UpstreamCupMemEngine,
)
from trace.vendor.cupmem_official.retrieval.embedding import BaseRetriever
from trace.vendor.cupmem_official.store_layer import ProfileStore


CUPMEM_SYSTEM_ARM = "cupmem_full"
# Historical import retained for existing result readers and active launchers.
CUPMEM_FULL_ARM = CUPMEM_SYSTEM_ARM
# ``cupmem_full`` remains the stable artifact identifier so completed and
# currently running experiments stay readable. Paper tables and all new
# human-facing reports use the canonical upstream method name ``CUPMem``.
CUPMEM_DISPLAY_NAME = "CUPMem"
CUPMEM_SYSTEM_IMPLEMENTATION_KIND = "official_pipeline_benchmark_adapter"
# Historical aliases retained for imports in active launchers.
CUPMEM_FULL_DISPLAY_NAME = CUPMEM_DISPLAY_NAME
CUPMEM_FULL_IMPLEMENTATION_KIND = CUPMEM_SYSTEM_IMPLEMENTATION_KIND
CUPMEM_UPSTREAM_COMMIT = "ea7d391103a151927cd29d2f01d87597a782bdcb"
CUPMEM_UPSTREAM_REPOSITORY = "https://github.com/icedreamc/STALE"
CUPMEM_METHOD = ReturnMethodSpec(
    method=CUPMEM_SYSTEM_ARM,
    display_name=CUPMEM_DISPLAY_NAME,
    category="complete_memory_system",
    produces_memory_view=False,
    paper_url="https://arxiv.org/abs/2605.06527",
    reference_url=CUPMEM_UPSTREAM_REPOSITORY,
    implementation_kind=CUPMEM_SYSTEM_IMPLEMENTATION_KIND,
    reference_commit=CUPMEM_UPSTREAM_COMMIT,
)
CUPMEM_TIMELINE_CHECKPOINT_SCHEMA = "cupmem_full_timeline_checkpoint_v1"
CUPMEM_CONTEXT_BUDGET_ADAPTER_VERSION = "context_budget_v2"


@contextmanager
def _exclusive_file_lock(path: str | None):
    if not path:
        yield
        return
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class LockedLLMClient(LLMClient):
    """Official LLM client serialized against other local actor jobs."""

    def __init__(
        self,
        *,
        request_lock_path: str | None = None,
        context_window_tokens: int = 32_768,
        default_max_tokens: int = 3_072,
        context_reserve_tokens: int = 512,
        minimum_output_tokens: int = 512,
        adaptive_context_retry: bool = True,
        **kwargs: Any,
    ):
        # The compatibility transport avoids inheriting experiment-wide SOCKS
        # proxy settings for the localhost actor endpoint. It implements the
        # exact chat.completions surface used by the pinned upstream client.
        if context_window_tokens < 1:
            raise ValueError("context_window_tokens must be positive")
        if default_max_tokens < 1:
            raise ValueError("default_max_tokens must be positive")
        if context_reserve_tokens < 0:
            raise ValueError("context_reserve_tokens cannot be negative")
        if minimum_output_tokens < 1:
            raise ValueError("minimum_output_tokens must be positive")
        if default_max_tokens + context_reserve_tokens >= context_window_tokens:
            raise ValueError("CUPMem output budget leaves no room for a prompt")

        self.context_window_tokens = int(context_window_tokens)
        self.default_max_tokens = int(default_max_tokens)
        self.context_reserve_tokens = int(context_reserve_tokens)
        self.minimum_output_tokens = int(minimum_output_tokens)
        self.adaptive_context_retry = bool(adaptive_context_retry)

        def openai_factory(**client_kwargs: Any) -> OpenAICompat:
            return OpenAICompat(
                **client_kwargs,
                default_max_tokens=self.default_max_tokens,
                context_reserve_tokens=self.context_reserve_tokens,
                minimum_output_tokens=self.minimum_output_tokens,
                adaptive_context_retry=self.adaptive_context_retry,
            )

        kwargs.setdefault("openai_cls", openai_factory)
        super().__init__(**kwargs)
        self.request_lock_path = request_lock_path

    def call_text(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        with _exclusive_file_lock(self.request_lock_path):
            try:
                return super().call_text(messages, **kwargs)
            except json.JSONDecodeError:
                # Recover only transport-cache corruption. Method outputs are
                # never repaired or reinterpreted here.
                cache_path = self._cache_path(messages)
                if cache_path is None or not cache_path.exists():
                    raise
                cache_path.unlink()
                return super().call_text(messages, **kwargs)


class RemoteEmbeddingRetriever(BaseRetriever):
    """CUPMem ``BaseRetriever`` backed by the shared Qwen embedding API."""

    def __init__(
        self,
        client: OpenAICompatibleEmbeddingClient,
        *,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        max_chars: int = 12_000,
    ) -> None:
        if batch_size < 1:
            raise ValueError("embedding batch_size must be positive")
        if max_chars < 128:
            raise ValueError("embedding max_chars is too small")
        self.client = client
        self.cache_dir = cache_dir
        self.batch_size = int(batch_size)
        self.max_chars = int(max_chars)
        self._memory_cache: dict[str, list[float]] = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _normalized_text(self, value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[: self.max_chars]

    def _key(self, text: str) -> str:
        payload = (self.client.model_id + "\0" + text).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _disk_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.json"

    def _read(self, key: str) -> list[float] | None:
        if key in self._memory_cache:
            return self._memory_cache[key]
        path = self._disk_path(key)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            vector = [float(value) for value in payload["embedding"]]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            try:
                path.unlink()
            except OSError:
                pass
            return None
        self._memory_cache[key] = vector
        return vector

    def _write(self, key: str, vector: list[float]) -> None:
        self._memory_cache[key] = vector
        path = self._disk_path(key)
        if path is None or path.exists():
            return
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"embedding": vector}, handle, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        results: list[list[float] | None] = [self._read(key) for key in keys]
        missing_indices = [index for index, value in enumerate(results) if value is None]
        for offset in range(0, len(missing_indices), self.batch_size):
            indices = missing_indices[offset : offset + self.batch_size]
            vectors = self.client.embed([texts[index] for index in indices])
            if len(vectors) != len(indices):
                raise ValueError("embedding response count mismatch")
            for index, vector in zip(indices, vectors):
                self._write(keys[index], vector)
                results[index] = vector
        return [value for value in results if value is not None]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            raise ValueError("embedding dimensions do not match")
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def rank(
        self,
        *,
        query_text: str,
        candidates: Iterable[Any],
        text_getter: Callable[[Any], str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        candidate_list = list(candidates)
        if not candidate_list or top_k <= 0:
            return []
        texts = [self._normalized_text(query_text)] + [
            self._normalized_text(text_getter(candidate))
            for candidate in candidate_list
        ]
        vectors = self._embed(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedding cache returned an incomplete batch")
        ranked = [
            {
                "score": self._cosine(vectors[0], vectors[index + 1]),
                "item": candidate,
            }
            for index, candidate in enumerate(candidate_list)
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[: int(top_k)]


class CupMemEngine(UpstreamCupMemEngine):
    """Official engine with dependency-injected remote retrieval only."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        embedding: BaseRetriever,
        thresholds: PipelineThresholds | None = None,
        trace_config: TraceConfig | None = None,
    ) -> None:
        # Mirrors the pinned upstream constructor except that an already-built
        # BaseRetriever is injected instead of loading a local checkpoint.
        self.llm = llm
        self.thresholds = thresholds or PipelineThresholds()
        self.trace_config = trace_config or TraceConfig()
        self.store = ProfileStore()
        self.embedding = embedding
        self.chunk_bank: list[SessionChunk] = []
        self.delta_store: list[SessionDelta] = []
        self._chunk_counter = 0
        self._delta_counter = 0
        self._proposal_counter = 0
        self.schema_text = schema_prompt_text()


@dataclass(frozen=True)
class CupMemSession:
    source_session_id: str
    timestamp: str
    messages: tuple[Mapping[str, object], ...]
    phase: str


@dataclass(frozen=True)
class CupMemQuery:
    label: str
    text: str


def _engine_contract(engine: CupMemEngine) -> dict[str, object]:
    embedding_client = getattr(engine.embedding, "client", None)
    return {
        "upstream_commit": CUPMEM_UPSTREAM_COMMIT,
        "actor_model": str(getattr(engine.llm, "model", "")),
        "embedding_model": str(getattr(embedding_client, "model_id", "")),
        "thresholds": dict(vars(engine.thresholds)),
        "trace_config": dict(vars(engine.trace_config)),
        "context_budget": {
            "adapter_version": CUPMEM_CONTEXT_BUDGET_ADAPTER_VERSION,
            "context_window_tokens": int(
                getattr(engine.llm, "context_window_tokens", 0)
            ),
            "default_max_tokens": int(
                getattr(engine.llm, "default_max_tokens", 0)
            ),
            "context_reserve_tokens": int(
                getattr(engine.llm, "context_reserve_tokens", 0)
            ),
            "minimum_output_tokens": int(
                getattr(engine.llm, "minimum_output_tokens", 0)
            ),
            "adaptive_context_retry": bool(
                getattr(engine.llm, "adaptive_context_retry", False)
            ),
        },
    }


def cupmem_full_timeline_fingerprint(
    *,
    engine: CupMemEngine,
    benchmark: str,
    sessions: Sequence[CupMemSession],
    departure_source_session_id: str | None,
    max_user_chars_per_session: int = 28_000,
) -> str:
    """Return a query-independent identity for one fully processed timeline."""

    normalized, _ = normalize_sessions_label_blind(
        sessions,
        max_user_chars_per_session=max_user_chars_per_session,
    )
    payload = {
        "schema_version": CUPMEM_TIMELINE_CHECKPOINT_SCHEMA,
        "contract": _engine_contract(engine),
        "benchmark": benchmark,
        "departure_source_session_id": departure_source_session_id,
        "max_user_chars_per_session": max_user_chars_per_session,
        "sessions": [
            {
                "source_session_id": session.source_session_id,
                "timestamp": session.timestamp,
                "phase": session.phase,
                "messages": [dict(message) for message in session.messages],
            }
            for session in normalized
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_engine_state(engine: CupMemEngine) -> dict[str, object]:
    return {
        "active_items": [
            {
                "bucket": bucket,
                "local_track": local_track,
                "items": [item.to_dict() for item in items],
            }
            for (bucket, local_track), items in sorted(engine.store.active_items.items())
        ],
        "stale_archive": [item.to_dict() for item in engine.store.stale_archive],
        "unknown_current": [
            item.to_dict() for item in engine.store.unknown_current.values()
        ],
        "stale_support_links": [
            link.to_dict() for link in engine.store.stale_support_links
        ],
        "store_counter": int(engine.store._counter),
        "chunk_bank": [chunk.to_dict() for chunk in engine.chunk_bank],
        "delta_store": [delta.to_dict() for delta in engine.delta_store],
        "chunk_counter": int(engine._chunk_counter),
        "delta_counter": int(engine._delta_counter),
        "proposal_counter": int(engine._proposal_counter),
    }


def _restore_engine_state(
    engine: CupMemEngine,
    state: Mapping[str, object],
) -> None:
    store = ProfileStore()
    active_items: defaultdict[tuple[str, str], list[ProfileItem]] = defaultdict(list)
    for track_payload in state.get("active_items", []):
        if not isinstance(track_payload, Mapping):
            raise ValueError("invalid CUPMem checkpoint active track")
        key = (
            str(track_payload.get("bucket", "")),
            str(track_payload.get("local_track", "")),
        )
        raw_items = track_payload.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError("invalid CUPMem checkpoint active items")
        active_items[key] = [ProfileItem(**dict(item)) for item in raw_items]
    store.active_items = active_items
    store.stale_archive = [
        ProfileItem(**dict(item)) for item in state.get("stale_archive", [])
    ]
    store.unknown_current = {}
    for raw_item in state.get("unknown_current", []):
        item = UnknownCurrent(**dict(raw_item))
        store.unknown_current[(item.bucket, item.local_track)] = item
    store.stale_support_links = [
        StaleSupportLink(**dict(item))
        for item in state.get("stale_support_links", [])
    ]
    store._counter = int(state.get("store_counter", 0))

    chunks: list[SessionChunk] = []
    for raw_chunk in state.get("chunk_bank", []):
        payload = dict(raw_chunk)
        payload["candidate_bucket_tracks"] = [
            BucketTrackRef(**dict(track))
            for track in payload.get("candidate_bucket_tracks", [])
        ]
        chunks.append(SessionChunk(**payload))

    engine.store = store
    engine.chunk_bank = chunks
    engine.delta_store = [
        SessionDelta(**dict(delta)) for delta in state.get("delta_store", [])
    ]
    engine._chunk_counter = int(state.get("chunk_counter", len(chunks)))
    engine._delta_counter = int(
        state.get("delta_counter", len(engine.delta_store))
    )
    engine._proposal_counter = int(state.get("proposal_counter", 0))


def build_cupmem_full_timeline_checkpoint(
    *,
    engine: CupMemEngine,
    benchmark: str,
    sessions: Sequence[CupMemSession],
    departure_source_session_id: str | None,
    max_user_chars_per_session: int = 28_000,
) -> dict[str, object]:
    """Process actor-visible sessions once and capture the complete CUPMem state."""

    engine.reset()
    if hasattr(engine.llm, "reset_usage_tracking"):
        engine.llm.reset_usage_tracking()
    normalized, normalization_map = normalize_sessions_label_blind(
        sessions,
        max_user_chars_per_session=max_user_chars_per_session,
    )
    session_logs: list[dict[str, object]] = []
    for index, session in enumerate(normalized):
        log = engine.process_session(
            session=[dict(message) for message in session.messages],
            session_index=index,
            session_time=session.timestamp,
        )
        log["adapter_source_session_id"] = session.source_session_id
        log["adapter_phase"] = session.phase
        session_logs.append(log)

    fingerprint = cupmem_full_timeline_fingerprint(
        engine=engine,
        benchmark=benchmark,
        sessions=sessions,
        departure_source_session_id=departure_source_session_id,
        max_user_chars_per_session=max_user_chars_per_session,
    )
    return {
        "schema_version": CUPMEM_TIMELINE_CHECKPOINT_SCHEMA,
        "fingerprint": fingerprint,
        "method": CUPMEM_FULL_ARM,
        "benchmark": benchmark,
        "departure_source_session_id": departure_source_session_id,
        "contract": _engine_contract(engine),
        "session_normalization": {
            "policy": "label_blind_actor_text_size_v1",
            "max_user_chars_per_session": max_user_chars_per_session,
            "mapping": list(normalization_map),
        },
        "session_logs": session_logs,
        "engine_state": _capture_engine_state(engine),
        "write_llm_usage": (
            engine.llm.get_usage_summary()
            if hasattr(engine.llm, "get_usage_summary")
            else {}
        ),
    }


def restore_cupmem_full_timeline_checkpoint(
    *,
    engine: CupMemEngine,
    checkpoint: Mapping[str, object],
) -> None:
    if checkpoint.get("schema_version") != CUPMEM_TIMELINE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported CUPMem timeline checkpoint schema")
    if checkpoint.get("contract") != _engine_contract(engine):
        raise ValueError("CUPMem timeline checkpoint runtime contract mismatch")
    state = checkpoint.get("engine_state")
    if not isinstance(state, Mapping):
        raise ValueError("CUPMem timeline checkpoint has no engine state")
    _restore_engine_state(engine, state)
    if hasattr(engine.llm, "reset_usage_tracking"):
        engine.llm.reset_usage_tracking()


def save_cupmem_full_timeline_checkpoint(
    path: Path,
    checkpoint: Mapping[str, object],
) -> None:
    """Atomically persist a compressed checkpoint."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(checkpoint, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cupmem_full_timeline_checkpoint(
    path: Path,
    *,
    expected_fingerprint: str,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != CUPMEM_TIMELINE_CHECKPOINT_SCHEMA
            or value.get("fingerprint") != expected_fingerprint
        ):
            return None
        return value
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


@contextmanager
def cupmem_full_timeline_checkpoint_lock(path: Path):
    """Prevent concurrent workers from rebuilding the same timeline state."""

    with _exclusive_file_lock(str(path.expanduser().resolve()) + ".lock"):
        yield


def run_cupmem_full_query_from_checkpoint(
    *,
    engine: CupMemEngine,
    checkpoint: Mapping[str, object],
    episode_id: str,
    query: CupMemQuery,
    checkpoint_reused: bool,
) -> dict[str, object]:
    """Restore an immutable timeline state and execute one independent query."""

    restore_cupmem_full_timeline_checkpoint(engine=engine, checkpoint=checkpoint)
    query_result = engine.answer_query(query_label=query.label, query_text=query.text)
    answer = query_result.get("answer", {})
    answer_text = (
        str(answer.get("answer", "")) if isinstance(answer, Mapping) else ""
    )
    normalization = checkpoint.get("session_normalization", {})
    return {
        "schema_version": "cupmem_full_benchmark_episode_v2",
        "method": CUPMEM_FULL_ARM,
        "display_name": CUPMEM_DISPLAY_NAME,
        "implementation_kind": CUPMEM_SYSTEM_IMPLEMENTATION_KIND,
        "upstream": {
            "repository": CUPMEM_UPSTREAM_REPOSITORY,
            "commit": CUPMEM_UPSTREAM_COMMIT,
            "vendored_code_modified": False,
        },
        "benchmark": checkpoint.get("benchmark", ""),
        "episode_id": episode_id,
        "gold_visible_to_actor": False,
        "benchmark_labels_visible_to_actor": False,
        "departure_source_session_id": checkpoint.get(
            "departure_source_session_id"
        ),
        "timeline_checkpoint": {
            "schema_version": CUPMEM_TIMELINE_CHECKPOINT_SCHEMA,
            "fingerprint": checkpoint.get("fingerprint", ""),
            "reused": bool(checkpoint_reused),
            "query_isolated_by_restore": True,
        },
        "session_normalization": deepcopy(normalization),
        "session_logs": deepcopy(checkpoint.get("session_logs", [])),
        "final_profile_snapshot": engine.store.to_snapshot(),
        "chunk_bank": [engine._chunk_trace_dict(chunk) for chunk in engine.chunk_bank],
        "delta_store": [engine._delta_trace_dict(delta) for delta in engine.delta_store],
        "query_logs": {query.label: query_result},
        "answers": {query.label: answer_text},
        "llm_usage": {
            "timeline_write": deepcopy(checkpoint.get("write_llm_usage", {})),
            "query": (
                engine.llm.get_usage_summary()
                if hasattr(engine.llm, "get_usage_summary")
                else {}
            ),
            "timeline_write_billed_this_episode": not checkpoint_reused,
            "timeline_write_amortized_across_queries": True,
        },
    }


def _split_text_label_blind(text: str, max_chars: int) -> tuple[str, ...]:
    normalized = str(text or "").strip()
    if not normalized:
        return ()
    if len(normalized) <= max_chars:
        return (normalized,)
    pieces: list[str] = []
    cursor = 0
    while cursor < len(normalized):
        end = min(len(normalized), cursor + max_chars)
        if end < len(normalized):
            boundary = max(
                normalized.rfind("\n", cursor, end),
                normalized.rfind(". ", cursor, end),
                normalized.rfind("。", cursor, end),
            )
            if boundary > cursor + max_chars // 2:
                end = boundary + 1
        pieces.append(normalized[cursor:end].strip())
        cursor = end
    return tuple(piece for piece in pieces if piece)


def normalize_sessions_label_blind(
    sessions: Sequence[CupMemSession],
    *,
    max_user_chars_per_session: int = 28_000,
) -> tuple[tuple[CupMemSession, ...], tuple[dict[str, object], ...]]:
    """Split oversized actor-visible text without inspecting task labels."""

    if max_user_chars_per_session < 512:
        raise ValueError("max_user_chars_per_session is too small")
    normalized: list[CupMemSession] = []
    mapping: list[dict[str, object]] = []
    for session in sessions:
        user_fragments: list[str] = []
        for message in session.messages:
            if str(message.get("role", "")).strip() != "user":
                continue
            user_fragments.extend(
                _split_text_label_blind(
                    str(message.get("content", "")),
                    max_user_chars_per_session,
                )
            )
        if not user_fragments:
            continue
        packed: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for fragment in user_fragments:
            separator_chars = 2 if current else 0
            if current and current_chars + separator_chars + len(fragment) > max_user_chars_per_session:
                packed.append(current)
                current = []
                current_chars = 0
            current.append(fragment)
            current_chars += separator_chars + len(fragment)
        if current:
            packed.append(current)
        output_indices: list[int] = []
        for part_index, fragments in enumerate(packed):
            output_indices.append(len(normalized))
            normalized.append(
                CupMemSession(
                    source_session_id=(
                        session.source_session_id
                        if len(packed) == 1
                        else f"{session.source_session_id}#part{part_index + 1}"
                    ),
                    timestamp=session.timestamp,
                    messages=({"role": "user", "content": "\n\n".join(fragments)},),
                    phase=session.phase,
                )
            )
        mapping.append(
            {
                "source_session_id": session.source_session_id,
                "phase": session.phase,
                "normalized_session_indices": output_indices,
                "part_count": len(output_indices),
            }
        )
    return tuple(normalized), tuple(mapping)


def _run_cupmem_full_episode_once(
    *,
    engine: CupMemEngine,
    episode_id: str,
    benchmark: str,
    sessions: Sequence[CupMemSession],
    queries: Sequence[CupMemQuery],
    departure_source_session_id: str | None,
    max_user_chars_per_session: int = 28_000,
) -> dict[str, object]:
    """Execute one CUPMem attempt at a fixed label-blind chunk size."""

    engine.reset()
    if hasattr(engine.llm, "reset_usage_tracking"):
        engine.llm.reset_usage_tracking()
    normalized, normalization_map = normalize_sessions_label_blind(
        sessions,
        max_user_chars_per_session=max_user_chars_per_session,
    )
    session_logs: list[dict[str, object]] = []
    for index, session in enumerate(normalized):
        log = engine.process_session(
            session=[dict(message) for message in session.messages],
            session_index=index,
            session_time=session.timestamp,
        )
        log["adapter_source_session_id"] = session.source_session_id
        log["adapter_phase"] = session.phase
        session_logs.append(log)
    query_logs: dict[str, object] = {}
    answers: dict[str, str] = {}
    for query in queries:
        result = engine.answer_query(query_label=query.label, query_text=query.text)
        query_logs[query.label] = result
        answer = result.get("answer", {})
        answers[query.label] = (
            str(answer.get("answer", "")) if isinstance(answer, Mapping) else ""
        )
    return {
        "schema_version": "cupmem_full_benchmark_episode_v1",
        "method": CUPMEM_FULL_ARM,
        "display_name": CUPMEM_DISPLAY_NAME,
        "implementation_kind": CUPMEM_SYSTEM_IMPLEMENTATION_KIND,
        "upstream": {
            "repository": CUPMEM_UPSTREAM_REPOSITORY,
            "commit": CUPMEM_UPSTREAM_COMMIT,
            "vendored_code_modified": False,
        },
        "benchmark": benchmark,
        "episode_id": episode_id,
        "gold_visible_to_actor": False,
        "benchmark_labels_visible_to_actor": False,
        "departure_source_session_id": departure_source_session_id,
        "session_normalization": {
            "policy": "label_blind_actor_text_size_v1",
            "max_user_chars_per_session": max_user_chars_per_session,
            "mapping": list(normalization_map),
        },
        "session_logs": session_logs,
        "final_profile_snapshot": engine.store.to_snapshot(),
        "chunk_bank": [engine._chunk_trace_dict(chunk) for chunk in engine.chunk_bank],
        "delta_store": [engine._delta_trace_dict(delta) for delta in engine.delta_store],
        "query_logs": query_logs,
        "answers": answers,
        "llm_usage": (
            engine.llm.get_usage_summary()
            if hasattr(engine.llm, "get_usage_summary")
            else {}
        ),
    }


def run_cupmem_full_episode(
    *,
    engine: CupMemEngine,
    episode_id: str,
    benchmark: str,
    sessions: Sequence[CupMemSession],
    queries: Sequence[CupMemQuery],
    departure_source_session_id: str | None,
    max_user_chars_per_session: int = 12_000,
    min_user_chars_per_session: int = 3_000,
    adaptive_session_bisection: bool = True,
) -> dict[str, object]:
    """Execute CUPMem with label-blind context-budget recovery.

    The official method remains untouched. Only the benchmark adapter's
    actor-visible session fragments are bisected when the backend reports
    that even its dynamically reduced completion budget cannot fit.
    """

    if min_user_chars_per_session < 512:
        raise ValueError("min_user_chars_per_session is too small")
    if min_user_chars_per_session > max_user_chars_per_session:
        raise ValueError(
            "min_user_chars_per_session cannot exceed the initial maximum"
        )

    attempt_limits: list[int] = []
    current_limit = int(max_user_chars_per_session)
    while True:
        attempt_limits.append(current_limit)
        try:
            result = _run_cupmem_full_episode_once(
                engine=engine,
                episode_id=episode_id,
                benchmark=benchmark,
                sessions=sessions,
                queries=queries,
                departure_source_session_id=departure_source_session_id,
                max_user_chars_per_session=current_limit,
            )
            result["context_budget_adapter"] = {
                "version": CUPMEM_CONTEXT_BUDGET_ADAPTER_VERSION,
                "label_blind": True,
                "adaptive_session_bisection": bool(
                    adaptive_session_bisection
                ),
                "session_max_chars_attempts": attempt_limits,
                "final_session_max_chars": current_limit,
            }
            return result
        except ContextWindowExceededError:
            if not adaptive_session_bisection:
                raise
            next_limit = max(
                int(min_user_chars_per_session),
                current_limit // 2,
            )
            if next_limit >= current_limit:
                raise
            current_limit = next_limit


# Canonical API for new code. The ``*_full`` spellings remain compatibility
# aliases for active launchers and immutable historical artifacts.
CupMemFullEngine = CupMemEngine
build_cupmem_timeline_checkpoint = build_cupmem_full_timeline_checkpoint
cupmem_timeline_checkpoint_lock = cupmem_full_timeline_checkpoint_lock
cupmem_timeline_fingerprint = cupmem_full_timeline_fingerprint
load_cupmem_timeline_checkpoint = load_cupmem_full_timeline_checkpoint
restore_cupmem_timeline_checkpoint = restore_cupmem_full_timeline_checkpoint
run_cupmem_episode = run_cupmem_full_episode
run_cupmem_query_from_checkpoint = run_cupmem_full_query_from_checkpoint
save_cupmem_timeline_checkpoint = save_cupmem_full_timeline_checkpoint


__all__ = [
    "CUPMEM_CONTEXT_BUDGET_ADAPTER_VERSION",
    "CUPMEM_DISPLAY_NAME",
    "CUPMEM_FULL_ARM",
    "CUPMEM_SYSTEM_ARM",
    "CUPMEM_SYSTEM_IMPLEMENTATION_KIND",
    "CUPMEM_FULL_DISPLAY_NAME",
    "CUPMEM_FULL_IMPLEMENTATION_KIND",
    "CUPMEM_METHOD",
    "CUPMEM_TIMELINE_CHECKPOINT_SCHEMA",
    "CUPMEM_UPSTREAM_COMMIT",
    "CupMemFullEngine",
    "CupMemEngine",
    "CupMemQuery",
    "CupMemSession",
    "LockedLLMClient",
    "RemoteEmbeddingRetriever",
    "build_cupmem_full_timeline_checkpoint",
    "build_cupmem_timeline_checkpoint",
    "cupmem_timeline_checkpoint_lock",
    "cupmem_timeline_fingerprint",
    "cupmem_full_timeline_checkpoint_lock",
    "cupmem_full_timeline_fingerprint",
    "load_cupmem_full_timeline_checkpoint",
    "load_cupmem_timeline_checkpoint",
    "normalize_sessions_label_blind",
    "restore_cupmem_full_timeline_checkpoint",
    "restore_cupmem_timeline_checkpoint",
    "run_cupmem_episode",
    "run_cupmem_full_episode",
    "run_cupmem_full_query_from_checkpoint",
    "run_cupmem_query_from_checkpoint",
    "save_cupmem_timeline_checkpoint",
    "save_cupmem_full_timeline_checkpoint",
]
