#!/usr/bin/env python3
"""Run configured paired MAS Return methods on official STALE Type-II records."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace.configuration import resolve_provider_config
from trace.providers import (  # noqa: E402
    Completion,
    OpenAICompatibleEmbeddingClient,
    OpenAICompatibleProvider,
    endpoint_origin,
)
from trace.stale_type2_methods import (  # noqa: E402
    CUPMEM_ARM,
    KMU_OP_ARM,
    SESSION_OBSERVATION_KEY_PREFIX,
    SESSION_TRANSPORT_PAGE_KEY,
    STALE_ALL_ARMS,
    TRACE_ARM,
    StaleArmSelection,
    StaleMemoryItem,
    answer_prompt,
    chunk_actor_sessions,
    chunk_cupmem_candidates,
    chunk_kmu_candidates,
    chunk_trace_candidate_observations,
    chunk_trace_later_observations,
    compile_trace_dependency_edges,
    cupmem_prompt,
    deterministic_arm_selections,
    expand_trace_retrieval,
    extraction_prompt,
    full_history_answer_prompt,
    memory_items_sha256,
    merge_cupmem_chunk_selections,
    kmu_prompt,
    kmu_chunk_index_contract,
    merge_kmu_chunk_selections,
    parse_answers,
    parse_cupmem_selection,
    parse_extraction,
    parse_indexed_kmu_selection,
    parse_judge,
    paginate_actor_sessions,
    parse_trace_selection,
    parse_trace_dependency_candidates,
    parse_trace_dependency_edges,
    parse_trace_dependency_verification,
    retrieve_for_queries,
    session_observation_items,
    trace_dependency_candidate_prompt,
    trace_dependency_scan_prompt,
    trace_dependency_verification_prompt,
    trace_prompt,
)
from trace.stale_trace_ablations import (  # noqa: E402
    STALE_TRACE_ABLATION_ARMS,
    TRACE_INVALIDATOR_EXPANSION_ARMS,
    TRACE_LIFECYCLE_RESOLUTION_ARMS,
    TRACE_WITHOUT_PROVENANCE_ARM,
    compile_stale_trace_ablations,
)
from trace.stale_type2_return import (  # noqa: E402
    OFFICIAL_JUDGE_SYSTEM_PROMPT,
    StaleType2MasReturnEpisode,
    StaleType2Record,
    STALE_SESSION_COUNT,
    actor_views,
    bind_records_to_sidecar,
    canonical_sha256,
    leakage_audit,
    load_official_stale_type2,
    load_sidecar_manifest,
    official_judge_user_prompt,
)
from trace.unified_mas_contract import (  # noqa: E402
    unified_mas_episode_record,
    validate_unified_mas_config,
)
SCHEMA = "stale_type2_mas_return_run_v7"
EXTRACTION_TRANSPORT_CONTRACT = (
    "stale_type2_extraction_transport_v6_session_index_enum_"
    "repair_preserving"
)
STALE_RUNNER_ARMS = STALE_ALL_ARMS
DIAGNOSTIC_ORACLES = (
    "full_history_oracle",
    "retrieval_oracle",
    "dependency_oracle",
)
DEFAULT_CONFIG = (
    ROOT / "configs" / "models/qwen35_122b_a10b/stale_type2.json"
)
_PRINT_LOCK = threading.Lock()


def _bucket_bound_records(
    bound: Sequence[tuple[StaleType2Record, StaleType2MasReturnEpisode]],
    config: Mapping[str, object],
) -> tuple[tuple[StaleType2Record, StaleType2MasReturnEpisode], ...]:
    """Apply the label-blind natural departure stratum declared by a config."""

    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        return tuple(bound)
    bucket = dataset.get("natural_departure_bucket")
    if not isinstance(bucket, Mapping):
        return tuple(bound)
    inclusive_range = bucket.get("inclusive_range")
    if not isinstance(inclusive_range, Sequence) or isinstance(
        inclusive_range, (str, bytes)
    ) or len(inclusive_range) != 2:
        raise ValueError("natural departure bucket requires inclusive_range")
    lower, upper = int(inclusive_range[0]), int(inclusive_range[1])
    selected = tuple(
        pair
        for pair in bound
        if lower <= pair[0].old_session_index <= upper
    )
    configured_records = dataset.get("records")
    if configured_records is not None and len(selected) != int(configured_records):
        raise ValueError(
            "natural departure bucket count mismatch: "
            f"configured={configured_records} selected={len(selected)}"
        )
    return selected


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _paired_arm_seed(uid: str, stage: str) -> int:
    """Hold stochastic transport fixed across matched method arms."""

    return _seed(uid, "paired_method_arm_seed_v1", stage)


def _completion_record(completion: Completion) -> dict[str, object]:
    return {
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "total_tokens": completion.total_tokens,
        "latency_seconds": completion.latency_seconds,
        "text_sha256": hashlib.sha256(completion.text.encode("utf-8")).hexdigest(),
    }


def _source_contract() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "trace" / "stale_type2_methods.py",
        ROOT / "src" / "trace" / "stale_trace_ablations.py",
        ROOT / "src" / "trace" / "return_methods" / "base.py",
        ROOT
        / "src"
        / "trace"
        / "return_methods"
        / "lifecycle"
        / "temporal_lww.py",
        ROOT
        / "src"
        / "trace"
        / "return_methods"
        / "lifecycle"
        / "kmu_op.py",
        ROOT
        / "src"
        / "trace"
        / "return_methods"
        / "temporal"
        / "memstrata.py",
        ROOT
        / "src"
        / "trace"
        / "return_methods"
        / "semantic_state"
        / "cupmem.py",
        ROOT
        / "src"
        / "trace"
        / "return_methods"
        / "transactional"
        / "memtx.py",
        ROOT / "src" / "trace" / "stale_type2_return.py",
        ROOT / "src" / "trace" / "providers.py",
        ROOT / "src" / "trace" / "unified_mas_contract.py",
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _write_json_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.replace(temporary, path)
    except PermissionError:
        # Object-storage/NFS mounts may reject an otherwise valid atomic rename.
        # The summary is derived and recoverable; stage/result receipts remain
        # immutable O_EXCL files.
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink()


def _read_stage(path: Path, input_sha256: str) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"cache stage is not an object: {path}")
    if value.get("input_sha256") != input_sha256:
        raise ValueError(f"cache input contract changed: {path}")
    return value


def _read_legacy_trace_candidate_cache(
    path: Path,
    *,
    expected_later_memory_ids: Sequence[str],
) -> Mapping[str, object] | None:
    """Read a validated v4 candidate result for transport-only migration."""

    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return None
    if value.get("schema_version") != (
        "stale_type2_trace_dependency_candidate_cache_v4"
    ):
        return None
    expected = [str(item) for item in expected_later_memory_ids]
    if value.get("classified_later_memory_ids") != expected:
        return None
    candidates = value.get("candidate_memory_ids")
    if not isinstance(candidates, Sequence) or isinstance(
        candidates, (str, bytes)
    ):
        return None
    allowed = set(expected)
    if any(str(candidate) not in allowed for candidate in candidates):
        return None
    return value


def _read_legacy_trace_verification_cache(
    path: Path,
    *,
    input_sha256: str,
    target_memory_id: str,
    candidate_memory_id: str,
) -> Mapping[str, object] | None:
    """Read only an exactly bound v4 verdict stored under the legacy path."""

    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return None
    if value.get("schema_version") != (
        "stale_type2_trace_dependency_verification_cache_v4"
    ):
        return None
    if value.get("input_sha256") != input_sha256:
        return None
    if value.get("target_memory_id") != target_memory_id:
        return None
    if value.get("candidate_memory_id") != candidate_memory_id:
        return None
    if not isinstance(value.get("accepted"), bool):
        return None
    if value.get("query_visible") is not False:
        return None
    if value.get("oracle_fields_visible") is not False:
        return None
    return value


def _extraction_tool_schema(
    max_items: int,
    *,
    allowed_session_indices: Sequence[int],
) -> dict[str, object]:
    """Return the transport schema for label-blind state extraction."""

    if max_items < 1:
        raise ValueError("max_items must be positive")
    session_indices = sorted({int(index) for index in allowed_session_indices})
    if not session_indices:
        raise ValueError("allowed_session_indices must not be empty")
    return {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "session_index": {
                            "type": "integer",
                            "enum": session_indices,
                        },
                        "state_key": {"type": "string"},
                        "statement": {"type": "string"},
                        "operation": {
                            "type": "string",
                            "enum": ["add", "update", "delete"],
                        },
                        "evidence_quote": {"type": "string"},
                    },
                    "required": [
                        "session_index",
                        "state_key",
                        "statement",
                        "operation",
                        "evidence_quote",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["memories"],
        "additionalProperties": False,
    }


def _answer_tool_schema() -> dict[str, object]:
    """Return the transport schema for the three official STALE answers."""

    keys = ("dim1_response", "dim2_response", "dim3_response")
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key in keys},
        "required": list(keys),
        "additionalProperties": False,
    }


def _trace_candidate_scan_token_budget(
    configured_max_tokens: int,
    candidate_count: int,
) -> int:
    """Reserve enough output for one decision per supplied candidate.

    This is a transport-only budget derived from the visible candidate count;
    it does not inspect benchmark labels, answers, or model output.
    """

    if configured_max_tokens < 1:
        raise ValueError("configured_max_tokens must be positive")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    return max(configured_max_tokens, 256 + 192 * candidate_count)


def _call_and_parse(
    provider: OpenAICompatibleProvider,
    *,
    prompt: str,
    system: str,
    seed: int,
    max_tokens: int,
    retries: int,
    parse: Callable[[str], Any],
    forced_tool: tuple[str, str, Mapping[str, object]] | None = None,
    structured_output_mode: str = "forced_tool",
) -> tuple[Any, list[dict[str, object]]]:
    if structured_output_mode not in {"forced_tool", "prompt_json"}:
        raise ValueError("invalid actor structured output mode")
    receipts: list[dict[str, object]] = []
    last_error: Exception | None = None
    repair_feedback: str | None = None
    for attempt in range(retries):
        effective_prompt = prompt
        if repair_feedback is not None:
            effective_prompt += (
                "\n\nCORRECTION REQUIRED: the previous response failed strict "
                "validation with this error: "
                + repair_feedback
                + ". Produce a fresh complete JSON object. Correct the named "
                "field, preserve every required row, use only supplied local "
                "indices or enum values, and output no prose or markdown."
            )
        try:
            if forced_tool is None or structured_output_mode == "prompt_json":
                if forced_tool is not None:
                    _, _, tool_parameters = forced_tool
                    effective_prompt = (
                        effective_prompt
                        + "\n\nReturn only one JSON object matching this schema: "
                        + json.dumps(
                            tool_parameters,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                completion = provider.complete(
                    effective_prompt,
                    system=system,
                    seed=seed + attempt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                structured_output_mode = "prompt_json"
            else:
                tool_name, tool_description, tool_parameters = forced_tool
                completion = provider.complete_with_forced_tool(
                    effective_prompt,
                    tool_name=tool_name,
                    tool_description=tool_description,
                    tool_parameters=tool_parameters,
                    system=system,
                    seed=seed + attempt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                actual_structured_output_mode = (
                    completion.transport_mode or "forced_tool"
                )
            if forced_tool is None or structured_output_mode == "prompt_json":
                actual_structured_output_mode = "prompt_json"
        except RuntimeError as error:
            last_error = error
            receipts.append(
                {
                    "attempt": attempt,
                    "prompt_sha256": hashlib.sha256(
                        effective_prompt.encode("utf-8")
                    ).hexdigest(),
                    "structured_output_mode": (
                        "forced_tool"
                        if forced_tool is not None
                        and structured_output_mode == "forced_tool"
                        else "prompt_json"
                    ),
                    "provider_error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        receipt = {
            "attempt": attempt,
            "prompt_sha256": hashlib.sha256(
                effective_prompt.encode("utf-8")
            ).hexdigest(),
            "completion": _completion_record(completion),
            "structured_output_mode": actual_structured_output_mode,
        }
        receipts.append(receipt)
        try:
            return parse(completion.text), receipts
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            receipt["parse_error"] = f"{type(error).__name__}: {error}"
            repair_feedback = (
                f"{type(error).__name__}: {error}"
            )[:600]
    raise ValueError(f"structured model stage failed: {last_error}")


def _provider(
    section: Mapping[str, object],
    execution: Mapping[str, object],
    *,
    request_lock_path: str | None = None,
) -> OpenAICompatibleProvider:
    env_name = str(section.get("api_key_env") or "").strip()
    api_key = os.environ.get(env_name) if env_name else None
    if env_name and not api_key:
        raise RuntimeError(f"missing API key environment variable: {env_name}")
    return OpenAICompatibleProvider(
        generation_base_url=str(section["api_base"]),
        generation_model=str(section["served_model_id"]),
        generation_api_key=api_key,
        timeout=float(section.get("model_timeout_seconds", execution["model_timeout_seconds"])),
        retries=int(section.get("request_retries", execution["model_retries"])),
        chat_response_format=(
            {"type": "json_object"}
            if bool(section.get("json_response_format", False))
            else None
        ),
        chat_template_kwargs=(
            section.get("chat_template_kwargs")
            if isinstance(section.get("chat_template_kwargs"), Mapping)
            else None
        ),
        empty_content_retries=int(
            section.get("empty_content_retries", 0)
        ),
        omit_temperature=bool(section.get("omit_temperature", False)),
        request_lock_path=request_lock_path,
        forced_tool_fallback_mode=str(
            section.get("forced_tool_fallback_mode", "disabled")
        ),
    )


def _embedding_provider(section: Mapping[str, object], execution: Mapping[str, object]) -> OpenAICompatibleEmbeddingClient:
    env_name = str(section.get("api_key_env") or "").strip()
    api_key = os.environ.get(env_name) if env_name else None
    if env_name and not api_key:
        raise RuntimeError(f"missing embedding API key environment variable: {env_name}")
    return OpenAICompatibleEmbeddingClient(
        base_url=str(section["api_base"]),
        model=str(section["served_model_id"]),
        api_key=api_key,
        timeout=float(section.get("timeout_seconds", execution["model_timeout_seconds"])),
        retries=int(section.get("request_retries", execution["model_retries"])),
    )


def _health(provider: Any, model: str, role: str) -> Mapping[str, object]:
    status = provider.health()
    if not status.get("ok"):
        raise RuntimeError(f"{role} health check failed: {status}")
    if model not in status.get("models", ()):
        raise RuntimeError(f"{role} model is absent from endpoint: {model}")
    return status


def _selection_from_record(value: Mapping[str, object]) -> StaleArmSelection:
    decisions = value.get("decisions") or ()
    if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
        raise ValueError("cached selection decisions must be an array")
    return StaleArmSelection(
        arm=str(value["arm"]),
        candidate_item_ids=tuple(str(item) for item in value["candidate_item_ids"]),
        selected_item_ids=tuple(str(item) for item in value["selected_item_ids"]),
        decisions=tuple(dict(item) for item in decisions if isinstance(item, Mapping)),
        parse_error=(str(value["parse_error"]) if value.get("parse_error") else None),
    )


def _metrics_from_judgment(judgment: Mapping[str, object]) -> dict[str, object]:
    passes = [
        bool(judgment[f"dim{index}_eval"]["pass"])
        for index in range(1, 4)
    ]
    return {
        "dim1_pass": passes[0],
        "dim2_pass": passes[1],
        "dim3_pass": passes[2],
        "overall_accuracy": sum(passes) / 3.0,
        "all_dimensions_pass": all(passes),
        "via": passes[2],
        "iir": passes[0] and passes[1],
        "iir_component_mean": sum(passes[:2]) / 2.0,
        "lifecycle_success": all(passes),
    }


def _trace_lifecycle_resolutions(
    items: Sequence[StaleMemoryItem],
    selection: StaleArmSelection,
    retrieved: Sequence[StaleMemoryItem],
    *,
    relation_source: str = "trace_verified_dependency_edge",
) -> tuple[Mapping[str, object], ...]:
    """Render only verified edges whose invalidators reached the answer view."""

    item_by_id = {item.memory_id: item for item in items}
    retrieved_ids = {item.memory_id for item in retrieved}
    seen: set[tuple[str, str]] = set()
    resolutions: list[Mapping[str, object]] = []
    for decision in selection.decisions:
        if decision.get("status") != "SUPERSEDED":
            continue
        target_id = str(decision.get("memory_id") or "")
        invalidator_id = str(decision.get("invalidated_by_memory_id") or "")
        target = item_by_id.get(target_id)
        invalidator = item_by_id.get(invalidator_id)
        if (
            target is None
            or invalidator is None
            or invalidator_id not in retrieved_ids
            or not target.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX)
        ):
            continue
        pair = (target_id, invalidator_id)
        if pair in seen:
            continue
        seen.add(pair)
        resolutions.append(
            {
                "status": "SUPERSEDED",
                "historical_target": target.compact_record(quote_chars=700),
                "current_invalidator": invalidator.compact_record(quote_chars=700),
                "relation_source": relation_source,
                "target_receipt_sha256": target.source_session_sha256,
                "invalidator_receipt_sha256": invalidator.source_session_sha256,
            }
        )
    return tuple(resolutions)


class StaleEpisodeRunner:
    def __init__(
        self,
        *,
        output_dir: Path,
        actor: OpenAICompatibleProvider,
        judge: OpenAICompatibleProvider,
        embedding: OpenAICompatibleEmbeddingClient,
        execution: Mapping[str, object],
        arms: Sequence[str],
        config_sha256: str,
        retry_seed_offset: int = 0,
        allow_anchor_fallback: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.actor = actor
        self.judge = judge
        self.embedding = embedding
        self.execution = execution
        self.actor_structured_output_mode = str(
            execution.get("actor_structured_output_mode", "forced_tool")
        )
        if self.actor_structured_output_mode not in {
            "forced_tool",
            "prompt_json",
        }:
            raise ValueError("invalid actor_structured_output_mode")
        self.arms = tuple(arms)
        raw_oracles = execution.get("diagnostic_oracles", ())
        if raw_oracles is True:
            raw_oracles = DIAGNOSTIC_ORACLES
        if raw_oracles in (False, None):
            raw_oracles = ()
        if not isinstance(raw_oracles, Sequence) or isinstance(
            raw_oracles, (str, bytes)
        ):
            raise ValueError("diagnostic_oracles must be an array or boolean")
        self.diagnostic_oracles = tuple(str(item) for item in raw_oracles)
        unknown_oracles = set(self.diagnostic_oracles) - set(DIAGNOSTIC_ORACLES)
        if unknown_oracles:
            raise ValueError(
                "unknown STALE diagnostic oracles: "
                + ",".join(sorted(unknown_oracles))
            )
        if len(self.diagnostic_oracles) != len(set(self.diagnostic_oracles)):
            raise ValueError("diagnostic_oracles must be unique")
        self.config_sha256 = config_sha256
        self.retry_seed_offset = retry_seed_offset
        self.allow_anchor_fallback = allow_anchor_fallback

    def _stage_seed(self, *parts: object) -> int:
        return (_seed(*parts) + self.retry_seed_offset) & 0x7FFFFFFF

    def _extract(
        self,
        record: StaleType2Record,
        episode: StaleType2MasReturnEpisode,
    ) -> tuple[StaleMemoryItem, ...]:
        views = actor_views(record, episode)
        all_items: list[StaleMemoryItem] = []
        covered_session_indices: set[int] = set()
        for agent_id in sorted(views):
            sessions = views[agent_id]["sessions"]
            assert isinstance(sessions, Sequence)
            for session in sessions:
                if not isinstance(session, Mapping):
                    raise TypeError("actor-visible session must be an object")
                session_index = int(session["session_index"])
                covered_session_indices.add(session_index)
                anchors = session_observation_items(
                    uid=record.uid,
                    agent_id=agent_id,
                    session=session,
                )
                all_items.extend(anchors)
            extraction_chunk_chars = min(
                int(self.execution["extraction_chunk_max_chars"]),
                24_000,
            )
            session_pages = paginate_actor_sessions(
                sessions,
                max_chars=extraction_chunk_chars,
            )
            chunks = chunk_actor_sessions(
                session_pages,
                max_chars=extraction_chunk_chars,
                max_sessions=4,
            )
            for chunk_index, chunk in enumerate(chunks):
                uses_transport_pages = any(
                    SESSION_TRANSPORT_PAGE_KEY in session
                    for session in chunk
                )
                allowed_session_indices = tuple(
                    sorted({int(item["session_index"]) for item in chunk})
                )
                max_items = min(
                    int(self.execution["extraction_max_items_per_chunk"]),
                    8,
                )
                tool_schema = _extraction_tool_schema(
                    max_items,
                    allowed_session_indices=allowed_session_indices,
                )
                prompt = extraction_prompt(
                    agent_id=agent_id,
                    sessions=chunk,
                    max_items=max_items,
                )
                input_sha = canonical_sha256(
                    {
                        "stage": (
                            "query_independent_chunk_enrichment_v6_paged"
                            if uses_transport_pages
                            else "query_independent_chunk_enrichment_v6_bounded"
                        ),
                        "model": self.actor.generation_model,
                        "prompt": prompt,
                        "transport_contract": EXTRACTION_TRANSPORT_CONTRACT,
                        "structured_output_mode": (
                            self.actor_structured_output_mode
                        ),
                        "allowed_session_indices": allowed_session_indices,
                        "tool_schema": tool_schema,
                    }
                )
                path = (
                    self.output_dir
                    / "cache"
                    / record.uid
                    / (
                        "extraction_v6_paged"
                        if uses_transport_pages
                        else "extraction_v6_bounded"
                    )
                    / f"{agent_id}_chunk_{chunk_index:03d}.json"
                )
                cached = _read_stage(path, input_sha)
                if cached is None:
                    fallback_error: str | None = None
                    try:
                        parsed, receipts = _call_and_parse(
                            self.actor,
                            prompt=prompt,
                            system=(
                                "You are a private-memory writer. Use only the "
                                "supplied actor-visible sessions and return strict "
                                "JSON."
                            ),
                            seed=self._stage_seed(
                                record.uid,
                                agent_id,
                                chunk_index,
                                (
                                    "extract_v6_paged"
                                    if uses_transport_pages
                                    else "extract_v6_bounded"
                                ),
                            ),
                            max_tokens=int(
                                self.execution["extraction_max_tokens"]
                            ),
                            retries=int(self.execution["format_retries"]),
                            parse=lambda text: parse_extraction(
                                text,
                                uid=record.uid,
                                agent_id=agent_id,
                                sessions=chunk,
                                max_items=max_items,
                                skip_ungrounded=True,
                            ),
                            forced_tool=(
                                "submit_extracted_memories",
                                "Submit the query-independent state memories.",
                                tool_schema,
                            ),
                            structured_output_mode=(
                                self.actor_structured_output_mode
                            ),
                        )
                    except ValueError as error:
                        if not self.allow_anchor_fallback:
                            raise
                        parsed = ()
                        fallback_error = f"{type(error).__name__}: {error}"
                        receipts = [
                            {
                                "structured_output_mode": "anchor_fallback",
                                "fallback_error": fallback_error,
                            }
                        ]
                    cached = {
                        "schema_version": (
                            "stale_type2_chunk_enrichment_cache_v6"
                        ),
                        "input_sha256": input_sha,
                        "transport_contract": EXTRACTION_TRANSPORT_CONTRACT,
                        "uid": record.uid,
                        "agent_id": agent_id,
                        "chunk_index": chunk_index,
                        "session_indices": [
                            int(item["session_index"]) for item in chunk
                        ],
                        "allowed_session_indices": list(
                            allowed_session_indices
                        ),
                        "tool_schema_sha256": canonical_sha256(tool_schema),
                        "deterministic_observation_coverage_external": True,
                        "query_visible": False,
                        "oracle_fields_visible": False,
                        "transport_paged": uses_transport_pages,
                        "fallback_to_session_anchors": fallback_error is not None,
                        "fallback_error": fallback_error,
                        "items": [item.record() for item in parsed],
                        "calls": receipts,
                    }
                    _write_json_once(path, cached)
                raw_items = cached.get("items")
                if not isinstance(raw_items, Sequence):
                    raise ValueError(f"cached extraction omitted items: {path}")
                all_items.extend(
                    StaleMemoryItem.from_record(item)
                    for item in raw_items
                    if isinstance(item, Mapping)
                )
        expected_indices = set(range(STALE_SESSION_COUNT))
        if covered_session_indices != expected_indices:
            raise RuntimeError("per-session extraction did not cover all sessions")
        deduplicated = {
            item.memory_id: item
            for item in sorted(all_items, key=lambda row: (row.session_index, row.memory_id))
        }
        return tuple(deduplicated.values())

    def _trace_dependency_selection(
        self,
        *,
        record: StaleType2Record,
        episode: StaleType2MasReturnEpisode,
        items: Sequence[StaleMemoryItem],
    ) -> tuple[StaleArmSelection, Mapping[str, object]]:
        observations = tuple(
            item
            for item in items
            if item.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX)
        )
        observed_sessions = {item.session_index for item in observations}
        if observed_sessions != set(range(STALE_SESSION_COUNT)):
            raise RuntimeError("TRACE requires observation coverage for all sessions")
        dependency_targets = tuple(
            item
            for item in observations
            if item.agent_id == episode.returning_agent_id
            and item.session_index <= episode.departure_after_session
        )
        if not dependency_targets:
            raise RuntimeError("TRACE requires returning-agent checkpoint memories")
        target_batch_size = int(
            self.execution.get("trace_dependency_target_batch_size", 4)
        )
        if target_batch_size < 1:
            raise ValueError("trace_dependency_target_batch_size must be positive")
        all_edges: list[Mapping[str, object]] = []
        scanned_pairs: set[tuple[str, str]] = set()
        expected_pairs = {
            (target.memory_id, later.memory_id)
            for target in dependency_targets
            for later in observations
            if later.session_index > target.session_index
        }
        scan_call_count = 0
        for start in range(0, len(dependency_targets), target_batch_size):
            targets = dependency_targets[start : start + target_batch_size]
            later_items = tuple(
                item
                for item in observations
                if item.session_index
                > min(target.session_index for target in targets)
            )
            later_chunks = chunk_trace_later_observations(
                targets,
                later_items,
                departure_after_session=episode.departure_after_session,
                max_prompt_chars=int(
                    self.execution["trace_dependency_max_prompt_chars"]
                ),
            )
            for chunk_index, later_chunk in enumerate(later_chunks):
                prompt = trace_dependency_scan_prompt(
                    targets,
                    later_chunk,
                    departure_after_session=episode.departure_after_session,
                )
                if len(prompt) > int(
                    self.execution["trace_dependency_max_prompt_chars"]
                ):
                    raise RuntimeError("TRACE dependency prompt exceeded budget")
                scan_max_tokens = int(
                    self.execution.get(
                        "trace_dependency_scan_max_tokens",
                        self.execution["governance_max_tokens"],
                    )
                )
                tool_parameters = {
                    "type": "object",
                    "properties": {
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_memory_id": {
                                        "type": "string",
                                        "enum": [
                                            item.memory_id for item in targets
                                        ],
                                    },
                                    "invalidated_by_memory_id": {
                                        "type": "string",
                                        "enum": [
                                            item.memory_id
                                            for item in later_chunk
                                        ],
                                    },
                                },
                                "required": [
                                    "target_memory_id",
                                    "invalidated_by_memory_id",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["edges"],
                    "additionalProperties": False,
                }
                input_sha = canonical_sha256(
                    {
                        "stage": "trace_chunked_dependency_edge_scan_v2",
                        "model": self.actor.generation_model,
                        "max_tokens": scan_max_tokens,
                        "prompt": prompt,
                        "structured_output_mode": (
                            self.actor_structured_output_mode
                        ),
                        "tool_parameters": tool_parameters,
                    }
                )
                path = (
                    self.output_dir
                    / "cache"
                    / record.uid
                    / "governance"
                    / f"trace_scan_t{start:03d}_c{chunk_index:03d}.json"
                )
                cached = _read_stage(path, input_sha)
                if cached is None:
                    edges, receipts = _call_and_parse(
                        self.actor,
                        prompt=prompt,
                        system=(
                            "You are the TRACE provenance and lifecycle dependency "
                            "scanner. Use only actor-visible observations and "
                            "return strict JSON."
                        ),
                        seed=self._stage_seed(
                            record.uid,
                            start,
                            chunk_index,
                            "trace_edge_scan_v2",
                        ),
                        max_tokens=scan_max_tokens,
                        retries=int(self.execution["format_retries"]),
                        parse=lambda text, targets=targets, later_chunk=later_chunk: (
                            parse_trace_dependency_edges(
                                text,
                                target_items=targets,
                                later_items=later_chunk,
                            )
                        ),
                        forced_tool=(
                            "emit_trace_dependency_edges",
                            "Return grounded invalidation edges found in this scan.",
                            tool_parameters,
                        ),
                        structured_output_mode=(
                            self.actor_structured_output_mode
                        ),
                    )
                    cached = {
                        "schema_version": (
                            "stale_type2_trace_dependency_edge_scan_cache_v2"
                        ),
                        "input_sha256": input_sha,
                        "uid": record.uid,
                        "target_memory_ids": [
                            item.memory_id for item in targets
                        ],
                        "later_memory_ids": [
                            item.memory_id for item in later_chunk
                        ],
                        "prompt_chars": len(prompt),
                        "query_visible": False,
                        "oracle_fields_visible": False,
                        "edges": [dict(item) for item in edges],
                        "calls": receipts,
                    }
                    _write_json_once(path, cached)
                raw_edges = cached.get("edges")
                if not isinstance(raw_edges, Sequence):
                    raise ValueError("TRACE dependency cache omitted edges")
                all_edges.extend(
                    dict(edge)
                    for edge in raw_edges
                    if isinstance(edge, Mapping)
                )
                for target in targets:
                    for later in later_chunk:
                        if later.session_index <= target.session_index:
                            continue
                        pair = (target.memory_id, later.memory_id)
                        if pair in scanned_pairs:
                            raise RuntimeError("TRACE scanned a dependency pair twice")
                        scanned_pairs.add(pair)
                scan_call_count += 1
        if scanned_pairs != expected_pairs:
            raise RuntimeError("TRACE dependency scan did not cover every pair")

        compiled = compile_trace_dependency_edges(
            dependency_targets,
            observations,
            all_edges,
        )
        observation_decisions = {
            str(row["memory_id"]): dict(row) for row in compiled
        }

        # Absence-period observations were not in the returning agent's stale
        # checkpoint.  They enter the return view as current candidates unless a
        # checkpoint target explicitly resolves to one as its invalidator.
        for item in observations:
            observation_decisions.setdefault(
                item.memory_id,
                {
                    "memory_id": item.memory_id,
                    "status": "ADMIT",
                    "invalidated_by_memory_id": None,
                },
            )

        observation_by_source: dict[str, list[StaleMemoryItem]] = {}
        for item in observations:
            observation_by_source.setdefault(item.source_session_sha256, []).append(item)
        decisions: list[Mapping[str, object]] = []
        for item in items:
            if item.memory_id in observation_decisions:
                decisions.append(observation_decisions[item.memory_id])
                continue
            source_observations = observation_by_source.get(
                item.source_session_sha256, []
            )
            inherited = next(
                (
                    observation_decisions[source.memory_id]
                    for source in source_observations
                    if observation_decisions[source.memory_id]["status"]
                    == "SUPERSEDED"
                ),
                None,
            )
            decisions.append(
                {
                    "memory_id": item.memory_id,
                    "status": (
                        "SUPERSEDED" if inherited is not None else "ADMIT"
                    ),
                    "invalidated_by_memory_id": (
                        inherited["invalidated_by_memory_id"]
                        if inherited is not None
                        else None
                    ),
                }
            )
        selected_ids = tuple(
            str(row["memory_id"])
            for row in decisions
            if row["status"] == "ADMIT"
        )
        return (
            StaleArmSelection(
                arm=TRACE_ARM,
                candidate_item_ids=tuple(item.memory_id for item in items),
                selected_item_ids=selected_ids,
                decisions=tuple(decisions),
            ),
            {
                "schema_version": "trace_dependency_scan_audit_v1",
                "checkpoint_target_count": len(dependency_targets),
                "observation_count": len(observations),
                "eligible_pair_count": len(expected_pairs),
                "scanned_pair_count": len(scanned_pairs),
                "each_pair_scanned_exactly_once": scanned_pairs == expected_pairs,
                "scan_call_count": scan_call_count,
                "candidate_edge_count": len(all_edges),
                "query_visible": False,
                "oracle_fields_visible": False,
            },
        )

    def _trace_dependency_selection_two_stage(
        self,
        *,
        record: StaleType2Record,
        episode: StaleType2MasReturnEpisode,
        items: Sequence[StaleMemoryItem],
    ) -> tuple[StaleArmSelection, Mapping[str, object]]:
        """Compile TRACE edges with a complete small-chunk scan and verifier."""

        observations = tuple(
            item
            for item in items
            if item.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX)
        )
        if {item.session_index for item in observations} != set(
            range(STALE_SESSION_COUNT)
        ):
            raise RuntimeError("TRACE requires observation coverage for all sessions")
        dependency_targets = tuple(
            item
            for item in observations
            if item.agent_id == episode.returning_agent_id
            and item.session_index <= episode.departure_after_session
        )
        if not dependency_targets:
            raise RuntimeError("TRACE requires returning-agent checkpoint memories")
        max_items_per_chunk = int(
            self.execution.get("trace_dependency_later_items_per_chunk", 4)
        )
        if max_items_per_chunk < 1:
            raise ValueError(
                "trace_dependency_later_items_per_chunk must be positive"
            )
        max_prompt_chars = int(
            self.execution.get("trace_dependency_max_prompt_chars", 24000)
        )
        expected_pairs = {
            (target.memory_id, later.memory_id)
            for target in dependency_targets
            for later in observations
            if later.session_index > target.session_index
        }
        scanned_pairs: set[tuple[str, str]] = set()
        proposed_pairs: set[tuple[str, str]] = set()
        scan_call_count = 0

        for target_index, target in enumerate(dependency_targets):
            later_items = tuple(
                item
                for item in observations
                if item.session_index > target.session_index
            )
            later_chunks = chunk_trace_candidate_observations(
                target,
                later_items,
                departure_after_session=episode.departure_after_session,
                max_prompt_chars=max_prompt_chars,
                max_items_per_chunk=max_items_per_chunk,
            )
            for chunk_index, later_chunk in enumerate(later_chunks):
                prompt = trace_dependency_candidate_prompt(
                    target,
                    later_chunk,
                    departure_after_session=episode.departure_after_session,
                )
                if len(prompt) > max_prompt_chars:
                    raise RuntimeError("TRACE candidate prompt exceeded budget")
                scan_max_tokens = _trace_candidate_scan_token_budget(
                    int(
                        self.execution.get(
                            "trace_dependency_scan_max_tokens",
                            self.execution["governance_max_tokens"],
                        )
                    ),
                    len(later_chunk),
                )
                tool_parameters = {
                    "type": "object",
                    "properties": {
                        "decisions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "later_index": {
                                        "type": "string",
                                        "enum": [
                                            str(index)
                                            for index, _ in enumerate(later_chunk)
                                        ],
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": [
                                            "POSSIBLE_INVALIDATOR",
                                            "NO_EVIDENCE",
                                        ],
                                    },
                                },
                                "required": [
                                    "later_index",
                                    "status",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["decisions"],
                    "additionalProperties": False,
                }
                input_sha = canonical_sha256(
                    {
                        "stage": (
                            "trace_dependency_candidate_scan_v5_"
                            "string_local_index"
                        ),
                        "model": self.actor.generation_model,
                        "max_tokens": scan_max_tokens,
                        "prompt": prompt,
                        "structured_output_mode": (
                            self.actor_structured_output_mode
                        ),
                        "tool_parameters": tool_parameters,
                    }
                )
                path = (
                    self.output_dir
                    / "cache"
                    / record.uid
                    / "governance_v4"
                    / f"candidate_t{target_index:03d}_c{chunk_index:03d}.json"
                )
                expected_chunk_ids = [item.memory_id for item in later_chunk]
                cached = _read_stage(path, input_sha)
                if cached is None:
                    legacy_path = (
                        self.output_dir
                        / "cache"
                        / record.uid
                        / "governance_v3"
                        / (
                            f"candidate_t{target_index:03d}_"
                            f"c{chunk_index:03d}.json"
                        )
                    )
                    legacy = _read_legacy_trace_candidate_cache(
                        legacy_path,
                        expected_later_memory_ids=expected_chunk_ids,
                    )
                    if legacy is not None:
                        cached = {
                            "schema_version": (
                                "stale_type2_trace_dependency_candidate_cache_v5"
                            ),
                            "input_sha256": input_sha,
                            "uid": record.uid,
                            "target_memory_id": target.memory_id,
                            "classified_later_memory_ids": expected_chunk_ids,
                            "candidate_memory_ids": list(
                                legacy["candidate_memory_ids"]
                            ),
                            "prompt_chars": len(prompt),
                            "max_tokens": scan_max_tokens,
                            "query_visible": False,
                            "oracle_fields_visible": False,
                            "calls": list(legacy.get("calls") or ()),
                            "transport_migrated_from": (
                                "integer_local_index_v4"
                            ),
                        }
                        _write_json_once(path, cached)
                if cached is None:
                    candidate_ids, receipts = _call_and_parse(
                        self.actor,
                        prompt=prompt,
                        system=(
                            "You are the TRACE high-recall lifecycle candidate "
                            "scanner. Classify every supplied later observation "
                            "and return strict JSON."
                        ),
                        seed=self._stage_seed(
                            record.uid,
                            target_index,
                            chunk_index,
                            "trace_candidate_scan_v5",
                        ),
                        max_tokens=scan_max_tokens,
                        retries=int(self.execution["format_retries"]),
                        parse=lambda text, target=target, later_chunk=later_chunk: (
                            parse_trace_dependency_candidates(
                                text,
                                target_item=target,
                                later_items=later_chunk,
                            )
                        ),
                        forced_tool=(
                            "classify_trace_dependency_candidates",
                            "Classify every grounded later observation.",
                            tool_parameters,
                        ),
                        structured_output_mode=self.actor_structured_output_mode,
                    )
                    cached = {
                        "schema_version": (
                            "stale_type2_trace_dependency_candidate_cache_v5"
                        ),
                        "input_sha256": input_sha,
                        "uid": record.uid,
                        "target_memory_id": target.memory_id,
                        "classified_later_memory_ids": [
                            item.memory_id for item in later_chunk
                        ],
                        "candidate_memory_ids": list(candidate_ids),
                        "prompt_chars": len(prompt),
                        "max_tokens": scan_max_tokens,
                        "query_visible": False,
                        "oracle_fields_visible": False,
                        "calls": receipts,
                    }
                    _write_json_once(path, cached)
                classified = cached.get("classified_later_memory_ids")
                candidate_ids = cached.get("candidate_memory_ids")
                if classified != expected_chunk_ids:
                    raise ValueError(
                        "TRACE candidate cache omitted classified observations"
                    )
                if not isinstance(candidate_ids, Sequence) or isinstance(
                    candidate_ids, (str, bytes)
                ):
                    raise ValueError("TRACE candidate cache omitted candidates")
                allowed = set(expected_chunk_ids)
                for candidate_id in candidate_ids:
                    pair = (target.memory_id, str(candidate_id))
                    if str(candidate_id) not in allowed:
                        raise ValueError("TRACE cache contains unknown candidate")
                    if pair in proposed_pairs:
                        raise RuntimeError("TRACE proposed a dependency pair twice")
                    proposed_pairs.add(pair)
                for later in later_chunk:
                    pair = (target.memory_id, later.memory_id)
                    if pair in scanned_pairs:
                        raise RuntimeError("TRACE scanned a dependency pair twice")
                    scanned_pairs.add(pair)
                scan_call_count += 1

        if scanned_pairs != expected_pairs:
            raise RuntimeError("TRACE candidate scan did not cover every pair")

        observation_by_id = {item.memory_id: item for item in observations}
        verified_pairs: set[tuple[str, str]] = set()
        verified_edges: list[Mapping[str, object]] = []
        verification_call_count = 0
        for pair_index, (target_id, candidate_id) in enumerate(
            sorted(
                proposed_pairs,
                key=lambda pair: (
                    observation_by_id[pair[0]].session_index,
                    observation_by_id[pair[1]].session_index,
                    pair,
                ),
            )
        ):
            current_pair = (target_id, candidate_id)
            target = observation_by_id[target_id]
            candidate = observation_by_id[candidate_id]
            prompt = trace_dependency_verification_prompt(
                target,
                candidate,
                departure_after_session=episode.departure_after_session,
            )
            tool_parameters = {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "INVALIDATES_CURRENT_USE",
                            "DOES_NOT_INVALIDATE",
                        ],
                    },
                },
                "required": [
                    "verdict",
                ],
                "additionalProperties": False,
            }
            input_sha = canonical_sha256(
                {
                    "stage": "trace_dependency_pair_verification_v4_bound_pair",
                    "model": self.actor.generation_model,
                    "prompt": prompt,
                    "tool_parameters": tool_parameters,
                }
            )
            path = (
                self.output_dir
                / "cache"
                / record.uid
                / "governance_v4"
                / f"verify_{pair_index:05d}.json"
            )
            cached = _read_stage(path, input_sha)
            if cached is None:
                legacy_path = (
                    self.output_dir
                    / "cache"
                    / record.uid
                    / "governance_v3"
                    / f"verify_{pair_index:05d}.json"
                )
                legacy = _read_legacy_trace_verification_cache(
                    legacy_path,
                    input_sha256=input_sha,
                    target_memory_id=target_id,
                    candidate_memory_id=candidate_id,
                )
                if legacy is not None:
                    cached = {
                        **dict(legacy),
                        "transport_migrated_from": "governance_v3_path",
                    }
                    _write_json_once(path, cached)
            if cached is None:
                accepted, receipts = _call_and_parse(
                    self.actor,
                    prompt=prompt,
                    system=(
                        "You are the TRACE grounded lifecycle verifier. Verify "
                        "only the supplied provenance-bound pair and return "
                        "strict JSON."
                    ),
                    seed=self._stage_seed(
                        record.uid,
                        target_id,
                        candidate_id,
                        "trace_pair_verification_v3",
                    ),
                    max_tokens=int(
                        self.execution.get(
                            "trace_dependency_verification_max_tokens", 512
                        )
                    ),
                    retries=int(self.execution["format_retries"]),
                    parse=lambda text, target=target, candidate=candidate: (
                        parse_trace_dependency_verification(
                            text,
                            target_item=target,
                            candidate_item=candidate,
                        )
                    ),
                    forced_tool=(
                        "verify_trace_dependency",
                        "Verify one grounded lifecycle dependency.",
                        tool_parameters,
                    ),
                    structured_output_mode=self.actor_structured_output_mode,
                )
                cached = {
                    "schema_version": (
                        "stale_type2_trace_dependency_verification_cache_v4"
                    ),
                    "input_sha256": input_sha,
                    "uid": record.uid,
                    "target_memory_id": target_id,
                    "candidate_memory_id": candidate_id,
                    "accepted": bool(accepted),
                    "prompt_chars": len(prompt),
                    "query_visible": False,
                    "oracle_fields_visible": False,
                    "calls": receipts,
                }
                _write_json_once(path, cached)
            if cached.get("target_memory_id") != target_id or cached.get(
                "candidate_memory_id"
            ) != candidate_id:
                raise ValueError("TRACE verification cache pair mismatch")
            if not isinstance(cached.get("accepted"), bool):
                raise ValueError("TRACE verification cache omitted verdict")
            if current_pair in verified_pairs:
                raise RuntimeError("TRACE verified a dependency pair twice")
            verified_pairs.add(current_pair)
            if cached["accepted"]:
                verified_edges.append(
                    {
                        "target_memory_id": target_id,
                        "invalidated_by_memory_id": candidate_id,
                    }
                )
            verification_call_count += 1

        if verified_pairs != proposed_pairs:
            raise RuntimeError("TRACE did not verify every proposed pair")

        compiled = compile_trace_dependency_edges(
            dependency_targets,
            observations,
            verified_edges,
        )
        observation_decisions = {
            str(row["memory_id"]): dict(row) for row in compiled
        }
        for item in observations:
            observation_decisions.setdefault(
                item.memory_id,
                {
                    "memory_id": item.memory_id,
                    "status": "ADMIT",
                    "invalidated_by_memory_id": None,
                },
            )

        observation_by_source: dict[str, list[StaleMemoryItem]] = {}
        for item in observations:
            observation_by_source.setdefault(item.source_session_sha256, []).append(item)
        decisions: list[Mapping[str, object]] = []
        for item in items:
            if item.memory_id in observation_decisions:
                decisions.append(observation_decisions[item.memory_id])
                continue
            inherited = next(
                (
                    observation_decisions[source.memory_id]
                    for source in observation_by_source.get(
                        item.source_session_sha256, []
                    )
                    if observation_decisions[source.memory_id]["status"]
                    == "SUPERSEDED"
                ),
                None,
            )
            decisions.append(
                {
                    "memory_id": item.memory_id,
                    "status": "SUPERSEDED" if inherited is not None else "ADMIT",
                    "invalidated_by_memory_id": (
                        inherited["invalidated_by_memory_id"]
                        if inherited is not None
                        else None
                    ),
                }
            )
        selected_ids = tuple(
            str(row["memory_id"])
            for row in decisions
            if row["status"] == "ADMIT"
        )
        return (
            StaleArmSelection(
                arm=TRACE_ARM,
                candidate_item_ids=tuple(item.memory_id for item in items),
                selected_item_ids=selected_ids,
                decisions=tuple(decisions),
            ),
            {
                "schema_version": "trace_dependency_scan_audit_v2",
                "mode": "complete_small_chunk_two_stage",
                "checkpoint_target_count": len(dependency_targets),
                "observation_count": len(observations),
                "eligible_pair_count": len(expected_pairs),
                "scanned_pair_count": len(scanned_pairs),
                "each_pair_scanned_exactly_once": scanned_pairs == expected_pairs,
                "scan_call_count": scan_call_count,
                "proposed_edge_count": len(proposed_pairs),
                "verified_pair_count": len(verified_pairs),
                "verification_call_count": verification_call_count,
                "verified_edge_count": len(verified_edges),
                # Backward-compatible name: only verified edges affect TRACE.
                "candidate_edge_count": len(verified_edges),
                "proposed_edges": [
                    {
                        "target_memory_id": target_id,
                        "invalidated_by_memory_id": candidate_id,
                    }
                    for target_id, candidate_id in sorted(proposed_pairs)
                ],
                "all_candidates_verified_once": verified_pairs == proposed_pairs,
                "max_later_items_per_chunk": max_items_per_chunk,
                "max_prompt_chars": max_prompt_chars,
                "query_visible": False,
                "oracle_fields_visible": False,
            },
        )

    def _bounded_cupmem_selection(
        self,
        *,
        record: StaleType2Record,
        episode: StaleType2MasReturnEpisode,
        items: Sequence[StaleMemoryItem],
    ) -> StaleArmSelection:
        """Run CUPMem as bounded cross-phase write-side adjudication."""

        max_prompt_chars = min(
            int(self.execution.get("governance_max_prompt_chars", 56_000)),
            56_000,
        )
        max_text_chars = int(
            self.execution.get("cupmem_candidate_text_chars", 160)
        )
        max_candidates_per_chunk = int(
            self.execution.get("cupmem_max_candidates_per_chunk", 112)
        )
        max_absence_items_per_chunk = int(
            self.execution.get("cupmem_max_absence_items_per_chunk", 96)
        )
        max_tokens = int(
            self.execution.get(
                "cupmem_governance_max_tokens",
                self.execution["governance_max_tokens"],
            )
        )
        chunks = chunk_cupmem_candidates(
            items,
            returning_agent_id=episode.returning_agent_id,
            max_prompt_chars=max_prompt_chars,
            max_text_chars=max_text_chars,
            max_candidates_per_chunk=max_candidates_per_chunk,
            max_absence_items_per_chunk=max_absence_items_per_chunk,
        )
        input_sha = canonical_sha256(
            {
                "stage": "cupmem_governance_v3_bounded_cross_phase",
                "model": self.actor.generation_model,
                "structured_output_mode": self.actor_structured_output_mode,
                "items_sha256": memory_items_sha256(items),
                "max_prompt_chars": max_prompt_chars,
                "max_text_chars": max_text_chars,
                "max_candidates_per_chunk": max_candidates_per_chunk,
                "max_absence_items_per_chunk": (
                    max_absence_items_per_chunk
                ),
                "max_tokens": max_tokens,
                "chunks": [
                    [item.memory_id for item in chunk] for chunk in chunks
                ],
            }
        )
        path = (
            self.output_dir
            / "cache"
            / record.uid
            / "governance"
            / "cupmem_v3_bounded_cross_phase.json"
        )
        cached = _read_stage(path, input_sha)
        if cached is None:
            chunk_selections: list[StaleArmSelection] = []
            chunk_audit: list[Mapping[str, object]] = []
            for chunk_index, chunk in enumerate(chunks):
                prompt = cupmem_prompt(
                    chunk,
                    max_text_chars=max_text_chars,
                )
                if len(prompt) > max_prompt_chars:
                    raise RuntimeError("bounded CUPMem prompt exceeded budget")
                candidate_indices = tuple(range(len(chunk)))
                tool_parameters: Mapping[str, object] = {
                    "type": "object",
                    "properties": {
                        "decisions": {
                            "type": "array",
                            "minItems": len(chunk),
                            "maxItems": len(chunk),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "candidate_index": {
                                        "type": "integer",
                                        "enum": candidate_indices,
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": [
                                            "ACTIVE",
                                            "STALE",
                                            "REPLACED",
                                            "UNKNOWN",
                                        ],
                                    },
                                    "replacement_candidate_index": {
                                        "anyOf": [
                                            {
                                                "type": "integer",
                                                "enum": candidate_indices,
                                            },
                                            {"type": "null"},
                                        ]
                                    },
                                },
                                "required": [
                                    "candidate_index",
                                    "status",
                                    "replacement_candidate_index",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["decisions"],
                    "additionalProperties": False,
                }
                chunk_input_sha = canonical_sha256(
                    {
                        "stage": "cupmem_governance_chunk_v3_local_index",
                        "model": self.actor.generation_model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "structured_output_mode": (
                            self.actor_structured_output_mode
                        ),
                        "tool_parameters": tool_parameters,
                    }
                )
                chunk_path = (
                    self.output_dir
                    / "cache"
                    / record.uid
                    / "governance"
                    / f"cupmem_v3_chunk_{chunk_index:03d}.json"
                )
                chunk_cached = _read_stage(chunk_path, chunk_input_sha)
                if chunk_cached is None:
                    def parse_chunk(
                        text: str,
                        *,
                        chunk: Sequence[StaleMemoryItem] = chunk,
                    ) -> StaleArmSelection:
                        selection = parse_cupmem_selection(text, chunk)
                        if selection.parse_error is not None:
                            raise ValueError(selection.parse_error)
                        return selection

                    selection, receipts = _call_and_parse(
                        self.actor,
                        prompt=prompt,
                        system=(
                            "You are the CUPMem lifecycle classifier. "
                            "Adjudicate this bounded write-side page and "
                            "return strict JSON only."
                        ),
                        seed=self._stage_seed(
                            record.uid,
                            CUPMEM_ARM,
                            f"governance_chunk_{chunk_index:03d}",
                        ),
                        max_tokens=max_tokens,
                        retries=int(self.execution["format_retries"]),
                        parse=parse_chunk,
                        forced_tool=(
                            "emit_cupmem_decisions",
                            "Return one CUPMem status for every supplied candidate.",
                            tool_parameters,
                        ),
                        structured_output_mode=(
                            self.actor_structured_output_mode
                        ),
                    )
                    chunk_cached = {
                        "schema_version": (
                            "stale_type2_cupmem_governance_chunk_cache_v3"
                        ),
                        "input_sha256": chunk_input_sha,
                        "uid": record.uid,
                        "chunk_index": chunk_index,
                        "candidate_item_ids": [
                            item.memory_id for item in chunk
                        ],
                        "prompt_chars": len(prompt),
                        "query_visible": False,
                        "oracle_fields_visible": False,
                        "selection": selection.record(),
                        "calls": receipts,
                    }
                    _write_json_once(chunk_path, chunk_cached)
                raw_selection = chunk_cached.get("selection")
                if not isinstance(raw_selection, Mapping):
                    raise ValueError(
                        f"cached CUPMem chunk omitted selection: {chunk_path}"
                    )
                chunk_selections.append(
                    _selection_from_record(raw_selection)
                )
                chunk_audit.append(
                    {
                        "chunk_index": chunk_index,
                        "candidate_count": len(chunk),
                        "prompt_chars": int(
                            chunk_cached.get("prompt_chars") or len(prompt)
                        ),
                    }
                )
            selection = (
                merge_cupmem_chunk_selections(
                    items,
                    chunks,
                    chunk_selections,
                )
                if chunks
                else StaleArmSelection(CUPMEM_ARM, (), ())
            )
            cached = {
                "schema_version": (
                    "stale_type2_cupmem_governance_cache_v3"
                ),
                "input_sha256": input_sha,
                "uid": record.uid,
                "mode": "bounded_cross_phase_complete_coverage_v1",
                "query_visible": False,
                "oracle_fields_visible": False,
                "chunk_count": len(chunks),
                "max_prompt_chars": max_prompt_chars,
                "max_text_chars": max_text_chars,
                "selection": selection.record(),
                "chunks": chunk_audit,
            }
            _write_json_once(path, cached)
        raw_selection = cached.get("selection")
        if not isinstance(raw_selection, Mapping):
            raise ValueError(f"cached governance omitted selection: {path}")
        return _selection_from_record(raw_selection)

    def _model_selection(
        self,
        *,
        record: StaleType2Record,
        episode: StaleType2MasReturnEpisode,
        items: Sequence[StaleMemoryItem],
        arm: str,
    ) -> StaleArmSelection:
        if arm == CUPMEM_ARM:
            return self._bounded_cupmem_selection(
                record=record,
                episode=episode,
                items=items,
            )
        if arm == TRACE_ARM:
            prompt = trace_prompt(
                items,
                departure_after_session=episode.departure_after_session,
            )
            parser = lambda text: parse_trace_selection(text, items)
            system = (
                "You are the TRACE provenance and lifecycle compiler. "
                "Use only actor-visible memories and return strict JSON."
            )
            statuses = ["ADMIT", "SUPERSEDED", "QUARANTINE"]
            link_key = "invalidated_by_memory_id"
            forced_schema = {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "memory_id": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": statuses,
                                },
                                link_key: {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "null"},
                                    ]
                                },
                            },
                            "required": ["memory_id", "status", link_key],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["decisions"],
                "additionalProperties": False,
            }
        elif arm == KMU_OP_ARM:
            prompt = kmu_prompt(
                items,
                returning_agent_id=episode.returning_agent_id,
            )
            parser = lambda text: parse_indexed_kmu_selection(
                text,
                items,
                returning_agent_id=episode.returning_agent_id,
            )
            system = (
                "You are the KMU operation-based memory updater. "
                "Return strict JSON only."
            )
            forced_schema = {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_index": {"type": "integer"},
                                "operation": {
                                    "type": "string",
                                    "enum": [
                                        "PASS",
                                        "REPLACE",
                                        "APPEND",
                                        "DELETE",
                                    ],
                                },
                                "target_candidate_index": {
                                    "anyOf": [
                                        {"type": "integer"},
                                        {"type": "null"},
                                    ]
                                },
                            },
                            "required": [
                                "candidate_index",
                                "operation",
                                "target_candidate_index",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["decisions"],
                "additionalProperties": False,
            }
        else:
            raise ValueError(arm)
        if arm == KMU_OP_ARM:
            return self._bounded_kmu_selection(
                record=record,
                episode=episode,
                items=items,
                system=system,
                forced_schema=forced_schema,
            )
        input_sha = canonical_sha256(
            {"stage": f"{arm}_governance_v2_forced_tool", "model": self.actor.generation_model, "prompt": prompt}
        )
        path = (
            self.output_dir
            / "cache"
            / record.uid
            / "governance"
            / f"{arm}_v2_forced_tool.json"
        )
        cached = _read_stage(path, input_sha)
        if cached is None:
            if not items:
                selection = StaleArmSelection(arm, (), ())
                receipts: list[dict[str, object]] = []
            else:
                selection, receipts = _call_and_parse(
                    self.actor,
                    prompt=prompt,
                    system=system,
                    seed=self._stage_seed(record.uid, arm, "governance"),
                    max_tokens=int(self.execution["governance_max_tokens"]),
                    retries=int(self.execution["format_retries"]),
                    parse=parser,
                    forced_tool=(
                        f"emit_{arm}_decisions",
                        "Return one complete lifecycle decision for every supplied memory candidate.",
                        forced_schema,
                    ),
                    structured_output_mode=(
                        self.actor_structured_output_mode
                    ),
                )
            cached = {
                "schema_version": "stale_type2_governance_cache_v1",
                "input_sha256": input_sha,
                "uid": record.uid,
                "query_visible": False,
                "oracle_fields_visible": False,
                "selection": selection.record(),
                "calls": receipts,
            }
            _write_json_once(path, cached)
        raw_selection = cached.get("selection")
        if not isinstance(raw_selection, Mapping):
            raise ValueError(f"cached governance omitted selection: {path}")
        return _selection_from_record(raw_selection)

    def _bounded_kmu_selection(
        self,
        *,
        record: StaleType2Record,
        episode: StaleType2MasReturnEpisode,
        items: Sequence[StaleMemoryItem],
        system: str,
        forced_schema: Mapping[str, object],
    ) -> StaleArmSelection:
        del forced_schema  # Every bounded chunk receives exact local enums below.
        max_prompt_chars = int(
            self.execution.get("governance_max_prompt_chars", 24000)
        )
        max_tokens = int(
            self.execution.get(
                "kmu_governance_max_tokens",
                self.execution["governance_max_tokens"],
            )
        )
        max_absence_items_per_chunk = int(
            self.execution.get("kmu_max_absence_items_per_chunk", 4)
        )
        chunks = chunk_kmu_candidates(
            items,
            returning_agent_id=episode.returning_agent_id,
            max_prompt_chars=max_prompt_chars,
            max_absence_items_per_chunk=max_absence_items_per_chunk,
        )
        input_sha = canonical_sha256(
            {
                "stage": "kmu_op_governance_v7_adaptive_nonself_fallback",
                "model": self.actor.generation_model,
                "structured_output_mode": self.actor_structured_output_mode,
                "items_sha256": memory_items_sha256(items),
                "max_prompt_chars": max_prompt_chars,
                "max_tokens": max_tokens,
                "max_absence_items_per_chunk": (
                    max_absence_items_per_chunk
                ),
                "chunks": [
                    [item.memory_id for item in chunk] for chunk in chunks
                ],
            }
        )
        path = (
            self.output_dir
            / "cache"
            / record.uid
            / "governance"
            / "kmu_op_v7_adaptive_nonself_fallback.json"
        )
        cached = _read_stage(path, input_sha)
        if cached is None:
            receipts: list[dict[str, object]] = []
            if not items:
                selection = StaleArmSelection(KMU_OP_ARM, (), ())
            elif not chunks:
                item_ids = tuple(item.memory_id for item in items)
                selection = StaleArmSelection(
                    KMU_OP_ARM,
                    item_ids,
                    item_ids,
                )
            else:
                def schema_for_chunk(
                    chunk: Sequence[StaleMemoryItem],
                    *,
                    exclude_incoming_targets: bool,
                ) -> Mapping[str, object]:
                    candidate_indices = tuple(range(len(chunk)))
                    absence_indices = tuple(
                        index
                        for index, item in enumerate(chunk)
                        if item.agent_id != episode.returning_agent_id
                    )
                    legal_targets = (
                        tuple(
                            index
                            for index in candidate_indices
                            if index not in absence_indices
                        )
                        if exclude_incoming_targets
                        else candidate_indices
                    )
                    return {
                        "type": "object",
                        "properties": {
                            "decisions": {
                                "type": "array",
                                "minItems": len(absence_indices),
                                "maxItems": len(absence_indices),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "candidate_index": {
                                            "type": "integer",
                                            "enum": absence_indices,
                                        },
                                        "operation": {
                                            "type": "string",
                                            "enum": [
                                                "PASS",
                                                "REPLACE",
                                                "APPEND",
                                                "DELETE",
                                            ],
                                        },
                                        "target_candidate_index": {
                                            "anyOf": [
                                                {
                                                    "type": "integer",
                                                    "enum": legal_targets,
                                                },
                                                {"type": "null"},
                                            ]
                                        },
                                    },
                                    "required": [
                                        "candidate_index",
                                        "operation",
                                        "target_candidate_index",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["decisions"],
                        "additionalProperties": False,
                    }

                def call_chunk(
                    chunk: Sequence[StaleMemoryItem],
                    *,
                    chunk_index: int,
                    cache_suffix: str,
                    exclude_incoming_targets: bool,
                ) -> tuple[StaleArmSelection, list[dict[str, object]]]:
                    prompt = kmu_prompt(
                        chunk,
                        returning_agent_id=episode.returning_agent_id,
                    )
                    chunk_schema = schema_for_chunk(
                        chunk,
                        exclude_incoming_targets=exclude_incoming_targets,
                    )
                    chunk_input_sha = canonical_sha256(
                        {
                            "stage": "kmu_op_governance_chunk_v7_adaptive",
                            "model": self.actor.generation_model,
                            "max_tokens": max_tokens,
                            "structured_output_mode": (
                                self.actor_structured_output_mode
                            ),
                            "exclude_incoming_targets": (
                                exclude_incoming_targets
                            ),
                            "prompt": prompt,
                        }
                    )
                    chunk_path = (
                        self.output_dir
                        / "cache"
                        / record.uid
                        / "governance"
                        / f"kmu_op_v7_chunk_{chunk_index:03d}_{cache_suffix}.json"
                    )
                    chunk_cached = _read_stage(chunk_path, chunk_input_sha)
                    if chunk_cached is None:
                        chunk_selection, chunk_receipts = _call_and_parse(
                            self.actor,
                            prompt=prompt,
                            system=system,
                            seed=self._stage_seed(
                                record.uid,
                                KMU_OP_ARM,
                                f"governance_chunk_{chunk_index:03d}_{cache_suffix}",
                            ),
                            max_tokens=max_tokens,
                            retries=int(self.execution["format_retries"]),
                            parse=lambda text, chunk=chunk: parse_indexed_kmu_selection(
                                text,
                                chunk,
                                returning_agent_id=episode.returning_agent_id,
                            ),
                            forced_tool=(
                                "emit_kmu_op_decisions",
                                "Return one complete lifecycle decision for every supplied absence candidate.",
                                chunk_schema,
                            ),
                            structured_output_mode=(
                                self.actor_structured_output_mode
                            ),
                        )
                        chunk_cached = {
                            "schema_version": (
                                "stale_type2_kmu_governance_chunk_cache_v7"
                            ),
                            "input_sha256": chunk_input_sha,
                            "uid": record.uid,
                            "chunk_index": chunk_index,
                            "cache_suffix": cache_suffix,
                            "exclude_incoming_targets": (
                                exclude_incoming_targets
                            ),
                            "candidate_item_ids": [
                                item.memory_id for item in chunk
                            ],
                            "query_visible": False,
                            "oracle_fields_visible": False,
                            "selection": chunk_selection.record(),
                            "calls": chunk_receipts,
                        }
                        _write_json_once(chunk_path, chunk_cached)
                    raw_selection = chunk_cached.get("selection")
                    if not isinstance(raw_selection, Mapping):
                        raise ValueError(
                            f"cached KMU chunk omitted selection: {chunk_path}"
                        )
                    raw_receipts = chunk_cached.get("calls")
                    return _selection_from_record(raw_selection), (
                        list(raw_receipts)
                        if isinstance(raw_receipts, Sequence)
                        else []
                    )

                chunk_selections: list[StaleArmSelection] = []
                for chunk_index, chunk in enumerate(chunks):
                    prompt = kmu_prompt(
                        chunk,
                        returning_agent_id=episode.returning_agent_id,
                    )
                    legacy_input_sha = canonical_sha256(
                        {
                            "stage": "kmu_op_governance_chunk_v4_local_index",
                            "model": self.actor.generation_model,
                            "max_tokens": max_tokens,
                            "structured_output_mode": (
                                self.actor_structured_output_mode
                            ),
                            "prompt": prompt,
                        }
                    )
                    legacy_path = (
                        self.output_dir
                        / "cache"
                        / record.uid
                        / "governance"
                        / f"kmu_op_v6_chunk_{chunk_index:03d}.json"
                    )
                    legacy_cached = _read_stage(legacy_path, legacy_input_sha)
                    fallback_reason: str | None = None
                    if legacy_cached is not None:
                        raw_legacy_selection = legacy_cached.get("selection")
                        if not isinstance(raw_legacy_selection, Mapping):
                            raise ValueError(
                                f"cached KMU chunk omitted selection: {legacy_path}"
                            )
                        chunk_selection = _selection_from_record(
                            raw_legacy_selection
                        )
                        raw_legacy_receipts = legacy_cached.get("calls")
                        chunk_receipts = (
                            list(raw_legacy_receipts)
                            if isinstance(raw_legacy_receipts, Sequence)
                            else []
                        )
                    else:
                        try:
                            chunk_selection, chunk_receipts = call_chunk(
                                chunk,
                                chunk_index=chunk_index,
                                cache_suffix="batch",
                                exclude_incoming_targets=False,
                            )
                        except ValueError as error:
                            fallback_reason = f"{type(error).__name__}: {error}"
                            predeparture = tuple(
                                item
                                for item in chunk
                                if item.agent_id == episode.returning_agent_id
                            )
                            incoming = tuple(
                                item
                                for item in chunk
                                if item.agent_id != episode.returning_agent_id
                            )
                            subchunks = tuple(
                                (*predeparture, item) for item in incoming
                            )
                            sub_selections: list[StaleArmSelection] = []
                            chunk_receipts = []
                            for sub_index, subchunk in enumerate(subchunks):
                                kmu_chunk_index_contract(
                                    subchunk,
                                    returning_agent_id=(
                                        episode.returning_agent_id
                                    ),
                                )
                                sub_selection, sub_receipts = call_chunk(
                                    subchunk,
                                    chunk_index=chunk_index,
                                    cache_suffix=(
                                        f"fallback_{sub_index:02d}"
                                    ),
                                    exclude_incoming_targets=True,
                                )
                                sub_selections.append(sub_selection)
                                chunk_receipts.extend(sub_receipts)
                            chunk_selection = merge_kmu_chunk_selections(
                                chunk,
                                subchunks,
                                sub_selections,
                                returning_agent_id=(
                                    episode.returning_agent_id
                                ),
                            )
                    chunk_selections.append(chunk_selection)
                    receipts.extend(
                        {
                            **receipt,
                            "chunk_index": chunk_index,
                            "chunk_candidate_count": len(chunk),
                            "adaptive_single_candidate_fallback": (
                                fallback_reason is not None
                            ),
                            "fallback_reason": fallback_reason,
                        }
                        for receipt in chunk_receipts
                    )
                selection = merge_kmu_chunk_selections(
                    items,
                    chunks,
                    chunk_selections,
                    returning_agent_id=episode.returning_agent_id,
                )
            cached = {
                "schema_version": "stale_type2_governance_cache_v7",
                "input_sha256": input_sha,
                "uid": record.uid,
                "query_visible": False,
                "oracle_fields_visible": False,
                "max_prompt_chars": max_prompt_chars,
                "chunk_count": len(chunks),
                "selection": selection.record(),
                "calls": receipts,
            }
            _write_json_once(path, cached)
        raw_selection = cached.get("selection")
        if not isinstance(raw_selection, Mapping):
            raise ValueError(f"cached KMU governance omitted selection: {path}")
        return _selection_from_record(raw_selection)

    def _retrievals(
        self,
        *,
        record: StaleType2Record,
        items: Sequence[StaleMemoryItem],
        selections: Mapping[str, StaleArmSelection],
    ) -> dict[str, tuple[StaleMemoryItem, ...]]:
        memory_arms = self.arms
        input_sha = canonical_sha256(
            {
                "stage": "question_time_embedding_retrieval_v1",
                "embedding_model": self.embedding.model_id,
                "items_sha256": memory_items_sha256(items),
                "queries": list(record.queries),
                "selections": {
                    arm: list(selections[arm].selected_item_ids)
                    for arm in memory_arms
                },
                "top_k_per_query": int(self.execution["retrieval_top_k_per_query"]),
                "max_union_items": int(self.execution["retrieval_max_union_items"]),
            }
        )
        path = self.output_dir / "cache" / record.uid / "retrieval.json"
        cached = _read_stage(path, input_sha)
        item_by_id = {item.memory_id: item for item in items}
        if cached is None:
            embedding_inputs = [
                f"{item.state_key}\n{item.statement}\n{item.evidence_quote}"
                for item in items
            ] + list(record.queries)
            vectors = self.embedding.embed(embedding_inputs) if embedding_inputs else []
            item_vectors = {
                item.memory_id: vectors[index] for index, item in enumerate(items)
            }
            query_vectors = vectors[len(items) :]
            retrieval_ids: dict[str, list[str]] = {}
            for arm in memory_arms:
                selected = (
                    tuple(items)
                    if arm in TRACE_INVALIDATOR_EXPANSION_ARMS
                    and self.execution.get("trace_retrieval_mode")
                    == "expand_invalidators_v1"
                    else tuple(
                        item_by_id[item_id]
                        for item_id in selections[arm].selected_item_ids
                    )
                )
                if selected:
                    retrieved = retrieve_for_queries(
                        selected,
                        record.queries,
                        item_embeddings=item_vectors,
                        query_embeddings=query_vectors,
                        top_k_per_query=int(
                            self.execution["retrieval_top_k_per_query"]
                        ),
                        max_union_items=int(
                            self.execution["retrieval_max_union_items"]
                        ),
                    )
                else:
                    retrieved = ()
                if (
                    arm in TRACE_INVALIDATOR_EXPANSION_ARMS
                    and self.execution.get("trace_retrieval_mode")
                    == "expand_invalidators_v1"
                ):
                    retrieved = expand_trace_retrieval(
                        items,
                        retrieved,
                        selections[arm],
                    )
                retrieval_ids[arm] = [item.memory_id for item in retrieved]
            cached = {
                "schema_version": "stale_type2_retrieval_cache_v1",
                "input_sha256": input_sha,
                "uid": record.uid,
                "query_visible": True,
                "oracle_fields_visible": False,
                "retrieved_item_ids": retrieval_ids,
            }
            _write_json_once(path, cached)
        raw = cached.get("retrieved_item_ids")
        if not isinstance(raw, Mapping):
            raise ValueError("cached retrieval omitted retrieved_item_ids")
        return {
            arm: tuple(item_by_id[str(item_id)] for item_id in raw[arm])
            for arm in memory_arms
        }

    def _answers(
        self,
        *,
        record: StaleType2Record,
        arm: str,
        items: Sequence[StaleMemoryItem] = (),
        prompt_override: str | None = None,
        lifecycle_resolutions: Sequence[Mapping[str, object]] = (),
    ) -> tuple[dict[str, str], Mapping[str, object]]:
        prompt = (
            prompt_override
            if prompt_override is not None
            else answer_prompt(
                items,
                record.queries,
                lifecycle_resolutions=lifecycle_resolutions,
            )
        )
        answer_seed = _paired_arm_seed(record.uid, "answer")
        input_sha = canonical_sha256(
            {
                "stage": "official_three_query_answer_v3_paired_seed",
                "model": self.actor.generation_model,
                "prompt": prompt,
                "seed": answer_seed,
            }
        )
        path = self.output_dir / "cache" / record.uid / "answers" / f"{arm}.json"
        cached = _read_stage(path, input_sha)
        if cached is None:
            answers, receipts = _call_and_parse(
                self.actor,
                prompt=prompt,
                system=(
                    "You are agent1, the Returning Agent. Answer the three "
                    "user questions directly using only the supplied "
                    "actor-visible context. Return strict JSON."
                ),
                seed=answer_seed,
                max_tokens=int(self.execution["answer_max_tokens"]),
                retries=int(self.execution["format_retries"]),
                parse=parse_answers,
                forced_tool=(
                    "submit_stale_answers",
                    "Submit the three answers to the official STALE queries.",
                    _answer_tool_schema(),
                ),
                structured_output_mode=self.actor_structured_output_mode,
            )
            cached = {
                "schema_version": "stale_type2_answer_cache_v2_paired_seed",
                "input_sha256": input_sha,
                "uid": record.uid,
                "arm": arm,
                "paired_seed": answer_seed,
                "retrieved_item_ids": [item.memory_id for item in items],
                "prompt_override": prompt_override is not None,
                "lifecycle_resolutions": [
                    dict(resolution) for resolution in lifecycle_resolutions
                ],
                "answers": answers,
                "calls": receipts,
            }
            _write_json_once(path, cached)
        raw_answers = cached.get("answers")
        if not isinstance(raw_answers, Mapping):
            raise ValueError("cached answer omitted answers")
        return parse_answers(json.dumps(raw_answers)), cached

    def _judge(
        self,
        *,
        record: StaleType2Record,
        arm: str,
        answers: Mapping[str, str],
    ) -> tuple[dict[str, object], Mapping[str, object]]:
        prompt = official_judge_user_prompt(record, answers)
        judge_seed = _paired_arm_seed(record.uid, "judge")
        input_sha = canonical_sha256(
            {
                "stage": "official_stale_all_in_one_judge_v2_paired_seed",
                "model": self.judge.generation_model,
                "system": OFFICIAL_JUDGE_SYSTEM_PROMPT,
                "prompt": prompt,
                "seed": judge_seed,
            }
        )
        path = self.output_dir / "cache" / record.uid / "judges" / f"{arm}.json"
        cached = _read_stage(path, input_sha)
        if cached is None:
            judgment, receipts = _call_and_parse(
                self.judge,
                prompt=prompt,
                system=OFFICIAL_JUDGE_SYSTEM_PROMPT,
                seed=judge_seed,
                max_tokens=int(self.execution["judge_max_tokens"]),
                retries=int(self.execution["judge_format_retries"]),
                parse=parse_judge,
            )
            cached = {
                "schema_version": "stale_type2_official_judge_cache_v2_paired_seed",
                "input_sha256": input_sha,
                "uid": record.uid,
                "arm": arm,
                "paired_seed": judge_seed,
                "evaluator_only": True,
                "judgment": judgment,
                "calls": receipts,
            }
            _write_json_once(path, cached)
        raw_judgment = cached.get("judgment")
        if not isinstance(raw_judgment, Mapping):
            raise ValueError("cached Judge stage omitted judgment")
        return parse_judge(json.dumps(raw_judgment)), cached

    def _diagnostic_oracle_results(
        self,
        *,
        record: StaleType2Record,
        episode: StaleType2MasReturnEpisode,
        items: Sequence[StaleMemoryItem],
        retrievals: Mapping[str, Sequence[StaleMemoryItem]],
    ) -> Mapping[str, object]:
        """Run explicitly separated evaluator-only diagnostic upper bounds."""

        if not self.diagnostic_oracles:
            return {}
        item_by_id = {item.memory_id: item for item in items}
        old_items = tuple(
            item for item in items if item.session_index == record.old_session_index
        )
        new_items = tuple(
            item for item in items if item.session_index == record.new_session_index
        )
        if not old_items or not new_items:
            raise RuntimeError("diagnostic oracle requires both relevant sessions")
        old_observation = next(
            (
                item
                for item in old_items
                if item.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX)
            ),
            None,
        )
        new_observation = next(
            (
                item
                for item in new_items
                if item.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX)
            ),
            None,
        )
        if old_observation is None or new_observation is None:
            raise RuntimeError("diagnostic oracle requires session observations")
        ideal_resolution = {
            "status": "SUPERSEDED",
            "historical_target": old_observation.compact_record(quote_chars=700),
            "current_invalidator": new_observation.compact_record(quote_chars=700),
            "relation_source": "diagnostic_dependency_oracle",
            "target_receipt_sha256": old_observation.source_session_sha256,
            "invalidator_receipt_sha256": new_observation.source_session_sha256,
        }

        actor_sessions = [
            session
            for view in actor_views(record, episode).values()
            for session in view["sessions"]
            if isinstance(session, Mapping)
        ]
        if {int(session["session_index"]) for session in actor_sessions} != set(
            range(STALE_SESSION_COUNT)
        ):
            raise RuntimeError("full-history diagnostic lost actor-visible sessions")

        base_arm = (
            TRACE_ARM
            if TRACE_ARM in retrievals
            else STATIC_ARM
            if STATIC_ARM in retrievals
            else next(iter(retrievals), None)
        )
        base_retrieval = (
            tuple(retrievals[base_arm]) if base_arm is not None else tuple(items)
        )
        dependency_items = tuple(
            sorted(
                {
                    item.memory_id: item
                    for item in (
                        *(
                            item
                            for item in base_retrieval
                            if item.session_index != record.old_session_index
                        ),
                        *new_items,
                    )
                }.values(),
                key=lambda item: (item.session_index, item.memory_id),
            )
        )
        retrieval_oracle_items = tuple(
            sorted(
                {item.memory_id: item for item in (*old_items, *new_items)}.values(),
                key=lambda item: (item.session_index, item.memory_id),
            )
        )

        results: dict[str, object] = {}
        for oracle in self.diagnostic_oracles:
            if oracle == "full_history_oracle":
                view_items: tuple[StaleMemoryItem, ...] = ()
                prompt_override = full_history_answer_prompt(
                    actor_sessions,
                    record.queries,
                )
                visible_session_indices = list(range(STALE_SESSION_COUNT))
                oracle_fields_used = ["none; actor-visible raw sessions only"]
            elif oracle == "retrieval_oracle":
                view_items = retrieval_oracle_items
                prompt_override = None
                visible_session_indices = sorted(
                    {item.session_index for item in view_items}
                )
                oracle_fields_used = [
                    "old_session_index",
                    "new_session_index",
                ]
            elif oracle == "dependency_oracle":
                view_items = dependency_items
                prompt_override = None
                visible_session_indices = sorted(
                    {item.session_index for item in view_items}
                )
                oracle_fields_used = [
                    "old_session_index",
                    "new_session_index",
                    "ideal_old_to_new_dependency",
                ]
            else:  # guarded in __init__
                raise AssertionError(f"unsupported diagnostic oracle: {oracle}")

            answers, answer_receipt = self._answers(
                record=record,
                arm=oracle,
                items=view_items,
                prompt_override=prompt_override,
                lifecycle_resolutions=(
                    (ideal_resolution,)
                    if oracle == "dependency_oracle"
                    else ()
                ),
            )
            judgment, judge_receipt = self._judge(
                record=record,
                arm=oracle,
                answers=answers,
            )
            results[oracle] = {
                "diagnostic_only": True,
                "excluded_from_main_summary": True,
                "final_answer_agent_id": episode.returning_agent_id,
                "base_retrieval_arm": (
                    base_arm if oracle == "dependency_oracle" else None
                ),
                "visible_session_indices": visible_session_indices,
                "retrieved_item_ids": [item.memory_id for item in view_items],
                "lifecycle_resolutions": (
                    [ideal_resolution]
                    if oracle == "dependency_oracle"
                    else []
                ),
                "actor_oracle_fields_visible": False,
                "evaluator_oracle_fields_used": oracle_fields_used,
                "official_answer_record": record.official_answer_record(answers),
                "official_judge": judgment,
                "metrics": _metrics_from_judgment(judgment),
                "actor_call_count": len(answer_receipt.get("calls") or ()),
                "judge_call_count": len(judge_receipt.get("calls") or ()),
            }
        # The ids above must all resolve to the same frozen checkpoint.
        if any(
            item_id not in item_by_id
            for result in results.values()
            for item_id in result["retrieved_item_ids"]
        ):
            raise RuntimeError("diagnostic oracle emitted an unknown memory_id")
        return results

    def run(
        self,
        record: StaleType2Record,
        episode: StaleType2MasReturnEpisode,
    ) -> Mapping[str, object]:
        final_path = self.output_dir / "results" / f"{record.uid}.json"
        if final_path.is_file():
            value = json.loads(final_path.read_text(encoding="utf-8"))
            if value.get("config_sha256") != self.config_sha256:
                raise ValueError(f"existing result config mismatch: {record.uid}")
            return value
        audit = leakage_audit(record, episode)
        if not audit["passed"] or not audit["assigned_exactly_once"]:
            raise RuntimeError(f"actor view audit failed: {record.uid}")
        items = self._extract(record, episode)
        checkpoint_sha256 = memory_items_sha256(items)
        selections = deterministic_arm_selections(
            items, returning_agent_id=episode.returning_agent_id
        )
        trace_dependency_audit: Mapping[str, object] | None = None
        trace_ablation_audit: Mapping[str, object] | None = None
        if CUPMEM_ARM in self.arms:
            selections[CUPMEM_ARM] = self._model_selection(
                record=record, episode=episode, items=items, arm=CUPMEM_ARM
            )
        if KMU_OP_ARM in self.arms:
            selections[KMU_OP_ARM] = self._model_selection(
                record=record, episode=episode, items=items, arm=KMU_OP_ARM
            )
        if TRACE_ARM in self.arms:
            trace_dependency_mode = self.execution.get("trace_dependency_mode")
            if trace_dependency_mode == "session_observation_chunked_scan_v1":
                selections[TRACE_ARM], trace_dependency_audit = (
                    self._trace_dependency_selection(
                        record=record,
                        episode=episode,
                        items=items,
                    )
                )
            elif (
                trace_dependency_mode
                == "session_observation_two_stage_scan_v2"
            ):
                selections[TRACE_ARM], trace_dependency_audit = (
                    self._trace_dependency_selection_two_stage(
                        record=record,
                        episode=episode,
                        items=items,
                    )
                )
            else:
                selections[TRACE_ARM] = self._model_selection(
                    record=record,
                    episode=episode,
                    items=items,
                    arm=TRACE_ARM,
                )
        requested_trace_ablations = set(self.arms).intersection(
            STALE_TRACE_ABLATION_ARMS
        )
        if requested_trace_ablations:
            if TRACE_ARM not in self.arms or trace_dependency_audit is None:
                raise ValueError(
                    "STALE TRACE ablations require the paired TRACE arm"
                )
            proposed_edges = trace_dependency_audit.get("proposed_edges")
            if not isinstance(proposed_edges, Sequence) or isinstance(
                proposed_edges, (str, bytes)
            ):
                raise ValueError("TRACE audit omitted proposed dependency edges")
            ablation_selections, trace_ablation_audit = (
                compile_stale_trace_ablations(
                    items,
                    returning_agent_id=episode.returning_agent_id,
                    departure_after_session=episode.departure_after_session,
                    base_selections=selections,
                    trace_selection=selections[TRACE_ARM],
                    proposed_edges=proposed_edges,
                )
            )
            selections.update(ablation_selections)
        retrievals = self._retrievals(
            record=record, items=items, selections=selections
        )
        method_results: dict[str, object] = {}
        for arm in self.arms:
            lifecycle_resolutions = (
                _trace_lifecycle_resolutions(
                    items,
                    selections[arm],
                    retrievals[arm],
                    relation_source=(
                        "trace_unverified_candidate_edge_ablation"
                        if arm == TRACE_WITHOUT_PROVENANCE_ARM
                        else "trace_verified_dependency_edge"
                    ),
                )
                if arm in TRACE_LIFECYCLE_RESOLUTION_ARMS
                and trace_dependency_audit is not None
                and trace_dependency_audit.get("mode")
                == "complete_small_chunk_two_stage"
                else ()
            )
            answers, answer_receipt = self._answers(
                record=record,
                arm=arm,
                items=retrievals[arm],
                lifecycle_resolutions=lifecycle_resolutions,
            )
            judgment, judge_receipt = self._judge(
                record=record, arm=arm, answers=answers
            )
            method_results[arm] = {
                "final_answer_agent_id": episode.returning_agent_id,
                "selection": selections[arm].record(),
                "retrieved_item_ids": [item.memory_id for item in retrievals[arm]],
                "lifecycle_resolutions": [
                    dict(resolution) for resolution in lifecycle_resolutions
                ],
                "official_answer_record": record.official_answer_record(answers),
                "official_judge": judgment,
                "metrics": _metrics_from_judgment(judgment),
                "actor_call_count": len(answer_receipt.get("calls") or ()),
                "judge_call_count": len(judge_receipt.get("calls") or ()),
            }
        diagnostic_results = self._diagnostic_oracle_results(
            record=record,
            episode=episode,
            items=items,
            retrievals=retrievals,
        )
        result = {
            "schema_version": SCHEMA,
            "created_at": _utc_now(),
            "config_sha256": self.config_sha256,
            "retry_seed_offset": self.retry_seed_offset,
            "allow_anchor_fallback": self.allow_anchor_fallback,
            "uid": record.uid,
            "source_record_sha256": record.source_record_sha256,
            "sidecar_episode_sha256": episode.record()["episode_sha256"],
            "mas_execution": unified_mas_episode_record(
                benchmark="STALE-TypeII-Official-MAS-Return",
                episode_id=episode.uid,
                assignments=episode.agent_session_indices,
                ordered_session_ids=tuple(range(STALE_SESSION_COUNT)),
                departure_after_session=episode.departure_after_session,
                return_after_session=episode.return_after_session,
                checkpoint_sha256=checkpoint_sha256,
            ),
            "official_record_modified": False,
            "actor_oracle_fields_visible": False,
            "extracted_memory_count": len(items),
            "extracted_memory_sha256": checkpoint_sha256,
            "memory_write_audit": {
                "policy": "exactly_one_lossless_provenance_anchor_per_session_v2",
                "official_session_count": STALE_SESSION_COUNT,
                "covered_session_count": len(
                    {
                        item.session_index
                        for item in items
                        if item.state_key.startswith(
                            SESSION_OBSERVATION_KEY_PREFIX
                        )
                    }
                ),
                "official_annotation_fields_read": False,
                "query_visible_during_write_or_governance": False,
            },
            "trace_dependency_audit": trace_dependency_audit,
            "trace_ablation_audit": trace_ablation_audit,
            "methods": method_results,
            "diagnostic_oracles": diagnostic_results,
            "metric_contract": {
                "schema_version": "stale_return_unified_metrics_v1",
                "via": "official_dim3_pass",
                "iir": "official_dim1_pass_and_official_dim2_pass",
                "iir_component_mean": "mean(official_dim1_pass,official_dim2_pass)",
                "lifecycle_success": "via_and_iir",
            },
        }
        _write_json_once(final_path, result)
        return result


def _summary(
    output_dir: Path,
    config_sha256: str,
    expected: int,
    arms: Sequence[str],
) -> Mapping[str, object]:
    rows = []
    for path in sorted((output_dir / "results").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("config_sha256") == config_sha256:
            rows.append(value)
    methods: dict[str, object] = {}
    for arm in arms:
        metrics = [row["methods"][arm]["metrics"] for row in rows]
        n = len(metrics)
        methods[arm] = {
            "records": n,
            "dim1_accuracy": sum(bool(item["dim1_pass"]) for item in metrics) / n if n else 0.0,
            "dim2_accuracy": sum(bool(item["dim2_pass"]) for item in metrics) / n if n else 0.0,
            "dim3_accuracy": sum(bool(item["dim3_pass"]) for item in metrics) / n if n else 0.0,
            "overall_accuracy": sum(float(item["overall_accuracy"]) for item in metrics) / n if n else 0.0,
            "all_dimensions_accuracy": sum(bool(item["all_dimensions_pass"]) for item in metrics) / n if n else 0.0,
            "via_accuracy": sum(bool(item["via"]) for item in metrics) / n if n else 0.0,
            "iir_accuracy": sum(bool(item["iir"]) for item in metrics) / n if n else 0.0,
            "iir_component_mean": sum(float(item["iir_component_mean"]) for item in metrics) / n if n else 0.0,
            "lifecycle_success_accuracy": sum(bool(item["lifecycle_success"]) for item in metrics) / n if n else 0.0,
        }
    diagnostic_names = sorted(
        {
            name
            for row in rows
            for name in (row.get("diagnostic_oracles") or {})
        }
    )
    diagnostics: dict[str, object] = {}
    for name in diagnostic_names:
        metrics = [
            row["diagnostic_oracles"][name]["metrics"]
            for row in rows
            if name in (row.get("diagnostic_oracles") or {})
        ]
        n = len(metrics)
        diagnostics[name] = {
            "diagnostic_only": True,
            "excluded_from_main_summary": True,
            "records": n,
            "dim1_accuracy": sum(bool(item["dim1_pass"]) for item in metrics) / n if n else 0.0,
            "dim2_accuracy": sum(bool(item["dim2_pass"]) for item in metrics) / n if n else 0.0,
            "dim3_accuracy": sum(bool(item["dim3_pass"]) for item in metrics) / n if n else 0.0,
            "overall_accuracy": sum(float(item["overall_accuracy"]) for item in metrics) / n if n else 0.0,
            "all_dimensions_accuracy": sum(bool(item["all_dimensions_pass"]) for item in metrics) / n if n else 0.0,
            "via_accuracy": sum(bool(item["via"]) for item in metrics) / n if n else 0.0,
            "iir_accuracy": sum(bool(item["iir"]) for item in metrics) / n if n else 0.0,
            "lifecycle_success_accuracy": sum(bool(item["lifecycle_success"]) for item in metrics) / n if n else 0.0,
        }
    return {
        "schema_version": "stale_type2_summary_v6",
        "updated_at": _utc_now(),
        "config_sha256": config_sha256,
        "completed_records": len(rows),
        "expected_records": expected,
        "complete": len(rows) == expected,
        "metric_contract": {
            "schema_version": "stale_return_unified_metrics_v1",
            "via": "official_dim3_pass",
            "iir": "official_dim1_pass_and_official_dim2_pass",
            "iir_component_mean": "mean(official_dim1_pass,official_dim2_pass)",
            "lifecycle_success": "via_and_iir",
        },
        "methods": methods,
        "diagnostic_oracles": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--uid", action="append", default=[])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--retry-seed-offset", type=int, default=0)
    parser.add_argument("--allow-anchor-fallback", action="store_true")
    parser.add_argument("--actor-request-lock-path")
    parser.add_argument("--judge-request-lock-path")
    parser.add_argument("--actor-model", help="Override model.served_model_id from config")
    parser.add_argument("--actor-base-url", help="Override model.api_base from config")
    parser.add_argument("--actor-api-key-env", help="Override model.api_key_env (pass empty string to clear)")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = resolve_provider_config(json.loads(config_path.read_text(encoding="utf-8")))
    if args.actor_model or args.actor_base_url or args.actor_api_key_env is not None:
        _model = dict(config.get("model", {}))
        if args.actor_model:
            _model["served_model_id"] = args.actor_model
        if args.actor_base_url:
            _model["api_base"] = args.actor_base_url
        if args.actor_api_key_env is not None:
            _model["api_key_env"] = args.actor_api_key_env or None
        config = {**config, "model": _model}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    execution = config["execution"]
    arms = tuple(str(arm) for arm in execution["fork_arms"])
    mas = config.get("mas")
    if not isinstance(mas, Mapping):
        raise ValueError("STALE config requires a mas contract")
    validate_unified_mas_config(mas)
    if not arms:
        raise ValueError("config must select at least one STALE arm")
    if len(arms) != len(set(arms)):
        raise ValueError("config STALE arms must be unique")
    unknown_arms = set(arms) - set(STALE_RUNNER_ARMS)
    if unknown_arms:
        raise ValueError(
            "config contains unsupported STALE arms: "
            + ",".join(sorted(unknown_arms))
        )
    dataset_path = ROOT / config["dataset"]["official_path"]
    sidecar_path = ROOT / config["dataset"]["sidecar_path"]
    records = load_official_stale_type2(dataset_path)
    episodes = load_sidecar_manifest(sidecar_path)
    bound = list(
        _bucket_bound_records(
            list(bind_records_to_sidecar(records, episodes)),
            config,
        )
    )
    if args.uid:
        requested = set(args.uid)
        bound = [pair for pair in bound if pair[0].uid in requested]
        missing = requested - {pair[0].uid for pair in bound}
        if missing:
            raise ValueError("unknown requested UIDs: " + ",".join(sorted(missing)))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        bound = bound[: args.limit]
    workers = args.workers or int(execution["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    if args.retry_seed_offset < 0:
        raise ValueError("retry-seed-offset must be non-negative")

    actor = _provider(
        config["model"],
        execution,
        request_lock_path=args.actor_request_lock_path,
    )
    judge = _provider(
        config["judge"],
        execution,
        request_lock_path=args.judge_request_lock_path,
    )
    embedding = _embedding_provider(config["embedding"], execution)
    health = {
        "actor": _health(actor, actor.generation_model, "actor"),
        "judge": _health(judge, judge.generation_model, "judge"),
        "embedding": _health(embedding, embedding.model_id, "embedding"),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": "stale_type2_run_manifest_v6",
        "created_at": _utc_now(),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "official_dataset_path": str(dataset_path.resolve()),
        "official_sidecar_path": str(sidecar_path.resolve()),
        "selected_records": len(bound),
        "official_record_modified": False,
        "fork_arms": list(arms),
        "diagnostic_oracles": list(
            execution.get("diagnostic_oracles")
            if isinstance(execution.get("diagnostic_oracles"), Sequence)
            and not isinstance(execution.get("diagnostic_oracles"), (str, bytes))
            else DIAGNOSTIC_ORACLES
            if execution.get("diagnostic_oracles") is True
            else ()
        ),
        "metric_contract": {
            "schema_version": "stale_return_unified_metrics_v1",
            "via": "official_dim3_pass",
            "iir": "official_dim1_pass_and_official_dim2_pass",
            "iir_component_mean": "mean(official_dim1_pass,official_dim2_pass)",
            "lifecycle_success": "via_and_iir",
        },
        "actor": {
            "endpoint_origin": endpoint_origin(actor.generation_base_url),
            "model": actor.generation_model,
        },
        "judge": {
            "endpoint_origin": endpoint_origin(judge.generation_base_url),
            "model": judge.generation_model,
            "official_prompt_sha256": hashlib.sha256(
                OFFICIAL_JUDGE_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        },
        "embedding": {
            "endpoint_origin": endpoint_origin(embedding.base_url),
            "model": embedding.model_id,
        },
        "health": health,
        "source_contract": _source_contract(),
    }
    run_manifest["source_contract_sha256"] = canonical_sha256(
        run_manifest["source_contract"]
    )
    manifest_path = output_dir / "run_manifest.json"
    expected_records = len(bound)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config_sha:
            raise ValueError("output directory belongs to a different config")
        expected_records = int(existing.get("selected_records", len(bound)))
    else:
        _write_json_once(manifest_path, run_manifest)

    if args.retry_seed_offset:
        retry_manifest = {
            "schema_version": "stale_type2_retry_manifest_v1",
            "created_at": _utc_now(),
            "config_sha256": config_sha,
            "retry_seed_offset": args.retry_seed_offset,
            "allow_anchor_fallback": args.allow_anchor_fallback,
            "selected_uids": [record.uid for record, _ in bound],
            "source_contract": _source_contract(),
        }
        retry_manifest["source_contract_sha256"] = canonical_sha256(
            retry_manifest["source_contract"]
        )
        retry_path = (
            output_dir
            / "retry_manifests"
            / f"seed_offset_{args.retry_seed_offset}.json"
        )
        if not retry_path.exists():
            _write_json_once(retry_path, retry_manifest)

    runner = StaleEpisodeRunner(
        output_dir=output_dir,
        actor=actor,
        judge=judge,
        embedding=embedding,
        execution=execution,
        arms=arms,
        config_sha256=config_sha,
        retry_seed_offset=args.retry_seed_offset,
        allow_anchor_fallback=args.allow_anchor_fallback,
    )
    failures: list[dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(runner.run, record, episode): record.uid
            for record, episode in bound
        }
        for future in as_completed(futures):
            uid = futures[future]
            try:
                future.result()
                completed += 1
                with _PRINT_LOCK:
                    print(f"[{completed}/{len(bound)}] completed uid={uid}", flush=True)
            except Exception as error:  # keep an intention-to-treat failure receipt
                failure = {
                    "uid": uid,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                failure_root = output_dir / "failures"
                if args.retry_seed_offset:
                    failure_root = (
                        failure_root
                        / f"retry_seed_offset_{args.retry_seed_offset}"
                    )
                failure_path = failure_root / f"{uid}.json"
                if not failure_path.exists():
                    _write_json_once(failure_path, failure)
                with _PRINT_LOCK:
                    print(f"FAILED uid={uid}: {type(error).__name__}: {error}", flush=True)
    summary = _summary(output_dir, config_sha, expected_records, arms)
    _replace_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} STALE episodes failed")


if __name__ == "__main__":
    main()
