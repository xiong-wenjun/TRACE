#!/usr/bin/env python3
"""Run paired Memora Static/Reset/Restore/TRACE RETURN episodes."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace.eight_agent_pipeline import (  # noqa: E402
    EVALUATOR_ID,
    ROUTER_ID,
    EvaluatorAgent,
    WORKER_IDS,
)
from trace.mas_return_pipeline import (  # noqa: E402
    AgentMemoryArm,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MAX_CHARS,
    EpisodeLongTermMemory,
    unified_five_task_agent_architecture_record,
)
from trace.memora_return import (  # noqa: E402
    MemoraQuestion,
    MemoraReturnEpisode,
    MemoraRubricJudgment,
    build_memora_return_forks,
    build_returning_worker_prompt,
    evidence_session_ids,
    forgetting_value_sessions,
    load_memora_manifest,
    load_memora_timeline,
    memora_unified_mas_episode_record,
    normalize_yes_no,
    score_memora_fama,
)
from trace.memora_predicted import (  # noqa: E402
    PredictedDependencyPlan,
    PredictedExtractionBatch,
    PredictedRouterPlan,
    PredictedStateFact,
    PredictedTrace,
    active_facts_at_departure,
    actor_session_records,
    build_predicted_return_forks,
    dynamic_selector_budget,
    extraction_prompt,
    parse_extraction,
    retrieve_actor_sessions,
    safe_question_record,
    shard_actor_sessions,
)
from trace.memory_backends import (  # noqa: E402
    ASVM_BACKEND,
    SCOPEDMEM_BACKEND,
    add_memory_backend_arguments,
    build_memory_backend_factory,
)
from trace.private_episodic_memory import (  # noqa: E402
    PrivateEpisodicMemory,
    PrivateMemoryRetrieval,
)
from trace.providers import (  # noqa: E402
    Completion,
    OpenAICompatibleEmbeddingClient,
    OpenAICompatibleProvider,
    endpoint_origin,
)
from trace.return_baselines import (  # noqa: E402
    CHECKPOINT_REPLAY_ARM,
    CUPMEM_ARM,
    LIFECYCLE_BASELINE_ARMS,
    LifecycleReturnArm,
    compile_checkpoint_replay,
    compile_cupmem,
    cupmem_adjudication_prompt,
)
from trace.return_methods.lifecycle import (  # noqa: E402
    KMU_OP_ARM,
    TEMPORAL_LWW_ARM,
    compile_indexed_kmu_operations,
    compile_router_temporal_lww,
    indexed_kmu_operation_prompt,
    router_return_candidates,
)
from trace.return_methods.temporal import (  # noqa: E402
    MEMSTRATA_ARM,
    compile_router_memstrata,
)
from trace.return_methods.transactional import (  # noqa: E402
    MEMTX_ARM,
    compile_router_memtx,
)
from trace.return_governance import ReturnMemoryGovernor  # noqa: E402
from trace.trace_method import (  # noqa: E402
    TRACE_ARM,
    canonical_return_arm,
    internal_return_arm,
)
from trace.router_return_protocol import (  # noqa: E402
    RouterReturnArm,
    RouterReturnFork,
    canonical_sha256,
)
from trace.router_return_governance import (  # noqa: E402
    render_router_fork,
)
SCHEMA = "memora_mas_return_run_v2"
DEFAULT_DATA_ROOT = (
    ROOT / "data" / "Memora" / "data"
)
DEFAULT_ARMS = (
    RouterReturnArm.STATIC.value,
    RouterReturnArm.RESET.value,
    RouterReturnArm.RESTORE_OLD.value,
    TRACE_ARM,
)


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("TRACE_EMBEDDING_BASE_URL"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("TRACE_EMBEDDING_MODEL"),
    )
    parser.add_argument(
        "--embedding-api-key-env",
        default="TRACE_EMBEDDING_API_KEY",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
    )
    parser.add_argument(
        "--embedding-max-chars",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_CHARS,
    )


def _build_embedding_provider(
    args: argparse.Namespace,
) -> OpenAICompatibleEmbeddingClient | None:
    if args.embedding_dimensions < 1:
        raise ValueError("embedding-dimensions must be positive")
    if args.embedding_max_chars < 1:
        raise ValueError("embedding-max-chars must be positive")
    if not args.embedding_base_url and not args.embedding_model:
        return None
    if not args.embedding_base_url or not args.embedding_model:
        raise ValueError(
            "embedding-base-url and embedding-model must be configured together"
        )
    api_key = (
        os.environ.get(args.embedding_api_key_env)
        if args.embedding_api_key_env
        else None
    )
    return OpenAICompatibleEmbeddingClient(
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        api_key=api_key,
        timeout=args.timeout,
        retries=args.retries,
    )


def _completion_record(completion: Completion) -> dict[str, object]:
    return {
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "total_tokens": completion.total_tokens,
        "latency_seconds": completion.latency_seconds,
        "text_sha256": hashlib.sha256(
            completion.text.encode("utf-8")
        ).hexdigest(),
    }


def _source_contract() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "trace" / "memora_predicted.py",
        ROOT / "src" / "trace" / "memora_return.py",
        ROOT / "src" / "trace" / "private_episodic_memory.py",
        ROOT / "src" / "trace" / "providers.py",
        ROOT / "src" / "trace" / "eight_agent_pipeline.py",
        ROOT / "src" / "trace" / "mas_return_pipeline.py",
        ROOT / "src" / "trace" / "trace_method.py",
        ROOT / "src" / "trace" / "router_return_protocol.py",
        ROOT / "src" / "trace" / "router_return_governance.py",
        ROOT / "src" / "trace" / "return_governance.py",
        ROOT / "src" / "trace" / "return_baselines.py",
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
        ROOT / "src" / "trace" / "unified_mas_contract.py",
        ROOT / "src" / "trace" / "memory_backends" / "__init__.py",
        ROOT / "src" / "trace" / "memory_backends" / "base.py",
        ROOT / "src" / "trace" / "memory_backends" / "factory.py",
        ROOT / "src" / "trace" / "memory_backends" / "mem0.py",
        ROOT / "src" / "trace" / "memory_backends" / "memobase.py",
        ROOT / "src" / "trace" / "memory_backends" / "lineage.py",
        ROOT / "src" / "trace" / "memory_backends" / "amem.py",
        ROOT / "src" / "trace" / "memory_backends" / "amem_runtime.py",
        ROOT / "src" / "trace" / "vendor" / "amem_official" / "memory_system.py",
        ROOT / "src" / "trace" / "vendor" / "amem_official" / "retrievers.py",
        ROOT / "src" / "trace" / "vendor" / "amem_official" / "UPSTREAM.json",
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _attach_lifecycle_baselines(
    bundle: Any,
    *,
    arms: Sequence[str],
    provider: OpenAICompatibleProvider | None,
    seed: int,
    cupmem_max_tokens: int,
    kmu_max_tokens: int | None = None,
) -> tuple[Any, tuple[dict[str, object], ...]]:
    """Compile requested Memora lifecycle baselines from one checkpoint."""

    requested = set(arms) & set(LIFECYCLE_BASELINE_ARMS)
    if not requested:
        return bundle, ()
    fork_by_arm = {item.arm.value: item for item in bundle.forks}
    actor_views = dict(bundle.actor_views)
    compilations: dict[str, object] = {}
    calls: list[dict[str, object]] = []
    effective_kmu_max_tokens = (
        cupmem_max_tokens if kmu_max_tokens is None else kmu_max_tokens
    )
    if effective_kmu_max_tokens < 1:
        raise ValueError("kmu_max_tokens must be positive")

    if CHECKPOINT_REPLAY_ARM in requested:
        replay = compile_checkpoint_replay(bundle.checkpoint)
        replay_fork = RouterReturnFork(
            arm=LifecycleReturnArm.CHECKPOINT_REPLAY,
            checkpoint_sha256=bundle.checkpoint.checkpoint_sha256,
            returning_principal_id=(
                bundle.checkpoint.returning_principal_id
            ),
            inherited_item_ids=replay.selected_item_ids,
            candidate_item_ids=replay.candidate_item_ids,
        )
        fork_by_arm[CHECKPOINT_REPLAY_ARM] = replay_fork
        actor_views[CHECKPOINT_REPLAY_ARM] = render_router_fork(
            replay_fork, bundle.checkpoint
        )
        compilations[CHECKPOINT_REPLAY_ARM] = replay.record()

    if TEMPORAL_LWW_ARM in requested:
        temporal = compile_router_temporal_lww(bundle.checkpoint)
        temporal_fork = RouterReturnFork(
            arm=LifecycleReturnArm.TEMPORAL_LWW,
            checkpoint_sha256=bundle.checkpoint.checkpoint_sha256,
            returning_principal_id=(
                bundle.checkpoint.returning_principal_id
            ),
            inherited_item_ids=temporal.selected_ids,
            candidate_item_ids=temporal.candidate_ids,
        )
        fork_by_arm[TEMPORAL_LWW_ARM] = temporal_fork
        actor_views[TEMPORAL_LWW_ARM] = render_router_fork(
            temporal_fork, bundle.checkpoint
        )
        compilations[TEMPORAL_LWW_ARM] = temporal.record()

    if MEMSTRATA_ARM in requested:
        memstrata = compile_router_memstrata(bundle.checkpoint)
        memstrata_fork = RouterReturnFork(
            arm=LifecycleReturnArm.MEMSTRATA,
            checkpoint_sha256=bundle.checkpoint.checkpoint_sha256,
            returning_principal_id=(
                bundle.checkpoint.returning_principal_id
            ),
            inherited_item_ids=memstrata.selected_ids,
            candidate_item_ids=memstrata.candidate_ids,
        )
        fork_by_arm[MEMSTRATA_ARM] = memstrata_fork
        actor_views[MEMSTRATA_ARM] = render_router_fork(
            memstrata_fork, bundle.checkpoint
        )
        compilations[MEMSTRATA_ARM] = memstrata.record()

    if MEMTX_ARM in requested:
        memtx = compile_router_memtx(bundle.checkpoint)
        memtx_fork = RouterReturnFork(
            arm=LifecycleReturnArm.MEMTX,
            checkpoint_sha256=bundle.checkpoint.checkpoint_sha256,
            returning_principal_id=(
                bundle.checkpoint.returning_principal_id
            ),
            inherited_item_ids=memtx.selected_ids,
            candidate_item_ids=memtx.candidate_ids,
        )
        fork_by_arm[MEMTX_ARM] = memtx_fork
        actor_views[MEMTX_ARM] = render_router_fork(
            memtx_fork, bundle.checkpoint
        )
        compilations[MEMTX_ARM] = memtx.record()

    if CUPMEM_ARM in requested:
        if provider is None:
            raise RuntimeError("CUPMem requires the actor model provider")
        prompt = cupmem_adjudication_prompt(bundle.checkpoint)
        cupmem = None
        for attempt in range(3):
            completion = provider.complete(
                prompt,
                system=(
                    "You are the CUPMem lifecycle classifier. Use only the "
                    "actor-visible candidate states and return strict JSON."
                ),
                seed=_arm_seed(
                    seed,
                    bundle.episode.episode_id,
                    CUPMEM_ARM,
                    f"classify:{attempt}",
                ),
                max_tokens=cupmem_max_tokens,
                temperature=0.0,
            )
            cupmem = compile_cupmem(bundle.checkpoint, completion.text)
            calls.append(
                {
                    "stage": "cupmem_lifecycle_classification",
                    "attempt": attempt,
                    "prompt_sha256": canonical_sha256(prompt),
                    "completion": _completion_record(completion),
                    "parse_error": cupmem.parse_error,
                    "official_evidence_visible": False,
                }
            )
            if cupmem.parse_error is None:
                break
        assert cupmem is not None
        if cupmem.parse_error is not None:
            calls[-1]["fallback"] = (
                "fail_closed_unknown_for_missing_or_invalid_items"
            )
        cupmem_fork = RouterReturnFork(
            arm=LifecycleReturnArm.CUPMEM,
            checkpoint_sha256=bundle.checkpoint.checkpoint_sha256,
            returning_principal_id=(
                bundle.checkpoint.returning_principal_id
            ),
            inherited_item_ids=cupmem.selected_item_ids,
            candidate_item_ids=cupmem.candidate_item_ids,
        )
        fork_by_arm[CUPMEM_ARM] = cupmem_fork
        actor_views[CUPMEM_ARM] = render_router_fork(
            cupmem_fork, bundle.checkpoint
        )
        compilations[CUPMEM_ARM] = cupmem.record()

    if KMU_OP_ARM in requested:
        if provider is None:
            raise RuntimeError("KMU-Op requires the actor model provider")
        candidates = router_return_candidates(bundle.checkpoint)
        prompt = indexed_kmu_operation_prompt(candidates)
        kmu = None
        for attempt in range(3):
            completion = provider.complete(
                prompt,
                system=(
                    "You are the KMU operation-based memory updater. Use "
                    "only actor-visible candidates and return strict JSON."
                ),
                seed=_arm_seed(
                    seed,
                    bundle.episode.episode_id,
                    KMU_OP_ARM,
                    f"classify:{attempt}",
                ),
                max_tokens=effective_kmu_max_tokens,
                temperature=0.0,
            )
            kmu = compile_indexed_kmu_operations(candidates, completion.text)
            calls.append(
                {
                    "stage": "kmu_operation_update",
                    "attempt": attempt,
                    "transport_adapter": "local_integer_candidate_index_v1",
                    "prompt_sha256": canonical_sha256(prompt),
                    "completion": _completion_record(completion),
                    "parse_error": kmu.parse_error,
                    "official_evidence_visible": False,
                }
            )
            if kmu.parse_error is None:
                break
        assert kmu is not None
        if kmu.parse_error is not None:
            calls[-1]["fallback"] = "fail_closed_on_invalid_operation_plan"
        kmu_fork = RouterReturnFork(
            arm=LifecycleReturnArm.KMU_OP,
            checkpoint_sha256=bundle.checkpoint.checkpoint_sha256,
            returning_principal_id=(
                bundle.checkpoint.returning_principal_id
            ),
            inherited_item_ids=kmu.selected_ids,
            candidate_item_ids=kmu.candidate_ids,
        )
        fork_by_arm[KMU_OP_ARM] = kmu_fork
        actor_views[KMU_OP_ARM] = render_router_fork(
            kmu_fork, bundle.checkpoint
        )
        compilations[KMU_OP_ARM] = kmu.record()

    ordered_forks = tuple(
        item
        for arm in (*[item.arm.value for item in bundle.forks], *arms)
        if (item := fork_by_arm.get(arm)) is not None
    )
    ordered_forks = tuple(
        {item.arm.value: item for item in ordered_forks}.values()
    )
    audit = dict(bundle.audit)
    audit["lifecycle_baselines"] = {
        "schema_version": "memora_lifecycle_baselines_v1",
        "same_checkpoint_all_arms": len(
            {item.checkpoint_sha256 for item in ordered_forks}
        )
        == 1,
        "official_evidence_runtime_reads": 0,
        "compilations": compilations,
    }
    return (
        replace(
            bundle,
            forks=ordered_forks,
            actor_views=actor_views,
            audit=audit,
        ),
        tuple(calls),
    )


def _extract_json(value: str) -> Mapping[str, object] | list[object]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        values: list[object] = []
        cursor = 0
        while cursor < len(text):
            match = re.search(r"[\[{]", text[cursor:])
            if match is None:
                break
            start = cursor + match.start()
            try:
                decoded_value, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            values.append(decoded_value)
            cursor = end
        if not values:
            raise
        if len(values) == 1:
            decoded = values[0]
        elif all(isinstance(item, Mapping) for item in values):
            # Small local models sometimes emit one row object per line
            # instead of wrapping them in the requested judgments array.
            decoded = values
        else:
            raise ValueError("judge output contains incompatible JSON values")
    if not isinstance(decoded, (Mapping, list)):
        raise ValueError("judge output must be one JSON object or list")
    return decoded


def _clip_middle(value: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(value) <= budget:
        return value
    marker = "\n...[fixed per-item actor-view clip]...\n"
    if budget <= len(marker) + 2:
        return value[-budget:]
    head = (budget - len(marker)) // 3
    tail = budget - len(marker) - head
    return value[:head] + marker + value[-tail:]


def _fit_actor_view(value: str, max_chars: int) -> str:
    """Fit a JSON workstate without dropping items or RETURN metadata."""
    if max_chars < 1:
        raise ValueError("max actor-view characters must be positive")
    if len(value) <= max_chars:
        return value
    decoded = json.loads(value)
    if not isinstance(decoded, Mapping):
        raise ValueError("actor view must be one JSON object")
    raw_items = decoded.get("workstate_items")
    if not isinstance(raw_items, list):
        raise ValueError("oversized actor view has no workstate_items list")
    items = [dict(item) for item in raw_items if isinstance(item, Mapping)]
    if len(items) != len(raw_items):
        raise ValueError("actor view contains a non-object workstate item")
    texts = [str(item.get("text") or "") for item in items]

    def render(per_item_budget: int) -> str:
        compact = dict(decoded)
        compact["workstate_items"] = [
            {
                **item,
                "text": _clip_middle(text, per_item_budget),
            }
            for item, text in zip(items, texts)
        ]
        return json.dumps(
            compact,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    low = 0
    high = max((len(text) for text in texts), default=0)
    best = render(0)
    if len(best) > max_chars:
        raise ValueError(
            "actor-view RETURN metadata alone exceeds character budget"
        )
    while low <= high:
        middle = (low + high) // 2
        candidate = render(middle)
        if len(candidate) <= max_chars:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _with_private_memory(
    prompt: str,
    retrieval: PrivateMemoryRetrieval,
    *,
    max_chars: int = 4000,
) -> str:
    context = retrieval.render(
        max_chars,
        include_utility=False,
    )
    return (
        prompt
        + "\n\n<principal_private_memory>\n"
        + (
            context
            or (
                "[none: this principal has no admissible prior experience "
                "for the current intent]"
            )
        )
        + "\n</principal_private_memory>\n"
        "Private memory is prior execution experience, not authoritative "
        "evidence. Ground every current factual output in the policy-visible "
        "input and current Router workstate."
    )


def _remember_extraction_batch(
    memory: EpisodeLongTermMemory,
    *,
    episode_id: str,
    question: MemoraQuestion,
    batch: PredictedExtractionBatch,
    intent: str,
) -> None:
    if not batch.facts:
        memory.remember(
            principal_id=batch.worker_id,
            task_id=(
                f"{episode_id}:extract:{batch.phase}:"
                + "-".join(str(item) for item in batch.session_ids)
            ),
            intent=intent,
            experience=json.dumps(
                {
                    "assignment": intent,
                    "outcome": "no grounded relevant fact extracted",
                    "session_ids": list(batch.session_ids),
                    "deterministic_fallback": batch.deterministic_fallback,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            attributes={
                "kind": "worker_extraction_experience",
                "phase": batch.phase,
                "question_id": question.question_id,
                "outcome": "empty",
                "return_admissible": False,
            },
        )
        return
    for fact in batch.facts:
        memory.remember(
            principal_id=batch.worker_id,
            task_id=(
                f"{episode_id}:extract:{batch.phase}:{fact.fact_id}"
            ),
            intent=intent,
            experience=json.dumps(
                {
                    "assignment": intent,
                    "state_key": fact.state_key,
                    "statement": fact.statement,
                    "operation": fact.operation,
                    "evidence_quote": fact.evidence_quote,
                    "session_id": fact.session_id,
                    "confidence": fact.confidence,
                    "outcome": "grounded fact accepted",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            dependency_ids=(fact.dependency_id,),
            source_workstate_id=fact.item_id,
            attributes={
                "kind": "worker_extraction_experience",
                "phase": batch.phase,
                "question_id": question.question_id,
                "operation": fact.operation,
                "outcome": "accepted",
                "return_admissible": batch.worker_id == WORKER_IDS[0],
            },
        )


def _seed_oracle_episode_memory(
    memory: EpisodeLongTermMemory,
    *,
    bundle: Any,
    question: MemoraQuestion,
) -> None:
    checkpoint = bundle.checkpoint
    memory.remember(
        principal_id=ROUTER_ID,
        task_id=f"{memory.episode_id}:router:plan",
        intent=f"Plan evidence work for {question.question}",
        experience=json.dumps(
            {
                "task_id": checkpoint.task_id,
                "obligations": [
                    item.record() for item in checkpoint.obligations
                ],
                "outcome": "post-absence checkpoint frozen",
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        attributes={
            "kind": "router_coordination_experience",
            "state_mode": "oracle",
        },
    )
    departure_item_ids: set[str] = set()
    for item in checkpoint.departure.workstate_items:
        departure_item_ids.add(item.item_id)
        memory.remember(
            principal_id=WORKER_IDS[0],
            task_id=f"{memory.episode_id}:prefix:{item.item_id}",
            intent=f"Materialize predeparture state for {question.question}",
            experience=item.text,
            obligation_ids=item.obligation_ids,
            dependency_ids=item.dependency_ids,
            source_workstate_id=item.item_id,
            attributes={
                "kind": "worker_extraction_experience",
                "phase": "predeparture",
                "state_mode": "oracle",
                "outcome": "accepted",
                "return_admissible": True,
            },
        )
    memory.quarantine_returning_worker()
    for item in checkpoint.current_workstate_items:
        if (
            item.item_id in departure_item_ids
            or item.writer_principal_id not in WORKER_IDS[1:]
        ):
            continue
        memory.remember(
            principal_id=item.writer_principal_id,
            task_id=f"{memory.episode_id}:absence:{item.item_id}",
            intent=f"Process absence-period state for {question.question}",
            experience=item.text,
            obligation_ids=item.obligation_ids,
            dependency_ids=item.dependency_ids,
            source_workstate_id=item.item_id,
            attributes={
                "kind": "worker_extraction_experience",
                "phase": "absence",
                "state_mode": "oracle",
                "outcome": "accepted",
                "return_admissible": False,
            },
        )


def _fork_episode_memory_arms(
    memory: EpisodeLongTermMemory,
    *,
    bundle: Any,
    arms: Sequence[str],
) -> dict[str, AgentMemoryArm]:
    forks = {item.arm.value: item for item in bundle.forks}
    result: dict[str, AgentMemoryArm] = {}
    for arm in arms:
        internal_arm = internal_return_arm(arm)
        try:
            fork = forks[internal_arm]
        except KeyError as error:
            raise KeyError(
                f"missing Router fork for private memory arm: {arm}"
            ) from error
        admitted_workstate_ids = (
            ()
            if arm
            in {
                RouterReturnArm.RESET.value,
                RouterReturnArm.RESET_REBRIEF.value,
            }
            else fork.inherited_item_ids
        )
        result[arm] = memory.fork_return_arm(
            arm=arm,
            admitted_workstate_ids=admitted_workstate_ids,
        )
    return result


def _judge_prompt(
    question: MemoraQuestion,
    answer: str,
    *,
    rubric_indices: Sequence[int] | None = None,
) -> str:
    indices = (
        tuple(range(len(question.rubric)))
        if rubric_indices is None
        else tuple(rubric_indices)
    )
    rubric = [
        {
            "index": index,
            "evaluation_question": question.rubric[
                index
            ].evaluation_question,
        }
        for index in indices
    ]
    return (
        "You are the isolated Memora Evaluator. Evaluate the candidate answer "
        "against every yes/no rubric question independently. Return exactly "
        "one JSON object with key 'judgments'. Its value must be a list of "
        "objects containing index and answer, where index is the integer "
        "copied from the rubric and answer is exactly 'yes' or 'no'. Do not "
        "omit, duplicate, or add indices. If the candidate does not mention "
        "or imply the requested fact, answer 'no'. Never return "
        "'not_applicable', 'unknown', or an explanation. The treatment arm "
        "is hidden and irrelevant.\n\n"
        f"<candidate_answer>\n{answer}\n</candidate_answer>\n\n"
        "<rubric>\n"
        + json.dumps(rubric, ensure_ascii=False, sort_keys=True)
        + "\n</rubric>"
    )


def _parse_judgments(
    question: MemoraQuestion,
    *,
    text: str,
) -> tuple[MemoraRubricJudgment, ...]:
    decoded = _extract_json(text)
    rows = decoded if isinstance(decoded, list) else decoded.get("judgments")
    if not isinstance(rows, list):
        raise ValueError("judge output is missing judgments")
    judgments_list: list[MemoraRubricJudgment] = []
    for position, row in enumerate(rows):
        if isinstance(row, Mapping):
            raw_index = row.get("index", row.get("rubric_index"))
            answer = next(
                (
                    row.get(key)
                    for key in (
                        "answer",
                        "judgment",
                        "prediction",
                        "response",
                        "label",
                    )
                    if row.get(key) is not None
                ),
                None,
            )
            if raw_index is None and len(row) == 1:
                only_key, only_value = next(iter(row.items()))
                if str(only_key).strip().isdigit():
                    raw_index = only_key
                    answer = only_value
            if answer is None:
                fallback_answers: set[str] = set()
                for key, value in row.items():
                    if key in {"index", "rubric_index"}:
                        continue
                    try:
                        fallback_answers.add(normalize_yes_no(value))
                    except ValueError:
                        continue
                if len(fallback_answers) == 1:
                    answer = fallback_answers.pop()
        else:
            raw_index = position
            answer = row
        if raw_index is not None:
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"judge rubric index is not an integer: {raw_index!r}"
                ) from error
            if index < 0 or index >= len(question.rubric):
                raise ValueError(f"judge rubric index is out of range: {index}")
            evaluation_question_id = question.rubric[
                index
            ].evaluation_question_id
        else:
            # Backward compatibility for providers that repeat the id despite
            # being asked for the shorter, less error-prone integer index.
            evaluation_question_id = str(
                row.get("evaluation_question_id") or ""
            ).strip()
            if not evaluation_question_id and position < len(question.rubric):
                evaluation_question_id = question.rubric[
                    position
                ].evaluation_question_id
        judgments_list.append(
            MemoraRubricJudgment(
                evaluation_question_id=evaluation_question_id,
                predicted_answer=normalize_yes_no(answer),
            )
        )
    judgments = tuple(judgments_list)
    ids = [item.evaluation_question_id for item in judgments]
    if len(ids) != len(set(ids)):
        raise ValueError("judge emitted duplicate rubric indices")
    return judgments


def _batch_judge(
    provider: OpenAICompatibleProvider,
    *,
    question: MemoraQuestion,
    answer: str,
    seed: int,
    max_tokens: int,
) -> tuple[
    tuple[MemoraRubricJudgment, ...],
    tuple[Completion, ...],
]:
    expected = {
        item.evaluation_question_id: index
        for index, item in enumerate(question.rubric)
    }
    collected: dict[str, MemoraRubricJudgment] = {}
    pending = tuple(range(len(question.rubric)))
    completions: list[Completion] = []
    last_error: Exception | None = None
    for attempt in range(3):
        completion = provider.complete(
            _judge_prompt(
                question,
                answer,
                rubric_indices=pending,
            ),
            system=(
                "You are the benchmark Evaluator. Ground truth and rubrics "
                "must never be disclosed to any task agent or governance "
                "component."
            ),
            seed=seed + attempt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        completions.append(completion)
        try:
            parsed = _parse_judgments(question, text=completion.text)
            for judgment in parsed:
                item_id = judgment.evaluation_question_id
                if item_id not in expected:
                    raise ValueError(
                        f"judge emitted unknown rubric id: {item_id}"
                    )
                if item_id in collected:
                    raise ValueError(
                        f"judge repeated completed rubric id: {item_id}"
                    )
                collected[item_id] = judgment
            pending = tuple(
                index
                for item_id, index in expected.items()
                if item_id not in collected
            )
            if not pending:
                break
        except ValueError as error:
            last_error = error
    if pending:
        missing = ",".join(
            question.rubric[index].evaluation_question_id
            for index in pending
        )
        detail = f"; last_error={last_error}" if last_error else ""
        output_excerpt = (
            completions[-1].text[:2000].encode("unicode_escape").decode()
            if completions
            else ""
        )
        raise ValueError(
            f"judge could not complete rubric after repair; missing={missing}"
            + detail
            + f"; last_output={output_excerpt}"
        )
    judgments = tuple(
        collected[item.evaluation_question_id] for item in question.rubric
    )
    score_memora_fama(question, judgments)
    return judgments, tuple(completions)


def _arm_seed(seed: int, episode_id: str, arm: str, stage: str) -> int:
    # All arms use the same stage seed.  ``arm`` remains in the call
    # signature so receipts can state the matched-arm contract, but it must
    # not introduce an avoidable randomization confound.
    del arm
    digest = hashlib.sha256(
        f"{seed}|{episode_id}|{stage}".encode()
    ).hexdigest()
    return int(digest[:8], 16)


def _generate_arm(
    *,
    episode: MemoraReturnEpisode,
    question: MemoraQuestion,
    arm: str,
    actor_view: str,
    actor_provider: OpenAICompatibleProvider,
    seed: int,
    actor_max_tokens: int,
    max_actor_view_chars: int,
    long_term_memory: EpisodeLongTermMemory | None = None,
    memory_arm: AgentMemoryArm | None = None,
) -> dict[str, object]:
    runtime = long_term_memory or EpisodeLongTermMemory.build(
        episode.episode_id
    )
    if memory_arm is None:
        runtime.quarantine_returning_worker()
        memory_arm = runtime.fork_return_arm(
            arm=arm,
            admitted_workstate_ids=(),
        )
    effective_actor_view = _fit_actor_view(
        actor_view,
        max_actor_view_chars,
    )
    returning_intent = (
        f"Return to answer {question.question} using current valid state"
    )
    returning_retrieval = runtime.recall(
        principal_id=WORKER_IDS[0],
        intent=returning_intent,
        bank=memory_arm.returning_worker,
        branch=arm,
    )
    worker_prompt = _with_private_memory(
        build_returning_worker_prompt(
            question=question,
            arm=arm,
            actor_view=effective_actor_view,
        ),
        returning_retrieval,
    )
    worker_completion = actor_provider.complete(
        worker_prompt,
        system=(
            "You are agent1, the Returning Agent in a five-task-agent MAS. "
            "Answer the user's final task directly from only your admitted "
            "return-memory view."
        ),
        seed=_arm_seed(seed, episode.episode_id, arm, "returning_worker"),
        max_tokens=actor_max_tokens,
        temperature=0.0,
    )
    returning_memory = runtime.remember(
        principal_id=WORKER_IDS[0],
        task_id=f"{episode.episode_id}:return:{arm}",
        intent=returning_intent,
        experience=worker_completion.text,
        bank=memory_arm.returning_worker,
        branch=arm,
        attributes={
            "kind": "returning_worker_experience",
            "arm": arm,
            "outcome": "final_answer_generated",
        },
    )
    answer = worker_completion.text.strip()
    if not answer:
        raise RuntimeError("agent1 produced an empty final answer")
    return {
        "arm": arm,
        "execution_contract": unified_five_task_agent_architecture_record(),
        "final_answer_agent_id": WORKER_IDS[0],
        "aggregator_agent_enabled": False,
        "actor_view_budget": {
            "maximum_chars": max_actor_view_chars,
            "original_chars": len(actor_view),
            "effective_chars": len(effective_actor_view),
            "items_dropped": 0,
        },
        "returning_worker": {
            "prompt_sha256": canonical_sha256(worker_prompt),
            "report": worker_completion.text,
            "completion": _completion_record(worker_completion),
            "ground_truth_visible": False,
            "memory_retrieval": returning_retrieval.record(),
            "stored_memory_id": returning_memory.memory_id,
            "memory_snapshot_sha256": (
                memory_arm.returning_worker.snapshot_sha256
            ),
        },
        "final_answer": {
            "principal_id": WORKER_IDS[0],
            "answer": answer,
            "answer_sha256": hashlib.sha256(
                answer.encode("utf-8")
            ).hexdigest(),
            "ground_truth_visible": False,
        },
        "answer_sha256": hashlib.sha256(
            answer.encode("utf-8")
        ).hexdigest(),
        "scored": False,
    }


def _generated_answer(generated_arm: Mapping[str, object]) -> str:
    """Read the unified agent1 answer, retaining old artifact compatibility."""

    final_answer = generated_arm.get("final_answer")
    if isinstance(final_answer, Mapping):
        answer = str(final_answer.get("answer") or "").strip()
        principal = str(final_answer.get("principal_id") or "")
        if principal != WORKER_IDS[0]:
            raise ValueError("final answer was not generated by agent1")
        return answer
    aggregation = generated_arm.get("aggregation")
    if isinstance(aggregation, Mapping):
        return str(aggregation.get("answer") or "").strip()
    return ""


def _score_generated_arm(
    *,
    episode: MemoraReturnEpisode,
    question: MemoraQuestion,
    generated_arm: Mapping[str, object],
    judge_provider: OpenAICompatibleProvider,
    seed: int,
    judge_max_tokens: int,
    long_term_memory: EpisodeLongTermMemory | None = None,
    evaluator_memory: PrivateEpisodicMemory | None = None,
) -> dict[str, object]:
    arm = str(generated_arm.get("arm") or "")
    if not arm:
        raise ValueError("generation artifact has an invalid arm")
    answer = _generated_answer(generated_arm)
    if not answer:
        raise ValueError("generation artifact has no frozen answer")
    expected_sha256 = str(generated_arm.get("answer_sha256") or "")
    actual_sha256 = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    if expected_sha256 != actual_sha256:
        raise ValueError("frozen answer digest mismatch")
    judgments, judge_completions = _batch_judge(
        judge_provider,
        question=question,
        answer=answer,
        seed=_arm_seed(seed, episode.episode_id, arm, "evaluator"),
        max_tokens=judge_max_tokens,
    )
    score = score_memora_fama(question, judgments)
    evaluator = EvaluatorAgent()
    evaluation = evaluator.evaluate(
        task_id=episode.episode_id,
        arm=arm,
        answer=answer,
        score=lambda _: (
            score.fama,
            {
                "fama": score.fama,
                "memory_presence_accuracy": (
                    score.memory_presence_accuracy
                ),
                "forgetting_absence_accuracy": (
                    score.forgetting_absence_accuracy
                ),
                "overall_accuracy": score.overall_accuracy,
            },
        ),
        scorer="memora_independent_strict_judge_fama_v2",
    )
    evaluator_bank = evaluator_memory or PrivateEpisodicMemory(
        owner_principal_id=EVALUATOR_ID,
        owner_instance_id=f"{EVALUATOR_ID}:{episode.episode_id}:{arm}",
        role_id="role:evaluator",
    )
    evaluation_experience = json.dumps(
        {
            "answer_sha256": actual_sha256,
            "score": score.record(),
            "scorer": evaluation.scorer,
            "outcome": "independent rubric evaluation completed",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if long_term_memory is None:
        stored_evaluation = evaluator_bank.remember(
            requesting_principal_id=EVALUATOR_ID,
            task_id=f"{episode.episode_id}:evaluate:{arm}",
            intent=f"Evaluate the frozen answer for {question.question}",
            experience=evaluation_experience,
            initial_utility=0.5,
            attributes={
                "kind": "evaluation_audit_experience",
                "arm": arm,
                "ground_truth_visible": True,
                "utility_learning_enabled": False,
            },
        )
    else:
        stored_evaluation = long_term_memory.remember(
            principal_id=EVALUATOR_ID,
            task_id=f"{episode.episode_id}:evaluate:{arm}",
            intent=f"Evaluate the frozen answer for {question.question}",
            experience=evaluation_experience,
            bank=evaluator_bank,
            branch=arm,
            attributes={
                "kind": "evaluation_audit_experience",
                "arm": arm,
                "ground_truth_visible": True,
            },
        )
    return {
        **dict(generated_arm),
        "scored": True,
        "evaluation": evaluation.record(),
        "evaluator_memory": {
            "stored_memory_id": stored_evaluation.memory_id,
            "snapshot": evaluator_bank.record(include_experience=True),
        },
        "score": score.record(),
        "judge": {
            "completion": _completion_record(judge_completions[0]),
            "completions": [
                _completion_record(item) for item in judge_completions
            ],
            "repair_calls": len(judge_completions) - 1,
            "rubric_item_count": len(judgments),
            "strict_yes_no_contract": True,
            "non_binary_labels_coerced": False,
            "ground_truth_visible_to_evaluator": True,
            "ground_truth_disclosed_to_other_agents": False,
        },
    }


def _run_arm(
    *,
    episode: MemoraReturnEpisode,
    question: MemoraQuestion,
    arm: str,
    actor_view: str,
    actor_provider: OpenAICompatibleProvider,
    judge_provider: OpenAICompatibleProvider,
    seed: int,
    actor_max_tokens: int,
    judge_max_tokens: int,
    max_actor_view_chars: int,
    long_term_memory: EpisodeLongTermMemory | None = None,
    memory_arm: AgentMemoryArm | None = None,
) -> dict[str, object]:
    """Backward-compatible combined path; formal runs use split scripts."""

    generated = _generate_arm(
        episode=episode,
        question=question,
        arm=arm,
        actor_view=actor_view,
        actor_provider=actor_provider,
        seed=seed,
        actor_max_tokens=actor_max_tokens,
        max_actor_view_chars=max_actor_view_chars,
        long_term_memory=long_term_memory,
        memory_arm=memory_arm,
    )
    return _score_generated_arm(
        episode=episode,
        question=question,
        generated_arm=generated,
        judge_provider=judge_provider,
        seed=seed,
        judge_max_tokens=judge_max_tokens,
        long_term_memory=long_term_memory,
        evaluator_memory=(
            memory_arm.evaluator if memory_arm is not None else None
        ),
    )


def _safe_episode_record(
    episode: MemoraReturnEpisode,
    *,
    departure_session_id: int,
    return_session_id: int,
    departure_fraction: float,
) -> dict[str, object]:
    """Record episode identity without serializing oracle witness labels."""

    return {
        "schema_version": "memora_predicted_episode_identity_v1",
        "episode_id": episode.episode_id,
        "cluster_id": episode.cluster_id,
        "period": episode.period,
        "persona": episode.persona,
        "task_type": episode.task_type,
        "question_id": episode.question_id,
        "timeline_sha256": episode.timeline_sha256,
        "source_revision": episode.source_revision,
        "runtime_event": {
            "departure_session_id": departure_session_id,
            "return_session_id": return_session_id,
            "departure_fraction": departure_fraction,
            "selection_policy": "fixed_timeline_fraction",
            "oracle_witness_consumed": False,
        },
    }


def _build_predicted_trace(
    *,
    episode: MemoraReturnEpisode,
    timeline: Any,
    question: MemoraQuestion,
    provider: OpenAICompatibleProvider,
    seed: int,
    departure_fraction: float,
    predeparture_top_k: int,
    absence_top_k: int,
    predeparture_recency_k: int,
    absence_recency_k: int,
    max_sessions_per_batch: int,
    minimum_confidence: float,
    base_selected_facts: int,
    max_selected_facts: int,
    prediction_max_tokens: int,
    structured_output_mode: str = "auto",
    long_term_memory: EpisodeLongTermMemory,
) -> tuple[PredictedTrace, tuple[dict[str, object], ...]]:
    """Run outcome-blind task-agent extraction with a deterministic scheduler."""

    if structured_output_mode not in {"auto", "forced_tool", "prompt_json"}:
        raise ValueError("invalid predicted structured output mode")
    sessions = tuple(sorted(timeline.sessions, key=lambda item: item.session_id))
    if len(sessions) < 2:
        raise ValueError("predicted RETURN requires at least two sessions")
    split = min(
        len(sessions) - 2,
        max(0, int(len(sessions) * departure_fraction) - 1),
    )
    departure_session_id = sessions[split].session_id
    return_session_id = sessions[-1].session_id
    question_record = safe_question_record(
        question.question_id,
        question.task_type,
        question.question,
        question.question_date,
    )
    receipts: list[dict[str, object]] = []

    def forced_tool_contract(stage: str) -> tuple[str, dict[str, object]]:
        if stage.startswith(("predeparture_extract:", "absence_extract:")):
            return (
                "emit_grounded_facts",
                {
                    "type": "object",
                    "properties": {
                        "facts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "session_id": {"type": "integer"},
                                    "state_key": {"type": "string"},
                                    "statement": {"type": "string"},
                                    "operation": {
                                        "type": "string",
                                        "enum": ["add", "update", "delete"],
                                    },
                                    "evidence_quote": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": [
                                    "session_id",
                                    "state_key",
                                    "statement",
                                    "operation",
                                    "evidence_quote",
                                    "confidence",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["facts"],
                    "additionalProperties": False,
                },
            )
        raise ValueError(f"no forced tool contract for stage: {stage}")

    def complete_and_parse(
        *,
        stage: str,
        prompt: str,
        system: str,
        parse: Any,
        memory_retrieval: PrivateMemoryRetrieval | None = None,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            completion_seed = _arm_seed(
                seed,
                episode.episode_id,
                "predicted",
                f"{stage}:{attempt}",
            )
            model_name = provider.generation_model.casefold()
            tool_name, tool_parameters = forced_tool_contract(stage)
            use_forced_tool = (
                structured_output_mode == "forced_tool"
                or (
                    structured_output_mode == "auto"
                    and model_name.startswith("claude-")
                )
            )
            completion_prompt = prompt
            if use_forced_tool:
                completion = provider.complete_with_forced_tool(
                    completion_prompt,
                    tool_name=tool_name,
                    tool_description=(
                        "Return the requested structured Router or Worker "
                        "result without any conversational answer."
                    ),
                    tool_parameters=tool_parameters,
                    system=system,
                    seed=completion_seed,
                    max_tokens=prediction_max_tokens,
                    temperature=0.0,
                )
                completion_mode = "forced_tool"
            else:
                if structured_output_mode == "prompt_json":
                    completion_prompt = (
                        prompt
                        + "\n\nReturn only one JSON object matching this schema: "
                        + json.dumps(
                            tool_parameters,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                completion = provider.complete(
                    completion_prompt,
                    system=system,
                    seed=completion_seed,
                    max_tokens=prediction_max_tokens,
                    temperature=0.0,
                )
                completion_mode = "prompt_json"
            receipt = {
                "stage": stage,
                "attempt": attempt,
                "prompt_sha256": canonical_sha256(completion_prompt),
                "completion": _completion_record(completion),
                "official_evidence_visible": False,
                "structured_output_mode": (
                    completion_mode
                ),
            }
            if memory_retrieval is not None:
                receipt["private_memory_retrieval"] = (
                    memory_retrieval.record()
                )
            receipts.append(receipt)
            try:
                return parse(completion.text)
            except (TypeError, ValueError, RuntimeError) as error:
                last_error = error
                receipt["parse_error"] = str(error)
        raise ValueError(
            f"predicted stage {stage} failed after repair: {last_error}"
        )

    question_text = str(question_record.get("question") or "").strip()
    task_type = str(question_record.get("task_type") or "").strip()
    retrieval_queries = tuple(
        item for item in (question_text, task_type) if item
    )
    router_plan = PredictedRouterPlan(
        obligation=question_text,
        retrieval_queries=retrieval_queries,
        raw_completion_sha256=canonical_sha256(
            {
                "component": "lifecycle_scheduler",
                "policy": "question_text_and_task_type_v1",
                "question": question_record,
            }
        ),
    )

    pre_records = actor_session_records(
        timeline,
        first_session_id=sessions[0].session_id,
        last_session_id=departure_session_id,
    )
    absence_records = actor_session_records(
        timeline,
        first_session_id=departure_session_id + 1,
        last_session_id=return_session_id,
    )
    pre_candidates = retrieve_actor_sessions(
        pre_records,
        question_record=question_record,
        retrieval_queries=router_plan.retrieval_queries,
        top_k=predeparture_top_k,
        recency_k=predeparture_recency_k,
    )
    absence_candidates = retrieve_actor_sessions(
        absence_records,
        question_record=question_record,
        retrieval_queries=router_plan.retrieval_queries,
        top_k=absence_top_k,
        recency_k=absence_recency_k,
    )
    batches: list[PredictedExtractionBatch] = []

    def degraded_empty_batch(
        *,
        phase: str,
        worker_id: str,
        records: Sequence[Mapping[str, object]],
        stage: str,
    ) -> PredictedExtractionBatch:
        return PredictedExtractionBatch(
            phase=phase,
            worker_id=worker_id,
            session_ids=tuple(int(item["session_id"]) for item in records),
            facts=(),
            rejected_rows=0,
            raw_completion_sha256=canonical_sha256(
                {
                    "stage": stage,
                    "status": "model_parse_failed_after_repair",
                }
            ),
            deterministic_fallback=True,
        )

    for batch_index, (worker_id, records) in enumerate(
        shard_actor_sessions(
            pre_candidates,
            (WORKER_IDS[0],),
            max_sessions_per_batch=max_sessions_per_batch,
        )
    ):
        extraction_intent = (
            "Extract grounded predeparture workstate for obligation "
            f"{router_plan.obligation}; sessions "
            + ",".join(str(item["session_id"]) for item in records)
        )
        worker_retrieval = long_term_memory.recall(
            principal_id=worker_id,
            intent=extraction_intent,
        )
        prompt = _with_private_memory(
            extraction_prompt(
                question_record=question_record,
                router_plan=router_plan,
                phase="predeparture",
                worker_id=worker_id,
                sessions=records,
            ),
            worker_retrieval,
        )
        stage = f"predeparture_extract:{batch_index}"
        try:
            extracted = complete_and_parse(
                stage=stage,
                prompt=prompt,
                system=(
                    "You are agent1 before departure. Extract only exact, "
                    "actor-visible personalized workstate."
                ),
                parse=lambda text, worker_id=worker_id, records=records: (
                    parse_extraction(
                        text,
                        phase="predeparture",
                        worker_id=worker_id,
                        sessions=records,
                        minimum_confidence=minimum_confidence,
                    )
                ),
                memory_retrieval=worker_retrieval,
            )
        except ValueError:
            extracted = degraded_empty_batch(
                phase="predeparture",
                worker_id=worker_id,
                records=records,
                stage=stage,
            )
        batches.append(extracted)
        _remember_extraction_batch(
            long_term_memory,
            episode_id=episode.episode_id,
            question=question,
            batch=extracted,
            intent=extraction_intent,
        )
    pre_current = active_facts_at_departure(
        tuple(fact for batch in batches for fact in batch.facts)
    )
    if not pre_current:
        repair_intent = (
            "Repair predeparture extraction for obligation "
            f"{router_plan.obligation}"
        )
        repair_retrieval = long_term_memory.recall(
            principal_id=WORKER_IDS[0],
            intent=repair_intent,
        )
        repair_prompt = _with_private_memory(
            extraction_prompt(
                question_record=question_record,
                router_plan=router_plan,
                phase="predeparture",
                worker_id=WORKER_IDS[0],
                sessions=pre_candidates,
            ),
            repair_retrieval,
        )

        def parse_nonempty_predeparture(text: str) -> PredictedExtractionBatch:
            batch = parse_extraction(
                text,
                phase="predeparture",
                worker_id=WORKER_IDS[0],
                sessions=pre_candidates,
                minimum_confidence=minimum_confidence,
            )
            if not batch.facts:
                raise ValueError(
                    "focused predeparture repair extracted no grounded fact"
                )
            return batch

        try:
            repaired = complete_and_parse(
                stage="predeparture_extract:focused_repair",
                prompt=repair_prompt,
                system=(
                    "You are agent1 before departure. If a retrieved session "
                    "contains a relevant fact, copy one exact complete user "
                    "message as evidence_quote. Otherwise return no facts."
                ),
                parse=parse_nonempty_predeparture,
                memory_retrieval=repair_retrieval,
            )
            batches.append(repaired)
            _remember_extraction_batch(
                long_term_memory,
                episode_id=episode.episode_id,
                question=question,
                batch=repaired,
                intent=repair_intent,
            )
            pre_current = active_facts_at_departure(repaired.facts)
        except ValueError:
            # Some fixed-50% episodes genuinely have no task-relevant V state.
            # Preserve them for intention-to-treat reporting with one
            # explicitly marked, raw-only continuity item; the stricter
            # DivergenceWitness gate will exclude them from causal analysis.
            if not pre_candidates:
                raise ValueError(
                    "fixed departure interval has no actor-visible session"
                )
            source = pre_candidates[-1]
            messages = [
                str(item.get("message") or "").strip()
                for item in source.get("conversation") or ()
                if isinstance(item, Mapping)
                and str(item.get("message") or "").strip()
            ]
            if not messages:
                raise ValueError(
                    "fixed departure interval has no actor-visible message"
                )
            quote = messages[-1]
            session_id = int(source["session_id"])
            source_sha256 = str(source["source_sha256"])
            fallback_id = "pf_" + canonical_sha256(
                {
                    "session_id": session_id,
                    "state_key": "workspace-continuity",
                    "statement": quote,
                    "source_sha256": source_sha256,
                    "deterministic_fallback": True,
                }
            )[:24]
            fallback_fact = PredictedStateFact(
                fact_id=fallback_id,
                session_id=session_id,
                phase="predeparture",
                worker_id=WORKER_IDS[0],
                state_key=f"workspace-continuity:{session_id}",
                statement=quote[:1200],
                operation="add",
                evidence_quote=quote[:500],
                source_sha256=source_sha256,
                confidence=0.0,
                quote_verified=True,
            )
            fallback_batch = PredictedExtractionBatch(
                phase="predeparture",
                worker_id=WORKER_IDS[0],
                session_ids=(session_id,),
                facts=(fallback_fact,),
                rejected_rows=0,
                raw_completion_sha256=canonical_sha256(
                    fallback_fact.record()
                ),
                deterministic_fallback=True,
            )
            batches.append(fallback_batch)
            _remember_extraction_batch(
                long_term_memory,
                episode_id=episode.episode_id,
                question=question,
                batch=fallback_batch,
                intent=repair_intent,
            )
            pre_current = (fallback_fact,)

    long_term_memory.quarantine_returning_worker()

    for batch_index, (worker_id, records) in enumerate(
        shard_actor_sessions(
            absence_candidates,
            WORKER_IDS[1:],
            max_sessions_per_batch=max_sessions_per_batch,
        )
    ):
        extraction_intent = (
            "Detect absence-period additions, updates, and deletions for "
            f"obligation {router_plan.obligation}; sessions "
            + ",".join(str(item["session_id"]) for item in records)
        )
        worker_retrieval = long_term_memory.recall(
            principal_id=worker_id,
            intent=extraction_intent,
        )
        prompt = _with_private_memory(
            extraction_prompt(
                question_record=question_record,
                router_plan=router_plan,
                phase="absence",
                worker_id=worker_id,
                sessions=records,
                predeparture_catalog=pre_current,
            ),
            worker_retrieval,
        )
        stage = f"absence_extract:{worker_id}:{batch_index}"
        try:
            extracted = complete_and_parse(
                stage=stage,
                prompt=prompt,
                system=(
                    "You are an active Worker while agent1 is absent. Detect "
                    "relevant additions, updates, and deletions from exact "
                    "raw conversation evidence."
                ),
                parse=lambda text, worker_id=worker_id, records=records: (
                    parse_extraction(
                        text,
                        phase="absence",
                        worker_id=worker_id,
                        sessions=records,
                        minimum_confidence=minimum_confidence,
                    )
                ),
                memory_retrieval=worker_retrieval,
            )
        except ValueError:
            extracted = degraded_empty_batch(
                phase="absence",
                worker_id=worker_id,
                records=records,
                stage=stage,
            )
        batches.append(extracted)
        _remember_extraction_batch(
            long_term_memory,
            episode_id=episode.episode_id,
            question=question,
            batch=extracted,
            intent=extraction_intent,
        )
    all_facts = tuple(fact for batch in batches for fact in batch.facts)
    current_facts = active_facts_at_departure(all_facts)
    if not current_facts:
        if not all_facts:
            raise ValueError("Predicted timeline has no grounded source fact")
        source = max(
            all_facts, key=lambda item: (item.session_id, item.fact_id)
        )
        tombstone_id = "pf_" + canonical_sha256(
            {
                "source_fact_id": source.fact_id,
                "state_key": "current-empty-state",
                "deterministic_current_fallback": True,
            }
        )[:24]
        tombstone = PredictedStateFact(
            fact_id=tombstone_id,
            session_id=source.session_id,
            phase=source.phase,
            worker_id=source.worker_id,
            state_key=f"current-empty-state:{source.session_id}",
            statement=(
                "No active value remains for this explicitly changed state: "
                + source.evidence_quote
            )[:1200],
            operation="add",
            evidence_quote=source.evidence_quote,
            source_sha256=source.source_sha256,
            confidence=0.0,
            quote_verified=True,
        )
        tombstone_batch = PredictedExtractionBatch(
            phase=source.phase,
            worker_id=source.worker_id,
            session_ids=(source.session_id,),
            facts=(tombstone,),
            rejected_rows=0,
            raw_completion_sha256=canonical_sha256(
                tombstone.record()
            ),
            deterministic_fallback=True,
        )
        batches.append(tombstone_batch)
        if source.worker_id != WORKER_IDS[0]:
            _remember_extraction_batch(
                long_term_memory,
                episode_id=episode.episode_id,
                question=question,
                batch=tombstone_batch,
                intent=(
                    "Record deterministic current-state continuity fallback"
                ),
            )
        all_facts = (*all_facts, tombstone)
        current_facts = (tombstone,)
    selector_budget = dynamic_selector_budget(
        current_facts,
        obligation_query_count=len(router_plan.retrieval_queries),
        base_selected_facts=base_selected_facts,
        max_selected_facts=max_selected_facts,
    )
    effective_selector_budget = selector_budget.effective_budget
    if effective_selector_budget < 1:
        raise ValueError("dynamic selector produced an empty budget")
    selected_current = tuple(
        sorted(
            current_facts,
            key=lambda item: (item.session_id, item.fact_id),
        )[:effective_selector_budget]
    )
    dependency_plan = PredictedDependencyPlan(
        selected_fact_ids=tuple(item.fact_id for item in selected_current),
        critical_fact_ids=tuple(item.fact_id for item in selected_current),
        rationale=(
            "Deterministic Lifecycle Scheduler preserves every current "
            "actor-grounded fact up to the frozen budget."
        ),
        raw_completion_sha256=canonical_sha256(
            {
                "component": "lifecycle_scheduler",
                "policy": "all_current_facts_chronological_v1",
                "selected_fact_ids": [
                    item.fact_id for item in selected_current
                ],
            }
        ),
        deterministic_fallback=True,
    )
    trace = PredictedTrace(
        departure_session_id=departure_session_id,
        return_session_id=return_session_id,
        departure_fraction=departure_fraction,
        router_plan=router_plan,
        predeparture_candidate_session_ids=tuple(
            int(item["session_id"]) for item in pre_candidates
        ),
        absence_candidate_session_ids=tuple(
            int(item["session_id"]) for item in absence_candidates
        ),
        extraction_batches=tuple(batches),
        dependency_plan=dependency_plan,
        selector_budget=selector_budget,
    )
    return trace, tuple(receipts)


def _ratio(numerator: set[int], denominator: set[int]) -> float | None:
    if not denominator:
        return None
    return len(numerator & denominator) / len(denominator)


def _precision(predicted: set[int], expected: set[int]) -> float | None:
    if not predicted:
        return None
    return len(predicted & expected) / len(predicted)


def _predicted_evaluator_metrics(
    *,
    question: MemoraQuestion,
    trace: PredictedTrace,
) -> dict[str, object]:
    """Use official labels only after the actor-visible trace is frozen."""

    facts = {item.fact_id: item for item in trace.facts}
    extracted = {item.session_id for item in facts.values()}
    selected = {
        facts[item].session_id
        for item in trace.dependency_plan.selected_fact_ids
        if item in facts
    }
    predicted_mutations = {
        item.session_id
        for item in facts.values()
        if item.operation in {"update", "delete"}
    }
    memory_sessions = set(evidence_session_ids(question.memory_evidence))
    forgetting_sessions = {
        session_id
        for _, session_id in forgetting_value_sessions(
            question.forgetting_evidence
        )
    }
    expected_relevant = memory_sessions | forgetting_sessions
    return {
        "schema_version": "memora_predicted_evaluator_metrics_v1",
        "computed_after_return_view_frozen": True,
        "consumed_by_lifecycle_scheduler_or_task_agents": False,
        "official_memory_session_count": len(memory_sessions),
        "official_forgetting_mutation_session_count": len(
            forgetting_sessions
        ),
        "predicted_fact_session_count": len(extracted),
        "selected_fact_session_count": len(selected),
        "fact_extraction_precision": _precision(extracted, expected_relevant),
        "fact_extraction_recall": _ratio(extracted, expected_relevant),
        "dependency_session_precision": _precision(selected, memory_sessions),
        "dependency_session_recall": _ratio(selected, memory_sessions),
        "update_delete_session_precision": _precision(
            predicted_mutations, forgetting_sessions
        ),
        "update_delete_session_recall": _ratio(
            predicted_mutations, forgetting_sessions
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument(
        "--api-key-env", default="TRACE_ACTOR_API_KEY"
    )
    _add_embedding_arguments(parser)
    add_memory_backend_arguments(parser)
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--judge-api-key-env", default="OPENAI_API_KEY"
    )
    parser.add_argument("--arms", type=_csv, default=DEFAULT_ARMS)
    parser.add_argument(
        "--state-mode",
        choices=("oracle", "predicted"),
        default="oracle",
        help=(
            "oracle reuses official evidence to materialize workstate; "
            "predicted uses only raw actor-visible sessions at runtime"
        ),
    )
    parser.add_argument(
        "--predicted-departure-fraction", type=float, default=0.50
    )
    parser.add_argument("--predicted-predeparture-top-k", type=int, default=10)
    parser.add_argument("--predicted-absence-top-k", type=int, default=14)
    parser.add_argument(
        "--predicted-predeparture-recency-k", type=int, default=2
    )
    parser.add_argument("--predicted-absence-recency-k", type=int, default=4)
    parser.add_argument(
        "--predicted-max-sessions-per-batch", type=int, default=6
    )
    parser.add_argument(
        "--predicted-minimum-confidence", type=float, default=0.45
    )
    parser.add_argument(
        "--predicted-base-selected-facts", type=int, default=12
    )
    parser.add_argument(
        "--predicted-max-selected-facts", type=int, default=16
    )
    parser.add_argument(
        "--predicted-max-tokens", type=int, default=1536
    )
    parser.add_argument(
        "--predicted-structured-output-mode",
        choices=("auto", "forced_tool", "prompt_json"),
        default="auto",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--actor-max-tokens", type=int, default=768)
    parser.add_argument("--cupmem-max-tokens", type=int, default=4096)
    parser.add_argument("--kmu-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--max-actor-view-chars", type=int, default=16_000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--empty-content-retries", type=int, default=2)
    parser.add_argument(
        "--actor-thinking-mode",
        choices=("default", "adaptive", "disabled"),
        default="default",
    )
    parser.add_argument(
        "--actor-omit-temperature",
        action="store_true",
        help="Omit deprecated temperature from Sonnet-compatible requests.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and audit all forks without calling a model.",
    )
    return parser


def _validate_arms(arms: Sequence[str]) -> tuple[str, ...]:
    canonical = tuple(canonical_return_arm(arm) for arm in arms)
    known = (
        {item.value for item in RouterReturnArm}
        | {TRACE_ARM}
        | set(LIFECYCLE_BASELINE_ARMS)
    )
    unknown = set(canonical) - known
    if unknown:
        raise ValueError("unknown RETURN arms: " + ",".join(sorted(unknown)))
    if len(canonical) != len(set(canonical)):
        raise ValueError("RETURN arms must be unique")
    return canonical


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = _validate_arms(args.arms)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.shard_count < 1:
        raise ValueError("shard count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard_count)")
    if not 0.0 < args.predicted_departure_fraction < 1.0:
        raise ValueError("predicted departure fraction must be in (0, 1)")
    for name in (
        "predicted_predeparture_top_k",
        "predicted_absence_top_k",
        "predicted_max_sessions_per_batch",
        "predicted_base_selected_facts",
        "predicted_max_selected_facts",
        "predicted_max_tokens",
        "cupmem_max_tokens",
        "kmu_max_tokens",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= args.predicted_minimum_confidence <= 1.0:
        raise ValueError("predicted minimum confidence must be in [0, 1]")
    if args.empty_content_retries < 0:
        raise ValueError("empty content retries must be non-negative")
    episodes = load_memora_manifest(args.manifest.resolve())
    if args.limit is not None:
        episodes = episodes[: args.limit]
    episodes = tuple(
        episode
        for index, episode in enumerate(episodes)
        if index % args.shard_count == args.shard_index
    )
    model_required = (
        not args.dry_run
        or args.state_mode == "predicted"
        or CUPMEM_ARM in arms
        or KMU_OP_ARM in arms
        or args.memory_backend not in {ASVM_BACKEND, SCOPEDMEM_BACKEND}
    )
    if model_required and (not args.base_url or not args.model):
        raise ValueError(
            "--base-url and --model are required for model-backed runs"
        )
    actor_provider: OpenAICompatibleProvider | None = None
    judge_provider: OpenAICompatibleProvider | None = None
    embedding_provider = _build_embedding_provider(args)
    actor_key = os.environ.get(args.api_key_env)
    if model_required:
        if not actor_key:
            raise RuntimeError(f"missing API key env: {args.api_key_env}")
        actor_provider = OpenAICompatibleProvider(
            generation_base_url=args.base_url,
            generation_model=args.model,
            generation_api_key=actor_key,
            timeout=args.timeout,
            retries=args.retries,
            empty_content_retries=args.empty_content_retries,
            omit_temperature=args.actor_omit_temperature,
            chat_request_overrides=(
                None
                if args.actor_thinking_mode == "default"
                else {"thinking": {"type": args.actor_thinking_mode}}
            ),
        )
    if model_required:
        assert actor_key is not None
    backend_factory = build_memory_backend_factory(
        args,
        generation_api_key=actor_key or "dry-run",
        embedding_provider=embedding_provider,
        default_storage_dir=(
            args.output_dir.resolve() / "native_backend_state"
        ),
    )
    if not args.dry_run:
        judge_url = args.judge_base_url or args.base_url
        judge_model = args.judge_model or args.model
        judge_key = os.environ.get(args.judge_api_key_env) or actor_key
        judge_provider = OpenAICompatibleProvider(
            generation_base_url=judge_url,
            generation_model=judge_model,
            generation_api_key=judge_key,
            timeout=args.timeout,
            retries=args.retries,
        )
    output_dir = args.output_dir.resolve()
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    timeline_cache: dict[tuple[str, str], Any] = {}
    completed = 0
    failed = 0
    started = time.perf_counter()
    source_contract = _source_contract()
    for index, episode in enumerate(episodes):
        output_path = episode_dir / (
            re.sub(r"[^a-zA-Z0-9._-]+", "_", episode.episode_id) + ".json"
        )
        if args.resume and output_path.is_file():
            completed += 1
            continue
        key = (episode.period, episode.persona)
        if key not in timeline_cache:
            timeline_cache[key] = load_memora_timeline(
                args.data_root.resolve(),
                period=episode.period,
                persona=episode.persona,
            )
        timeline = timeline_cache[key]
        question = timeline.by_question_id[episode.question_id]
        try:
            governor = ReturnMemoryGovernor(
                "memora-return-authority",
                hashlib.sha256(
                    (
                        "memora-return-governor|"
                        + episode.episode_id
                        + "|"
                        + str(args.seed)
                    ).encode()
                ).digest(),
            )
            prediction_receipts: tuple[dict[str, object], ...] = ()
            lifecycle_baseline_calls: tuple[dict[str, object], ...] = ()
            predicted_metrics: dict[str, object] | None = None
            episode_memory = backend_factory.build_episode(
                episode.episode_id
            )
            if args.state_mode == "predicted":
                assert actor_provider is not None
                trace, prediction_receipts = _build_predicted_trace(
                    episode=episode,
                    timeline=timeline,
                    question=question,
                    provider=actor_provider,
                    seed=args.seed,
                    departure_fraction=args.predicted_departure_fraction,
                    predeparture_top_k=args.predicted_predeparture_top_k,
                    absence_top_k=args.predicted_absence_top_k,
                    predeparture_recency_k=(
                        args.predicted_predeparture_recency_k
                    ),
                    absence_recency_k=args.predicted_absence_recency_k,
                    max_sessions_per_batch=(
                        args.predicted_max_sessions_per_batch
                    ),
                    minimum_confidence=args.predicted_minimum_confidence,
                    base_selected_facts=(
                        args.predicted_base_selected_facts
                    ),
                    max_selected_facts=args.predicted_max_selected_facts,
                    prediction_max_tokens=args.predicted_max_tokens,
                    structured_output_mode=(
                        args.predicted_structured_output_mode
                    ),
                    long_term_memory=episode_memory,
                )
                bundle = build_predicted_return_forks(
                    timeline,
                    episode,
                    question_record=question.actor_record(),
                    trace=trace,
                    governor=governor,
                )
                # The official annotations enter only after all arm views are
                # frozen and are never copied into an actor-visible record.
                predicted_metrics = _predicted_evaluator_metrics(
                    question=question,
                    trace=trace,
                )
                episode_record = _safe_episode_record(
                    episode,
                    departure_session_id=trace.departure_session_id,
                    return_session_id=trace.return_session_id,
                    departure_fraction=trace.departure_fraction,
                )
            else:
                bundle = build_memora_return_forks(
                    timeline,
                    episode,
                    governor=governor,
                )
                _seed_oracle_episode_memory(
                    episode_memory,
                    bundle=bundle,
                    question=question,
                )
                episode_record = episode.record()
            bundle, lifecycle_baseline_calls = _attach_lifecycle_baselines(
                bundle,
                arms=arms,
                provider=actor_provider,
                seed=args.seed,
                cupmem_max_tokens=args.cupmem_max_tokens,
                kmu_max_tokens=args.kmu_max_tokens,
            )
            departure_session_id = (
                trace.departure_session_id
                if args.state_mode == "predicted"
                else episode.witness.departure_session_id
            )
            return_session_id = (
                trace.return_session_id
                if args.state_mode == "predicted"
                else episode.witness.return_session_id
            )
            memory_arms = _fork_episode_memory_arms(
                episode_memory,
                bundle=bundle,
                arms=arms,
            )
            record: dict[str, object] = {
                "schema_version": SCHEMA,
                "mas_execution": memora_unified_mas_episode_record(
                    timeline,
                    episode,
                    departure_session_id=departure_session_id,
                    return_session_id=return_session_id,
                    checkpoint_sha256=bundle.checkpoint.checkpoint_sha256,
                ),
                "state_mode": args.state_mode,
                "episode": episode_record,
                "question_actor_record": question.actor_record(),
                "fork_bundle": bundle.record(),
                "selected_arms": list(arms),
                "dry_run": args.dry_run,
                "source_contract_sha256": canonical_sha256(source_contract),
                "memory_backend": backend_factory.record(),
                "created_at_utc": _utc_now(),
                "arms": {},
            }
            if args.state_mode == "predicted":
                record["prediction_calls"] = list(prediction_receipts)
                record["prediction_quality_evaluator_only"] = (
                    predicted_metrics
                )
            if lifecycle_baseline_calls:
                record["lifecycle_baseline_calls"] = list(
                    lifecycle_baseline_calls
                )
            if not args.dry_run:
                assert actor_provider is not None
                assert judge_provider is not None
                arm_records: dict[str, object] = {}
                for arm in arms:
                    actor_view = bundle.actor_views[internal_return_arm(arm)]
                    arm_record = _run_arm(
                        episode=episode,
                        question=question,
                        arm=arm,
                        actor_view=actor_view,
                        actor_provider=actor_provider,
                        judge_provider=judge_provider,
                        seed=args.seed,
                        actor_max_tokens=args.actor_max_tokens,
                        judge_max_tokens=args.judge_max_tokens,
                        max_actor_view_chars=args.max_actor_view_chars,
                        long_term_memory=episode_memory,
                        memory_arm=memory_arms[arm],
                    )
                    arm_records[arm] = arm_record
                record["arms"] = arm_records
                record["provider"] = {
                    "actor": {
                        "origin": endpoint_origin(args.base_url),
                        "model": args.model,
                    },
                    "judge": {
                        "origin": endpoint_origin(
                            args.judge_base_url or args.base_url
                        ),
                        "model": args.judge_model or args.model,
                    },
                    "embedding": (
                        {
                            "origin": endpoint_origin(
                                args.embedding_base_url
                            ),
                            "model": args.embedding_model,
                            "dimensions": args.embedding_dimensions,
                        }
                        if embedding_provider is not None
                        else None
                    ),
                    "api_keys_persisted": False,
                }
            elif actor_provider is not None:
                record["provider"] = {
                    "actor": {
                        "origin": endpoint_origin(args.base_url),
                        "model": args.model,
                    },
                    "judge": None,
                    "api_keys_persisted": False,
                }
            record["long_term_memory"] = episode_memory.record(
                include_private_experience=True
            )
            record["record_sha256"] = canonical_sha256(record)
            output_path.write_text(
                json.dumps(
                    record, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
            failure_path = output_path.with_suffix(".failure.json")
            if failure_path.is_file():
                failure_path.unlink()
            completed += 1
        except Exception as error:  # fail one episode, preserve campaign
            failed += 1
            failure_path = output_path.with_suffix(".failure.json")
            failure_path.write_text(
                json.dumps(
                    {
                        "schema_version": "memora_mas_return_failure_v1",
                        "episode_id": episode.episode_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "created_at_utc": _utc_now(),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "progress": f"{index + 1}/{len(episodes)}",
                    "completed": completed,
                    "failed": failed,
                    "episode_id": episode.episode_id,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    campaign = {
        "schema_version": "memora_mas_return_campaign_receipt_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "data_root": str(args.data_root.resolve()),
        "requested_episodes": len(episodes),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "completed": completed,
        "failed": failed,
        "arms": list(arms),
        "state_mode": args.state_mode,
        "predicted_runtime_contract": (
            {
                "departure_fraction": args.predicted_departure_fraction,
                "predeparture_top_k": args.predicted_predeparture_top_k,
                "absence_top_k": args.predicted_absence_top_k,
                "predeparture_recency_k": (
                    args.predicted_predeparture_recency_k
                ),
                "absence_recency_k": args.predicted_absence_recency_k,
                "max_sessions_per_batch": (
                    args.predicted_max_sessions_per_batch
                ),
                "minimum_confidence": (
                    args.predicted_minimum_confidence
                ),
                "max_selected_facts": (
                    args.predicted_max_selected_facts
                ),
                "base_selected_facts": (
                    args.predicted_base_selected_facts
                ),
                "selector_budget_policy": (
                    "dynamic_obligation_coverage_v1"
                ),
                "structured_output_mode": (
                    args.predicted_structured_output_mode
                ),
                "official_evidence_runtime_reads": 0,
            }
            if args.state_mode == "predicted"
            else None
        ),
        "dry_run": args.dry_run,
        "source_contract": source_contract,
        "source_contract_sha256": canonical_sha256(source_contract),
        "elapsed_seconds": time.perf_counter() - started,
        "created_at_utc": _utc_now(),
    }
    receipt_name = (
        "campaign_receipt.json"
        if args.shard_count == 1
        else (
            f"campaign_receipt.shard_{args.shard_index:02d}_of_"
            f"{args.shard_count:02d}.json"
        )
    )
    (output_dir / receipt_name).write_text(
        json.dumps(campaign, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(campaign, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
