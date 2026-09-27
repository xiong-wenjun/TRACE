#!/usr/bin/env python3
"""Generate and freeze Memora RETURN answers without running an evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from trace.memora_return import (  # noqa: E402
    build_memora_return_forks,
    load_memora_manifest,
    load_memora_timeline,
    memora_unified_mas_episode_record,
)
from trace.mas_return_pipeline import (  # noqa: E402
    EpisodeLongTermMemory,
    unified_five_task_agent_architecture_record,
)
from trace.providers import (  # noqa: E402
    OpenAICompatibleProvider,
    endpoint_origin,
)
from trace.return_governance import ReturnMemoryGovernor  # noqa: E402
from trace.router_return_protocol import (  # noqa: E402
    RouterReturnArm,
    canonical_sha256,
)
from scripts.experiments.run_memora_mas_return import (  # noqa: E402
    DEFAULT_ARMS,
    DEFAULT_DATA_ROOT,
    _add_embedding_arguments,
    _attach_lifecycle_baselines,
    _build_predicted_trace,
    _build_embedding_provider,
    _csv,
    _generate_arm,
    _fork_episode_memory_arms,
    _safe_episode_record,
    _seed_oracle_episode_memory,
    _source_contract,
    _validate_arms,
)
from trace.memora_predicted import (  # noqa: E402
    build_predicted_return_forks,
)
from trace.memory_backends import (  # noqa: E402
    add_memory_backend_arguments,
    build_memory_backend_factory,
)
from trace.trace_method import internal_return_arm  # noqa: E402
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generation_contract(
    args: argparse.Namespace,
    *,
    arms: Sequence[str],
) -> dict[str, object]:
    """Return every semantic generation setting needed to reproduce an answer."""
    embedding_model = getattr(args, "embedding_model", None)
    embedding_base_url = getattr(args, "embedding_base_url", None)
    embedding_dimensions = getattr(
        args,
        "embedding_dimensions",
        None,
    )
    return {
        "schema_version": "memora_generation_contract_v2",
        "mas_architecture": unified_five_task_agent_architecture_record(),
        "model": args.model,
        "endpoint_origin": endpoint_origin(args.base_url),
        "state_mode": args.state_mode,
        "arms": list(arms),
        "seed": args.seed,
        "actor_max_tokens": args.actor_max_tokens,
        "actor_thinking_mode": getattr(
            args,
            "actor_thinking_mode",
            "default",
        ),
        "actor_omit_temperature": bool(
            getattr(args, "actor_omit_temperature", False)
        ),
        "empty_content_retries": getattr(
            args,
            "empty_content_retries",
            0,
        ),
        "cupmem_max_tokens": getattr(args, "cupmem_max_tokens", 4096),
        "kmu_max_tokens": getattr(args, "kmu_max_tokens", 4096),
        "max_actor_view_chars": args.max_actor_view_chars,
        "long_term_memory": {
            "backend": getattr(args, "memory_backend", "asvm-vector"),
            "amem": ({"evo_threshold": args.amem_evo_threshold,
                      "max_tokens": args.amem_max_tokens, "temperature": 0.0}
                     if getattr(args, "memory_backend", "asvm-vector") == "amem" else None),
            "native_embedding_dimensions": getattr(
                args,
                "memory_backend_embedding_dimensions",
                None,
            ),
            "scope": "one_longitudinal_episode",
            "principal_isolation": True,
            "return_branching": "same_quarantined_snapshot_per_arm",
            "semantic_backend": {
                "mem0": "mem0_native_vector_search",
                "memobase": "memobase_native_profile_search",
                "amem": "amem_native_agentic_search",
            }.get(
                getattr(args, "memory_backend", "asvm-vector"),
                (
                    "dense_embedding_cosine"
                    if embedding_model
                    else "token_cosine"
                ),
            ),
            "embedding_model": embedding_model,
            "embedding_dimensions": (
                embedding_dimensions
                if embedding_model
                else None
            ),
            "embedding_endpoint_origin": (
                endpoint_origin(embedding_base_url)
                if embedding_base_url
                else None
            ),
            "utility_learning_enabled": False,
        },
        "predicted": {
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
            "minimum_confidence": args.predicted_minimum_confidence,
            "base_selected_facts": getattr(
                args,
                "predicted_base_selected_facts",
                12,
            ),
            "max_selected_facts": args.predicted_max_selected_facts,
            "selector_budget_policy": "dynamic_obligation_coverage_v1",
            "max_tokens": args.predicted_max_tokens,
            "structured_output_mode": getattr(
                args,
                "predicted_structured_output_mode",
                "auto",
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--api-key-env", default="TRACE_ACTOR_API_KEY"
    )
    _add_embedding_arguments(parser)
    add_memory_backend_arguments(parser)
    parser.add_argument("--arms", type=_csv, default=DEFAULT_ARMS)
    parser.add_argument(
        "--state-mode",
        choices=("oracle", "predicted"),
        default="predicted",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--actor-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--actor-thinking-mode",
        choices=("default", "adaptive", "disabled"),
        default="default",
        help=(
            "Explicit Claude-compatible thinking policy; default omits the "
            "request field."
        ),
    )
    parser.add_argument(
        "--actor-omit-temperature",
        action="store_true",
        help="Omit deprecated temperature from Sonnet-compatible requests.",
    )
    parser.add_argument("--cupmem-max-tokens", type=int, default=4096)
    parser.add_argument("--kmu-max-tokens", type=int, default=4096)
    parser.add_argument("--max-actor-view-chars", type=int, default=16_000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--empty-content-retries",
        type=int,
        default=2,
        help=(
            "Retry valid HTTP responses whose stop/length envelope contains "
            "neither visible text nor a required tool call."
        ),
    )
    parser.add_argument("--resume", action="store_true")
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
    parser.add_argument("--predicted-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--predicted-structured-output-mode",
        choices=("auto", "forced_tool", "prompt_json"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = _validate_arms(args.arms)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard contract")
    if not 0.0 < args.predicted_departure_fraction < 1.0:
        raise ValueError("predicted departure fraction must be in (0, 1)")
    if args.empty_content_retries < 0:
        raise ValueError("empty content retries must be non-negative")
    if args.cupmem_max_tokens < 1 or args.kmu_max_tokens < 1:
        raise ValueError("lifecycle baseline token budgets must be positive")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env: {args.api_key_env}")
    provider = OpenAICompatibleProvider(
        generation_base_url=args.base_url,
        generation_model=args.model,
        generation_api_key=api_key,
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
    embedding_provider = _build_embedding_provider(args)
    backend_factory = build_memory_backend_factory(
        args,
        generation_api_key=api_key,
        embedding_provider=embedding_provider,
        default_storage_dir=(
            args.output_dir.resolve() / "native_backend_state"
        ),
    )
    episodes = load_memora_manifest(args.manifest.resolve())
    if args.limit is not None:
        episodes = episodes[: args.limit]
    episodes = tuple(
        episode
        for index, episode in enumerate(episodes)
        if index % args.shard_count == args.shard_index
    )
    generation_dir = args.output_dir.resolve() / "generations"
    generation_dir.mkdir(parents=True, exist_ok=True)
    timeline_cache: dict[tuple[str, str], Any] = {}
    source_contract = {
        **_source_contract(),
        str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    generation_contract = _generation_contract(
        args,
        arms=arms,
    )
    generation_contract_sha256 = canonical_sha256(generation_contract)
    completed = 0
    failed = 0
    started = time.perf_counter()
    for index, episode in enumerate(episodes):
        path = generation_dir / (
            re.sub(r"[^a-zA-Z0-9._-]+", "_", episode.episode_id) + ".json"
        )
        if args.resume and path.is_file():
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
        stage = "initialize_episode"
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
            prediction_calls: tuple[dict[str, object], ...] = ()
            episode_memory = backend_factory.build_episode(
                episode.episode_id
            )
            if args.state_mode == "predicted":
                stage = "predicted_trace"
                trace, prediction_calls = _build_predicted_trace(
                    episode=episode,
                    timeline=timeline,
                    question=question,
                    provider=provider,
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
                    ablation_arms=arms,
                )
                episode_record = _safe_episode_record(
                    episode,
                    departure_session_id=trace.departure_session_id,
                    return_session_id=trace.return_session_id,
                    departure_fraction=trace.departure_fraction,
                )
            else:
                stage = "oracle_return_forks"
                bundle = build_memora_return_forks(
                    timeline, episode, governor=governor
                )
                _seed_oracle_episode_memory(
                    episode_memory,
                    bundle=bundle,
                    question=question,
                )
                episode_record = episode.record()
            stage = "lifecycle_baselines"
            bundle, lifecycle_baseline_calls = _attach_lifecycle_baselines(
                bundle,
                arms=arms,
                provider=provider,
                seed=args.seed,
                cupmem_max_tokens=args.cupmem_max_tokens,
                kmu_max_tokens=args.kmu_max_tokens,
            )
            memory_arms = _fork_episode_memory_arms(
                episode_memory,
                bundle=bundle,
                arms=arms,
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
            generated_arms: dict[str, object] = {}
            for arm in arms:
                stage = f"generate_arm:{arm}"
                actor_view = bundle.actor_views[internal_return_arm(arm)]
                generated_arm = _generate_arm(
                    episode=episode,
                    question=question,
                    arm=arm,
                    actor_view=actor_view,
                    actor_provider=provider,
                    seed=args.seed,
                    actor_max_tokens=args.actor_max_tokens,
                    max_actor_view_chars=args.max_actor_view_chars,
                    long_term_memory=episode_memory,
                    memory_arm=memory_arms[arm],
                )
                generated_arms[arm] = generated_arm
            record: dict[str, object] = {
                "schema_version": "memora_mas_return_generation_v1",
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
                "prediction_calls": list(prediction_calls),
                "long_term_memory": episode_memory.record(
                    include_private_experience=True
                ),
                "generation_contract": generation_contract,
                "generation_contract_sha256": generation_contract_sha256,
                "memory_backend": backend_factory.record(),
                "source_contract": source_contract,
                "source_contract_sha256": canonical_sha256(source_contract),
                "created_at_utc": _utc_now(),
                "arms": generated_arms,
                "provider": {
                    "origin": endpoint_origin(args.base_url),
                    "model": args.model,
                    "api_key_persisted": False,
                },
            }
            if lifecycle_baseline_calls:
                record["lifecycle_baseline_calls"] = list(
                    lifecycle_baseline_calls
                )
            record["generation_sha256"] = canonical_sha256(record)
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            failure = path.with_suffix(".failure.json")
            if failure.is_file():
                failure.unlink()
            completed += 1
        except Exception as error:
            failed += 1
            path.with_suffix(".failure.json").write_text(
                json.dumps(
                    {
                        "schema_version": "memora_generation_failure_v1",
                        "episode_id": episode.episode_id,
                        "stage": stage,
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
                    "phase": "generate",
                    "progress": f"{index + 1}/{len(episodes)}",
                    "completed": completed,
                    "failed": failed,
                    "episode_id": episode.episode_id,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    receipt = {
        "schema_version": "memora_generation_campaign_v1",
        "phase": "generate",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "requested_episodes": len(episodes),
        "completed": completed,
        "failed": failed,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "arms": list(arms),
        "state_mode": args.state_mode,
        "generation_contract": generation_contract,
        "generation_contract_sha256": generation_contract_sha256,
        "memory_backend": backend_factory.record(),
        "provider": {
            "origin": endpoint_origin(args.base_url),
            "model": args.model,
            "api_key_persisted": False,
        },
        "source_contract": source_contract,
        "source_contract_sha256": canonical_sha256(source_contract),
        "elapsed_seconds": time.perf_counter() - started,
        "created_at_utc": _utc_now(),
    }
    receipt_name = (
        "generation_campaign.json"
        if args.shard_count == 1
        else (
            f"generation_campaign.shard_{args.shard_index:02d}_of_"
            f"{args.shard_count:02d}.json"
        )
    )
    (args.output_dir.resolve() / receipt_name).write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
