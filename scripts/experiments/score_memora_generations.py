#!/usr/bin/env python3
"""Score frozen Memora generations with an independent strict yes/no judge."""

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
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from trace.memora_return import (  # noqa: E402
    evidence_session_ids,
    forgetting_value_sessions,
    load_memora_manifest,
    load_memora_timeline,
)
from trace.mas_return_pipeline import (  # noqa: E402
    EpisodeLongTermMemory,
)
from trace.providers import (  # noqa: E402
    OpenAICompatibleProvider,
    endpoint_origin,
)
from trace.router_return_protocol import canonical_sha256  # noqa: E402
from trace.trace_method import (  # noqa: E402
    canonical_return_arm,
    canonicalize_arm_mapping,
)
from scripts.experiments.run_memora_mas_return import (  # noqa: E402
    DEFAULT_ARMS,
    DEFAULT_DATA_ROOT,
    _add_embedding_arguments,
    _build_embedding_provider,
    _csv,
    _generated_answer,
    _score_generated_arm,
    _source_contract,
    _validate_arms,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_file_name(episode_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", episode_id) + ".json"


def _load_and_validate_generation(
    generation_path: Path,
    *,
    episode_id: str,
    arms: Sequence[str],
) -> dict[str, object]:
    arms = tuple(canonical_return_arm(arm) for arm in arms)
    if not generation_path.is_file():
        raise ValueError(
            f"missing frozen generation artifact: {generation_path}"
        )
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if generation.get("schema_version") != "memora_mas_return_generation_v1":
        raise ValueError(
            f"unsupported generation artifact schema: {generation_path}"
        )
    expected = str(generation.get("generation_sha256") or "")
    unsigned = dict(generation)
    unsigned.pop("generation_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise ValueError(
            f"generation artifact digest mismatch: {generation_path}"
        )
    if generation.get("episode", {}).get("episode_id") != episode_id:
        raise ValueError(
            f"generation artifact episode mismatch: {generation_path}"
        )
    raw_generated_arms = generation.get("arms")
    if not isinstance(raw_generated_arms, Mapping):
        raise ValueError(
            f"generation artifact has no arms: {generation_path}"
        )
    generated_arms = canonicalize_arm_mapping(raw_generated_arms)
    for arm in arms:
        generated_arm = generated_arms.get(arm)
        if not isinstance(generated_arm, Mapping):
            raise ValueError(
                f"generation artifact is missing arm {arm}: "
                f"{generation_path}"
            )
        answer = _generated_answer(generated_arm)
        expected_answer = str(generated_arm.get("answer_sha256") or "")
        if (
            not answer
            or hashlib.sha256(answer.encode("utf-8")).hexdigest()
            != expected_answer
        ):
            raise ValueError(
                f"frozen answer digest mismatch for {arm}: "
                f"{generation_path}"
            )
    contract = generation.get("generation_contract")
    contract_digest = generation.get("generation_contract_sha256")
    if contract is not None or contract_digest is not None:
        if (
            not isinstance(contract, Mapping)
            or canonical_sha256(contract) != contract_digest
        ):
            raise ValueError(
                f"generation contract digest mismatch: {generation_path}"
            )
    elif not isinstance(generation.get("migration"), Mapping):
        raise ValueError(
            "generation contract is required for non-migrated artifacts: "
            f"{generation_path}"
        )
    normalized = dict(generation)
    normalized["arms"] = generated_arms
    return normalized


def _episode_memory_from_generation_record(
    value: Mapping[str, object],
    *,
    embedding_provider: Any = None,
) -> EpisodeLongTermMemory:
    """Restore the lifecycle view used by evaluator-only score bookkeeping.

    External native backends persist their semantic state separately and wrap
    the ordinary lifecycle/provenance record in ``lifecycle_shadow``.  Scoring
    never re-runs native retrieval or answer generation; it only needs this
    shadow to append the independent evaluator audit memory.
    """

    schema_version = str(value.get("schema_version") or "")
    lifecycle_record: Mapping[str, object] = value
    if schema_version == "external_episode_memory_v1":
        if value.get("governance_backend_separation") is not True:
            raise ValueError(
                "external episode memory must declare backend separation"
            )
        raw_shadow = value.get("lifecycle_shadow")
        if not isinstance(raw_shadow, Mapping):
            raise ValueError(
                "external episode memory requires a lifecycle_shadow"
            )
        outer_episode_id = str(value.get("episode_id") or "")
        shadow_episode_id = str(raw_shadow.get("episode_id") or "")
        if not outer_episode_id or outer_episode_id != shadow_episode_id:
            raise ValueError(
                "external episode memory lifecycle_shadow episode mismatch"
            )
        lifecycle_record = raw_shadow
    return EpisodeLongTermMemory.from_record(
        lifecycle_record,
        embedding_provider=embedding_provider,
    )


def _ratio(left: set[int], right: set[int]) -> float | None:
    return len(left & right) / len(right) if right else None


def _precision(left: set[int], right: set[int]) -> float | None:
    return len(left & right) / len(left) if left else None


def _prediction_quality(
    question: Any,
    generation: Mapping[str, object],
) -> dict[str, object] | None:
    bundle = generation.get("fork_bundle")
    if not isinstance(bundle, Mapping):
        return None
    trace = bundle.get("trace")
    if not isinstance(trace, Mapping):
        return None
    batches = trace.get("extraction_batches")
    dependency = trace.get("dependency_plan")
    if not isinstance(batches, list) or not isinstance(dependency, Mapping):
        return None
    facts: dict[str, Mapping[str, object]] = {}
    for batch in batches:
        if not isinstance(batch, Mapping):
            continue
        for fact in batch.get("facts") or ():
            if isinstance(fact, Mapping) and fact.get("fact_id"):
                facts[str(fact["fact_id"])] = fact
    extracted = {int(item["session_id"]) for item in facts.values()}
    selected = {
        int(facts[fact_id]["session_id"])
        for fact_id in dependency.get("selected_fact_ids") or ()
        if str(fact_id) in facts
    }
    mutations = {
        int(item["session_id"])
        for item in facts.values()
        if item.get("operation") in {"update", "delete"}
    }
    memory = set(evidence_session_ids(question.memory_evidence))
    forgetting = {
        session_id
        for _, session_id in forgetting_value_sessions(
            question.forgetting_evidence
        )
    }
    relevant = memory | forgetting
    return {
        "schema_version": "memora_predicted_evaluator_metrics_v1",
        "computed_only_in_score_phase": True,
        "consumed_by_router_workers_or_aggregator": False,
        "official_memory_session_count": len(memory),
        "official_forgetting_mutation_session_count": len(forgetting),
        "predicted_fact_session_count": len(extracted),
        "selected_fact_session_count": len(selected),
        "fact_extraction_precision": _precision(extracted, relevant),
        "fact_extraction_recall": _ratio(extracted, relevant),
        "dependency_session_precision": _precision(selected, memory),
        "dependency_session_recall": _ratio(selected, memory),
        "update_delete_session_precision": _precision(
            mutations, forgetting
        ),
        "update_delete_session_recall": _ratio(mutations, forgetting),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--judge-api-key-env", default="OPENAI_API_KEY"
    )
    parser.add_argument("--arms", type=_csv, default=DEFAULT_ARMS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    _add_embedding_arguments(parser)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = _validate_arms(args.arms)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard contract")
    judge_key = os.environ.get(args.judge_api_key_env)
    if not judge_key:
        raise RuntimeError(
            f"missing judge API key env: {args.judge_api_key_env}"
        )
    judge = OpenAICompatibleProvider(
        generation_base_url=args.judge_base_url,
        generation_model=args.judge_model,
        generation_api_key=judge_key,
        timeout=args.timeout,
        retries=args.retries,
    )
    embedding_provider = _build_embedding_provider(args)
    episodes = load_memora_manifest(args.manifest.resolve())
    if args.limit is not None:
        episodes = episodes[: args.limit]
    input_dir = args.input_dir.resolve()
    generation_dir = (
        input_dir / "generations"
        if (input_dir / "generations").is_dir()
        else input_dir
    )
    frozen_generations: dict[str, dict[str, object]] = {}
    contract_digests: set[str] = set()
    for episode in episodes:
        generation_path = generation_dir / _artifact_file_name(
            episode.episode_id
        )
        generation = _load_and_validate_generation(
            generation_path,
            episode_id=episode.episode_id,
            arms=arms,
        )
        frozen_generations[episode.episode_id] = generation
        digest = generation.get("generation_contract_sha256")
        if digest:
            contract_digests.add(str(digest))
    if len(contract_digests) > 1:
        raise ValueError(
            "generation artifacts do not share one frozen generation contract"
        )
    episodes = tuple(
        episode
        for index, episode in enumerate(episodes)
        if index % args.shard_count == args.shard_index
    )
    output_dir = (args.output_dir or args.input_dir).resolve()
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    timeline_cache: dict[tuple[str, str], Any] = {}
    source_contract = {
        **_source_contract(),
        str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    scoring_contract = {
        "schema_version": "memora_scoring_contract_v1",
        "judge_model": args.judge_model,
        "judge_endpoint_origin": endpoint_origin(args.judge_base_url),
        "arms": list(arms),
        "seed": args.seed,
        "judge_max_tokens": args.judge_max_tokens,
        "strict_yes_no_contract": True,
        "non_binary_policy": "reject_and_retry_without_label_coercion",
        "long_term_memory": {
            "semantic_backend": (
                "dense_embedding_cosine"
                if embedding_provider is not None
                else "token_cosine"
            ),
            "embedding_model": args.embedding_model,
            "embedding_dimensions": (
                args.embedding_dimensions
                if embedding_provider is not None
                else None
            ),
            "embedding_endpoint_origin": (
                endpoint_origin(args.embedding_base_url)
                if embedding_provider is not None
                else None
            ),
        },
    }
    scoring_contract_sha256 = canonical_sha256(scoring_contract)
    completed = 0
    failed = 0
    started = time.perf_counter()
    for index, episode in enumerate(episodes):
        file_name = _artifact_file_name(episode.episode_id)
        generation_path = generation_dir / file_name
        output_path = episode_dir / file_name
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
            generation = frozen_generations[episode.episode_id]
            expected = str(generation.get("generation_sha256") or "")
            generated_arms = generation.get("arms")
            raw_episode_memory = generation.get("long_term_memory")
            episode_memory = (
                _episode_memory_from_generation_record(
                    raw_episode_memory,
                    embedding_provider=embedding_provider,
                )
                if isinstance(raw_episode_memory, Mapping)
                else None
            )
            scored_arms: dict[str, object] = {}
            for arm in arms:
                memory_arm = (
                    episode_memory.branches.get(arm)
                    if episode_memory is not None
                    else None
                )
                scored_arms[arm] = _score_generated_arm(
                    episode=episode,
                    question=question,
                    generated_arm=generated_arms[arm],
                    judge_provider=judge,
                    seed=args.seed,
                    judge_max_tokens=args.judge_max_tokens,
                    long_term_memory=episode_memory,
                    evaluator_memory=(
                        memory_arm.evaluator
                        if memory_arm is not None
                        else None
                    ),
                )
            record: dict[str, object] = {
                "schema_version": "memora_mas_return_run_v3",
                "state_mode": generation["state_mode"],
                "episode": generation["episode"],
                "question_actor_record": generation[
                    "question_actor_record"
                ],
                "fork_bundle": generation["fork_bundle"],
                "selected_arms": list(arms),
                "generation_contract": generation.get(
                    "generation_contract"
                ),
                "generation_contract_sha256": generation.get(
                    "generation_contract_sha256"
                ),
                "scoring_contract": scoring_contract,
                "scoring_contract_sha256": scoring_contract_sha256,
                "generation_artifact": {
                    "path": str(generation_path),
                    "file_sha256": hashlib.sha256(
                        generation_path.read_bytes()
                    ).hexdigest(),
                    "generation_sha256": expected,
                    "answers_regenerated_during_scoring": False,
                },
                "prediction_quality_evaluator_only": _prediction_quality(
                    question, generation
                ),
                "source_contract": source_contract,
                "source_contract_sha256": canonical_sha256(source_contract),
                "created_at_utc": _utc_now(),
                "arms": scored_arms,
                "provider": {
                    "actor": generation["provider"],
                    "judge": {
                        "origin": endpoint_origin(args.judge_base_url),
                        "model": args.judge_model,
                        "api_key_persisted": False,
                    },
                },
            }
            if episode_memory is not None:
                record["long_term_memory"] = episode_memory.record(
                    include_private_experience=True
                )
            record["record_sha256"] = canonical_sha256(record)
            output_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            failure = output_path.with_suffix(".failure.json")
            if failure.is_file():
                failure.unlink()
            completed += 1
        except Exception as error:
            failed += 1
            output_path.with_suffix(".failure.json").write_text(
                json.dumps(
                    {
                        "schema_version": "memora_score_failure_v1",
                        "episode_id": episode.episode_id,
                        "generation_path": str(generation_path),
                        "generation_preserved": generation_path.is_file(),
                        "answers_regenerated": False,
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
                    "phase": "score",
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
        "schema_version": "memora_score_campaign_v1",
        "phase": "score",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "generation_dir": str(generation_dir),
        "requested_episodes": len(episodes),
        "completed": completed,
        "failed": failed,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "arms": list(arms),
        "scoring_contract": scoring_contract,
        "scoring_contract_sha256": scoring_contract_sha256,
        "judge": {
            "origin": endpoint_origin(args.judge_base_url),
            "model": args.judge_model,
            "strict_yes_no_contract": True,
            "non_binary_labels_coerced": False,
            "api_key_persisted": False,
        },
        "source_contract": source_contract,
        "source_contract_sha256": canonical_sha256(source_contract),
        "elapsed_seconds": time.perf_counter() - started,
        "created_at_utc": _utc_now(),
    }
    receipt_name = (
        "score_campaign.json"
        if args.shard_count == 1
        else (
            f"score_campaign.shard_{args.shard_index:02d}_of_"
            f"{args.shard_count:02d}.json"
        )
    )
    (output_dir / receipt_name).write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
