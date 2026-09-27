#!/usr/bin/env python3
"""Run the sharded full ManBench disputed-return panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from trace.manbench_return import (  # noqa: E402
    MANBENCH_RETURN_ARMS,
    MANBENCH_SUPPORTED_ARMS,
    build_manbench_balanced_scenarios,
    load_manbench_examples,
    manbench_dataset_sha256,
    run_manbench_return_episode,
)
from trace.manbench_provenance import (  # noqa: E402
    ProvenanceStressSpec,
    PROVENANCE_STRESS_SCENARIOS,
    PROVENANCE_STRESS_AGENT_COUNTS,
    stress_specs,
    validate_stress_metadata,
)
from trace.mas_return_pipeline import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MAX_CHARS,
)
from trace.memory_backends import (  # noqa: E402
    AgentScopedVersionedMemoryBackend,
)
from trace.configuration import resolve_provider_config
from trace.providers import (  # noqa: E402
    OpenAICompatibleEmbeddingClient,
)
from trace.unified_mas_contract import (  # noqa: E402
    ACTIVE_AGENT_IDS,
    task_agent_ids_for_active_count,
    validate_unified_mas_config,
)
import hashlib
import time
from trace.providers import OpenAICompatibleProvider  # noqa: E402 (already imported below for embedding)


def _sha256_path(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_provider(config, *, section_name):
    section = config[section_name]
    execution = config["execution"]
    api_key_env = section.get("api_key_env")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise RuntimeError(f"missing {section_name} API key env: {api_key_env}")
    return OpenAICompatibleProvider(
        generation_base_url=str(section["api_base"]),
        generation_model=str(section["served_model_id"]),
        generation_api_key=api_key,
        timeout=float(section.get("model_timeout_seconds", execution["model_timeout_seconds"])),
    )


def _healthy_models(provider, *, role):
    health = {}
    for attempt in range(5):
        health = provider.health()
        if health.get("ok") and provider.generation_model in health.get("models", ()):
            return health
        if attempt < 4:
            time.sleep(float(attempt + 1))
    if not health.get("ok"):
        raise RuntimeError(f"{role} endpoint health check failed: {health}")
    raise RuntimeError(f"configured {role} model is not served by the healthy endpoint")


DEFAULT_CONFIG = ROOT / "configs" / "variants/qwen35_122b_a10b/manbench_return_backend_qwen_full.json"


def _build_memory_embedding_provider(
    config: Mapping[str, Any],
) -> OpenAICompatibleEmbeddingClient | None:
    backend = config.get("memory_backend")
    if backend is None:
        return None
    if not isinstance(backend, Mapping):
        raise ValueError("memory_backend must be an object")
    if str(backend.get("type") or "asvm") != "asvm":
        raise ValueError("ManBench currently requires the ASVM backend")
    embedding = backend.get("embedding")
    if embedding is None:
        return None
    if not isinstance(embedding, Mapping):
        raise ValueError("memory_backend.embedding must be an object")
    env_name = str(embedding.get("api_key_env") or "").strip()
    api_key = os.environ.get(env_name) if env_name else None
    if env_name and not api_key:
        raise RuntimeError(
            f"missing memory embedding API key environment: {env_name}"
        )
    return OpenAICompatibleEmbeddingClient(
        base_url=str(embedding["api_base"]),
        model=str(embedding["served_model_id"]),
        api_key=api_key,
        timeout=float(embedding.get("timeout_seconds", 180)),
        retries=int(embedding.get("request_retries", 3)),
    )


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_progress(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class _ExecutionCase:
    lifecycle: Any
    provenance_stress: ProvenanceStressSpec | None = None

    @property
    def artifact_key(self) -> str | None:
        return (
            self.provenance_stress.case_id
            if self.provenance_stress is not None
            else None
        )

    @property
    def episode_id(self) -> str:
        base = self.lifecycle.episode_id
        return (
            self.provenance_stress.episode_id(base)
            if self.provenance_stress is not None
            else base
        )


def _artifact_path(
    output_dir: Path,
    task_name: str,
    source_index: int,
    *,
    artifact_key: str | None = None,
) -> Path:
    root = output_dir / "episodes"
    if artifact_key:
        root = root / "provenance_stress" / artifact_key
    return root / task_name / f"{source_index:04d}.json"


def _is_return_eligible(artifact: Mapping[str, Any]) -> bool:
    value = artifact.get("return_evaluation_eligible")
    if not isinstance(value, bool):
        raise ValueError("ManBench artifact lacks frozen eligibility status")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--episode-id", action="append", dest="episode_ids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--actor-model", help="Override model.served_model_id from config")
    parser.add_argument("--actor-base-url", help="Override model.api_base from config")
    parser.add_argument("--actor-api-key-env", help="Override model.api_key_env (pass empty string to clear)")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard contract")
    config = resolve_provider_config(json.loads(args.config.read_text(encoding="utf-8")))
    if args.actor_model or args.actor_base_url or args.actor_api_key_env is not None:
        _model = dict(config.get("model", {}))
        if args.actor_model:
            _model["served_model_id"] = args.actor_model
        if args.actor_base_url:
            _model["api_base"] = args.actor_base_url
        if args.actor_api_key_env is not None:
            _model["api_key_env"] = args.actor_api_key_env or None
        config = {**config, "model": _model}
    mas = config.get("mas")
    if not isinstance(mas, Mapping):
        raise ValueError("ManBench config requires a mas contract")
    validate_unified_mas_config(mas)
    dataset_root = ROOT / str(config["dataset"]["root"])
    dataset_sha256 = manbench_dataset_sha256(dataset_root)
    if dataset_sha256 != config["dataset"]["dataset_sha256"]:
        raise ValueError("ManBench dataset digest mismatch")
    examples = load_manbench_examples(dataset_root)
    expected_examples = int(config["dataset"]["expected_examples"])
    if len(examples) != expected_examples:
        raise ValueError(
            f"expected {expected_examples} examples, found {len(examples)}"
        )
    scenarios = build_manbench_balanced_scenarios(examples)
    stress_config = config.get("provenance_stress")
    stress_matrix: tuple[ProvenanceStressSpec | None, ...] = (None,)
    if stress_config is not None:
        if not isinstance(stress_config, Mapping):
            raise ValueError("provenance_stress must be an object")
        if int(stress_config.get("departing_agent_count", 1)) != 1:
            raise ValueError(
                "provenance_stress fixes departing_agent_count=1"
            )
        if stress_config.get("actor_visible", False) is not False:
            raise ValueError(
                "provenance_stress metadata must remain actor-hidden"
            )
        configured_scenarios = stress_config.get(
            "scenarios", PROVENANCE_STRESS_SCENARIOS
        )
        configured_counts = stress_config.get(
            "active_agent_counts", PROVENANCE_STRESS_AGENT_COUNTS
        )
        if isinstance(configured_scenarios, (str, bytes)) or not isinstance(
            configured_scenarios, list
        ):
            raise ValueError("provenance_stress.scenarios must be a list")
        if isinstance(configured_counts, (str, bytes)) or not isinstance(
            configured_counts, list
        ):
            raise ValueError(
                "provenance_stress.active_agent_counts must be a list"
            )
        stress_matrix = stress_specs(configured_scenarios, configured_counts)
    requested = set(args.episode_ids or ())
    valid_requested = {
        execution_id
        for lifecycle in scenarios
        for stress in stress_matrix
        for execution_id in (
            lifecycle.episode_id
            if stress is None
            else stress.episode_id(lifecycle.episode_id),
        )
    }
    unknown = requested - valid_requested
    if unknown:
        raise ValueError("unknown episode IDs: " + ", ".join(sorted(unknown)))
    selected: list[_ExecutionCase] = []
    for index, lifecycle in enumerate(scenarios):
        for stress in stress_matrix:
            case = _ExecutionCase(lifecycle, stress)
            if index % args.shard_count != args.shard_index:
                continue
            if requested and case.episode_id not in requested:
                continue
            selected.append(case)
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]

    provider = _build_provider(config, section_name="model")
    health = _healthy_models(provider, role="actor")
    execution = config["execution"]
    backend_config = config.get("memory_backend") or {}
    if not isinstance(backend_config, Mapping):
        raise ValueError("memory_backend must be an object")
    embedding_provider = _build_memory_embedding_provider(config)
    embedding_dimensions = int(
        backend_config.get(
            "embedding_dimensions",
            DEFAULT_EMBEDDING_DIMENSIONS,
        )
    )
    embedding_max_chars = int(
        backend_config.get(
            "embedding_max_chars",
            DEFAULT_EMBEDDING_MAX_CHARS,
        )
    )
    if embedding_dimensions < 1 or embedding_max_chars < 1:
        raise ValueError("ASVM embedding dimensions/chars must be positive")
    requested_arms = tuple(
        str(item) for item in execution.get("fork_arms", MANBENCH_RETURN_ARMS)
    )
    unknown_arms = set(requested_arms) - set(MANBENCH_SUPPORTED_ARMS)
    if unknown_arms:
        raise ValueError(
            "unknown ManBench Return arms: " + ",".join(sorted(unknown_arms))
        )
    if not requested_arms or len(requested_arms) != len(set(requested_arms)):
        raise ValueError("ManBench Return arms must be non-empty and unique")
    completed = 0
    resumed = 0
    failed = 0
    eligible = 0
    excluded = 0
    progress_path = (
        args.output_dir / "progress" / f"shard-{args.shard_index:02d}.json"
    )
    for position, execution_case in enumerate(selected, start=1):
        lifecycle = execution_case.lifecycle
        example = lifecycle.departure_example
        artifact_path = _artifact_path(
            args.output_dir,
            example.task_name,
            example.source_index,
            artifact_key=execution_case.artifact_key,
        )
        failure_root = args.output_dir / "failures"
        if execution_case.artifact_key:
            failure_root = failure_root / "provenance_stress" / execution_case.artifact_key
        failure_path = (
            failure_root
            / example.task_name
            / f"{example.source_index:04d}.json"
        )
        if artifact_path.exists():
            if not args.resume:
                raise FileExistsError(
                    f"episode exists without --resume: {example.episode_id}"
                )
            resumed += 1
            existing_artifact = json.loads(
                artifact_path.read_text(encoding="utf-8")
            )
            if execution_case.provenance_stress is not None:
                metadata = existing_artifact.get("provenance_stress")
                if not isinstance(metadata, Mapping):
                    raise ValueError(
                        "stress artifact lacks provenance metadata: "
                        + execution_case.episode_id
                    )
                validate_stress_metadata(
                    metadata,
                    expected_spec=execution_case.provenance_stress,
                )
            if _is_return_eligible(existing_artifact):
                eligible += 1
            else:
                excluded += 1
        else:
            try:
                call_cache_root = args.output_dir / "call_cache"
                if execution_case.artifact_key:
                    call_cache_root = (
                        call_cache_root / execution_case.artifact_key
                    )
                artifact = run_manbench_return_episode(
                    example=example,
                    scenario=lifecycle.scenario,
                    current_example=lifecycle.current_example,
                    provider=provider,
                    cache_dir=(
                        call_cache_root
                        / example.task_name
                        / f"{example.source_index:04d}"
                    ),
                    verifier_confidence_threshold=float(
                        execution["verifier_confidence_threshold"]
                    ),
                    format_retries=int(execution["format_retries"]),
                    baseline_max_tokens=int(
                        execution["baseline_max_tokens"]
                    ),
                    social_max_tokens=int(execution["social_max_tokens"]),
                    verifier_max_tokens=int(execution["verifier_max_tokens"]),
                    answer_max_tokens=int(execution["answer_max_tokens"]),
                    kmu_max_tokens=int(
                        execution.get("kmu_max_tokens", 512)
                    ),
                    cupmem_max_tokens=int(
                        execution.get("cupmem_max_tokens", 512)
                    ),
                    arms=requested_arms,
                    active_agent_ids=(
                        execution_case.provenance_stress.active_agent_ids
                        if execution_case.provenance_stress is not None
                        else ACTIVE_AGENT_IDS
                    ),
                    provenance_stress=execution_case.provenance_stress,
                    memory_backend=(
                        AgentScopedVersionedMemoryBackend.build(
                            example.episode_id,
                            worker_principal_ids=(
                                task_agent_ids_for_active_count(
                                    len(
                                        execution_case.provenance_stress.active_agent_ids
                                    )
                                )
                                if execution_case.provenance_stress is not None
                                else None
                            ),
                            embedding_provider=embedding_provider,
                            embedding_dimensions=embedding_dimensions,
                            embedding_max_chars=embedding_max_chars,
                        )
                    ),
                )
                artifact["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                artifact["dataset_sha256"] = dataset_sha256
                artifact["experiment_episode_id"] = execution_case.episode_id
                _write_json_once(artifact_path, artifact)
                failure_path.unlink(missing_ok=True)
                completed += 1
                if _is_return_eligible(artifact):
                    eligible += 1
                else:
                    excluded += 1
            except Exception as error:  # noqa: BLE001
                failed += 1
                _write_progress(
                    failure_path,
                    {
                        "schema_version": "manbench_return_failure_v1",
                        "episode_id": example.episode_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
        _write_progress(
            progress_path,
            {
                "schema_version": "manbench_balanced_return_progress_v3",
                "shard_count": args.shard_count,
                "shard_index": args.shard_index,
                "selected": len(selected),
                "processed": position,
                "completed": completed,
                "resumed": resumed,
                "failed": failed,
                "eligible_return_episodes": eligible,
                "excluded_return_episodes": excluded,
                "last_episode_id": example.episode_id,
                "last_scenario": lifecycle.scenario,
                "last_current_example_id": lifecycle.current_example.episode_id,
                "last_experiment_episode_id": execution_case.episode_id,
                "provenance_stress_case": (
                    execution_case.provenance_stress.record()
                    if execution_case.provenance_stress is not None
                    else None
                ),
            },
        )

    summary = {
        "schema_version": "manbench_balanced_return_batch_shard_v3",
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "dataset_sha256": dataset_sha256,
        "dataset_examples": len(examples),
        "balanced_scenarios": len(scenarios),
        "provenance_stress_cells": [
            stress.record() for stress in stress_matrix if stress is not None
        ],
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "selected": len(selected),
        "completed": completed,
        "resumed": resumed,
        "failed": failed,
        "eligible_return_episodes": eligible,
        "excluded_return_episodes": excluded,
        "model": provider.generation_model,
        "memory_backend": {
            "type": str(backend_config.get("type") or "asvm"),
            "semantic_backend": (
                "dense_embedding_cosine"
                if embedding_provider is not None
                else "token_cosine"
            ),
            "embedding": (
                embedding_provider.safe_metadata()
                if embedding_provider is not None
                else None
            ),
            "embedding_dimensions": embedding_dimensions,
            "embedding_max_chars": embedding_max_chars,
        },
        "evaluated_arms": list(requested_arms),
        "health": health,
    }
    summary_path = (
        args.output_dir / "shards" / f"shard-{args.shard_index:02d}.json"
    )
    if args.resume and summary_path.exists():
        summary_path.chmod(0o644)
        try:
            _write_progress(summary_path, summary)
        finally:
            summary_path.chmod(0o444)
    else:
        _write_json_once(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
