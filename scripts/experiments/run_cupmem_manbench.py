#!/usr/bin/env python3
"""Run commit-pinned CUPMem on frozen ManBench-Return episodes.

The source run and CUPMem actor must use the same actor family for a
formal comparison.  The runner reuses only actor-visible task/evidence fields;
gold labels remain evaluator-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace.manbench_return import (  # noqa: E402
    _bounded_question,
    parse_choice_index,
)
from trace.providers import OpenAICompatibleEmbeddingClient  # noqa: E402
from trace.return_methods.semantic_state.cupmem import (  # noqa: E402
    CUPMEM_SYSTEM_ARM,
    CupMemEngine,
    CupMemQuery,
    CupMemSession,
    LockedLLMClient,
    RemoteEmbeddingRetriever,
    run_cupmem_episode,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_gzip_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _completed_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        return value.get("method") == CUPMEM_SYSTEM_ARM and bool(value.get("metrics"))
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return False


def _metric(value: bool) -> dict[str, int | float]:
    return {"numerator": int(value), "denominator": 1, "value": float(value)}


def _active_source_sessions(result: Mapping[str, object]) -> set[str]:
    deltas = result.get("delta_store")
    delta_sessions = {
        str(item.get("delta_id")): str(item.get("session_id"))
        for item in deltas or ()
        if isinstance(item, Mapping) and item.get("delta_id") and item.get("session_id")
    }
    snapshot = result.get("final_profile_snapshot")
    if not isinstance(snapshot, Mapping):
        return set()
    active = snapshot.get("active_profile")
    if not isinstance(active, Mapping):
        return set()
    source_sessions: set[str] = set()
    for rows in active.values():
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            source_delta_ids = item.get("source_delta_ids")
            if isinstance(source_delta_ids, list) and source_delta_ids:
                source = delta_sessions.get(str(source_delta_ids[-1]))
                if source:
                    source_sessions.add(source)
                    continue
            source = str(item.get("created_session_id") or "").strip()
            if source:
                source_sessions.add(source)
    return source_sessions


def _actor_visible_sessions(artifact: Mapping[str, object]) -> tuple[CupMemSession, ...]:
    baseline = artifact["baseline_reality"]
    current = artifact["current_example"]
    if not isinstance(baseline, Mapping) or not isinstance(current, Mapping):
        raise ValueError("frozen ManBench artifact has malformed actor state")
    departure = artifact["departure_example"]
    if not isinstance(departure, Mapping):
        raise ValueError("frozen ManBench artifact has no departure example")
    departure_question = _bounded_question(str(departure["question"]))
    current_question = _bounded_question(str(current["question"]))
    baseline_content = (
        "Returning agent's active task before departure:\n"
        + departure_question
        + "\n\nAgent1's own answer before departure:\n"
        + str(baseline["raw_output"])
    )
    evidence = artifact.get("active_agent_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("frozen ManBench artifact has no active-agent evidence")
    current_changed = (
        str(current.get("episode_id")) != str(departure.get("episode_id"))
    )
    absence_lines = []
    if current_changed:
        absence_lines.append(
            "While agent1 was absent, the scheduler changed the active task to:\n"
            + current_question
        )
    else:
        absence_lines.append(
            "While agent1 was absent, active agents discussed the same task."
        )
    for agent_id, text in sorted(evidence.items()):
        absence_lines.append(f"{agent_id}: {text}")
    return (
        CupMemSession(
            source_session_id="departure-answer",
            timestamp="logical-phase-0",
            messages=({"role": "user", "content": baseline_content},),
            phase="predeparture",
        ),
        CupMemSession(
            source_session_id="absence-update",
            timestamp="logical-phase-1",
            messages=({"role": "user", "content": "\n\n".join(absence_lines)},),
            phase="absence",
        ),
    )


def _eligible_source_paths(root: Path) -> tuple[Path, ...]:
    paths = []
    for path in sorted((root / "episodes").glob("*/*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if bool(value.get("return_evaluation_eligible")):
            paths.append(path)
    return tuple(paths)


def _build_engine(args: argparse.Namespace) -> CupMemEngine:
    actor_key = os.environ.get(args.actor_api_key_env, "") or "local"
    embedding_key = (
        os.environ.get(args.embedding_api_key_env, "")
        if args.embedding_api_key_env
        else None
    )
    shared_cache = args.shared_cache_dir.resolve()
    llm = LockedLLMClient(
        model=args.actor_model,
        api_key=actor_key,
        base_url=args.actor_base_url,
        cache_dir=shared_cache / "llm",
        request_lock_path=args.actor_request_lock_path,
    )
    embedding_client = OpenAICompatibleEmbeddingClient(
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        api_key=embedding_key,
        timeout=args.timeout,
        retries=args.retries,
    )
    retriever = RemoteEmbeddingRetriever(
        embedding_client,
        cache_dir=shared_cache / "embeddings",
        batch_size=args.embedding_batch_size,
        max_chars=args.embedding_max_chars,
    )
    return CupMemEngine(llm=llm, embedding=retriever)


def _iter_results(output_dir: Path) -> Iterable[Mapping[str, object]]:
    for path in sorted((output_dir / "results").glob("*/*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, Mapping):
            yield value


def _summary(output_dir: Path, expected: int) -> dict[str, object]:
    rows = tuple(_iter_results(output_dir))
    scenarios: dict[str, object] = {}
    for scenario in ("old_valid", "old_stale"):
        subset = [row for row in rows if row.get("scenario") == scenario]
        n = len(subset)
        metrics = [row["metrics"] for row in subset]
        scenarios[scenario] = {
            "records": n,
            "faa": sum(float(item["final_answer_accuracy"]["value"]) for item in metrics) / n if n else 0.0,
            "via": sum(float(item["valid_information_availability"]["value"]) for item in metrics) / n if n else 0.0,
            "iir": sum(float(item["invalid_information_rejection"]["value"]) for item in metrics) / n if n else 0.0,
            "wsar": sum(float(item["wrong_state_admission_rate"]["value"]) for item in metrics) / n if n else 0.0,
        }
    present = [value for value in scenarios.values() if value["records"]]
    macro = {
        key: sum(float(value[key]) for value in present) / len(present) if present else 0.0
        for key in ("faa", "via", "iir", "wsar")
    }
    summary = {
        "schema_version": "cupmem_full_manbench_summary_v1",
        "updated_at": _utc_now(),
        "method": CUPMEM_SYSTEM_ARM,
        "records": len(rows),
        "expected_eligible_records": expected,
        "complete": len(rows) == expected,
        "scenario_macro_average": macro,
        "scenarios": scenarios,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-cache-dir", type=Path, required=True)
    parser.add_argument("--actor-base-url", default=os.environ.get("TRACE_ACTOR_BASE_URL") or "https://dashscope-us.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--actor-model", default=os.environ.get("TRACE_ACTOR_MODEL") or "Qwen3.5-122B-A10B")
    parser.add_argument("--actor-api-key-env", default="TRACE_ACTOR_API_KEY")
    parser.add_argument("--embedding-base-url", default=os.environ.get("TRACE_EMBEDDING_BASE_URL") or "https://dashscope-us.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--embedding-model", default=os.environ.get("TRACE_EMBEDDING_MODEL") or "Qwen3-Embedding-8B")
    parser.add_argument("--embedding-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--actor-request-lock-path", default="/dev/shm/cupmem_full_qwen_actor.lock")
    parser.add_argument("--session-max-chars", type=int, default=28_000)
    parser.add_argument("--embedding-max-chars", type=int, default=12_000)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-bounded-question-contract",
        action="store_true",
        help=(
            "Recompute only source episodes already marked "
            "question_prompt_truncated, while resume skips all other results."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    source_root = args.source_run.resolve()
    all_paths = _eligible_source_paths(source_root)
    paths = tuple(
        path
        for index, path in enumerate(all_paths)
        if index % args.shard_count == args.shard_index
    )
    if args.limit is not None:
        paths = paths[: args.limit]
    output_dir = args.output_dir.resolve()
    engine = _build_engine(args)
    completed = 0
    failed = 0
    for position, source_path in enumerate(paths, start=1):
        relative = source_path.relative_to(source_root / "episodes")
        result_path = output_dir / "results" / relative.with_suffix(".json.gz")
        failure_path = output_dir / "failures" / relative
        artifact = json.loads(source_path.read_text(encoding="utf-8"))
        refresh_bounded = (
            args.refresh_bounded_question_contract
            and bool(artifact.get("question_prompt_truncated"))
        )
        if args.resume and _completed_result(result_path) and not refresh_bounded:
            completed += 1
            continue
        try:
            scenario = str(artifact["lifecycle_scenario"]["scenario"])
            current = artifact["current_example"]
            if not isinstance(current, Mapping):
                raise ValueError("current example is malformed")
            sessions = _actor_visible_sessions(artifact)
            result = run_cupmem_episode(
                engine=engine,
                episode_id=str(current["episode_id"]),
                benchmark="ManBench-Balanced-Return",
                sessions=sessions,
                queries=(
                    CupMemQuery(
                        label="final_query",
                        text=_bounded_question(str(current["question"])),
                    ),
                ),
                departure_source_session_id="departure-answer",
                max_user_chars_per_session=args.session_max_chars,
            )
            answer = str(result["answers"]["final_query"])
            choices = tuple(str(item) for item in current["choices"])
            choice_index = parse_choice_index(answer, len(choices))
            active_sessions = _active_source_sessions(result)
            valid_session = "s_000" if scenario == "old_valid" else "s_001"
            invalid_session = "s_001" if scenario == "old_valid" else "s_000"
            metrics = {
                "final_answer_accuracy": _metric(choice_index == int(current["gold_index"])),
                "valid_information_availability": _metric(valid_session in active_sessions),
                "invalid_information_rejection": _metric(invalid_session not in active_sessions),
                "wrong_state_admission_rate": _metric(invalid_session in active_sessions),
            }
            result.update(
                {
                    "created_at": _utc_now(),
                    "scenario": scenario,
                    "source_artifact_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "source_actor_generation_reused": True,
                    "source_gold_fields_visible_to_cupmem": False,
                    "choice_index": choice_index,
                    "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                    "active_source_sessions": sorted(active_sessions),
                    "metrics": metrics,
                    "evaluator": {
                        "gold_index": int(current["gold_index"]),
                        "valid_source_session": valid_session,
                        "invalid_source_session": invalid_session,
                    },
                }
            )
            _write_gzip_atomic(result_path, result)
            failure_path.unlink(missing_ok=True)
            completed += 1
            print(f"[{position}/{len(paths)}] completed {relative}", flush=True)
        except Exception as error:  # noqa: BLE001 - durable formal runner
            failed += 1
            _write_json_atomic(
                failure_path,
                {
                    "schema_version": "cupmem_full_failure_v1",
                    "updated_at": _utc_now(),
                    "source": str(relative),
                    "method": CUPMEM_SYSTEM_ARM,
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                },
            )
            print(
                f"[{position}/{len(paths)}] failed {relative}: "
                f"{type(error).__name__}: {str(error)[:300]}",
                flush=True,
            )
        _summary(output_dir, len(all_paths))
    summary = _summary(output_dir, len(all_paths))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if failed == 0 and completed == len(paths) else 2


if __name__ == "__main__":
    raise SystemExit(main())
