#!/usr/bin/env python3
"""Run commit-pinned CUPMem on Memora MAS Return episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from trace.memora_return import (  # noqa: E402
    load_memora_manifest,
    load_memora_timeline,
    score_memora_fama,
)
from trace.providers import (  # noqa: E402
    OpenAICompatibleEmbeddingClient,
    OpenAICompatibleProvider,
)
from trace.return_methods.semantic_state.cupmem import (  # noqa: E402
    CUPMEM_SYSTEM_ARM,
    CupMemEngine,
    CupMemQuery,
    CupMemSession,
    LockedLLMClient,
    RemoteEmbeddingRetriever,
    build_cupmem_full_timeline_checkpoint,
    cupmem_full_timeline_checkpoint_lock,
    cupmem_full_timeline_fingerprint,
    load_cupmem_full_timeline_checkpoint,
    run_cupmem_full_query_from_checkpoint,
    save_cupmem_full_timeline_checkpoint,
)
from scripts.experiments.run_memora_mas_return import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    _batch_judge,
    _completion_record,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


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
        return value.get("method") == CUPMEM_SYSTEM_ARM and bool(value.get("score"))
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return False


def _build_engine(args: argparse.Namespace) -> tuple[CupMemEngine, OpenAICompatibleProvider]:
    actor_key = os.environ.get(args.actor_api_key_env, "") or "local"
    judge_key = os.environ.get(args.judge_api_key_env, "")
    embedding_key = (
        os.environ.get(args.embedding_api_key_env, "")
        if args.embedding_api_key_env
        else None
    )
    if not judge_key:
        raise RuntimeError(f"missing Judge API key: {args.judge_api_key_env}")
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
    judge = OpenAICompatibleProvider(
        generation_base_url=args.judge_base_url,
        generation_model=args.judge_model,
        generation_api_key=judge_key,
        timeout=args.timeout,
        retries=args.retries,
        chat_response_format={"type": "json_object"},
        request_lock_path=args.judge_request_lock_path,
    )
    return CupMemEngine(llm=llm, embedding=retriever), judge


def _summary(output_dir: Path, expected: int) -> dict[str, object]:
    scores: list[Mapping[str, object]] = []
    for path in sorted((output_dir / "results").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        score = value.get("score")
        if isinstance(score, Mapping):
            scores.append(score)
    n = len(scores)
    total = sum(int(item["total_evaluations"]) for item in scores)
    correct = sum(int(item["correct_evaluations"]) for item in scores)
    memory_total = sum(int(item["memory_presence_total"]) for item in scores)
    memory_correct = sum(int(item["memory_presence_correct"]) for item in scores)
    stale_total = sum(int(item["forgetting_absence_total"]) for item in scores)
    stale_correct = sum(int(item["forgetting_absence_correct"]) for item in scores)
    summary = {
        "schema_version": "cupmem_full_memora_summary_v1",
        "updated_at": _utc_now(),
        "method": CUPMEM_SYSTEM_ARM,
        "records": n,
        "expected_records": expected,
        "complete": n == expected,
        "overall_accuracy": correct / total if total else 0.0,
        "via_accuracy": memory_correct / memory_total if memory_total else 0.0,
        "iir_accuracy": stale_correct / stale_total if stale_total else 0.0,
        "fama_macro": sum(float(item["fama"]) for item in scores) / n if n else 0.0,
        "counts": {
            "rubric_correct": correct,
            "rubric_total": total,
            "valid_current_correct": memory_correct,
            "valid_current_total": memory_total,
            "stale_rejection_correct": stale_correct,
            "stale_rejection_total": stale_total,
        },
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--timeline-checkpoint-dir",
        type=Path,
        help="Query-independent CUPMem states; defaults below shared-cache-dir.",
    )
    parser.add_argument("--departure-fraction", type=float, required=True)
    parser.add_argument("--actor-base-url", default=os.environ.get("TRACE_ACTOR_BASE_URL") or "https://dashscope-us.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--actor-model", default=os.environ.get("TRACE_ACTOR_MODEL") or "Qwen3.5-122B-A10B")
    parser.add_argument("--actor-api-key-env", default="TRACE_ACTOR_API_KEY")
    parser.add_argument("--judge-base-url", default=os.environ.get("TRACE_JUDGE_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--judge-model", default=os.environ.get("TRACE_JUDGE_MODEL") or "gpt-5.6-terra")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-base-url", default=os.environ.get("TRACE_EMBEDDING_BASE_URL") or "https://dashscope-us.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--embedding-model", default=os.environ.get("TRACE_EMBEDDING_MODEL") or "Qwen3-Embedding-8B")
    parser.add_argument("--embedding-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--actor-request-lock-path", default="/dev/shm/cupmem_full_qwen_actor.lock")
    parser.add_argument("--judge-request-lock-path", default="/dev/shm/cupmem_full_terra_judge.lock")
    parser.add_argument("--session-max-chars", type=int, default=28_000)
    parser.add_argument("--embedding-max-chars", type=int, default=12_000)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.departure_fraction < 1.0:
        raise ValueError("departure fraction must be in (0, 1)")
    episodes = load_memora_manifest(args.manifest.resolve())
    if args.limit is not None:
        episodes = episodes[: args.limit]
    output_dir = args.output_dir.resolve()
    engine, judge = _build_engine(args)
    timeline_cache: dict[tuple[str, str], object] = {}
    checkpoint_cache: dict[str, dict[str, object]] = {}
    checkpoint_dir = (
        args.timeline_checkpoint_dir.resolve()
        if args.timeline_checkpoint_dir is not None
        else args.shared_cache_dir.resolve() / "timeline_checkpoints_v1"
    )
    completed = 0
    failed = 0
    for position, episode in enumerate(episodes, start=1):
        path = output_dir / "results" / f"{_safe_name(episode.episode_id)}.json.gz"
        failure_path = output_dir / "failures" / f"{_safe_name(episode.episode_id)}.json"
        if args.resume and _completed_result(path):
            completed += 1
            continue
        try:
            key = (episode.period, episode.persona)
            if key not in timeline_cache:
                timeline_cache[key] = load_memora_timeline(
                    args.data_root.resolve(),
                    period=episode.period,
                    persona=episode.persona,
                )
            timeline = timeline_cache[key]
            ordered = tuple(sorted(timeline.sessions, key=lambda item: item.session_id))
            split = min(
                len(ordered) - 2,
                max(0, int(len(ordered) * args.departure_fraction) - 1),
            )
            departure_id = ordered[split].session_id
            question = timeline.by_question_id[episode.question_id]
            sessions = tuple(
                CupMemSession(
                    source_session_id=f"session:{session.session_id}",
                    timestamp=session.date,
                    messages=tuple(
                        {
                            "role": (
                                "user"
                                if "user" in turn.speaker.casefold()
                                else "assistant"
                            ),
                            "content": turn.message,
                        }
                        for turn in session.conversation
                    ),
                    phase=(
                        "predeparture"
                        if session.session_id <= departure_id
                        else "absence"
                    ),
                )
                for session in ordered
            )
            fingerprint = cupmem_full_timeline_fingerprint(
                engine=engine,
                benchmark="Memora-MAS-Return",
                sessions=sessions,
                departure_source_session_id=f"session:{departure_id}",
                max_user_chars_per_session=args.session_max_chars,
            )
            checkpoint_path = checkpoint_dir / f"{fingerprint}.json.gz"
            checkpoint = checkpoint_cache.get(fingerprint)
            checkpoint_reused = checkpoint is not None
            if checkpoint is None:
                with cupmem_full_timeline_checkpoint_lock(checkpoint_path):
                    checkpoint = load_cupmem_full_timeline_checkpoint(
                        checkpoint_path,
                        expected_fingerprint=fingerprint,
                    )
                    checkpoint_reused = checkpoint is not None
                    if checkpoint is None:
                        checkpoint = build_cupmem_full_timeline_checkpoint(
                            engine=engine,
                            benchmark="Memora-MAS-Return",
                            sessions=sessions,
                            departure_source_session_id=f"session:{departure_id}",
                            max_user_chars_per_session=args.session_max_chars,
                        )
                        if checkpoint.get("fingerprint") != fingerprint:
                            raise RuntimeError("CUPMem checkpoint fingerprint drift")
                        save_cupmem_full_timeline_checkpoint(
                            checkpoint_path,
                            checkpoint,
                        )
                checkpoint_cache[fingerprint] = checkpoint
                # Session traces can be large. Keep only the current timeline
                # in memory; older states remain available in compressed form.
                if len(checkpoint_cache) > 1:
                    checkpoint_cache = {fingerprint: checkpoint}
            result = run_cupmem_full_query_from_checkpoint(
                engine=engine,
                checkpoint=checkpoint,
                episode_id=episode.episode_id,
                query=CupMemQuery(label="final_query", text=question.question),
                checkpoint_reused=checkpoint_reused,
            )
            answer = str(result["answers"]["final_query"])
            judgments, completions = _batch_judge(
                judge,
                question=question,
                answer=answer,
                seed=args.seed,
                max_tokens=args.judge_max_tokens,
            )
            score = score_memora_fama(question, judgments)
            result.update(
                {
                    "created_at": _utc_now(),
                    "departure_fraction": args.departure_fraction,
                    "departure_session_id": departure_id,
                    "return_session_id": ordered[-1].session_id,
                    "selection_policy": "fixed_timeline_session_fraction",
                    "shared_memory_continuous_during_absence": True,
                    "question_actor_record": question.actor_record(),
                    "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                    "score": score.record(),
                    "judge": {
                        "completions": [_completion_record(item) for item in completions],
                        "rubric_item_count": len(judgments),
                        "ground_truth_visible_to_evaluator_only": True,
                    },
                }
            )
            _write_gzip_atomic(path, result)
            failure_path.unlink(missing_ok=True)
            completed += 1
            print(f"[{position}/{len(episodes)}] completed {episode.episode_id}", flush=True)
        except Exception as error:  # noqa: BLE001 - durable formal runner
            failed += 1
            _write_json_atomic(
                failure_path,
                {
                    "schema_version": "cupmem_full_failure_v1",
                    "updated_at": _utc_now(),
                    "episode_id": episode.episode_id,
                    "method": CUPMEM_SYSTEM_ARM,
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                },
            )
            print(
                f"[{position}/{len(episodes)}] failed {episode.episode_id}: "
                f"{type(error).__name__}: {str(error)[:300]}",
                flush=True,
            )
        _summary(output_dir, len(episodes))
    summary = _summary(output_dir, len(episodes))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if failed == 0 and completed == len(episodes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
