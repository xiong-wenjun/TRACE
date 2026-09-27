#!/usr/bin/env python3
"""Run commit-pinned CUPMem on official STALE Type-II Return episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace.configuration import resolve_provider_config
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
    run_cupmem_episode,
)
from trace.stale_type2_methods import parse_judge  # noqa: E402
from trace.stale_type2_return import (  # noqa: E402
    OFFICIAL_JUDGE_SYSTEM_PROMPT,
    StaleType2Record,
    build_mas_return_episode,
    load_official_stale_type2,
    official_judge_user_prompt,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _unlink_temporary_best_effort(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        if error.errno != errno.EPERM:
            raise


def _commit_temporary(path: Path, temporary: Path) -> None:
    """Publish a complete temporary file on rename-restricted filesystems."""

    try:
        os.replace(temporary, path)
        return
    except OSError as error:
        if error.errno not in {errno.EPERM, errno.EXDEV}:
            raise

    payload = temporary.read_bytes()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_gzip_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        _commit_temporary(path, temporary)
    finally:
        _unlink_temporary_best_effort(temporary)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _commit_temporary(path, temporary)
    finally:
        _unlink_temporary_best_effort(temporary)


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


def _metrics(judgment: Mapping[str, object]) -> dict[str, object]:
    passes = [
        bool(judgment[f"dim{index}_eval"]["pass"])  # type: ignore[index]
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


def _bucket_records(
    records: Sequence[StaleType2Record], config: Mapping[str, object]
) -> tuple[StaleType2Record, ...]:
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        return tuple(records)
    bucket = dataset.get("natural_departure_bucket")
    if not isinstance(bucket, Mapping):
        return tuple(records)
    inclusive_range = bucket.get("inclusive_range")
    if not isinstance(inclusive_range, Sequence) or len(inclusive_range) != 2:
        raise ValueError("natural departure bucket requires inclusive_range")
    lower, upper = int(inclusive_range[0]), int(inclusive_range[1])
    return tuple(
        record
        for record in records
        if lower <= record.old_session_index <= upper
    )


def _build_engine(
    *, config: Mapping[str, object], output_dir: Path
) -> tuple[CupMemEngine, OpenAICompatibleProvider]:
    model = config["model"]
    judge = config["judge"]
    embedding = config["embedding"]
    execution = config["execution"]
    if not all(isinstance(item, Mapping) for item in (model, judge, embedding, execution)):
        raise ValueError("config model/judge/embedding/execution sections must be objects")
    actor_env = str(model.get("api_key_env") or "").strip()
    judge_env = str(judge.get("api_key_env") or "").strip()
    embedding_env = str(embedding.get("api_key_env") or "").strip()
    actor_key = os.environ.get(actor_env, "") if actor_env else "local"
    judge_key = os.environ.get(judge_env, "") if judge_env else ""
    embedding_key = os.environ.get(embedding_env, "") if embedding_env else None
    if actor_env and not actor_key:
        raise RuntimeError(f"missing actor key environment variable: {actor_env}")
    if judge_env and not judge_key:
        raise RuntimeError(f"missing judge key environment variable: {judge_env}")
    if embedding_env and not embedding_key:
        raise RuntimeError(f"missing embedding key environment variable: {embedding_env}")
    request_lock = str(
        execution.get("actor_request_lock_path")
        or os.environ.get("TRACE_GENERATION_REQUEST_LOCK")
        or "/tmp/trace_qwen_generation.lock"
    )
    llm = LockedLLMClient(
        model=str(model["served_model_id"]),
        api_key=actor_key or "local",
        base_url=str(model["api_base"]),
        cache_dir=output_dir / "cache" / "cupmem_llm",
        request_lock_path=request_lock,
        context_window_tokens=int(
            execution.get("cupmem_full_context_window_tokens", 32_768)
        ),
        default_max_tokens=int(
            execution.get("cupmem_full_default_max_tokens", 3_072)
        ),
        context_reserve_tokens=int(
            execution.get("cupmem_full_context_reserve_tokens", 512)
        ),
        minimum_output_tokens=int(
            execution.get("cupmem_full_min_output_tokens", 512)
        ),
        adaptive_context_retry=bool(
            execution.get("cupmem_full_adaptive_context_retry", True)
        ),
    )
    embedding_client = OpenAICompatibleEmbeddingClient(
        base_url=str(embedding["api_base"]),
        model=str(embedding["served_model_id"]),
        api_key=embedding_key,
        timeout=float(embedding.get("timeout_seconds", 180)),
        retries=int(embedding.get("request_retries", 3)),
    )
    retriever = RemoteEmbeddingRetriever(
        embedding_client,
        cache_dir=output_dir / "cache" / "embeddings",
        batch_size=int(execution.get("embedding_batch_size", 32)),
        max_chars=int(execution.get("embedding_max_chars", 12_000)),
    )
    judge_provider = OpenAICompatibleProvider(
        generation_base_url=str(judge["api_base"]),
        generation_model=str(judge["served_model_id"]),
        generation_api_key=judge_key or None,
        timeout=float(judge.get("model_timeout_seconds", 600)),
        retries=int(judge.get("request_retries", 5)),
        chat_response_format=(
            {"type": "json_object"}
            if bool(judge.get("json_response_format", True))
            else None
        ),
        request_lock_path=(
            str(execution["judge_request_lock_path"])
            if execution.get("judge_request_lock_path")
            else None
        ),
    )
    return CupMemEngine(llm=llm, embedding=retriever), judge_provider


def _judge_answers(
    *,
    provider: OpenAICompatibleProvider,
    record: StaleType2Record,
    answers: Mapping[str, str],
    max_tokens: int,
    format_retries: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    prompt = official_judge_user_prompt(record, answers)
    receipts: list[dict[str, object]] = []
    last_error: Exception | None = None
    for attempt in range(format_retries + 1):
        completion = provider.complete(
            prompt,
            system=OFFICIAL_JUDGE_SYSTEM_PROMPT,
            seed=_seed(record.uid, CUPMEM_SYSTEM_ARM, "judge", attempt),
            max_tokens=max_tokens,
        )
        receipts.append(
            {
                "attempt": attempt + 1,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.total_tokens,
                "latency_seconds": completion.latency_seconds,
                "text_sha256": hashlib.sha256(
                    completion.text.encode("utf-8")
                ).hexdigest(),
            }
        )
        try:
            return parse_judge(completion.text), receipts
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            last_error = error
    raise RuntimeError(f"Judge returned invalid structured output: {last_error}")


def _summary(output_dir: Path, expected: int) -> dict[str, object]:
    metrics: list[Mapping[str, object]] = []
    for path in sorted((output_dir / "results").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        metric = value.get("metrics")
        if isinstance(metric, Mapping):
            metrics.append(metric)
    n = len(metrics)
    summary = {
        "schema_version": "cupmem_full_stale_summary_v1",
        "updated_at": _utc_now(),
        "method": CUPMEM_SYSTEM_ARM,
        "records": n,
        "expected_records": expected,
        "complete": n == expected,
        "dim1_accuracy": sum(bool(item["dim1_pass"]) for item in metrics) / n if n else 0.0,
        "dim2_accuracy": sum(bool(item["dim2_pass"]) for item in metrics) / n if n else 0.0,
        "dim3_accuracy": sum(bool(item["dim3_pass"]) for item in metrics) / n if n else 0.0,
        "overall_accuracy": sum(float(item["overall_accuracy"]) for item in metrics) / n if n else 0.0,
        "via_accuracy": sum(bool(item["via"]) for item in metrics) / n if n else 0.0,
        "iir_accuracy": sum(bool(item["iir"]) for item in metrics) / n if n else 0.0,
        "lifecycle_success_accuracy": sum(bool(item["lifecycle_success"]) for item in metrics) / n if n else 0.0,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--expected-total-records",
        type=int,
        help="Shared-output total used by concurrent fixed-subset lanes.",
    )
    parser.add_argument(
        "--uid",
        action="append",
        default=[],
        help="Run only this official record UID; repeat for a fixed subset.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_provider_config(json.loads(args.config.read_text(encoding="utf-8")))
    dataset = config["dataset"]
    execution = config["execution"]
    records = _bucket_records(
        load_official_stale_type2(ROOT / str(dataset["official_path"])),
        config,
    )
    if args.uid:
        requested = set(args.uid)
        records = tuple(record for record in records if record.uid in requested)
        missing = requested - {record.uid for record in records}
        if missing:
            raise ValueError(
                "unknown requested UIDs: " + ",".join(sorted(missing))
            )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        records = records[: args.limit]
    expected_total_records = (
        int(args.expected_total_records)
        if args.expected_total_records is not None
        else len(records)
    )
    if expected_total_records < len(records):
        raise ValueError(
            "expected_total_records cannot be smaller than this lane"
        )
    output_dir = args.output_dir.resolve()
    engine, judge = _build_engine(config=config, output_dir=output_dir)
    completed = 0
    failed = 0
    for position, record in enumerate(records, start=1):
        result_path = output_dir / "results" / f"{record.uid}.json.gz"
        failure_path = output_dir / "failures" / f"{record.uid}.json"
        if args.resume and _completed_result(result_path):
            completed += 1
            continue
        try:
            lifecycle = build_mas_return_episode(record)
            sessions = tuple(
                CupMemSession(
                    source_session_id=f"session:{index}",
                    timestamp=record.timestamps[index],
                    messages=tuple(message.record() for message in session),
                    phase=(
                        "predeparture"
                        if index <= lifecycle.departure_after_session
                        else "absence"
                    ),
                )
                for index, session in enumerate(record.sessions)
            )
            queries = tuple(
                CupMemQuery(label=f"dim{index}_query", text=text)
                for index, text in enumerate(record.queries, start=1)
            )
            result = None
            transient_retries = int(
                execution.get("cupmem_full_episode_transient_retries", 2)
            )
            for transient_attempt in range(transient_retries + 1):
                try:
                    result = run_cupmem_episode(
                        engine=engine,
                        episode_id=record.uid,
                        benchmark="STALE-TypeII-Official-MAS-Return",
                        sessions=sessions,
                        queries=queries,
                        departure_source_session_id=(
                            f"session:{lifecycle.departure_after_session}"
                        ),
                        max_user_chars_per_session=int(
                            execution.get(
                                "cupmem_full_session_max_chars",
                                12_000,
                            )
                        ),
                        min_user_chars_per_session=int(
                            execution.get(
                                "cupmem_full_min_session_chars",
                                3_000,
                            )
                        ),
                        adaptive_session_bisection=bool(
                            execution.get(
                                "cupmem_full_adaptive_session_bisection",
                                True,
                            )
                        ),
                    )
                    break
                except (URLError, TimeoutError, ConnectionError):
                    if transient_attempt >= transient_retries:
                        raise
                    time.sleep(2.0 * (transient_attempt + 1))
            if result is None:
                raise RuntimeError("CUPMem episode produced no result")
            answers = {
                f"dim{index}_response": str(
                    result["answers"][f"dim{index}_query"]  # type: ignore[index]
                )
                for index in range(1, 4)
            }
            judgment, judge_receipts = _judge_answers(
                provider=judge,
                record=record,
                answers=answers,
                max_tokens=int(execution.get("judge_max_tokens", 2048)),
                format_retries=int(execution.get("judge_format_retries", 2)),
            )
            result.update(
                {
                    "created_at": _utc_now(),
                    "mas_execution": lifecycle.record(),
                    "official_record_modified": False,
                    "actor_oracle_fields_visible": False,
                    "official_answer_record": record.official_answer_record(answers),
                    "official_judge": judgment,
                    "judge_receipts": judge_receipts,
                    "metrics": _metrics(judgment),
                }
            )
            _write_gzip_atomic(result_path, result)
            failure_path.unlink(missing_ok=True)
            completed += 1
            print(
                f"[{position}/{len(records)}] completed uid={record.uid}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - durable formal runner
            failed += 1
            failure = {
                "schema_version": "cupmem_full_failure_v1",
                "updated_at": _utc_now(),
                "uid": record.uid,
                "method": CUPMEM_SYSTEM_ARM,
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            }
            _write_json_atomic(failure_path, failure)
            print(
                f"[{position}/{len(records)}] failed uid={record.uid} "
                f"error={type(error).__name__}: {str(error)[:300]}",
                flush=True,
            )
        _summary(output_dir, expected_total_records)
    summary = _summary(output_dir, expected_total_records)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if failed == 0 and completed == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
