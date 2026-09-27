#!/usr/bin/env python3
"""Prepare once, then run five isolated STALE mechanism workers concurrently."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trace.configuration import resolve_provider_config
import run_stale_type2_mas_return as native
from trace.ablations.core import CoreArm
from trace.ablations.stale import (
    Budgets, compile_arm, answer_prompt, parse_plan, planner_prompt,
    explicit_invalidations,
)
from trace.ablations.planning import (
    candidate_closure, evidence_catalog, catalog_pages, merge_shortlists,
    planning_prompt, parse_requirements, parse_shortlist, render_evidence,
)
from trace.ablations.support_recovery import recover_empty_support
from trace.stale_type2_methods import (
    StaleMemoryItem, session_observation_items, memory_items_sha256,
    SESSION_OBSERVATION_KEY_PREFIX, cosine_similarity, retrieve_for_queries,
)
from trace.stale_type2_return import (
    actor_views, bind_records_to_sidecar, load_official_stale_type2,
    load_sidecar_manifest, canonical_sha256,
)

SCHEMA = "stale_core_mechanisms_v2_task_conditioned"


def configured_arms(config):
    names = config["execution"].get("fork_arms", [a.value for a in CoreArm])
    if not names or len(set(names)) != len(names) or CoreArm.TRACE.value not in names:
        raise ValueError("arm panel must be unique and contain TRACE")
    return tuple(CoreArm(name) for name in names)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    """One writer per file; local POSIX cache storage allows atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def source_contract():
    paths = [Path(__file__).resolve(), Path(native.__file__).resolve(),
             *sorted((ROOT / "src/trace/ablations").glob("*.py")),
             *[ROOT / "src/trace" / f for f in (
                 "return_governance.py", "private_episodic_memory.py", "providers.py",
                 "stale_type2_methods.py", "stale_type2_return.py")]]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_inputs(config, limit=None):
    records = load_official_stale_type2(ROOT / config["dataset"]["official_path"])
    episodes = load_sidecar_manifest(ROOT / config["dataset"]["sidecar_path"])
    bound = list(bind_records_to_sidecar(records, episodes))
    if len(bound) != config["dataset"]["records"]:
        raise ValueError("dataset count changed")
    subset = config["study"].get("uid_subset")
    if subset is not None:
        if not subset or len(subset) != len(set(subset)) or not set(subset).issubset({r.uid for r, _ in bound}):
            raise ValueError("invalid predeclared UID subset")
        bound = [(r, e) for r, e in bound if r.uid in set(subset)]
    return bound[:limit] if limit else bound


def load_upstream(source, record, episode, expected_model):
    """Whitelist upstream state; historical answers and scores never enter preparation."""
    path = source / "results" / f"{record.uid}.json"
    result = read(path)
    manifest = read(source / "run_manifest.json")
    if manifest["actor"]["model"] != expected_model:
        raise ValueError("upstream actor differs from configured Qwen")
    if result["source_record_sha256"] != record.source_record_sha256:
        raise ValueError("upstream source record mismatch")
    if result["sidecar_episode_sha256"] != episode.record()["episode_sha256"]:
        raise ValueError("upstream return boundary mismatch")
    items = []
    session_digests = set()
    for agent, view in sorted(actor_views(record, episode).items()):
        for session in view["sessions"]:
            anchors = session_observation_items(uid=record.uid, agent_id=agent, session=session)
            items.extend(anchors)
            session_digests.update(m.source_session_sha256 for m in anchors)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    for folder in ("extraction_v6_bounded", "extraction_v6_paged"):
        for cache in sorted((source / "cache" / record.uid / folder).glob("*.json")):
            value = read(cache)
            if value["uid"] != record.uid or value.get("query_visible") is not False or value.get("oracle_fields_visible") is not False:
                raise ValueError("upstream extraction is not label-blind")
            parsed = [StaleMemoryItem.from_record(m) for m in value["items"]]
            if any(m.source_session_sha256 not in session_digests for m in parsed):
                raise ValueError("extracted memory has an unknown source")
            items.extend(parsed)
            hashes[str(cache)] = hashlib.sha256(cache.read_bytes()).hexdigest()
    index = {m.memory_id: m for m in sorted(items, key=lambda m: (m.session_index, m.memory_id))}
    items = tuple(index.values())
    if memory_items_sha256(items) != result["extracted_memory_sha256"]:
        raise ValueError("frozen extraction digest mismatch; do not mix cache versions")
    selection = result["methods"]["trace"]["selection"]
    if set(selection["candidate_item_ids"]) != set(index):
        raise ValueError("upstream selection candidate pool changed")
    invalid = {d["memory_id"]: d["invalidated_by_memory_id"]
               for d in selection["decisions"] if d["status"] == "SUPERSEDED"}
    if any(a not in index or b not in index or index[b].session_index <= index[a].session_index
           for a, b in invalid.items()):
        raise ValueError("invalid upstream temporal relation")
    return items, invalid, hashes


def providers(config, out):
    actor = native._provider(config["model"], config["execution"])
    judge = native._provider(config["judge"], config["execution"])
    embedding = native._embedding_provider(config["embedding"], config["execution"])
    runner = native.StaleEpisodeRunner(output_dir=out / "shared_generation", actor=actor,
        judge=judge, embedding=embedding, execution=config["execution"],
        arms=("trace",), config_sha256=canonical_sha256(config))
    return actor, judge, embedding, runner


def prepare_one(config, out, record, episode, actor, embedding):
    if config["study"].get("preparation_protocol") == "task_conditioned_support_review_v3":
        return prepare_support_review(config, out, record, episode, actor)
    if config["study"].get("preparation_protocol") == "task_conditioned_evidence_v2":
        return prepare_task_one(config, out, record, episode, actor, embedding)
    path = out / "prepared" / f"{record.uid}.json"
    if path.exists():
        return
    items, invalid, hashes = load_upstream(Path(config["upstream_root"]), record, episode,
                                         config["model"]["served_model_id"])
    # Observation anchors cover all sessions uniformly. No query-based preselection.
    anchors = tuple(m for m in items if m.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX))
    prompt = planner_prompt(anchors)
    if len(prompt) > config["study"]["planner_max_prompt_chars"]:
        raise ValueError("planner prompt exceeds the predeclared limit")
    plan_path = out / "preparation_cache" / record.uid / "plan.json"
    if plan_path.exists():
        cached = read(plan_path)
        if cached["prompt_sha256"] != canonical_sha256(prompt):
            raise ValueError("planner input changed")
        plan, calls = cached["plan"], cached["calls"]
    else:
        plan, calls = native._call_and_parse(actor, prompt=prompt,
            system="Extract evidence requirements only from supplied observations. Return strict JSON.",
            seed=native._paired_arm_seed(record.uid, "core_obligations_v1"),
            max_tokens=config["study"]["planner_max_tokens"],
            retries=config["execution"]["format_retries"], parse=lambda text: parse_plan(text, anchors))
        write(plan_path, {"plan": plan, "calls": calls, "prompt_sha256": canonical_sha256(prompt)})
    intent = "\n".join(o["description"] for o in plan["obligations"])
    rank_path = out / "preparation_cache" / record.uid / "relevance.json"
    if rank_path.exists():
        ranks = read(rank_path)
    else:
        vectors = embedding.embed([m.statement + "\n" + m.evidence_quote[:700] for m in items] + [intent])
        ranks = {m.memory_id: cosine_similarity(vectors[i], vectors[-1]) for i, m in enumerate(items)}
        write(rank_path, ranks)
    prepared = {"schema": SCHEMA, "uid": record.uid,
        "source_record_sha256": record.source_record_sha256,
        "departure_after_session": episode.departure_after_session,
        "items": [m.record() for m in items], "plan": plan,
        "explicit_invalidations": explicit_invalidations(items), "implicit_invalidations": invalid,
        "relevance_scores": ranks, "upstream_file_hashes": hashes,
        "new_planning_calls": calls, "query_visible_during_preparation": False,
        "oracle_fields_visible_during_preparation": False,
        "private_memory_origin": "source_bound_departure_observations",
        "upstream_cost_note": "Frozen extraction and dependency judgments reused; their original token cost is not zero and is not included in incremental tokens."}
    write(path, prepared)
    print(f"PREPARED {record.uid}", flush=True)


def support_review_inputs(config, record, episode):
    """Validate original serialized evidence before any model call or normalization."""
    study = config["study"]
    base = Path(study["base_prepared_root"])
    manifest = read(base / "manifest.json")
    if manifest["config_sha256"] != study["base_config_sha256"] or record.uid not in manifest["uids"]:
        raise ValueError("base preparation provenance changed")
    prepared = read(base / "prepared" / f"{record.uid}.json")
    if (prepared["source_record_sha256"] != record.source_record_sha256
            or prepared["departure_after_session"] != episode.departure_after_session
            or prepared["task_queries_sha256"] != canonical_sha256(list(record.queries))
            or prepared.get("oracle_fields_visible_during_preparation") is not False
            or prepared.get("preparation_protocol") != "task_conditioned_evidence_v2"):
        raise ValueError("base preparation input boundary mismatch")
    items, invalid, _ = load_upstream(Path(config["upstream_root"]), record, episode,
                                     config["model"]["served_model_id"])
    # from_record strips boundary whitespace. Compare the actual frozen records,
    # so harmless deserialization cannot masquerade as a provenance change.
    if (canonical_sha256([m.record() for m in items]) != canonical_sha256(prepared["items"])
            or invalid != prepared["implicit_invalidations"]
            or explicit_invalidations(items) != prepared["explicit_invalidations"]):
        raise ValueError("base evidence or lifecycle state changed")
    candidate_ids = set(prepared["candidate_ids"])
    if not candidate_ids.issubset({m.memory_id for m in items}):
        raise ValueError("base preparation has unknown candidates")
    catalog = evidence_catalog([m for m in items if m.memory_id in candidate_ids])
    return prepared, items, catalog, manifest


def prepare_support_review(config, out, record, episode, actor):
    """Reuse only frozen shared preparation; never import answers or scores."""
    path = out / "prepared" / f"{record.uid}.json"
    if path.exists():
        return
    study = config["study"]
    prepared, items, catalog, manifest = support_review_inputs(config, record, episode)
    base_sha = canonical_sha256(prepared)
    plan, support_review, calls = recover_empty_support(prepared["plan"], catalog, record.queries,
        lambda stage, prompt, parse: planner_call(actor, config, out, record.uid, stage, prompt, parse),
        max_chars=study["planner_max_prompt_chars"], max_spans=study["support_review_max_spans"])
    prepared = {**prepared, "schema": config["schema_version"], "plan": plan,
        "base_prepared_sha256": base_sha, "base_config_sha256": manifest["config_sha256"],
        "preparation_protocol": study["preparation_protocol"], "support_review": support_review,
        "reused_planning_calls": prepared["new_planning_calls"], "new_planning_calls": calls,
        "rendered_evidence": render_evidence(plan, items)}
    write(out / "support_reviews" / f"{record.uid}.json", support_review)
    write(path, prepared)
    print(f"PREPARED {record.uid} review=" + json.dumps(support_review, ensure_ascii=False), flush=True)


def planner_call(actor, config, out, uid, stage, prompt, parse):
    """Persist every attempt, including invalid responses and transport failures."""
    stage_dir = out / "preparation_cache" / uid / stage
    prompt_sha = canonical_sha256(prompt)
    success_path = stage_dir / "success.json"
    if success_path.exists():
        cached = read(success_path)
        if cached["prompt_sha256"] != prompt_sha:
            raise ValueError("planner stage input changed")
        return parse(cached["response_text"]), cached["calls"]
    calls = [read(p) for p in sorted(stage_dir.glob("attempt_*.json"))]
    feedback = ""
    seed = native._paired_arm_seed(uid, "core_task_v2:" + stage)
    attempts = config["study"].get("planner_format_retries", config["execution"]["format_retries"])
    attempt_base = len(calls)
    for attempt in range(attempts):
        effective = prompt + feedback
        receipt = {"attempt": attempt_base + attempt, "stage": stage,
                   "prompt_sha256": canonical_sha256(effective)}
        try:
            system = ("Select only supplied evidence indices and requested review fields. Return strict JSON."
                      if stage.startswith("support_") else "Select only supplied evidence indices. Return strict JSON.")
            completion = actor.complete(effective, system=system,
                seed=seed + attempt_base + attempt, temperature=0.0,
                max_tokens=config["study"]["planner_max_tokens"])
            receipt["completion"] = native._completion_record(completion)
            receipt["response_text"] = completion.text
            value = parse(completion.text)
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            feedback = "\nCorrect the previous validation error and return a fresh JSON object: " + receipt["error"][:500]
            calls.append(receipt)
            write(stage_dir / f"attempt_{attempt_base + attempt:03d}.json", receipt)
            continue
        calls.append(receipt)
        write(stage_dir / f"attempt_{attempt_base + attempt:03d}.json", receipt)
        write(success_path, {"prompt_sha256": prompt_sha, "response_text": completion.text, "calls": calls})
        return value, calls
    raise ValueError(f"planner {stage} failed after {attempts} attempts: {calls[-1]['error']}")


def prepare_task_one(config, out, record, episode, actor, embedding):
    path = out / "prepared" / f"{record.uid}.json"
    if path.exists():
        return
    items, invalid, hashes = load_upstream(Path(config["upstream_root"]), record, episode,
                                         config["model"]["served_model_id"])
    explicit = explicit_invalidations(items)
    rank_path = out / "preparation_cache" / record.uid / "task_retrieval.json"
    retrieval_input = canonical_sha256({"items": memory_items_sha256(items),
        "queries": list(record.queries), "embedding": config["embedding"],
        "top_k": config["execution"]["retrieval_top_k_per_query"],
        "union": config["execution"]["retrieval_max_union_items"]})
    if rank_path.exists():
        ranks = read(rank_path)
        if ranks["input_sha256"] != retrieval_input:
            raise ValueError("retrieval input changed")
    else:
        vectors = embedding.embed([f"{m.state_key}\n{m.statement}\n{m.evidence_quote}" for m in items]
                                  + list(record.queries))
        item_vectors = {m.memory_id: vectors[i] for i, m in enumerate(items)}
        query_vectors = vectors[len(items):]
        retrieved = retrieve_for_queries(items, record.queries, item_embeddings=item_vectors,
            query_embeddings=query_vectors, top_k_per_query=config["execution"]["retrieval_top_k_per_query"],
            max_union_items=config["execution"]["retrieval_max_union_items"])
        ranks = {"input_sha256": retrieval_input, "seed_ids": [m.memory_id for m in retrieved],
            "scores": {m.memory_id: max(cosine_similarity(item_vectors[m.memory_id], q)
                                       for q in query_vectors) for m in items}}
        write(rank_path, ranks)
    # Common candidate construction must not use the ablated implicit edges.
    # Keep every raw session anchor, plus query retrieval and explicit successors.
    # Paged preparation covers their full stored text instead of prefix truncation.
    seeds = [*ranks["seed_ids"], *[m.memory_id for m in items
                                 if m.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX)]]
    candidates = candidate_closure(items, seeds, explicit)
    catalog = evidence_catalog(candidates)
    max_chars = config["study"]["planner_max_prompt_chars"]
    calls = []
    selected_catalog = catalog
    if len(planning_prompt(catalog, record.queries)) > max_chars:
        pages = catalog_pages(catalog, record.queries, max_chars)
        shortlists = []
        for i, page in enumerate(pages):
            shortlist, receipts = planner_call(actor, config, out, record.uid, f"shortlist_{i:02d}",
                planning_prompt(page, record.queries, shortlist=True),
                lambda text, page=page: parse_shortlist(text, page))
            shortlists.append(shortlist)
            calls.extend(receipts)
        selected_catalog = merge_shortlists(shortlists, catalog, record.queries, max_chars)
    prompt = planning_prompt(selected_catalog, record.queries)
    if len(prompt) > max_chars:
        raise ValueError("planner prompt exceeds declared budget")
    plan, receipts = planner_call(actor, config, out, record.uid, "requirements", prompt,
        lambda text: parse_requirements(text, selected_catalog, record.queries))
    calls.extend(receipts)
    prepared = {"schema": SCHEMA, "uid": record.uid,
        "source_record_sha256": record.source_record_sha256,
        "departure_after_session": episode.departure_after_session,
        "items": [m.record() for m in items], "plan": plan,
        "explicit_invalidations": explicit, "implicit_invalidations": invalid,
        "relevance_scores": ranks["scores"], "upstream_file_hashes": hashes,
        "new_planning_calls": calls,
        "query_visible_during_preparation": True, "oracle_fields_visible_during_preparation": False,
        "preparation_protocol": config["study"]["preparation_protocol"],
        "requirement_semantics": "current_state_or_verified_invalidation",
        "task_queries_sha256": canonical_sha256(list(record.queries)),
        "retrieval_seed_ids": ranks["seed_ids"], "candidate_ids": [m.memory_id for m in candidates],
        "catalog_span_count": len(catalog), "planner_visible_span_count": len(selected_catalog),
        "rendered_evidence": render_evidence(plan, items), "include_lifecycle_resolutions": True,
        "private_memory_origin": "source_bound_departure_observations",
        "upstream_cost_note": "Frozen extraction and dependency judgments reused; their original token cost is not zero and is not included in incremental tokens."}
    write(path, prepared)
    print(f"PREPARED {record.uid}", flush=True)


def prepare(config, out, bound):
    actor = native._provider(config["model"], config["execution"])
    embedding = native._embedding_provider(config["embedding"], config["execution"])
    errors = []
    with ThreadPoolExecutor(max_workers=config["study"]["preparation_workers"]) as pool:
        pending = {pool.submit(prepare_one, config, out, r, e, actor, embedding): r.uid for r, e in bound}
        for future in as_completed(pending):
            uid = pending[future]
            try:
                future.result()
            except Exception as error:
                message = {"uid": uid, "type": type(error).__name__, "error": str(error)}
                write(out / "preparation_failures" / f"{uid}.json", message)
                print(f"PREPARATION_FAILED {uid} {type(error).__name__}: {error}", flush=True)
                errors.append(message)
    write(out / "preparation_done.json", {"finished_at": now(), "expected": len(bound),
          "ready": sum((out / "prepared" / f"{r.uid}.json").exists() for r, _ in bound), "failures": errors})
    return int(bool(errors))


def incremental_tokens(receipts):
    return sum(c.get("completion", {}).get("total_tokens", 0) for c in receipts)


def run_one(config, out, record, arm, runner, signing_key):
    prepared = read(out / "prepared" / f"{record.uid}.json")
    view = compile_arm(prepared, arm, Budgets(**config["study"]["budgets"]),
        key=signing_key, queries=record.queries, condition=config["study"]["condition"])
    # Persist the exact intervention before any downstream generation.
    write(out / "arms" / arm.value / "views" / f"{record.uid}.json", view)
    prompt = answer_prompt(view["content"], record.queries)
    input_id = canonical_sha256({"prompt": prompt, "uid": record.uid,
        "actor": config["model"], "judge": config["judge"], "execution": config["execution"]})
    # Identical actor inputs share answers AND judgments across processes.
    with lock(out / "locks" / f"{input_id}.lock"):
        answers, answer_receipt = runner._answers(record=record, arm=input_id, prompt_override=prompt)
        judgment, judge_receipt = runner._judge(record=record, arm=input_id, answers=answers)
    result = {"schema": config.get("schema_version", SCHEMA), "uid": record.uid, "arm": arm.value,
        "finished_at": now(), "condition": config["study"]["condition"],
        "prepared_sha256": canonical_sha256(prepared), "answer_input_sha256": input_id,
        "answers": answers, "judgment": judgment, "metrics": native._metrics_from_judgment(judgment),
        "view_status": view["compilation"]["status"],
        "raw_obligation_coverage": view["raw_obligation_coverage"],
        "unavailable_dependencies_before_selection": view["unavailable_dependencies_before_selection"],
        "private_reexposed_source_ids": view["private_reexposed_source_ids"],
        "support_review": prepared.get("support_review"),
        "incremental_actor_tokens": incremental_tokens(prepared["new_planning_calls"]) + incremental_tokens(answer_receipt["calls"]),
        "reused_planning_tokens": incremental_tokens(prepared.get("reused_planning_calls", [])),
        "judge_tokens": incremental_tokens(judge_receipt["calls"]),
        "full_lifecycle_tokens": None,
        "cost_scope": "new shared planning/review plus answer; reused planning separately reported; historical extraction/invalidation excluded; judge separate",
        "answer_receipt": answer_receipt, "judge_receipt": judge_receipt}
    write(out / "arms" / arm.value / "results" / f"{record.uid}.json", result)
    print(f"COMPLETED {arm.value} {record.uid}", flush=True)


def arm_worker(config, out, bound, arm):
    _, _, _, runner = providers(config, out)
    key = (out / "signing.key").read_bytes()
    pending = {r.uid: r for r, _ in bound}
    deadline = time.monotonic() + config["study"].get("preparation_wait_seconds", 86400)
    failed = []
    while pending:
        progress = False
        for uid, record in list(pending.items()):
            if (out / "arms" / arm.value / "results" / f"{uid}.json").exists():
                pending.pop(uid)
                progress = True
                continue
            if not (out / "prepared" / f"{uid}.json").exists():
                continue
            try:
                run_one(config, out, record, arm, runner, key)
            except Exception as error:
                write(out / "arms" / arm.value / "failures" / f"{uid}.json",
                      {"uid": uid, "type": type(error).__name__, "error": str(error)})
                print(f"FAILED {arm.value} {uid} {type(error).__name__}: {error}", flush=True)
                failed.append(uid)
            pending.pop(uid)
            progress = True
            summarize(out, bound)
        if pending and (out / "preparation_done.json").exists() and not progress:
            failed.extend(pending)
            break
        if pending and not progress:
            if time.monotonic() >= deadline:
                failed.extend(pending)
                break
            time.sleep(3)
    write(out / "arms" / arm.value / "done.json", {"finished_at": now(), "failed_uids": failed,
          "expected": len(bound), "complete": not failed})
    summarize(out, bound)
    return int(bool(failed))


def summarize(out, bound):
    with lock(out / "locks/summary.lock"):
        arm_names = read(out / "manifest.json")["arms"]
        rows = {arm.value: {p.stem: read(p) for p in (out / "arms" / arm.value / "results").glob("*.json")}
                for arm in map(CoreArm, arm_names)}
        paired = set.intersection(*(set(r) for r in rows.values()))
        report = {"updated_at": now(), "expected_per_arm": len(bound), "paired_completed": len(paired),
                  "complete": len(paired) == len(bound), "arms": {}}
        report["preparation"] = {
            "ready": len(list((out / "prepared").glob("*.json"))),
            "unresolved_failures": sum(not (out / "prepared" / p.name).exists()
                                       for p in (out / "preparation_failures").glob("*.json"))}
        review_states = {}
        for p in (out / "support_reviews").glob("*.json"):
            for q in read(p).get("questions", []):
                review_states[q["status"]] = review_states.get(q["status"], 0) + 1
        report["preparation"]["support_review_states"] = review_states
        for arm, results in rows.items():
            metrics = [results[uid]["metrics"] for uid in sorted(paired)]
            report["arms"][arm] = {"completed": len(results), "paired_metrics": {
                name: sum(float(m[name]) for m in metrics) / len(metrics) if metrics else None
                for name in ("overall_accuracy", "via", "iir", "lifecycle_success")}}
            report["arms"][arm]["paired_diagnostics"] = {
                "reset_count": sum(results[u]["view_status"] == "reset_fallback" for u in paired),
                "missing_source_support_count": sum(bool(results[u].get("unavailable_dependencies_before_selection")) for u in paired),
                "same_actor_input_as_trace": sum(results[u]["answer_input_sha256"] == rows["trace"][u]["answer_input_sha256"] for u in paired),
                "private_reexposure_count": sum(bool(results[u]["private_reexposed_source_ids"]) for u in paired)}
            report["arms"][arm]["unresolved_failures"] = sum(p.stem not in results
                for p in (out / "arms" / arm / "failures").glob("*.json"))
        write(out / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("launch", "prepare", "arm", "preflight", "summarize"), default="launch")
    parser.add_argument("--arm", choices=[x.value for x in CoreArm])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = resolve_provider_config(read(args.config))
    arms = configured_arms(config)
    out = args.output_dir.resolve()
    bound = load_inputs(config, args.limit)
    if args.mode == "preflight":
        # Exhaustively validate all frozen inputs without a generation request.
        for record, episode in bound:
            if config["study"].get("preparation_protocol") == "task_conditioned_support_review_v3":
                support_review_inputs(config, record, episode)
            else:
                load_upstream(Path(config["upstream_root"]), record, episode, config["model"]["served_model_id"])
        actor, judge, embedding, _ = providers(config, out)
        for provider, model, role in [(actor, actor.generation_model, "actor"),
                (judge, judge.generation_model, "judge"), (embedding, embedding.model_id, "embedding")]:
            native._health(provider, model, role)
        print(json.dumps({"ok": True, "validated_uids": len(bound)}))
        return
    if args.mode == "summarize":
        summarize(out, bound)
        return
    if args.mode in {"prepare", "arm"}:
        manifest = read(out / "manifest.json")
        if manifest["source_contract"] != source_contract() or manifest["config_sha256"] != canonical_sha256(config):
            raise ValueError("source or configuration changed after launch")
        if manifest["uids"] != [r.uid for r, _ in bound]:
            raise ValueError("worker UID set differs from launch manifest")
        if args.mode == "prepare":
            sys.exit(prepare(config, out, bound))
        if not args.arm:
            raise ValueError("arm worker requires --arm")
        if CoreArm(args.arm) not in arms:
            raise ValueError("arm is not in the configured panel")
        sys.exit(arm_worker(config, out, bound, CoreArm(args.arm)))
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / "locks/launcher.lock"):
        manifest = {"schema": config.get("schema_version", SCHEMA), "created_at": now(), "config_sha256": canonical_sha256(config),
            "source_contract": source_contract(), "uids": [r.uid for r, _ in bound],
            "arms": [x.value for x in arms], "condition": config["study"]["condition"],
            "statistical_scope": config.get("statistical_status", "STALE-derived diagnostic independent of historical main-table results")}
        if (out / "manifest.json").exists():
            old = read(out / "manifest.json")
            if any(old[k] != manifest[k] for k in ("source_contract", "config_sha256", "uids", "arms")):
                raise ValueError("refusing to mix a different experiment into this output directory")
            # A resume must not mistake the previous producer's terminal marker
            # for the new producer's completion while its workers are waiting.
            stamp = str(time.time_ns())
            for marker in [out / "preparation_done.json", out / "terminal.json",
                           *[out / "arms" / a.value / "done.json" for a in arms]]:
                if marker.exists():
                    archive = out / "attempt_receipts" / stamp / marker.relative_to(out)
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(marker, archive)
        else:
            write(out / "manifest.json", manifest)
            fd = os.open(out / "signing.key", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(os.urandom(32))
        base = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                "--output-dir", str(out)]
        if args.limit:
            base += ["--limit", str(args.limit)]
        children = {}
        for name, extra in [("prepare", ["--mode", "prepare"]),
                           *[(a.value, ["--mode", "arm", "--arm", a.value]) for a in arms]]:
            log = out / "logs" / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                children[name] = subprocess.Popen(base + extra, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        write(out / "launch.json", {"started_at": now(), "pids": {n: p.pid for n, p in children.items()}})
        print(json.dumps({"output_dir": str(out), "pids": {n: p.pid for n, p in children.items()}}), flush=True)
        exits = {"prepare": children["prepare"].wait()}
        if not (out / "preparation_done.json").exists():
            write(out / "preparation_done.json", {"finished_at": now(),
                  "fatal_producer_exit": exits["prepare"]})
        exits.update({name: process.wait() for name, process in children.items() if name != "prepare"})
        summarize(out, bound)
        write(out / "terminal.json", {"finished_at": now(), "exit_codes": exits,
              "complete": read(out / "summary.json")["complete"]})
        if any(exits.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
