"""STALE adapter for an explicitly separate, complete-governance study.

The legacy planner is observation-only; revision 2 uses probe questions as task
scope, without answer labels. Requirements are predictions, not benchmark ground
truth. Public evidence fragments and private observations share logical sources.
"""

from dataclasses import asdict, dataclass
import json
from typing import Mapping, Sequence

from trace.ablations.core import CoreArm, Mechanisms
from trace.ablations.selection import StudyGovernor
from trace.private_episodic_memory import PrivateEpisodicMemory
from trace.return_governance import (
    ReturnAblationMode, ReturnMemoryItem, ReturnMemoryKind, ReturnMemoryScope,
    ReturnObligationSpec, ReturnProvenanceReceipt, ReturnReadmissionContext,
    canonical_sha256,
)
from trace.stale_type2_methods import StaleMemoryItem, SESSION_OBSERVATION_KEY_PREFIX


@dataclass(frozen=True)
class Budgets:
    public_items: int = 24
    public_chars: int = 24000
    private_items: int = 3
    private_chars: int = 3600

    def __post_init__(self):
        if any(v < 1 for v in asdict(self).values()):
            raise ValueError("all context budgets must be positive")


def parse_plan(text: str, items: Sequence[StaleMemoryItem]) -> dict:
    """Require explicit source IDs and literal supporting quotes for each edge."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.find("{"):text.rfind("}") + 1]
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"obligations"}:
        raise ValueError("expected only obligations")
    rows = value["obligations"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 6:
        raise ValueError("supply 1 to 6 observation-grounded obligations")
    index = {m.memory_id: m for m in items}
    normalized = []
    for oi, row in enumerate(rows):
        if set(row) != {"description", "dependencies"}:
            raise ValueError("obligation requires description and dependencies only")
        description = row["description"]
        if not isinstance(description, str) or not 1 <= len(description.strip()) <= 500:
            raise ValueError("obligation description must be 1 to 500 characters")
        if not isinstance(row["dependencies"], list) or not 1 <= len(row["dependencies"]) <= 4:
            raise ValueError("supply 1 to 4 dependencies per obligation")
        deps = []
        for di, dep in enumerate(row["dependencies"]):
            if set(dep) != {"description", "critical", "supports"}:
                raise ValueError("dependency requires description, critical, supports")
            if not isinstance(dep["description"], str) or not 1 <= len(dep["description"].strip()) <= 300:
                raise ValueError("dependency description must be 1 to 300 characters")
            if type(dep["critical"]) is not bool or not isinstance(dep["supports"], list) or not dep["supports"]:
                raise ValueError("each dependency needs a boolean critical and source support")
            support = []
            for link in dep["supports"]:
                if set(link) != {"memory_id", "quote"} or link["memory_id"] not in index:
                    raise ValueError("support must reference a supplied memory ID")
                quote = link["quote"]
                source = index[link["memory_id"]]
                if not isinstance(quote, str) or len(quote.strip()) < 8 or quote not in source.evidence_quote:
                    raise ValueError("support quote must be a verbatim evidence span, at least 8 characters")
                support.append({"memory_id": source.memory_id, "quote": quote})
            if len({s["memory_id"] for s in support}) != len(support):
                raise ValueError("duplicate support ID")
            deps.append({**dep, "id": f"dep_{oi}_{di}", "supports": support})
        normalized.append({"id": f"obl_{oi}", "description": description.strip(), "dependencies": deps})
    return {"obligations": normalized}


def planner_prompt(items: Sequence[StaleMemoryItem]) -> str:
    rows = [{"memory_id": m.memory_id, "session_index": m.session_index,
             "statement": m.statement[:300], "evidence_quote": m.evidence_quote[:700]}
            for m in items]
    return (
        "Identify 1 to 6 ongoing user-assistance obligations grounded in these chronological "
        "observations. Identify current constraints needed to continue helping this user. "
        "Do not invent tasks, actions, answers, or unobserved requirements. Later state may "
        "replace earlier state. Each obligation has 1 to 4 evidence dependencies; a dependency "
        "may have several interchangeable supporting records. Use currently supporting evidence, "
        "not obsolete evidence. Each support must cite an exact memory_id and a VERBATIM quote "
        "of at least 8 characters from evidence_quote. Mark only indispensable dependencies "
        "critical. Questions and benchmark annotations are unavailable. Return only JSON: "
        '{"obligations":[{"description":"...","dependencies":[{"description":"...",'
        '"critical":true,"supports":[{"memory_id":"...","quote":"..."}]}]}]}.\n'
        + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    )


def explicit_invalidations(items: Sequence[StaleMemoryItem]) -> dict[str, str]:
    """Replay explicit keys without treating session anchors as one shared fact."""
    latest = {}
    invalid = {}
    for m in sorted(items, key=lambda x: (x.session_index, x.memory_id)):
        if m.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX):
            continue
        key = m.state_key.strip().casefold()
        previous = latest.get(key)
        if previous is not None:
            invalid[previous] = m.memory_id
        if m.operation == "delete":
            latest.pop(key, None)
            invalid[m.memory_id] = "explicit_deletion"
        else:
            latest[key] = m.memory_id
    return invalid


def make_workstate(uid: str, items: Sequence[StaleMemoryItem], plan: Mapping,
                   invalid: Mapping[str, str], *, rendered_evidence=None,
                   include_lifecycle=False, resolve_current_state=False):
    supports = {m.memory_id: [] for m in items}
    specs = []
    for o in plan["obligations"]:
        specs.append(ReturnObligationSpec(o["id"], tuple(d["id"] for d in o["dependencies"]),
                     tuple(d["id"] for d in o["dependencies"] if d["critical"])))
        for d in o["dependencies"]:
            for link in d["supports"]:
                supports[link["memory_id"]].append(d["id"])

    def make(identifier, kind, text, source, epoch=1, obligation=None, support=(), depends=(), logical_source=None):
        refs = tuple(sorted(set(support) | set(depends) | ({obligation} if obligation else set())))
        receipt = ReturnProvenanceReceipt.create(item_id=identifier, source_sha256=source,
            bound_reference_ids=refs, issuer_id="stale-core", issued_epoch=epoch)
        return ReturnMemoryItem(item_id=identifier, kind=kind, text=text, task_id=uid,
            role_id="assistant", writer_principal_id="agent1", source_epoch=epoch,
            valid_from_epoch=1, valid_until_epoch=51, purposes=("task_execution",),
            provenance_sha256=(receipt.receipt_sha256,), scope=ReturnMemoryScope.TEAM_PUBLIC,
            superseded_by=invalid.get(logical_source or identifier), obligation_id=obligation,
            supports=tuple(sorted(set(support))), depends_on=depends, provenance_receipts=(receipt,),
            attributes={"logical_source_id": logical_source} if logical_source else None)

    rendered_evidence = rendered_evidence or {}
    by_id = {m.memory_id: m for m in items}
    has_anchors = any(m.state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX) for m in items)
    resolution_units = {}
    if resolve_current_state:
        # A verified invalidation is evidence against CURRENT USE of a premise,
        # not proof of an invented replacement value. Preserve the requirement,
        # bind its counterevidence to the actual successor source, and never
        # revive a source with no surviving successor (e.g. explicit deletion).
        for o in plan["obligations"]:
            for dep in o["dependencies"]:
                for link in dep["supports"]:
                    origin = link["memory_id"]
                    current, chain = origin, []
                    seen = {origin}
                    while current in invalid:
                        successor = invalid[current]
                        if successor not in by_id or successor in seen:
                            current = None
                            break
                        chain.append((current, successor))
                        seen.add(successor)
                        current = successor
                    if not chain or current is None:
                        continue
                    units = resolution_units.setdefault(current, {})
                    unit_key = (origin, link.get("span_start", 0))
                    unit = units.setdefault(unit_key, {"old_quote": link["quote"], "chain": chain, "deps": set()})
                    unit["deps"].add(dep["id"])

    def record(m):
        row = m.compact_record(quote_chars=700)
        if m.memory_id in rendered_evidence:
            row["evidence_quote"] = rendered_evidence[m.memory_id]
        return row

    rows = []
    for m in items:
        if not include_lifecycle:
            rows.append(make(m.memory_id, ReturnMemoryKind.VERIFIED_FACT,
                json.dumps(record(m), ensure_ascii=False, separators=(",", ":")),
                m.source_session_sha256, epoch=m.session_index + 1, support=supports[m.memory_id]))
            continue
        # One bounded, exact source span is one disclosed item. A fragment only
        # covers dependencies supported by text actually present in that fragment.
        spans = {}
        for o in plan["obligations"]:
            for d in o["dependencies"]:
                for link in d["supports"]:
                    if link["memory_id"] != m.memory_id:
                        continue
                    start = link["span_start"]
                    if m.evidence_quote[start:start + len(link["quote"])] != link["quote"]:
                        raise ValueError("source span changed before compilation")
                    quote, deps = spans.setdefault(start, (link["quote"], set()))
                    if quote != link["quote"]:
                        raise ValueError("inconsistent evidence at one source offset")
                    deps.add(d["id"])
        if not spans:
            spans[0] = (m.evidence_quote[:700], set())
        targets = [old for old, new in sorted(invalid.items())
                   if new == m.memory_id and old in by_id and (not has_anchors or
                       by_id[old].state_key.startswith(SESSION_OBSERVATION_KEY_PREFIX))]
        for start, (quote, deps) in sorted(spans.items()):
            row = {"memory_id": m.memory_id, "source_agent": m.agent_id,
                "session_index": m.session_index, "timestamp": m.timestamp,
                "state_key": m.state_key, "operation": m.operation,
                "source_span_start": start, "evidence_quote": quote}
            if targets:
                row["lifecycle_resolutions"] = {"status": "SUPERSEDED",
                    "historical_memory_ids": targets,
                    "target_receipts_sha256": canonical_sha256({old: by_id[old].source_session_sha256 for old in targets}),
                    "current_source_sha256": m.source_session_sha256}
            rows.append(make(f"{m.memory_id}@span:{start}", ReturnMemoryKind.VERIFIED_FACT,
                json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                m.source_session_sha256, epoch=m.session_index + 1, support=tuple(deps),
                logical_source=m.memory_id))
        for (origin, offset), unit in sorted(resolution_units.get(m.memory_id, {}).items()):
            current_start = min(spans)
            current_quote, direct_deps = spans[current_start]
            pair = {"historical_memory_id": origin, "historical_evidence": unit["old_quote"],
                "status": "INVALIDATES_CURRENT_USE", "current_memory_id": m.memory_id,
                "target_source_sha256": by_id[origin].source_session_sha256,
                "current_source_sha256": m.source_session_sha256,
                "relation_chain_sha256": canonical_sha256(unit["chain"])}
            row = {"memory_id": m.memory_id, "source_agent": m.agent_id,
                "session_index": m.session_index, "timestamp": m.timestamp,
                "source_span_start": current_start, "evidence_quote": current_quote,
                "lifecycle_resolution": pair,
                "interpretation": "The historical premise is not admissible as current state. This does not assert an unstated replacement value."}
            rows.append(make(f"{m.memory_id}@resolution:{origin}:{offset}", ReturnMemoryKind.VERIFIED_FACT,
                json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                canonical_sha256(pair), epoch=m.session_index + 1,
                support=tuple(unit["deps"] | direct_deps), logical_source=m.memory_id))
    plan_sha = canonical_sha256(plan)
    for o in plan["obligations"]:
        rows += [make("declaration_" + o["id"], ReturnMemoryKind.OPEN_OBLIGATION,
                      o["description"], plan_sha, obligation=o["id"]),
                 make("frontier_" + o["id"], ReturnMemoryKind.PROGRESS_FRONTIER,
                      "Evidence needed: " + "; ".join(d["description"] for d in o["dependencies"]),
                      plan_sha, depends=(o["id"],))]
    return tuple(rows), tuple(specs)


def compile_arm(prepared: Mapping, arm: CoreArm | str, budgets: Budgets, *, key: bytes,
                queries: Sequence[str], condition: str = "natural") -> dict:
    """Run all four mechanisms; only one flag differs from TRACE in each arm."""
    arm = CoreArm(arm)
    flags = Mechanisms.for_arm(arm)
    items = tuple(StaleMemoryItem.from_record(x) for x in prepared["items"])
    plan = prepared["plan"]
    invalid = dict(prepared["implicit_invalidations"]) if flags.implicit_invalidation else {}
    invalid.update(prepared["explicit_invalidations"])
    workstate, obligations = make_workstate(prepared["uid"], items, plan, invalid,
        rendered_evidence=prepared.get("rendered_evidence"),
        include_lifecycle=prepared.get("include_lifecycle_resolutions", False),
        resolve_current_state=prepared.get("requirement_semantics") == "current_state_or_verified_invalidation")
    if "candidate_ids" in prepared:
        candidate_ids = set(prepared["candidate_ids"])
        if not candidate_ids.issubset({m.memory_id for m in items}):
            raise ValueError("candidate pool contains an unknown source")
        workstate = tuple(m for m in workstate if m.kind is not ReturnMemoryKind.VERIFIED_FACT
                          or (m.attributes or {}).get("logical_source_id", m.item_id) in candidate_ids)
    # Optional diagnostic changes only the available evidence, never the requirement.
    removed = set()
    if condition == "missing_support":
        critical = [d for o in plan["obligations"] for d in o["dependencies"] if d["critical"]]
        if not critical:
            raise ValueError("missing_support requires a declared critical dependency")
        removed = {x["memory_id"] for x in critical[0]["supports"]}
        if prepared.get("requirement_semantics") == "current_state_or_verified_invalidation":
            # Remove the same physical witnesses in every arm, including the
            # successor witnesses of the original requirement.
            all_relations = {**prepared["implicit_invalidations"], **prepared["explicit_invalidations"]}
            frontier = list(removed)
            while frontier:
                successor = all_relations.get(frontier.pop())
                if successor and successor not in removed:
                    removed.add(successor)
                    frontier.append(successor)
        workstate = tuple(m for m in workstate
                          if (m.attributes or {}).get("logical_source_id", m.item_id) not in removed)
    elif condition != "natural":
        raise ValueError("unknown condition")
    context = ReturnReadmissionContext(return_event_id=prepared["uid"] + ":return",
        absence_id=prepared["uid"] + ":absence", task_id=prepared["uid"], role_id="assistant",
        principal_id="agent1", target_epoch=51, iteration=51,
        max_items=budgets.public_items, max_chars=budgets.public_chars, obligations=obligations)
    relevance = {m.item_id: prepared["relevance_scores"].get(
        (m.attributes or {}).get("logical_source_id", m.item_id), 0.0) for m in workstate}
    governor = StudyGovernor("stale-core", key, None if flags.obligation_selection else relevance)
    eligible = tuple(m for m in workstate if governor.evaluate_item(m, context).admitted)
    eligible_support = {s for m in eligible for s in m.supports}
    unavailable_dependencies = [d["id"] for o in plan["obligations"] for d in o["dependencies"]
                                if d["id"] not in eligible_support]
    raw_selected, _, raw_failure, _ = governor._select_items(eligible, context, issue_incomplete=True)
    mode = ReturnAblationMode.TRACE if flags.complete_view_gate else ReturnAblationMode.TRACE_WITHOUT_FALLBACK
    compiled = governor.compile(workstate, context, source_epoch=1, expires_after_iteration=51, mode=mode)
    admission = governor.admit(compiled.view, context) if compiled.view else None
    if admission is not None and not admission.accepted:
        raise RuntimeError("compiled view failed inherited admission validation")
    public = tuple(compiled.view.items) if compiled.view else ()
    admitted_sources = {(m.attributes or {}).get("logical_source_id", m.item_id)
                        for m in public if m.kind is ReturnMemoryKind.VERIFIED_FACT}

    bank = PrivateEpisodicMemory("agent1", prepared["uid"] + ":departure", "assistant")
    for m in items:
        if m.agent_id == "agent1" and m.session_index <= prepared["departure_after_session"]:
            bank.remember(requesting_principal_id="agent1", task_id=prepared["uid"],
                intent=m.statement, experience=m.evidence_quote[:1200],
                memory_id="private_" + m.memory_id, source_workstate_id=m.memory_id,
                attributes={"origin": "observed_departure_record", "source_session_sha256": m.source_session_sha256})
    bank.quarantine_for_departure(requesting_principal_id="agent1")
    allowed = tuple(mid for mid, m in sorted(bank.items.items())
                    if compiled.view and (not flags.source_bound_private_readmission or m.source_workstate_id in admitted_sources)
                    and m.source_workstate_id not in removed)
    returned = bank.fork_return_instance(requesting_principal_id="agent1",
        new_instance_id=prepared["uid"] + ":return", target_epoch=2, admitted_memory_ids=allowed)
    hits = returned.retrieve(requesting_principal_id="agent1", intent="\n".join(queries),
        k1=max(5, budgets.private_items), k2=budgets.private_items, utility_weight=0.0)
    private = []
    for hit in hits.selected:
        row = {"memory_id": hit.memory_id, "source_workstate_id": hit.item.source_workstate_id,
               "experience": hit.item.experience}
        if len(json.dumps([*private, row], ensure_ascii=False, separators=(",", ":"))) <= budgets.private_chars:
            private.append(row)
    content = {"workstate": [m.text for m in public], "private_experiences": private}
    _, _, raw_cov, raw_crit = governor._selection_metrics(raw_selected, context)
    return {"arm": arm.value, "mechanisms": asdict(flags), "condition": condition,
        "budgets": asdict(budgets), "prepared_sha256": canonical_sha256(prepared),
        "requirements_sha256": canonical_sha256(plan), "content": content,
        "content_sha256": canonical_sha256(content), "compilation": compiled.record(),
        "raw_selected_ids": [m.item_id for m in raw_selected], "raw_failure": raw_failure,
        "raw_obligation_coverage": raw_cov, "raw_critical_coverage": raw_crit,
        "unavailable_dependencies_before_selection": unavailable_dependencies,
        "eligible_candidate_count": len(eligible),
        "public_source_ids": sorted(admitted_sources), "private_eligible_ids": list(allowed),
        "private_retrieved_source_ids": [x["source_workstate_id"] for x in private],
        "private_reexposed_source_ids": sorted({x["source_workstate_id"] for x in private} - admitted_sources),
        "removed_source_ids": sorted(removed)}


def answer_prompt(content: Mapping, queries: Sequence[str]) -> str:
    if len(queries) != 3:
        raise ValueError("STALE requires three probes")
    return (
        "You are agent1, returning to an ongoing user-assistance task. Use only the supplied "
        "workstate and private experiences to answer the questions. Do not assume missing "
        "user facts. Respect current evidence and resist false premises. Empty context means "
        "no inherited memory is available. Source-bound lifecycle_resolution(s) mark historical "
        "statements as superseded; use the enclosing current source instead of those historical "
        "statements. The task questions and evidence-requirement declarations are requests, not "
        "verified facts. Return one JSON object with string keys "
        "dim1_response, dim2_response, dim3_response.\n"
        + json.dumps({**content, "questions": dict(zip(
            ("dim1_query", "dim2_query", "dim3_query"), queries))},
            ensure_ascii=False, separators=(",", ":"))
    )
