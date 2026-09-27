"""Task-conditioned evidence requirements with program-bound source excerpts.

Only probe questions and checkpoint evidence cross this boundary. The model
selects local integer indices; it cannot write answers, quotes or source IDs
into the Return View. This protocol is distinct from pre-query compilation.
"""

import json
from collections.abc import Mapping, Sequence

from trace.stale_type2_methods import StaleMemoryItem


def evidence_catalog(items: Sequence[StaleMemoryItem], span_chars: int = 700) -> list[dict]:
    """Expose every character of the shortlisted evidence, in bounded spans."""
    catalog = []
    for item in items:
        for start in range(0, len(item.evidence_quote), span_chars):
            catalog.append({"index": len(catalog), "memory_id": item.memory_id,
                "session_index": item.session_index, "start": start,
                "text": item.evidence_quote[start:start + span_chars]})
    return catalog


def candidate_closure(items: Sequence[StaleMemoryItem], seeds: Sequence[str],
                      relations: Mapping[str, str]) -> tuple[StaleMemoryItem, ...]:
    """Include all available successor evidence, without filtering any arm."""
    index = {m.memory_id: m for m in items}
    chosen = set(seeds)
    if not chosen.issubset(index):
        raise ValueError("unknown retrieval candidate")
    frontier = list(chosen)
    while frontier:
        next_id = relations.get(frontier.pop())
        if next_id in index and next_id not in chosen:
            chosen.add(next_id)
            frontier.append(next_id)
    return tuple(m for m in items if m.memory_id in chosen)


def _load(text):
    text = text.strip()
    if text.startswith("```"):
        text = text[text.find("{"):text.rfind("}") + 1]
    return json.loads(text)


def _indices(value, allowed, limit):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"evidence indices must be an array of at most {limit} integers")
    if any(type(i) is not int or i not in allowed for i in value):
        raise ValueError("evidence index is not in the supplied catalog")
    return list(dict.fromkeys(value))


def _rows(text, field):
    value = _load(text)
    if not isinstance(value, dict) or set(value) != {"requirements"}:
        raise ValueError("expected requirements only")
    rows = value["requirements"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("exactly three requirements, one per question, are required")
    if any(not isinstance(r, dict) or set(r) != {"query_index", field}
           or type(r["query_index"]) is not int for r in rows):
        raise ValueError("invalid requirement fields")
    if sorted(r["query_index"] for r in rows) != [0, 1, 2]:
        raise ValueError("query indices must be 0, 1, 2 exactly once")
    return sorted(rows, key=lambda r: r["query_index"])


def parse_shortlist(text: str, catalog: Sequence[Mapping]) -> list[list[int]]:
    allowed = {e["index"] for e in catalog}
    return [_indices(r["evidence_indices"], allowed, 12)
            for r in _rows(text, "evidence_indices")]


def parse_requirements(text: str, catalog: Sequence[Mapping], queries: Sequence[str]) -> dict:
    index = {e["index"]: e for e in catalog}
    obligations = []
    for row in _rows(text, "evidence_groups"):
        qi = row["query_index"]
        groups = row["evidence_groups"]
        if not isinstance(groups, list) or not 1 <= len(groups) <= 4:
            raise ValueError("one to four evidence groups required; use [] within a group for missing evidence")
        deps = []
        for di, group in enumerate(groups):
            selected = _indices(group, index, 8)
            # Empty support remains an explicit unmet requirement, never success.
            supports = [{"memory_id": index[i]["memory_id"], "quote": index[i]["text"],
                         "span_start": index[i]["start"]} for i in selected]
            deps.append({"id": f"q{qi}_dep{di}", "description": f"Evidence requirement {di + 1} for question {qi + 1}",
                         "critical": True, "supports": supports})
        obligations.append({"id": f"q{qi}", "description": queries[qi], "dependencies": deps})
    return {"obligations": obligations}


def planning_prompt(catalog: Sequence[Mapping], queries: Sequence[str], *, shortlist=False) -> str:
    instruction = (
        "The task is to answer these THREE user questions after return. Identify evidence "
        "requirements for ONLY these questions, not tasks from unrelated past conversations. "
        "Questions can contain outdated or false premises: never treat their claims as facts. "
        "Use the historical evidence below, including later events that change the applicability "
        "of an older fact even when the topics differ. Do not answer the questions. "
        "Do not output prose, quotes, descriptions, or memory IDs. Output integer catalog indices only. "
    )
    if shortlist:
        instruction += (
            "This is one page of evidence. For each question return up to 12 potentially useful "
            "indices from this page, including both historical premises and relevant later changes. "
            "If nothing is relevant, return an empty array. Return JSON: "
            '{"requirements":[{"query_index":0,"evidence_indices":[0,1]},'
            '{"query_index":1,"evidence_indices":[]},{"query_index":2,"evidence_indices":[2]}]}.'
        )
    else:
        instruction += (
            "For each question identify 1 to 4 indispensable evidence needs. Each inner group "
            "lists up to 8 alternative evidence indices for ONE need; different groups are "
            "jointly required. A need concerns a state attribute or constraint, not a fixed "
            "historical value: include evidence of its different values and state-changing events "
            "as alternatives when they address that same need. Do not add requirements for "
            "incidental topics. If a necessary need lacks evidence use an empty inner group. "
            "Return JSON: "
            '{"requirements":[{"query_index":0,"evidence_groups":[[0,1],[2]]},'
            '{"query_index":1,"evidence_groups":[[]]},{"query_index":2,"evidence_groups":[[3]]}]}.'
        )
    # Source IDs are bound in code; the model only needs local indices and dates.
    return instruction + "\n" + json.dumps({"questions": list(queries),
        "evidence": [{k: e[k] for k in ("index", "session_index", "text")} for e in catalog]},
        ensure_ascii=False, separators=(",", ":"))


def catalog_pages(catalog, queries, max_chars):
    pages, page = [], []
    for entry in catalog:
        if len(planning_prompt([*page, entry], queries, shortlist=True)) > max_chars:
            if not page:
                raise ValueError("one evidence span exceeds planner budget")
            pages.append(page)
            page = []
        page.append(entry)
    if page:
        pages.append(page)
    return pages


def merge_shortlists(shortlists, catalog, queries, max_chars):
    """Round-robin across questions/pages; never privilege the first sessions."""
    by_index = {e["index"]: e for e in catalog}
    candidates, seen = [], set()
    for rank in range(12):
        for qi in range(3):
            for page in shortlists:
                if rank >= len(page[qi]):
                    continue
                i = page[qi][rank]
                if i not in seen:
                    seen.add(i)
                    candidates.append(by_index[i])
    selected = []
    for entry in candidates:
        if len(planning_prompt([*selected, entry], queries)) <= max_chars:
            selected.append(entry)
    return sorted(selected, key=lambda e: e["index"])


def render_evidence(plan: Mapping, items: Sequence[StaleMemoryItem]) -> dict[str, str]:
    """Bind selected spans to exact source offsets and preserve them for the actor."""
    index = {m.memory_id: m for m in items}
    spans = {}
    for o in plan["obligations"]:
        for d in o["dependencies"]:
            for link in d["supports"]:
                mid, start, quote = link["memory_id"], link["span_start"], link["quote"]
                if mid not in index or index[mid].evidence_quote[start:start + len(quote)] != quote:
                    raise ValueError("evidence span does not match its bound source offset")
                spans.setdefault(mid, {})[start] = quote
    return {mid: "\n[... intervening source text omitted ...]\n".join(q for _, q in sorted(parts.items()))
            for mid, parts in spans.items()}
