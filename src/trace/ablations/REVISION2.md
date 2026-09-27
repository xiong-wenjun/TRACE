# Task-conditioned STALE core ablations

Use `configs/ablations/stale_core_qwen200_v2.json` and
`scripts/experiments/run_stale_core_ablation.py`. The five arms and their four
boolean interventions are unchanged; the common task/evidence interface is
revised. The v1 runtime and outputs are preserved separately.

## Why the initial panel was stopped

Of 39 compiled v1 TRACE views, 21 reset; every reset had at least one predicted
requirement whose supporting sources were all ineligible. Thirteen issued views
selected only sessions 0--5. The observation-only planner treated unrelated
historical conversations as current obligations, while evidence prefixes and
generated quotes introduced information loss and parsing failures. These are
additional adapter bottlenecks, not evidence that the original main-table
TRACE implementation has the reported 20% Overall score.

## Information boundary

**The three STALE probe questions define the returning task and are visible
before public selection in v2.** This is a task-conditioned diagnostic, not the
paper's pre-query main-table protocol. All five arms see the same questions,
checkpoint, predicted requirements, candidate pool and budgets. Answer labels,
oracle annotations and scoring rubrics are not provided to preparation,
selection or answering. The judge sees them only during scoring.

This change fixes the scope of the evaluated task, but also changes the
information available to the selector. Therefore a v2-versus-v1 improvement
cannot be attributed to one of the four components. Do not merge their scores
or claim v2 establishes pre-query admission performance.

## Shared evidence preparation

1. Validate the frozen extraction and implicit relation map against source
   records, return boundaries and recorded input hashes. Only state inputs
   are imported; historical answers and scores are never used by the methods.
2. Retrieve ten items per question with a union cap of thirty. The common pool
   includes these items, all raw session anchors, and explicit successors.
   Candidate construction does not use the ablated implicit relations.
3. Expose the full stored evidence in exact 700-character spans. When needed,
   paginate within a 64,000-character prompt limit. Each page nominates at most
   twelve spans per question; consolidation rotates across questions and pages
   under the same character limit instead of privileging early sessions.
4. Qwen returns integer evidence indices only, grouped into one to four required
   evidence needs per question. Within a group, up to eight spans are alternatives
   for the same state attribute or constraint, including its historical values
   and later state changes. Different groups are jointly required.
5. Code binds indices to original source IDs, exact offsets and text. Repeated
   indices are deduplicated without adding evidence. Empty support remains an
   unmet requirement and may trigger reset; it is not a preparation failure or
   fictitious coverage. All generation attempts and validation failures are saved.

The obligation descriptions are the actual task questions. The planner cannot
write free-form answers or asserted facts into the Return View. Predicted support
groups remain model judgments, not ground-truth coverage labels. Binding a quote
to its source does not prove that its meaning supports the predicted requirement.

## Governance and fair interventions

The production governor still enforces authorization, epoch, applicability,
provenance, coverage, signing and admission. Public budgets remain 24 items and
24,000 item-text characters; private budgets remain three items and 3,600
serialized characters. All five arms share these limits.

Supporting spans selected beyond an evidence prefix are retained in the
actor-visible source item. A source is split into individually bounded evidence
fragments; only a fragment containing a bound span covers that span's dependency.
Fragments retain the same logical source for source-level invalidation and
private readmission, and each consumes a public item slot. This preserves the
governor's 4,096-character atomic-item limit as well as the total budget.

For the current-state questions in this protocol, a verified invalidation is
counterevidence against using the historical premise now. A bound resolution
record can cover that evidence need using the surviving successor source and
the verified relation. It does **not** assert an unspecified replacement value
or prove that an arbitrary factual assertion is true. Requirements are unchanged;
their effective support graph is projected through the enabled lifecycle edges.
The resolution contains the historical quote, current evidence and both source
receipts, and counts as a disclosed item. Explicit deletion with no successor,
missing sources and absent requirements still fail closed. Explicit relations
take precedence over implicit ones.

Verified lifecycle relations are attached to their current source items, restoring
the type of information rendered by the native STALE adapter. The implicit ablation
disables implicit exclusion and its relation annotations, while explicit
updates remain. The relevance control retains the same requirements and final
coverage gate. The gate control uses the same pre-gate selection. The private
control preserves the public view and owner isolation.

The common requirement planner does not receive implicit decisions. Its output
and public candidate pool must be identical across arms. A missing-support
diagnostic is implemented but is not part of the natural 200-record launch.
Per-episode views record missing support before selection separately from
coverage failure under a budget. Identical actor inputs share both generation
and judgment caches, so no-op interventions are not compared through extra
sampling noise.

## Validation and launch

Regression tests cover source offsets, unknown indices, scope binding, paging,
missing evidence, explicit/implicit isolation, private ownership, gate bypass,
signatures and budgets. The fixed first three records are used for an end-to-end
smoke; success means valid preparation, five complete outputs per UID and correct
intervention contracts. A favorable accuracy ordering is not a launch criterion.

```bash
python3 scripts/experiments/run_stale_core_ablation.py \
  --mode preflight --config configs/ablations/stale_core_qwen200_v2.json \
  --output-dir /cache/trace-core-study/preflight

python3 scripts/experiments/run_stale_core_ablation.py \
  --config configs/ablations/stale_core_qwen200_v2.json \
  --output-dir /cache/trace-core-study/new-run
```

Launch creates a shared preparation process plus five concurrent arm workers.
Use a fresh output root for this protocol. `summary.json` includes paired metrics,
preparation failures, reset counts, unavailable supports and private re-exposure.
Only full paired coverage permits a complete panel; partial prefixes can be biased
by preparation success. A reset stays in the evaluation and is answered/scored
with empty inherited context. Failure records are preserved for explicit recovery.

`incremental_actor_tokens` covers new planning and answering, including recorded
format retries. It excludes reused extraction and implicit verification costs;
`full_lifecycle_tokens` remains null. Judge costs are separate. Do not compare
these incremental counts with historical full-lifecycle Tokens/episode.
