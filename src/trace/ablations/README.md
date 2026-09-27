# TRACE core mechanism study

**Current protocol: [Revision 2](REVISION2.md).** The text below documents the
preserved observation-only v1 protocol, whose run was stopped for adapter
diagnosis. It is retained for reproduction, not as the current launch guide.

This opt-in package evaluates the complete return-governance chain on STALE
Type II. It does not alter the production STALE runner or overwrite historical
results. Its full TRACE reference must be rerun alongside its four controls;
it is a **new STALE-derived protocol**, not the historical main-table protocol.

| Arm | Only intervention |
| --- | --- |
| `trace` | All four mechanisms enabled |
| `without_implicit_invalidation` | Remove implicit invalidation relations; keep explicit replay and all item gates |
| `without_obligation_selection` | Relevance ordering instead of coverage gain; keep mandatory declarations, budgets and coverage gate |
| `without_complete_view_gate` | Allow the same nonempty partial selection; retain all item, identity, signature and budget checks |
| `without_source_bound_private_readmission` | Permit the owner's departure memories regardless of public source admission; public view and private retrieval budget unchanged |

`core.py` defines immutable interventions. `selection.py` specializes only the
selector of the existing signed-view governor. `stale.py` maps observations,
requirements and private memory to existing governance primitives.

## Shared preparation and information boundaries

The configured completed Qwen run supplies frozen observation extraction and
implicit invalidation decisions. Import validates each source record, return
boundary, candidate pool, source digest and aggregate extraction digest. Only
these upstream state fields are imported; historical answers and scores are
never supplied to planning, admission or answer generation.

One Qwen planning call sequence proposes up to six ongoing obligations from
all chronological session anchors, before probe questions are exposed. Every
dependency requires a source ID and a literal supporting quote. Requirements
and alternative support edges are fixed across the five arms. They are model
predictions, **not independent ground-truth coverage labels**. Source receipts
authenticate their bindings; they do not certify semantic correctness.

The planner sees the first 300 statement characters and first 700 evidence
characters of each session anchor, uniformly across all sessions. This bounded
representation is part of the new protocol. Relevance scores use these fixed
obligation descriptions, not probe questions. Public evidence uses the common
700-character quote representation. The governor preserves its item-level
authorization, epoch, applicability and bound-receipt predicates.

Private memory contains only the returning agent's actual departure observations
and extracted records, bound to their original workstate IDs. These are stored
observations, not fabricated reflections or newly measured agent trajectories.
The existing private-memory implementation enforces owner isolation, quarantine,
forking and retrieval. Probe questions become visible only after public admission,
for private retrieval and final answering.

Default budgets are 24 public items / 24,000 public item-text characters and
3 private items / 3,600 serialized private characters. These are fixed before
outcome inspection, not dynamically enlarged to ensure coverage. A RESET supplies
empty inherited context and is still answered/scored on all three probes. This
keeps every example in the evaluation denominator.

The optional `missing_support` condition removes all source supports for the
first declared critical dependency while retaining the original requirement.
It is a separately named controlled diagnostic, never pooled into natural results.
Ordinary episodes may not activate the gate. The per-arm views preserve raw
selection, coverage, public IDs and private re-exposure IDs so non-interventions
are reported as such rather than interpreted from generation noise.

## Execution

Use `configs/ablations/stale_core_qwen200.json` with
`scripts/experiments/run_stale_core_ablation.py`. Existing judge credentials are
loaded externally; no credentials belong in this package or result manifests.

```bash
python3 scripts/experiments/run_stale_core_ablation.py \
  --mode preflight --config configs/ablations/stale_core_qwen200.json \
  --output-dir /cache/trace-core-study/preflight

python3 scripts/experiments/run_stale_core_ablation.py \
  --config configs/ablations/stale_core_qwen200.json \
  --output-dir /cache/trace-core-study/run
```

Launch starts one shared preparation producer and **five concurrent arm
processes**, one per method. Each arm processes ready examples serially. Shared
content-addressed, locked generation caches reuse both answers and judgments
when interventions leave the actual actor input unchanged. A private per-run
signing key is generated with mode 0600. Launch manifests bind the full UID set,
configuration and source code. Use a frozen runtime copy for long runs.

`prepared/` stores shared input. `arms/<arm>/views/` records the actual
intervention before generation, `results/` contains scored episodes, and
`failures/` preserves failures. `summary.json` compares only the intersection
of completed UIDs. `terminal.json` reports process exits and full paired
coverage; a running process or partial prefix is not a completed experiment.
Relaunching with the identical source/configuration resumes missing work.

`incremental_actor_tokens` includes new requirement planning and answering.
Historical extraction/invalidation costs are reused but **not free**; full
lifecycle tokens are left null until a separate upstream cost audit is merged.
Judge tokens are reported separately. Do not plot the incremental count as
`Tokens/episode` alongside historical full-lifecycle costs.

## Validation

`tests/test_stale_core_ablations.py` checks isolated flags, explicit replay,
same-selection gate bypass, relevance selection retaining coverage enforcement,
same-public-view private bypass, owner isolation, context budgets, signed-view
validation, literal support binding and absence of probes in planner inputs.
