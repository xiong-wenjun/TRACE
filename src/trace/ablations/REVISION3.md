# Bounded support review and gate revalidation

The completed v2 run and its negative gate result remain immutable. Revision 3
is a development retest of the same inspected 200 records, not independent
confirmation and not the paper's pre-query experiment.

Revision 3 reuses frozen shared preparation, validates source records, task
questions, return boundary, extraction and lifecycle decisions, and imports no
answers or scores. It reviews every question with an empty evidence group, not
a hard-coded UID set or only dimension 3. Other questions are unchanged.

Each unresolved question gets one complete paginated scan of the original
candidate catalog, then one semantic support review of up to 48 nominated and
adjacent spans. Each prompt remains within 64,000 characters. The scan asks for
current personal constraints relevant to the new action, not a historical
occurrence of that exact action. The review can produce at most four jointly
required groups of up to eight alternative source indices. Existing nonempty
requirements remain required. Exact offsets and quotes are bound by code;
review prose remains audit metadata and cannot enter the answer context.

The first smoke revealed that the semantic reviewer conflated grounded personal
context with enough information for an unconditional final recommendation. The
review contract now distinguishes those: established current constraints can
support a correction, conditional answer, or focused clarification. Missing
scenario details are not invented and cannot erase existing state evidence.
Incidental background and the question's own historical premises remain
insufficient. This is a preparation correction; the gate is not relaxed.

Support states are explicit: `supported`, `not_found_after_review`,
`uncertain_after_review`, and `confirmed_no_source`. The last state is permitted
only when the supplied physical catalog is empty. A model's unsuccessful search
cannot establish absolute absence. All unresolved states retain the original
coverage gate. No empty group is silently marked as satisfied.

The gate comparison uses only `trace` and `without_complete_view_gate`, with
identical shared preparation and pre-gate selection, 24 public items / 24,000
characters and three private items / 3,600 characters. No answer-prompt, judge,
source-eligibility, temporal invalidation, signing, or gate threshold changes
are introduced. This isolates the gate comparison *within v3*. Comparing v2 and
v3 is a development comparison of the revised preparation, not an independent
mechanism claim.

New review and answer tokens are reported as incremental cost; reused initial
planning tokens are reported separately. Historical extraction/invalidation
costs are still excluded, and full-lifecycle cost is not claimed. Identical
inputs share generation and judge results within each run.

Synthetic tests cover recovery of evidence beyond a source prefix, correct
binding, unknown indices, absence-vs-uncertainty, retention of existing needs,
prompt budgets, unrelated context, and fail-closed behavior after support is
removed. The engineering smoke contains the first two reset and first two
nonreset records in the original manifest, selected without reference to scores.
Smoke acceptance depends on valid intervention contracts and terminal coverage,
not on whether full TRACE wins.

Run `configs/ablations/stale_gate_qwen200_v3.json` in a fresh output root after
the smoke. The runner now respects the configured arm panel and records the
explicit development scope in its manifest. The original five-arm configuration
continues to select all five arms.
