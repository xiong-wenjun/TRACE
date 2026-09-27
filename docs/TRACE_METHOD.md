# TRACE method reference

## Claim boundary

The treatment is a single authenticated RETURN event: the old workload instance is absent, the team and task evolve, and a fresh instance of the same principal is readmitted. Instance removal and creation are internal mechanics, not separate experimental events. Successor inheritance and REPLACE are out of scope.

## TRACE

TRACE (Temporal Return Admission under Context Evolution) compiles a bounded, signed workstate view using:

1. principal and role authorization;
2. epoch freshness and supersession checks;
3. task and purpose applicability;
4. provenance-bound state references;
5. obligation-aware selection under item and character budgets;
6. fail-closed reset or block behavior when critical coverage is impossible.

The Router view contains role workstate, not raw cross-principal private memory, tools, an expert plan, or an answer. Each principal has a separate episodic bank. At RETURN, agent1's quarantined bank is forked into the paired arms, and a private experience is readmitted only when its bound source workstate is inherited by that arm. Validity is therefore a hard gate before private-memory relevance retrieval; utility learning is not used.

## Main estimands

Memora estimates state retention and stale-state rejection under temporal departure. Fluent proposal quality remains a secondary endpoint. A positive state-use result does not imply TRACE improves generic innovation, safety, or feasibility scores.

## Baselines

- `static_no_churn`: raw current state; retained as a diagnostic.
- `static_compact`: current state with TRACE's budget and serializer.
- `reset`: no inherited or reacquired state.
- `reset_rebrief`: reset, then fixed-order retrieval of current public state under the same budget.
- `restore_old`: departure snapshot only.
- `validity_filter_only`: valid return candidates in frozen order without the obligation selector.
- `full_merge`: old and new state concatenated before the common prompt limit.
- `trace`: full validity checks plus obligation-aware compact selection.

The fairness claim rests on comparisons with `static_compact`, `validity_filter_only`, `reset_rebrief`, and `full_merge`, not on RESET alone.

## Agent topology

Memora, STALE Type II, and ManBench use one paper-facing task-agent topology. Benchmark-native data, departure boundaries, questions, and scorers remain unchanged.

| Component | Responsibility | Counted as task agent |
| --- | --- | --- |
| `agent1` | Returning Agent; owns predeparture state and directly answers the final task after return | yes |
| `agent2`--`agent5` | Active Agents; independently process absence-period work | yes |
| Lifecycle Scheduler | Deterministically assigns work, records lifecycle events, and freezes the checkpoint | no |
| Governance verifier | Resolves a ManBench conflict without benchmark labels | no |
| Judge | Reads hidden benchmark rubrics only after the answer is frozen | no |

The frozen topology is `N_task=5`, `D=1`, and `A=4`. All methods fork from one immutable post-absence checkpoint. Only the return-memory governance method may differ between arms. No separate task-level Router or Aggregator produces the final answer.

### Benchmark bindings

- Memora assigns sessions up to the preregistered 25%, 50%, or 75% departure boundary to `agent1`; later sessions are assigned exactly once to `agent2`--`agent5` in chronological round-robin order.
- STALE Type II departs immediately after the official old-evidence session and returns after all 50 official sessions, before the three official probes. Early/Middle/Late are natural departure-position strata.
- ManBench uses two predeparture logical steps (question and correct anchor), then one independent evidence-generation step for each of `agent2`--`agent5`. `agent1` returns after the four evidence steps and before the final multiple-choice task. Four agents cover the five native social roles; `agent5` receives the last two roles.

### Logical independence versus physical concurrency

The four Active Agents have separate identities, prompts, assignments, and receipts; they do not share private context. Calls inside one episode currently run in deterministic serial order, followed by a synchronization barrier before checkpointing. Parallelism is available only across episodes or external shards. Results therefore claim **logically independent active agents**, not simultaneous wall-clock model execution.

Every result persists a `mas_execution` receipt containing the complete work assignment, lifecycle boundary, checkpoint digest, task-agent roster, execution mode, and Judge-isolation declaration.

## Method interfaces

The code and paper separate three different intervention layers. A method may appear in only the panel appropriate to its layer.

| Paper name | Stable arm | Layer | Formal use |
| --- | --- | --- | --- |
| Static | `static` / `static_no_churn` | No lifecycle governance | Unified governance table |
| Restore | `restore_old` | Restore-only policy | Unified governance table |
| Reset | `reset` | Destructive reset policy | Unified governance table |
| MemStrata | `memstrata` | Temporal freshness governance | Unified governance table |
| MemTX | `memtx` | Shared-memory transaction governance | Unified governance table |
| TRACE | `trace` | Evidence-grounded re-entry governance | Unified governance table |
| CUPMem | `cupmem_full` (legacy artifact key) | Complete memory system | STALE native audit |
| No Defense | `no_defense` | Answer-time control | ManBench prompt-defense panel |
| Cognitive Anchoring--Return | `cognitive_anchoring` | Prompt defense | ManBench prompt-defense panel |
| Source Scrutiny--Return | `source_scrutiny` | Prompt defense | ManBench prompt-defense panel |

The historical `cupmem` arm is a lightweight candidate-pool adapter registered as `CUPMem-Adapted (diagnostic)` and excluded from formal paper tables. It must not be reported as CUPMem.

### Unified governance fairness contract

- Every governance method branches from the same frozen pre-method candidate pool.
- Actors cannot see benchmark labels, gold answers, explanations, or Judge rubrics.
- MemStrata and MemTX retain their exact public names in tables; the artifact records state that they are mechanism reimplementations.
- TRACE consumes the same actor-visible evidence and cannot use an oracle invalidation label.
- Changing the arm set or prompt contract requires a new output directory.

### ManBench answer-time contract

`No Defense`, `Cognitive Anchoring--Return`, and `Source Scrutiny--Return` all receive the identical raw two-candidate view: agent1's independently produced predeparture answer (`answer-old`), the absence-period group challenger (`answer-challenger`), and the same current question and choice order. None of these three arms modifies memory eligibility, so final-answer accuracy is their primary metric. VIA, IIR, and WSAR denominators are recorded as zero and must be shown as `--` in the paper panel.

The frozen Qwen configuration is `configs/variants/qwen35_122b_a10b/manbench_balanced_return_backend_qwen_prompt_defense_formal.json`.

### Implementation fidelity

| Method | Code provenance | Required wording |
| --- | --- | --- |
| MemStrata | Paper-mechanism reimplementation | `MemStrata` with an implementation-fidelity footnote |
| MemTX | Pinned manager/conflict/state/dependency subset | `MemTX` with a mechanism-reimplementation footnote |
| CUPMem | Unmodified pinned upstream package plus infrastructure adapters | `CUPMem` |
| Cognitive Anchoring--Return | Original prompt principles adapted to re-entry | `Cognitive Anchoring--Return` |
| Source Scrutiny--Return | Original prompt principles adapted to re-entry | `Source Scrutiny--Return` |

Never describe the three governance rows collectively as unmodified upstream code. Every formal artifact must retain `implementation_kind` and reference commit metadata where available.

## Code layout

```
src/trace/return_methods/
  base.py                           # shared records and ReturnMethodSpec
  registry.py                       # validated catalog; no method logic
  policies/static.py, restore.py, reset.py, checkpoint_replay.py
  governance/trace.py               # canonical TRACE entry point
  temporal/memstrata.py
  semantic_state/cupmem.py          # official CUPMem pipeline adapter
  semantic_state/cupmem_adapter.py  # diagnostic candidate adapter (excluded from paper tables)
  semantic_state/cupmem_full.py     # legacy import shim only
  transactional/memtx.py
  prompt_defense/no_defense.py, cognitive_anchoring.py, source_scrutiny.py
```

New code imports `TRACE` from `return_methods.governance.trace`, official `CUPMem` from `return_methods.semantic_state.cupmem`, and simple controls from `return_methods.policies`. Historical `latch`, `review`, `static_no_churn`, and `cupmem_full` keys are normalized only by the registry or compatibility modules.

## Router state machine

The Router maintains a task DAG, current obligations, public workstate, membership status, a departure checkpoint, and an immutable post-absence checkpoint. All experimental arms fork from that same checkpoint.

TRACE converts relevant Router items into signed memory items, applies authorization, freshness, applicability, provenance, and disclosure-budget checks, then selects a view covering the returning role's critical dependencies. The fresh returning instance can read only the admitted view.

The benchmark compiles matched fairness controls: current-state compact selection (`static_compact`), validity filtering without obligation selection (`validity_filter_only`), reset followed by same-budget current-state rebrief (`reset_rebrief`), and unfiltered old/current concatenation (`full_merge`). Private memory is principal-confined and never copied to another agent. Hidden evaluator relations and expected outcomes never enter actor prompts.

## Ablation and security controls

### Performance ablations

The required matched panel is `Static / Static-Compact / Reset / Reset-Rebrief / Restore / Validity-Filter-Only / Full-Merge / TRACE`. All arms share actor model, Judge, task, pre-RETURN execution, checkpoint, terminal prompts, output budgets, and prompt-character limits.

### Security controls

The deterministic panel separately tests: wrong principal or role; stale, expired, or superseded items; missing or mismatched provenance receipts; replay after view expiry; tampered signatures; insufficient critical-obligation coverage; weakened freshness, provenance, and fail-closed variants.

Security negative controls diagnose the mechanism and are not benchmark performance baselines. Current runtime enums, schemas, and output fields use `TRACE`; historical `review`/`latch` values are accepted only by the centralized artifact-compatibility parser.
