# TRACE benchmarks and experimental reference

## Memora

### Upstream

Upstream: `geniesinc/Memora` at `a6493188efc836d6511ed5e4163fe3ba87da30ff`. Do not upgrade without a re-freeze of all Memora output, re-review of the episode contract, and an updated provenance record.

### Episode contract

- **V** — the set of sessions verified as valid state for the returning agent at the return boundary.
- **N** — the total number of sessions in the departure window assigned to the returning agent before departure.
- **X** — the crossing-point session index that defines the departure boundary.

Each episode record contains the episode ID, persona, departure boundary (X), the full session sequence, the return question, the ground-truth answer, the temporal cluster ID, and a schema-version tag. The benchmark assigns predeparture sessions (up to X) to `agent1` and post-departure sessions to `agent2`--`agent5` in chronological round-robin order.

### Commands

```bash
python scripts/experiments/materialize_memora_return.py --help
python eval.py memora --help
python eval.py memora-score --help
python analyze.py memora --help
```

The maintained Memora entry points are ordinary repository scripts; the
root wrappers preserve their argument contracts. See the root README for a
predicted-state generation and independent-scoring example.

### Primary panel statistics

137 questions across 30 persona-period clusters. All episodes are frozen; the cluster assignments and question order are fixed. Do not add episodes or reorder clusters without a full re-freeze.

---

## STALE Type II

### Data contract

Source file: `data/stale/T1_T2_400_FULL.json`. SHA-256: `5f3ec375179e20e2e94469e018189188f34e2e7e5f21cbecbd99fcfa648c1876`. 200 T2 records are used for all formal results; T1 records are not included in any formal table. Do not modify the source file.

### Departure strata

Early, Middle, and Late are natural departure-position strata derived from the official STALE session order. Stratum assignment is determined solely by a record's departure position in that ordering; no label is consulted. All three strata use the same agent topology and arm set.

### TRACE dependency protocol

TRACE on STALE uses a two-stage dependency resolution.

1. **Scanner** — identifies candidate dependency references in the returning agent's predeparture state.
2. **Grounded verifier** — checks each candidate against absence-period workstate receipts and assigns a validity tag (`valid` / `stale` / `absent`).

The grounded verifier may not consult the official STALE labels, gold answers, or temporal provenance annotations during arm execution. The validity tag is derived solely from evidence visible to the returning agent at the return boundary.

### Diagnostic oracle conditions

Three oracle conditions are run as diagnostics and are excluded from the main result table:

- `oracle_valid` — all dependency candidates pre-tagged as valid.
- `oracle_stale` — all dependency candidates pre-tagged as stale.
- `oracle_absent` — all dependency candidates pre-tagged as absent.

These conditions bound the performance envelope of the scanner and verifier; they are not baselines.

---

## ManBench (balanced return)

### Construction protocol

ManBench-Balanced-Return constructs Old-Valid and Old-Stale scenarios using position parity only: Old-Valid scenarios occupy even positions; Old-Stale scenarios occupy odd positions. The gold label is never consulted during construction. The correct-anchor assignment and the true validity label are therefore logically independent in the dataset.

### Metrics

- **FAA** (Final Answer Accuracy): fraction of final answers matching the reference.
- **VIA** (Valid Information Availability): valid-state availability under the retained artifact contract.
- **IIR** (Invalid Item Rejection rate): fraction of stale items successfully rejected.
- **WSAR** (Wrong-State Admission Rate): admission of invalid workstate under the retained artifact contract; lower is better.

FAA is the primary metric. The overall score is the equal-weight macro average of the Old-Valid and Old-Stale subscores. Micro averaging is incorrect because the two pools may have unequal sizes after stratification.

### Audit invariants

- No scenario appears in both Old-Valid and Old-Stale pools.
- The `answer-old` and `answer-challenger` fields are frozen before the final multiple-choice task is assembled.
- Judge reads only the frozen final answer and the benchmark rubric, never the full generation trace.
- Position-parity assignment is immutable after the dataset freeze.

---

## ManBench provenance stress

Active diagnostic benchmark. Tests four cells that exercise how return governance handles conflicting or missing source receipts.

| Cell | Description |
| --- | --- |
| `independent_majority` | Each active agent produces an independent receipt; no shared provenance. |
| `replicated_majority` | Multiple active agents share a common source receipt. |
| `missing_receipt` | One or more active agents produce state with no attached receipt. |
| `conflicting_receipt` | Active agents produce receipts that contradict each other on shared items. |

Config: `configs/variants/qwen35_122b_a10b/manbench_provenance_stress_qwen.json`
Launcher: `scripts/launchers/launch_manbench_provenance_stress_qwen.sh`
Runner: `src/trace/manbench_provenance.py`

Results from this benchmark are diagnostic only and are not included in the main paper table.

---

## Memory backends

### ScopedMem

`ScopedMem` is the paper/display name for the existing agent-scoped versioned reference store. `--memory-backend scopedmem` selects the same algorithm as the legacy `asvm-vector` option. The legacy default, class imports, and receipt ID `asvm_vector_v1` are retained for old commands and running or resumed experiments. `ScopedMemoryBackend` is an additional public class alias.

Backend portability (selecting among ScopedMem, Mem0, Memobase, A-MEM) is currently supported only in the Memora 50% departure experiment. STALE and ManBench runners do not expose native backend selection.

### Native backend contract

Each native driver implements `reset`, `write`, `list`, `import_entries`, `search`, and `record`. Native memory formation and ranking remain in the selected system. TRACE's shadow bank stores identity, epoch, and workstate receipts only; it never stores fallback retrieval text.

A complete native checkpoint is frozen at the first return branch and reused across all arms. Drift between branches is an error. Source namespaces are quarantined after departure.

A merged record is admissible only if all its recorded source workstates are admitted and provenance is complete. Mem0 and Memobase do not expose a complete formation read set, so the ledger conservatively unions all successful inputs in the namespace, including buffered writes that emitted no profile. A-MEM follows the same conservative policy. This is an upper bound on source dependence and may reduce retention compared with exact lineage.

Ledger receipts persist separately from native text. Checkpoint import must preserve text, source IDs, and record coverage without repeating native formation. Malformed output, ambiguous mutation failure, truncated snapshots, and untracked text changes are errors, not empty successful memory. A failed namespace must be reset before reuse.

Storage: native SQLite, Chroma, and Qdrant state must be placed on local disk (on a Linux host, `/cache`), not on a shared NFS result directory. Source receipt writes retain atomic replace where supported; EPERM/EXDEV triggers a narrow direct-write/fsync fallback. Interrupted or malformed JSON remains an error on restart.

| Backend | Formation and retrieval | Checkpoint import |
| --- | --- | --- |
| ScopedMem | Existing scoped/versioned reference store | Existing immutable branch implementation |
| Mem0 | `mem0ai==2.0.19`, native `add(infer=True)` and `search`; reconcile full before/after state including updates/deletions | `add(infer=False)`, text/coverage validation |
| Memobase | Official REST API at pinned server revision; insert chat then synchronous flush; query-conditioned native profile ranking | Direct profile insertion retaining native topic/subtopic attributes |
| A-MEM | Pinned upstream `ceffb860...`, native analysis, linking/evolution and `search_agentic`, scoped Chroma | Copy notes, embeddings, and native index; remove dangling links; no LLM or re-embedding |

A-MEM compatibility changes and fidelity limits are in `src/trace/vendor/amem_official/ADAPTER_NOTES.md`; its MIT license and upstream hashes are retained. Memobase POSTs are not retried after uncertain transport completion. Snapshot profile retrieval has no token cap; ranked actor retrieval retains its separate token cap. Mem0 snapshots fail above 1,000 records rather than silently truncating.

### Smoke tests

```
python -m trace.memory_backends.smoke --backend scopedmem
python -m trace.memory_backends.smoke --backend mem0
python -m trace.memory_backends.smoke --backend memobase
python -m trace.memory_backends.smoke --backend amem
```

---

## CUPMem

Paper name: CUPMem. Stable artifact key: `cupmem_full`. Upstream: `icedreamc/STALE` at commit `ea7d391103a151927cd29d2f01d87597a782bdcb`. Vendored at `src/trace/vendor/cupmem_official`. The local provenance file retains the license, repository URL, and exact commit.

CUPMem is used as a complete, unmodified upstream memory system with infrastructure adapters for the TRACE harness. Do not describe it as a mechanism reimplementation or a prompt-defense method.

Pipeline adapter: `src/trace/return_methods/semantic_state/cupmem.py`. The historical lightweight candidate-pool adapter (`cupmem_adapter.py`) is registered as `CUPMem-Adapted (diagnostic)` and is excluded from formal paper tables. `cupmem_full.py` is a legacy import shim only.

CUPMem is included in the STALE native audit panel. Its formal artifact key is `cupmem_full` in all output files and the retention manifest. The legacy redirect `docs/CUPMEM_FULL_BASELINE.md` — which read "The canonical method contract is now documented in `CUPMEM_BASELINE.md`" — has been removed; this file is the canonical source.

---

## Experiments and leakage rules

### Freeze rules

- Benchmark data, episode assignments, cluster IDs, and question order are frozen before any generation run.
- Actor prompts, Judge rubrics, and arm configs are frozen per run; changes require a new output directory.
- No run output may be relabeled or merged with a run from a different config.
- The retention manifest is generated by `scripts/evaluation/verify_retained_results.py` and is never written by hand.

### Leakage prohibitions

- Actors must never receive benchmark labels, gold answers, explanations, relevant-session annotations, or Judge rubrics.
- Raw generations, request caches, API logs, and private benchmark material are not committed to the repository.
- Judge isolation must be declared in the `mas_execution` receipt for every formal run.
- `private/runtime/*.env` is in `.gitignore` and must never be copied into repository artifacts.

### Actor model disclosure

Primary Memora panel (137 questions, 30 clusters):

| Departure boundary | Actor model |
| --- | --- |
| 25% | `gpt-5.6-sol` |
| 50% | `gpt-5.6-terra` |
| 75% | `gpt-5.6-sol` |

All three identities must be disclosed in the paper. Do not describe all three conditions with a single actor name.

---

## Human evaluation

### Protocol

288 yes/no judgments in the frozen evaluation packet. Two raters evaluate each item independently before discussion.

### Reporting

- Inter-rater agreement: Cohen's kappa (primary statistic), reported with confidence intervals.
- Ordinal correlation: Spearman where applicable.
- F1 is not appropriate for overall proposal quality judgments; use accuracy or percent agreement.
- Do not report only raw percent agreement without kappa.

### Packet integrity

The 288-item packet is frozen. Items may not be added, removed, or relabeled after the freeze. Any deviation requires a new packet ID and a fresh run by both raters.
