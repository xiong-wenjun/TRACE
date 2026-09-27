# TRACE code map

## Core mechanism

- `src/trace/return_governance.py`: item validity, provenance receipts,
  obligation-aware selection, signed Return Views, admission, ablations, and
  fail-closed behavior.
- `src/trace/router_return_protocol.py`: task DAG, membership state,
  obligations, shared workstate, departure and post-absence checkpoints, and
  matched Return forks.
- `src/trace/router_return_governance.py`: converts Router state into
  TRACE and fairness-control views.
- `src/trace/private_episodic_memory.py`: principal-confined memory,
  departure quarantine, and source-workstate-bound Return forks.
- `src/trace/return_methods/governance/trace.py`: canonical TRACE method
  identity and public compiler entry point. `trace_method.py` is restricted to
  frozen `review`/`latch` artifact compatibility.
- `src/trace/unified_mas_contract.py`: one five-task-agent Return
  architecture shared by Memora, STALE, and ManBench adapters.
- `src/trace/return_baselines.py`: compatibility facade for historical
  imports; new code imports each method from its own package.
- `src/trace/return_methods/`: modular method registry. Lifecycle
  policies live under `lifecycle/`; prompt-only defenses live under
  `prompt_defense/` and are never reported as memory-governance methods.
  Published governance adapters are separated by mechanism: MemStrata under
  `temporal/`, CUPMem under `semantic_state/`, and MemTX under
  `transactional/`.
- `src/trace/return_methods/semantic_state/cupmem.py`: external
  infrastructure adapter for the
  commit-pinned, unmodified official CUPMem write/query pipeline under
  `src/trace/vendor/cupmem_official/`; its public name is `CUPMem`.
  `cupmem_adapter.py` is diagnostic-only and `cupmem_full.py` is an import shim
  for frozen runners.
- `src/trace/vendor_support/openai_compat.py`: minimal localhost chat
  transport used when the official SDK is unavailable or proxy-incompatible.

## Benchmark adapters

- `memora_return.py`, `memora_predicted.py`, `mas_return_pipeline.py`, and
  `eight_agent_pipeline.py`: Memora materialization, controlled Return arms,
  model execution, and independent scoring.
- `stale_type2_return.py` and `stale_type2_methods.py`: official STALE Type-II
  adaptation, natural departure strata, dependency scan, and six-arm scoring.
- `manbench_return.py`: balanced Old-Valid/Old-Stale disputed Return without
  actor-visible gold anchors.

## Reproduction entry points

Root `eval.py` and `analyze.py` forward to the corresponding scripts below.
`prepare_data.py` installs and verifies the pinned inputs under `data/`.

- Memora: `materialize_memora_return.py`, `generate_memora_mas_return.py`,
  `score_memora_generations.py`, and `report_memora_return_statistics.py`.
- STALE Type II: `materialize_stale_type2_mas_return.py`,
  `run_stale_type2_mas_return.py`, and `audit_stale_type2_mas_return.py`.
- ManBench: `run_manbench_return_batch.py` and
  `report_manbench_return_statistics.py`.
- Human evaluation: the export/report tools in `scripts/evaluation/`.

See `scripts/experiments/README.md` for the maintained entry-point index and
runtime artifact policy.

The active tree intentionally excludes general membership control planes,
framework connector matrices, unrelated AIME/BBH/EvalPlus/LongMemEval
snapshots, raw results, API logs, and downloaded upstream repositories.
