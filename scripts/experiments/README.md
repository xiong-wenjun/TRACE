# Experiment entry points

This directory contains Python experiment entry points for materialization, generation, scoring, and reporting.
Shell launchers and retry/resume scripts are maintained separately under scripts/launchers/.
Method implementations do not live here: lifecycle policies, published governance adapters, and
prompt defenses are registered under src/trace/return_methods/.

## Memora

- `generate_memora_mas_return.py`: materialize one shared predicted-state
  checkpoint and generate independent method forks.
- `score_memora_generations.py`: independently score frozen generations.
- `report_memora_return_statistics.py`: paired, persona-period clustered
  reporting for a completed panel.

## STALE Type II

- `materialize_stale_type2_mas_return.py`: build metadata-only MAS Return
  sidecars without modifying official records.
- `run_stale_type2_mas_return.py`: execute natural Early/Middle/Late strata.
- `audit_stale_type2_mas_return.py`: verify leakage, coverage, and contracts.

## ManBench-Balanced-Return

- `run_manbench_return_batch.py`: run sharded Old-Valid/Old-Stale episodes.
- `report_manbench_return_statistics.py`: report balanced and mixed metrics.

## CUPMem system baseline

The implementation is exposed as
`trace.return_methods.semantic_state.cupmem`; all paper-facing output
uses the name `CUPMem`.

- `run_cupmem_memora.py`: full write/query pipeline at 25%/50%/75%.
- `run_cupmem_stale.py`: official Type-II sessions and official Judge.
- `run_cupmem_manbench.py`: frozen common Qwen Return episodes.
- `scripts/launchers/launch_cupmem_qwen_all_benchmarks.sh`: resume-safe three-benchmark
launcher that runs only CUPMem. `cupmem_full` is its stable historical
artifact identifier, not a paper-facing method name.

The commit pin, upstream-code audit, label isolation, and adapter contract are
documented in `docs/TRACE_BENCHMARKS.md`.

## Runtime rules

- Keep API credentials in ignored runtime environments; never place them in
  commands, configurations, logs, or artifacts.
- Use a new output directory whenever the method set or execution contract
  changes.
- A direct method comparison must share one frozen pre-method checkpoint and
  branch into independent method-specific memory views.
- Raw outputs belong under ignored `results/`; commit only explicit retained
  summaries listed by the result-retention policy.
- Actor calls must not receive hidden benchmark labels, explanations, or
  evaluator rubrics.
