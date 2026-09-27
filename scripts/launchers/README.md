# Launcher entry points
This directory contains the maintained shell launchers and retry/resume entry points.
Each launcher delegates to a Python runner in scripts/experiments/ and keeps
configuration files under configs/.

## Canonical launchers

- launch_cupmem_qwen_all_benchmarks.sh
- launch_memora_deepseek_three_departures.sh
- launch_memora_qwen_backend_portability.sh
- launch_stale_qwen_cupmem.sh
- launch_stale_qwen_memstrata_memtx.sh
- launch_stale_qwen_six_governance.sh
- launch_stale_deepseek_core.sh
- launch_stale_gemini_core.sh
- launch_manbench_return_formal.sh
- retry_memora_deepseek_failures_after_main.sh

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
