# Selected benchmark configurations

Each model directory contains at most one JSON per available benchmark.

| Directory | Memora | ManBench | STALE Type II |
| --- | --- | --- | --- |
| `qwen35_122b_a10b` | `memora.json`: Pareto specification, 50% departure | `manbench.json`: balanced-return full panel | `stale_type2.json`: 200-record, six-arm panel |
| `gemini_3_flash` | no model-specific JSON | `manbench.json`: four-core full panel | `stale_type2.json`: Early 64-record four-core stratum |
| `deepseek_v4_flash_0731` | no model-specific JSON | `manbench.json`: MemStrata/MemTX full panel | `stale_type2.json`: Early 64-record four-core stratum |

These different arm sets and splits must not be presented as a matched
cross-model comparison. Middle/Late strata and other panels are in
`../variants/`; their dedicated launchers still cover the three strata.
Memora launchers consume `configs/shared/memora_return_confirmation_manifest.json`.
The Qwen Memora JSON is a study specification, not a directly runnable
ManBench/STALE-style configuration.

These are anonymized derivatives, not byte-identical frozen originals.
Endpoints use official providers and credentials are environment-variable names.
Benchmark selection, arm sets, seeds, and original model IDs are preserved.
The IDs may require an explicit override on a different provider; changing a
model is a new experiment, not exact reproduction. Always use a fresh output
directory. See [../../SUBMISSION.md](../../SUBMISSION.md).
