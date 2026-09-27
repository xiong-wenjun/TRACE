# Configuration index

```text
configs/
  models/       one selected configuration per model and available benchmark
  shared/       dataset manifests, UID strata, and development/test splits
  variants/     additional panels still used by launchers or documentation
  ablations/    mechanism ablations; require separately prepared artifacts
```

Start with [models/README.md](models/README.md). Root-level JSON files have been
consolidated; runners, launchers, tests, and internal JSON references use
the new locations. Eighteen unreferenced development/recovery variants are
not included in this release. Three duplicates were merged into selected
model files.

Shared manifests are experimental inputs, not redundant model configurations.
Additional variants differ in methods, transport settings, or data selection
and are not interchangeable with the selected model files.

The release records three pinned benchmark inputs. Run `python prepare_data.py`
to download STALE and Memora; ManBench is bundled:
ManBench under `data/manbench/`, Memora under
`data/Memora/data/`, and STALE Type-II under
`data/stale/`. Their upstream licenses and revisions are
recorded in [`../data/README.md`](../data/README.md).

The distributable configurations use public endpoint defaults and environment
variables instead of private infrastructure. This changes paths and
credential environment-variable names. Configurations therefore do not retain
their original frozen hashes. Benchmark data, UID selection, arm sets, seeds,
and method logic are preserved. Never resume an existing
private run using these rewritten configurations; use a fresh output directory.

See [../SUBMISSION.md](../SUBMISSION.md) for provider setup and release notes.
