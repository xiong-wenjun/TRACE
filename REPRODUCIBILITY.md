# Reproducibility guide

This repository separates benchmark materialization, actor generation,
independent judging, and statistical reporting. Raw API responses and secrets
are local artifacts; frozen protocols, split manifests, code, and retained
summary manifests are version controlled.

## 1. Environment

TRACE requires Python 3.10 or newer. Create an isolated environment and install
the package in editable mode:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,cupmem,figures]'
```

The official CUPMem audit additionally requires:

```bash
python -m pip install -e '.[cupmem]'
```

Copy `.env.example` to an ignored runtime file and set provider credentials in
the process environment. Never add credentials to a configuration, command
line, log, or result artifact.

## 2. Method identity

The validated catalog is `trace.return_methods.RETURN_METHOD_REGISTRY`.
Paper-facing names are `Static`, `Restore`, `Reset`, `MemStrata`, `MemTX`,
`TRACE`, and `CUPMem`. The keys `static_no_churn`, `review`, `latch`, and
`cupmem_full` exist only for frozen-artifact compatibility.

Each implementation has its own module:

```text
return_methods/policies/{static,restore,reset,checkpoint_replay}.py
return_methods/governance/trace.py
return_methods/temporal/memstrata.py
return_methods/transactional/memtx.py
return_methods/semantic_state/cupmem.py
return_methods/prompt_defense/*.py
```

`semantic_state/cupmem_adapter.py` is a diagnostic mechanism adaptation and is
not the official CUPMem system baseline.

## 3. Paired evaluation contract

For governance comparisons, materialize one actor-visible checkpoint per
episode and branch every method from that checkpoint. A new method set, prompt
contract, model, departure policy, or provider transport requires a new output
directory. Actor calls must never receive hidden labels, gold answers,
explanations, relevant-session annotations, or Judge rubrics.

Maintained entry points are indexed in `scripts/experiments/README.md`; frozen
configuration semantics are indexed in `configs/README.md`. Run generation and
judging as separate phases whenever the benchmark runner supports it.

## 4. Validation

Run the included entry-point and data-integrity checks before using this release:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m pytest -q -p no:cacheprovider
```

These checks do not rerun model-based experiments. Before freezing an experiment,
verify the intended configuration, dataset digest, method registry,
coverage count, unresolved failures, and result-contract fingerprint. Report
complete-case results only as diagnostics; formal tables require the declared
coverage target or an explicit missingness analysis.

## 5. Artifact retention

Raw generations, request caches, API logs, and private benchmark material are
ignored by Git. Publication artifacts must follow `results/README.md` and be
listed explicitly in the retention manifest. Vendored upstream code must keep
its license, repository URL, and exact commit in its local provenance file.
