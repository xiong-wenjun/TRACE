# TRACE: Governing Memory Validity in Evolving Multi-Agent Systems

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for **TRACE**, a memory-governance method for agents returning to an
evolving multi-agent system.

**Wenjun Xiong, Shengtao Zhang, Shangding Gu, Bo Tang, Zhiyu Li, Feiyu Xiong,
Ying Wen, Muning Wen**

Correspondence: [Ying Wen](mailto:ying.wen@sjtu.edu.cn) and
[Muning Wen](mailto:muningwen@sjtu.edu.cn).
Code contact: [Wenjun Xiong](mailto:xiongwenjun@sjtu.edu.cn).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)
- [Result Analysis](#result-analysis)
- [Repository Structure](#repository-structure)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Overview

An agent's memory can remain relevant after the state it describes has changed.
Resetting the agent loses useful history; restoring all of its memory can
reintroduce invalid assumptions. TRACE governs which memory may be used when an
authenticated agent returns under the same principal and role.

TRACE checks authorization, temporal validity, applicability, and provenance,
then selects a bounded Return View that covers the agent's current obligations.
Private episodic memory is readmitted only when its source workstate survives
these checks. The mechanism and its scope are described in
[TRACE_METHOD.md](docs/TRACE_METHOD.md).

| Benchmark | Evaluation setting | Data location |
| --- | --- | --- |
| ManBench-Return | Old-Valid and Old-Stale conditions in multi-agent interaction | `data/manbench/` |
| STALE Type II | Implicit invalidation through state changes | `data/stale/` |
| Memora | Explicit updates and forgetting in conversation histories | `data/Memora/` |

## Installation

```bash
git clone https://github.com/xiong-wenjun/TRACE.git
cd TRACE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10 or newer is required. Batch launchers require Bash 4+; the Python
entry points can be used directly. Native memory-backend experiments have
additional dependencies:

```bash
python -m pip install -e '.[memory-backends]'
```

Set provider credentials through the environment:

```bash
cp .env.example .env
# Fill in the variables required by your actor, judge, and embedding provider.
set +x
set -a
source .env
set +a
```

Model IDs and endpoint settings are recorded in the selected configurations.
Use the exact models required by the intended experiment; see
[provider configuration](SUBMISSION.md#provider-configuration) for overrides.

## Quick Start

ManBench inputs are included in Git. Download the pinned STALE and Memora
inputs before evaluating those benchmarks:

```bash
python prepare_data.py --dataset manbench --verify-only
python prepare_data.py --dataset stale
python prepare_data.py --dataset Memora
```

The downloader verifies file sizes and SHA-256 hashes and refuses to replace
existing files with different content. Dataset licenses, source revisions, and
checksums are recorded under [data/](data/README.md).

Explore the available options without making model requests:

```bash
python eval.py --help
python eval.py manbench --help
python eval.py stale --help
python eval.py memora --help
python analyze.py --help
python -m pytest -q
```

## Evaluation

### ManBench-Return

Run a small evaluation with the selected Qwen configuration:

```bash
python eval.py manbench \
  --config configs/models/qwen35_122b_a10b/manbench.json \
  --max-episodes 2 \
  --output-dir results/manbench_quickstart
```

Use `--actor-model`, `--actor-base-url`, and `--actor-api-key-env` to select
another available actor explicitly. Remove `--max-episodes` for the configured
full panel. A small run is a smoke check, not a paper result.

### STALE Type II

```bash
python eval.py stale \
  --config configs/models/qwen35_122b_a10b/stale_type2.json \
  --limit 2 --workers 1 \
  --output-dir results/stale_quickstart
```

The selected Qwen configuration uses the official 200-record T2 split. Other
model configurations may select different strata; see
[configs/models/README.md](configs/models/README.md) before comparing them.

### Memora

Generation and judging are separate phases. Set `TRACE_ACTOR_BASE_URL`,
`TRACE_ACTOR_MODEL`, `TRACE_ACTOR_API_KEY`, `TRACE_JUDGE_BASE_URL`,
`TRACE_JUDGE_MODEL`, and `OPENAI_API_KEY` for the intended providers.

```bash
python eval.py memora \
  --manifest configs/shared/memora_return_confirmation_manifest.json \
  --state-mode predicted --predicted-departure-fraction 0.50 \
  --base-url "$TRACE_ACTOR_BASE_URL" --model "$TRACE_ACTOR_MODEL" \
  --api-key-env TRACE_ACTOR_API_KEY \
  --limit 2 --output-dir results/memora_quickstart

python eval.py memora-score \
  --manifest configs/shared/memora_return_confirmation_manifest.json \
  --input-dir results/memora_quickstart \
  --limit 2 \
  --judge-base-url "$TRACE_JUDGE_BASE_URL" --judge-model "$TRACE_JUDGE_MODEL" \
  --judge-api-key-env OPENAI_API_KEY
```

The root entry points forward arguments to the maintained experiment scripts.
Frozen full-panel commands, departure settings, baselines, and backend options
remain in [scripts/experiments/](scripts/experiments/README.md) and
[scripts/launchers/](scripts/launchers/README.md).

## Result Analysis

```bash
python analyze.py manbench --output-dir results/manbench_quickstart
python analyze.py memora results/memora_quickstart
```

STALE writes its summaries during evaluation. Memora analysis requires complete
matched method outputs; missing or incomplete runs must not be reported as a
completed comparison. Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the paired
contract and [results/README.md](results/README.md) for artifact retention.

## Repository Structure

```text
TRACE/
├── README.md                 Overview, installation, and examples
├── eval.py                   Benchmark evaluation and Memora scoring
├── analyze.py                Existing statistical report entry points
├── prepare_data.py           Pinned dataset download and verification
├── requirements.txt          Standard evaluation dependencies
├── configs/                  Model settings, frozen protocols, and splits
├── data/
│   ├── manbench/             Bundled ManBench inputs and notices
│   ├── stale/                STALE metadata; full data downloaded on demand
│   └── Memora/               Memora metadata; full data downloaded on demand
├── src/trace/                Governance, memory backends, and adapters
├── scripts/                  Experiments, launchers, and evaluation utilities
├── tests/                    Public entry-point and dataset integrity checks
├── docs/                     Method and benchmark documentation
├── paper/                    Figure-generation source
└── results/                  Local generated outputs (ignored by Git)
```

[CODEBASE_MAP.md](CODEBASE_MAP.md) maps the underlying implementation modules.

## Acknowledgements

We thank the authors of [ManBench / Mandela-Effect](https://github.com/bluedream02/Mandela-Effect),
[STALE](https://huggingface.co/datasets/STALEproj/STALE), and
[Memora](https://github.com/geniesinc/Memora) for their benchmark releases.
The repository presentation follows the simple entry-point and documentation
style of Mandela-Effect. Vendored CUPMem and A-MEM code retains its original
licenses and commit provenance under [src/trace/vendor/](src/trace/vendor/).

## Citation

```bibtex
@misc{xiong2026trace,
  title  = {TRACE: Governing Memory Validity in Evolving Multi-Agent Systems},
  author = {Wenjun Xiong and Shengtao Zhang and Shangding Gu and Bo Tang and
            Zhiyu Li and Feiyu Xiong and Ying Wen and Muning Wen},
  year   = {2026},
  url    = {https://github.com/xiong-wenjun/TRACE}
}
```

Machine-readable software citation metadata is available in [CITATION.cff](CITATION.cff).

## License

TRACE code is released under the [MIT License](LICENSE). Third-party code and
datasets retain their own licenses; see [data/README.md](data/README.md) and
the vendored notices.
