# Release notes and provider setup

This Git repository is the named TRACE code release. Its source originated
from a review snapshot; the release adds public entry points, dataset download
and checksum metadata, author attribution, and GitHub documentation.

## Configuration layout

- `configs/models/`: selected model and benchmark configurations.
- `configs/shared/`: frozen dataset, split, and UID manifests.
- `configs/variants/`: additional panels used by maintained launchers.
- `configs/ablations/`: mechanism ablations requiring the documented inputs.

## Provider configuration

Only environment-variable **names**, never real API keys, are stored in JSON.
Copy `.env.example` to `.env` locally and fill the required provider keys.
Launchers source it with tracing disabled. For direct Python invocation, load
your local environment first:

```bash
set +x
set -a
. ./.env
set +a
```

Default endpoints were checked against official provider documentation:

| Provider | Endpoint default | Credential variable |
| --- | --- | --- |
| OpenAI / judge | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| DeepSeek | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` |
| Qwen / embeddings | `https://dashscope-us.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |
| Claude variants | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |

Sources: [OpenAI authentication](https://developers.openai.com/api/reference/overview),
[DeepSeek API](https://api-docs.deepseek.com/),
[Gemini compatibility](https://ai.google.dev/gemini-api/docs/openai),
[Qwen compatibility](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope),
[Claude compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk).

Qwen uses a public US regional endpoint to avoid embedding a workspace ID;
use an endpoint and key from the same region. A local Memobase URL is an
optional memory service, not a model API. Test fixtures use reserved example
domains and dummy credentials; they are not author credentials.

Original experiment model IDs are preserved as records of the intended
models; they are **not** a claim that every ID is available at these official
endpoints. No newer model is silently substituted. Before an online run,
verify the exact actor, judge, and embedding models, API dialect, rate limits,
and transport parameters. Different endpoints/models can change results.

JSON-based ManBench, STALE, STALE ablation, and CUPMem-STALE runners accept
optional `TRACE_ACTOR_*`, `TRACE_JUDGE_*`, and `TRACE_EMBEDDING_*` environment
overrides: `BASE_URL`, `MODEL`, and `API_KEY_ENV`. The last is the **name** of
a credential variable. CLI overrides, where supported, take precedence.
CLI-only runners expose corresponding command-line options; consult `--help`.
Never send a provider key to another provider's endpoint.

## Reproduction boundaries

- Seeds, method sets, splits, benchmark bytes, and core algorithms are retained.
  Directory and provider configuration changes alter configuration hashes; use
  fresh result directories instead of resuming incompatible frozen runs.
- Gemini and DeepSeek selected STALE configurations cover the Early stratum;
  dedicated launchers select their additional Middle and Late variants.
- The Qwen Memora JSON describes a Pareto study. Memora execution uses a shared
  manifest plus CLI settings rather than the ManBench/STALE JSON runner format.
- ManBench is bundled; STALE and Memora are installed from fixed upstream
  revisions by `prepare_data.py`. See [data/README.md](data/README.md).
- Raw model outputs, private runtime environments, and previously frozen
  experiment results are not published in this source release.
- Figure-generation source retains its plotted values; generated figures and
  CSVs can be recreated locally. Check their experimental scope before reuse.

## Validation

```bash
python -m pytest -q
python prepare_data.py --dataset manbench --verify-only
# After downloading all datasets:
python prepare_data.py --dataset all --verify-only
```

The included tests check the public entry points and dataset integrity rules.
They do not rerun model-based experiments or establish benchmark performance.
`SOURCE_MANIFEST.json` records the published source file hashes, excluding
itself and downloaded inputs. Dataset bytes have separate checksum manifests.

The legacy `audit_anonymous_release.py` and `package_anonymous_release.py`
utilities are retained for source-snapshot workflows. They must not be run
against a Git checkout as an anonymity guarantee: this release intentionally
contains named authors and public repository links.
