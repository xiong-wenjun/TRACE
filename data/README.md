# Benchmark data

The maintained dataset directories are `manbench`, `stale`, and `Memora`.
Each directory contains `UPSTREAM.json` and `FILES.sha256.json`, recording the
source revision and the exact input bytes used by the existing TRACE source
snapshot. These are data-integrity records, not experimental result claims.

| Dataset | Revision | License | Distribution |
| --- | --- | --- | --- |
| [ManBench](https://github.com/bluedream02/Mandela-Effect) | `2688076944fb026cebc894c25f600627e325b8a8` | MIT | 20 task JSON files and license included |
| [STALE](https://huggingface.co/datasets/STALEproj/STALE) | `617c51dc200b5ab09970834144c7e51c77959af0` | CC-BY-4.0; LongMemEval MIT notice retained | Official full JSON downloaded from the pinned revision |
| [Memora](https://github.com/geniesinc/Memora) | `a6493188efc836d6511ed5e4163fe3ba87da30ff` | Apache-2.0 | Complete `data/` tree downloaded from the pinned revision |

From the repository root:

```bash
python prepare_data.py --dataset all
python prepare_data.py --dataset all --verify-only
```

STALE's full JSON is 305,908,212 bytes. It is excluded from Git, as is Memora's
downloaded data tree. The downloader uses pinned upstream sources and checks
each installed file against the recorded SHA-256 digest. Existing mismatching
files cause an error and are never silently replaced. No API key is required.

The original upstream README and licenses remain in each dataset directory.
Upstream README links describe the full upstream repository and may reference
files outside this data-only distribution. TRACE-specific commands are in the
root README. The historical STALE projection receipt is not distributed here:
this release uses the complete official haystack, not that projection.

ManBench configurations use `data/manbench/bbh_all_small`. STALE configurations
use `data/stale/T1_T2_400_FULL.json`, and Memora runners use `data/Memora/data`.
