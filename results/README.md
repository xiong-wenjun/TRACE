# Local result artifacts

Raw model generations, API logs, caches, scored episodes, and retry shards are
local research artifacts and are ignored by Git. They can be large and may
contain provider metadata, so do not add them with `git add -f`.

Before a paper release, select the canonical immutable run directories and
generate `formal/RETENTION_MANIFEST.json` with schema
`return_experiment_retention_manifest_v1`. Each asset must record its relative
path, file count, byte size, and deterministic tree SHA-256. Verify it with:

```bash
PYTHONPATH=src python3 scripts/evaluation/verify_retained_results.py
```

No retention manifest is fabricated automatically: its presence is a claim
that the listed local trees are the publication artifacts and have been
reviewed and frozen.
