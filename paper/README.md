# Paper artifacts

This directory holds artifacts that belong to the paper itself, as opposed to
`docs/`, which documents the method, benchmarks, and execution protocols.

## `figures/`

Figure-generation source, with figure-level values embedded in the script.
Generated CSV and binary outputs are not included in the source release.
The script writes both the CSV and the PDF/PNG pair
into the directory it is given, so regenerating in place is:

```bash
python3 paper/figures/generate_qwen_stale60_cost_figures.py \
  --output-dir paper/figures
```

It needs `matplotlib` and `numpy`, which are not part of the `dev` extra:

```bash
python -m pip install matplotlib numpy
```

## `human_review/`

Blinded human-review packets. These are generated on demand by the export
tools in `scripts/evaluation/` and are not tracked in Git. See
`docs/TRACE_BENCHMARKS.md` for the annotation and
Judge-calibration protocol.
