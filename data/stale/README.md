---
license: cc-by-4.0
---

# STALE: State Tracking And Latent Evaluation

## Dataset Structure

Main fields include:

- `uid`: unique instance identifier.
- `M_old`: earlier user observation.
- `M_new`: later user observation that implicitly invalidates or changes the state implied by `M_old`.
- `explanation`: short description of the underlying state change.
- `probing_queries`: three evaluation queries:
  - `dim1_query`: explicit state validation.
  - `dim2_query`: stale-premise robustness.
  - `dim3_query`: implicit downstream planning or recommendation.
- `haystack_session`: long-context dialogue history used for evaluation.
- `relevant_session_index`: indices of sessions containing the relevant old and new observations.
- `timestamps`: timestamps associated with the dialogue sessions.
- `type`: conflict type.

## License

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Some distractor/noise dialogue sessions included in the packaged haystacks are derived from LongMemEval materials; the original LongMemEval MIT license notice is provided in `LongMemEval_LICENSE`.