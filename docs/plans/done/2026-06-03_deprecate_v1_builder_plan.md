# Deprecate the v1 pair builder

**Status: IMPLEMENTED** (2026-06-03) — v1 removed (−1,260 lines); harness `check` 8/8 bit-exact.

Retire the v1 dataset builder in `src/datasets/dataset_segment_pairs.py`. v2 has been the
default since 2026-05-11 and is the only path any bundle reaches. This is plan **P5** of
`docs/plans/2026-06-03_dataset_split_refactor_plan.md`, scoped out as its own change.

## Why it's safe (verified 2026-06-03)
- `conf/dataset/default.yaml: pair_builder_version: v2` — every bundle inherits v2.
- No bundle sets `pair_builder_version: v1`, and no bundle sets the v1-only knob *values*
  (`pair_mode: unordered`, `canonicalize_pair_orientation: true`, `allow_same_func_negatives: true`).
- The 6 v1 builder functions form a closed call-graph; their only external entry is the
  `elif PAIR_BUILDER_VERSION == 'v1'` dispatch branch.
- No code imports the v1 functions (scripts invoke the CLI as a subprocess); no test references v1.
- bunya (`# STATUS: not maintained`) inherits v2 and doesn't use v1 behavior.

## What v2 gave over v1 (context)
Coverage-first negatives, within-split `pair_key` dedup, per-sequence exposure tracking,
metadata-axis annotations — plus everything since (regimes, seq_disjoint/cluster_disjoint
routing, metadata_holdout, pair_key_alphabet). All v2-only.

## Change
- Remove the 6 v1 builder functions: `canonicalize_pair_orientation`, `create_positive_pairs`,
  `create_negative_pairs`, `split_dataset`, `save_split_output`, `generate_all_cv_folds`.
- Remove the `elif PAIR_BUILDER_VERSION == 'v1'` dispatch branch.
- Dispatch becomes "v2, else reject": a non-v2 `pair_builder_version` raises a clear retirement
  error (same pattern as the retired `year_train`/`year_test` keys).
- Keep: the CLI (argparse, config load, filtering/enrichment) and the v2 branch. The file stays
  the Stage-3 entry point (scripts + the regression harness invoke it as a subprocess).

## Verification
- `scripts/split_regression_harness.py check` green (8/8) before and after — v1 removal must not
  perturb any v2 path.
- `python -m py_compile` the file (syntax).
- Grep: no remaining references to the removed functions.

## Capability note
v1-only behaviors (unordered pairs, same-function negatives, orientation canonicalization) are
removed. Unused by any bundle; recoverable via git history if ever needed.

## Docs to update
CLAUDE.md (Pipeline table), `.claude/memory.md`. The v2 design spec
(`docs/plans/done/2026-05-11_design_dataset_gen_v2.md`) stays as the v2 contract.
