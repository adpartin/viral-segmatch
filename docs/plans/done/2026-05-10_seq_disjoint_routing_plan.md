# seq_disjoint routing — Plan

**Status: IMPLEMENTED** (2026-05-11)
**Date:** 2026-05-10
**Results:** `docs/results/2026-05-11_exp4a_seq_disjoint_results.md`

## Context

Mode #3 (sequence-level leakage) of the 5-mode taxonomy
(`docs/methods/leakage.md`) is the only mode where the v2
builder is still complicit: the same `dna_hash` appears in different
pairs across train/val/test (see Exp 1 of
`docs/plans/2026-05-07_leakage_diagnostics_plan.md`, table at the top of
that file). On `dataset_flu_ha_na_20260508_171512`, 9.0–14.4% of
val/test DNAs are also in train; for `seq_hash` the share is 24.6–33.0%.

The diagnostics plan sketched two mitigations:

- **strict_dedup** (drop duplicate DNAs across the whole dataset) —
  simpler, more aggressive, drops 50–80% of pairs, breaks DNA-level
  coverage by construction.
- **seq_disjoint routing** (this plan) — partition unique DNAs across
  splits and route each pair into the split where both its DNAs sit.
  Preserves both protein-level and DNA-level coverage; drops zero pairs
  on the current Flu A datasets (see measurement below).

## Algorithm

1. Build the global positive table via `create_positive_pairs_v2`
   (unchanged). The `pos_df['assembly_id_a'].is_unique` invariant still
   holds — each row is one isolate, one positive pair.
2. **Bipartite-component routing.** Build the bipartite graph: nodes =
   `('a', dna_hash_a) ∪ ('b', dna_hash_b)`, edges = unique
   `(dna_hash_a, dna_hash_b)` tuples in `pos_df`. Compute connected
   components via union-find. Each component is indivisible: assigning
   the whole component to one split guarantees no cross-split DNA
   leakage from positives in that component.
3. **LPT-greedy bin-pack.** Sort components by size desc, then by root
   id for deterministic tie-breaking. Assign each component to the bin
   with the largest remaining-capacity deficit (proportional to the
   target ratios — 80/10/10 by default). Components are NEVER split, so
   the routing drops **zero positive pairs** by construction.
4. Per-split positives are then fed to the existing
   `create_negative_pairs_v2` unchanged. Because that sampler operates
   entirely on `pos_df` (not on the global `df` — see its docstring
   line 268-274 of `dataset_segment_pairs_v2.py`), per-split negatives
   inherit the seq_disjoint property automatically: their candidate
   partners are sourced from rows of the per-split `pos_df`, whose
   DNAs are already disjoint from the other splits' DNAs.

Why no extra plumbing is needed:
- Coverage floor scales with per-split `pos_df` (already per-split).
- `forbidden_pair_keys` threading still prevents cross-split neg-neg
  pair_key collisions.
- The cross-split pair_key assertion at line 1402 + the isolate-disjoint
  tripwire at line 1428 remain unchanged.
- `compute_axis_flags` and `compute_exposure_stats` are per-split
  already.

## Component-size measurement (flu_ha_na_neg_regimes, 2026-05-10)

Source: `dataset_flu_ha_na_neg_regimes_20260510_120116`

- 58,826 positives, 54,305 unique HA-DNAs, 51,466 unique NA-DNAs.
- **46,954 connected components.**
- 89.2% are singletons (size 1, 41,898 components).
- Largest component: **178 pairs (0.30%)**.
- Top-100 components account for 4.80% of pairs.
- LPT-greedy bin-pack achieves **80.00 / 10.00 / 10.00** exactly. Zero drops.

The dataset is essentially a forest of tiny stars; nothing forces
intra-component splitting. The routing should be a clean replacement of
the existing isolate-level shuffle-split.

## Files touched

- `src/datasets/_pair_helpers.py`
  + `bipartite_components(pos_df)` — union-find on
    `(dna_hash_a, dna_hash_b)` edges; returns one component-id per row.
  + `seq_disjoint_route_pos_df(pos_df, train_ratio, val_ratio, seed)`
    — LPT-greedy routing; returns `(train_pos, val_pos, test_pos, audit_dict)`.
- `src/datasets/dataset_segment_pairs_v2.py`
  + new `split_strategy_mode` parameter on `split_dataset_v2`.
  + dispatcher branch (random | seq_disjoint).
  + writes `audit_dict` into `duplicate_stats['seq_disjoint_audit']`.
  + `save_split_output_v2` emits `seq_disjoint_audit.json` if present.
- `src/datasets/dataset_segment_pairs.py`
  + extract `dataset.split_strategy.mode` from config; pass to
    `split_dataset_v2`.
  + `_validate_v2_config` rejects unknown modes; `split_dataset_v2`
    rejects `seq_disjoint` + CV-override path.
- `conf/dataset/default.yaml`
  + new `dataset.split_strategy: { mode: random }` block.
- `conf/bundles/flu_ha_na_seq_disjoint.yaml`
  + inherits `flu_ha_na_neg_regimes`; sets `mode: seq_disjoint`.

## Out of scope (this iteration)

- `strict_dedup` mode (slot is reserved in the dispatcher; not
  implemented — seq_disjoint achieves the same scientific test with
  better data retention).
- CV interaction (the routing branch errors clearly if combined with
  CV-override path).
- ESM-2 feature_source variant (one bundle change away once the routing
  is in).
- mmseqs2 / cosine cluster-disjoint routing — see
  `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` for the
  follow-up that builds on this scaffolding.

## Success criteria

1. Stage 3 build completes for `flu_ha_na_seq_disjoint` bundle.
2. `seq_disjoint_audit.json` present and well-formed
   (n_components, achieved_split_pcts, top_10 component sizes).
3. `split_overlap_stats.csv`: every dna_hash `overlap_with_{other_split}`
   row equals 0 (the headline before/after diagnostic).
4. `pair_key` cross-split assertion at line 1402 still passes.
5. Existing v2 mode=random bundles unaffected (regression: rebuilding
   `flu_ha_na_neg_regimes` produces identical pair counts and
   `dataset_stats.json` aggregate fields).
