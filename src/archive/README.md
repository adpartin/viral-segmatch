# src/archive — retired scripts

Scripts kept for provenance only. **Nothing here is imported by live code**, and nothing here is
maintained: paths, CLI defaults, and cluster sources are frozen as of the archive date and are
expected to rot. Read them as a record of how a result was produced, not as runnable tools.

A `docs/results/` write-up citing one of these is not a reason to restore it — those docs are the
historical record, and this directory is where the code they cite lives.

## Contents

Archived 2026-07-30, during the bigraph/fragmentation consolidation:

| Script | Why archived | Replacement |
|---|---|---|
| `bigraph_pair_metadata.py` | superseded plotting; carried an aa-only `pair_key_to_metadata` copy | `src/analysis/plot_cc_metadata.py`; helper generalized into `_pair_helpers.pair_key_to_metadata` |
| `bigraph_pair_feasibility.py` | superseded barplot; rode the protein-dedup pair universe (25% nt_cds undercount) and a hardcoded cluster root | `src/analysis/plot_cc_sizes.py` |
| `bigraph_reassort_check.py` | one-off aa-only validation of the cut vs reassortment signal | none — finding recorded in `docs/results/2026-06-04_bigraph_megacc_structure_and_cutting.md` |
| `bigraph_cut_subtype.py` | its `pair_key_to_subtype` was absorbed by the generalized helper | `_pair_helpers.pair_key_to_metadata(..., fields=('hn_subtype',))` |

Archived 2026-07-31 — the aa-only CV harness and its dependencies, retired in favour of the
production 2D-CD path (`dataset_pairs_cc.py` + Stage 4) reading `cc_{source}` artifacts:

| Script | Why archived | Replacement |
|---|---|---|
| `cluster_pair_weight_topk.py` | **misleading**: built ONE aa-keyed universe then looped BOTH alphabets over it, emitting rows *labelled* `nt_cds` computed from 58,826 aa pairs. Also the home of `load_pair_universe` | `cc_cluster_composition.csv` (per-CC cluster composition) + `cc_summary.json` (per-slot floor), from the production universe |
| `cluster_disjoint_cv_experiment.py` | aa-only score-vs-t harness (raised `NotImplementedError` on nt: the nt k-mer cache is contig-level while clusters are CDS-level). Last run 2026-06-07 | production 2D-CD builder + Stage 4 |
| `cluster_disjoint_regime_cv.py` | aa-only per-regime TPR/TNR companion, same guard. Last run 2026-06-08 | production 2D-CD builder + Stage 4 |
| `_cv_sampling.py` | the harness's atom assignment; no production builder ever imported it | `dataset_pairs_cc.assign_atoms_prod` |
| `_cv_features.py` | k-mer feature assembly for the harness only | Stage-4 feature path |
| `verify_cc_reproduction.py` | verified the 2D-CD builder reproduced the harness — moot once the harness is retired | — |

`verify_membership_swap.py` was **deleted outright** (2026-07-31), not archived: a one-shot check
for the 2026-06-05 membership-table swap whose inputs no longer exist (`cds_final.parquet` renamed
to `cds_dna_final.parquet`; `cluster_memb_{aa,nt_cds}.parquet` never built), so it could not run at
all. Recoverable from git history at `4e41dfb` if ever needed.

`_gen1_bigraph.py` holds `build_cluster_bigraph`, the Gen-1 "map hashes to clusters, then build"
adapter, moved here 2026-07-31 once no live code called it.

Archived 2026-08-05, during the fragmentation cleanup:

| Script | Why archived | Replacement |
|---|---|---|
| `_drop_budget_cut.py` | holds `apply_drop_budget_cut` + `DropBudgetExceeded`: the hardcoded-80/10/10 form of `fragment_to_targets`, serving the 2D-CD **holdout** that the K-fold builder superseded. No bundle or config group ever declared the `split_strategy.drop_budget` knob that reached it | `_megacc_cut.fragment_to_targets` for arbitrary targets; `_megacc_cut.fragment_until` for the production K-fold path |

The drop-budget **mechanism** was not retired with it: `fragment_until` caps its cuts with
`max_drop_frac`, wired as `split_strategy.edge_cut.max_drop_frac`, and that is what the production
2D-CD path uses. Its wiring (`split_strategy.drop_budget` through `dataset_segment_pairs.py` →
`dataset_segment_pairs_v2.py` → `_split_helpers.cluster_disjoint_route_pos_df`) was removed at the
same time, so the config key no longer does anything.

**The `load_pair_universe` caveat** applies to everything above that uses it: by default it dedups
on `prot_hash` for **every** alphabet, so its nt_cds numbers count 58,826 HA-NA pairs where the
nt_cds-keyed universe has 79,347 (both via `load_pair_universe` on `cds_dna_final.parquet`; the
production universe, after the v2 filters, is 78,764 — a third, separate quantity). Treat any
nt_cds figure these produced as aa-deduped.

Imports here are repointed when a live symbol they depend on moves — enough to keep them readable
and importable, nothing more. As of 2026-07-31 all 12 modules import cleanly, but that is a
courtesy, not a guarantee: no test covers them and they are not run.
