# OOD vs random split: is the OOD split itself hard?

**Status: IN PROGRESS**

Date: 2026-07-21

Scope: build **two datasets from the same positive pairs** at **matched size** -- an **OOD** one
(train/val/test cluster-disjoint) and a **non-OOD** one (train/test share clusters) -- train both,
and read the test-metric gap. That gap is the split's effect (the OOD penalty), isolated from size.
nt_cds HA-NA on the production `_ood` clusters. Spun out of P5 of the edge-cut plan.

Related:
- `docs/plans/2026-07-17_2d_cc_edge_cut_fragmentation_plan.md` -- the **mechanism** this builds on
  (`_megacc_cut.fragment_until` wired into `dataset_pairs_cc.assign_atoms_prod`; P2 DONE). This plan
  is that plan's former P5, grown into its own experiment.
- `docs/results/2026-07-14_cc_ood_threshold_size_decoupling.md` -- "size, not threshold"; this asks
  whether the *split* adds difficulty on top of size.
- `docs/plans/2026-07-08_single_segment_ood_clusters_plan.md` -- the `_ood` clusters we operate on.
- `docs/plans/2026-06-09_cc_dataset_cv_plan.md` -- the 2D-CD builder (`dataset_pairs_cc.py`).
- `docs/methods/glossary.md` -- atom, mega-CC, straddling pair, edge min-cut, within_fold.

---

## 1. Question

`2026-07-14` showed test performance tracks the number of independent atoms, not the OOD threshold.
Open question: at a **fixed** size, is a **cluster-disjoint (OOD)** split harder than a **random
(non-OOD)** split? If OOD is harder at matched size, the gap is the split, not data scarcity.

## 2. The two datasets

- **OOD dataset** -- train/val/test cluster-disjoint (a test pair's clusters are unseen in train).
- **non-OOD dataset** -- train/test share clusters (in-distribution).

Same positive pairs, same size -> the only difference is the split. The gap between them is the OOD
penalty.

## 3. Why the naive "reshuffle the fold CSVs" idea fails

At `m_pos_per_cc=1` each cluster sits in exactly one positive pair (atom = one bipartite CC, which
owns its clusters), so **any** split of those positives is already cluster-disjoint -- reshuffling
gives a second OOD dataset, and the only cross-split mixing would be via relocated negatives (a
confound). A real non-OOD dataset needs clusters to **recur across positives**, which `m_pos=1`
forbids. So both arms need many pairs per fragment.

## 4. Construction -- one pool, two splits

- **Pool.** Edge-cut fragment the mega-CC at `t` into N atoms and keep **all** pairs per fragment
  (`m_pos_per_cc: null`, not 1 -- recurrence is the point). Only the fragmentation straddlers are
  dropped, from both datasets. (Small CCs are ~1% of pairs at low `t`; exclude them only at higher
  `t`, where they would sit in one fold and dilute the non-OOD split.)
- **OOD dataset.** Route whole fragments to folds -- the existing GroupKFold-by-`atom_id` path
  (`make_folds_within_fold`). No cluster spans folds.
- **non-OOD dataset.** Split the same pairs at the **pair** level (random K-fold, ignoring fragment
  boundaries) -- new `make_folds_random`. A dominant cluster (NA_0, HA_0) is in many pairs within
  its fragment, so a pair-level split scatters those pairs across folds and the cluster lands in
  more than one fold -- train/test overlap.
- **Negatives.** `within_fold` synthesizes negatives per split, per arm, from that split's own
  positives (`dataset_pairs_cc.py:344`) -- there is **no shared negative pool**. Same procedure and
  ratio, so the split is the only manipulated variable; the negatives then differ only as a
  consequence (OOD arm: unseen-cluster negatives; non-OOD arm: seen-cluster negatives). within_fold
  is *required* for the OOD dataset (a cross-split negative would put a test cluster in train) and
  **not required** for the non-OOD dataset -- used there only to keep the two builds identical. Both
  arms draw endpoints from the same pool `P`, so the non-OOD arm adds no new sequence; per-split
  counts match (`round(r x n_split_pos)`) while content differs by independent sampling (a sequence
  can appear in a non-OOD negative yet in no OOD negative -- harmless, never a new sequence).
- **Matched size (OOD is the constraint; non-OOD matches it) -- kept simple.** One run, in memory
  (`out_dir/ood/`, `out_dir/non_ood/`) -- no two-run handshake, no template file. Build the OOD arm
  **first**; it gives every positive a home fold. For the non-OOD arm, **permute the home-fold
  labels across positives, preserving the per-fold histogram** -- a column shuffle. Same per-fold
  sizes by construction; `make_folds_random` takes the OOD per-fold sizes as an argument. Aggregate
  test size matches automatically (each pair is tested once); per-fold matching keeps each fold's
  OOD and non-OOD models on equal train sizes.

## 5. What the non-OOD dataset is, and the `t` trade

- **A mixture, not uniformly in-distribution.** A test pair is "seen" if one of its clusters is also
  in train. The dominant clusters recur and are almost always in train, so most test pairs are seen;
  but a test pair whose two clusters are both rare can have neither in train -- that pair is
  individually OOD. So the non-OOD arm is *mostly* in-distribution with a tail of OOD pairs (what a
  real random split looks like), while the OOD arm has *every* test pair OOD by construction; the
  measured gap is between those two.
- **`t` sets a trade.** Lower `t` -> larger, denser mega-CC -> richer pool and more recurrence, but
  the edge-cut floor (NA_0 ~37% at t095) caps N and leaves one ~37% fragment that cannot be split --
  it lands whole in one fold and unbalances the OOD folds. Higher `t` -> thinner pool, more even
  atoms. Pick `t` from the per-`t` mega-CC fractions (to measure). If that ~37% test fold muddies
  the OOD metric, cap pairs per atom (`m_pos`) so the largest atom is <= ~1/K and let the non-OOD
  arm match the smaller size.

## 6. Code changes (in `dataset_pairs_cc.py` unless noted)

- **Rename** `make_folds` -> `make_folds_within_cc` (it is the within_cc path; the current name is
  too generic and confuses it with `make_folds_within_fold`).
- **New** `make_folds_random(pos_full, k_folds, val_ratio, seed, *, per_fold_sizes, ...)`: pair-level
  K-fold (plain, NOT GroupKFold-by-atom), pair-level val carve (not `_carve_val_atoms`, which is
  atom-grouped), hitting `per_fold_sizes` from the OOD arm, then reuses `within_fold_negatives`.
- **`m_pos_per_cc: null` (keep all pairs).** Relax `_resolve_spec` (`:539`) to accept `None`
  (`CCSpec.m_pos: int | None`); the cap site `if spec.m_pos:` (`:688`) already treats `None` as
  no-cap, and CLAUDE.md/glossary already document "null = no cap." (Stop-gap with no code change: a
  large int.)
- **Paired-run flow.** One entry builds pool -> OOD arm -> non-OOD arm (matched) -> writes
  `out_dir/{ood,non_ood}/fold_k/...` + a paired `cv_info.json`. Config selects it. A new bundle
  (edge_cut on, `m_pos=null`, paired mode, `target_atoms`, `K`) -- the existing
  `flu_ha_na_cc_nt_cds_ood_edge_cut` sets `m_pos=1` (wrong here).
- **Build emits artifacts for post-hoc plotting (do not couple plotting into the build).** Plotting
  is always standalone functions over a run dir; "during build" just calls them inline. So the build
  persists what later plots need and cannot recover otherwise:
  - the per-pair `cc_id` / `natural_cc_id` / `atom_id`,
  - the **dropped pairs** (`fragment_until` returns `dropped_pos` at `_megacc_cut.py:397`;
    `assign_atoms_prod` currently discards it as `_dropped_pos` at `:179`) -- else the straddling
    edges are lost after the build,
  - the resolved `cluster_id_path`.
  Metadata (`host`/`hn_subtype`) is **not** persisted -- it is re-derivable post-hoc from
  `assembly_id` (in the fold CSVs) via the same `enrich_prot_data_with_metadata` the build uses.

## 7. Verification and plots

All plot functions are **lean and parametrized** -- `alphabet`, `representation`, `slot`, and a
generic group/color column are arguments; nothing hard-codes an alphabet, a representation, or
"colored by cluster." Every plot is a standalone function over a run dir (post-hoc; callable inline).

- **V1 -- numeric overlap verifier (crucial).** Read both arms' fold CSVs; join the per-slot hash
  (`cds_dna_hash_a/b` for nt_cds) -> `cluster_id` via `load_cluster_lookup`; assert the OOD arm has
  **0** cross-fold cluster overlap on both slots and the non-OOD arm has **> 0**. (`_PAIR_COLUMNS`
  carries all three per-slot hashes, so the join needs no schema change; the CC builder does not add
  `cluster_id` to the CSVs.)
- **V2 -- 4-panel split-colored UMAP.** OOD-left, OOD-right, non-OOD-left, non-OOD-right; each point
  is a pair's slot-side sequence, colored by train/val/test. OOD panels should show split-separated
  clusters; non-OOD panels should show a shared cluster as **mixed colors in one region** -- the
  visual proof of overlap. Two layers so representation is swappable:
  - a **representation provider** `hashes -> vector matrix`, dispatched by
    `representation in {kmer_nt_cds, esm2, ...}` (kmer_nt_cds joins `cds_dna_hash` -> k-mer vector;
    esm2 joins `prot_seq`/`prot_hash` -> the ESM-2 cache; the fold CSVs carry `cds_dna_hash` and
    `prot_seq` but not `cds_dna_seq`, so it is a hash->vector lookup),
  - a **generic `umap_scatter(X, groups, ...)`** that projects and colors by `groups`, agnostic to
    the representation and to what `groups` mean.
  Default **kmer_nt_cds** -- axis-matched to the nt_cds split and to the model's features; ESM-2 is
  protein-level and aa-only (it blurs synonymous variants), a fallback. Reuse the projection +
  scatter/legend mechanics from `plot_clusters.py` (`plot_cluster_umap` / `_load_cluster_embeddings`).
- **P1 -- `_ood` CC-size barplots** per alphabet/`t`/schema-pair: reuse `plot_cc_sizes.py` as-is
  (it reads `cc_pair_sizes.csv` / `cc_pair_sizes_post_edge_cut.csv` from the run). This is the `_ood`
  version of the old analysis `2D_cluster_sizes`.
- **P2 -- per-CC metadata bars** (`hn_subtype`/`host`) on the `_ood` fragments: per-CC stacked
  composition bars, ranked by CC size. Drive from the build's `_ood` frames/artifacts (not the
  analysis `load_pair_universe`, which does the nt_cds protein-only dedup the builder avoids).
- **P3 -- metadata of dropped edges and of the fragments**: same stacked bars over the captured
  `dropped_pos` (the straddling pairs) and over the kept fragments -- what the cut removed vs what
  each atom is made of.

**Shared primitives (extract to durable homes; `bigraph_*` scripts stay in place -- their retirement
is a separate long-term effort, out of scope here).**
- the stacked-composition barplot drawing -> `src/utils/plot_utils.py` (next to `size_barplot`),
  generic (no alphabet / metadata-field baked in); adapt from `bigraph_pair_metadata.py`.
- `pair_key_to_metadata` (per-pair modal metadata over the isolates a pair co-occurs in) ->
  `src/datasets/_pair_helpers.py` (no such helper exists in `src/datasets/` today), next to
  `canonical_pair_key` / `build_cooccurrence_set`; generalize from
  `bigraph_pair_metadata.pair_key_to_metadata`.

## 8. Task list

1. Extract shared primitives: `plot_utils.stacked_composition_barplot` +
   `_pair_helpers.pair_key_to_metadata` (leave `bigraph_*` untouched).
2. `m_pos_per_cc: null` -- relax `_resolve_spec` + `CCSpec.m_pos` type.
3. Capture `dropped_pos` and emit the per-pair artifact (`cc_id`/`natural_cc_id`/`atom_id`),
   `dropped_pairs.csv`, and the resolved `cluster_id_path` in the builder.
4. Rename `make_folds` -> `make_folds_within_cc`.
5. `make_folds_random` (pair-level K-fold + pair-level val + `within_fold_negatives`; accepts the
   OOD per-fold sizes).
6. Paired-run flow + bundle (one run, `out_dir/{ood,non_ood}/`, `m_pos=null`, edge_cut on).
7. V1 numeric overlap verifier (hash->cluster join-back).
8. V2 representation-agnostic UMAP (provider + `umap_scatter`), kmer_nt_cds default.
9. P1 `_ood` CC-size barplots (reuse `plot_cc_sizes.py`).
10. P2 per-CC metadata bars (`_ood`, from build artifacts).
11. P3 dropped-edge + fragment metadata bars.
12. **(GATED)** Stage 4: train both arms, compare AUC/F1; repeated over cut-seed x fold-seed for a CI
    on the gap. **Do not launch without explicit confirmation.**

## 9. Open questions / gating

- **`t` choice.** t095 = rich pool but a lopsided OOD arm (one ~37% fragment); a higher `t` = thinner
  pool, more even atoms. Needs the per-`t` `_ood` mega-CC fractions (to measure).
- **UMAP representation.** kmer_nt_cds default (axis-matched); confirm before building V2.
- **Exclude small CCs?** Only matters at higher `t` (~1% at t095) -- optional.
- **Optional 3rd arm.** Cluster-disjoint folds on **set-cover** clusters (`clusters_nt_cds`) at the
  same N separates the OOD penalty from the plain disjointness penalty. Out of the core 2-arm
  contrast; add only if the OOD-vs-random gap motivates it.
- **Repeats.** cut-seed x fold-seed for CIs; `n_repeats>1` raises today and the edge_cut seed is tied
  to the master seed -- independent variation is a later step.
- **Gating.** Stage-4 training / any AUC run is GATED -- no launch without explicit confirmation.
