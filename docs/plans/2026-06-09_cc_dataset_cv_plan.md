# CC-based dataset generation + cross-validation (Tier 1)

**Status: IN PROGRESS** — Phase-1 **core IMPLEMENTED + verified + committed**
(`09fb2c2`, 2026-06-10): the `_cv_sampling`→`_cc_helpers` move (§4.A; T1.1 atom
equivalence + T1.2 byte-identical) and `dataset_pairs_cc.py` (§4.B; aa HA-NA t099 —
schema, feature keys, cluster-disjoint + within-CC, pair_key/label all verified).
**Remaining Phase 1:** Hydra/`--config_bundle` + §11 knobs (scaffold uses argparse);
full `save_split_output_v2` reuse (parquet/audit/plots) vs. the slim CSV writer;
front-end metadata enrich/filter (scaffold is unfiltered). Then **Phase 2** (nt_cds:
`kmer_features_nt_cds` + the `dna_hash`→`cds_dna_hash` mislabel fix), **Phase 3** (nt_ctg).

**Branch:** `feature/cc-dataset-cv` (off `master`; master already carries the Phase-2 nt_cds machinery).
**Scope:** Tier 1 only. Tier 2 (the symmetric column rename) is a separate future migration — see §9.

---

## 1. Goal

A new **maintained** Stage-3-style script `src/datasets/dataset_pairs_cc.py` that materializes
**cluster-disjoint K-fold** train/val/test pair sets (one set per fold) usable by the **maintained
Stage 4** (`train_pair_classifier.py`) + `analyze_stage4_train.py`. It must work for **aa /
nt_cds / nt_ctg**. This graduates the CC approach out of the in-memory analysis harness
(`src/analysis/_cv_*`) into the maintained pipeline.

## 2. Why a new script (not v2's cluster_disjoint mode)

`dataset_segment_pairs_v2.py` (maintained) already has a cluster-disjoint CV path
(`generate_all_cluster_disjoint_cv_folds_v2`), but it is **single-slot only** (bilateral CC k-fold
explicitly raises in `cluster_disjoint_route_pos_df`) and uses **cross-isolate** coverage-first
negatives. The CC track established two things v2 does not do:

- **bilateral CC GroupKFold** (atoms = bipartite connected components on `(cluster_id_a, cluster_id_b)`), and
- **within-CC negatives** (every negative drawn from the same CC as its positives).

`dataset_pairs_cc.py` adds exactly these; everything else reuses v2.

**Background (why two DNA notions exist).** The project went prot_seq+ESM-2 → contig-DNA k-mers →
prot_seq k-mers → mmseqs2 clustering (aa, then DNA). DNA *clustering* was switched to **CDS** on the
assumption that clustering should be coding-only (never tested against contig clustering); k-mer
*features* stayed **contig**. That split is the root of the `dna_hash` (contig) vs `cds_dna_hash`
(CDS) duality in §3 — and adding nt_ctg finally lets us test the assumption.

## 3. Facts established (verified 2026-06-09)

- **Three split hashes, all md5**: `seq_hash`=md5(`prot_seq`), `cds_dna_hash`=md5(`cds_dna`),
  `dna_hash`=md5(`dna_seq`=contig). The **ESM-2 embedding cache** uses **sha1** (`esm2_utils.py:643`) —
  a separate namespace, not a split key.
- **nt_ctg hash = `dna_hash`** — already exists (computed Stage 3 by `attach_dna_to_prot_df`, in every
  pair CSV). Only the nt_ctg **clusters** are missing.
- **⚠ `dna_hash` is overloaded across layers (verified 2026-06-09 — landmine):** in **production**
  (`_pair_helpers.py:142`, `_PAIR_COLUMNS`) `dna_hash` = md5(contig `dna_seq`) = the **nt_ctg** key
  (PB2 isolate: `ed2c86dd…`). But the **analysis** `load_pair_universe`
  (`cluster_pair_weight_topk.py:123/127`) renames `cds_dna_hash` → `dna_hash_a/b`, so there
  `dna_hash` = md5(`cds_dna`) = **CDS** (same isolate: `4f8cb952…`). Same name, opposite molecule.
  The §4.A move **must** rename the analysis nt_cds key `dna_hash_a/b` → `cds_dna_hash_a/b`, so the
  canonical mapping holds everywhere — **aa → `seq_hash` (protein), nt_cds → `cds_dna_hash` (CDS),
  nt_ctg → `dna_hash` (contig)** — and never reuse the analysis `dna_hash`-means-CDS code beside
  production's `dna_hash`-means-contig.
- **Stage 4 keys k-mer features on `(assembly_id, brc_fea_id)` (aa) / `(assembly_id, genbank_ctg_id)` (nt)**
  — i.e. on `brc_a/b` and `ctg_a/b`, **not** on the hash columns (`kmer_utils._pair_side_col`). So hash
  column names are free; the CSV must carry `assembly_id_a/b`, `brc_a/b`, `ctg_a/b`, `label`.
- **Output schema** = `_PAIR_COLUMNS` (`dataset_segment_pairs_v2.py:89`): `pair_key, assembly_id_a/b,
  brc_a/b, ctg_a/b, seq_a/b, dna_seq_a/b, seg_a/b, func_a/b, seq_hash_a/b, dna_hash_a/b,
  cds_dna_hash_a/b, label, neg_regime, metadata_match_count`.
- **Dependency rule** (`_megacc_cut.py:18`): `src/datasets` must **not** import `src/analysis`
  (direction is analysis → datasets). This forces the primitive move in §4.A.

Source-read basis: full read of `dataset_segment_pairs.py`, `dataset_segment_pairs_v2.py`,
`_pair_helpers.py`, `_split_helpers.py`, `_negative_regime_sampling.py`, `_megacc_cut.py` (~6,600 lines).

## 4. Architecture

### 4.A Move CC primitives into `src/datasets/_cc_helpers.py` (dependency-forced shape)
Full read of `_cv_sampling.py` (2026-06-09): `assign_atoms`/`_fragment_atoms` import `load_cluster_map`,
`fragment_weighted`, `uniform_targets` from **`src.analysis`** (lines 43–44) — which `_cc_helpers` (in
`src/datasets`) cannot import (dependency rule; the wall `_megacc_cut.py` documented). So:
- **Atom assignment is NOT ported.** Production already has the equivalent, alphabet-clean:
  `_split_helpers.attach_cluster_ids(pos_df, cluster_lookup, pos_hash_col)` +
  `_pair_helpers.bipartite_components(col_a='cluster_id_a', col_b='cluster_id_b')`. `dataset_pairs_cc.py`
  calls those directly. **This dissolves the mislabel** — production keys nt_cds on `cds_dna_hash`
  correctly, so there is no `dna_hash`-means-CDS path to rename; the mislabeled analysis
  `_HASH`/`assign_atoms` path is simply dropped, not migrated.
- **`_cc_helpers.py` holds only the genuinely-new pieces** (datasets-only deps: `_pair_helpers`,
  `_negative_regime_sampling`): the within-CC negative sampler (`sample_random_within_cc_negatives`;
  `sample_regime_negatives` = placeholder) + the per-CC isolate-pool bridge (adapted
  `build_isolate_context`), with its membership-hash column made **alphabet-specific**
  (`seq_hash`/`cds_dna_hash`/`dna_hash`) instead of the hardcoded `seq_hash`.
- **'cut' fragmentation deferred** (needs `fragment_weighted`): Phase 1 is natural CCs only; when
  fragmentation lands, reuse `_megacc_cut.apply_drop_budget_cut` (already in `src/datasets`).
- The analysis CV harness (`cluster_disjoint_*`) then imports the sampler + bridge **back from
  `src/datasets/_cc_helpers`** and switches its atom-assignment to the production primitives.

### 4.B `src/datasets/dataset_pairs_cc.py` (new, alongside `dataset_segment_pairs_v2.py`)
Thin orchestrator — **reuse** unless marked NEW:
- Front-end: reuse v2's `load → attach_dna_to_prot_df → enrich → filter_by_metadata → narrow to schema_pair`.
- Positives: `create_positive_pairs_v2` (one pair per isolate, deduped by `pair_key`).
- Cooccur blocking: `build_cooccurrence_set` + `canonical_pair_key`.
- Atoms: `attach_cluster_ids` + `bipartite_components(col_a='cluster_id_a', col_b='cluster_id_b')`
  (bilateral CC). For nt_ctg, add `dna_hash` to `load_cluster_lookup`'s accepted hash columns.
- **Folds (NEW)**: `GroupKFold(n_splits=k, shuffle=True, random_state=seed)` on `atom_id`.
- Per fold: `test` = held-out atom group; `val` = **group-aware** carve from the remaining atoms (whole
  atoms, never split an atom across train/val); `train` = the rest.
- **Within-CC negatives (NEW)**: per split, draw random non-cooccurring pairs **within each CC**, reject
  cooccur/dup, then **enrich to `_PAIR_COLUMNS`** by joining `df` on `(assembly_id, function)` (same
  per-side fields `create_positive_pairs_v2` produces). Regime-targeted negatives = **placeholder**.
- **Singleton / positive-only-CC flag (NEW)**: a CC that cannot form any within-CC negative (the 1×1 case)
  → `--keep_positive_only_ccs` (default **drop**, keeping each contributing CC class-balanced).
- Output: `fold_{k}/{train,val,test}_pairs.csv` (+ `.parquet`) via `save_split_output_v2` (reuse, possibly
  slimmed). Canonical filenames per fold dir ⇒ each `fold_k/` **is** a drop-in Stage-4 dataset.
- `--n_repeats`: **placeholder** arg (only `n_repeats=1` wired; `>1` raises with a note). Reserved layout
  `repeat_{r}/fold_{k}/`.

### 4.C Naming
Keep `_PAIR_COLUMNS` names verbatim (drop-in Stage-4 + consistent with every dataset). Route column names
through the central `_PAIR_COLUMNS` constant so the future Tier-2 rename is a single-point change.

## 5. Alphabet readiness + what each needs

| alphabet | split key | clusters | membership | k-mer features |
|---|---|---|---|---|
| aa | `seq_hash` | ✅ `clusters_aa` | ✅ | ✅ `kmer_features_aa_*` |
| nt_cds | `cds_dna_hash` | ✅ `clusters_nt_cds` | ✅ | ❌ build `kmer_features_nt_cds_*` |
| nt_ctg | `dna_hash` | ❌ build `clusters_nt_ctg` | ❌ build | rename existing `nt` → `kmer_features_nt_ctg_*` |

## 6. Prerequisites (data + feature gen)

1. **`clusters_nt_ctg` + `cluster_memb_nt_ctg`**: operationalize the contig exporter
   (`export_function_ctg_fasta`) in `clustering_utils`/`build_mmseqs_clusters` (today
   `NotImplementedError`), run the t100→t090 sweep, extend `build_cluster_membership` for nt_ctg.
   UTR-trimming caveat: keep contigs as-is (NOTE already banked at `clustering_utils.py` `_COLS_BY_ALPHABET`).
2. **k-mer features**: extend `compute_kmer_features.py` to emit `kmer_features_nt_cds_k*` (from
   `cds_final.cds_dna`, keyed by `(assembly_id, function)`) and rename the existing `nt` →
   `kmer_features_nt_ctg_k*` (from `genome_final.dna_seq`, keyed by `(assembly_id, genbank_ctg_id)`).
   Extend `kmer_utils._occurrence_col`/`_pair_side_col` accordingly.

## 7. Phasing

- **Phase 1 — aa (existing data, zero new compute).** §4.A move + §4.B script. **Validate** the materialized
  5-fold reproduces the in-memory aa CV (`cluster_disjoint_regime_cv`). Locks architecture + schema + naming.
- **Phase 2 — nt_cds.** Build `kmer_features_nt_cds`; run `dataset_pairs_cc` for nt_cds; validate.
- **Phase 3 — nt_ctg.** Contig exporter → `clusters_nt_ctg` + membership + `kmer_features_nt_ctg`; run; validate.

## 8. Settled decisions

- val carved from train, group-aware; output = `_PAIR_COLUMNS` in `fold_{k}/` dirs with canonical filenames.
- negatives within-CC only; random now, regime = placeholder.
- positive-only CCs: flag, default drop.
- `n_repeats`: placeholder.
- CC primitives owned by `src/datasets/_cc_helpers.py`; analysis imports them back.
- production hash names kept; analysis `dna_hash`→`cds_dna_hash` mislabel fixed.

## 9. Deferred / future

- **Tier 2 symmetric rename** (planned, separate migration): `prot_seq`/`cds_dna_seq`/`ctg_dna_seq` +
  matching `*_hash` (NOT `aa_*` — a *seq*/*hash* is molecule-level → `prot`/`dna` per the glossary).
  Rewrites `protein_final`/`genome_final`/`cds_final` parquets + every run dir + all readers ⇒ a data
  migration, all-or-nothing across the repo.
- Regime-targeted within-CC negatives (placeholder now).
- Repeated K-fold (`n_repeats > 1`).

## 10. Open during implementation

- Slim vs full `save_split_output_v2` reuse (its `dataset_stats` expects `exposure_tables` /
  `duplicate_stats` — the CC path may want a slimmer saver).
- Factor the v2 front-end into a shared function vs call it.

## 11. Config (knobs)

All in existing blocks — no parallel scheme. Reads via Hydra / `--config_bundle` (like v2).

**Reuse unchanged:** `split_strategy.{cluster_id_path, cluster_id_threshold, cluster_alphabet,
feasibility.*}`, `n_folds` (= K), `neg_to_pos_ratio`, `val_ratio`, `negative_sampling`
(`null` → within-CC **random**; a regime dict → within-CC **regime** [placeholder, raises in Phase 1]),
the whole front-end filter/sample block.

**New:**
- `dataset.n_repeats: 1` — placeholder (only `1` wired; `>1` raises).
- `dataset.split_strategy.drop_single_isolate_ccs: true` — drop CCs with a single isolate (no
  within-CC negative possible — the dominant degenerate case). **Not** bare `singleton`:
  `_pair_helpers.bipartite_components` already reports `singleton_components` = single-*pair* CCs,
  which can have ≥2 isolates and *do* yield negatives.
- `dataset.split_strategy.mode += cluster_disjoint_cc` — explicit CC-builder mode; the v2 validator
  rejects it, so a CC bundle can't be misrun through `dataset_segment_pairs_v2`.
- `dataset.split_strategy.cluster_alphabet += nt_ctg` — code validates `{aa, nt_cds}` today; the
  comment at `conf/dataset/default.yaml:221` is also stale (`'nt'` → `'nt_cds'`), fix in the same edit.
- `conf/kmer/default.yaml` `alphabet`: hard-rename `nt → nt_ctg`, add `nt_cds`; **regenerate** the
  k-mer caches (no alias). Any bundle/code referencing `kmer.alphabet: nt` moves to `nt_ctg`.
