# CC-based dataset generation + cross-validation (Tier 1)

**Status: IMPLEMENTED** (2026-07-01) — Phase-1 **(a) Hydra wiring + (c) front-end DONE + verified +
committed** (`48a3e26`): `dataset_pairs_cc.py` is now a maintained, config-driven Stage-3
builder — `--config_bundle` + §11 knobs (incl. `drop_negative_infeasible_ccs` — structural
negative-infeasibility, and `m_pos_per_cc` default 1), v2's front-end (enrich/filter) reused by calling
its helpers, `cluster_disjoint_cc` rejected by `_validate_v2_config`, and
`conf/bundles/flu_ha_na_cc_aa.yaml`. The within-CC negative pool is restricted to the
front-end-filtered `df`. Verified end-to-end on aa HA-NA t099: 928 pos / 928 neg across 928
atoms, cluster-disjoint (0 seqhash/pair_key overlap within folds; each atom tested once);
every kept atom balanced. (Earlier: the `_cv_sampling`→`_cc_helpers` move + builder core in
`09fb2c2`, T1.1 atom-equivalence + T1.2 byte-identical.)
**Phase 1 (b) RESOLVED:** the slim CSV writer is sufficient — a Stage-4 MLP run
(`flu_ha_na_cc_aa`, within_cc, fold_0) consumes `fold_k/{train,val,test}_pairs.csv` end-to-end
(load -> kmer join on `brc_a/b` at 100% coverage -> MLP train -> eval -> artifacts) with no
`save_split_output_v2` reuse needed. (That fold sat at chance — AUC-ROC 0.498 — the honest
within-CC difficulty, not a plumbing gap: a 5-fold LGBM is **also ~chance** on within_cc
(AUC-ROC 0.507 ± 0.030), while within_fold — same positives, cross-CC negatives — reaches
AUC-ROC 0.870. So the cluster shortcut is worth ~0.37 AUC, and aa k=3 kmers carry no learnable
HA–NA specific-pairing signal under within_cc at t099.) The **§7 lock is CLOSED**: the
builder structurally reproduces the in-memory CV (identical universe / `cooccur` / pools / atom
partition; within-CC negatives differ only by label-dependent random seeding) — see §7 and
`scripts/verify_cc_reproduction.py`. **All three molecule axes (aa / nt_cds / nt_ctg)
DONE + verified in the builder**: the negative samplers are alphabet-aware (key pair_key +
enrichment on `_POS_HASH[alphabet]`, not a hardcoded `prot_hash`), the gates allow
`{aa, nt_cds, nt_ctg}` with `pair_key_alphabet == cluster_alphabet` enforced (single
molecule axis), and nt_cds attaches `cds_dna_hash` in `build_frontend` (from
`cds_dna_final.parquet`; because pair_key == cluster alphabet the positives carry POPULATED
cds_dna_hash_{a,b}, so the cluster join never hits the §13c empty-column crash). Bundles
`flu_ha_na_cc_aa` (aa), `flu_ha_na_cc_nt_cds`, `flu_ha_na_cc_nt_ctg` all build end-to-end:
pair_key 100% on the configured axis (cds / contig), cluster-disjoint, balanced, each pair
tested once, both within_cc + within_fold. aa + nt_ctg outputs byte-identical across the
nt_cds change. **Remaining:** Stage-4 k-mer features (`kmer_features_nt_cds/nt_ctg`) — a
separate downstream item, not part of the builder.

**Branch:** `feature/cc-dataset-cv` (off `master`; master already carries the Phase-2 nt_cds machinery).
**Scope:** Tier 1 only. Tier 2 (the symmetric column rename) is a separate future migration — see §9.

**Reconciled 2026-07-01 (pre-merge):** Phases 1–3 (the 2D-CD builder for aa / nt_cds / nt_ctg)
are DONE + verified; the slim writer and §7 reproduction are closed. Deferred future work is out
of this plan's scope and tracked elsewhere: **cut fragmentation** →
`docs/plans/2026-06-04_2d_cd_drop_budget_router_plan.md`; **regime-targeted within-CC negatives**
+ **repeated CV (`n_repeats>1`)** → `BACKLOG.md` § "CC dataset CV — deferred".

---

## 1. Goal

A new **maintained** Stage-3-style script `src/datasets/dataset_pairs_cc.py` — the **2D-CD
builder** — that materializes **bilateral cluster-disjoint (2D-CD)** train/val/test pair sets
with **within-CC negatives**, usable by the **maintained Stage 4**
(`train_pair_classifier.py`) + `analyze_stage4_train.py`. It emits either **2D-CD holdout**
(one 80/10/10 set) or **2D-CD CV** (one set per fold), and must work for **aa / nt_cds /
nt_ctg**. This graduates the CC approach out of the in-memory analysis harness
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

**Method naming** (glossary-anchored): the builder is the **2D-CD builder**; the *method* is
**2D-CD with within-CC negatives** (the within-CC negatives distinguish it from v2's existing
2D-CD, which uses cross-isolate coverage negatives). "2D-CD CV" / "2D-CD holdout" name only the
two output modes — never call the builder itself "2D-CD CV". New canonical terms `within-CC
negative` and `singleton` live in `docs/methods/glossary.md`.

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

- **Phase 1 — aa (existing data, zero new compute).** §4.A move + §4.B script. **Validated** (the §7
  lock): on aligned inputs (unfiltered, t099, m_pos=1, ratio=1, random negs, seed=42) the materialized
  dataset structurally reproduces the in-memory aa CV (`cluster_disjoint_regime_cv`) — identical positive
  universe (58,826), `cooccur` set, isolate pools, and **atom partition** (4,125 CCs). Within-CC negatives
  differ in identity only because the two atom-assignment implementations label CCs differently and the
  sampler seeds on `seed+cc_id` (different valid random draws; counts match to ±1 CC) — not a faithfulness
  gap. Reusable check: `scripts/verify_cc_reproduction.py`. Locks architecture + schema + naming.
- **Phase 2 — nt_cds.** Build `kmer_features_nt_cds`; run `dataset_pairs_cc` for nt_cds; validate.
- **Phase 3 — nt_ctg.** Contig exporter → `clusters_nt_ctg` + membership + `kmer_features_nt_ctg`; run; validate.

## 8. Settled decisions

- val carved from train, group-aware; output = `_PAIR_COLUMNS` in `fold_{k}/` dirs with canonical filenames.
- negatives within-CC only; random now, regime = placeholder.
- **Negative-infeasible CCs: flag, default drop.** A CC is *negative-infeasible* when no within-CC
  negative can be drawn — every recombination of its distinct slot-A × slot-B sequences reconstructs
  a positive. The **singleton** CC (one `pair_key` after Mode #1 dedup) is the base case; the superset
  also covers dense CCs where every cross pairing co-occurs (e.g. all positives share one slot). Knob:
  `drop_negative_infeasible_ccs` (default true), computed **structurally** and seed-independently
  (`compute_negative_infeasible_ccs`, mirroring the within-CC sampler) and applied identically under
  both `negative_scope` values, so within_cc and within_fold route the same CC universe. See glossary
  (*Negative-infeasible CC*). (History: the knob was `drop_single_isolate_ccs` → `drop_singleton_ccs`
  → `drop_negative_infeasible_ccs`; the 2026-06 audit found `drop_singleton_ccs` conflated *singleton*
  = 1 pair_key with the broader negative-infeasible set, and that within_cc/within_fold used divergent
  predicates — unified here on the structural test.)
- **Mode #2 (sequence-level label imbalance) is a control, not a defect.** Keeping singletons
  leaves their two sequences only-positive (Mode #2 unmitigable for them); dropping them makes
  Mode #2 *mitigable* — but the mitigation itself is per-sequence coverage (cf. v2's coverage
  phase, `dataset_segment_pairs_v2.py:819`), still a future option. Leakage coverage of this
  builder: #1 ✅ (pair_key dedup + cooccur block), #2 choice (above), #3 ✅ by construction (a
  whole CC sits in one fold), #4 ✅ (the point), #5 placeholder (random within-CC → natural regime
  mix; targeted regimes deferred).
- `n_repeats`: placeholder.
- CC primitives owned by `src/datasets/_cc_helpers.py`; analysis imports them back.
- production hash names kept; analysis `dna_hash`→`cds_dna_hash` mislabel fixed.

## 9. Deferred / future

- **Tier 2 symmetric rename** (planned, separate migration): `prot_seq`/`cds_dna_seq`/`ctg_dna_seq` +
  matching `*_hash` (NOT `aa_*` — a *seq*/*hash* is molecule-level → `prot`/`dna` per the glossary),
  and the **file** renames `genome_final → ctg_dna_final`, `cds_final → cds_dna_final` (keep
  `protein_final`) to mirror the columns and drop the "genome" misnomer (it's contigs).
  - **Hash persistence is uneven today** (verified via parquet schema): only `cds_final.parquet`
    persists hashes (`seq_hash` + `cds_dna_hash`); `protein_final.parquet` and `genome_final.parquet`
    carry the sequence but no hash column — so `seq_hash` is recomputed by every reader and
    `dna_hash` is computed at Stage 3 by `attach_dna_to_prot_df`. Tier-2 should **produce + persist
    each hash at the stage that creates its sequence** (`seq_hash` in `protein_final` Stage 1,
    `dna_hash` in `ctg_dna_final` Stage 1, `cds_dna_hash` in `cds_dna_final` Stage 1.5) and make
    **Stage 3 a pure consumer (join)** — removing the on-the-fly `dna_hash` computation and the
    scattered `seq_hash` recomputes. (Also fix CLAUDE.md's stale "seq_hash written by Stage 1".)
  - Rewrites those parquets + every run dir + all readers ⇒ a data migration, all-or-nothing across
    the repo. **Test bar = bit-exact / ε=0** (pure relabeling, content unchanged): run
    `scripts/split_regression_harness.py` + rebuild representative datasets and confirm
    byte-identical, like the Phase-2 pair_key migration.
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
- `dataset.split_strategy.drop_negative_infeasible_ccs: true` — drop CCs with no drawable within-CC
  negative (every slot-A × slot-B recombination reconstructs a positive; singleton CCs are the base
  case). Computed structurally (`compute_negative_infeasible_ccs`), seed-independent, applied
  identically under both `negative_scope` values — **parity: within_cc and within_fold both keep 928
  on aa HA-NA t099** (within_fold previously kept 1,219; the 291-CC gap is closed). Renamed from
  `drop_singleton_ccs` (the audit found that name conflated *singleton* = 1 pair_key with the broader
  negative-infeasible set; see glossary *Negative-infeasible CC*). (Note: single-*cluster*-pair CCs
  holding multiple seq pairs are not singletons and **do** yield negatives.)
- `dataset.split_strategy.mode += cluster_disjoint_cc` — explicit CC-builder mode; the v2 validator
  rejects it, so a CC bundle can't be misrun through `dataset_segment_pairs_v2`.
- `dataset.split_strategy.cluster_alphabet += nt_ctg` — code validates `{aa, nt_cds}` today; the
  comment at `conf/dataset/default.yaml:221` is also stale (`'nt'` → `'nt_cds'`), fix in the same edit.
- `conf/kmer/default.yaml` `alphabet`: hard-rename `nt → nt_ctg`, add `nt_cds`; **regenerate** the
  k-mer caches (no alias). Any bundle/code referencing `kmer.alphabet: nt` moves to `nt_ctg`.

## 12. Tests, bundle, next step

**Tests / non-regression.** `dataset_pairs_cc.py` is additive (new file) — it can't regress v2.
The only v2-touching change is adding the `mode=cluster_disjoint_cc` rejection to
`_validate_v2_config`; when that lands, run `scripts/split_regression_harness.py` (bit-exact
holdout goldens under `tests/golden/split_regression`, "8/8") + rebuild one existing bundle to
confirm unchanged. The new builder itself is checked by the 4 invariants + the §7 in-memory
reproduction (no golden exists for it yet — create one once the output schema is locked).
`flu_ha_na_tight.yaml` is a metadata-filtered **research** bundle (seq_disjoint), **not** a
regression bundle — don't use it for that.

**CC bundle.** None exists yet (scaffold is argparse-only). Create `conf/bundles/flu_ha_na_cc.yaml`
during the Hydra wiring, **deriving from `flu_ha_na_cluster_t099.yaml`** (inherits cluster path +
threshold), setting `mode=cluster_disjoint_cc` + the §11 knobs. Not `flu_ha_na_tight` (wrong
routing + heavy filters).

**Next step (recommended).** Phase-1 (a)+(c): Hydra/`--config_bundle` + front-end + §11 knobs
(one change — wiring brings the config-driven front-end). **Defer (b)** the full
`save_split_output_v2` reuse (it needs `exposure_tables` + `duplicate_stats` the CC path doesn't
produce, and runs audit hard-fails) — keep the slim writer until a real bundle shows what Stage 4
/ `analyze_stage4_train` actually consume. **Fork to confirm before coding:** §7's lock is
"materialized folds reproduce the in-memory CV"; the committed verification so far is
invariant-level only. Decide whether the invariants suffice to lock, or close the §7 reproduction
first (cheaper before the front-end filter changes the population).
