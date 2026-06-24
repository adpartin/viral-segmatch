# nt_cds / nt_ctg disambiguation + symmetric hash/sequence/file refactor

**Status: Phase A + Phase B COMPLETE; Phase C (the 3 experiments) next.** Phase A: A1 registry + `tests/test_schema.py`; A2 Stage-1/1.5 producers + non-destructive data migration; the keystone (Stage-3 pair-table → `schema.build_pair_columns()`, all 5 builders) + the `bipartite_components` seq_disjoint fix; all consumers (Stage 4 vocab/bugs, 18 analysis/viz files incl. the `dna_hash` mislabel fix); routing maps → registry (`clustering_utils`/`build_mmseqs_clusters`/`build_cluster_membership`, `compute_seq_hash`→`compute_prot_hash`); k-mer `nt`→`nt_ctg` (builder/loader/Stage 4 + 6 renamed data files + configs). **Gate-A:** Stage-3 dataset ε=0 (byte-identical modulo rename) + Stage-4 `nt_ctg` k-mer training end-to-end verified. Phase-A commits: `38fac8e` keystone, `6f5e4cf` consumers, `181c576` routing maps, `591605a` k-mer. **Phase B done:** B3 v2 ungate (`8580623`); B1 `kmer_features_nt_cds_k3` (`ae55f79`+`dd62942`); B2 `clusters_nt_ctg` + `cluster_memb_nt_ctg` (`1c3423c`); plus the `plot_kmer_routing_geometry` nt_ctg viz cutover the B2 consumer-smoke surfaced (`4601e38`); B4 `cluster_disjoint_feasibility` registry cutover + measured feasibility (`ab916a8`; HA/NA operable at t100/t099 only on all three alphabets) + `clusters_aa` `seq_hash`→`prot_hash` migration; B1-k6 `kmer_features_nt_cds_k6` (`2e6f3c0`, 868,240×4096, ~1.008B nnz). **Phase B COMPLETE.** **Phase C:** the 3 experiments (incl. C1 baseline F1≈0.958). Loose ends: `compute_kmer_features` docstring labels; `train_pair_baselines` `kmer_alphabet` provenance. Decisions locked (§11).
**Branch:** `feature/nt-cds-ctg-refactor` (based on `feature/cc-dataset-cv`).
**Source:** the 2026-06-21 read-only pipeline audit (7-slice fan-out, Stage 1→4 + configs + analysis + docs). Load-bearing findings re-verified directly (see §5).

---

## 1. Goal

Disambiguate the historically-overloaded `nt` alphabet into **`nt_cds`** (CDS DNA) and **`nt_ctg`** (contig DNA) across the *entire* pipeline, and make sequence / hash / file naming symmetric and molecule-level. Introduce **one schema registry** as the single source of truth for the per-alphabet column mapping, so the drift that produced the `dna_hash` mislabel cannot recur. Validate by reproducing the v2 2D-CD holdout baseline (≈0.96), then run two clean experiments.

**Why now:** the production `nt` k-mer path is *contig*-level while `nt` clustering is *CDS*-level — a collision. `docs/results/2026-05-15_cluster_disjoint_nt_results.md` ran **nt_cds clusters + nt_ctg features** and got HA/NA t100 F1 ≈ 0.958. The 2026-06-02 Phase-2 migration split *clustering* into `nt_cds` but left the k-mer/feature side and the hashes overloaded. This refactor finishes the split end-to-end.

**Replication target:** v2's 2D-CD *holdout* (`dataset_segment_pairs_v2`, `mode=cluster_disjoint`, `n_folds=null`), not the CC K-fold builder.

---

## 2. Target scheme (the trinity)

| representation | alphabet | seq column | hash column | file | k-mer occurrence key | cluster dir | membership |
|---|---|---|---|---|---|---|---|
| protein | `aa` | `prot_seq` | `prot_hash` | `protein_final` | `brc_fea_id` | `clusters_aa` | `cluster_memb_aa` |
| CDS DNA | `nt_cds` | `cds_dna_seq` | `cds_dna_hash` | `cds_dna_final` | `brc_fea_id`¹ | `clusters_nt_cds` | `cluster_memb_nt_cds` |
| contig DNA | `nt_ctg` | `ctg_dna_seq` | `ctg_dna_hash` | `ctg_dna_final` | `genbank_ctg_id` | `clusters_nt_ctg` | `cluster_memb_nt_ctg` |

¹ CDS↔protein↔`brc_fea_id` are 1-1, so nt_cds k-mers key on `brc_fea_id` — **no new pair column needed**.

**Naming rules** (glossary `aa/nt vs protein/DNA`): alphabet values (`aa`/`nt_cds`/`nt_ctg`) are *alphabet/residue level* → enum, cluster dirs, k-mer feature names, config only. Molecule names (`prot`/`cds_dna`/`ctg_dna`) are *molecule/sequence level* → sequence columns, hashes, file basenames. `protein_final` keeps its name (molecule-level; not `aa_final`). Pair tables carry all three types' `_a`/`_b` columns simultaneously; the active one is selected by `pair_key_alphabet` / `cluster_alphabet`. md5 for all three hashes; ESM-2 cache key stays `sha1(prot_seq)` (separate namespace, untouched).

---

## 3. Step 1 — the schema registry (foundation)

New leaf module **`src/utils/schema.py`** (importable by both `datasets` and `analysis`, respecting the analysis→datasets one-way rule). A frozen typed registry generalizing the existing `clustering_utils._COLS_BY_ALPHABET`:

```
@dataclass(frozen=True)
class AlphabetSchema:
    alphabet: str         # 'aa' | 'nt_cds' | 'nt_ctg'
    seq_col: str          # prot_seq | cds_dna_seq | ctg_dna_seq
    hash_col: str         # prot_hash | cds_dna_hash | ctg_dna_hash
    file_basename: str    # protein_final | cds_dna_final | ctg_dna_final
    occurrence_col: str   # brc_fea_id | brc_fea_id | genbank_ctg_id
    cluster_dir: str      # clusters_aa | clusters_nt_cds | clusters_nt_ctg
    memb_basename: str    # cluster_memb_aa | ...
    kmer_basename: str    # kmer_features_aa | ...
SCHEMA: dict[str, AlphabetSchema]              # the one source of truth
def hash_col_ab(alphabet) -> (str, str)        # ('{hash_col}_a', '{hash_col}_b')
def seq_col_ab(alphabet)  -> (str, str)
def build_pair_columns() -> list[str]          # constructs _PAIR_COLUMNS
```

**Repoint these existing drifted maps at `SCHEMA`** (this dedup *is* the mislabel fix):

| current map | file | becomes |
|---|---|---|
| `_COLS_BY_ALPHABET` | `clustering_utils.py:40` | the seed; move/generalize into `schema.py` |
| `_MEMB`, `_MEMB_HASH` | `_cc_helpers.py:53,60` | `SCHEMA[t].memb_basename` / `.hash_col` |
| `_POS_HASH`, `_SIDE_SRC`, `_SIDE_RENAME` | `dataset_pairs_cc.py:60,63,65` | `SCHEMA` |
| `_HASH` (×3, the **mislabel**) | `_cv_sampling.py:58`, `_cv_features.py:32`, `cluster_disjoint_regime_cv.py:80` | `hash_col_ab(t)` |
| `_ROOT` (×4) | `_cv_sampling.py:55`, `bigraph_pair_feasibility/metadata`, `cluster_count_vs_threshold` | `SCHEMA[t].cluster_dir` |
| `_KEY`, `_MEMB_FILE` | `cluster_source.py:47,49` | `SCHEMA` |
| `_occurrence_col`, `_pair_side_col` | `kmer_utils.py:22,32` | `SCHEMA[t].occurrence_col` |
| `_PAIR_COLUMNS` | `dataset_segment_pairs_v2.py:89` | `build_pair_columns()` |

**Placement rationale (schema contract, not config):** column names are a structural invariant the whole codebase agrees on — not an experiment knob. A Python registry gives type-safety / import-time failure / rename-symbol refactoring and carries derivation logic (`_a`/`_b`, `build_pair_columns`, the `brc_fea_id` 1-1 rule); a YAML config gives none of that, surfaces typos at runtime deep in a join, couples low-level helpers (which run outside Hydra entrypoints) to the config runtime, and — worst — *signals tunability*, inviting the per-bundle override/drift we are removing. Clean split: **`conf/paths` owns WHERE** (dirs, `data_version`, virus root); **`schema.py` owns WHAT** (column names + file basenames + the registry). Column names are virus-agnostic, so there is no `conf/virus` pull either.

---

## 4. Rename map (CONTEXT-SPLIT — not a global find/replace)

| kind | old → new | scope |
|---|---|---|
| alphabet (features) | `nt` → `nt_ctg` | k-mer/feature/contig path |
| alphabet (clustering) | add `nt_ctg`; `nt_cds` exists | clustering already 3-way-aware |
| hash | `seq_hash` → `prot_hash` | **everywhere** (largest blast radius) |
| hash | `dna_hash` → `ctg_dna_hash` | **only** Stage-3/pair-table/contig contexts |
| hash | `dna_hash` → `cds_dna_hash` | **only** the analysis-CV universe (mislabel) |
| hash | `cds_dna_hash` | unchanged |
| seq col | `seq_a/b`→`prot_seq_a/b`; `dna_seq_a/b`→`ctg_dna_seq_a/b`; src `cds_dna`→`cds_dna_seq`; src `dna_seq`→`ctg_dna_seq` | `prot_seq` src stays |
| file | `genome_final`→`ctg_dna_final`; `cds_final`→`cds_dna_final` | `protein_final` stays |
| artifact | `kmer_features_nt_*`→`kmer_features_nt_ctg_*`; add `kmer_features_nt_cds_*` | clusters/membership already 3-way |
| NOT renamed | `brc_a/b`, `ctg_a/b`, `genbank_ctg_id`, `brc_fea_id` | occurrence-id join keys, not hashes |

**The critical asymmetry:** `dna_hash` means **opposite molecules in different layers** — *contig* in production Stage-3 pair tables (`_PAIR_COLUMNS`, `attach_dna_to_prot_df`, and their consumers `audit_split_leakage`, `mmd_per_*`, `analyze_predictions_stratified`), but *CDS* in the analysis-CV universe (`_cv_*._HASH`, fed by `cluster_pair_weight_topk.load_pair_universe`, which renames `cds_dna_hash`→`dna_hash_a/b`). A blind `dna_hash → ctg_dna_hash` global replace **corrupts the analysis path**. Fix the analysis side at its single source: `cluster_pair_weight_topk.load_pair_universe`.

---

## 5. Critical hazards (verified)

1. **Dual `dna_hash`** (above) — #1 correctness risk. *Verified directly:* `_PAIR_COLUMNS` `dna_hash_a/b` = `md5(dna_seq=contig)`; `_cv_sampling._HASH['nt_cds'] = ('dna_hash_a','dna_hash_b')` = CDS.
2. **Family-key couplings** — column names built by template `f'{family}_hash_{side}'`, `family ∈ ('seq','dna')`, in `_pair_helpers.bipartite_components` / `seq_disjoint_route_pos_df`, `_split_helpers._build_audit`, and v2 overlap reports; plus audit-JSON keys `seq_hash_overlap`/`dna_hash_overlap` written to disk. Must change in lockstep or lookups **silently return empty**. The registry replaces the template with explicit `SCHEMA` lookups.
3. **Hash-at-source** — only `cds_dna_hash` is produced at its source (Stage 1.5). `prot_hash`(=`seq_hash`) is recomputed by ~every reader; `ctg_dna_hash`(=`dna_hash`) is built at Stage 3 (`attach_dna_to_prot_df`). Target: produce `prot_hash` + `ctg_dna_hash` at Stage 1, make Stage 3 a pure consumer.
4. **nt_cds features don't exist end-to-end** — `compute_kmer_features` is hardcoded `{nt, aa}`; `nt` reads contig. Needs a CDS path (reads `cds_dna_final`, keys `brc_fea_id`). No new pair column (¹).
5. **ESM-2 cache key** is `sha1(prot_seq)` (local var, `esm2_utils.py:643`) — separate namespace; leave alone.

---

## 6. Migration phases

### Phase A — Foundation (pure refactor, hard cutover, regression-gated). No new capability, no numeric change.
- **A1.** `src/utils/schema.py` + repoint the maps in §3 + `_PAIR_COLUMNS = build_pair_columns()`.
- **A2. Stage 1/1.5** (`preprocess_flu.py`, `extract_cds_dna.py`, `cds_utils.py`): produce + persist `prot_hash` (before the `protein_final` write) and `ctg_dna_hash` (before the `ctg_dna_final` write); rename seq cols (`dna_seq`→`ctg_dna_seq`, `cds_dna`→`cds_dna_seq`; `prot_seq` stays) and files (`genome_final`→`ctg_dna_final`, `cds_final`→`cds_dna_final`); `extract_cds_dna` *reads* `prot_hash` instead of recomputing.
- **A3. Stage 2/2.5** (`compute_kmer_features.py`, `kmer_utils.py`, `clustering_utils.py`, `build_mmseqs_clusters.py`): k-mer builder `nt`→`nt_ctg` (rename outputs `kmer_features_nt_*`→`kmer_features_nt_ctg_*`); clustering repoint to `SCHEMA`, cluster-parquet column `seq_hash`→`prot_hash`; `nt_cds`/`nt_ctg` stay gated (enabled in Phase B).
- **A4. Stage 3** (`dataset_segment_pairs{,_v2}.py`, `_pair_helpers.py`, `_split_helpers.py`, `_cc_helpers.py`, `dataset_pairs_cc.py`): rename pair cols via registry; `_POS_HASH`/`_MEMB_HASH` (`nt_ctg`→`ctg_dna_hash`); family-key couplings → registry-driven; `load_cluster_lookup` `prot_hash` (+ accept `ctg_dna_hash`); `attach_dna_to_prot_df` reads `ctg_dna_hash` from source.
- **A5. Stage 4** (`train_pair_classifier.py`, `train_pair_baselines.py`, `_pair_features.py`): alphabet vocab `{nt,aa}`→`{nt_ctg,nt_cds,aa}`; fix swap-test `alphabet=` bug (`:1569`); record `kmer.alphabet` in `training_info.json`; fix `_pair_features` stale docstring.
- **A6. Analysis**: reconcile the 3 `_HASH` copies to the registry (**mislabel fix**, at source in `cluster_pair_weight_topk.load_pair_universe`); split dual `dna_hash` (Stage-3-CSV consumers → `ctg_dna_hash`; CV universe → `cds_dna_hash`); unify bare `nt` (feasibility/cluster → `nt_cds`; k-mer/mmd → `nt_ctg`).
- **A7. Configs**: `conf/kmer/default.yaml` `nt`→`nt_ctg`; `flu_ha_na.yaml` + `flu_pb2_pb1.yaml` `kmer.alphabet: nt`→`nt_ctg`; comment/path refs (`genome_final`→`ctg_dna_final`, etc.).
- **A8. Docs**: update the 6 canonical docs — `CLAUDE.md` (hash + `aa/nt` conventions, pipeline table, source-file comments), `.claude/memory.md`, `docs/methods/{glossary,kmer_features,splits,clusters}.md`. Leave `docs/results/*` and historical `docs/plans/*` as record.
- **Gate A:** regression net (§7) green.

### Phase B — Enable nt_cds + nt_ctg (additive)
- **B1.** [DONE — `ae55f79`+`dd62942`+`2e6f3c0`] Build `kmer_features_nt_cds_k{3,6}` — extend `compute_kmer_features` with the nt_cds path (reads `cds_dna_final.cds_dna_seq`, keys `brc_fea_id`). k3 + k6 both built; k6 = 868,240×4096, ~1.008B nnz, validated via the Stage-4 loaders (100% index alignment with cds_dna_final).
- **B2.** [DONE — `1c3423c`] Build `clusters_nt_ctg/` + `cluster_memb_nt_ctg.parquet` (mmseqs contig sweep; ungate `nt_ctg` in `clustering_utils`/`build_mmseqs_clusters`). Input path: `function` attached to `ctg_dna_final` via a verified 1-1 join from `cds_dna_final` (`attach_function_to_contigs`). Sweep = 11 thresholds (t100–t090) × 8 functions, 868K contigs, ~21 min, 12G. Consumer smoke (v2 bilateral `cluster_disjoint`, nt_ctg, t100): 0% cluster_id cross-split leakage, 2,519/2,519 attach. Surfaced + fixed the `plot_kmer_routing_geometry` nt_ctg viz cutover (`4601e38`).
- **B3.** [DONE — `8580623`] Ungate `nt_cds` + `nt_ctg` in `dataset_segment_pairs_v2` (`cluster_alphabet ∈ {aa,nt_cds,nt_ctg}`; pos-hash via `schema.hash_col`). `dataset_pairs_cc` nt ungate left out (not on the experiment path).
- **B4.** [DONE — `ab916a8`] nt_ctg feasibility pre-flight. Cut `cluster_disjoint_feasibility.py`'s CLI over to the registry (`--cds_final`→`--cds_dna_final`, add `--ctg_dna_final`/`--function_source`, `--alphabet {aa,nt}`→`{aa,nt_cds,nt_ctg}`, slot/cluster hash via `schema.hash_col`). **Measured (HA/NA, largest bipartite-component %, the right metric):** operable at **t100, t099 only** on all three alphabets — t100 nt 1.4–1.5% / aa 49.0%, t099 nt 69% / aa 79.6% (feasible), t098 ≥88% → degenerate, down to ~99% at t090. nt_cds and nt_ctg curves are near-superimposed; aa collapses earlier (protein-identity merges synonymous DNA variants). Data fix: migrated `clusters_aa` (99 parquets `seq_hash`→`prot_hash`, value-preserving — `seq_hash`==`md5(prot_seq)`==`prot_hash`, 100% verified) + regenerated `cluster_memb_aa`; the only pre-refactor-named cluster artifact. **Out of scope (this plan):** fragmenting the t098+ mega-CC via the `drop_budget` edge min-cut (`_megacc_cut.py`, `2026-06-04_2d_cd_drop_budget_router_plan.md`) to unlock lower thresholds — a separate later exploration.

### Phase C — Experiments (v2 2D-CD holdout)
- **C1. Baseline reproduction** [DONE] — nt_cds clusters (t100) + nt_ctg features. Bundle `flu_ha_na_cluster_nt_cds_t100` (reconstructed from the surviving PB2-PB1 id100 `resolved_config`; HA-NA id100 dirs were deleted). **Reproduced:** LGBM test F1 **0.9569** (AUC-ROC 0.9920), MLP **0.9649** — vs the historical id100 LGBM 0.958, i.e. on-target to −0.1 pp. Run under the operationally-proper **nt_cds pair_key** (Option B, matches exp1); at strict t100 the pair_key change is negligible (the −1.9 pp seen at t099 doesn't apply). Confirms the refactor preserved the known result + the full Phase-C pipeline works end-to-end. (A `pair_key_alphabet=aa` historical-fidelity variant needs a v2 fix — the nt_cds-clusters + protein-pair_key combo hit a Phase-2 regression where `cds_dna_hash` isn't populated; deferred, optional.)
- **C2. Exp 1** [DONE] — nt_cds clusters (t100) + nt_cds features (clean CDS). Bundle `flu_ha_na_cluster_nt_cds_t100_feat_nt_cds` (inherits C1's routing, overrides k-mer to nt_cds k6; reuses the C1 dataset via Stage 3/4 decoupling). **LGBM test F1 0.9532** (AUC-ROC 0.9904), MLP ~0.964 (confirmatory). Isolates the FEATURE axis vs C1 (routing + pair_key fixed): nt_cds vs nt_ctg features are ~equal at t100 (LGBM 0.9532 vs C1 0.9569, −0.4 pp) — the contig's flanking UTR adds ~nothing.
- **C3. Exp 2** — nt_ctg clusters + nt_ctg features (clean contig). Isolates the cluster axis vs the baseline. **Reported per option (2): "contig-as-submitted, UTR artifact included"** — a sensitivity check, with the UTR-confound caveat (§9) written into the interpretation.

---

## 7. Cutover + regression net

Hard cutover (no aliases). Regression discipline matches the Phase-2 ε=0 migration:
- **Golden snapshot (before Phase A):** one **aa** dataset+training and the production **nt_ctg** (`flu_ha_na` nt k6) dataset+training.
- **After Phase A, rebuild + diff:** aa → pairs CSV + metrics **byte-/ε=0 identical**; nt_ctg → values + metrics identical, only column *names* differ (compare modulo the §4 map).
- **Baseline (C1):** HA/NA t100 F1 within a tight band of 0.958.
- A snapshot/compare script (extend the `verify_cc_reproduction.py` style) gates the cutover. The family-key hazard (§5.2) is the thing this net most needs to catch — an empty-set silent failure would pass code but change metrics.

---

## 8. Data rebuilds forced by the cutover

- **Rename + add columns:** `protein_final` (+`prot_hash`), `genome_final`→`ctg_dna_final` (+`ctg_dna_hash`, `dna_seq`→`ctg_dna_seq`), `cds_final`→`cds_dna_final` (`cds_dna`→`cds_dna_seq`, `seq_hash`→`prot_hash`).
- **Rename:** `kmer_features_nt_*` → `kmer_features_nt_ctg_*`.
- **Build new:** `kmer_features_nt_cds_k{3,6}`; `clusters_nt_ctg/` + `cluster_memb_nt_ctg.parquet`.
- **Not migrated:** existing dataset/model run dirs keep their frozen `resolved_config` + old column names (historical artifacts).

---

## 9. Pre-existing bugs folded in

- `train_pair_classifier.py:1569` — swap-test `KmerPairDataset(...)` missing `alphabet=` → silently defaults to `nt` (already wrong for aa).
- Neither trainer records `kmer.alphabet` in `training_info.json` (becomes load-bearing once `nt` splits).
- `_pair_features.py` module docstring stale (claims the k-mer path is concat-only; it supports the full interaction set).
- `clustering_utils.clean_nt_for_mmseqs(dna_seq)` — param named `dna_seq` but cleans **CDS**.

**Exp-2 UTR caveat (carried, not fixed — option 2):** contig DNA carries inconsistent flanking UTR (`genome_final`: `contig_len − cds_len` 0–310 nt, median 36; ~25% zero-UTR — a submission artifact, `clustering_utils.py:43-49`, unresolved 2026-06-08). nt_ctg clustering is therefore confounded by submission practice. Exp 2 runs as-is and reports this explicitly; UTR normalization is deferred (§12).

---

## 10. Per-file implementation checklist

Most column references collapse to registry lookups once §3 lands; the manual sites are the producers, the gates, the recompute fixes, the family-key templates, the literal `'nt'` strings, and the bugs. By area (full line-level occurrences in the 2026-06-21 audit):

- **Stage 1/1.5** — `preprocess_flu.py` (add `prot_hash`@~1224, `ctg_dna_hash`@~1263; rename `DNA_SEQ_COL_NAME`, files), `extract_cds_dna.py` (`_OUTPUT_COLUMNS` `seq_hash`→`prot_hash`, `cds_dna`→`cds_dna_seq`; read `prot_hash` not recompute@174; out filename), `cds_utils.py` (`compute_cds_dna_hash` ok; `extract_cds_dna` return var; docstrings).
- **Stage 2/2.5** — `compute_kmer_features.py` (`{nt,aa}`→3-way, `SEQ_COL`/`INPUT_BASENAME`, output basenames@285-287, **add nt_cds path**), `kmer_utils.py` (`_occurrence_col`/`_pair_side_col`/defaults → registry), `clustering_utils.py` (`_COLS_BY_ALPHABET`→`schema.py`; `compute_seq_hash`→`compute_prot_hash`; `clean_nt_for_mmseqs` param), `build_mmseqs_clusters.py` (`seq_hash`→`prot_hash` col@457; `--cds_final`→`--cds_dna_final`; usecols).
- **Stage 3** — `dataset_segment_pairs_v2.py` (`_PAIR_COLUMNS`→`build_pair_columns`; rename maps@198-248; `_cooccur_hash_col`; overlap-report family loops@1857,2519,2676; `nt_ctg` gate@1577; validator@3015), `dataset_segment_pairs.py` (recompute@430; stale `'nt'` comments@530), `_pair_helpers.py` (`attach_dna_to_prot_df`@142 recompute→read; `build_cooccurrence_set` default; `bipartite_components`/`seq_disjoint` family templates), `_split_helpers.py` (`load_cluster_lookup`@57; `attach_cluster_ids` default; `_build_audit` family loops + audit keys), `_cc_helpers.py` (`_MEMB`/`_MEMB_HASH`→registry), `dataset_pairs_cc.py` (`_POS_HASH`/`_SIDE_*`→registry; `compute_negative_infeasible_ccs`/`within_*_negatives` hash args; aa-only gates@369,372).
- **Stage 4** — `train_pair_classifier.py` (`KmerPairDataset` alphabet branch@274; required-col@1322; `kmer.alphabet` read@1300; EMBED_DIM@1306; swap-test@1569; provenance@1591), `train_pair_baselines.py` (`_resolve_kmer_alphabet`@95; provenance@326), `_pair_features.py` (`{nt,aa}`@345; default@281; docstring).
- **Analysis** — `cluster_pair_weight_topk.load_pair_universe` (**source of the mislabel**: `cds_dna_hash`→`dna_hash_a/b`@122-127 → `cds_dna_hash_a/b`); `_cv_sampling`/`_cv_features`/`cluster_disjoint_regime_cv` `_HASH`→registry; the `dna_hash`-from-Stage-3-CSV consumers (`audit_split_leakage`, `mmd_per_{pair,slot}`, `analyze_predictions_stratified`) → `ctg_dna_hash`; bare-`nt` unification (`cluster_disjoint_feasibility`, `single_slot_*`, `cluster_analysis_summary`); `cluster_disjoint_cv_experiment` mismatch guard@323.
- **Configs** — `conf/kmer/default.yaml:6`; `conf/bundles/flu_ha_na.yaml:47`; `conf/bundles/flu_pb2_pb1.yaml:47`; `conf/dataset/default.yaml` comments.
- **Docs** — `CLAUDE.md` (lines ~295/298 conventions, 65-66 pipeline table, 107-108 source comments); `.claude/memory.md`; `docs/methods/{glossary,kmer_features,splits,clusters}.md`.

---

## 11. Decisions (locked)

- **nt_cds occurrence key** — **`brc_fea_id`** (CDS↔protein 1-1; no new pair column).
- **Regression reference bundles** — golden snapshot uses one exact **aa** bundle + the production **nt_ctg** (`flu_ha_na` nt k6) bundle; the specific aa bundle is chosen at the Phase-A gate.
- **nt_ctg feasibility thresholds** — output of B4 (`tXXX` notation); no presumption that `t100`/`t099` carry over from nt_cds.
- **Seq-side pair column** — **`prot_seq_a/b`** (molecule-level; resolves the `seq_a/b` TODO at `_PAIR_COLUMNS:94`).

---

## 12. Out of scope / deferred

- **nt_ctg UTR normalization** — exp 2 runs artifact-included (option 2); normalization deferred until/unless the artifact proves to dominate.
- **CC-builder (`dataset_pairs_cc`) nt experiments** — the experiments use the v2 holdout; Phase B only *ungates* nt in the CC builder, it does not run CC nt CV.
- **Bunya** — not maintained.

---

## A3 execution notes — the consumer cutover (= §6 steps A3–A6) — resume anchor

> Label note: "A3" here (and in the commit messages) is used loosely for the whole consumer cutover — the registry repoint + rename across Stage 2, Stage 3, Stage 4, and analysis, i.e. §6 steps **A3–A6**. The §6 sub-step numbers (A3=Stage 2, A4=Stage 3, …) still apply within.

**Data already migrated** (additive, non-breaking; gitignored; originals + `.bak` intact): `protein_final` (+`prot_hash`), `ctg_dna_final` (was `genome_final`), `cds_dna_final` (was `cds_final`), all 11 `clusters_aa/t*/combined_cluster.parquet` + `cluster_memb_aa` (+`prot_hash`). nt_cds clusters/memb already carry `cds_dna_hash`. `scripts/migrate_to_nt_schema.py` = the Stage-1/1.5 record (cluster `+prot_hash` was an inline additive copy).

**Code repoint done:** `_cc_helpers._MEMB{,_HASH}` → registry (tested). Remaining below.

**Substring-safe rename rules (DO NOT blind global-replace):**
- `seq_hash` → `prot_hash` — safe global (no superstring).
- `dna_hash` → `ctg_dna_hash` **only** in Stage-3/pair-table/contig context, and **never** inside `cds_dna_hash` (which contains `dna_hash`). Use `(?<!cds_)dna_hash`.
- analysis-universe `dna_hash` → **`cds_dna_hash`** (the mislabel: `cluster_pair_weight_topk.load_pair_universe` + the 3 `_HASH` copies) — opposite target from Stage 3.
- `dna_seq` → `ctg_dna_seq`, never inside `cds_dna_seq`. Use `(?<!cds_)dna_seq`.
- pair cols `seq_a`/`seq_b` → `prot_seq_a`/`prot_seq_b` — **not** substring-safe (`seq_a` ⊂ `ctg_dna_seq_a`); edit by context, *after* the `dna_seq` rename.

**Order:** (1) **keystone** — v2 `_PAIR_COLUMNS = schema.build_pair_columns()` + rename all literals in `dataset_segment_pairs_v2`, `_pair_helpers`, `_split_helpers`, `dataset_pairs_cc` (unblocks `_POS_HASH`, `load_cluster_lookup`, and the family-key `f'{family}_hash_{side}'` templates). (2) routing maps → registry (`dataset_pairs_cc._POS_HASH`, `clustering_utils._COLS_BY_ALPHABET`, `kmer_utils`, `cluster_source`, the `build_mmseqs_clusters` producer + cluster-parquet `seq_hash`→`prot_hash`). (3) Stage 2 kmer `nt`→`nt_ctg` (rename the 6 `kmer_features_nt_*` files + the `{nt,aa}` vocab; add the nt_cds builder in Phase B). (4) Stage 4 vocab + 3 bugs (swap-test `alphabet=`, provenance, `_pair_features` docstring). (5) analysis mislabel + dual-`dna_hash` split + bare-`nt`. (6) configs (`conf/kmer/default`, `flu_ha_na`, `flu_pb2_pb1`). (7) docs (6 canonical). (8) **Gate-A regression** (aa byte-exact; nt_ctg identical-modulo-rename; baseline nt **t100** F1≈0.958). At cutover-end: drop the transitional `seq_hash`/`dna_seq` columns + archive `genome_final`/`cds_final`/`.bak` → `_pre_nt_refactor_archive_<date>/`.

**Resume (start here):** branch `feature/nt-cds-ctg-refactor`; last *code* commit `f88cffa` (`_cc_helpers`→registry) — any commits after it are plan/docs only, so `git log --oneline -1` gives the true HEAD; tracked tree clean (untracked `reports/` + `cluster_disjoint_ood_audit.py` are intentional — never commit the latter). First action: `python tests/test_schema.py` (confirm the registry baseline), then begin the keystone (order step 1). **Keystone gotcha** (only-in-head, now banked): the per-side maps that build `_PAIR_COLUMNS` rows must move in lockstep with `_PAIR_COLUMNS = schema.build_pair_columns()` — `dataset_segment_pairs_v2`'s side-rename (~L198-206, source→pair-side: `prot_seq→seq`, `dna_seq→dna_seq`, `seq_hash→seq_hash`, `dna_hash→dna_hash`, `cds_dna_hash→cds_dna_hash`) and `dataset_pairs_cc._SIDE_SRC`/`_SIDE_RENAME` (~L63-67); their pair-side *targets* become the new molecule names (or derive from the registry). Per keystone file, verify with `py_compile` + `grep -nE "seq_hash|dna_hash|dna_seq|seq_[ab]"` and confirm only `cds_dna_*` / `ctg_dna_*` remain. Use `schema.SCHEMA` / `schema.hash_col_ab` / `schema.seq_col_ab` for all alphabet→column lookups (the `dna_hash` context-split: contig→`ctg_dna_hash` in Stage 3, CDS→`cds_dna_hash` in the analysis universe).

---

## Phase B checkpoint + resume (updated 2026-06-24)

**Phase A: COMPLETE** (code cutover + Gate-A + docs + contract tests). Commits `38fac8e` (keystone) → `ea8d82d` (docs) + `ca4cc16` (tests). Registry `src/utils/schema.py` = single source of truth; 19 contract tests (`test_schema` 10, `test_kmer_utils` 3, `test_clustering_utils` 6).

**Phase B done:**
- **B3** — v2 `nt_ctg` ungated: `cluster_alphabet ∈ {aa,nt_cds,nt_ctg}`, pos-hash via `schema.hash_col(cluster_alphabet)` (`8580623`). nt_cds cluster path validated (existing `clusters_nt_cds`): cluster_id cross-split leakage 0/2,044.
- **B1 (nt_cds features) DONE + validated**: `extract_cds_dna` now outputs `brc_fea_id` + `canonical_segment` (`ae55f79`; re-derived cds_dna_final); `kmer_features_nt_cds_k3` built; bundle `flu_ha_na_kmer_nt_cds_k3` (`dd62942`); Stage-4 nt_cds k-mer training Test F1 0.66. → **exp1 path ready** (nt_cds clusters + nt_cds features).
- **B2 (nt_ctg clusters) DONE + validated** (`1c3423c`): `attach_function_to_contigs` attaches `function` to `ctg_dna_final` via a verified 1-1 join from `cds_dna_final` on `(assembly_id, genbank_ctg_id)` (0 multi-function contigs). `build_mmseqs_clusters --ctg_dna_final` mode + nt_ctg ungate in `clustering_utils`; `build_cluster_membership --alphabet nt_ctg`. Sweep: 11 thresholds (t100–t090) × 8 functions, ~21 min, 12G; `cluster_memb_nt_ctg` 868,240×19, all mapped. Consumer smoke (v2 bilateral `cluster_disjoint`, nt_ctg, t100): **0% cluster_id cross-split leakage, 2,519/2,519 attach**. Surfaced + fixed the `plot_kmer_routing_geometry` nt_ctg viz cutover (`4601e38`).

- **B4 (nt_ctg feasibility) DONE + validated** (`ab916a8`): cut `cluster_disjoint_feasibility.py` to the registry (`--cds_dna_final`/`--ctg_dna_final`/`--function_source`, `--alphabet {aa,nt_cds,nt_ctg}`, slot hash via `schema.hash_col`, nt_ctg function-join). **Measured (HA/NA, largest bipartite-component % — the right metric, not cluster count):** operable at **t100/t099 only** on all three alphabets; t098 ≥88% → degenerate. nt_cds≈nt_ctg curves (near-superimposed); aa collapses earlier (t100 already 49% vs nt 1.4%, since protein-identity merges synonymous DNA variants). Data fix: migrated `clusters_aa` `seq_hash`→`prot_hash` (99 parquets, value-preserving; `seq_hash`==`md5(prot_seq)`==`prot_hash`; the only pre-refactor-named cluster artifact) + regenerated `cluster_memb_aa`. **Out of scope (this plan):** fragmenting the t098+ mega-CC via the `drop_budget` edge min-cut (`_megacc_cut.py`, `2026-06-04_2d_cd_drop_budget_router_plan.md`) to unlock lower thresholds.

- **B1-k6 DONE** (`2e6f3c0`): `kmer_features_nt_cds_k6` (868,240×4096, ~1.008B nnz, 1.6GB; bundle `flu_ha_na_kmer_nt_cds_k6`), validated via the Stage-4 loaders (100% index alignment with cds_dna_final). **→ Phase B COMPLETE.**

**Phase C (experiments) — start here:**
- **C1 baseline** — nt_cds clusters + nt_ctg features → reproduce HA/NA t100 F1≈0.958 (needs no new data; all caches built). Reconstruct the deleted `flu_ha_na_cluster_nt_id100` bundle from its run's `resolved_config.yaml`. Then **C2/exp1** (all nt_cds), **C3/exp2** (all nt_ctg, UTR-caveated, §9/§12). Phase-C router = v2 **2D-CD holdout** (bilateral `cluster_disjoint` + `drop_budget`), which can reach below the strict t100/t099 ceiling measured in B4.

**Loose ends (documented):** orchestrator (`dataset_segment_pairs.py`) still reads `cds_final` (works — has `cds_dna_hash` — but should be `cds_dna_final`); `compute_kmer_features` docstring labels (`nt:`→`nt_ctg:`); `train_pair_baselines` `kmer_alphabet` provenance (needs param threading); cc nt ungate (not on the experiment path); `compute_kmer_features` `INPUT_BASENAME` is `.csv` (resolves to parquet via `load_dataframe`, but ext-agnostic is cleaner).

**Resume:** `git log --oneline -1` for HEAD (was `ab916a8`); confirm the 19-test baseline by running each file separately — `python tests/test_schema.py`, then `tests/test_kmer_utils.py`, then `tests/test_clustering_utils.py` (passing all three to one `python` invocation runs only the first as `__main__`); tracked tree clean (untracked `reports/` + `cluster_disjoint_ood_audit.py` intentional — never commit the latter).
