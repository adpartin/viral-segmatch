# Phase 2 — pair_key alphabet migration + clustering enum cleanup

**Status: IN PROGRESS** (pre-flight DONE; commits 1-6 of 7 DONE; HEAD runnable; pair_key semantic change live + test datasets regenerated + models retrained; aa regression guard PASSED at ε = 0 across all 4 cells × 3 post-hoc artifacts; nt_cds vs aa headline delta measured)

Implementation plan for the bundled Phase 2 work, combining:

- **Clustering cleanup plan Phase 2** (`docs/plans/2026-06-02_clustering_cleanup_plan.md`): alphabet enum `{aa, nt} → {aa, nt_cds, nt_ctg}`; `parse_cluster_tsv` writes alphabet-specific hash column; drop the rename hack in `attach_cluster_ids`.
- **Pair_key alphabet plan** (`docs/plans/2026-06-02_pair_key_alphabet_plan.md`): adopt alphabet-specific `pair_key` — aa keeps `seq_hash`-based; nt_cds switches to `cds_dna_hash`-based (universe inflates by +34.9 % on HA-NA, +57.0 % on PB2-PB1 per § 4 measurements).

The two changes touch the same artifact set (~25-30 bundles, ~50-80
datasets, ~100+ trained models) and should land together so users never
see a half-migrated state.

**Bundled IN** (per Alex 2026-06-02 v2):

- `idXXX → tXXX` notation rename (pair_key plan § 7.1; CLAUDE.md
  "Threshold notation" docs-only adoption now finishes its
  code/dir/bundle migration). All four levels: helper + 12+ readers,
  cluster dirs (`clusters_*/tXXX/`), bundle YAML internal refs, and
  bundle filenames (`*_id095_*.yaml` → `*_t095_*.yaml`). Cascade
  detail in § 3.1, § 3.2.

**Bundled OUT** (per Alex 2026-06-02):

- v1 builder deletion (pair_key plan § 7.3) — dedicated branch
  later. Don't risk breaking anything Alex might still need. v1
  stays protein-only; documented divergence from v2 if anyone
  reactivates it.
- nt_ctg operationalization (Phase 3 of clustering cleanup plan) —
  enum value reserved but raises `NotImplementedError`.
- 1-NN baseline retraining — explicitly excluded (long wall time,
  inflated nt universe makes the run intractable; aa/nt MLP+LGBM
  agreement is sufficient regression + delta signal).

---

## 1. Scope

| Surface | Phase 2 action |
|---|---|
| Alphabet enum (code) | `{aa, nt}` → `{aa, nt_cds, nt_ctg}`; `nt_ctg` reserved (raises) |
| `_COLS_BY_ALPHABET` (clustering_utils) | Rename `nt` key → `nt_cds`; reserve `nt_ctg` |
| `parse_cluster_tsv` | New `alphabet` param; writes alphabet-specific hash column name (`seq_hash` / `cds_dna_hash`) instead of legacy `seq_hash` everywhere |
| `attach_cluster_ids` (_split_helpers) | Drop the rename hack at line 102 — join column now matches naturally |
| `canonical_pair_key` (v2 builder) | Alphabet-aware: aa = `canonical(seq_hash_a, seq_hash_b)`; nt_cds = `canonical(cds_dna_hash_a, cds_dna_hash_b)` |
| v2 `cooccur_pairs` | Built per alphabet to match the pair_key family |
| v2 negative-sampling `forbidden_pair_keys` | Per alphabet |
| Bundle YAML key | `split_strategy.cluster_alphabet: aa\|nt` → `aa\|nt_cds`; `pair_key_alphabet` added if needed (else inferred from `cluster_alphabet`) |
| Cluster parquet dirs | `data/processed/flu/<version>/clusters_nt/` → `clusters_nt_cds/`; old `clusters_nt/` and `clusters_aa/idXXX/` archived as `clusters_{aa,nt}_archive_<YYYYMMDD>/idXXX/` |
| Cluster parquet threshold dirs | `clusters_*/idXXX/` → `clusters_*/tXXX/` (new only; old archived) |
| Cluster parquet contents | Regenerate under new column convention (same clustering, just column rename + dir rename) |
| `_threshold_label` helper | `f"id{pct:03d}"` → `f"t{pct:03d}"` |
| Threshold readers (12+ files) | Update `f"id{...}"` patterns; default-threshold constants; plot labels |
| Bundle YAML internal `cluster_id_path` refs | `clusters_aa/id099/...` → `clusters_aa/t099/...` |
| Bundle filenames | `*_id095_*.yaml` → `*_t095_*.yaml` (~25-30 files) |
| Cross-refs to bundle filenames | CLAUDE.md, memory.md, docs/results/, docs/methods/, aggregator code |
| Active datasets | Regenerate. aa: byte-identical pair_keys → byte-identical splits. nt_cds: inflated pair_keys → different splits. NEW run dirs use `_t099_`; OLD `_id099_` dirs stay archived. |
| Active models (MLP + LGBM) | Retrain. aa: byte-identical metrics expected (regression guard). nt_cds: shift expected (the headline measurement). |

---

## 2. Pre-Phase-2 settle-up

Pair_key plan § 9 open questions, status:

- **§ 9.1 measure deltas**: DONE 2026-06-02 (§ 4 of that plan, 35 % /
  57 %).
- **§ 9.2 idXXX → tXXX bundling**: RESOLVED — bundled in.
- **§ 9.3 v1 builder**: RESOLVED — kept, dedicated branch later.
- **§ 9.4 audit-only stepping stone**: REJECTED by choosing Option B.
- **§ 9.5 publication deadline**: REMOVED.

Pre-flight measurements to take BEFORE touching code (~30 min each):

1. **Baseline performance** — re-run the existing aa and nt MLP +
   LGBM on the test bundles selected in § 3.3, exact seeds +
   bundles, save the metrics CSVs. Cross-check against the matching
   pre-existing runs in `models/flu/July_2025/runs/` to confirm
   reproducibility.

   **DONE 2026-06-02**. Test bundle set was revised (Alex's call):
   `flu_{ha_na,pb2_pb1}_cluster_nt_id099` (exercises cluster_disjoint
   routing) + `flu_{ha_na,pb2_pb1}_regimes` (exercises pair-universe
   dedup, no cluster routing). 8 baseline rows recorded in
   `results/flu/July_2025/runs/phase2_preflight/baselines_manifest.csv`.

   **Reproducibility verdict: ε = 0 on the full pipeline.** Stage 3
   pair tables, MLP metrics, and LGBM metrics are all byte-identical
   across independent re-runs (same seed + same data + same code).
   No GPU nondeterminism observed on V100. The regression-guard
   tolerance threshold in § 4.4 collapses from "≤ seed-variance band
   (~0.005-0.01)" to "= 0"; any aa drift post-migration is a real
   signal, not rerun noise.
2. **Active dataset inventory**: snapshot `data/datasets/flu/July_2025/runs/`
   to a manifest CSV (bundle, alphabet, threshold, n_pairs, mtime).
   Used to confirm regen coverage in § 3.3 and to identify dirs that
   will get the new `_t099_` naming convention.
3. **Active bundle inventory**: list every bundle that sets
   `cluster_alphabet: nt` (must be updated) + every bundle with
   `_idXXX_` in its filename (must be renamed). Grep + manifest CSV.

---

## 3. Implementation sub-phases

### 3.1 Code changes (no data touched yet)

**Status: DONE 2026-06-02 (commit `a3aeb53`).** 40 files changed
(12 .py files, 27 bundle YAMLs, conf/dataset/default.yaml).

**One PR commit, all schema-level changes:**

- `src/utils/clustering_utils.py`:
  - Rename `_COLS_BY_ALPHABET` key `'nt'` → `'nt_cds'`.
  - Add `'nt_ctg'` placeholder that maps to its own seq/hash cols and
    raises `NotImplementedError` from `export_function_fasta` / dispatcher.
  - Update validation messages, docstrings, type hints.
- `src/utils/clustering_utils.py::parse_cluster_tsv`:
  - Add `alphabet: str` parameter (required).
  - Dispatch output column name from `_COLS_BY_ALPHABET[alphabet]['hash_col']`.
- `src/datasets/_split_helpers.py::attach_cluster_ids`:
  - Drop the `lookup.rename(columns={'seq_hash': pos_col_a, ...})` hack;
    direct merge on the alphabet-correct column name.
  - `pos_hash_col` parameter validation now ensures alignment with the
    cluster parquet's column.
- `src/datasets/_pair_helpers.py::canonical_pair_key`:
  - Add `alphabet: str = 'aa'` parameter; doc explains the alphabet-
    specific behavior. Body is unchanged (still `'__'.join(sorted(...))`
    on whatever the caller passes), but callers must now pass the
    right hash family.
- `src/datasets/dataset_segment_pairs_v2.py`:
  - `create_positive_pairs_v2`: derive `pair_key` from the alphabet-
    appropriate hash columns (aa → `seq_hash_{a,b}`; nt_cds →
    `cds_dna_hash_{a,b}`).
  - `compute_cooccur_pairs` (in `_pair_helpers`): build the set
    per alphabet; v2 caller picks which.
  - `create_negative_pairs_v2`: `cooccur_pairs` + `forbidden_pair_keys`
    must match the alphabet of `pair_key` in the dedup loop.
  - `split_dataset_v2`: global pair_key dedup at the right alphabet.
- `src/analysis/build_mmseqs_clusters.py`:
  - Update `cluster_one_function_one_threshold(alphabet)` validation:
    `'aa' | 'nt_cds'` (with `nt_ctg` reserved-not-implemented).
  - CLI `--alphabet` choices follow.
- `conf/bundles/*.yaml`: 4 active bundles have
  `split_strategy.cluster_alphabet: nt` — rename to `nt_cds`. Sweep.
- `conf/dataset/default.yaml`: comment update.
- `conf/virus/flu.yaml`: no change needed.

**idXXX → tXXX cascade** (bundled per Alex 2026-06-02):

- `src/analysis/build_mmseqs_clusters.py::_threshold_label`:
  `f"id{pct:03d}"` → `f"t{pct:03d}"`. Removes the TODO at line
  119. Affects all NEW cluster parquet dirs going forward.
- 12+ reader files emit or display `idXXX` patterns; sweep each:
  - `cluster_disjoint_feasibility.py:65`,
    `aa_nt_cluster_crosstab.py:59`,
    `dataset_segment_pairs_v2.py:2891-2892` (docstring path
    examples — also reflect the dir change to `tXXX/`),
    `aggregate_mmd_single_slot_sweep.py:336, 392, 451`,
    `bipartite_graph_properties.py:71` (`_DEFAULT_THRESHOLDS`),
    `coupling_visualizations.py:148, 176`,
    `cluster_pair_weight_topk.py:99` (`_DEFAULT_THRESHOLDS`),
    `cluster_pair_coupling_precheck.py:193`,
    `plot_idxx_sweep_geometry.py:171` (script body; filename rename
    `plot_idxx_sweep_geometry.py` → `plot_tXX_sweep_geometry.py`
    is OUT of scope for this PR — too much cross-doc cascade for a
    cosmetic filename).
- Bundle YAML internal refs: `cluster_id_path:
  .../clusters_aa/id099/...` → `clusters_aa/t099/...`. Mechanical
  sweep across the active bundle set.
- Bundle filename renames: `flu_*_cluster_*_id095_*.yaml` →
  `flu_*_cluster_*_t095_*.yaml`. ~25-30 files. Includes the 6
  HAonly + 6 PB2only single-slot bundles + 2 cluster_id99
  bilateral pairs + a handful of legacy bundles. Cross-ref sweep
  in CLAUDE.md, .claude/memory.md, docs/methods/, docs/plans/
  (active only — done/ and results/ left as historical),
  aggregator code that hard-codes bundle names.
- NEW run dirs (Phase 2 onward) use `_t099_` in their
  timestamped names; OLD `_id099_` run dirs stay as-is, archived
  in their existing locations. Aggregator path globs need to
  handle both patterns — survey aggregators first; document
  any that need updating.

**Validation at end of 3.1**: import-check + unit-level smoke
test of `parse_cluster_tsv` and the new alphabet-aware
`canonical_pair_key`. Smoke-test that `_threshold_label(0.99)`
returns `"t099"`. No data regen yet.

### 3.2 Cluster artifact regeneration (data-only)

**One PR commit, regen + dir flip:**

- Archive old: rename
  `data/processed/flu/July_2025/clusters_aa/` →
  `data/processed/flu/July_2025/clusters_aa_archive_<YYYYMMDD>/` and
  `data/processed/flu/July_2025/clusters_nt/` →
  `data/processed/flu/July_2025/clusters_nt_archive_<YYYYMMDD>/`.
  Following the precedent of `clusters_aa_easy_cluster_archive/`
  (memory.md): preserves rollback path without inflating active
  state. Archive dirs are gitignored (1-2 GB each).
- Regenerate `data/processed/flu/July_2025/clusters_aa/tXXX/`
  parquets via `build_mmseqs_clusters.py` — same clustering
  decisions, new column name + new threshold-label prefix. ~10 min
  wall (Phase 1 timing data, 64 threads).
- Generate `data/processed/flu/July_2025/clusters_nt_cds/tXXX/`
  parquets (new dir, new prefix). ~10 min wall.

**Validation at end of 3.2 (§ 4.1)**: cluster decisions byte-
identical for both alphabets vs the archived old parquets.
Pre-Phase-2 cluster_rep mapping == post-regen cluster_rep mapping
(mod column name + dir/threshold-label rename).

### 3.3 Dataset regeneration (test subset only)

**One PR commit, narrow scope to confirm the migration works
before bulk regen.**

Test bundle set (4 bundles total):

- HA-NA bilateral cluster_t099 aa (regenerated from existing
  `flu_ha_na_cluster_t099.yaml` → `flu_ha_na_cluster_t099.yaml`)
- HA-NA bilateral cluster_t099 nt_cds (regenerated from existing
  `flu_ha_na_cluster_nt_t099.yaml` → `flu_ha_na_cluster_nt_cds_t099.yaml`)
- PB2-PB1 bilateral cluster_t099 aa (similar)
- PB2-PB1 bilateral cluster_t099 nt_cds (similar)

Picked because (a) bilateral cluster_id99 is the canonical
operating point (per memory.md), (b) both pairs at the same
threshold give the aa/nt_cds asymmetry on two biological systems
with known coupling differences (HA-NA's antigenic-subtype
coupling vs PB2-PB1's polymerase co-conservation), (c) 4
datasets is small enough to be fast (~30 min total) and large
enough to surface bugs.

- aa datasets: expected byte-identical pair_keys, splits, audit
  outputs. Validate as § 4.2 negative control.
- nt_cds datasets: expected inflated pair_key universe (+34.9 % on
  HA-NA, +57.0 % on PB2-PB1). Validate magnitude matches the § 4
  measurement; record per-bundle counts in audit.

If § 4.2 (datasets) + § 4.4 (aa metrics) + § 4.5 (nt_cds metrics)
all pass on the test set, expand to the full active-bundle set as
a follow-up commit (or follow-up PR). Don't bulk-regen
~50-80 datasets until the small set validates.

### 3.4 Model retraining (MLP + LGBM only; no 1-NN)

**One PR commit per (bundle × algorithm) batch on the test set:**

- aa retraining: same bundles, same seeds, regenerated datasets.
  Expected: byte-identical training rows → byte-identical metrics
  (within reproducibility noise).
- nt_cds retraining: same bundles, same seeds, regenerated datasets
  with inflated pair-universe. Expected: metric shift. Direction +
  magnitude is the result.
- Wall time: per CLAUDE.md "Stage 4 (training) is modest", ~30 min
  per (bundle × seed × algorithm) for MLP; LGBM is faster. With 4
  bundles × 3 seeds × 2 algorithms = 24 runs, budget ~6-8 hours
  (or parallel).
- Explicit exclusion: 1-NN cosine-margin baseline. Skipped per Alex
  (long wall time on inflated nt; aa MLP+LGBM consistency is
  sufficient regression signal).

After the test set passes § 4.4 + § 4.5, expand retraining to the
full set of active bundles in a follow-up commit.

---

## 4. Validation matrix

The aa/nt asymmetry is the central design of the test plan. Aa
serves as the regression guard (anything that's NOT supposed to
change MUST stay byte-identical); nt provides the headline
measurement (the bias's magnitude).

### 4.1 Cluster parquets

| | aa | nt_cds |
|---|---|---|
| Cluster count per (function, threshold) | byte-identical | byte-identical |
| Cluster-rep mapping per `seq_hash` (aa) / `cds_dna_hash` (nt) | byte-identical | byte-identical |
| Column name | `seq_hash` (unchanged) | `cds_dna_hash` (was `seq_hash`) |
| Dir name | `clusters_aa/idXXX/` (unchanged) | `clusters_nt_cds/idXXX/` (was `clusters_nt/`) |

**Test**: re-run validation script from Phase 1
(`/tmp/validate_phase1_*.py` shape) on M1 + HA + NA × both
alphabets at t099. Map keys via the new column name.

### 4.2 Datasets

| | aa | nt_cds |
|---|---|---|
| Pair universe size | byte-identical (HA-NA: 58,826) | inflated (HA-NA: 79,347, +34.9 %) |
| `pair_key` set in train/val/test | byte-identical | shifted |
| Split sizes | byte-identical | shifted |
| Negative pair counts | byte-identical | shifted |
| Audit JSON content | byte-identical | shifted |

**Test**: pick one HA-NA aa bundle (bilateral cluster_id99) +
one nt_cds bundle, regenerate, diff train_pairs.csv /
val_pairs.csv / test_pairs.csv at the row level:
- aa: row-by-row identical when sorted by pair_key.
- nt_cds: row count grows by ~35 %; spot-check that the new pair_keys
  are alphabet-distinguishable variants of the protein universe.

### 4.3 Audit shape

| | aa | nt_cds |
|---|---|---|
| Bipartite-CC sizes (cluster_id_a × cluster_id_b) | byte-identical | shifted (larger universe → potentially different CC structure) |
| `seq_hash_overlap`, `dna_hash_overlap` audit fields | byte-identical for aa | nt audit re-keyed to `cds_dna_hash` |
| `cluster_alphabet` audit field | `aa` (unchanged) | `nt_cds` (was `nt`) |

### 4.4 Model retraining — aa (regression guard)

For each (bundle, seed, algorithm) ∈ {aa bundles} × {seeds in use}
× {MLP, LGBM}:

- Compute |Δ metric| for F1, AUC-ROC, MCC.
- Threshold: **ε = 0** (per § 2 pre-flight: end-to-end pipeline is
  bit-exact across reruns; same seed + same data + same code yields
  byte-identical metrics.csv).
- **Any aa metric drift > 0 is a Phase-2 bug**, not a methodology
  change. Fix before proceeding.

### 4.5 Model retraining — nt_cds (positive expectation)

For each (bundle, seed, algorithm) ∈ {nt_cds bundles} × {seeds in
use} × {MLP, LGBM}:

- Compute Δ metric for F1, AUC-ROC, MCC.
- Expected direction: ambiguous. The bias direction analysis (§ 2
  of pair_key plan) said nt training was under-counting its training
  diversity. Adding distinct pair_keys could:
  - Improve metrics (more genuine training rows; better generalization)
  - Hurt metrics (more correlated rows from the same protein-pair
    inflate within-split similarity; harder generalization)
- Magnitude is the headline result. Report per-bundle Δ; expect
  noticeable swings on PB2-PB1 (where the universe inflation is
  largest at +57 %).
- Update the relevant Key Findings in CLAUDE.md / docs/results/
  with the new numbers + the pair_key convention they were measured
  under.

---

## 5. Risks + rollback

| Risk | Mitigation | Rollback path |
|---|---|---|
| aa metric drift (regression bug) | § 4.4 catches it before nt_cds work starts | Revert § 3.1-3.3 commits; fix the bug; retry |
| nt_cds dataset build fails (e.g., negative sampler can't find enough valid pairs in inflated universe) | Coverage-first sampler from v2 is robust; pre-flight feasibility check on one bundle first | Pin nt_cds bundles to old protein pair_key; ship Phase 2 with only the cosmetic enum rename, defer pair_key change to follow-up |
| Bipartite mega-CC behaves worse on inflated nt_cds universe (e.g., feasibility ceiling tightens further) | Cluster_disjoint feasibility CSV regen catches it pre-dataset | Document the new feasibility curve; bundles that go infeasible get marked NOT_MAINTAINED |
| Bundle-schema breakage if any consumer reads old `cluster_alphabet: nt` value | Grep + audit before merging; all bundle YAMLs flipped in one commit | Single revert commit on the bundle YAML changes |
| Old cluster parquets still on disk at `clusters_nt/` cause stale-cache hits | Rename to `clusters_nt_archive_<YYYYMMDD>/` after § 4.1 validation passes (don't delete; archive preserves rollback) | Re-extract from `clusters_nt_cds/` (lossless) |
| Trained-model `training_info.json` reads `cluster_alphabet: nt` from history | Provenance only; trained weights unaffected | Mark old runs with the old alphabet enum in the manifest; new runs use the new value |

---

## 6. Affected publication numbers

Per pair_key plan § 5.4: do NOT re-run old experiments. Pin to old
convention with explicit labels. The CLAUDE.md "Key Experimental
Findings" entries that need a "(pair_key=protein)" label appended:

- 2026-05-24 single-slot HA-NA HAonly idXX-sweep — uses protein
  pair_key, stays as-is.
- 2026-05-26 PB2-PB1 PB2only idXX-sweep — same.
- 2026-05-15 Experiment B-nt — uses protein pair_key; the
  "1-NN ≥ LGBM at every routing" finding is conditional on the
  protein-collapsed universe. Add caveat.
- 2026-05-31 multiplicity-skew (Task 4) — explicitly tied to
  protein pair_key collapse; the finding survives but the framing
  needs an addendum noting that nt_cds pair_key would weaken the
  collapse mechanism.

New experiments under Phase 2 should always cite the alphabet they
used.

---

## 7. Commit shape

Seven commits on `feature/phase2-pair-key-migration`, in order:

1. **DONE 2026-06-02 (commit `a3aeb53`)** — `refactor(clustering): alphabet enum {aa,nt_cds,nt_ctg}; parse_cluster_tsv writes alphabet-specific hash column; threshold label id -> t` — § 3.1 code only. 40 files: clustering_utils, _split_helpers, dataset_segment_pairs_v2, build_mmseqs_clusters, 9 reader files, 27 bundle YAMLs + conf/dataset/default.yaml. HEAD intentionally non-runnable until commit 3 regenerates cluster parquets at the new tXXX/ dirs.
2. **DONE 2026-06-02 (commit `852a2c2`)** — `chore: rename idXXX bundle filenames to tXXX + targeted cross-ref sweep`. 40 files: 34 bundle renames + 6 cross-ref edits in active docs + CLAUDE.md "Threshold notation" rewrite. Sweep protects run-dir refs (`dataset_*_idXXX_*_<ts>`, `training_*_idXXX_*_<ts>`, `baseline_*_idXXX_*_<ts>`) from rewrite since those dirs physically exist on disk; 4 such refs preserved in `phase2_preflight_baselines.md`. HEAD still non-runnable (bundle `cluster_id_path` values point at `clusters_*/tXXX/` which don't exist on disk yet — commit 3 archives + regenerates).
3. **DONE 2026-06-02** — `data(clusters): archive old clusters_{aa,nt}; regenerate cluster_{aa,nt_cds}/tXXX/ parquets under new column convention`. Archive renames: `clusters_aa/` → `clusters_aa_archive_20260603/`, `clusters_nt/` → `clusters_nt_archive_20260603/`. Regen: 88 fresh cells per alphabet (11 thresholds × 8 functions), 64 threads, 11 min aa + 14 min nt_cds wall time. **Validation: 176/176 cells byte-identical cluster_rep mapping vs archived parquets** (`/tmp/validate_commit3_regen.py`; aa cell uses `seq_hash` column, nt_cds uses `cds_dna_hash`, content otherwise identical). Discovered + fixed in this commit: `run_mmseqs_easy_clust` validation still rejected 'nt_cds' (missed in commit 1's sweep — checked the dispatcher in `_clean_for_mmseqs` and `export_function_fasta` but not the mmseqs subprocess invoker). **HEAD now runnable**: bundle `cluster_id_path: clusters_*/t099/...` paths resolve. Data itself is gitignored; this commit lands the bugfix to `clustering_utils.py` + status updates.
4. **DONE 2026-06-02** — `feat(v2): alphabet-specific pair_key for nt_cds; aa unchanged`. Adds `pair_key_alphabet` parameter to `create_positive_pairs_v2`, `create_negative_pairs_v2`, `split_dataset_v2`, `generate_all_cv_folds_v2`, `generate_all_cluster_disjoint_cv_folds_v2`. `build_cooccurrence_set` gains `hash_col` param. New `attach_cds_dna_hash_to_prot_df` helper for orchestrator pre-attach. `_PAIR_COLUMNS` extended with `cds_dna_hash_{a,b}` (populated for nt_cds, NA for aa). `_side` includes `cds_dna_hash` when present in df. Orchestrator (`dataset_segment_pairs.py`) reads `dataset.split_strategy.pair_key_alphabet` from config (default inference: ties to `cluster_alphabet` for cluster_disjoint, else 'aa'). `load_cluster_lookup` accepts either `seq_hash` or `cds_dna_hash` column (Phase 2 alphabet-specific parquets). Additional cleanup discovered during smoke testing: `_validate_v2_config` `cluster_alphabet` check rejected 'nt_cds'; legacy 'nt' value now rejected. Smoke test results: HA-NA cluster_t099 (aa) = 145,972 total pair rows; HA-NA cluster_nt_t099 (nt_cds) = 196,912 = **+34.9 % inflation**, matching the pair_key plan § 4 universe-size delta prediction exactly (79,347 / 58,826 = 1.349).
5. **DONE 2026-06-02** — `data(datasets): regenerate test bundles (HA-NA + PB2-PB1 bilateral t099, aa + nt_cds = 4 datasets) under new pair_key convention`. All 4 datasets built end-to-end in ~5 min each. Plus 2 regimes-bundle re-runs for the aa regression guard (HA-NA + PB2-PB1 regimes). Validation results: (a) nt_cds inflation matches § 4 prediction exactly — HA-NA: 145,972 → 196,912 rows = **+34.9 %**; PB2-PB1: 131,645 → 206,742 rows = **+57.0 %**. (b) aa regimes (pair_key='aa' default for random routing) — `(pair_key, label)` rows **content byte-identical** to pre-flight datasets (HA-NA 146K rows, PB2-PB1 132K rows). CSV bytes differ only because `_PAIR_COLUMNS` was extended with `cds_dna_hash_{a,b}` (NA on aa flow); pair content unchanged. Regression-guard ε = 0 honored at the dataset level.
6. **DONE 2026-06-03** — `data(models): retrain MLP + LGBM on test datasets; aa regression guard + nt cluster before/after measurement`. 12 runs end-to-end (6 MLPs across 2 GPU chains + 6 LGBMs sequential on CPU; ~90 min MLP wall time, ~30 min LGBM). Results manifest at `docs/results/2026-06-03_phase2_postmigration_metrics.md`. Comparisons are all "same bundle, same algorithm, before code change vs after code change". (a) **aa regimes regression guard PASSED at ε = 0** across all 4 (bundle × algorithm) cells × all 3 post-hoc artifacts (12 md5 checks; every byte matched pre-flight). Confirms the pair_key migration is a pure refactor for the aa pipeline at the model-output level. (b) **nt cluster_t099 before/after deltas** — same bundle ID renamed (`flu_*_cluster_nt_id099` → `flu_*_cluster_nt_t099`), cluster ASSIGNMENTS byte-identical pre/post (validated in commit 3), only pair_key semantics changed (protein → CDS DNA). All 4 experiments degrade in aggregate: HA-NA mild (−1 to −2 pp F1, −2 to −4 pp MCC), PB2-PB1 large (MLP −5 pp F1 / −8 pp MCC; LGBM −13 pp F1 / −23 pp MCC / −10 pp AUC-ROC). Per-regime breakdown shows HA-NA degradation concentrates in `host_subtype_only` (+17-22 pp fp_rate) and `host_subtype_year` (+24-26 pp); PB2-PB1 MLP shows uniform degradation across all 8 negative regimes; PB2-PB1 LGBM collapses positive recall (−15.8 pp) AND degrades most negatives. The +34.9 %/+57.0 % pair-universe inflation magnitudes track the HA-NA-mild vs PB2-PB1-large F1-drop ratio. Disentanglement (aa cluster atoms × nt_cds pair_key) is out of scope for Phase 2. Data is gitignored; this commit lands the manifest + plan + memory updates.
7. `docs: Key Findings caveats (pair_key=protein) on past results + new nt_cds numbers from test bundles` — § 6.

**Pre-flight (preceding the 7 commits)**: scoping at `f09d9a1`; baselines + ε=0 finding at `7c6a143`.

If § 4 validation passes on the test set, a follow-up PR
expands § 3.3 + § 3.4 to the full active-bundle set.

---

## 8. Out of scope for this Phase 2

- v1 builder deletion (`src/datasets/dataset_segment_pairs.py`) —
  dedicated branch later per Alex. v1 stays as-is (protein-only
  pair_key); if anyone reactivates it, that's a known divergence
  from v2's new behavior.
- nt_ctg operationalization — Phase 3 of clustering cleanup plan.
  Enum value reserved here (raises `NotImplementedError`) so the
  follow-up doesn't require a second enum migration.
- 1-NN cosine-margin baseline retraining — long wall time on
  inflated nt; aa MLP+LGBM consistency is sufficient regression
  signal per Alex.
- Re-publication of past experiments — pinned to protein pair_key
  with caveats per § 6.
- Bulk dataset/model regen for the full active-bundle set —
  follow-up PR after the test-set validation in § 4 passes.
- Filename rename of `plot_idxx_sweep_geometry.py` — too much
  cross-doc cascade for a cosmetic rename; the script body's
  internal `idxx` references get the `tXX` treatment but the
  filename stays as a separate micro-PR if desired.

---

## 9. Open questions — resolved

Alex's 2026-06-02 decisions:

1. **Pre-Phase-2 baseline runs**: run fresh baselines on the test
   bundles (§ 3.3 set); cross-check against existing
   `models/flu/July_2025/runs/` entries. Existing runs are a
   sanity check, not the anchor — the fresh runs are.
2. ~~Publication deadline~~ — REMOVED from the plan.
3. **Bundle scope**: narrow test set first (4 bundles, § 3.3);
   expand to remaining active bundles only if test validates.
4. **Cluster_nt disposition**: rename to
   `clusters_nt_archive_<YYYYMMDD>/` (don't delete). Archive
   preserves rollback indefinitely.
5. **idXXX → tXXX**: bundled IN (was outside scope in v1 of this
   plan; revised per Alex's pushback). Cascade detail in § 3.1,
   § 3.2, § 7.

---

## See also

- `docs/plans/2026-06-02_clustering_cleanup_plan.md` — Phase 2
  scoping context (clustering side).
- `docs/plans/2026-06-02_pair_key_alphabet_plan.md` — pair_key side
  + the universe-delta measurements driving the decision.
- `docs/plans/done/2026-06-02_clustering_phase1_*.md` (when moved
  to done/) — Phase 1 reference precedent.
- CLAUDE.md "Threshold notation", "Sequence hashes", "aa/nt vs
  protein/DNA" — surrounding conventions.
- `.claude/memory.md` "Recent run outputs (May 2026) — explicit
  file paths" — inventory of active bundles + run dirs for § 2
  pre-flight + § 3.3 regen scope.
