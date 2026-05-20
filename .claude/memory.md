# viral-segmatch — Project Memory

This file is version-controlled in the repo (.claude/memory.md) so it is available on every machine.
Claude: read this at the start of every session. Update it when decisions change or new findings emerge.

---

## Project Summary
Flu A viral segment co-occurrence prediction. ESM-2 protein embeddings (frozen) + MLP binary classifier.
Primary virus: Influenza A. Bunya support exists but NOT actively maintained.

## Pipeline (4 stages)
- Stage 1: `src/preprocess/preprocess_flu.py` → `data/processed/flu/{version}/protein_final.csv` + `genome_final.csv` (run once). Invoked by `scripts/stage1_preprocess_flu.sh`. Unified protein + genome extraction in one pass. (The legacy `preprocess_flu_protein.py` was deleted 2026-05-13 after the unified script absorbed its logic; recoverable via git history if ever needed.)
- Stage 2: `src/embeddings/compute_esm2_embeddings.py` → `data/embeddings/flu/{version}/master_esm2_embeddings.h5` (run once)
- Stage 3: `src/datasets/dataset_segment_pairs.py` → `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/` (per experiment)
- Stage 4: `src/models/train_pair_classifier.py` → `models/flu/{version}/runs/training_{bundle}_{ts}/` (per experiment)
- Shell wrappers: `scripts/stage2_esm2.sh`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh`

## Config System
Hydra + bundle-per-experiment. `conf/bundles/{bundle}.yaml` = one file per named experiment.
Bundle naming: currently inconsistent across generations. Planned general signature (not yet enforced):
  `{virus}_{proteins}[_{n_isolates}][_{slot_transform}_{interaction}][_{data_filter}]`
  e.g. `flu_ha_na_5ks_slot_norm_unit_diff_h3n2` -- renaming existing bundles is a future task.
Config loader: `src/utils/config_hydra.py` via `hydra.compose(config_name="bundles/{name}")`.
No root config -- bundles are loaded directly. `src/utils/config.py` and `conf/config.yaml` deleted (legacy).

## Bundle Organization (see conf/bundles/README.md for full detail)
- Each bundle has `# STATUS: active|ablation|experimental|legacy|not maintained` header.
- Three generations: Gen1 (`flu_base.yaml`), Gen2 (`flu_schema.yaml`), Gen3 (`flu_schema_raw_*` and the active `flu_28_major_protein_pairs_master` chain).
- Base bundles must stay flat (moving them breaks Hydra defaults chains in children).
- Base bundles use the `_base.yaml` suffix (renamed 2026-05-10: `flu.yaml` -> `flu_base.yaml`, `bunya.yaml` -> `bunya_base.yaml`) to disambiguate from `conf/virus/<name>.yaml` which lives in a different config group.
- `conf/bundles/paper/` reserved for publication experiments.
- Best model: `flu_schema_raw_slot_norm_unit_diff` (slot_norm + unit_diff, HA/NA).
- Regime-aware leaves (renamed 2026-05-12): `flu_ha_na_regimes`, `flu_pb2_pb1_regimes` (was `*_neg_regimes`) -- inherit from `flu_ha_na` / `flu_pb2_pb1` and add `dataset.negative_sampling.regime_targets`. Uniform `0.10 x 7 + 0.30 x 1` regime mix (`host_subtype_year` keeps the 0.30 weight; the other 7 regimes are uniform at 0.10). Per-bundle baseline blocks dropped now that defaults live in `conf/baselines/default.yaml`.
- Active HA/NA and PB2/PB1 bundles (`flu_ha_na.yaml`, `flu_pb2_pb1.yaml`) bake in `split_strategy.mode=seq_disjoint` + `hash_key=seq` + Test 3 interaction (`slot_transform=unit_norm`, `interaction=unit_diff+prod`). Earlier dedicated leaves (`flu_ha_na_seq_disjoint.yaml`, `flu_ha_na_metadata_holdout_temporal.yaml`, `flu_ha_na_interactions.yaml`, `flu_{ha_na,pb2_pb1}_human_h3n2*.yaml`) were retired (2026-05-12) now that the equivalent knobs live in the main bundles or in `conf/dataset/default.yaml`.
- Metadata-holdout demo bundles (2026-05-11, current): `flu_ha_na_holdout_year.yaml`, `flu_ha_na_holdout_host.yaml`, `flu_ha_na_holdout_subtype.yaml` exercise the year-range / host / subtype axes of the general `dataset.metadata_holdout` mechanism.

## Key Findings
- ESM-2 `unit_diff` > `concat` on homogeneous data (H3N2-only): AUC 0.96 vs 0.50
- K-mer concat does NOT collapse on H3N2 (AUC 0.985) -- concat failure is ESM-2-specific, not interaction-specific
- K-mer dominates ESM-2 on H3N2: k-mer unit_diff AUC 0.988 vs ESM-2 unit_diff AUC 0.957; k-mers are interaction-agnostic
- K-mer (k=6, 4096-dim) matches or exceeds ESM-2 on mixed-subtype HA/NA (AUC 0.982 vs 0.966-0.975)
- LayerNorm (`slot_norm`) critical for ESM-2 on homogeneous subsets
- Delayed learning on H3N2 + unit_diff: increase patience to 40+
- High FP rate on filtered datasets (year/host/geo) -- root cause identified 2026-05-07 as **demographic-shortcut leakage** (mode #5): FP rate climbs 30-50x from `none_match` to `host_subtype_year` regime, and MORE capacity exploits the shortcut MORE (h=[10] vs h=[200]). Construction-time mitigation landed: regime-aware negative sampling. See `docs/results/2026-05-07_metadata_shortcut_negatives.md`.
- **Temporal holdout**: K-mer AUC 0.941 vs ESM-2 AUC 0.891 (train 2021-2023, test 2024); k-mers generalize better across flu seasons
- **Experiment B-nt feasibility ceiling = aa ceiling on Flu A (2026-05-15)**: nt CDS-level cluster_disjoint hits the same bipartite mega-component collapse as aa. Only id100 and id099 are operable on the full corpus; id095 and below dump >98% of pairs into one component on BOTH alphabets. The corpus metadata structure (8 segments × 16 dominant HxNy subtype × host × year cells) dominates the alphabet choice. The hope that nt's synonymous diversity would unlock lower-threshold splits did not pan out. `docs/results/2026-05-15_cluster_disjoint_nt_results.md`.
- **1-NN cosine margin >= LGBM at every cluster_disjoint routing (2026-05-15)**: head-to-head on 8 cells (HA/NA × PB2/PB1 × {seq_disjoint, aa id099, nt id100, nt id099}). 1-NN matches LGBM at id100/seq_disjoint cells (within ~1 pp F1) and OUTPERFORMS LGBM at id099 cells (+16 pp F1 on HA/NA aa id099, +7 pp on PB2/PB1 aa id099). Going-in hypothesis "1-NN drops more than LGBM under cluster_disjoint" did NOT hold. Interpretation: cluster_disjoint weakens the near-neighbor signal gradually (every test pair still has a closest train pair, just farther away), and 1-NN stays well-calibrated under that weakening while LGBM's tree splits rely on signal that does not generalize across the cluster boundary. The "MLP/LGBM vs 1-NN" leakage doctrine is informative as a residual-leakage gauge but does NOT by itself confirm cluster_disjoint removed leakage. See `docs/results/2026-05-15_cluster_disjoint_nt_results.md` § "1-NN cosine margin (leakage upper bound)".

## Roadmap (02/10/2026 meeting + March 2026 updates) -- for publication
1. Cross-validation -- IMPLEMENTED, needs end-to-end run
2. Large dataset (full Flu A ~100K isolates, HPC) -- supported, not yet run
3. Temporal holdout -- IMPLEMENTED, needs dedup fix + re-run
4. K-mer + MLP -- DONE; k-mer + XGBoost/LightGBM still TODO
5. PB2/PB1 + H3N2 bundle -- optional (one new bundle)
11. All protein pairs (C(8,2)=28 pairs of 8 major proteins) -- NOT IMPLEMENTED, HPC
12. FP/FN ratio diagnosis + mitigation -- NOT IMPLEMENTED; see `roadmap_v1.md` Task 12
    - Diagnostics first (embedding distances, probability histograms, pair-level metadata matrix)
    - Data-centric: hard negative mining (highest priority), negative ratio, curriculum learning
    - Model-centric: focal loss, contrastive learning (if simpler approaches fail)

## Publication Strategy (March 2026)
- **Paper 1 (biology, primary):** Segment matching for data remediation + surveillance.
  Target: Bioinformatics / PLOS Comp Bio / Genome Biology. See `paper_outline_v1.md`.
- **Paper 2 (ML, follow-up):** ESM-2 concat collapse + GenSLM. Target: NeurIPS/ICML workshop.
- Paper outline: `paper_outline_v1.md` (v1), `paper_outline_v2.md` (v2, current)
- Applications: data remediation (BV-BRC), wastewater surveillance, reassortment detection (future)

## Directory Structure (post-cleanup Feb 2026)
- `.claude/` -- settings.json (permissions) + memory.md (this file)
- `eda/` -- exploratory scripts (bunya EDA moved here; NOT pipeline)
- `examples/` -- HuggingFace reference scripts (NOT pipeline)
- `flu_genomes_eda.py` stays in `src/preprocess/` -- generates flu_genomes_metadata_parsed.csv (pipeline input)

## Not Maintained
- `src/preprocess/preprocess_bunya_protein.py`, `conf/bundles/bunya_base.yaml` (renamed from `bunya.yaml` in the 2026-05-10 base-bundle cleanup; still flagged not-maintained in its STATUS comment)
- `old_scripts/` was deleted entirely on 2026-05-13; recover via git history (`git log --diff-filter=D --diff-filter=R -- old_scripts/`) if ever needed.

## Removed (2026-05-11)
- `dataset.year_train` / `dataset.year_test`: legacy temporal-holdout config keys + the v1 `generate_temporal_split` function. Replaced by the general `dataset.metadata_holdout`. Old bundles using these keys raise at config-load time with a migration message pointing at `docs/plans/done/2026-05-11_metadata_holdout_plan.md`.
- `unknown_metadata_neg` regime: retired as the 9th regime. Sampler is now 8 regimes; null on any axis classifies as no-match on that axis via the existing 8-tuple mapping. Validator rejects the key in `regime_targets`. Bundles updated.
- `performance_metrics_by_metadata.png` + `error_rates_by_metadata.png`: both halves of the old `_plot_errors_by_metadata` are deprecated. The function is deleted. The level2_by_{host,subtype,year_bin}.png trio (both-sides-matching, with mixed/unknown/other residual buckets) subsumes the performance side; `error_by_match_count.{csv,png}` + the per-FP/FN drill-down CSVs (`false_positives_detailed.csv`, `false_negatives_detailed.csv`) cover the error-direction side.
- `error_analysis_by_host.csv` / `error_analysis_by_hn_subtype.csv` / `error_analysis_by_year.csv`: the `legacy_axes` loop in `analyze_errors_by_metadata` is gone. Those three CSVs fed only the deleted plot and used the same both-sides-matching schema as `level2_by_*.csv` but with fewer metrics and no residual buckets.
- `segment_metrics.csv`: no longer emitted by the default v2 pipeline. Degenerate under v2 (schema_ordered fixes a single seg_pair, so the CSV was a one-row restatement of `metrics.csv`). The function `analyze_segment_performance` stays in the module for v1 / future cross-protein runs but is not invoked from `main()`.

## Temporal Holdout (IMPLEMENTED — initial runs complete)
- Bundles: `flu_schema_raw_slot_norm_unit_diff_temporal` (ESM-2), `flu_schema_raw_kmer_k6_slot_norm_unit_diff_temporal` (k-mer)
- Train 2021-2023 (~20K isolates), val+test 2024 (~17K isolates)
- Notable subtype shift: H5N1 24%→41%, H3N2 40%→32% (2024 avian flu surge)
- **Pair-key dedup issue**: 42% of val/test pairs removed (positive pair_keys overlap with train due to same strains across years), creating 25/75 label imbalance in val/test. Needs fix before publication — likely disable dedup for temporal mode (approach A). See plan doc.
- **Initial results** (with dedup artifact, threshold=0.5):
  - ESM-2: AUC 0.891, F1 0.734, FP/FN=64.1
  - K-mer (k=6): AUC 0.941, F1 0.832, FP/FN=11.6
  - K-mers substantially outperform ESM-2 on temporal holdout; gap wider than on random splits
  - Both show AUC drop vs random splits (~0.97), confirming genuine temporal difficulty
- See `docs/plans/temporal_holdout_plan.md` for full analysis and results

## In Development
- `src/utils/dna_utils.py` -- DNA QC utilities (summarize_dna_qc complete, clean_dna_sequences untested)

## Recently Promoted to Production
- **2026-05-20: cluster sweep cleanup + docs/results migration batch.** Multi-commit cleanup of the clustering doc + script family:
  - `seq_redundancy_per_function.py`: append/merge stats CSV (commit `74e798e`); skip missing functions for nt mode (`67ddb62`); strict-to-loose threshold ordering in auto-gen markdown (`ce4e408`). Default output now `<out_root>/redundancy_summary.md`, not docs/.
  - `cluster_disjoint_feasibility.py`: default `--out_csv` → `results/.../cluster_disjoint_feasibility/feasibility_<pair>_<alphabet>.csv` (commit `8c9a733`). 4 stale CSVs in docs/results/ + 2 stale seq_redundancy markdowns deleted (`6e6fcb9`).
  - `cluster_analysis_summary.py`: float-precision fix in `build_mutations_tolerated` (`int(L*(1-t))` → `int(L-ceil(L*t))`, commit `67ddb62`); defensive `keep_default_na=False` on `compute_length_stats` (`25280e2`).
  - `clustering_overview.md` §8 rewrite with one-unit-resolution PB2/PB1 collapse trajectory (commit `f91b805`); §9 feasibility table extended to id098/097/096; §2.3/§5/§7/§10 freshness fixes (`ce4e408`).
  - `leakage_definitions.md` Exp 5 status flipped "pending" → "IMPLEMENTED 2026-05-15" with results refs.
  - New convention: `function_short='NA'` (Neuraminidase) is the canonical pandas-NaN trap. Any CSV with that column must use `keep_default_na=False, na_values=['']`. Documented in CLAUDE.md Conventions; defensive guards added at consumer sites.
  - DataSAIL Phase 0 bake-off PAUSED with quantified route-vs-drop finding: C2 drops 51–71% of HA/NA pairs across K∈{10,50} × ε∈{0.05,0.50}; I2 has no objective function (`solve(1, constraints, ...)` — feasibility-only). Plan + results at `docs/plans/2026-05-19_datasail_bakeoff_plan.md` and `docs/results/2026-05-20_datasail_phase0_results.md`. New `BACKLOG.md` tracks remaining forward-looking items.
- **2026-05-18: AUC metric names standardized.** Project-wide rename to a single convention. Snake_case identifiers (dict keys, CSV columns, variable names, config values) are `auc_roc` and `auc_pr`; display strings (titles, log messages, doc prose) are `AUC-ROC` and `AUC-PR`. sklearn function names (`roc_auc_score`, `average_precision_score`) are external and left as-is. Concrete changes: training script's early-stop key `'auc'` → `'auc_roc'`; history keys `val_auc`/`train_auc` → `val_auc_roc`/`train_auc_roc`; `compute_pair_metrics` dict key `'auc'` → `'auc_roc'`; `analyze_stage4_train.py` and aggregators' `'avg_precision'` → `'auc_pr'`; `_HIGHER_IS_BETTER_METRICS` updated; YAML comments in `conf/training/base.yaml` and `conf/bundles/bunya_base.yaml` updated. Display strings in 2 docs flipped from `PR-AUC` to `AUC-PR`. Existing run artifacts (metrics.csv, training_info.json from previous runs) are NOT back-compat — old runs need re-training or will fail to load in the aggregators. Branch: `chore/standardize-auc-metric-names`.
- **2026-05-15: clustering methodology consolidated**. New comprehensive methods doc `docs/methods/clustering_overview.md` (~640 lines) for a reader unfamiliar with mmseqs2 / sequence-space clustering — covers the mmseqs cascade, the three knobs (`--min-seq-id`, `-c`/`--cov-mode`, `--dbtype`), per-function corpus redundancy on Flu A, mutations-tolerated-per-threshold tables for both alphabets, bipartite-component routing math, and the feasibility ceiling. New post-hoc structural script `src/analysis/cluster_analysis_summary.py` emits `cluster_summary.csv`, `mutations_tolerated_table.csv`, and three plots (unique-sequence retention, cluster collapse, bipartite feasibility) under `results/flu/{version}/runs/cluster_analysis/`. Two renames for symmetry: `src/analysis/protein_redundancy_per_function.py` -> `seq_redundancy_per_function.py` (the script handles both alphabets), and `data/processed/flu/{version}/clusters/` -> `clusters_aa/` (pairs with `clusters_nt/`). The renamed seq_redundancy script now writes `<out_root>/runtime.json` with per-(function, threshold) wall times + a rollup; measured: aa easy-cluster 3,011 s total for 50 runs vs nt easy-linclust 986 s for 48 runs (median 4.8 s vs 6.7 s; max 570 s aa PA@id100 vs 217 s nt PB1@id100). Routing-equivalence reference section added to `docs/methods/leakage_definitions.md` covering aa cluster_id100 ≈ seq_disjoint hash_key=seq vs nt cluster_id100 ≠ seq_disjoint hash_key=dna and the flag-by-flag mmseqs argument semantics.
- **2026-05-15: Experiment B-nt end-to-end** (nt-level cluster_disjoint). Stage 1.5 CDS extractor (`src/utils/cds_utils.py` + `src/preprocess/extract_cds_dna.py`) emits `data/processed/flu/{version}/cds_final.parquet` (868K rows on Flu A July 2025, 871 MB, 100% translate-back-validated for the 8 majors). IUPAC-aware translator resolves synonymous codons. `parse_location` rejects `length<=0` (BV-BRC sentinel for incomplete spliced annotations; ~30% of M42 rows). Spliced extraction validated on the real corpus (M2 99%, NEP 99.7%, M42 68% with -1-sentinel drops, NS3 99.9%, PA-X 99.6%, PB2-splice 100%). Stage 3 wiring: new `dataset.split_strategy.cluster_alphabet: aa|nt` (default `aa`); when `nt`, attaches `cds_dna_hash_{a,b}` to pos_df via `_pair_helpers.attach_cds_dna_hash_to_pos_df` and routes the bipartite-CC on those columns. `cluster_disjoint_route_pos_df` and `attach_cluster_ids` gain `pos_hash_col` param (default `seq_hash`). 4 new bundles: `flu_{ha_na,pb2_pb1}_cluster_nt_id{100,099}.yaml` (id095/id090 deleted as infeasible per the bipartite-CC pre-flight). nt cluster artifacts at `data/processed/flu/{version}/clusters_nt/id{NN}/`. mmseqs2 uses `--dbtype 2` for nt (not `--search-type 3` — easy-cluster/linclust accept `--dbtype` through createdb). KEY FINDINGS: (a) bipartite mega-component collapse hits at the SAME thresholds on aa and nt on Flu A (only id100/id099 feasible); the corpus structure dominates the alphabet choice. (b) 1-NN cosine margin >= LGBM at every routing, with the widest gap at aa id099 (1-NN beats LGBM by 16 pp F1 on HA/NA, 7 pp on PB2/PB1) — opposite of the going-in hypothesis. Read: cluster_disjoint weakens the near-neighbor signal gradually, not absolutely. Plot script: `src/analysis/plot_aa_vs_nt_cluster_disjoint.py`. Plot/CSV: `results/flu/July_2025/runs/cluster_aa_vs_nt/cluster_aa_vs_nt.{png,csv}`. Plan: `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` § B-nt. Results: `docs/results/2026-05-15_cluster_disjoint_nt_results.md`.
- **2026-05-15: NaN-aware dual-level quality filter at Stage 1**. `apply_protein_basic_filters` now drops on `quality=='Poor'` (genome-level) OR `feature_quality=='Poor'` (per-protein, set only on the 8 majors). `apply_genome_basic_filters` adds the analogous `contig_quality=='Poor'` drop. `NaN` means "no claim made" (legitimate BV-BRC state for auxiliary proteins on `feature_quality`) and is NOT treated as Poor — filter only fires on the explicit `'Poor'` string. On the Flu A July 2025 corpus this is a no-op (0 rows dropped); defense-in-depth for future GTO releases. preprocess.md tech-debt items #5 and #6 marked Resolved.
- **2026-05-15: regime-aware coverage phase** (`dataset.negative_sampling.regime_aware_coverage: bool`, default false). New `_negative_regime_sampling.py` helpers (`COVERAGE_PRIORITY_CHAIN`, `build_cell_regime_partners`) + new branch in `create_negative_pairs_v2` that walks the priority chain (host_subtype_year first, none_match last) with per-regime budget=10 and a last-resort uniform fallback for coverage guarantee. Surfaced via `rejection_stats['coverage_regime_aware']` (per-regime attempts/acceptances, `fell_back_to_uniform`). Validated on tight bundles (Phase 5 of plan): host_subtype_year TNR +25 pp (HA/NA) / +30 pp (PB2/PB1), FPR −22 pp / −27 pp, on the same hard test set. Applied to cluster_id99 cells (`docs/results/2026-05-15_cluster_id99_racov_results.md`): FPs dropped 65–84% across 4 cells with AUC-PR essentially unchanged. 8 unit tests in `tests/test_regime_aware_coverage.py`. Plan: `docs/plans/done/2026-05-14_regime_aware_coverage_plan.md`. Bundles: `flu_{ha_na,pb2_pb1}_tight_racov.yaml`, `flu_{ha_na,pb2_pb1}_cluster_id99[_r3]_racov.yaml`. KNOWN COST: full-corpus Stage 3 builds slow from ~1.5 min to 16–62 min when n_cells is large (2,916 cells); two optimizations sketched in the cluster_id99 results doc (lower per-regime budget, cache first-regime per self_cell).
- **2026-05-15: cluster_disjoint routing** (`dataset.split_strategy.mode: cluster_disjoint`, requires `cluster_id_path` + optional `cluster_id_threshold`). mmseqs2 protein clustering at thresholds {1.00, 0.99, 0.95, 0.90, 0.80}; pair routing by bipartite-CC LPT-greedy on `(cluster_id_a, cluster_id_b)` (mirrors seq_disjoint but on cluster_ids). Routing helper `src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df`. Per-function cluster artifacts live in `data/processed/flu/July_2025/clusters_aa/id{NN}/{HA,NA,PB2,PB1,...}_cluster.parquet` + `combined_cluster.parquet` (the `clusters_aa/` dir was renamed from `clusters/` 2026-05-15 for symmetry with `clusters_nt/`). KEY FINDING: bipartite mega-component collapses to >80% of deduped pairs below id099 on the full corpus (`data/processed/flu/July_2025/clusters_aa/redundancy_summary.md` § feasibility table; pre-flight script `src/analysis/cluster_disjoint_feasibility.py`). On flu A, only id100 (≈ seq_disjoint) and id099 are non-trivially feasible. id099 vs seq_disjoint shows F1 −14 to −27 pp on same dataset — direct measure of mode #4 cluster leakage. Plan: `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` (covers both Experiment B = aa, IMPLEMENTED 2026-05-15, and Experiment B-nt = nt, also IMPLEMENTED 2026-05-15 — see the "Experiment B-nt end-to-end" entry above). Bundles: `flu_{ha_na,pb2_pb1}_cluster_id{99,95,90}.yaml` (id095/id090 marked infeasible) + `_r3` ratio=3 variants + `_racov` variants. Results: `docs/results/2026-05-14_cluster_disjoint_id99_results.md`. Methods: `docs/methods/clustering_overview.md`.
- **2026-05-15: h5py made lazy** across `src/utils/{esm2,embedding}_utils.py`, `src/models/{train_pair_classifier,_pair_features}.py`. ESM-2 paths get `import h5py` inside the functions that touch HDF5; k-mer-only paths no longer crash on env-level libhdf5/h5py ABI mismatch (which happens after conda installs that pull a different libhdf5 — observed after `conda install -c bioconda mmseqs2`).
- **2026-05-12/13 docs + codebase cleanup batch**:
  - `docs/methods/` fully refreshed (preprocess, leakage_definitions, kmer_features, feature_normalization, pipeline_overview) with corpus-backed claims, current file:line refs, and current production settings (seq_disjoint, unit_norm, 4-baselines, 8-regime). Pruned ~75 lines of paper-defense prose and trivial placeholder subsections.
  - New methods doc: `docs/methods/gto_format_reference.md` — comprehensive GTO JSON schema walk-through with corpus statistics; companion `docs/examples/example_gto_excerpt.json`.
  - New methods doc: `docs/methods/dataset_construction_v2_workflow.md` — phase-by-phase v2 builder walkthrough with verified numbers from `dataset_flu_ha_na_regimes_20260512_114205`; companion frozen `docs/examples/dataset_flu_ha_na_regimes_20260512_114205_dataset_stats.json` (HEAD at run time: 45a2ed3).
  - New plan: `docs/plans/2026-05-12_codon_aware_kmer_features_plan.md` — interpretability/feasibility note on whether current k-mer feature attributions can be read as codon-level signal (short answer: no; current stride-1 features mix all three frames + UTRs + introns).
  - Merged plan: `docs/plans/2026-05-12_model_validation_plan.md` — combines the former `LEARNING_VERIFICATION.md` (Part A, IMPLEMENTED training-mechanics checks) + the former `baseline_validation_experiments_plan.md` (Part B, DRAFT signal-source adversarial ablations). Both source docs deleted.
  - Doc deletes (top-level `docs/`): STRESS_TEST_VISUALS, DUPLICATE_SEQUENCE_HANDLING, geo_location_analysis, PAIR_EMBEDDING_ORDERING_AND_NORM_ARTIFACT, PROJECT_REPORTS (bucket A — obsolete standalone); EXP_RESULTS_STATUS, EXPERIMENT_RESULTS_ANALYSIS, EXPERIMENT_TRACKING_GUIDE, ESM2_EMBEDDING_SCALING, COMPUTE_EMBEDDINGS_ANALYSIS (bucket B — redundant); STRATIFIED_EXPERIMENTS_PLAN (top-level "PLAN" that should never have been there).
  - Rename: `docs/CONFIGURATION_GUIDE.md` → `docs/conf_guide.md` (shorter, matches lowercase convention).
  - Plan moves: `docs/plans/design_dataset_gen_v2.md` → `docs/plans/done/2026-05-11_design_dataset_gen_v2.md` (with status header). Plan deletes: `lr_baseline_integration_plan.md` (4/4 steps done), `negative_pair_rng_fix_plan.md` (Source A done; Source B moot under seq_disjoint).
  - Plan prunes: `code_cleanup_plan.md` dropped its 70-line item 6 (Gen3 bundle rename — every flu_schema_raw_* target was retired in 9d3b96d); `design_improve_plan.md` collapsed sections 2–4 to one-line future-cleanup notes (only section 1 implemented).
  - Retired the abandoned experiment-tracking system: deleted `experiments/registry.yaml` (3,411 lines, last updated Jan 28) and the orphaned `src/utils/experiment_registry.py` module.
  - `documentation/` directory refresh: deleted `pipeline-overview.md` (subsumed), `configuration.md` (was thin pointer), `model-improvements.md` (historical), `stages/preprocessing.md` (subsumed), `analysis/presentation-plots.md` (script was deleted earlier). Refreshed `README.md` (thin index), `installation.md`, `quick-start.md`, `troubleshooting.md`, `analysis/results-analysis.md`, `development/code-structure.md` for current production state.
  - `old_scripts/` deleted entirely on 2026-05-13 (23 files, including 14 already-superseded files referenced in its own README).
  - `eda/` cleanup on 2026-05-13: deleted `probe_metadata_lookup.py` (the `metadata.first()` lookup it probed is no longer used) + 4 v2/regime verification scripts (`e2e_regime`, `edge_cases_regime`, `integration_split_v2`, `sanity_regime_counts`); refreshed docstrings on the 4 survivors (2 bunya + 2 dna_coverage feasibility) with explicit status headers flagging stale embedded inputs.
  - `.claude/settings.json` tightened (2026-05-13): added `Bash(rm *)` + `Bash(curl *)` to deny; dropped redundant `Bash(mkdir -p *)` from allow.
- **2026-05-12 batch** (feature branch `feature/normalized-pair-interactions`):
  - `dataset.split_strategy.hash_key: seq | dna` (default **seq**, protein-level). `seq_disjoint` routing now partitions on the **protein** hash by default — the stricter choice. `hash_key=dna` remains available as the legacy nucleotide-level partition (looser; still appropriate when the downstream feature is DNA-derived k-mer). Audit JSON always reports both `seq_hash_overlap` and `dna_hash_overlap` (one is the guarantee, the other diagnostic). Validator rejects values other than `{seq, dna}`. Implementation: `bipartite_components` and `seq_disjoint_route_pos_df` accept `hash_key` param; saver hard-fails only on the active family.
  - `slot_transform: unit_norm` (parameter-free L2 row-norm: `x -> x / max(||x||, 1e-8)`) and `interaction: unit_prod` (`(a*b) / max(||a*b||, 1e-8)`). Both available on the MLP path (`train_pair_classifier.py`) and on the ESM-2 + k-mer baseline paths (`_pair_features.py`, `kmer_utils.py`). `interaction: unit_diff` simultaneously switched from signed `(a-b)/||a-b||` to abs-then-normalize `|a-b|/||a-b||` (symmetric in a, b — matches the existing `diff = |a-b|` semantics).
  - K-mer baseline path no longer rejects non-`concat` interactions or non-`none` slot transforms. Whitelist is now `{none, unit_norm}` for slot_transform, full token set for interaction (concat / diff / unit_diff / prod / unit_prod and combinations). The legacy `interaction='concat'`-only guard from 2026-05-10 is gone.
  - K-mer feature computation generalized to arbitrary alphabets. `sequences_to_sparse_kmer_matrix(..., alphabet='ACGT')` — pass `'ACDEFGHIKLMNPQRSTVWY'` for protein k-mers. Vocab is `|alphabet|^k` lexicographic. Docstring spells out the rationale for (a) fixed exhaustive vocab (cross-run column alignment without handshake), (b) float32 cell dtype (int8 overflow + normalization-ready + downstream-lib compat).
  - Centralized sklearn-baseline defaults: `conf/baselines/default.yaml` (logistic / lgbm / knn1_margin / knn_vote — `feature_scaling`, model hyperparameters, `n_jobs`) wired via `# @package bundles` and an additional `- /baselines: default` entry in `flu_base.yaml`'s defaults list. Single source of truth replaces the previous in-code-only `cfg.get(key, default)` calls. LGBM `n_jobs` default lowered from -1 to 16 to play nicer when multiple LGBM runs share a node.
  - Stage 3 viz: new `plots/split_composition.png` — stacked bar per split (positive + per-regime negs) with achieved/requested neg:pos ratios. Yellow caption flags `metadata_holdout` active and notes that `train_ratio/val_ratio/test_ratio` are ignored under holdout. Per-split distribution titles (host / subtype / year / heatmap) now include `(n=X, Y% of dataset)` from new `isolate_share` field in `dataset_stats.json::split_sizes`. End-of-Stage-3 stdout banner prints a one-line split summary with isolate counts, percentages, and per-slot driver text (random / carved from train / filter spec).
  - Stage 3 viz cleanup: k-mer PCA scree bumped to top-12 PCs; `kmer_pca_cumvar.png` dropped (redundant with scree's cumulative line); per-split `kmer_pca_segments_{train,val,test}.png` dropped (intent unclear — colored by protein but didn't expose side). Removes `_plot_pca_cumvar`, `_plot_kmer_unique_segments`, and orphaned helper `_plot_features_by_category_2d`.
  - Aggregator output convention: write under `results/{virus}/{data_version}/runs/` (mirrors `data/datasets/` and `models/`). Stop writing aggregator artifacts under `docs/results/`. Script docstrings + example commands updated accordingly. The `docs/` tree is reserved for hand-authored writeups and plans.
- **`pair_builder_version: v2` is now the DEFAULT** (2026-05-11; was v1). The CLI in `dataset_segment_pairs.py` dispatches to v2 unless a bundle explicitly sets `pair_builder_version: v1`. None of the active bundles use v1-only knobs (`pair_mode: unordered`, `allow_same_func_negatives: true`, `canonicalize_pair_orientation: true`), so the flip is a no-op for active runs and lets the 28-pair sweep bundles (which never set the key) silently get v2.
- **Metadata holdout (cross-population split)** (2026-05-11): `dataset.metadata_holdout: {train: {filter}, test: {filter}, val: null | {filter}}` builds train/val/test isolate sets from per-slot metadata filters and threads them through `split_dataset_v2`'s `*_isolates_override` hook. Each filter accepts `host` / `hn_subtype` / `year` / `geo_location` / `passage` as scalar OR list (set membership, any length), plus `year_range: [min, max]` for inclusive year ranges (`year_range` mutually exclusive with `year`). Val carves `val_ratio` (0.1 default) off train when `val: null`. Validator + helper enforce: unknown axis keys -> error; `host_range`-style misuse on unordered axes -> error; empty train/val/test pool -> error; multi-slot overlap -> error with first 20 offending assembly_ids. Drops are recorded in `metadata_holdout_dropped.csv` next to `dataset_stats.json` (one row per isolate that matched no slot, with metadata cols + `excluded_reason` text). Mutually exclusive with CV (`n_folds > 1`) and with non-random `split_strategy.mode`. Replaces the retired `year_train`/`year_test` mechanism (year-axis holdout is the degenerate case). Plan: `docs/plans/done/2026-05-11_metadata_holdout_plan.md`. Helpers in `src/datasets/_pair_helpers.py` (`filter_by_metadata` extended; new `compute_metadata_holdout_isolates`). Tests: `tests/test_metadata_holdout.py` (27 tests covering happy paths + ~12 failure modes).
- **Ambiguous `hn_subtype` drop in Stage 3** (2026-05-11): isolates whose `hn_subtype` doesn't match `^H\\d+N\\d+$` are dropped right after metadata enrichment, with a WARNING reporting the count + per-value breakdown. On the full Flu A enriched set this is ~1,006 of ~108,530 isolates (~0.93%): mostly `HN` (1,004) plus singletons `H1N` / `H3N`. Toggle via `dataset.drop_ambiguous_subtype: true` (default). Implemented by `_pair_helpers.drop_ambiguous_hn_subtype()` (returns `(filtered_df, summary)` with `value_counts` per dropped value). 4 unit tests in `tests/test_metadata_holdout.py`.
- **Regime taxonomy 9 -> 8** (2026-05-11): `unknown_metadata_neg` retired. `classify_pair_regime` now treats null on either side of an axis as no-match on that axis (instead of catch-all-routing to the unknown bucket). `compute_match_count` returns `int` (was `Optional[int]`) -- null contributes 0. Validator rejects `unknown_metadata_neg` in `regime_targets` with a migration message. Aggregator and analyze_stage4_train use the 8-regime order; the `unknown_metadata_neg` column / footnote / agg-bucket are gone. Tests updated: `tests/test_negative_regime_sampling.py` rewrites the null-case to the 8-regime mapping; `tests/test_level1_neg_regimes.py` drops the synthetic unknown-axis row.
- **Aggregator extended** (2026-05-11): `src/analysis/aggregate_baselines_vs_mlp.py` now renders a per-row "Aggregate" sidebar (AUC-ROC, AUC-PR, F1, MCC, read from each run's `post_hoc/metrics.csv`) to the right of the per-regime heatmap, plus a companion `baselines_vs_mlp_overall.png` horizontal grouped-bars view. Adaptive colormap: vmin auto-promotes to 0.7 when every observed value is >= 0.7, so the typical "everything-above-0.9 wall of green" gets visible contrast; falls back to 0.0 when any value is below the threshold so genuinely broken models still saturate red. xtick labels rotated 35deg (was 0/25deg) so `host_subtype_year` and `unknown_metadata_neg` no longer collide. CSV gains 4 aggregate-metric columns appended after the 9 regime columns.
- **Stage 1 reference doc** (2026-05-11): `docs/methods/preprocess.md` -- GTO field map (verified against a real Flu A GTO), `protein_final.csv` / `genome_final.csv` construction, filter pipeline, 9-row tech-debt table (flagged: HA1/HA2 aux mapping is dead code; ESM-2 prep thresholds hard-coded in main; `gto['id']` silently ignored in favor of filename-derived assembly_id).
- Unified Flu preprocessing (`src/preprocess/preprocess_flu.py`) -- emits both `protein_final.csv` and `genome_final.csv` in a single pass. Wired up via `scripts/stage1_preprocess_flu.sh`. Design rationale (why two output files, QC step coverage) is folded into `docs/methods/preprocess.md`; the standalone `docs/genome_pipeline_design.md` historical doc was retired 2026-05-15.
- Metadata-aware negative sampling (v2 builder, 2026-05-10). Opt-in via `dataset.negative_sampling.regime_targets`. 9 mutually-exclusive regimes over `(host, hn_subtype, year_or_year_bin)`. Coverage > quotas (hard invariant); manifest reports per-regime target/available/coverage_placed/fill_placed/achieved/shortfall. Pair CSVs gain `neg_regime` and `metadata_match_count`. Output `negative_regime_manifest.{json,csv}`. Pre-requisite bug fixes also landed: `compute_axis_flags` now keys on `assembly_id` (was `seq_hash`, wrong by 5-34% on positives), and `compute_metadata_coverage` per-seq null counter dropna-first. Plan: `docs/plans/done/2026-05-09_metadata_aware_negatives_plan.md`.
- Level 1 8-regime stratified eval (post-hoc, 2026-05-10; trimmed 9->8 on 2026-05-11). Replaces the prior 4-bucket `level1_pair_regime.{csv,png}` with two views over the v2 sampler's 8-regime taxonomy: `level1_neg_regimes.{csv,png}` (per-regime TPR/TNR, 9 buckets = positive + 8 negs, seagreen/crimson) and `level1_neg_regimes_agg.{csv,png}` (aggregated by metadata-match count 0/1/2/3, seagreen/indigo). Both prefer the v2-written `neg_regime`/`metadata_match_count` columns when present; fall back to deriving from per-side host/hn_subtype/year for legacy datasets. Sanity test: `tests/test_level1_neg_regimes.py`.
- Multi-baseline runner (2026-05-10): `src/models/train_pair_baselines.py` accepts `--baseline NAME [NAME ...]` (or reads `baselines.enabled` from the bundle), shares feature materialization across baselines (kmer densification ~5-10 min cold; per-baseline scaling cheap), per-baseline isolation on failure, batch summary at the end. Four registered baselines: `logistic`, `lgbm`, `knn1_margin` (cosine 1-NN margin, the leakage diagnostic), `knn_vote` (sklearn KNeighborsClassifier with configurable k + weights, default k=5/distance). Bundle hook: `baselines: {enabled: [...]}` + `baseline_<name>: {...}`. The kmer-only `slot_transform=slot_norm` rejection in `_pair_features.py` still stands; `train_pair_baselines.py` silently coerces to 'none' for kmer baselines (with a printed note) so MLP-tuned bundles don't break the baseline run.
- Cross-model heatmap aggregator (2026-05-10, autodiscovery added 2026-05-11): `src/analysis/aggregate_baselines_vs_mlp.py`. Reads each entry's `post_hoc/level1_neg_regimes.csv`, builds a (n_models x 10 regimes) matrix, renders a single PNG heatmap (RdYlGn colormap, [0,1]) + CSV. Default caption flags the featurization mismatch. Two modes: (a) `--bundle <name> [--runs_root <dir>]` autodiscovers the latest `training_<bundle>_<TS>` and `baseline_<name>_<bundle>_<TS>` runs, cross-checks `training_info.json::dataset_dir` is identical across picks, refuses to aggregate on mismatch; bundle match is anchored so `--bundle flu_ha_na` does NOT pick up `flu_ha_na_seq_disjoint`. (b) `--model_dirs <paths...>` explicit list for ad-hoc cross-bundle comparisons. Canonical row order: MLP, logistic, lgbm, knn1_margin, knn_vote.
- seq_disjoint split mode (2026-05-11, extended 2026-05-12 with `hash_key`): `dataset.split_strategy.mode={random,seq_disjoint}` and `dataset.split_strategy.hash_key={seq,dna}` in v2 builder. **Default is `hash_key=seq` (protein-level)** — the stricter guarantee. Bipartite-CC LPT-greedy on the chosen hash family; whole components are indivisible so cross-split sequence leakage is impossible by construction on that family. Audit always reports overlap counters for BOTH seq_hash and dna_hash (one is the guarantee, the other is diagnostic). Zero pair drops on Flu A: HA/NA seq routing yields 21,719 components / largest 11,748 (20% of total), PB2/PB1 yields 14,924 components / largest 20,214 (38%). Both hit 80/10/10 within 0.0011%. The v2 negative sampler operates entirely on per-split `pos_df`, so per-split negatives inherit the property automatically. Saver writes `seq_disjoint_audit.json` (re-checked on full pairs incl. negatives) and hard-fails on non-zero overlap of the active family. Single-split only — CV combo is rejected loudly. Plan: `docs/plans/done/2026-05-10_seq_disjoint_routing_plan.md`. Earlier results: `docs/results/2026-05-11_exp4a_seq_disjoint_results.md` (those numbers were under `hash_key=dna`, the prior default; current default re-runs differ slightly on per-regime TPR/TNR).
- Two regime-aware bundles (2026-05-10; pb2_pb1 flattened 2026-05-11): `conf/bundles/flu_ha_na_neg_regimes.yaml` and `conf/bundles/flu_pb2_pb1_neg_regimes.yaml`. Both inherit from their parent (flu_ha_na / flu_pb2_pb1), set `neg_to_pos_ratio: 2.0` (gives the regime-aware fill phase room to bias the mix; the coverage phase eats ~|pos_pairs| at ratio=1.0), and use the uniform 8-regime mix (`host_subtype_year: 0.30` on the hardest regime; the other 7 regimes are uniform at 0.10). Both enable the 4 baselines (logistic, lgbm, knn1_margin, knn_vote).
- post_hoc/metrics.png (2026-05-10): single-figure bar chart with [F1 (binary), F1 (macro), AUC-ROC, AUC-PR, MCC] -- drops Accuracy (uninformative on imbalanced binary). Lives next to `metrics.csv` in `post_hoc/`. Replaces the legacy `results/.../training_<bundle>_<TS>/training_analysis/{metrics_summary,results_misc,model_calibration}.png` triple. `src/analysis/create_presentation_plots.py` deleted entirely (all four functions were either degenerate under schema_ordered or redundant with `post_hoc/`); `scripts/stage4_train.sh` no longer invokes it. Existing legacy `training_analysis/` dirs left in place; only new runs stop creating them.
- Bundle base rename pattern (2026-05-10): `conf/bundles/flu.yaml` → `flu_base.yaml` (trimmed to defaults + `run_suffix` + `master_seed`/`process_seeds`; HA/NA-specific overrides removed -- they were always overridden by `flu_28_major_protein_pairs_master.yaml` anyway). `conf/bundles/bunya.yaml` → `bunya_base.yaml` (pure rename; bundle still flagged as not-maintained). Naming convention: legacy/Gen-1 base configs use the `_base.yaml` suffix; `flu` and `virus` were ambiguously named across config groups before this.
- Methods docs added (2026-05-10):
  - `docs/methods/feature_normalization.md` -- (model x feature_source) preprocessing matrix; per-model rationale (LR needs StandardScaler for per-feature regularization; trees scale-invariant; cosine k-NN normalizes vector lengths internally; MLP uses learned LayerNorm); per-feature properties (ESM-2 subspace offset; k-mer length confound); common pitfalls (StandardScaler ≠ L2-norm; cosine implicitly L2-normalizes; slot_transform vs feature_scaling are different concepts).
  - `docs/methods/pipeline_overview.md` -- multi-audience synthesis (biologists, bioinformaticians, data scientists). Task framing → reassortment biology → 4 stages → filtering order → pair construction (positive dedup, negative coverage, the 8 regimes) → train/val/test balance impact → features → models → heatmap interpretation → 5-mode leakage map. Refreshed 2026-05-12 to align with 8-regime taxonomy + current production state.
  - `docs/post_hoc_analysis_design.md` -- Level 1 (8-regime + agg) + outputs schema. Refreshed 2026-05-13 to drop the legacy `unknown_metadata_neg` references.

## HPC
- For 8-GPU dev cluster (no scheduler): Python subprocess launcher with CUDA_VISIBLE_DEVICES per fold.
- Polaris (ALCF), PBS job arrays. Do NOT use Hydra's submitit launcher (SLURM only).
- See `polaris_plan.md` for Task 11 plan: phases 0-3, env setup, bundle design, queue strategy, scripts.
- See `speed_up.md` for training optimizations (67% speedup: batch_size=128, eval_train_metrics=false, pin_memory=true).
- See `docs/hardware_notes.md` for HW/SW interaction notes: `pin_memory` + cudaHostAlloc serialization under ensemble packing, `num_workers=0` rationale, TF GPU pre-allocation prevention, L3 cache vs working-set, and `[Extra]` background topics (CUDA async, AMP, CUDA_VISIBLE_DEVICES remapping, pinned-memory budget, ensemble packing vs PBS arrays).
- **Batch vs interactive env**: PBS batch mode doesn't source dotfiles. The fix is `#!/bin/bash -l` (login shell shebang) which loads the full default env (PrgEnv-nvidia, Cray PALS/mpiexec, libfabric, CUDA). No manual PATH or LD_LIBRARY_PATH needed — just conda + venv on top. Interactive mode uses `scripts/polaris_env.sh`.

## Cross-validation (IMPLEMENTED — branch: feature/cross-validation)
### Output structure
- Stage 3 runs ONCE → `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/`
  - Nested: `fold_0/`, `fold_1/`, …, `fold_{N-1}/` each with train/val/test CSVs, stats, plots
  - Top-level `cv_info.json` with fold isolate assignments and seeds
- Stage 4 trains per fold → `models/flu/{version}/runs/training_{bundle}_fold{k}_{ts}/`
  - Each training dir has `test_predicted.csv`, `optimal_threshold.txt`
  - `train_pair_classifier.py` auto-invokes `analyze_stage4_train.py` at end of training, writing artifacts to `<training_run>/post_hoc/`. Guardrailed — post-hoc failure logs `WARNING:` but never changes training exit code. Opt-out with `--skip_post_hoc`.
  - Backfill / refresh post_hoc over an entire sweep: `bash scripts/run_allpairs_post_hoc.sh <TAG>` (used for legacy runs predating in-train integration, or after analysis code evolves).
- After all folds: `cv_run_manifest.json` (dataset run dir), `cv_summary.csv/json`
- Cross-pair aggregation (`aggregate_allpairs_results.py`) emits `allpairs_summary.csv` + `allpairs_summary_fp_fn.csv`; also flags pairs with incomplete `post_hoc/` coverage (columns `post_hoc_n_with/n_total/missing_folds`).

### Config
- `conf/dataset/default.yaml`: `n_folds: null`, `fold_id: null` (null = single-split, backward compat)
- `conf/bundles/flu_schema_raw_slot_norm_unit_diff_cv5.yaml`: inherits base bundle, adds `n_folds: 5`

### Key implementation details
- `split_dataset()` gains `train/val/test_isolates_override` params (None = existing behavior)
- `generate_all_cv_folds()`: KFold on isolates; `val_frac = val_ratio / (1 - 1/n_folds)` for consistent val size
- Fold seed = `master_seed + fold_i` for reproducible but distinct negative sampling
- `--fold_id` added to training script (optional; appends `fold_{k}/` to `--dataset_dir`)
- `save_split_output()`: extracted helper used by both single-split and CV paths

### Launchers
- `scripts/run_cv_lambda.py`: subprocess.Popen per fold, CUDA_VISIBLE_DEVICES=gpu_k, saves manifest, calls aggregation
- `scripts/run_cv_polaris.pbs`: STAGE=dataset|train|aggregate; train uses PBS_ARRAY_INDEX=fold_id
- `scripts/aggregate_cv_results.py`: reads manifest or --training_dirs, computes mean±std, writes cv_summary.*

### Hydra limitation: no subdirectory bundles
CV bundle is flat (`conf/bundles/flu_schema_raw_slot_norm_unit_diff_cv5.yaml`), not in `paper/`.
Hydra's package resolution double-nests inherited configs from subdirs, breaking `get_virus_config_hydra`.
`conf/bundles/paper/` kept as directory but YAML bundles must stay flat. See README.md in bundles/.

### Next steps for CV
- Run dry run: `python scripts/run_cv_lambda.py --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 --dry_run`
- Run full CV: `python scripts/run_cv_lambda.py --config_bundle flu_schema_raw_slot_norm_unit_diff_cv5 --gpus 0 1 2 3 4`

## Stage 3/4 Decoupling (IMPLEMENTED — branch: feature/decouple-dataset-training)
- `stage4_train.sh` requires `--dataset_dir` explicitly; no bundle extraction from path
- `--allow_bundle_mismatch` flag removed (no longer needed)
- Training script saves `training_info.json` with full provenance (config_bundle, dataset_dir, HPs)
- Both shell scripts slimmed to ~60-100 lines matching the lean stage1/stage2b pattern
- Workflow: Stage 3 once → Stage 4 N times with different training bundles

## Task 11: All Protein-Pair Combinations (8x8 Heatmap) — ON HOLD since 2026-04-06
Goal: 28 pairwise combinations C(8,2) of 8 major proteins (PB2, PB1, PA, HA, NP, NA, M1, NS1).
Full run: 28 pairs × 12 CV folds × 100 epochs × ~111K isolates (Polaris).

**Current status (2026-05-10)**: paused since the Phase 3 failure analysis (2026-04-06).
Priority shifted to metadata-aware negative sampling (mode #5) and the
heatmap aggregator (`aggregate_baselines_vs_mlp.py`). When Task 11
resumes, the regime-aware bundles can drop into the 28-pair grid by
swapping the master bundle's `dataset.negative_sampling` block.

### What's built
- **Master bundle**: `conf/bundles/flu_28_major_protein_pairs_master.yaml` — Phase 3 production settings (full dataset, 12-fold CV, batch_size=128, optimized training). Children override only `schema_pair`.
- **28 child bundles**: `conf/bundles/flu_28p_{protA}_{protB}.yaml` — generated by `scripts/generate_all_pairs_bundles.py`
- **Phase 0 launcher**: `scripts/run_allpairs_polaris_phase0.sh` — sequential single-GPU for validation
- **Prod launcher**: `scripts/run_allpairs_polaris_prod.sh` — multi-node PBS job (28 nodes, prod queue). Each node runs `run_cv_lambda.py` with 4 GPUs for 12-fold CV.
- **Full plan**: `polaris_plan.md` — phases 0-3, env setup, bundle design, HPC reference

### What needs to be built
1. **Cross-pair aggregation + heatmap** — collect 28 CV summaries into 8×8 AUC/F1 matrix for paper
2. **Resolved config snapshot** — added `save_config()` calls to Stage 3 and Stage 4 (saves `resolved_config.yaml` in output dir). Not yet tested in a run.

### Key constraints
- Master bundle is set to Phase 3 defaults (patience=100, effectively no early stopping). For Phases 0-2, temporarily edit the master.
- Data is on Polaris (copied from Lambda 2026-03-27): `protein_final.csv`, `kmer_k6_*`, ESM-2 embeddings.
- N=12 folds chosen for perfect GPU packing (12/4 = 3 waves, no idle GPU).

### Phase status (2026-04-02)
- **Phase 0** — COMPLETE (single pair, single GPU, sequential)
- **Phase 1** — COMPLETE (single pair, 12-fold CV, 4 GPUs)
- **Phase 2** — COMPLETE (Steps 1-6). Key results:
  - Steps 2-4: interactive `qsub -I` with SSH (Step 2) and mpiexec (Step 4)
  - Step 5: batch `qsub` with 2 pairs — 2/2 SUCCEEDED
  - Step 6: full 28-pair batch — 23/28 SUCCEEDED, 5 failed (CUDA OOM on specific nodes, transient)
  - Root cause of early env failures: `#!/bin/bash -l` needed for login shell in batch mode
- **Phase 3** — FAILED (walltime kill after 6h, 0/336 folds completed). Root cause diagnosed and fixed. Ready to re-test.
- **Branch**: `feature/polaris-mpiexec` (master as of latest commits)

### Phase 3 failure root cause + fixes (2026-04-02 → 2026-04-06)
**Problem**: Training ~15-34x slower per batch on Polaris vs Lambda (7.5 min/epoch vs 29s).
Three root causes identified:
1. **Memory explosion**: `KmerPairDataset` densified the ENTIRE 868K×4096 sparse matrix (14.2 GB) per fold. 4 concurrent folds = 56.9 GB → folds 1-11 OOM. Only fold 0 survived per pair.
2. **Cache thrashing**: 14.2 GB array >> L3 cache (~32-64 MB). Shuffled batch access = pure cache misses.
3. **Per-item overhead**: `self.pairs.iloc[idx]['label']` (pandas iloc) + `torch.tensor()` copy per item.

**Round 1 fixes** (in `src/models/train_pair_classifier.py`):
- Matrix subsetting: only densify rows used by this fold's pairs (~3.5 GB vs 14.2 GB). 4 folds fit: 14 GB vs 56.9 GB.
- Label pre-extraction: `self.labels = pairs['label'].values` — eliminates pandas iloc.
- Zero-copy tensors: `torch.from_numpy()` instead of `torch.tensor()` (safe with num_workers=0).
- tqdm `miniters=50` to reduce Lustre I/O from 1385 writes/epoch to ~28.
- `NUM_WORKERS` hard-coded to 0 (was configurable but num_workers>0 is 87% slower + incompatible with torch.from_numpy).
- Level 1 profiling added: per-epoch data_time/compute_time/eval_time in training_history.csv.

**Test run results (2026-04-06)**: Folds 0,1,3 ran 6 epochs before job died. Fold 2 OOM'd at model.to(device).
Profiling showed data loading is STILL 96-97% of epoch time (~460s data, ~4s compute, ~14s eval).
Matrix subsetting fixed memory but NOT speed. Two remaining issues:

**Round 2 diagnosis (2026-04-06)**:
1. **Fold 2 OOM**: TensorFlow installed in Polaris system conda env, loaded transitively by HuggingFace `transformers`
   (via esm2_utils.py). TF's default is to eagerly allocate all GPU memory. Added defensive env vars
   (`TF_CPP_MIN_LOG_LEVEL=3`, `TF_FORCE_GPU_ALLOW_GROWTH=true`) at top of script before any imports.
   Also added `torch.cuda.mem_get_info()` diagnostic before model.to(device) to capture GPU state.
2. **Data loading 460s/epoch**: Strongest hypothesis is `pin_memory=True` with 4 concurrent folds.
   pin_memory forces cudaHostAlloc (CUDA driver call that serializes across processes) for every batch.
   1385 batches/epoch × 4 folds = 5540 driver calls/epoch contending on same driver lock.
   The 18% speedup from pin_memory was benchmarked on Lambda (single fold) — likely counterproductive
   with 4 concurrent folds on Polaris. Added Level 2 diagnostic (micro-benchmark 10 batches with
   pin_memory=True vs False) to confirm before changing the config.

**Additional changes**:
- `run_cv_lambda.py`: `--skip_dataset` now auto-discovers latest dataset dir (no longer requires `--dataset_run_dir`).
- `src/analysis/analyze_training_profile.py`: post-hoc profiling aggregation (single run, CV, all-pairs modes).
- Cross-pair aggregation wired into prod launcher (`aggregate_allpairs_results.py`).

See `polaris_plan.md` for detailed step-by-step checklists per phase.

## Deferred (was "What's Next (immediate)" pre-2026-05-07)
**H3N2 all-pairs sweep** -- rerun the 28 protein-pair x 12-fold CV with the
`dataset.hn_subtype=H3N2` filter applied via the `--filter`/`--tag`
mechanism (no new bundles). Plumbing committed on feature branch, runbook
at `docs/allpairs_filter_sweep_runbook.md`. **On hold** since the
regime-aware sampler (2026-05-10) addresses the metadata-shortcut concern
more comprehensively than a single-subtype filter; the H3N2-only view is
still a valid within-subtype experiment, just not the next priority.
Plumbing summary: `--override` in `dataset_segment_pairs.py` /
`train_pair_classifier.py` / `run_cv_lambda.py`; `--filter` + `--filter-tag`
in `run_allpairs_polaris_prod.sh`; `--tag` in `aggregate_allpairs_results.py`.

## What's Next (beyond Task 11)
- **Mode #3 (sequence-level leakage) — IMPLEMENTED 2026-05-11; default tightened 2026-05-12** via seq_disjoint routing. Default `hash_key=seq` (protein-level — stricter than the original dna-level partition). Cross-split overlap on the chosen hash family is 0 by construction with zero pair drops on both HA/NA (largest component 20%) and PB2/PB1 (largest component 38%). Initial finding (2026-05-11, under `hash_key=dna`): MLP host_subtype_year TNR drops 0.872 → 0.834 vs random split. Initial finding under `hash_key=seq` (2026-05-12): on PB2/PB1, 1-NN edges MLP on aggregate MCC (0.900 vs 0.887) — consistent with conserved proteins offering fewer truly novel eval examples. Baselines barely move on per-regime TPR/TNR. strict_dedup deliberately deferred (seq_disjoint achieves the same test with no data loss). Cluster-level leakage (mode #4) remains open — see `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md`.
- **Cross-population holdout (metadata_holdout) — IMPLEMENTED 2026-05-11.** See "Recently Promoted to Production" above. The natural next experiment is to actually train MLP + baselines on `flu_ha_na_metadata_holdout_temporal` (year-axis 2021-2023 train vs 2024 test) and on multi-axis bundles (e.g., human-H3N2-2021-2023 vs pig-H1N1-2024); aggregate via the heatmap aggregator with `--bundle flu_ha_na_metadata_holdout_*`. The aggregator's adaptive vmin will make small AUC drops visible.
- Re-test mode #5 on regime-aware datasets via the heatmap (`aggregate_baselines_vs_mlp.py`). Regime-aware sampling is the construction-time mitigation; the heatmap on `host_subtype_year` TNR is the verification.
- Run a properly-trained MLP on `flu_*_neg_regimes` (the 2-epoch smoke MLP we used for the aggregator demo isn't a fair comparison vs the 4 baselines; need full ~100 epochs).
- Fix pair_key dedup for temporal holdout -- re-run for clean metrics
- Run cross-validation end-to-end on a regime-aware bundle (see CV section above)
- FP/FN diagnostics (Task 12) -- mostly subsumed by the per-regime heatmap now; revisit if any regime shows surprising behavior
- Quantify unlinked BV-BRC records (ask Jim) -- scopes the remediation demo

## Env Management
**Rule**: bioconda / kalininalab CLI tools and experimental Python packages
live in **dedicated conda envs**, never in the main `segmatch` pipeline env.

**Why**: bioconda installs frequently pull a different `libhdf5` build than
conda-forge, which breaks the precompiled `h5py` wheel in the same env. Two
incidents:
- 2026-05-15: `conda install -c bioconda mmseqs2` into `cepi` broke `h5py`
  -> forced the lazy-h5py refactor in `src/utils/{esm2,embedding}_utils.py`
  and `src/models/{train_pair_classifier,_pair_features}.py` (so k-mer-only
  paths no longer crash on h5py import).
- 2026-05-19: `conda install -c conda-forge -c bioconda -c kalininalab datasail`
  into `cepi` broke `h5py` again (`undefined symbol: H5Pset_fapl_ros3`),
  also pulling ~55 transitive solver deps (cvxpy, scip, suitesparse, ipopt,
  mumps, ...). `conda remove --force datasail` removed only the package, not
  the transitive deps -- env stayed polluted.

**How to apply**:
1. Pipeline env: `segmatch` (built from `environment.yml`, conda-forge only).
2. CLI-only tools: dedicated env per tool; expose binary path via Hydra
   config. mmseqs2 is the existing example -- `clustering_utils.py:172`
   takes `mmseqs_bin: str` already, so no code change needed beyond pointing
   at `/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs`.
3. Experimental Python tools (e.g., `datasail`): dedicated env; run as
   subprocess from `segmatch` if needed, or read its outputs (split CSVs)
   back into segmatch.
4. **Never `conda remove --force` to clean up bioconda damage** -- it leaves
   transitive deps orphaned. Rebuild the env from `environment.yml` instead.

**Envs as of 2026-05-19**:
- `cepi`: legacy pipeline env, polluted by DataSAIL leftovers + bio CLI tools.
  Retire after `segmatch` is validated by one Stage 3+4 run.
- `segmatch`: clean pipeline env, built from `environment.yml` 2026-05-19.
  **Validated end-to-end 2026-05-20** via Stage 3 + Stage 4 on
  `flu_ha_na` with `dataset.max_isolates_to_process=2000
  training.epochs=5`: Stage 3 in 2 min, Stage 4 in 2 min on GPU,
  model learned (Test F1=0.802, AUC-ROC=0.92). `cepi` env can be retired.
- `datasail`: dedicated env for the DataSAIL bake-off
  (`feature/datasail-bakeoff` branch). Built 2026-05-19.
- `mmseqs2`: dedicated env, built 2026-05-19. mmseqs2 version `18.8cc5c`.
  Binary path: `/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs`.

**`MMSEQS_BIN` env var convention** (wired 2026-05-19): on Lambda, set
`export MMSEQS_BIN=/homes/apartin/miniconda3/envs/mmseqs2/bin/mmseqs` in
your shell rc (or per-session before running `seq_redundancy_per_function.py`).
`run_mmseqs_easy_cluster` in `src/utils/clustering_utils.py:172` resolves the
binary in this order: explicit `mmseqs_bin=` kwarg -> `MMSEQS_BIN` env var
-> `'mmseqs'` on PATH. `seq_redundancy_per_function.py` exposes a
`--mmseqs_bin` CLI override. Hydra config was considered but skipped: the
only mmseqs caller is an argparse script, and on Polaris `module load`
puts mmseqs on PATH so the default works there.

## User Preferences
- Concise responses, no emojis unless asked
- No unnecessary refactoring beyond what's asked
- Always ask before destructive operations (rm, git reset --hard, git push --force, etc.)
- CLAUDE.md is the authoritative project context; .claude/memory.md is the compact working memory
- Both files are in the repo -- update them when decisions change
- **One script per purpose**: follow the existing pattern in `src/analysis/` — propose a dedicated script with a clear name (e.g., `aggregate_cv_results.py`) rather than hedging between existing scripts. Commit to the obvious answer.
- **Code priority order**: correctness > readability > efficiency. Optimize for the next reader, not the next clock cycle. Reach for performance changes only when measured (or when efficiency is correctness-critical, e.g., when the naive version is intractable at expected input sizes).
- **Communication style**: prefer common words; use jargon only when it carries meaning the plain term doesn't. Don't cut technical content; cut hedges and filler. Concrete numbers, file:line refs, and observed data beat hedged adjectives.
- **Accuracy over confidence**: state only what is verified against a source actually checked in this session (paper passage, code at file:line, observed command output). When uncertain, say so with what would resolve it ("haven't read §X", "need to grep Y"). Don't pattern-match across sources without verification — superficially synonymous terms (e.g. DataSAIL I2, Park & Marcotte C3, segmatch seq_disjoint) may differ in dimensionality, what gets discarded, or which axes they cover. Hedges are fine when the uncertainty is named; vague hedging is not.
- **No commits without explicit instruction**: never run `git commit` until the user says "commit" (or equivalent) for the specific change. Stage and prepare diffs freely, but wait for the green light.
- **Refer to Claude as "Claude"** in committed docs and writeups, not "I" or "my proposal." First-person is fine in conversation; documents persist and read better in third-person.
