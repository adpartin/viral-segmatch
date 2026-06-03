# CLAUDE.md — Project Context for Claude Code

This file provides persistent context for Claude Code sessions. Update it as the project evolves.
Also read `.claude/memory.md` for compact, up-to-date project memory (pipeline state, recent decisions).

---

## Session Startup Checklist

At the start of each session:
1. Read `.claude/memory.md` (in the repo) for current project state.
2. Check if the machine-local auto-memory `MEMORY.md` (path provided in the system prompt) exists.
   If it does not exist, create it with exactly this content:
   ```
   Project memory has moved into the repo for portability across machines.
   Read: .claude/memory.md (in the repo root)
   This machine-local file is no longer updated.
   ```
3. Check `docs/plans/` for any in-progress plans (status != IMPLEMENTED).
   If found, read the plan and offer to resume implementation.

### Plans directory (`docs/plans/`)

Plans are saved in the repo (not machine-local `~/.claude/`) for cross-machine portability.
When creating a plan during plan mode, save it to `docs/plans/<descriptive_name>_plan.md` before
starting implementation. Mark the plan's status right after the title heading:
- `**Status: IN PROGRESS**` — plan approved, implementation underway
- `**Status: IMPLEMENTED**` — implementation complete

When a plan is fully implemented, mark `**Status: IMPLEMENTED**` and move it to `docs/plans/done/`.
Active plans stay in `docs/plans/`; completed plans live in `docs/plans/done/`.

---

## Approval Required

Always ask for explicit confirmation before running any of the following — even if you think it is safe:

- `rm *` or any file/directory deletion
- `git rm *`
- `git reset --hard *` or `git reset --mixed *`
- `git push --force *` or `git push -f *`
- `git branch -D *`
- `git rebase *`
- `git clean *`
- Any command that modifies shared infrastructure, sends messages, or affects state outside this repo

---

## What This Project Does

**viral-segmatch**: Predicts whether two viral protein segments co-occur in the same isolate (binary classification). Given embeddings of two protein segments from the same virus, the model learns to distinguish true co-occurring pairs (positive) from artificially mixed pairs (negative).

**Approach**: Frozen ESM-2 protein embeddings (1280-dim, `esm2_t33_650M_UR50D`) → pairwise interaction feature (e.g., `unit_diff`, `concat`) → MLP binary classifier.

**Primary virus**: Influenza A (Flu A). Bunyavirales support exists but is not actively maintained.

---

## Pipeline Stages

Stages 1–2 run once per dataset (shared across experiments). Stages 3–4 are experiment-specific.

| Stage | Script | Output | Runs |
|-------|--------|--------|------|
| 1. Preprocess | `src/preprocess/preprocess_flu.py` | `data/processed/flu/{version}/protein_final.csv` + `genome_final.csv` | Once |
| 1.5. CDS extraction (opt) | `src/preprocess/extract_cds_dna.py` | `data/processed/flu/{version}/cds_final.parquet` | Once, for nt cluster_disjoint |
| 2. Embeddings | `src/embeddings/compute_esm2_embeddings.py` | `data/embeddings/flu/{version}/master_esm2_embeddings.h5` | Once |
| 3. Dataset | `src/datasets/dataset_segment_pairs.py` (CLI) → `dataset_segment_pairs_v2.py` (default builder since 2026-05-11) | `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/` | Per experiment |
| 4. Train | `src/models/train_pair_classifier.py` | `models/flu/{version}/runs/training_{bundle}_{ts}/` | Per experiment |

Shell wrappers: `scripts/stage1_preprocess_flu.sh`, `scripts/stage2_esm2.sh`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh` (MLP only), `scripts/stage4_baselines.sh` (one baseline per call), `scripts/stage4_full.sh` (MLP + every baseline in `bundle.baselines.enabled` — bug fixed 2026-05-25 to read `.enabled` correctly), `scripts/stage4_sweep.sh` (Stage 4 across multiple datasets × seeds in parallel — added 2026-05-25), `scripts/mmd_sweep.sh` (S1+S2 MMD across multiple datasets for one feature_space × label_filter — added 2026-05-25). Stage 1.5 has no shell wrapper — invoked directly when needed (`python src/preprocess/extract_cds_dna.py --config_bundle <virus_bundle>`).

**Stage 3/4 decoupling**: Stage 4's shell script requires `--dataset_dir` explicitly and
does not extract or validate a bundle name from the dataset path. This allows running
Stage 3 once and Stage 4 multiple times with different training bundles against the same
dataset. Provenance is tracked via `training_info.json` saved in the training output dir.

---

## Configuration System

**Hydra** with a bundle-per-experiment pattern.

- Bundles: `conf/bundles/{bundle_name}.yaml` — one file per named experiment
- Virus configs: `conf/virus/flu.yaml`, `conf/virus/bunya.yaml`
- Data paths: `conf/paths/flu.yaml`, `conf/paths/bunya.yaml`
- Defaults: `conf/dataset/default.yaml`, `conf/embeddings/default.yaml`, `conf/training/base.yaml`, `conf/baselines/default.yaml`
- Config loader: `src/utils/config_hydra.py`

**Convention**: One bundle = one reproducible experiment. Bundle names encode the experiment:
`flu_{proteins}_{n_isolates}[_{modifiers}]`, e.g., `flu_ha_na_5ks`, `flu_schema_raw_slot_norm_unit_diff_h3n2`.

Key bundle parameters: `virus.selected_functions`, `dataset.max_isolates_to_process`, `dataset.hn_subtype`, `dataset.year`, `dataset.host`, `dataset.split_strategy.{mode,hash_key,cluster_alphabet,single_slot}`, `dataset.metadata_holdout`, `training.slot_transform`, `training.interaction`. Sklearn-baseline knobs live under `baseline_<name>.*` at the bundle root (defaults inherited from `conf/baselines/default.yaml`).

`split_strategy.single_slot` (added 2026-05-24): `null` (default, bilateral cluster_disjoint) | `'a'` (constrain slot-a clusters only, slot-b unconstrained) | `'b'` (constrain slot-b only). Only consumed under `mode: cluster_disjoint`. Unlocks idXX thresholds the bilateral path collapses on; the unconstrained slot may still shift via biological coupling (HA↔NA via subtype on Flu A HA-NA). See `docs/results/2026-05-24_cluster_disjoint_feasibility_HA_NA.md` and `docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`.

**Bundle organization** (see `conf/bundles/README.md` for full detail):
- Each bundle has a `# STATUS: active|ablation|experimental|legacy|not maintained` header comment.
- Bundles form inheritance chains via Hydra `defaults`. Base bundles (e.g., `flu_schema.yaml`,
  `flu_schema_raw_slot_norm_unit_diff.yaml`) must stay flat; only leaf bundles can be moved to subdirs.
- `conf/bundles/paper/` — reserved for publication experiments (CV, temporal holdout, large dataset).
- Three generations: Gen1 (`flu.yaml` base, no schema ordering), Gen2 (`flu_schema.yaml` base,
  none+concat), Gen3 (`flu_schema_raw_*`, current best: `slot_norm + unit_diff`).


---

## Active Source Files

```
src/
  preprocess/
    preprocess_flu.py               # Stage 1 (ACTIVE): GTO → protein_final.csv + genome_final.csv
    extract_cds_dna.py              # Stage 1.5 (optional): protein_final + genome_final → cds_final.parquet (CDS DNA + cds_dna_hash). Prereq for nt cluster_disjoint.
    flu_genomes_eda.py              # Generates flu_genomes_metadata_parsed.csv (run once)
    preprocess_bunya_protein.py     # Bunya preprocessing (NOT actively maintained; reference-only)
  embeddings/
    compute_esm2_embeddings.py      # Stage 2: sequences → ESM-2 HDF5 cache
    compute_kmer_features.py        # Stage 2b: DNA (or protein, via alphabet param) → k-mer sparse matrix
  datasets/
    dataset_segment_pairs.py        # Stage 3: pairs + train/val/test splits
    dataset_segment_pairs_v2.py     # v2 builder (default; coverage-first + regime-aware)
    _pair_helpers.py                # Shared helpers (seq_disjoint routing, filters)
    _split_helpers.py               # cluster_disjoint routing (mmseqs2-based)
    _negative_regime_sampling.py    # 8-regime classifier + priority chain for racov
  models/
    train_pair_classifier.py  # Stage 4: MLP classifier training
  analysis/
    analyze_stage4_train.py         # Confusion matrix, ROC, FP/FN analysis
    visualize_dataset_stats.py      # PCA plots, interaction diagnostics
    aggregate_experiment_results.py # Cross-bundle summary tables
    visualize_cv_results.py         # CV visualization (error bars, ROC curves)
    analyze_stage1_preprocess.py    # Preprocessing QC
    analyze_stage2_embeddings.py    # Embedding quality checks
    analyze_stage3_datasets.py      # Dataset balance/distribution
    build_mmseqs_clusters.py             # mmseqs2 per-function cluster sweep (aa easy-cluster, nt easy-linclust); emits redundancy_stats.csv, runtime.json, and redundancy_summary.md alongside the data (not in docs/)
    cluster_disjoint_feasibility.py      # Bilateral bipartite-CC feasibility pre-flight (aa via --protein_final, nt via --cds_final)
    single_slot_cluster_disjoint_feasibility.py  # Single-slot atom feasibility pre-flight (reuses bilateral's helpers); sweeps both slots × thresholds
    cluster_analysis_summary.py          # Post-hoc structural summary: 8x2 redundancy table, mutations-tolerated per threshold, cluster-collapse + bipartite-feasibility plots
    plot_aa_vs_nt_cluster_disjoint.py    # LGBM + 1-NN cluster-disjoint test-metric comparison across routings
    aggregate_cluster_disjoint_ratios.py # Ratio-sweep aggregator
    mmd_per_slot.py                      # S1 per-slot MMD (RBF + permutation test) on PCA-50 ESM-2 / aa k-mer / nt k-mer
    mmd_per_pair.py                      # S2 per-pair MMD on the production Test 3 interaction (slot_transform=unit_norm + interaction=unit_diff+prod)
    aggregate_mmd_single_slot_sweep.py   # Sweep rollup + plots (MMD-vs-idXX, perf-vs-idXX, perf-vs-MMD). Parametrized for any pair/direction since 2026-05-26 (flags: --routing_direction, --training_bundle, --slot_a_display/--slot_b_display, --out_subdir, --skip_reference_baselines)
    cluster_pair_coupling_precheck.py    # Pre-dataset Cramér's V check for V(slot-a cluster × slot-b cluster) and × whole-isolate H_N_ subtype, parametrized for any schema_pair. Used to falsify the "single-slot decouples slots" prediction before launching a sweep.
  utils/
    config_hydra.py                 # Hydra config loader (primary)
    esm2_utils.py                   # ESM-2 tokenization, batch embedding
    embedding_utils.py              # Load/index embeddings from HDF5
    metadata_enrichment.py          # Load flu_genomes_metadata_parsed.csv
    seed_utils.py                   # Hierarchical seed system
    learning_verification_utils.py  # Karpathy-style sanity checks
    plot_config.py                  # Colors, protein name mapping
    gto_utils.py, protein_utils.py, path_utils.py, timer_utils.py
    kmer_utils.py                   # Load k-mer features, pair construction
    clustering_utils.py             # mmseqs2 wrappers (FASTA export, TSV parse); alphabet={aa,nt}, algorithm={cluster,linclust}
    cds_utils.py                    # CDS reconstruction (parse_location, extract_cds_dna, translate_dna with IUPAC, compute_cds_dna_hash)
    dna_utils.py                    # DNA QC utilities (in development)
    dim_reduction_utils.py          # PCA/UMAP wrappers
```

---

## Key Experimental Findings (as of Mar 2026)

> **All findings below were measured under the protein pair_key
> convention** (pair_key = canonical hash of `seq_hash_a` / `seq_hash_b`
> = protein-level dedup). Phase 2 (2026-06-02 → 2026-06-03) introduced
> a CDS-DNA pair_key option for nt_cds cluster_disjoint bundles
> (`split_strategy.pair_key_alphabet`); under that option silent
> codon variants of the same protein-pair count as distinct positives,
> inflating the pair universe +34.9 % (HA-NA) / +57.0 % (PB2-PB1) and
> opening a new "DNA-variant" leakage mode (silent variants split
> across train/test). New experiments after 2026-06-03 should cite
> the `pair_key_alphabet` they used; default for cluster_disjoint
> on nt_cds remains `nt_cds`. See `docs/plans/2026-06-02_pair_key_alphabet_plan.md`
> and `docs/results/2026-06-03_phase2_postmigration_metrics.md`.

- **`unit_diff` > `concat` on homogeneous data (ESM-2 only)**: ESM-2 `concat` fails on H3N2-only HA/NA (AUC=0.50); `unit_diff` succeeds (AUC=0.96). Direction of embedding difference carries genuine biological signal; magnitude is a shortcut.
- **ESM-2 `concat` collapse is not H3N2-specific**: Task 11 all-pairs sweeps (Apr 2026) show PB1/PA collapses on BOTH the full unfiltered dataset (AUC=0.4971) and the H3N2-filtered dataset (AUC=0.4981) with ESM-2 concat. K-mer concat never collapses on any pair in any sweep. The subspace-offset failure is a property of ESM-2's protein-type geometry, not of a specific subtype.
- **K-mer concat does NOT collapse on H3N2**: K-mer concat achieves AUC=0.985 on H3N2-only, proving the concat failure is specific to ESM-2's embedding geometry (subspace offset between protein types), not concatenation as an interaction.
- **K-mer dominates ESM-2 on homogeneous data**: On H3N2-only, k-mer unit_diff AUC=0.988 vs ESM-2 unit_diff AUC=0.957. K-mer features are interaction-agnostic (unit_diff≈concat) because sparse frequency vectors don't have ESM-2's subspace offset problem.
- **LayerNorm (`slot_norm`) is critical for homogeneous subsets**: Without it, raw HA/NA embeddings live in slightly different subspaces; `unit_diff` then picks up slot offset rather than biological signal.
- **Delayed learning on H3N2 + unit_diff**: Characteristic plateau-then-breakthrough (~epochs 10–32, seed-dependent). Increase `patience` to 40+ for H3N2-only runs.
- **High FP rate on filtered datasets**: Likely due to subtype/host confounders in negative pairs. Hypothesis: model learns "same population" rather than "same isolate." Hard negatives or strict metadata filtering can test this.
- **K-mer (k=6, 4096-dim) matches or exceeds ESM-2 on mixed-subtype HA/NA**: AUC 0.982 vs 0.966–0.975. Both interactions work with k-mers.
- **K-mer interaction sweep on HA/NA (Tests 1–4, 2026-05-12)**: With Test 1 = `[|u-v|, u*v]` (raw), Test 2 = same on L2-normalized slots, Test 3 = `[|u_n-v_n|/||·||, u_n*v_n]`, Test 4 = same with `unit_prod` on the second term — all four lie within ~0.5% on F1 and ~0.1% on AUC-ROC. Test 3 narrowly leads on most aggregate metrics. Differences are at seed-noise level on a single-seed run; multiple seeds needed before claiming any winner. Important detail: `unit_diff` is element-wise abs first then L2-normalize (matches the symmetric `diff = |u-v|` semantics); previously it was a *signed* normalization.
- **seq_disjoint scales to conserved proteins (2026-05-12)**: PB2/PB1 with `hash_key=seq` produces 14,924 components, largest = 20,214 pairs (~38% of total), and still achieves 80/10/10 within 0.0011%. The dominant components fit inside the train target so the bin-packer hits the ratios cleanly. HA/NA with the same routing: 21,719 components, largest 11,748 (~20%). Both runs drop zero pairs.
- **1-NN edges MLP on PB2/PB1 under seq_disjoint** (MCC 0.900 vs 0.887). Consistent with the conservation-effect interpretation: PB2/PB1 has fewer distinct proteins overall, so eval splits contain fewer truly novel proteins and lookup-style baselines have an easier time.
- **Experiment B-nt feasibility ceiling = aa ceiling on Flu A (2026-05-15, pair_key=protein)**: nt CDS-level cluster_disjoint hits the same bipartite mega-component collapse as aa cluster_disjoint, at the same thresholds (only id100 and id099 are operable on the full corpus; id095 and below dump >98% of pairs into one component on both alphabets). The hope that nt's higher synonymous diversity would unlock lower-threshold splits did not pan out — the corpus's metadata structure dominates the alphabet choice. See `docs/results/2026-05-15_cluster_disjoint_nt_results.md`. **Caveat under nt_cds pair_key**: the bipartite CC structure is unchanged (clustering atoms preserved), so the mega-CC FRACTION of pairs stays the same; only the absolute pair counts inflate uniformly. Feasibility ceiling is robust to the pair_key change.
- **1-NN cosine margin ≥ LGBM at every cluster_disjoint routing (2026-05-15, pair_key=protein)**: ran 1-NN + LGBM head-to-head on 8 cells (HA/NA × PB2/PB1 × {seq_disjoint, aa id099, nt id100, nt id099}). 1-NN matches LGBM at id100/seq_disjoint cells and OUTPERFORMS LGBM at id099 cells (+16 pp F1 on HA/NA aa id099, +7 pp on PB2/PB1 aa id099). Going-in hypothesis "1-NN drops more than LGBM under cluster_disjoint" did not survive. Read: cluster_disjoint weakens the near-neighbor signal *gradually* rather than eliminating it; 1-NN's prediction-by-nearest-pair stays well-calibrated under that weakening while LGBM's tree splits rely on signal that doesn't generalize across the cluster boundary. The "MLP vs 1-NN" leakage doctrine is informative as a residual-leakage gauge but does not by itself confirm that cluster_disjoint removed leakage. See `docs/results/2026-05-15_cluster_disjoint_nt_results.md` § "1-NN cosine margin (leakage upper bound)". **Caveat under nt_cds pair_key**: the 1-NN ≥ LGBM ordering is conditional on the protein-collapsed pair universe. Under nt_cds pair_key, silent-variant training neighbors offer additional "near-identical" matches at inference (same protein, slightly different DNA), so 1-NN's advantage would likely strengthen further. Not re-measured in commit 6 because the post-Phase-2 nt cluster routing shifted ~20 % of protein-pairs across splits (see `docs/results/2026-06-03_phase2_postmigration_metrics.md` Section C), confounding a direct 1-NN-vs-LGBM re-test.
- **Single-slot HA-only cluster_disjoint sweep on HA-NA produces monotone MMD ↑ and monotone test perf ↓ across id100..id095 (2026-05-24, multi-seed 2026-05-25, pair_key=protein)**: 6 datasets built under new `single_slot` routing mode, all feasible 80/10/10. S1 HA MMD grows monotonically with id↓ (ESM-2 22.6×, aa k=3 33.7×); S1 NA MMD also grows (~9-13×) with a non-monotone dip at id097; S2 pair MMD tracks HA closely. Models trained per dataset (MLP + LGBM + 1-NN cosine margin on aa k=3 + Test 3); MLP trained with 3 seeds (42, 43, 44), F1 std ≤ 0.005 — 4.4 pp F1 drop is ~80σ. F1-vs-MMD² scatter is **nearly linear** across all three models. id097 ≈ id098 F1 plateau aligns with id097 NA-MMD dip (perf doesn't drop where MMD doesn't grow), confirmed real under multi-seed. Negative-pair MMD also shifts monotonically; perf drop is **joint pos+neg shift**, not positive-driven. Pre-registered "S1 NA stays near random" was FALSIFIED — biological coupling confirmed: HA-cluster boundary ≈ NA-subtype boundary (Cramér's V drops 0.98 → 0.82 across id100→id095; never below 0.5 on any axis). Not tested: ESM-2 training cross-check, NA-only direction (slot-symmetry check), PB2-PB1 (different biology, no subtype coupling), CV-fold variance. See `docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`.
- **PB2-PB1 single-slot PB2-only sweep is the falsification sibling — H1-H5 all supported, ~half the shift and ~half the F1 drop of HA-NA (2026-05-26, pair_key=protein)**: same 6-threshold infra applied to PB2/PB1 (no antigenic subtype on either slot). Cramér's V pre-check confirmed PB2-PB1 cluster-cluster coupling is consistently 0.08-0.15 below HA-NA's at every threshold (V=0.76→0.41 vs HA-NA 0.85→0.53 across id100→id095). Sweep results: constrained-slot (PB2) pos-MMD grows 6.7× (vs HA's 33.7×); unconstrained-slot (PB1) pos-MMD grows 5.7× (vs NA's 12.9×); pair pos-MMD grows 6.5×. `label=both` (the joint distribution the MLP trains on) gives the cleanest comparable summary: PB2 6.7×, PB1 2.8×, pair 6.9×. Negative-pair MMD grows MORE than positive in most cells (PB2 9.5×, PB1 10.0×, pair 11.4×) — joint pos+neg shift framing reaffirmed. Held-out F1 drop id100→id095 is ~half of HA-NA: MLP 2.5pp (vs 4.6pp), LGBM 2.5pp (vs 5.9pp), 1-NN 2.2pp (vs 4.7pp). **Caveat: model ordering on PB2/PB1 is 1-NN > MLP ≈ LGBM at every threshold**, reversing HA-NA's MLP > 1-NN > LGBM. The trained models aren't outperforming nearest-neighbor lookup on this pair — strong "memorization-test" signal that residual metadata-driven leakage survives cluster_disjoint single-slot on PB2/PB1. Multi-axis metadata holdout (year + host + subtype) on top of cluster_disjoint is the natural next experiment. Anomalies in `label=both`: PB1 dips at id095 (0.035→0.026, -25%) — pos and neg shifting partially out-of-phase on the unconstrained slot; at id100 PB2 is non-significant (p=0.35) but PB1 already detects shift (p=0.002) — reversed from HA-NA where the constrained slot was first to detect. Sigma values for PB2/PB1: σ_S1=11.3474, σ_S2=0.4144 (vs HA-NA 29.3192 / 1.0720; PB2/PB1 features more concentrated). Not tested: ESM-2 cross-check, PB1-only direction, nt sweep (feasible only id100..id097 on PB2/PB1 nt), axis-flag composition sanity. See `docs/results/2026-05-26_pb2_pb1_PB2only_idXX_sweep.md`.

---

## Recent run outputs (May 2026) — explicit file paths

Pointer index for the most recent results so a cold session can jump
straight to the right writeup / plot / CSV / bundle. Plots and CSVs
in `results/.../sweep_aggregate/` are gitignored (regenerable from the
aggregator); writeups and bundles are committed.

### Single-slot HA-only idXX sweep (the headline experiment, 2026-05-24 → 25)

- **Writeup**: `docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`
  — full tables (MMD pos/neg/both × 2 feature spaces; perf mean±std
  for MLP, single-seed for LGBM/1-NN), Cramér's V coupling
  discussion, MMD↔perf relationship, caveats, end-to-end reproduce
  commands (calls the wrapper scripts below).
- **Plots** (under `results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate/ha_na_HAonly/` — moved 2026-05-26 to match the per-sweep namespacing convention; PB2/PB1 outputs live alongside under `pb2_pb1_PB2only/`):
  - `sweep_mmd_vs_idxx.png` — MMD trajectories (HA, NA, pair × ESM-2, aa k=3).
  - `sweep_perf_vs_idxx.png` — Test F1 / AUC-ROC / MCC vs idXX (MLP mean±std band, per-seed dots, LGBM + 1-NN single seed).
  - `sweep_perf_vs_mmd_pair_kmer_aa_{pos,neg,both}.png` — F1 vs S2 pair MMD² scatter, one per label_filter, annotated by idXX.
  - `coupling/sweep_coupling_vs_idxx.png` — Cramér's V per axis (NA subtype, host, year-bin) vs idXX.
  - `coupling/sweep_coupling_purity_bars.png` — % of pairs in ≥95%-pure HA clusters per axis.
  - `coupling/sweep_coupling_heatmap_id098.png` — top-30 HA clusters × NA subtype.
- **CSVs** (same dir):
  - `sweep_combined.csv` — MMD long-format (idxx × role × feature_space × label_filter).
  - `sweep_perf.csv` — perf long-format (idxx × model × seed).
  - `sweep_perf_summary.csv` — perf wide-format (mean/std/min/max/count).
  - `reference_baselines.csv` — random / seq_disjoint / cluster_id099 MMD baselines for cross-reference.
  - `coupling/coupling_summary.csv` — Cramér's V + purity per (idxx, axis).
- **Bundles** (committed in `conf/bundles/`):
  - Six datasets: `flu_ha_na_cluster_aa_id{100,099,098,097,096,095}_HAonly.yaml`.
  - Training: `flu_ha_na_kmer_aa_k3.yaml` (aa k=3 + Test 3).
- **Run dirs**: 18 model dirs under `models/flu/July_2025/runs/training_flu_ha_na_kmer_aa_k3_HAonly_id{idxx}[_seedN]_<ts>/` plus matching `baseline_lgbm_*` and `baseline_knn1_margin_*` (12 baseline dirs, single-seed).

### Feasibility pre-flight (2026-05-24, what motivated single-slot)

- **Writeup**: `docs/results/2026-05-24_cluster_disjoint_feasibility_HA_NA.md` — bilateral vs single-slot, aa vs nt; tabular pre-flight at id100..id085.
- **CSVs** (under `results/flu/July_2025/runs/cluster_disjoint_feasibility/`):
  - Bilateral: `feasibility_ha_na_{aa,nt}.csv`.
  - Single-slot: `single_slot_feasibility_ha_na_{aa,nt}.csv`.

### PB2-PB1 single-slot PB2-only sweep (falsification sibling, 2026-05-26)

- **Writeup**: `docs/results/2026-05-26_pb2_pb1_PB2only_idXX_sweep.md` —
  definitions of constrained/unconstrained/pair MMD upfront, full MMD
  tables across all 3 label_filters (pos / neg / both with `both` as
  headline), Cramér's V coupling pre-check (V=0.76→0.41 vs HA-NA's
  0.85→0.53), perf tables (MLP mean±std across 3 seeds; LGBM + 1-NN
  single seed), H1-H5 hypothesis verdict table.
- **Plots** (under `results/flu/July_2025/runs/split_separation_mmd/sweep_aggregate/pb2_pb1_PB2only/`):
  - `sweep_mmd_vs_idxx.png` — single panel (aa k=3 only).
  - `sweep_perf_vs_idxx.png` — F1 / AUC-ROC / MCC × MLP, LGBM, 1-NN.
  - `sweep_perf_vs_mmd_pair_kmer_aa_{pos,neg,both}.png` — F1 vs S2 pair MMD².
- **CSVs** (same dir): `sweep_combined.csv`, `sweep_perf.csv`,
  `sweep_perf_summary.csv` (mean±std across seeds). Reference baselines
  intentionally absent (`--skip_reference_baselines` — HA-NA baselines
  would be misleading on PB2/PB1 axes).
- **Pre-flight CSVs** (under `results/flu/July_2025/runs/cluster_disjoint_feasibility/`):
  - `single_slot_feasibility_pb2_pb1_{aa,nt}.csv` — Phase 1.
  - `cluster_pair_coupling_precheck_{pb2_pb1,ha_na}_aa.csv` — Phase 2
    Cramér's V (PB2/PB1 + HA-NA reference, apples-to-apples).
- **Bundles** (committed in `conf/bundles/`):
  - Six datasets: `flu_pb2_pb1_cluster_aa_id{100,099,098,097,096,095}_PB2only.yaml`.
  - Training: `flu_pb2_pb1_kmer_aa_k3.yaml` (pre-existing).
- **Run dirs**: 18 MLP dirs under
  `models/flu/July_2025/runs/training_flu_pb2_pb1_kmer_aa_k3_PB2only_id{idxx}[_seedN]_<ts>/`
  plus matching `baseline_lgbm_*` and `baseline_knn1_margin_*` (the
  redundant seed=43/44 baseline dirs were created before the
  `stage4_sweep.sh` baseline-skip patch took effect; their contents
  are byte-identical to seed=42).

### MMD baselines (context for the sweep, 2026-05-24)

- **S1 (per-slot)**: `docs/results/2026-05-24_mmd_per_slot_results.md` — random / seq_disjoint / bilateral cluster_id099 × HA / NA × ESM-2 / aa k=3 / nt k=6.
- **S2 (per-pair, Test 3)**: `docs/results/2026-05-24_mmd_per_pair_results.md` — same 3 routings × 3 feature spaces on the pair representation.
- Baseline phase-2 CSVs: `results/.../split_separation_mmd/phase2_perm_<routing>_<role>_<fs>[_test3].csv`.

### DataSAIL L(π) (negative result, 2026-05-24)

- **Writeup**: `docs/results/2026-05-24_datasail_lpi_results.md` — mmseqspp similarity collapses all routings to the partition-shape constant ~0.34; MMD took over as the working metric. See BACKLOG.md "DataSAIL follow-ups" #1 (marked DONE).

### Wrapper scripts to rebuild any of the above

- `scripts/stage4_sweep.sh` — Stage 4 across multiple datasets × seeds in parallel (bash-and-zsh portable). `--help` for full flag list. Baselines run only on the first seed since 2026-05-26 (avoids 3× redundant LGBM/1-NN runs; commit 412ac88).
- `scripts/mmd_sweep.sh` — S1 HA + S1 NA + S2 pair MMD sweep across multiple datasets for one (feature_space × label_filter) configuration.
- `scripts/pb2_pb1_phase4_5_launch.sh` — one-shot launcher for the PB2-PB1 PB2-only Phase 4 (MMD sweep × 3 label filters) + Phase 5 (training × 3 seeds) parallel run. Self-probes σ_S1 and σ_S2 from id099 PB2-only before dispatching the sweeps. Easy to copy + edit for the next pair/direction.
- `python -m src.analysis.aggregate_mmd_single_slot_sweep` — produces every CSV + plot listed above (auto-detects available seeds, label_filters, feature spaces). Defaults reproduce HA-NA; flags override for PB2/PB1 and future pairs.

---

## Roadmap (from 02/10/2026 meeting)

Priority experiments for publication:
1. **Cross-validation** (N splits, mean ± std metrics) — `dataset.n_folds` in production (single-dataset N-folds builder, per-fold D3 feasibility, Stage 4 auto-detect of `fold_*/`). Canonical reference: `docs/methods/splits.md` § 2. Historical design log: `docs/plans/done/2026-05-27_kfold_variance_estimation_plan.md`. Remaining work (seq_disjoint / bilateral / metadata_holdout k-fold; variance-source extensions): `docs/plans/2026-05-28_kfold_remaining.md`.
2. **Large dataset** (full Flu A, ~100K isolates) — HPC required (Polaris)
3. **Temporal holdout** (train 2021–2023, test 2024) — use `dataset.metadata_holdout` under v2 (year-axis is the degenerate case). The legacy `year_train`/`year_test` keys were retired 2026-05-11.
4. **Genome features** (k-mers + XGBoost/LightGBM, then GenSLM) — unified `preprocess_flu.py` now in production (emits `genome_final.csv` alongside `protein_final.csv`)
5. **PB2/PB1 + H3N2 bundle** — trivial; one new bundle
6. **Accuracy vs genetic distance** — needs clade metadata from BV-BRC

---

## HPC (ALCF Polaris)

- PBS job arrays, not SLURM. Hydra's `submitit` launcher is SLURM-only — do not use it.
- For CV/parallel runs: PBS job array where `PBS_ARRAY_INDEX` maps to fold ID; pass as Hydra override `dataset.fold_id=${FOLD}`.
- For 8-GPU dev cluster (no scheduler): Python launcher using `subprocess.Popen` per fold, each with `CUDA_VISIBLE_DEVICES=K`.
- Stage 2 (embeddings) is the most GPU-intensive. Stage 4 (training) is modest.

---

## What Is NOT Maintained

- `src/preprocess/preprocess_bunya_protein.py` — Bunya preprocessing; see maintenance note in file
- `conf/bundles/bunya_base.yaml` — Bunya experiment config (renamed 2026-05-10 from `bunya.yaml`); see maintenance note in file

## What Is In Development (Not Yet Production)

- `src/utils/dna_utils.py` — DNA sequence QC utilities
- `dataset.split_strategy.single_slot` cluster_disjoint mode (added 2026-05-24):
  exercised on Flu A HA-NA HA-only at id100..id095 (6 bundles, all built
  successfully, MMD + MLP/LGBM/1-NN results landed). Untested directions:
  NA-only, PB2-PB1, nt cluster_alphabet, multi-seed. Routing + audit code
  in `src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df` and
  `src/datasets/dataset_segment_pairs_v2.py::split_dataset_v2`.

Note: unified Flu preprocessing (`preprocess_flu.py`), the temporal-holdout
mechanism, and Experiment B-nt (nt-level cluster_disjoint) were previously
listed here. All are now in production. Preprocessing entry point:
`scripts/stage1_preprocess_flu.sh`. Temporal-holdout `year_train` /
`year_test` keys were retired 2026-05-11 in favor of the more general
`dataset.metadata_holdout` (year-axis is its degenerate case, multi-axis
cross-population holdouts are now first-class); see
`docs/plans/done/2026-05-11_metadata_holdout_plan.md`. Experiment B-nt
landed 2026-05-15 — CDS extractor (`src/preprocess/extract_cds_dna.py`),
`dataset.split_strategy.cluster_alphabet: aa|nt` knob, 4 nt cluster bundles,
LGBM + 1-NN results; on Flu A only id100/id099 are feasible (same ceiling
as aa); see `docs/results/2026-05-15_cluster_disjoint_nt_results.md`.

K-mer aa support is in production end-to-end as of 2026-05-13 (Stage 2b
CLI, loader, MLP, baselines, bundles). `kmer.alphabet ∈ {nt, aa}`,
default `nt`. Bundles: `flu_{ha_na,pb2_pb1}_kmer_{nt,aa}_k3.yaml`.
See `docs/plans/2026-05-13_aa_kmer_and_cache_symmetry_plan.md`.

Scaling limits (exhaustive `|alphabet|^k` vocab; current pipeline does
NOT do observed-vocab or feature hashing):

- **nt** practical to k≈10 (4^10 ≈ 1M cols, ~2 GB MLP first layer at default `hidden_dims=[512,…]`). Production: k=6.
- **aa** practical ceiling is k=4 (20^4 = 160K cols, ~330 MB first layer). Current bundles use k=3 (8K cols).
- **aa k=5** breaks GPU: 3.2M cols → 1.6B params in the first linear layer (~6.5 GB just for weights), plus per-row densification of 12+ MB makes batches huge.
- **aa k=6** breaks build: 64M-entry vocab list + dict ≈ 10 GB Python memory at compute time. First MLP layer would also be 32B params.
- **aa k≥7** OOMs anywhere.

If aa k≥5 is needed, would require redesign: observed-vocab (only
enumerate k-mers seen in the data) or feature hashing
(`HashingVectorizer`-style). See "Scaling and practical limits"
in `docs/methods/kmer_features.md` for the full numeric breakdown.

## Aggregator Output Convention

Machine-generated aggregator outputs (`baselines_vs_mlp_heatmap.png`,
`baselines_vs_mlp_overall.png`, `baselines_vs_mlp.csv` and similar) live
under `results/{virus}/{data_version}/runs/`, mirroring the layout of
`data/datasets/` and `models/`. The `docs/` tree is reserved for
hand-authored writeups, plans, and methods notes — not for files an
aggregator script writes on every run.

---

## Directory Layout

```
viral-segmatch/
├── CLAUDE.md                   # This file (auto-loaded by Claude Code)
├── .claude/
│   ├── settings.json           # Claude Code permissions (deny/allow rules)
│   └── memory.md               # Compact project memory — read this every session
├── README.md                   # Project overview
├── roadmap_v1.md               # Experiment plan v1 (02/10/2026 meeting)
├── roadmap_v2.md               # Experiment plan v2 (updated 03/12/2026 meeting)
├── paper_outline_v1.md         # Paper outline v1 (initial draft)
├── paper_outline_v2.md         # Paper outline v2 (updated 03/12/2026 meeting)
├── _ongoing_work.md            # Technical notes on interactions, findings
├── _notes.txt                  # Ad-hoc questions and TODOs
├── src/                        # Python source code
├── scripts/                    # Shell pipeline wrappers (stage2–4)
├── conf/                       # Hydra configs
│   └── bundles/                # One YAML per named experiment
├── eda/                        # Exploratory analysis scripts (not pipeline)
├── examples/                   # HuggingFace reference scripts (not pipeline)
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Technical docs
├── documentation/              # User guides
└── data/                       # Not in git; symlinked raw data
```

---

## Per-machine Git Setup

Run once after cloning on each new machine (writes to `.git/config`, not tracked by git):
```bash
git config pull.rebase true   # avoid "need to reconcile divergent branches" on git pull
```

---

## Conventions

- **Experiment naming**: `{virus}_{proteins}_{n_isolates}[_{modifiers}]`
- **Timestamps**: All run directories include `YYYYMMDD_HHMMSS`
- **Shared vs. run-specific**: Preprocessing and embeddings are shared per `{virus}/{data_version}`. Datasets and models are per run in `runs/` subdirectories.
- **Seed system**: Hierarchical — `master_seed` derives all process seeds. See `docs/SEED_SYSTEM.md`.
- **Metrics**: `metrics.csv` carries F1 (binary + macro), precision, recall, AUC-ROC, AUC-PR, MCC, Brier, BCE loss. Early-stop options in `train_pair_classifier.py`: `loss`, `f1`, `auc_roc`, `auc_pr`, `mcc`. Naming convention (standardized 2026-05-18): snake_case identifiers are `auc_roc` / `auc_pr` (dict keys, CSV columns, variable names, config values); display strings are `AUC-ROC` / `AUC-PR` (titles, log messages, doc prose). The sklearn-function names `roc_auc_score` and `average_precision_score` are external and left alone. Train targets neg:pos = `neg_to_pos_ratio` (default 1.0); val/test drift to ~1.07–1.20× neg-heavy because v2's coverage phase overshoots `requested_negatives` to honor the per-sequence coverage minimum.
- **Proteins**: `preprocess_flu.py` maps GTO replicon functions to standard protein names (PB2, PB1, PA, HA, NP, NA, M1, M2, NEP).
- **Threshold notation**: `tXXX` (zero-padded, e.g., `t095`) denotes the mmseqs identity threshold (`--min-seq-id`) at `0.XXX`. Canonical across docs prose, plot labels, code (helper `_threshold_label` emits `tXXX`), bundle YAML filenames (`flu_*_t095_*.yaml`), and bundle internal `cluster_id_path` refs (`clusters_*/t095/...`). Adopted docs-only 2026-05-29; code/bundle migration finished in Phase 2 of clustering cleanup plan (commits 1-2 on `feature/phase2-pair-key-migration`, 2026-06-02). **Asymmetry until commits 3-7 land**: on-disk cluster parquets still live at `clusters_*/idXXX/` (commit 3 archives + regenerates at `tXXX/`); existing dataset and training run dirs at `data/datasets/.../runs/dataset_*_idXXX_*_<ts>` and `models/.../runs/training_*_idXXX_*_<ts>` retain their pre-Phase-2 names (new runs from commit 5 onward use `_tXXX_`).
- **Sequence hashes**: `seq_hash = md5(prot_seq)` (protein), `dna_hash = md5(dna_seq)` (nucleotide). The unprefixed `seq_hash` is the *protein* hash for historical reasons; DNA gets the explicit `dna_` prefix. In pair tables these become `seq_hash_a`/`seq_hash_b` and `dna_hash_a`/`dna_hash_b`. `seq_hash` is written by Stage 1 (`preprocess_flu.py`) and flows through Stage 3 unchanged; `dna_hash` is added by `attach_dna_to_prot_df` in Stage 3. Note: the ESM-2 embedding cache key uses `sha1(prot_seq)` (`esm2_utils.py`) — a separate namespace that is never joined back to `seq_hash`.
- **Log messages**: No emojis in print/log output. Use text prefixes instead: `ERROR:` (fatal, script will raise/exit), `WARNING:` (non-fatal but noteworthy), `Done.` (success). Decorative emojis (`📊`, `🔍`, etc.) should be removed — the surrounding text is sufficient.
- **Leakage terminology**: when discussing dataset / evaluation issues, use the canonical names from the table in `docs/plans/2026-05-07_leakage_diagnostics_plan.md`: same-pair leakage, sequence-level label imbalance, sequence-level leakage, cluster leakage, demographic shortcut leakage. Don't invent new synonyms; if a new mode is found, it should be added to that table first.
- **aa/nt vs protein/DNA**: use `aa` and `nt` at the **alphabet / residue level** (kmer alphabet, per-residue identity, "1 aa change", "97% nt identity", "201 aa" for protein length). Use `protein` and `DNA` at the **molecule / sequence level** (protein sequence, DNA sequence, protein cluster, DNA contig, protein hash, DNA hash). The pairing is aa↔nt and protein↔DNA — don't cross the streams ("aa vs DNA" is asymmetric and should be either "aa vs nt" or "protein vs DNA" depending on whether the comparison is residue-level or sequence-level). `seq_hash` is a historical name but conceptually a *protein* hash; `dna_hash` is conceptually a *DNA* hash.
- **Reading CSVs with `function_short`**: any CSV containing a `function_short` column has the literal string `'NA'` (Neuraminidase) as a value. Default pandas `read_csv()` parses `'NA'` as NaN and **silently drops the Neuraminidase rows**. Always read these CSVs with `keep_default_na=False, na_values=['']` — empty cells still become NaN (preserves missing-value detection in metadata columns like `passage`, `host`, `year`), but the literal `'NA'` stays as a string. Audited 2026-05-20: source pipeline CSVs (`protein_final.csv`, `train_pairs.csv`, etc.) use full function names so are safe; derived CSVs that use `function_short` are vulnerable. Affected files: `redundancy_stats.csv`, `mutations_tolerated_table.csv`, `sequence_length_summary.csv`, `cluster_disjoint_feasibility_*.csv`. Already patched at `cluster_analysis_summary.py:100,120`; the docstring at `load_redundancy_stats` explains the trap inline.
- **Bash tool calls**: prefer single-command invocations over compound chains (`&&`, `;`, `$(...)`, `bash -c '...'`) — the `.claude/settings.json` allow-list matcher only auto-approves statically-parseable commands, so chains trigger manual approval prompts. Use compound only when atomicity matters (`git add X && git commit ...`) or the construct is fundamentally one shell idiom (`until ... do sleep ... done` polling, true pipelines).
- **Documentation language**: prefer plain technical verbs (`removes`, `drops`, `reads`, `writes`, `joins`) over decorative or colloquial alternatives (`scrubs`, `massages`, `munges`, `slurps`). The docs are technical references, not blog posts. Use the same word for the same thing/action throughout the repo — don't synonym-vary for variety's sake.
- **Docs describe current state, not history**. Method and reference docs (`docs/methods/`, `CLAUDE.md`, `.claude/memory.md`) read as a stable description of how things are now — not a log of how they got that way. Don't write "the new X column shows Y", "adding the id094..id091 columns localizes the cliff", or "since 2026-05-26 the sweep covers …". Just describe what's true. Historical framing (algorithm switches, dataset version transitions, one-time events) belongs in result writeups (`docs/results/`) or plan docs (`docs/plans/`), where temporal language is appropriate. Good: "PB2 aa drops most at id093 → id092 (1,085 → 112, −90%)". Bad: "Adding the id094..id091 columns localizes the cliff: PB2 aa drops most at …".
- **Claims match verified evidence — no more, no less**. Don't under-claim ("7 of 8 functions show X" when you checked all 8 and they all do); don't over-claim ("every function shows X" when only PB2 and PB1 were spot-checked). The rule isn't "use round numbers"; it's "say what the evidence supports". When the scope of verification is partial, state it ("checked PB2 and PB1; not confirmed on the other 6") rather than rounding up to a universal claim.
- **Concrete numbers in takeaways when the magnitude IS the point**. "PB2 id093 → id092: 1,085 → 112 (−90%)" beats "PB2 drops sharply" because the magnitude is the takeaway. Reserve qualitative descriptors (`sharply`, `gradually`, `monotone`, `flat`) for cases where the SHAPE matters more than the magnitude, or where they summarize a trend whose numbers are visible in the table just above.
- **Design symmetry: check before proposing**. Before naming a field, data structure, or API surface, list the dimensions it covers (slot a/b, routing modes, alphabets, splits, etc.) and verify the proposal is uniform across each. Names that fit the current example better than the alternatives bake in assumptions and usually have to be redesigned. Example: `slot_b_seq_disjoint_audit` only fits `single_slot='a'` runs; `slot_a` and `slot_b` symmetric keys with routing mode at the top of the structure fit all modes uniformly. Redesign before showing.
