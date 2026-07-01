# Architecture & Reference

Descriptive reference for viral-segmatch — moved out of `CLAUDE.md` so the always-on context
stays lean. `CLAUDE.md` keeps the **behavioral rules**; this file is the **map** (pipeline,
config, source files, findings, roadmap, HPC). Read it when you need orientation on a subsystem.

---

## What This Project Does

Predicts whether two viral protein segments co-occur in the same isolate (binary classification).
Given embeddings of two protein segments from the same virus, the model distinguishes true
co-occurring pairs (positive) from artificially mixed pairs (negative).

**Approach**: Frozen ESM-2 protein embeddings (1280-dim, `esm2_t33_650M_UR50D`) → pairwise
interaction feature (e.g. `unit_diff`, `concat`) → MLP binary classifier. **Primary virus**:
Influenza A. Bunyavirales support exists but is not actively maintained.

---

## Pipeline Stages

Stages 1–2 run once per dataset (shared across experiments). Stages 3–4 are experiment-specific.

| Stage | Script | Output | Runs |
|-------|--------|--------|------|
| 1. Preprocess | `src/preprocess/preprocess_flu.py` | `data/processed/flu/{version}/protein_final.csv` + `ctg_dna_final.csv` | Once |
| 1.5. CDS extraction (opt) | `src/preprocess/extract_cds_dna.py` | `data/processed/flu/{version}/cds_dna_final.parquet` | Once, for nt_cds cluster_disjoint |
| 2. Embeddings | `src/embeddings/compute_esm2_embeddings.py` | `data/embeddings/flu/{version}/master_esm2_embeddings.h5` | Once |
| 3. Dataset | `src/datasets/dataset_segment_pairs.py` (CLI) → `dataset_segment_pairs_v2.py` (only builder; v1 retired 2026-06-03) | `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/` | Per experiment |
| 4. Train | `src/models/train_pair_classifier.py` | `models/flu/{version}/runs/training_{bundle}_{ts}/` | Per experiment |

Shell wrappers: `scripts/stage1_preprocess_flu.sh`, `stage2_esm2.sh`, `stage3_dataset.sh`,
`stage4_train.sh` (MLP only), `stage4_baselines.sh` (one baseline per call), `stage4_full.sh`
(MLP + every baseline in `bundle.baselines.enabled`), `stage4_sweep.sh` (Stage 4 across datasets ×
seeds in parallel), `mmd_sweep.sh` (S1+S2 MMD across datasets for one feature_space × label_filter).
Stage 1.5 has no wrapper — `python src/preprocess/extract_cds_dna.py --config_bundle <virus_bundle>`.

A separate **2D-CD CC builder** (`src/datasets/dataset_pairs_cc.py`) materializes bilateral
cluster-disjoint connected-component K-fold datasets with within-CC / within-fold negatives;
see `docs/plans/2026-06-09_cc_dataset_cv_plan.md`.

**Stage 3/4 decoupling**: Stage 4's shell script requires `--dataset_dir` explicitly and does not
extract/validate a bundle name from the dataset path — run Stage 3 once, Stage 4 many times with
different training bundles against the same dataset. Provenance is tracked via `training_info.json`.

---

## Configuration System

**Hydra** with a bundle-per-experiment pattern.

- Bundles: `conf/bundles/{bundle_name}.yaml` — one file per named experiment
- Virus configs: `conf/virus/flu.yaml`, `bunya.yaml`; data paths: `conf/paths/flu.yaml`, `bunya.yaml`
- Defaults: `conf/dataset/default.yaml`, `conf/embeddings/default.yaml`, `conf/training/base.yaml`, `conf/baselines/default.yaml`
- Config loader: `src/utils/config_hydra.py`

**Convention**: one bundle = one reproducible experiment. Names encode the experiment:
`flu_{proteins}_{n_isolates}[_{modifiers}]`, e.g. `flu_ha_na`, `flu_ha_na_cluster_t099`.

Key bundle parameters: `virus.selected_functions`, `dataset.max_isolates_to_process`,
`dataset.hn_subtype`, `dataset.year`, `dataset.host`,
`dataset.split_strategy.{mode,hash_key,cluster_alphabet,single_slot}`, `dataset.metadata_holdout`,
`training.slot_transform`, `training.interaction`. Sklearn-baseline knobs live under
`baseline_<name>.*` at the bundle root (defaults from `conf/baselines/default.yaml`).

`split_strategy.single_slot`: `null` (default, bilateral cluster_disjoint) | `'a'` | `'b'`.
Only consumed under `mode: cluster_disjoint`.

**Bundle organization** (see `conf/bundles/README.md`): each bundle has a
`# STATUS: active|ablation|experimental|legacy|not maintained` header. Bundles form inheritance
chains via Hydra `defaults`; base bundles stay flat, only leaf bundles move to subdirs.
`conf/bundles/paper/` is reserved for publication experiments. Three generations: Gen1
(`flu.yaml` base), Gen2 (`flu_schema.yaml` base, none+concat), Gen3 (`flu_schema_raw_*`, retired
2026-05-12; its `slot_norm + unit_diff` config remains the reference ESM-2 result). Active leaves
(`flu_ha_na`, `flu_pb2_pb1`) descend from `flu_28_major_protein_pairs_master`.

---

## Active Source Files

```
src/
  preprocess/
    preprocess_flu.py               # Stage 1: GTO → protein_final.csv + ctg_dna_final.csv
    extract_cds_dna.py              # Stage 1.5: protein_final + ctg_dna_final → cds_dna_final.parquet
    flu_genomes_eda.py              # Generates flu_genomes_metadata_parsed.csv (run once)
    preprocess_bunya_protein.py     # Bunya (NOT maintained; reference-only)
    build_mmseqs_clusters.py        # Stage 2.5: per-function mmseqs2 cluster sweep (aa+nt)
  embeddings/
    compute_esm2_embeddings.py      # Stage 2: sequences → ESM-2 HDF5 cache
    compute_kmer_features.py        # Stage 2b: DNA or protein → k-mer sparse matrix
  datasets/
    dataset_segment_pairs.py        # Stage 3: pairs + train/val/test splits (CLI)
    dataset_segment_pairs_v2.py     # v2 builder (default; coverage-first + regime-aware)
    dataset_pairs_cc.py             # 2D-CD CC builder (cluster-disjoint CC K-fold; within-CC/fold negs)
    _pair_helpers.py                # Shared helpers (seq_disjoint routing, filters, hash attach)
    _split_helpers.py               # cluster_disjoint routing (mmseqs2-based)
    _cc_helpers.py                  # within-CC isolate pool + negative samplers (CC builder)
    _megacc_cut.py                  # mega-CC edge min-cut (drop-budget 2D-CD; spectral/KL)
    _negative_regime_sampling.py    # 8-regime classifier + priority chain for racov
  models/
    train_pair_classifier.py        # Stage 4: MLP classifier training
    train_pair_baselines.py         # Stage 4: sklearn baselines (logistic, lgbm, knn1_margin, knn_vote)
  analysis/
    analyze_stage4_train.py         # Confusion matrix, ROC, FP/FN analysis
    analyze_stage{1,2,3}_*.py       # Per-stage QC scripts
    visualize_dataset_stats.py      # PCA plots, interaction diagnostics
    aggregate_experiment_results.py # Cross-bundle summary tables
    visualize_cv_results.py         # CV visualization
    cluster_disjoint_feasibility.py # Bilateral bipartite-CC feasibility pre-flight
    single_slot_cluster_disjoint_feasibility.py
    cluster_analysis_summary.py     # Post-hoc structural cluster summary
    plot_aa_vs_nt_cluster_disjoint.py    # LGBM + 1-NN cluster-disjoint comparison
    cluster_source.py               # Membership-backed cluster-map source (registry-driven)
    mmd_per_slot.py / mmd_per_pair.py / aggregate_mmd_single_slot_sweep.py
  utils/
    config_hydra.py                 # Hydra config loader (primary)
    schema.py                       # Per-alphabet column/file registry (aa, nt_cds, nt_ctg) — single source of truth
    esm2_utils.py, embedding_utils.py, kmer_utils.py
    metadata_enrichment.py, seed_utils.py, learning_verification_utils.py, plot_config.py
    clustering_utils.py             # mmseqs2 wrappers; alphabet={aa,nt_cds,nt_ctg}
    cds_utils.py                    # CDS reconstruction + cds_dna_hash
    dna_utils.py                    # DNA QC utilities (in development)
    dim_reduction_utils.py, gto_utils.py, protein_utils.py, path_utils.py, timer_utils.py
```

---

## Key Experimental Findings

> All findings below were measured under the **protein pair_key** convention unless stated.
> Phase 2 (2026-06-02 → 06-03) added a CDS-DNA pair_key option for nt_cds cluster_disjoint
> bundles (`split_strategy.pair_key_alphabet`); under it, silent codon variants count as distinct
> positives — cite the `pair_key_alphabet` used for experiments after 2026-06-03. See
> `docs/plans/2026-06-02_pair_key_alphabet_plan.md`, `docs/results/2026-06-03_phase2_postmigration_metrics.md`.

- **`unit_diff` > `concat` on homogeneous data (ESM-2 only)**: ESM-2 `concat` collapses on subtype-filtered HA/NA (AUC≈0.50); `unit_diff` succeeds. K-mer concat never collapses. `slot_norm` LayerNorm is required for ESM-2 unit_diff on homogeneous subsets.
- **K-mer ≥ ESM-2 on homogeneous data**: k=6 (4096-dim) matches/exceeds ESM-2 on mixed-subtype HA/NA; dominates on H3N2-only. K-mer features are interaction-agnostic.
- **K-mer interaction sweep (Tests 1–4, HA/NA)**: all four within seed noise; Test 3 narrowly leads (needs multi-seed before picking a winner). `unit_diff` semantics = element-wise abs first, then L2-normalize.
- **seq_disjoint scales to conserved proteins**: PB2/PB1 with `hash_key=seq` gives clean 80/10/10 despite high conservation. 1-NN edges MLP on PB2/PB1 under seq_disjoint.
- **Experiment B-nt feasibility ceiling = aa ceiling on Flu A**: nt CDS-level cluster_disjoint hits the same bipartite mega-component collapse at the same thresholds. Only id100/id099 operable on the full corpus on either alphabet. See `docs/results/2026-05-15_cluster_disjoint_nt_results.md`.
- **1-NN cosine margin ≥ LGBM at every cluster_disjoint routing**; cluster_disjoint weakens the near-neighbor signal gradually rather than eliminating it.
- **Single-slot HA-only sweep (HA-NA, id100..id095)**: monotone MMD↑ and test perf↓; biological coupling (HA-cluster ≈ NA-subtype boundary). See `docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`.
- **PB2-PB1 PB2-only sweep (falsification sibling)**: ~half the shift/F1 drop of HA-NA; model ordering REVERSES (1-NN > MLP ≈ LGBM) — residual metadata-driven leakage survives cluster_disjoint single-slot here. See `docs/results/2026-05-26_pb2_pb1_PB2only_idXX_sweep.md`.
- **2D-CD within_cc removes the cluster shortcut**: HA-NA at t099, within_cc is ~chance for both MLP and LGBM across aa k=3 and nt_cds k=6; within_fold reaches 0.87 (shortcut ≈ 0.37 AUC). See `docs/results/2026-06-29_cc_within_cc_vs_within_fold.md`.

---

## Recent Run Outputs

`docs/results/` holds the canonical writeups. Most-cited:
- `2026-05-24_single_slot_HAonly_idXX_sweep.md` — headline single-slot experiment
- `2026-05-24_cluster_disjoint_feasibility_HA_NA.md` — pre-flight that motivated single-slot
- `2026-05-26_pb2_pb1_PB2only_idXX_sweep.md` — PB2-PB1 falsification sibling
- `2026-05-24_mmd_per_{slot,pair}_results.md` — MMD baselines
- `2026-05-15_cluster_disjoint_nt_results.md` — Experiment B-nt
- `2026-06-03_phase2_postmigration_metrics.md` — pair_key migration impact
- `2026-06-29_cc_within_cc_vs_within_fold.md` — 2D-CD within_cc vs within_fold (cluster shortcut ≈ 0.37 AUC)

Per-sweep aggregator outputs (gitignored) live under
`results/{virus}/{data_version}/runs/<analysis_name>/sweep_aggregate/<pair_direction>/`. Aggregator:
`python -m src.analysis.aggregate_mmd_single_slot_sweep`. Rebuild wrappers: `scripts/stage4_sweep.sh`,
`scripts/mmd_sweep.sh`, `scripts/pb2_pb1_phase4_5_launch.sh`.

**Aggregator output convention**: machine-generated aggregator outputs (`baselines_vs_mlp_*.{png,csv}`
and similar) live under `results/{virus}/{data_version}/runs/`, mirroring `data/datasets/` and
`models/`. The `docs/` tree is reserved for hand-authored writeups, plans, and methods notes.

---

## Roadmap (from 02/10/2026 meeting)

1. **Cross-validation** (N splits, mean ± std) — `dataset.n_folds`. Ref: `docs/methods/splits.md` § 2; remaining work `docs/plans/2026-05-28_kfold_remaining.md`.
2. **Large dataset** (full Flu A, ~100K isolates) — HPC required (Polaris).
3. **Temporal holdout** (train 2021–2023, test 2024) — `dataset.metadata_holdout` (year-axis is the degenerate case; legacy `year_train`/`year_test` retired 2026-05-11).
4. **Genome features** (k-mers + XGBoost/LightGBM, then GenSLM).
5. **PB2/PB1 + H3N2 bundle** — trivial; one new bundle.
6. **Accuracy vs genetic distance** — needs clade metadata from BV-BRC.

---

## HPC (ALCF Polaris)

- PBS job arrays, not SLURM. Hydra's `submitit` launcher is SLURM-only — do not use it.
- CV/parallel: PBS job array where `PBS_ARRAY_INDEX` → fold ID, passed as `dataset.fold_id=${FOLD}`.
- 8-GPU dev cluster (no scheduler): Python launcher using `subprocess.Popen` per fold, each with `CUDA_VISIBLE_DEVICES=K`.
- Stage 2 (embeddings) is the most GPU-intensive; Stage 4 (training) is modest.

---

## Maintenance Status

**NOT maintained**: `src/preprocess/preprocess_bunya_protein.py`, `conf/bundles/bunya_base.yaml`.

**In development**: `src/utils/dna_utils.py`; `dataset.split_strategy.single_slot` (exercised on
HA-NA HA-only and PB2-PB1 PB2-only; untested: NA-only / PB1-only, nt cluster_alphabet on single_slot,
CV variance).

**K-mer aa scaling limits** (exhaustive `|alphabet|^k` vocab; no observed-vocab/hashing yet): nt
practical to k≈10 (production k=6); aa ceiling k=4 (20^4=160K cols; current bundles k=3); aa k≥5
OOMs. See `docs/methods/kmer_features.md`.

---

## Directory Layout

```
viral-segmatch/
├── CLAUDE.md                   # Always-on behavioral rules (auto-loaded)
├── docs/architecture.md        # This file — descriptive reference
├── .claude/
│   ├── settings.json           # Permissions + hooks
│   └── memory.md               # Compact project memory — read every session
├── roadmap_v{1,2}.md, paper_outline_v{1,2}.md, _ongoing_work.md, _notes.txt
├── src/                        # Python source (preprocess, embeddings, datasets, models, analysis, utils)
├── scripts/                    # Shell pipeline wrappers (stage1–4)
├── conf/                       # Hydra configs (conf/bundles/ = one YAML per experiment)
├── eda/, examples/, notebooks/ # Exploratory (not pipeline)
├── docs/                       # Technical docs (plans, methods, results)
├── documentation/              # User guides
└── data/                       # Not in git; symlinked raw data
```
