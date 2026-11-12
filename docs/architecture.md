# Architecture & Reference

Descriptive reference for viral-segmatch — moved out of `CLAUDE.md` so the always-on context
stays lean. `CLAUDE.md` keeps the **behavioral rules**; this file is the **map** (pipeline,
config, source files, findings, roadmap, HPC). Read it when you need orientation on a subsystem.

---

## What This Project Does

Predicts whether two viral protein segments co-occur in the same isolate (binary classification).
Given embeddings of two protein segments from the same virus, the model distinguishes true
co-occurring pairs (positive) from artificially mixed pairs (negative). A mixed pair counts as a
negative only when its two sequences co-occur in no isolate of the corpus, which Stage 3 enforces
with the **co-occurrence blocking set** (`docs/methods/glossary.md`).

**Approach**: one feature type per run, set by `training.feature_source` — frozen ESM-2
protein embeddings (`esm2`; 1280-dim, `esm2_t33_650M_UR50D`), k-mer counts (`kmer`), or per-site
codes, one per nucleotide, codon or amino acid (`site`) → pairwise interaction feature (e.g.
`unit_diff`, `concat`) → MLP or sklearn-baseline classifier (logistic, LightGBM, 1-NN margin,
kNN vote). **Primary virus**: Influenza A. Bunyavirales support exists but is not actively
maintained.

---

## Pipeline Stages

Stages 1–2 run once per dataset (shared across experiments). Stages 3–4 are experiment-specific.

| Stage | Script | Output | Runs |
|-------|--------|--------|------|
| 1. Preprocess | `src/preprocess/preprocess_flu.py` | `data/processed/flu/{version}/protein_final.{csv,parquet}` + `ctg_dna_final.{csv,parquet}` | Once |
| 1.5. CDS extraction | `src/preprocess/extract_cds_dna.py` | `data/processed/flu/{version}/cds_dna_final.parquet` | Once; read by nt_cds runs and per-site features |
| 2. ESM-2 embeddings | `src/embeddings/compute_esm2_embeddings.py` | `data/embeddings/flu/{version}/master_esm2_embeddings.h5` | Once |
| 2b. K-mer features | `src/embeddings/compute_kmer_features.py` | `kmer_features_{alphabet}_k{k}.*`, alongside the ESM-2 cache | Once per alphabet and k |
| 2c. Per-site features | `src/embeddings/compute_site_features.py` | `site_features_{unit}_{protein}_{pin}nt.*`, alongside the k-mer cache | Once per protein, unit and pinned length |
| 3. Dataset | `src/datasets/dataset_segment_pairs.py` (CLI) → `dataset_segment_pairs_v2.py` (only builder; v1 retired 2026-06-03) | `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/` | Per experiment |
| 4. Train | `src/models/train_pair_classifier.py` | `models/flu/{version}/runs/training_{bundle}_{ts}/` | Per experiment |

Shell wrappers: `scripts/stage1_preprocess_flu.sh`, `stage2_esm2.sh`, `stage2b_kmer.sh`,
`stage3_dataset.sh`, `stage3_cc.sh` (2D-CD CC builder), `stage4_train.sh` (MLP only),
`stage4_baselines.sh` (one baseline per call), `stage4_full.sh` (MLP + every baseline in
`bundle.baselines.enabled`), `stage4_sweep.sh` (Stage 4 across datasets ×
seeds in parallel), `mmd_sweep.sh` (S1+S2 MMD across datasets for one feature_space × label_filter).
Stage 1.5 has no wrapper — `python src/preprocess/extract_cds_dna.py --config_bundle <virus_bundle>`.

A separate **2D-CD CC builder** (`src/datasets/dataset_pairs_cc.py`) materializes bilateral
cluster-disjoint connected-component K-fold datasets with within-CC / within-fold negatives;
see `docs/plans/done/2026-06-09_cc_dataset_cv_plan.md`.

**Stage 3/4 decoupling**: Stage 4's shell script requires `--dataset_dir` explicitly and does not
extract/validate a bundle name from the dataset path — run Stage 3 once, Stage 4 many times with
different training bundles against the same dataset. Provenance is tracked via `training_info.json`.

---

## Configuration System

**Hydra** with a bundle-per-experiment pattern.

- Bundles: `conf/bundles/{bundle_name}.yaml` — one file per named experiment
- Virus configs: `conf/virus/flu.yaml`, `bunya.yaml`; data paths: `conf/paths/flu.yaml`, `bunya.yaml`
- Defaults: `conf/dataset/default.yaml`, `conf/embeddings/default.yaml`, `conf/kmer/default.yaml`,
  `conf/site/default.yaml`, `conf/training/base.yaml`, `conf/baselines/default.yaml`
- Experiment specs: `conf/experiments/*.yaml` — the settings one experiment must have, applied to
  and checked against every bundle it uses (`src/utils/experiment_spec.py`)
- Config loader: `src/utils/config_hydra.py`

**Convention**: one bundle = one reproducible experiment. Names encode the experiment:
`flu_{proteins}_{n_isolates}[_{modifiers}]`, e.g. `flu_ha_na`, `flu_ha_na_h3n2_2024_random_cv4`.

Key bundle parameters: `virus.selected_functions`, `dataset.max_isolates_to_process`,
`dataset.hn_subtype`, `dataset.year`, `dataset.host`,
`dataset.split_strategy.{mode,hash_key,cluster_alphabet,single_slot}`, `dataset.metadata_holdout`,
`training.feature_source`, `training.slot_transform`, `training.interaction`. Sklearn-baseline
knobs live under `baseline_<name>.*` at the bundle root (defaults from
`conf/baselines/default.yaml`).

`split_strategy.single_slot`: `null` (default, bilateral cluster_disjoint) | `'a'` | `'b'`.
Only consumed under `mode: cluster_disjoint`.

**Bundle organization** (see `conf/bundles/README.md`): bundles form inheritance chains via Hydra
`defaults`. 64 of the 130 bundles carry a `# STATUS:` header (`active`, `production`,
`experimental`, `ablation`, `legacy` or `not maintained`). `conf/bundles/flu_28/` holds 28
per-pair bundles that inherit from `flu_28_major_protein_pairs_master`; every other bundle sits
at the top level.

---

## Active Source Files

The pipeline and production files. `src/analysis/` is summarised in one line.

```
src/
  preprocess/
    preprocess_flu.py               # Stage 1: GTO → protein_final + ctg_dna_final
    extract_cds_dna.py              # Stage 1.5: Stage 1 outputs → cds_dna_final.parquet
    build_clusters.py               # mmseqs2 clusters per function and identity threshold
    build_cluster_membership.py     # per-record cluster-membership table
    build_mmseqs_clusters.py        # back-compat shim: set-cover cluster builder
    build_ood_clusters.py           # back-compat shim: OOD cluster builder
    flu_genomes_eda.py              # writes flu_genomes_metadata_parsed.csv (run once)
    preprocess_bunya_protein.py     # Bunya (NOT maintained; reference-only)
  embeddings/
    compute_esm2_embeddings.py      # Stage 2: proteins → ESM-2 HDF5 cache
    compute_kmer_features.py        # Stage 2b: DNA or protein → k-mer sparse matrix
    compute_site_features.py        # Stage 2c: CDS → per-site codes (nt, codon or aa)
  datasets/
    dataset_segment_pairs.py        # Stage 3 CLI; builds with dataset_segment_pairs_v2
    dataset_segment_pairs_v2.py     # Stage 3 builder (coverage-first negatives, within-split dedup)
    dataset_pairs_cc.py             # Stage 3 CC builder: 2D-CD cluster-disjoint K-fold datasets
    build_cc_structure.py           # Stage 2.5: positive pair universe + per-t CC structure
    build_random_arm.py             # random-split control arm for an existing k-fold dataset
    _pair_helpers.py                # loading, filtering, splitting; co-occurrence blocking set
    _split_helpers.py               # split routing for non-default modes
    _bigraph.py                     # cluster-level bigraph builder
    _megacc_cut.py                  # mega-CC edge min-cut
    _cc_helpers.py                  # within-CC isolate pool + negative sampling (CC builder)
    _negative_sampling.py           # negative-pair samplers shared by both builders
    _negative_regime_sampling.py    # metadata-aware negative regimes
    _positive_pair_selection.py     # positives with unique sequence endpoints per slot
  models/
    train_pair_classifier.py        # Stage 4: MLP classifier
    train_pair_baselines.py         # Stage 4: sklearn baselines
    baselines/                      # logistic, lgbm, knn1_margin, knn_vote
    baseline_fits.py                # launch one baseline fit; decide whether to skip it
    _pair_features.py               # baseline pair features (esm2, kmer or site)
    _pair_metrics.py                # metrics shared by the MLP trainer and the baselines
  utils/
    schema.py                       # per-alphabet column/file registry — single source of truth
    config_hydra.py                 # Hydra config loader
    experiment_spec.py              # read an experiment spec; check configs against it
    experiment_utils.py             # experiment tracking: git info, file hashes
    dataset_manifest.py             # manifest of pair datasets built together
    clustering_utils.py             # mmseqs2 wrappers; alphabet={aa,nt_cds,nt_ctg}
    cluster_source.py               # cluster-map lookup; imported only by src/archive/ scripts
    cds_utils.py                    # CDS extraction, translation, terminal stop
    site_utils.py                   # per-site matrix, cache reading, pair features
    kmer_utils.py                   # k-mer loading and pair features
    esm2_utils.py                   # ESM-2 model loading, embedding compute, cache save/load
    embedding_utils.py              # embedding loading and visualization
    fasta_utils.py                  # read FASTA sequences or headers
    metadata_enrichment.py          # isolate metadata join, population filters and labels
    path_utils.py, seed_utils.py, timer_utils.py, torch_utils.py
    protein_utils.py, dna_utils.py, gto_utils.py
    plot_config.py, plot_utils.py, dim_reduction_utils.py, learning_verification_utils.py
  analysis/                         # 71 scripts: per-stage QC, aggregation, plots, MMD, CC and
                                    #   bigraph diagnostics, audits, protein alignment
  archive/                          # inert; see Layering
```

### Layering

"Production" means code that participates in producing train/val/test splits, **regardless of
directory**. It lives in `src/datasets/` or `src/utils/`; `src/analysis/` may import from those,
never the reverse. The check:

```bash
grep -rnE "^\s*(from|import)\s+src\.analysis" src/datasets/ src/utils/
```

**One documented exception**, which the grep above will report: `dataset_segment_pairs_v2` imports
`src.analysis.visualize_dataset_stats`. It is allowed to stand because it is not split-producing —
the import is lazy (function-local), reached only under `if generate_visualizations`, and wrapped
in try/except so a failure degrades to a `WARNING:` rather than affecting the dataset. Moving the
visualizer to `src/utils/` would drag its plotting stack into the production layer to no benefit.
Any *new* `src/analysis` import from `src/datasets/` or `src/utils/` is a violation, not a second
exception.

`src/archive/` is inert: nothing may import from it, and it may reference anything.

---

## Results for the paper plan

`docs/results/` holds the hand-written results. These are the ones
`docs/plans/2026-10-08_paper_plan.md` cites, by task:

- **1. Establish the methods on `Human-H3N2-2024`**
  - `2026-09-07_cds_length_survey.md` — per-protein CDS completeness and length, corpus-wide and
    for Human-H3N2-2024.
  - `2026-09-08_cds_pair_capacity.md` — positive-pair capacity of the 28 schema pairs on
    Human-H3N2-2024.
  - `2026-09-08_h3n2_2024_progress_report.md` — progress report and scope decision: the dataset
    population, the four-schema-pair experiment, and where the codon-site signal sits.
- **2. Expand the population. Evaluate across hosts, subtypes, and years.**
  - `2026-11-09_protein_alignment.md` — protein alignment of the 8 modelled proteins on the Task 2
    population, to decide which records and columns to keep for codon features.
- **3. Test on future-year pairs** — no results doc yet.

**Aggregator output convention**: machine-generated aggregator outputs (`baselines_vs_mlp_*.{png,csv}`
and similar) live under `results/{virus}/{data_version}/runs/`, mirroring `data/datasets/` and
`models/`. The `docs/` tree is reserved for hand-authored writeups, plans, and methods notes.

---

## Roadmap

Current planning is in `docs/plans/2026-10-08_paper_plan.md` (Tasks 1–3). Open todos are in
`BACKLOG.md`; active and completed plans are in `docs/plans/`.

---

## HPC (ALCF Polaris)

- PBS job arrays, not SLURM. Hydra's `submitit` launcher is SLURM-only — do not use it.
- CV/parallel: PBS job array where `PBS_ARRAY_INDEX` → fold ID, passed as `dataset.fold_id=${FOLD}`
  (`scripts/run_cv_polaris.pbs`).
- 8-GPU dev cluster (no scheduler): Python launcher using `subprocess.Popen` per fold, each with
  `CUDA_VISIBLE_DEVICES=K` (`scripts/run_cv_lambda.py`).
- 28-pair sweeps: `scripts/run_allpairs_polaris_prod.sh` gives each node one protein pair, which
  runs 12-fold CV on the node's 4 GPUs through `run_cv_lambda.py`. See `polaris_plan.md`.
- Stage 2 (embeddings) is the most GPU-intensive; Stage 4 (training) is modest.

---

## Maintenance Status

**NOT maintained**: `src/preprocess/preprocess_bunya_protein.py`, `conf/bundles/bunya_base.yaml`.

**In development**: `dataset.split_strategy.single_slot` (exercised on
HA-NA HA-only and PB2-PB1 PB2-only; untested: NA-only / PB1-only, nt cluster_alphabet on single_slot,
CV variance).

**K-mer scaling limits**: the vocabulary is exhaustive (`|alphabet|^k`, no observed-vocab or
hashing), so the practical ceilings are about k=10 for nt and k=4 for aa; see
`docs/methods/kmer_features.md`. Production uses nt k=6.

---

## Directory Layout

```
viral-segmatch/
├── CLAUDE.md                   # Always-on behavioral rules (auto-loaded)
├── README.md, BACKLOG.md       # Project overview; triaged todo list
├── polaris_plan.md             # Task 11 (28-pair sweeps) on Polaris: plan, runs, results
├── docs/architecture.md        # This file — descriptive reference
├── .claude/
│   ├── settings.json           # Permissions + hooks
│   └── memory.md               # Compact project memory — read every session
├── src/                        # Python source (preprocess, embeddings, datasets, models, analysis, utils)
├── scripts/                    # Shell pipeline wrappers (stage1–4) and CV launchers
├── conf/                       # Hydra configs (conf/bundles/ = one YAML per experiment)
├── tests/                      # test_*.py modules
├── eda/, examples/, notebooks/ # Exploratory (not pipeline)
├── docs/                       # Technical docs (plans, methods, results)
├── documentation/              # User guides
├── data/                       # Inputs and Stage 1–3 outputs; only data/raw/ links + README tracked
└── models/, results/, reports/ # Run outputs, aggregates and reports (gitignored)
```
