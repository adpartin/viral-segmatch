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
| 3. Dataset | `src/datasets/dataset_segment_pairs.py` (CLI) → `dataset_segment_pairs_v2.py` (only builder; v1 retired 2026-06-03) | `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/` | Per experiment |
| 4. Train | `src/models/train_pair_classifier.py` | `models/flu/{version}/runs/training_{bundle}_{ts}/` | Per experiment |

Shell wrappers: `scripts/stage1_preprocess_flu.sh`, `scripts/stage2_esm2.sh`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh` (MLP only), `scripts/stage4_baselines.sh` (one baseline per call), `scripts/stage4_full.sh` (MLP + every baseline in `bundle.baselines.enabled`), `scripts/stage4_sweep.sh` (Stage 4 across multiple datasets × seeds in parallel), `scripts/mmd_sweep.sh` (S1+S2 MMD across multiple datasets for one feature_space × label_filter). Stage 1.5 has no shell wrapper — invoked directly (`python src/preprocess/extract_cds_dna.py --config_bundle <virus_bundle>`).

**Stage 3/4 decoupling**: Stage 4's shell script requires `--dataset_dir` explicitly and does not extract or validate a bundle name from the dataset path. This allows running Stage 3 once and Stage 4 multiple times with different training bundles against the same dataset. Provenance is tracked via `training_info.json` saved in the training output dir.

---

## Configuration System

**Hydra** with a bundle-per-experiment pattern.

- Bundles: `conf/bundles/{bundle_name}.yaml` — one file per named experiment
- Virus configs: `conf/virus/flu.yaml`, `conf/virus/bunya.yaml`
- Data paths: `conf/paths/flu.yaml`, `conf/paths/bunya.yaml`
- Defaults: `conf/dataset/default.yaml`, `conf/embeddings/default.yaml`, `conf/training/base.yaml`, `conf/baselines/default.yaml`
- Config loader: `src/utils/config_hydra.py`

**Convention**: One bundle = one reproducible experiment. Bundle names encode the experiment:
`flu_{proteins}_{n_isolates}[_{modifiers}]`, e.g., `flu_ha_na_5ks`, `flu_ha_na_regimes`.

Key bundle parameters: `virus.selected_functions`, `dataset.max_isolates_to_process`, `dataset.hn_subtype`, `dataset.year`, `dataset.host`, `dataset.split_strategy.{mode,hash_key,cluster_alphabet,single_slot}`, `dataset.metadata_holdout`, `training.slot_transform`, `training.interaction`. Sklearn-baseline knobs live under `baseline_<name>.*` at the bundle root (defaults inherited from `conf/baselines/default.yaml`).

`split_strategy.single_slot`: `null` (default, bilateral cluster_disjoint) | `'a'` (constrain slot-a clusters only) | `'b'` (constrain slot-b only). Only consumed under `mode: cluster_disjoint`.

**Bundle organization** (see `conf/bundles/README.md`):
- Each bundle has a `# STATUS: active|ablation|experimental|legacy|not maintained` header.
- Bundles form inheritance chains via Hydra `defaults`. Base bundles must stay flat; only leaf bundles can be moved to subdirs.
- `conf/bundles/paper/` — reserved for publication experiments.
- Three generations: Gen1 (`flu.yaml` base, no schema ordering), Gen2 (`flu_schema.yaml` base, none+concat), Gen3 (`flu_schema_raw_*` base, retired 2026-05-12; its `slot_norm + unit_diff` config remains the reference ESM-2 result). Active leaves (`flu_ha_na`, `flu_pb2_pb1`) descend from `flu_28_major_protein_pairs_master`.

---

## Active Source Files

```
src/
  preprocess/
    preprocess_flu.py               # Stage 1: GTO → protein_final.csv + genome_final.csv
    extract_cds_dna.py              # Stage 1.5: protein_final + genome_final → cds_final.parquet
    flu_genomes_eda.py              # Generates flu_genomes_metadata_parsed.csv (run once)
    preprocess_bunya_protein.py     # Bunya (NOT maintained; reference-only)
    build_mmseqs_clusters.py        # Stage 2.5: per-function mmseqs2 cluster sweep (easy-linclust, aa+nt); producer for cluster_disjoint routing
  embeddings/
    compute_esm2_embeddings.py      # Stage 2: sequences → ESM-2 HDF5 cache
    compute_kmer_features.py        # Stage 2b: DNA or protein → k-mer sparse matrix
  datasets/
    dataset_segment_pairs.py        # Stage 3: pairs + train/val/test splits (CLI)
    dataset_segment_pairs_v2.py     # v2 builder (default; coverage-first + regime-aware)
    _pair_helpers.py                # Shared helpers (seq_disjoint routing, filters)
    _split_helpers.py               # cluster_disjoint routing (mmseqs2-based)
    _megacc_cut.py                  # mega-CC edge min-cut (drop-budget 2D-CD; spectral/KL)
    _negative_regime_sampling.py    # 8-regime classifier + priority chain for racov
  models/
    train_pair_classifier.py        # Stage 4: MLP classifier training
  analysis/
    analyze_stage4_train.py         # Confusion matrix, ROC, FP/FN analysis
    visualize_dataset_stats.py      # PCA plots, interaction diagnostics
    aggregate_experiment_results.py # Cross-bundle summary tables
    visualize_cv_results.py         # CV visualization
    analyze_stage{1,2,3}_*.py       # Per-stage QC scripts
    cluster_disjoint_feasibility.py # Bilateral bipartite-CC feasibility pre-flight
    single_slot_cluster_disjoint_feasibility.py  # Single-slot atom feasibility pre-flight
    cluster_analysis_summary.py     # Post-hoc structural summary
    plot_aa_vs_nt_cluster_disjoint.py    # LGBM + 1-NN cluster-disjoint comparison
    mmd_per_slot.py                 # S1 per-slot MMD (RBF + permutation test)
    mmd_per_pair.py                 # S2 per-pair MMD on production interaction
    aggregate_mmd_single_slot_sweep.py   # Sweep rollup + plots (parametrized for any pair/direction)
  utils/
    config_hydra.py                 # Hydra config loader (primary)
    esm2_utils.py                   # ESM-2 tokenization, batch embedding
    embedding_utils.py              # Load/index embeddings from HDF5
    metadata_enrichment.py          # Load flu_genomes_metadata_parsed.csv
    seed_utils.py                   # Hierarchical seed system
    learning_verification_utils.py  # Karpathy-style sanity checks
    plot_config.py                  # Colors, protein name mapping
    kmer_utils.py                   # Load k-mer features, pair construction
    clustering_utils.py             # mmseqs2 wrappers; alphabet={aa,nt}, algorithm={cluster,linclust}
    cds_utils.py                    # CDS reconstruction (parse_location, extract_cds_dna, translate_dna, compute_cds_dna_hash)
    dna_utils.py                    # DNA QC utilities (in development)
    dim_reduction_utils.py          # PCA/UMAP wrappers
    gto_utils.py, protein_utils.py, path_utils.py, timer_utils.py
```

---

## Key Experimental Findings

> **All findings below were measured under the protein pair_key
> convention** unless stated otherwise. Phase 2 (2026-06-02 → 2026-06-03)
> introduced a CDS-DNA pair_key option for nt_cds cluster_disjoint
> bundles (`split_strategy.pair_key_alphabet`); under that option silent
> codon variants of the same protein-pair count as distinct positives,
> inflating the pair universe and opening a new "DNA-variant" leakage
> mode. New experiments after 2026-06-03 should cite the
> `pair_key_alphabet` they used. See
> `docs/plans/2026-06-02_pair_key_alphabet_plan.md` and
> `docs/results/2026-06-03_phase2_postmigration_metrics.md`.

Headline takeaways (full detail in `docs/results/`):

- **`unit_diff` > `concat` on homogeneous data (ESM-2 only)**: ESM-2 `concat` collapses on subtype-filtered HA/NA (AUC≈0.50); `unit_diff` succeeds. K-mer concat never collapses — the failure is a property of ESM-2's protein-type geometry, not concatenation. `slot_norm` LayerNorm is required for ESM-2 unit_diff on homogeneous subsets.
- **K-mer ≥ ESM-2 on homogeneous data**: k=6 (4096-dim) matches or exceeds ESM-2 on mixed-subtype HA/NA. On H3N2-only k-mer dominates. K-mer features are interaction-agnostic.
- **K-mer interaction sweep (Tests 1–4, HA/NA)**: all four within seed noise; Test 3 narrowly leads on aggregate metrics. Multi-seed needed before picking a winner. `unit_diff` semantics = element-wise abs first, then L2-normalize.
- **seq_disjoint scales to conserved proteins**: PB2/PB1 with `hash_key=seq` produces clean 80/10/10 splits despite high conservation. 1-NN edges MLP on PB2/PB1 under seq_disjoint (consistent with conservation interpretation).
- **Experiment B-nt feasibility ceiling = aa ceiling on Flu A**: nt CDS-level cluster_disjoint hits the same bipartite mega-component collapse at the same thresholds. Only id100 and id099 are operable on the full corpus on either alphabet. See `docs/results/2026-05-15_cluster_disjoint_nt_results.md`.
- **1-NN cosine margin ≥ LGBM at every cluster_disjoint routing**: 1-NN matches LGBM at id100/seq_disjoint and outperforms at id099. cluster_disjoint weakens the near-neighbor signal gradually rather than eliminating it.
- **Single-slot HA-only sweep on HA-NA (id100..id095)**: monotone MMD↑ and monotone test perf↓; F1-vs-MMD² scatter is nearly linear. Negative-pair MMD also shifts monotonically — perf drop is joint pos+neg shift. Biological coupling confirmed: HA-cluster boundary ≈ NA-subtype boundary. See `docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md`.
- **PB2-PB1 PB2-only sweep (falsification sibling)**: same infra applied with no antigenic subtype on either slot. ~half the shift and ~half the F1 drop of HA-NA. Model ordering REVERSES on PB2/PB1: 1-NN > MLP ≈ LGBM at every threshold — strong "memorization-test" signal that residual metadata-driven leakage survives cluster_disjoint single-slot here. See `docs/results/2026-05-26_pb2_pb1_PB2only_idXX_sweep.md`.

---

## Recent run outputs

The `docs/results/` directory holds the canonical writeups. Most-cited (2026-05/06):

- `2026-05-24_single_slot_HAonly_idXX_sweep.md` — headline single-slot experiment
- `2026-05-24_cluster_disjoint_feasibility_HA_NA.md` — pre-flight that motivated single-slot
- `2026-05-26_pb2_pb1_PB2only_idXX_sweep.md` — PB2-PB1 falsification sibling
- `2026-05-24_mmd_per_{slot,pair}_results.md` — MMD baselines
- `2026-05-15_cluster_disjoint_nt_results.md` — Experiment B-nt
- `2026-06-03_phase2_postmigration_metrics.md` — pair_key migration impact

Per-sweep aggregator outputs (gitignored plots/CSVs) live under
`results/{virus}/{data_version}/runs/<analysis_name>/sweep_aggregate/<pair_direction>/`.
The aggregator is `python -m src.analysis.aggregate_mmd_single_slot_sweep`
(auto-detects seeds, label_filters, feature spaces; flags parametrize pair/direction).

Wrapper scripts for rebuilding sweeps:
- `scripts/stage4_sweep.sh` — Stage 4 across multiple datasets × seeds in parallel
- `scripts/mmd_sweep.sh` — S1+S2 MMD sweep for one (feature_space × label_filter)
- `scripts/pb2_pb1_phase4_5_launch.sh` — one-shot PB2-PB1 launcher (template for new directions)

---

## Roadmap (from 02/10/2026 meeting)

Priority experiments for publication:
1. **Cross-validation** (N splits, mean ± std metrics) — `dataset.n_folds`. Canonical reference: `docs/methods/splits.md` § 2. Remaining work: `docs/plans/2026-05-28_kfold_remaining.md`.
2. **Large dataset** (full Flu A, ~100K isolates) — HPC required (Polaris)
3. **Temporal holdout** (train 2021–2023, test 2024) — use `dataset.metadata_holdout` (year-axis is the degenerate case). Legacy `year_train`/`year_test` keys retired 2026-05-11.
4. **Genome features** (k-mers + XGBoost/LightGBM, then GenSLM)
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

- `src/preprocess/preprocess_bunya_protein.py` — Bunya preprocessing
- `conf/bundles/bunya_base.yaml` — Bunya experiment config

## What Is In Development

- `src/utils/dna_utils.py` — DNA QC utilities
- `dataset.split_strategy.single_slot`: exercised on Flu A HA-NA HA-only and PB2-PB1 PB2-only. Untested: NA-only / PB1-only directions, nt cluster_alphabet on single_slot, CV variance.

K-mer aa scaling limits (exhaustive `|alphabet|^k` vocab; no observed-vocab or hashing yet):
- **nt** practical to k≈10 (4^10 ≈ 1M cols). Production: k=6.
- **aa** practical ceiling k=4 (20^4 = 160K cols). Current bundles use k=3.
- **aa k≥5** OOMs — would need observed-vocab or feature hashing. See `docs/methods/kmer_features.md`.

---

## Aggregator Output Convention

Machine-generated aggregator outputs (`baselines_vs_mlp_*.{png,csv}` and similar) live
under `results/{virus}/{data_version}/runs/`, mirroring the layout of `data/datasets/`
and `models/`. The `docs/` tree is reserved for hand-authored writeups, plans, and methods notes.

---

## Directory Layout

```
viral-segmatch/
├── CLAUDE.md                   # This file (auto-loaded by Claude Code)
├── .claude/
│   ├── settings.json           # Claude Code permissions
│   └── memory.md               # Compact project memory — read every session
├── README.md
├── roadmap_v{1,2}.md           # Experiment plan (v2 updated 03/12/2026)
├── paper_outline_v{1,2}.md     # Paper outline (v2 updated 03/12/2026)
├── _ongoing_work.md            # Technical notes on interactions, findings
├── _notes.txt                  # Ad-hoc questions and TODOs
├── src/                        # Python source code
├── scripts/                    # Shell pipeline wrappers (stage1–4)
├── conf/                       # Hydra configs
│   └── bundles/                # One YAML per named experiment
├── eda/                        # Exploratory analysis (not pipeline)
├── examples/                   # HuggingFace reference scripts
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Technical docs (plans, methods, results)
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

- **Experiment naming**: `{virus}_{proteins}_{n_isolates}[_{modifiers}]`.
- **Timestamps**: All run directories include `YYYYMMDD_HHMMSS`.
- **Shared vs. run-specific**: Preprocessing and embeddings are shared per `{virus}/{data_version}`. Datasets and models are per run in `runs/` subdirectories.
- **Seed system**: Hierarchical — `master_seed` derives all process seeds. See `docs/SEED_SYSTEM.md`.
- **Metrics**: `metrics.csv` carries F1 (binary + macro), precision, recall, AUC-ROC, AUC-PR, MCC, Brier, BCE loss. Early-stop options: `loss`, `f1`, `auc_roc`, `auc_pr`, `mcc`. Naming: snake_case identifiers are `auc_roc` / `auc_pr` (dict keys, CSV columns, variables, config values); display strings are `AUC-ROC` / `AUC-PR`. Sklearn names `roc_auc_score` / `average_precision_score` are external and left alone. Train targets neg:pos = `neg_to_pos_ratio` (default 1.0); val/test drift to ~1.07–1.20× neg-heavy because v2's coverage phase overshoots.
- **Proteins**: `preprocess_flu.py` maps GTO replicon functions to standard protein names (PB2, PB1, PA, HA, NP, NA, M1, M2, NEP).
- **Threshold notation**: `tXXX` (zero-padded, e.g., `t095`) denotes the mmseqs identity threshold at `0.XXX`. Canonical across docs, plot labels, code, bundle YAML filenames, and `cluster_id_path` refs. **Asymmetry (Phase 2)**: on-disk cluster parquets now live at `clusters_*/tXXX/` (pre-Phase-2 `idXXX` + easy-cluster artifacts archived under `clusters_*_archive_*`); existing dataset and training run dirs retain their pre-Phase-2 `idXXX` names.
- **Sequence hashes**: `seq_hash = md5(prot_seq)` (protein), `dna_hash = md5(dna_seq)` (nucleotide). In pair tables: `seq_hash_a`/`seq_hash_b`, `dna_hash_a`/`dna_hash_b`. `seq_hash` written by Stage 1; `dna_hash` added by `attach_dna_to_prot_df` in Stage 3. ESM-2 cache key uses `sha1(prot_seq)` — separate namespace, never joined back to `seq_hash`.
- **Log messages**: No emojis. Use text prefixes: `ERROR:` (fatal), `WARNING:` (non-fatal), `Done.` (success).
- **Leakage terminology**: use canonical names from `docs/plans/2026-05-07_leakage_diagnostics_plan.md`: same-pair leakage, sequence-level label imbalance, sequence-level leakage, cluster leakage, demographic shortcut leakage. New modes should be added to that table first.
- **aa/nt vs protein/DNA**: `aa` and `nt` at the **alphabet/residue level** (kmer alphabet, "1 aa change", "97% nt identity"). `protein` and `DNA` at the **molecule/sequence level** (protein sequence, DNA contig, protein hash). The pairing is aa↔nt and protein↔DNA — don't cross the streams.
- **Reading CSVs with `function_short`**: any CSV containing a `function_short` column has the literal string `'NA'` (Neuraminidase) as a value. Default `pd.read_csv()` parses `'NA'` as NaN and **silently drops Neuraminidase rows**. Always read with `keep_default_na=False, na_values=['']`. Source pipeline CSVs use full function names (safe); derived CSVs using `function_short` are vulnerable (`redundancy_stats.csv`, `mutations_tolerated_table.csv`, `sequence_length_summary.csv`, `cluster_disjoint_feasibility_*.csv`).
- **Bash tool calls**: prefer single-command invocations over compound chains (`&&`, `;`, `$(...)`, `bash -c '...'`) — the allow-list matcher only auto-approves statically-parseable commands. Use compound only when atomicity matters (`git add X && git commit ...`) or it's fundamentally one shell idiom.
- **Documentation language**: prefer plain technical verbs (`removes`, `drops`, `reads`, `writes`, `joins`) over decorative alternatives (`scrubs`, `massages`, `munges`, `slurps`). Use the same word for the same thing throughout the repo.
- **Terminology**: `docs/methods/glossary.md` is the canonical glossary (graph-theory + project-specific terms). Use its exact terms in code, docs, and analysis; when introducing a new term, add it there first (same discipline as **Leakage terminology** above). Don't coin a synonym for a term that already has a glossary entry.
- **Docs describe current state, not history**. Method and reference docs (`docs/methods/`, `CLAUDE.md`, `.claude/memory.md`) read as a stable description of how things are now. Don't write "the new X column shows Y" or "since 2026-05-26 the sweep covers …". Historical framing belongs in `docs/results/` or `docs/plans/`.
- **Claims match verified evidence — no more, no less**. Don't under-claim or over-claim; state the scope of verification ("checked PB2 and PB1; not confirmed on the other 6") rather than rounding to a universal claim.
- **Verify before asserting; flag the unverified**. Claims about what *exists* (files, functions, flags, bundles), *current values*, or *what code does* must be checked against the source (Read / Grep / run) in the same turn before stating them — never from memory or inference. For anything not checked (design reasoning, recall, prediction), say so inline ("unverified", "likely", "would need to check X"). A bare factual claim with no evidence and no hedge is a bug — this is the behavioral trigger behind **Claims match verified evidence** above.
- **Concrete numbers in takeaways when the magnitude IS the point**. "PB2 id093 → id092: 1,085 → 112 (−90%)" beats "PB2 drops sharply". Reserve qualitative descriptors for cases where SHAPE matters more than magnitude.
- **Design symmetry: check before proposing**. Before naming a field, data structure, or API surface, list the dimensions it covers (slot a/b, routing modes, alphabets, splits) and verify the proposal is uniform across each. Names that fit the current example better than the alternatives bake in assumptions and have to be redesigned.
- **Commits are explicit-only**. Never run `git commit` / `git commit --amend` on Claude's own initiative. Commit only on an explicit user instruction — either for a specific change ("commit this") or a standing authorization scoping a batch/session ("as you go, commit each fix you finish"). Otherwise: stage, show the diff, draft the message, and stop. `git commit` is allow-listed (no prompt) so authorized commits don't stall unattended runs — which makes this rule the sole guard against unsolicited commits, so apply it strictly.
