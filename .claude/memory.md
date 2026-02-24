# viral-segmatch — Project Memory

This file is version-controlled in the repo (.claude/memory.md) so it is available on every machine.
Claude: read this at the start of every session. Update it when decisions change or new findings emerge.

---

## Project Summary
Flu A viral segment co-occurrence prediction. ESM-2 protein embeddings (frozen) + MLP binary classifier.
Primary virus: Influenza A. Bunya support exists but NOT actively maintained.

## Pipeline (4 stages)
- Stage 1: `src/preprocess/preprocess_flu_protein.py` → `data/processed/flu/{version}/protein_final.csv` (run once)
- Stage 2: `src/embeddings/compute_esm2_embeddings.py` → `data/embeddings/flu/{version}/master_esm2_embeddings.h5` (run once)
- Stage 3: `src/datasets/dataset_segment_pairs.py` → `data/datasets/flu/{version}/runs/dataset_{bundle}_{ts}/` (per experiment)
- Stage 4: `src/models/train_esm2_frozen_pair_classifier.py` → `models/flu/{version}/runs/training_{bundle}_{ts}/` (per experiment)
- Shell wrappers: `scripts/stage2_esm2.sh`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh`

## Config System
Hydra + bundle-per-experiment. `conf/bundles/{bundle}.yaml` = one file per named experiment.
Bundle naming: `flu_{proteins}_{n_isolates}[_{modifiers}]`
Config loader: `src/utils/config_hydra.py` via `hydra.compose(config_name="bundles/{name}")`.
No root config -- bundles are loaded directly. `src/utils/config.py` and `conf/config.yaml` deleted (legacy).

## Bundle Organization (see conf/bundles/README.md for full detail)
- Each bundle has `# STATUS: active|ablation|experimental|legacy|not maintained` header.
- Three generations: Gen1 (flu.yaml base), Gen2 (flu_schema.yaml base), Gen3 (flu_schema_raw_* -- current)
- Base bundles must stay flat (moving them breaks Hydra defaults chains in children)
- `conf/bundles/paper/` reserved for publication experiments
- Best model: `flu_schema_raw_slot_norm_unit_diff` (slot_norm + unit_diff, HA/NA)

## Key Findings
- `unit_diff` > `concat` on homogeneous data (H3N2-only): AUC 0.96 vs 0.50
- LayerNorm (`slot_norm`) critical for homogeneous subsets
- Delayed learning on H3N2 + unit_diff: increase patience to 40+
- High FP rate on filtered datasets (year/host/geo) -- likely population-level confounders

## Roadmap (02/10/2026 meeting) -- for publication
1. Cross-validation (fold_id/n_folds in dataset config + PBS job array on Polaris) -- NEXT
2. Large dataset (full Flu A ~100K isolates, HPC)
3. Temporal holdout (year_train/year_test config fields)
4. Genome features (k-mers + LightGBM; start from preprocess_bunya_dna.py -> preprocess_flu_dna.py)
5. PB2/PB1 + H3N2 bundle (trivial, one new bundle)

## Directory Structure (post-cleanup Feb 2026)
- `.claude/` -- settings.json (permissions) + memory.md (this file)
- `eda/` -- exploratory scripts (bunya EDA moved here; NOT pipeline)
- `examples/` -- HuggingFace reference scripts (NOT pipeline)
- `old_scripts/` -- superseded scripts (NOT maintained)
- `flu_genomes_eda.py` stays in `src/preprocess/` -- generates flu_genomes_metadata_parsed.csv (pipeline input)

## Not Maintained
- `old_scripts/`, `src/preprocess/preprocess_bunya_protein.py`, `conf/bundles/bunya.yaml`

## In Development
- `src/preprocess/preprocess_bunya_dna.py` -- template for preprocess_flu_dna.py
- `src/utils/dna_utils.py` -- DNA QC utilities
- Cross-validation support (fold_id/n_folds) -- actively being designed
- Temporal holdout split logic (year_train/year_test)

## HPC
- Polaris (ALCF), PBS job arrays. Do NOT use Hydra's submitit launcher (SLURM only).
- For 8-GPU dev cluster (no scheduler): Python subprocess launcher with CUDA_VISIBLE_DEVICES per fold.
- CV design: run dataset_segment_pairs.py once to generate all N folds, then train in parallel.

## User Preferences
- Concise responses, no emojis unless asked
- No unnecessary refactoring beyond what's asked
- Always ask before destructive operations (rm, git reset --hard, git push --force, etc.)
- CLAUDE.md is the authoritative project context; .claude/memory.md is the compact working memory
- Both files are in the repo -- update them when decisions change
