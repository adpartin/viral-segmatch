# Code Cleanup Plan

**Status: ONGOING**

Non-urgent code quality improvements. Items here are not blocking any experiments
but would improve readability, consistency, and maintainability.

---

## 1. Harmonize Stage 2 featurization scripts

**Files**: `src/embeddings/compute_esm2_embeddings.py`, `src/embeddings/compute_kmer_features.py`

Both are Stage 2 featurization scripts but were written at different times and have
different code organization (section separators, CLI layout, logging style, path
resolution). Aligning their structure would make them easier to read side-by-side
and debug consistently.

---

## 2. Add per-code DNA ambiguity breakdown (match protein_utils pattern)

**Files**: `src/utils/dna_utils.py`

`protein_utils.analyze_protein_ambiguities()` provides a detailed per-residue-type
breakdown (X, B, Z, *, terminal vs internal stops, positions, etc.).
`dna_utils.summarize_dna_qc()` currently lumps all IUPAC ambiguity codes (N, R, Y,
S, W, K, M, B, D, H, V) into a single `ambig_count`/`ambig_frac`. We should consider
adding a per-code breakdown so downstream consumers can distinguish "mostly Ns" (general
sequencing uncertainty) from "many two-base ambiguities" (partial base calls).

See the IUPAC reference table in the `dna_utils.py` module docstring.

---

## 3. Slim down shell script wrappers

**Files**: `scripts/stage2_esm2.sh`, `scripts/stage2b_kmer.sh`

**DONE (stage3, stage4)**: `stage3_dataset.sh` and `stage4_train.sh` were rewritten to
match the lean `stage1_preprocess_flu.sh` pattern as part of the Stage 3/4 decoupling work.

**Remaining**: `stage2_esm2.sh` still has the verbose pattern (git provenance blocks,
`log()` helpers, elaborate headers/footers, registry integration). Should be refactored
to match the lean pattern used by stage1, stage2b, stage3, and stage4.

---

## 4. Consider renaming `src/embeddings/` directory

**Files**: `src/embeddings/`

The directory currently holds `compute_esm2_embeddings.py` and `compute_kmer_features.py`.
K-mer frequency vectors are not embeddings in the ML sense (they are hand-crafted features,
not learned representations). A name like `src/featurize/` or `src/features/` would better
reflect that this directory contains Stage 2 scripts that convert raw sequences into numerical
feature vectors — regardless of whether the method is a pretrained model (ESM-2) or a
counting procedure (k-mers). Would also need to update imports, CLAUDE.md, and shell scripts.

Similarly, `conf/embeddings/` (currently just `default.yaml` for ESM-2 settings) should be
renamed to `conf/featurize/` or `conf/features/` for consistency. Do both renames together.

**Note (March 2026):** K-mer scripts (`compute_kmer_features.py`, `kmer_utils.py`) are now
in active use with tested bundles, strengthening the case for this rename.

---

## 5. Revisit Stage 4 training script naming / structure

**Files**: `src/models/train_esm2_frozen_pair_classifier.py`

The script now supports multiple feature sources (ESM-2, k-mer) via `config.training.feature_source`,
so the name `train_esm2_frozen_pair_classifier.py` is misleading. Two options to discuss:

- **Rename** to something general like `train_pair_classifier.py` (the script already handles
  both feature types through the same MLP architecture).
- **Keep as-is** for MLP-based training and create a separate script for tree-based models
  (XGBoost/LightGBM) that would be used with large-k k-mers (k=10, 1M-dim features).

Decision depends on whether we want one training entry point or separate scripts per model family.
