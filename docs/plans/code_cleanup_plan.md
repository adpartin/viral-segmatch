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

**Files**: `scripts/stage2_esm2.sh`, `scripts/stage2b_kmer.sh`, `scripts/stage3_dataset.sh`, `scripts/stage4_train.sh`

The existing shell wrappers are verbose (100-400 lines) with duplicated boilerplate:
git provenance blocks, `log()` helpers, elaborate headers/footers, registry integration.
Most of this repeats what the Python scripts already print.

`scripts/stage1_preprocess_flu.sh` (~45 lines) demonstrates the leaner pattern:
compact arg parsing, `tee` to timestamped log, latest-log symlink, proper exit code.
The other stage scripts should be refactored to match.

---

## 4. Consider renaming `src/embeddings/` directory

**Files**: `src/embeddings/`

The directory currently holds `compute_esm2_embeddings.py` and `compute_kmer_features.py`.
K-mer frequency vectors are not embeddings in the ML sense (they are hand-crafted features,
not learned representations). A name like `src/featurize/` or `src/features/` would better
reflect that this directory contains Stage 2 scripts that convert raw sequences into numerical
feature vectors — regardless of whether the method is a pretrained model (ESM-2) or a
counting procedure (k-mers). Would also need to update imports, CLAUDE.md, and shell scripts.
