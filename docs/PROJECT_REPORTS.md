# Project Reports

This document contains periodic project summaries and updates.

---

## Report: December 2025

We developed and tested a segment-matching pipeline for segmented viruses using ESM-2, a protein language model (pLM), with influenza A and Bunyavirales as test cases. The pipeline takes curated GTO files, extracts protein sequences, computes ESM-2 embeddings, constructs segment-pair datasets with strict isolate-level splits and leakage controls, and trains an MLP classifier that sits on top of a frozen ESM-2 to predict whether two proteins originate from the same viral isolate. Extensive leakage mitigation (isolate-level splitting, symmetric pair de-duplication, blocking contradictory negatives, and ensuring non-overlapping pairs across train/validation/test) and sanity checks (initialization-loss and label-shuffling experiments) confirm that the models are learning a stable underlying signal of isolate origin.

We quantified how well a frozen pLM can support isolate-level segment matching across different conservation regimes. For Bunyavirales "core" proteins, the classifier achieves strong performance (F1 ≈ 0.91). For influenza A, we observe a clear relationship between segment conservation and performance: variable segments HA-NA reach F1≈0.92 (AUC≈0.95); mixed PB2-HA-NA dataset yield intermediate performance (F1≈0.86); and highly conserved polymerase segments PB2-PB1-PA saturate at F1≈0.75 (AUC≈0.75), with the model essentially converging within the first training epoch. A deliberate attempt to overfit the model with PB2-PB1-PA, using a tiny train set, reaches the same performance ceiling rather than collapsing on the held-out test set, indicating that there is limited but real signal in conserved proteins, with the same performance ceiling of F1≈0.75.

These results provide a quantified assessment of pLMs in the context of isolate resolution signal, as well as highlight the impact of segment conservation and potential limitations on downstream tasks. In the next phase, we plan to: (a) further harden the dataset construction by stratifying splits by host organisms, collection date, and HA/NA subtype; (b) test richer pairwise feature interactions (|embₐ − emb_b|, embₐ ⊙ emb_b); (c) explore alternative architectures and learning regimes such as contrastive learning with LoRA fine-tuning, and (d) genome-level models such as GenSLM-ESM.

### Performance Summary

| Virus / Setup | Segment Type | Test F1 | Test AUC |
|---------------|--------------|---------|----------|
| Bunyavirales (core proteins) | Mixed core segments | 0.91 | 0.93 |
| Flu A: HA–NA | Variable | 0.92 | 0.95 |
| Flu A: PB2–HA–NA | Mixed | 0.86 | 0.92 |
| Flu A: PB2–PB1–PA | Conserved (polymerase) | 0.75 | 0.75 |

---

## Report: March 2026

Building on the Dec 2025 conservation hypothesis results, we investigated interaction features, normalization strategies, and an alternative feature source (k-mer genome features). The pipeline now supports Gen3 bundles (`flu_schema_raw_slot_norm_*`) with `unit_diff` and `concat` interactions, per-slot LayerNorm (`slot_norm`), and k-mer (k=6, 4096-dim) genome features as an alternative to ESM-2 protein embeddings.

**Key findings:**

1. **Interaction matters for ESM-2 on homogeneous data.** When filtering to a single subtype (H3N2-only), `concat` collapses to chance (AUC 0.498) while `unit_diff` succeeds (AUC 0.957). The `concat` interaction exploits a magnitude/ordering shortcut that is available in mixed-subtype data (where HA and NA from different subtypes occupy distinct embedding subspaces) but uninformative when all isolates share the same subtype. `unit_diff` (L2-normalized signed difference) strips magnitude and retains only the direction of the embedding difference, which carries genuine biological signal.

2. **LayerNorm (`slot_norm`) is critical for ESM-2 on homogeneous subsets.** Without per-slot normalization, raw HA and NA embeddings live in slightly different subspaces; `unit_diff` then picks up the systematic slot offset rather than within-subtype biological signal.

3. **K-mer (k=6) features match or exceed ESM-2 and do not suffer the concat collapse.** On mixed-subtype HA-NA: k-mer AUC 0.982 vs ESM-2 AUC 0.966–0.975. On H3N2-only: k-mer AUC 0.988 (unit_diff) / 0.985 (concat) vs ESM-2 AUC 0.957 (unit_diff) / 0.498 (concat). The concat collapse is specific to ESM-2's embedding geometry — k-mer sparse frequency vectors do not have a protein-type subspace offset, making them interaction-agnostic.

4. **Pipeline improvements.** Stage 3 (dataset) and Stage 4 (training) are now decoupled: create a dataset once, train with different bundles against it. Cross-validation support (`n_folds`/`fold_id`) is implemented. Provenance tracking via `training_info.json`.

### Performance Summary (Gen3, HA-NA, 5K isolates)

| Feature Source | Interaction | Filter | Test F1 | Test AUC |
|---|---|---|---|---|
| ESM-2 | unit_diff | mixed-subtype | 0.917 | 0.966 |
| ESM-2 | concat | mixed-subtype | 0.929 | 0.975 |
| ESM-2 | unit_diff | H3N2-only | 0.878 | 0.957 |
| ESM-2 | concat | H3N2-only | 0.570 | 0.498 |
| K-mer (k=6) | unit_diff | mixed-subtype | 0.957 | 0.982 |
| K-mer (k=6) | concat | mixed-subtype | 0.951 | 0.982 |
| K-mer (k=6) | unit_diff | H3N2-only | 0.963 | 0.988 |
| K-mer (k=6) | concat | H3N2-only | 0.958 | 0.985 |

**Next steps:** Cross-validation (N-fold with robust mean +/- std), temporal holdout (train 2021–2023, test 2024), baseline validation experiments (embedding shuffle, swap-slot) to confirm sequence-level learning, and k-mer + XGBoost/LightGBM comparison.

---

