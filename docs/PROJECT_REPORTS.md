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

--

