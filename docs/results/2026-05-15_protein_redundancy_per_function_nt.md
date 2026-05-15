# Per-function redundancy (nt (CDS DNA)) — mmseqs2 sweep

**Date.** 2026-05-15.
**Input.** `data/processed/flu/July_2025/cds_final.parquet`.
**Alphabet.** nt (CDS DNA).
**Tool.** mmseqs2 `easy-linclust --min-seq-id <th> -c 0.8 --cov-mode 0 --dbtype 2`.
**Script.** `src/analysis/protein_redundancy_per_function.py`.

## Method

For each major protein function, dedup `cds_dna` on `cds_dna_hash` (md5 of the CDS DNA), export to FASTA, and cluster at multiple nt-identity thresholds with mmseqs2 `easy-linclust --dbtype 2`. IUPAC ambiguity codes (N, R, Y, ...) are left in place — mmseqs scores them natively. CDS is reconstructed by `src/preprocess/extract_cds_dna.py` from Stage 1 outputs (validated via translate-back). linclust (linear-time, less sensitive) was chosen over the sensitive easy-cluster path because easy-cluster's prefilter is an order of magnitude slower on the longer nt sequences while producing within-noise different cluster counts on this corpus.

## Results — cluster-size distribution per (function, threshold)

### threshold = 0.80

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 85 | 21,978 | 15,971 | 1,147 | 10 | 0.247 |
| PB1 | 67,034 | 10 | 66,971 | 60,948 | 6,744 | 1 | 0.600 |
| PB2 | 67,341 | 5 | 67,335 | 64,641 | 40,402 | 1 | 0.600 |
| nan | 58,887 | 91 | 18,613 | 15,687 | 938 | 3 | 0.319 |

### threshold = 0.85

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 141 | 20,553 | 11,709 | 604 | 8 | 0.184 |
| PB1 | 67,034 | 18 | 40,973 | 36,794 | 10,575 | 11 | 0.222 |
| PB2 | 67,341 | 24 | 33,693 | 30,568 | 9,060 | 7 | 0.208 |
| nan | 58,887 | 134 | 20,016 | 11,146 | 725 | 5 | 0.201 |

### threshold = 0.90

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 275 | 18,779 | 2,994 | 271 | 7 | 0.156 |
| PB1 | 67,034 | 121 | 21,853 | 17,192 | 197 | 3 | 0.355 |
| PB2 | 67,341 | 180 | 21,818 | 12,119 | 102 | 2 | 0.378 |
| nan | 58,887 | 250 | 13,077 | 5,226 | 314 | 7 | 0.144 |

### threshold = 0.95

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 1,277 | 14,068 | 495 | 44 | 3 | 0.280 |
| PB1 | 67,034 | 742 | 18,205 | 1,638 | 40 | 2 | 0.336 |
| PB2 | 67,341 | 954 | 19,028 | 687 | 37 | 2 | 0.369 |
| nan | 58,887 | 1,108 | 15,020 | 545 | 42 | 3 | 0.277 |

### threshold = 0.99

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 12,150 | 4,815 | 38 | 5 | 1 | 0.668 |
| PB1 | 67,034 | 14,990 | 6,122 | 29 | 3 | 1 | 0.784 |
| PB2 | 67,341 | 11,484 | 6,593 | 40 | 4 | 1 | 0.727 |
| nan | 58,887 | 12,092 | 5,636 | 33 | 5 | 1 | 0.690 |

### threshold = 1.00

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 64,526 | 16 | 2 | 1 | 1 | 0.989 |
| PB1 | 67,034 | 66,138 | 16 | 2 | 1 | 1 | 0.989 |
| PB2 | 67,341 | 66,475 | 15 | 2 | 1 | 1 | 0.990 |
| nan | 58,887 | 57,987 | 55 | 2 | 1 | 1 | 0.988 |

## Reading the table

- `n_sequences`: unique protein sequences input to clustering (constant across thresholds for a given function).
- `n_clusters`: clusters produced at this threshold. Smaller = more aggressive collapse.
- `largest_cluster`: dominant cluster size. If this exceeds the max per-split capacity (10% of n_pairs at 80/10/10), the routing is forced.
- `fraction_singletons`: clusters of size 1 / total clusters. Higher = more sequences with no near-neighbor.

## Related

- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` — parent plan (Experiment B).
- `docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md` — the diagnostic that motivated this sweep.
