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
| M1 | 32,413 | 2 | 32,412 | 32,087 | 29,170 | 16,206 | 0.500 |
| NA | 58,887 | 91 | 18,613 | 15,687 | 938 | 3 | 0.319 |
| NP | 52,800 | 3 | 52,797 | 51,741 | 42,238 | 2 | 0.333 |
| NS1 | 38,039 | 8 | 35,578 | 33,257 | 12,370 | 8 | 0.250 |
| PA | 65,242 | 3 | 65,239 | 63,934 | 52,191 | 2 | 0.333 |
| PB1 | 67,034 | 10 | 66,971 | 60,948 | 6,744 | 1 | 0.600 |
| PB2 | 67,341 | 5 | 67,335 | 64,641 | 40,402 | 1 | 0.600 |

### threshold = 0.85

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 141 | 20,553 | 11,709 | 604 | 8 | 0.184 |
| M1 | 32,413 | 5 | 32,399 | 31,103 | 19,441 | 5 | 0.200 |
| NA | 58,887 | 134 | 20,016 | 11,146 | 725 | 5 | 0.201 |
| NP | 52,800 | 12 | 20,873 | 20,404 | 16,411 | 7 | 0.083 |
| NS1 | 38,039 | 91 | 15,197 | 11,751 | 115 | 5 | 0.220 |
| PA | 65,242 | 13 | 36,548 | 34,489 | 17,165 | 6 | 0.077 |
| PB1 | 67,034 | 18 | 40,973 | 36,794 | 10,575 | 11 | 0.222 |
| PB2 | 67,341 | 24 | 33,693 | 30,568 | 9,060 | 7 | 0.208 |

### threshold = 0.90

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 275 | 18,779 | 2,994 | 271 | 7 | 0.156 |
| M1 | 32,413 | 48 | 11,039 | 10,800 | 483 | 7 | 0.125 |
| NA | 58,887 | 250 | 13,077 | 5,226 | 314 | 7 | 0.144 |
| NP | 52,800 | 66 | 16,501 | 14,748 | 585 | 5 | 0.106 |
| NS1 | 38,039 | 196 | 10,491 | 6,076 | 120 | 4 | 0.276 |
| PA | 65,242 | 47 | 21,466 | 20,095 | 3,217 | 10 | 0.085 |
| PB1 | 67,034 | 121 | 21,853 | 17,192 | 197 | 3 | 0.355 |
| PB2 | 67,341 | 180 | 21,818 | 12,119 | 102 | 2 | 0.378 |

### threshold = 0.95

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 1,277 | 14,068 | 495 | 44 | 3 | 0.280 |
| M1 | 32,413 | 619 | 7,825 | 1,110 | 10 | 1 | 0.515 |
| NA | 58,887 | 1,108 | 15,020 | 545 | 42 | 3 | 0.277 |
| NP | 52,800 | 1,182 | 13,174 | 609 | 11 | 1 | 0.561 |
| NS1 | 38,039 | 800 | 8,476 | 585 | 30 | 2 | 0.372 |
| PA | 65,242 | 719 | 18,043 | 1,774 | 36 | 2 | 0.344 |
| PB1 | 67,034 | 742 | 18,205 | 1,638 | 40 | 2 | 0.336 |
| PB2 | 67,341 | 954 | 19,028 | 687 | 37 | 2 | 0.369 |

### threshold = 0.99

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 12,150 | 4,815 | 38 | 5 | 1 | 0.668 |
| M1 | 32,413 | 10,227 | 2,660 | 24 | 2 | 1 | 0.855 |
| NA | 58,887 | 12,092 | 5,636 | 33 | 5 | 1 | 0.690 |
| NP | 52,800 | 11,627 | 4,443 | 32 | 4 | 1 | 0.779 |
| NS1 | 38,039 | 12,012 | 2,644 | 24 | 3 | 1 | 0.807 |
| PA | 65,242 | 11,184 | 4,317 | 43 | 4 | 1 | 0.737 |
| PB1 | 67,034 | 14,990 | 6,122 | 29 | 3 | 1 | 0.784 |
| PB2 | 67,341 | 11,484 | 6,593 | 40 | 4 | 1 | 0.727 |

### threshold = 1.00

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 65,414 | 64,526 | 16 | 2 | 1 | 1 | 0.989 |
| M1 | 32,413 | 31,974 | 18 | 2 | 1 | 1 | 0.990 |
| NA | 58,887 | 57,987 | 55 | 2 | 1 | 1 | 0.988 |
| NP | 52,800 | 52,097 | 16 | 2 | 1 | 1 | 0.990 |
| NS1 | 38,039 | 37,458 | 34 | 2 | 1 | 1 | 0.990 |
| PA | 65,242 | 64,406 | 16 | 2 | 1 | 1 | 0.990 |
| PB1 | 67,034 | 66,138 | 16 | 2 | 1 | 1 | 0.989 |
| PB2 | 67,341 | 66,475 | 15 | 2 | 1 | 1 | 0.990 |

## Reading the table

- `n_sequences`: unique protein sequences input to clustering (constant across thresholds for a given function).
- `n_clusters`: clusters produced at this threshold. Smaller = more aggressive collapse.
- `largest_cluster`: dominant cluster size. If this exceeds the max per-split capacity (10% of n_pairs at 80/10/10), the routing is forced.
- `fraction_singletons`: clusters of size 1 / total clusters. Higher = more sequences with no near-neighbor.

## Related

- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` — parent plan (Experiment B).
- `docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md` — the diagnostic that motivated this sweep.
