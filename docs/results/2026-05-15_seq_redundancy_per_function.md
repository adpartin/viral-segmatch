# Per-function redundancy (aa) — mmseqs2 sweep

**Date.** 2026-05-15.
**Input.** `data/processed/flu/July_2025/protein_final.parquet`.
**Alphabet.** aa.
**Tool.** mmseqs2 `easy-cluster --min-seq-id <th> -c 0.8 --cov-mode 0`.
**Script.** `src/analysis/seq_redundancy_per_function.py`.

## Method

For each major protein function, dedup `prot_seq` on md5(`prot_seq.rstrip('*')`), export to FASTA, and cluster at multiple aa-identity thresholds with mmseqs2 `easy-cluster`. `X` residues are left in place (mmseqs handles them natively); internal `*` rows would be dropped but none exist in this corpus.

## Results — cluster-size distribution per (function, threshold)

### threshold = 0.80

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 41,896 | 23 | 12,513 | 11,984 | 3,921 | 690 | 0.043 |
| M1 | 4,771 | 2 | 4,770 | 4,722 | 4,293 | 2,385 | 0.500 |
| M2 | 8,173 | 14 | 8,137 | 7,080 | 10 | 1 | 0.571 |
| NA | 37,488 | 39 | 14,495 | 13,875 | 1,161 | 95 | 0.179 |
| NEP | 6,896 | 19 | 6,853 | 5,621 | 9 | 1 | 0.632 |
| NP | 17,684 | 2 | 17,683 | 17,506 | 15,914 | 8,842 | 0.500 |
| NS1 | 22,225 | 10 | 11,915 | 11,265 | 5,415 | 61 | 0.100 |
| PA | 34,217 | 2 | 34,216 | 33,873 | 30,794 | 17,108 | 0.500 |
| PB1 | 31,226 | 2 | 31,225 | 30,912 | 28,102 | 15,613 | 0.500 |
| PB2 | 33,663 | 2 | 33,662 | 33,325 | 30,295 | 16,831 | 0.500 |

### threshold = 0.90

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 41,896 | 110 | 10,159 | 7,484 | 753 | 25 | 0.082 |
| M1 | 4,771 | 7 | 3,630 | 3,468 | 2,017 | 60 | 0.286 |
| M2 | 8,173 | 56 | 2,962 | 2,374 | 225 | 2 | 0.393 |
| NA | 37,488 | 108 | 7,643 | 6,700 | 808 | 24 | 0.157 |
| NEP | 6,896 | 31 | 3,842 | 3,328 | 302 | 1 | 0.516 |
| NP | 17,684 | 7 | 17,643 | 16,586 | 7,075 | 2 | 0.429 |
| NS1 | 22,225 | 98 | 4,523 | 3,827 | 186 | 5 | 0.163 |
| PA | 34,217 | 3 | 34,214 | 33,529 | 27,371 | 2 | 0.333 |
| PB1 | 31,226 | 4 | 31,223 | 30,286 | 21,856 | 1 | 0.750 |
| PB2 | 33,663 | 2 | 33,662 | 33,325 | 30,295 | 16,831 | 0.500 |

### threshold = 0.95

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 41,896 | 711 | 4,695 | 1,098 | 90 | 3 | 0.370 |
| M1 | 4,771 | 26 | 2,323 | 2,088 | 380 | 1 | 0.500 |
| M2 | 8,173 | 426 | 1,459 | 229 | 21 | 2 | 0.413 |
| NA | 37,488 | 625 | 4,964 | 914 | 92 | 4 | 0.302 |
| NEP | 6,896 | 135 | 1,976 | 1,067 | 43 | 3 | 0.326 |
| NP | 17,684 | 44 | 8,270 | 7,135 | 172 | 4 | 0.341 |
| NS1 | 22,225 | 485 | 3,306 | 1,036 | 51 | 2 | 0.392 |
| PA | 34,217 | 158 | 23,718 | 4,086 | 4 | 1 | 0.759 |
| PB1 | 31,226 | 50 | 23,316 | 15,120 | 75 | 1 | 0.700 |
| PB2 | 33,663 | 26 | 27,019 | 21,790 | 233 | 1 | 0.577 |

### threshold = 0.99

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 41,896 | 11,039 | 2,089 | 35 | 4 | 1 | 0.753 |
| M1 | 4,771 | 698 | 613 | 133 | 6 | 1 | 0.658 |
| M2 | 8,173 | 7,525 | 28 | 3 | 1 | 1 | 0.954 |
| NA | 37,488 | 10,184 | 1,761 | 31 | 4 | 1 | 0.720 |
| NEP | 6,896 | 2,662 | 286 | 22 | 4 | 1 | 0.716 |
| NP | 17,684 | 1,981 | 1,568 | 111 | 10 | 1 | 0.529 |
| NS1 | 22,225 | 6,313 | 1,122 | 32 | 4 | 1 | 0.657 |
| PA | 34,217 | 10,450 | 4,697 | 8 | 1 | 1 | 0.955 |
| PB1 | 31,226 | 10,782 | 5,393 | 6 | 1 | 1 | 0.962 |
| PB2 | 33,663 | 7,935 | 4,857 | 14 | 1 | 1 | 0.942 |

### threshold = 1.00

| function_short | n_sequences | n_clusters | largest_cluster | p99_cluster_size | p90_cluster_size | median_cluster_size | fraction_singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| HA | 41,896 | 41,708 | 6 | 1 | 1 | 1 | 0.996 |
| M1 | 4,771 | 4,633 | 9 | 2 | 1 | 1 | 0.983 |
| M2 | 8,173 | 7,525 | 28 | 3 | 1 | 1 | 0.954 |
| NA | 37,488 | 37,102 | 22 | 1 | 1 | 1 | 0.992 |
| NEP | 6,896 | 6,405 | 20 | 2 | 1 | 1 | 0.954 |
| NP | 17,684 | 17,258 | 17 | 2 | 1 | 1 | 0.985 |
| NS1 | 22,225 | 21,864 | 13 | 1 | 1 | 1 | 0.991 |
| PA | 34,217 | 34,153 | 4 | 1 | 1 | 1 | 0.998 |
| PB1 | 31,226 | 30,808 | 20 | 2 | 1 | 1 | 0.988 |
| PB2 | 33,663 | 33,573 | 17 | 1 | 1 | 1 | 0.998 |

## Reading the table

- `n_sequences`: unique protein sequences input to clustering (constant across thresholds for a given function).
- `n_clusters`: clusters produced at this threshold. Smaller = more aggressive collapse.
- `largest_cluster`: dominant cluster size. If this exceeds the max per-split capacity (10% of n_pairs at 80/10/10), the routing is forced.
- `fraction_singletons`: clusters of size 1 / total clusters. Higher = more sequences with no near-neighbor.

## Related

- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` — parent plan (Experiment B).
- `docs/results/2026-05-13_aa_vs_nt_similarity_leakage.md` — the diagnostic that motivated this sweep.
