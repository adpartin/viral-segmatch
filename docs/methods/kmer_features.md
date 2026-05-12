# K-mer feature pipeline

> Companion figure: `docs/figures/kmer_method_overview.png` (+ `.pdf`).
> Script to regenerate: `src/analysis/plot_kmer_method.py`.
> This doc is draft methods text for `paper_outline_v2.md` §2.x.
>
> See also:
> - `docs/methods/gto_format_reference.md` — schema-level reference for the
>   GTO input (the `contigs[]` array this pipeline reads). Covers
>   what's annotated vs implicit (UTRs, introns) and the per-segment
>   non-coding fraction (median ≤ 3.01% across all 8 flu segments in
>   this corpus) — relevant to interpreting k-mer feature attributions.
> - `docs/plans/2026-05-12_codon_aware_kmer_features_plan.md` —
>   interpretability note on whether current k-mer attributions can be
>   read as codon-level signal (short answer: no; current stride-1
>   features mix all three frames + UTRs + introns).

## Overview

For each genomic segment (one DNA contig per influenza segment), we compute
a **bag-of-k-mers** frequency representation. Each segment becomes a fixed
4,096-dimensional vector whose i-th entry is the number of times the i-th
lexicographically-ordered 6-mer occurs in the segment's DNA. Stacked across
all segments in the dataset, this yields a sparse matrix of shape
(`N_segments`, `4^k`) which is the feature cache for the pair classifier.

Pair features for classification are formed by looking up the two segments'
k-mer vectors and applying a fixed interaction (concat, diff, unit_diff, or
element-wise product) before passing to an MLP.

The production configuration is **k = 6**, **no normalization** (raw integer
counts), **all overlapping windows** (stride = 1), **ACGT-only alphabet**
(skip any window containing an ambiguous base).

`sequences_to_sparse_kmer_matrix` also accepts an `alphabet` parameter
(added 2026-05-12), so the same machinery can compute protein k-mers
over the canonical 20-AA alphabet `'ACDEFGHIKLMNPQRSTVWY'`. This is in
production code but not currently used in any active modeling bundle.
Practical ceiling is k≈4 (160K cols) before the exhaustive vocabulary
becomes impractical.

## Data flow

| Stage | Script | Input | Output | Level |
|---|---|---|---|---|
| 1 | `src/preprocess/preprocess_flu.py` | GTO JSON files (one per isolate) | `genome_final.csv` (one row per contig) | contigs extracted from `gto.contigs[]` |
| 2b | `src/embeddings/compute_kmer_features.py` | `genome_final.csv` (column `dna_seq`) | `kmer_features_k6.npz` (sparse CSR), `kmer_features_k6_index.parquet` (row→key), `kmer_features_k6_metadata.json` | one vector per contig |
| 3 | `src/datasets/dataset_segment_pairs.py` | `protein_final.csv`, `genome_final.csv` | pair CSVs carrying `(assembly_id_{a,b}, ctg_{a,b}, label, …)` | pair rows |
| 4 (train) | `src/models/train_pair_classifier.py` + `src/utils/kmer_utils.py::get_kmer_pair_features` | pair CSVs + k-mer npz + index | MLP predictions | pair features |

Storage for the production Flu-A July 2025 dataset
(`data/embeddings/flu/July_2025/kmer_features_k6_*`, verified against
the on-disk artifact 2026-05-12):

| Quantity | Value |
|---|---:|
| Segments (matrix rows) | 868,240 |
| Vocabulary (matrix columns, 4^6) | 4,096 |
| Non-zero entries | 1,049,945,579 |
| Sparsity (fraction of zero cells) | 70.48% |
| Avg distinct 6-mers per segment | 1,209.3 |
| NPZ file size on disk | 1.78 GB |
| Index parquet size | 9.8 MB |

## GTO → contigs

A Genome Typing Object (GTO) is a JSON document with two relevant arrays:
`contigs[]` (one entry per genomic segment, with keys `id`, `replicon_type`,
`dna`, `length`) and `features[]` (one entry per CDS, with
`protein_translation`). For the k-mer pipeline only `contigs[]` is used.
Stage 1 emits one row per contig to `genome_final.csv`, preserving
`(assembly_id, genbank_ctg_id, canonical_segment, dna_seq)` plus summary
statistics (length, GC content, ambiguous-base counts).

## K-mer counting

For a DNA string `s` and window size `k`, the count vector is built by
sliding a width-`k` window with step 1 across `s`. A window is recorded only
if every character is in `{A, C, G, T}` (uppercase, after `.upper()`).
Windows containing any ambiguous base (N, R, Y, W, S, K, M, B, D, H, V,
gaps) are silently skipped. No reverse-complement canonicalization, no
strand awareness.

```python
def compute_kmer_counts(seq: str, k: int, alphabet: str = 'ACGT') -> Counter:
    counts = Counter()
    seq_upper = seq.upper()
    valid = set(alphabet)
    for i in range(len(seq_upper) - k + 1):
        kmer = seq_upper[i:i + k]
        if all(c in valid for c in kmer):
            counts[kmer] += 1
    return counts
```

(Current signature in `src/embeddings/compute_kmer_features.py:56`.
Default `alphabet='ACGT'` matches the production DNA path; pass
`'ACDEFGHIKLMNPQRSTVWY'` for protein k-mers.)

The output vector has dimension `4^k` (k=6 → 4,096). Columns are ordered
lexicographically: index 0 is `AAAAAA`, index 1 is `AAAAAC`, …, index 4,095
is `TTTTTT`.

Counts are stored **unnormalized** in the production cache. Length
normalization (`l1` → frequencies) and magnitude normalization (`l2` → unit
vectors) are supported config options (`kmer.normalize`) but have not been
used in paper experiments to date; the MLP + `slot_norm` combination
appears to handle raw counts well (see `roadmap_v2.md` Runtime Analysis).

## Sparse storage

The per-segment vectors stack into a single `scipy.sparse.csr_matrix` saved
as `kmer_features_k6.npz`. A sidecar parquet file
(`kmer_features_k6_index.parquet`) maps `(assembly_id, genbank_ctg_id,
canonical_segment)` tuples to row indices. CSR format gives O(1) dense-row
extraction for batch k-mer lookup during training. On-disk footprint on
the full Flu-A July 2025 dataset is **1.78 GB** for the NPZ plus 9.8 MB
for the index parquet (verified 2026-05-12). Cells are `float32` (not
counts dtype) because (a) common k-mers in long sequences can exceed
int8's 127 cap, (b) normalization paths require fractional values
anyway, and (c) downstream consumers (sklearn, LightGBM) expect float
matrices — see the docstring of
`sequences_to_sparse_kmer_matrix` for the full rationale.

## Pair feature construction

For a protein pair (`protein_a`, `protein_b`) generated by Stage 3, the
segment identities are `(assembly_id_a, ctg_a)` and `(assembly_id_b, ctg_b)`
respectively. At train time each pair row is converted to:

1. row_a = index_lookup(assembly_id_a, ctg_a)
2. row_b = index_lookup(assembly_id_b, ctg_b)
3. v_a = kmer_matrix[row_a].todense()  (4,096-dim)
4. v_b = kmer_matrix[row_b].todense()  (4,096-dim)
5. pair_feature = f(v_a, v_b)

Supported interactions `f(·, ·)` (from
`src/utils/kmer_utils.py::get_kmer_pair_features` and the mirroring
MLP path `src/models/train_pair_classifier.py::_compute_interaction`):

| Token | Formula | Output dim |
|---|---|---:|
| `concat` | `[v_a ; v_b]` | 8,192 |
| `diff` | `|v_a − v_b|` (element-wise abs) | 4,096 |
| `unit_diff` | `|v_a − v_b| / ‖ |v_a − v_b| ‖_2` (abs first, then L2-normalize — symmetric in `a`, `b`) | 4,096 |
| `prod` | `v_a ⊙ v_b` (element-wise) | 4,096 |
| `unit_prod` | `(v_a ⊙ v_b) / ‖v_a ⊙ v_b‖_2` | 4,096 |

Tokens compose with `+`, e.g. `unit_diff + prod` produces an
`[unit_diff(v_a,v_b) ; prod(v_a,v_b)]` 8,192-dim vector. Note: as of
2026-05-12, `unit_diff` is **abs-based** (element-wise abs first, then
L2-normalize). The previous signed-`unit_diff` is no longer in the code.

A per-slot transform can be applied to `v_a` and `v_b` before the
interaction; `get_kmer_pair_features` accepts `slot_transform ∈
{'none', 'unit_norm'}`. `'unit_norm'` is a parameter-free L2 row-norm
that mirrors the MLP's `slot_transform: 'unit_norm'` (added
2026-05-12); LayerNorm-style `slot_norm` is **not** offered for k-mer
features because non-negative count vectors don't benefit from it (a
ValueError is raised — `get_kmer_pair_features` lines 113–119).

The original 28-pair sweep (Task 11, `paper_outline_v2.md` §11) used
`concat` + MLP `slot_norm`. The HA/NA and PB2/PB1 deep-dive bundles
(Tests 1–4, 2026-05-12) use `unit_norm + (unit_diff + prod)` — Test 3
of that sweep narrowly leads on aggregate metrics, but all four tests
lie within ~0.5% F1 / ~0.1% AUC-ROC on a single-seed run (seed-noise
level). The 28-pair headline numbers below were computed under the
older configuration; current production for HA/NA and PB2/PB1 uses Test 3.

## Anticipated reviewer questions

These are likely questions with short answers. Some are already covered by
roadmap/audit docs; cross-references given.

### Why k-mers at all?

Reference-free, alignment-free, CPU-cheap, and interpretable (each
feature is a specific 6-mer count). For influenza A, k-mer MLP matches
or exceeds ESM-2 on the full 28-pair sweep — median AUC 0.994 vs ESM-2
0.976 (see `roadmap_v2.md` §11 for the per-pair table).

### Why k = 6?

k = 6 is the smallest k at which the 4^k = 4,096 vocabulary is richer
than the longest segment (~2,300 bp → ~2,295 overlapping 6-mers), so an
individual 6-mer count is informative rather than saturated. We did not
tune k exhaustively.

### Why not normalize counts?

Raw counts encode both composition *and* length. Since segment lengths
differ substantially across canonical segments (corpus medians:
S1 = 2,296 nt, S2 = 2,301, S3 = 2,189, S4 = 1,715, S5 = 1,524, S6 = 1,420,
S7 = 993, S8 = 860 — see `docs/methods/gto_format_reference.md` §6.5), raw
counts carry some length information as a confound. Two responses:

1. A per-slot transform is applied to `v_a` and `v_b` before the
   interaction. Two options are supported and produce nearly
   identical results on Tests 1–4 (2026-05-12) for HA/NA — within
   ~0.5% F1 / ~0.1% AUC-ROC on a single seed:
   - **`slot_norm`** (MLP-only): LayerNorm per slot. Critical for
     ESM-2; useful for k-mer on homogeneous subsets.
   - **`unit_norm`** (parameter-free, supported in both MLP and
     `get_kmer_pair_features`): L2 row-norm per slot before the
     interaction. Equivalent in effect to applying `kmer.normalize=l2`
     at the feature level but applied at training time, so the cached
     NPZ remains raw counts.
2. We have not run a head-to-head comparison of `l1`-normalized k-mers
   (a feature-level fix, `kmer.normalize=l1`) vs raw + `slot_norm` /
   `unit_norm` on the full 28-pair sweep. Possible future ablation.

For paper text: counts are stored raw; per-slot normalization happens
at training time; flag the length-confound as a known property and an
ablation candidate.

### What about ambiguous bases (N, R, Y, …)?

All windows containing any non-`ACGT` character are **skipped**, not
imputed. For the full Flu-A July 2025 dataset this removes a negligible
fraction of windows (flu reference genomes are mostly unambiguous; the
`ambig_frac` column in `genome_final.csv` captures the per-segment rate
and is near 0 for well-assembled sequences). A segment with pathological
ambiguity would end up with a partial k-mer profile; the existing Stage 1
QC (drops `genome_missing_seqs.csv`, `genome_poor_quality.csv`) filters
those out before Stage 2b runs.

### What about reverse complement?

No. The k-mer counter operates on the DNA string as stored in the GTO,
which is the sense strand for each contig. We do not sum counts of each
k-mer with its reverse complement. For segment co-occurrence this matches
the input encoding the protein-based ESM-2 pipeline also uses (each
segment has one canonical sequence per GTO). Reverse-complement
canonicalization would halve the effective vocabulary (to ~2,080 unique
canonical 6-mers for odd-k, or similar) and could be tried as an
ablation.

### Are k-mers position-aware?

No — the representation is a multiset. Two sequences with the same 6-mer
composition but different arrangements will produce identical feature
vectors. For this task that's acceptable: flu segments are compact enough
that compositional difference between segments (e.g., PB2 vs HA) is
already strong, and compositional difference between isolates of the
same segment is a direct proxy for evolutionary divergence.

### How does the k-mer vector link to a protein pair?

Stage 3 produces pair rows with `(assembly_id_a, ctg_a)` and
`(assembly_id_b, ctg_b)` where `ctg_x` is `genbank_ctg_id` — the same key
used to index the k-mer matrix. The training dataloader
(`KmerPairDataset`) builds a composite key
`f"{assembly_id}::{genbank_ctg_id}"` and looks up both sides of each pair
via the parquet index.

### Does this introduce leakage?

Two sub-questions:

1. *Can the same k-mer vector end up in train and test?* Only if two
   rows share `(assembly_id, genbank_ctg_id)`. Under current default
   (`split_strategy.mode: seq_disjoint`, `hash_key: seq`, in production
   for `flu_ha_na` and `flu_pb2_pb1`), the split routing partitions
   positives by the chosen hash family before negatives are added, so
   cross-split overlap on `seq_hash` is **0 by construction** (audit
   field `seq_hash_overlap_full_pairs_*` in `split_strategy_audit.json`).
   `dna_hash` overlap is also typically 0 under `hash_key=seq` (it's
   computed as a diagnostic — see `dataset_segment_pairs_v2.py:607`).
2. *Can semantically-identical k-mer vectors end up in different splits
   via different `genbank_ctg_id` values?* Under `hash_key=seq`, two
   isolates that share a protein sequence (same `seq_hash`) are routed
   to the same split, so this is precluded at the protein level.
   Synonymous-codon variants of the same protein could still diverge at
   the DNA level — for stricter DNA-level disjointness, set
   `hash_key=dna`. The full leakage taxonomy is in
   `docs/methods/leakage_definitions.md` (mode #3 — sequence-level
   leakage). Audit script: `src/analysis/audit_split_leakage.py`.

### Why does the matrix have 868,240 rows?

108,530 isolates × 8 segments = **868,240** exactly. Every assembly in
the Flu-A July 2025 corpus has all 8 segments deposited (verified:
108,530 / 108,530 = 100.00% have 8 contigs in `genome_final.parquet`),
and Stage 1 QC drops no rows on the genome side for this corpus. The
matrix is keyed by segment, not by protein — multiple proteins can
share one segment (M1/M2 on S7; NS1/NEP on S8; PA/PA-X on S3; and
PB1/PB1-F2/PB1-N40 on S2; PB2/PB2-S1 on S1). Those proteins receive
**identical** k-mer vectors when looked up, which is semantically
correct (they share contig DNA) and is why the 28-pair Task 11 bundles
select exactly one function per segment.


## Figure legend (draft for paper)

> **Figure X. K-mer feature pipeline.**
> **(A)** Each input is a Genome Typing Object (GTO) — one JSON file per
> influenza-A isolate, with a `contigs[]` array (one entry per genomic
> segment) and a `features[]` array (one entry per CDS). Stage 1 extracts
> both tracks. **(B)** The genome-side output, `genome_final.csv`, has one
> row per `(assembly_id, segment)` pair with the raw DNA sequence and
> length. Approximate segment lengths shown. **(C)** Stage 2b counts
> overlapping 6-mers in every DNA sequence with a unit-stride window.
> Windows containing any non-`ACGT` base are skipped. **(D)** Per-segment
> count vectors are stacked into a sparse matrix (868,240 segments × 4,096
> 6-mers; 70.5% zero; avg 1,209 distinct 6-mers per segment). The matrix
> is keyed by `(assembly_id, genbank_ctg_id)`. **(E)** For each protein
> pair generated by Stage 3, the two relevant k-mer rows are looked up and
> combined by a fixed interaction (concat, |diff|, unit_diff, or
> elementwise product) before the MLP classifier predicts whether the two
> segments co-occur in the same isolate. Multiple proteins on the same
> contig (M1/M2, NS1/NEP, PA/PA-X) share a k-mer vector; the paper's
> 28-pair sweeps select one function per segment to avoid this case.
