# Clustering overview (aa and nt) for cluster_disjoint splits

This reference covers: what protein (aa) and nucleotide (nt) sequence
clustering does in this pipeline, what mmseqs2's parameters mean, and
how a per-function clustering is used to generate train/val/test splits.

For the *experimental findings* on Flu A see
`docs/results/2026-05-15_cluster_disjoint_nt_results.md` (model training
results) and two sequence redundancy autogen docs
`data/processed/flu/July_2025/clusters_aa/redundancy_summary.md` (aa) and
`data/processed/flu/July_2025/clusters_nt/redundancy_summary.md` (nt). This
doc explains the *machinery*; those docs report the *numbers we got*.

For the routing-equivalence semantics across the four implemented modes
(seq_disjoint hash_key={seq,dna}, cluster_disjoint
cluster_alphabet={aa,nt}), see `docs/methods/leakage_definitions.md`
§ "Routing equivalence and mmseqs argument semantics".

---

## 1. Why we cluster in this pipeline

The segmatch classifier predicts whether two sequences from different proteins/segments co-occur in
the same flu isolate. A random train/val/test split can result
in highly similar sequence pairs being distributed across train,
validation, and test sets (the very similar pairs can be called
*near-neighbors*). Thus, test pairs typically have a train neighbor
that's extremely similar at the sequence level, and the model can
exploit this similarity to make accurate predictions without learning any pair-co-occurrence biology. We
define this as leakage mode #4 ("cluster leakage" / "near-neighbor
leakage") in `docs/methods/leakage_definitions.md`.

Clustering uses an alignment-identity threshold *t*. For each input
sequence X (aa or nt), pairwise alignment with every candidate sequence Y
yields a percent-identity number; Y is in X's neighborhood at
threshold t if identity(X, Y) ≥ t and the coverage rule of §3.2 is
met. Intuitively, this defines a *radius* around X — the set of
sequences alignable to it at identity ≥ t. Two sequences whose radii
overlap (i.e., are pairwise within threshold t of each other)
collapse into the same cluster id by *transitive closure*. In mmseqs, the threshold
t is set by `--min-seq-id` (formal semantics in §3.1).

**Simplified single-sequence view.** The schematic
below uses a single-sequence simplification of the splitting task to
show *why* random splitting leaks and *how* clustering by radius
mitigates it. The real `segmatch` task operates on *pairs* (two
sequences per training example, one per function slot), which adds a
second dimension — covered at the end of this section, with the full
pair-level construction in §7.

**Step 1 — random split lets near-neighbors leak.** Consider A1 and A2, two
protein sequences differing by 1 aa. A random shuffle can assign them
to different splits.

```
   ●  A1               assigned to TRAIN
   │
   │   ← 1 aa apart (near-neighbor pair in sequence space)
   │
   ●  A2               assigned to TEST
```

The model memorizes A1's neighborhood at training time and trivially
predicts A2 at evaluation — without learning anything about the sequence biology. Test sequences typically have a train-side
neighbor within a handful of aa edits, allowing for shortcut learning.

**Step 2 — clustering at radius t collapses near-neighbors to one cluster id.**
Sequences within identity threshold t of each other join the same
cluster (transitive closure on the pairwise-alignment relation; see
the operational definition above). At t = 0.99 on a 760-aa protein
the radius admits up to 7 aa mismatches (§5).

```
   ┌─────────────┐
   │   ●  A1     │
   │      │      │  cluster C_γ   ← A1 and A2 now in the same
   │   ●  A2     │                  cluster (1-aa edit ≤ radius
   └─────────────┘                  at t = 0.99)

   ┌─────────────┐
   │  ●  B1      │
   │     ●  B2   │  cluster C_α   (3 near-identical sequences)
   │  ●  B3      │
   └─────────────┘

   ┌─────┐
   │ ● C │          cluster C_β   (isolated sequence)
   └─────┘
```

**Step 3 — partition by CLUSTER, not by sequence.** Each cluster is
indivisible: every member goes to one split.

```
   C_α ───► train       (B1, B2, B3 all in train)
   C_β ───► train       (C in train)
   C_γ ───► test        (A1 and A2 both in test — the
                         near-neighbor edge is no longer
                         across the train/test boundary)
```

The cluster-disjoint rule (single-sequence version) is: no cluster id
appears on both sides of any split boundary. Near-neighbor edges that
exist *within* a cluster therefore cannot straddle splits, mitigating
mode #4 by construction at the chosen radius t.

**The radius t controls the trade-off:**

- `t = 1.0` — pairs with 100% identity over the aligned region cluster
together. Approximately equivalent to `seq_disjoint` on 7 of the 8 major
functions, but looser on NA where more sequences of different lengths
cluster via the coverage rule (check §3.2 for actual numbers).
`seq_disjoint` itself splits on seq hash of the full sequence (§7.3).
- `t = 0.99` — Up to ~1% mismatches over the aligned region cluster
together.
- `t = 0.90` — Up to 10% mismatches over the aligned region cluster
together.

A lower `t` removes more leakage but also merges more of the corpus
into mega-clusters. Once a single cluster contains more than ~80% of
all pairs it can't fit into a 0.8 of train, and 80/10/10 routing
becomes structurally impossible.

**From single sequences to pairs.** The schematic above discusses
clustering in the context of a single function (e.g., HA sequences
clustered into multiple cluster ids). The actual segmatch task includes
two dimensions: each training example is a *pair* ([HA, NA], or [PB2, PB1],...),
and the goal is to predict pair co-occurrence. Cluster-disjoint
routing on pairs has additional mechanics:

1. **Cluster per function, independently.** HA sequences get HA
   cluster ids (`HA_c1`, `HA_c2`, …); NA sequences get NA cluster
   ids (`NA_c1`, `NA_c2`, …). mmseqs never sees two functions at once.
2. **Each positive pair carries a tuple of two cluster ids.** A
   positive pair `(HA_i, NA_i)` — both proteins from isolate i —
   becomes the cluster tuple `(cluster(HA_i), cluster(NA_i))`, e.g.,
   `(HA_c1, NA_c5)`. Note that negative pairs are constructed within
   each split *after* the routing decision; the bipartite graph that
   drives routing is built from positives only.
3. **Build a bipartite graph** with HA clusters on one side, NA
   clusters on the other; an edge `HA_ck ─ NA_cm` exists iff some
   pair has that combination.
4. **Bipartite connected components** on this graph are the
   indivisible routing units. All pairs in a CC must go to the same
   split.
5. **Bin-pack CCs into 80/10/10** via LPT-greedy.

§7 walks through this paired construction with a bipartite-graph
ASCII diagram and the LPT-greedy routing details. The single-sequence
intuition above is the prerequisite for the paired version.

The clustering itself happens once per (data version, alphabet (`aa`|`nt`), identity
threshold (`id99`|`id98`|...)). Stage 3 reads the cluster lookups when
`dataset.split_strategy.mode: cluster_disjoint` is set in the bundle.

---

## 2. Sequence-space clustering 101

> Note that in §2, "pair" refers to
> *two sequences being aligned by mmseqs2* (the O(N²) alignment problem
> mmseqs2's k-mer prefilter solves), not the (HA, NA) co-occurring
> pairs from §1. mmseqs2 operates per function (or segment) on single sequences;
> the path from per-function clusters to training-pair routing is in §7.

### 2.1 What sequenece "similarity" means

mmseqs2 represents similarity of two sequences using **percent
identity**. With the default `--seq-id-mode 0` (other values: 1 = shorter
sequence, 2 = longer sequence — discussed below), identity is computed using

```
identity = n_identical / alignment_length
```

where:

- `identity` ∈ [0, 1] is the percent identity — the value compared
  against the threshold `t` which is set by `--min-seq-id`.
- `n_identical` = the number of alignment columns in which both
  sequences carry the same residue (a *match*). Gap columns are
  not matches by definition.
- `alignment_length` = the total number of columns in the local
  alignment, counting **matches**, **mismatches** (columns where both
  sequences have a residue but they differ), and **gaps** (columns
  where one sequence has a residue but the other has a placeholder
  `-`, representing an insertion in one or a deletion in the other).

`alignment_length` is *not* the length of either input sequence — it
can exceed both when one sequence has insertions relative to the
other. An `--min-seq-id` threshold of 0.95 is a *floor* — only
sequence pairs with `identity` ≥ 0.95 are admitted to the same cluster.

The match unit is **residues**: amino acids (aa) for proteins,
nucleotides (nt) for DNA. Length is counted also in residues. As a
first-order intuition, when the alignment spans both sequences
end-to-end with no gaps (typical when comparing two sequences of
similar length within a function), "id 0.95" on a 760-aa PB2 protein
admits ~38 aa mismatches within a cluster (for empirical numbers
see §5).

The flag `--seq-id-mode` is not listed in the user guide
https://mmseqs.com/latest/userguide.pdf. The canonical reference for
every parameter can be obtained with `mmseqs <subcommand> --help`, e.g.
`mmseqs easy-linclust --help`. For `--seq-id-mode` we get:

```
--seq-id-mode INT     0: alignment length 1: shorter, 2: longer sequence [0]
```

**Examples** below provide intuition for the identity formula and
its interaction with the coverage rule. In each case, "mode N" refers
to `--seq-id-mode N` (the identity denominator); the coverage mode is
set by separate `--cov-mode` discussed in §3.2.

```
Case 1: length-variants — identity passes, coverage passes (cluster together)
─────────────────────────────────────────────
Seq A:   M K T V R Q E L K                     (9 residues; length-variant)
Seq B:   M K T V R Q E L K L                   (10 residues; full)
         = = = = = = = = = ─                   (= aligned, ─ unaligned in B)

alignment_length  =  9
n_identical       =  9
identity (mode 0) = 9 / 9 = 1.00      ✓ identity passes even at t = 1.0

cov(A)  = 9 / 9  = 1.00               ✓
cov(B)  = 9 / 10 = 0.90               ✓ (≥ 0.8)

BOTH coverages ≥ 0.8 under --cov-mode 0, so this pair DOES cluster
despite the sequences being different lengths. This is the empirical
"NA stalk-deletion isoforms cluster with NA stalk-full" pattern at
t = 1.0 (see §3.1; §6.4's NA id100 row pools 6.9% of the NA corpus
into one cluster).

Case 2: fragment vs full protein — identity passes, coverage fails (does not cluster)
─────────────────────────────────────────────
Seq A:   M K T V R Q E L K L                                     (10 residues; fragment)
Seq B:   M K T V R Q E L K L Q W S P R M N K T L H A V S Q E S F (40 residues; full)
         = = = = = = = = = = ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ (= aligned, ─ unaligned in B)

alignment_length  = 10
n_identical       = 10
identity (mode 0) = 10 / 10 = 1.00      ✓ identity passes even at t = 1.0

cov(A)  = 10 / 10 = 1.00                ✓
cov(B)  = 10 / 40 = 0.25                ✗ fails -c 0.8 (under --cov-mode 0)

Under --cov-mode 0 BOTH sequences must have ≥80% of their residues
inside the aligned region, so the pair does NOT cluster despite
identity = 1.00.

Case 3: no gaps, one mismatch
─────────────────────────────────────────────
Seq A:   M K T V R Q E L K L            (10 residues)
Seq B:   M K T V R Q E L K Y            (10 residues)
status:  = = = = = = = = = X            (= match, X mismatch)

alignment_length  = 10  (columns in the alignment)
n_identical       =  9
identity (mode 0) = 9 / 10 = 0.90 ← that's what mmseqs compares to --min-seq-id

Case 4: one internal gap (different-length sequences)
─────────────────────────────────────────────
Seq A:   M K T - R Q E L K L            (residues:  9; alignment row: 10 cols)
Seq B:   M K T V R Q E L K L            (residues: 10; alignment row: 10 cols)
status:  = = = G = = = = = =            (G = gap)

alignment_length  = 10  (every column counts: matches + mismatches + gaps)
n_identical       =  9  (gap columns are not identical by definition)
identity (mode 0) = 9 / 10 = 0.90
identity (mode 1) = 9 /  9 = 1.00  ← over shorter sequence  (9 residues)
identity (mode 2) = 9 / 10 = 0.90  ← over longer sequence  (10 residues)

We pin --seq-id-mode to 0 — see §3.2 for the coverage discussion that
interacts with this choice.
```

Cases 1 and 2 together demonstrate §3.1's "modulo coverage" caveat:
the coverage gate stops fragment-vs-full matches (Case 2) from
clustering at t = 1.0, but still permits length-variants (Case 1)
where ≤20% of each sequence is unaligned (e.g., NA stalk-deletion
isoforms aligning to NA stalk-full proteins — see §4). See §3.2 for
the full coverage discussion and `--cov-mode` alternatives.

Note that the same threshold is **biologically stricter on shorter proteins**
(fewer absolute mutations admitted). See § 5 for a per-function table.

### 2.2 easy-cluster vs easy-linclust

mmseqs2 provides two main methods for clustering:

- **`easy-cluster`** — runs standard cascaded clustering workflow, using multiple sensitivity stages and optional reassignment; generally more sensitive, but slower.
-  **`easy-linclust`** — runs the Linclust workflow, designed for near-linear-time clustering of very large sequence sets; much faster, but usually less sensitive than cascaded clustering.

**Choice on Flu A: symmetric easy-linclust on both alphabets** (since
2026-05-22). The wrapper at `src/utils/clustering_utils.py::run_mmseqs_easy_clust`
defaults to `algorithm='linclust'` and is what
`seq_redundancy_per_function.py` invokes for both the aa and nt
sweeps. Decision-relevant mmseqs2 flags are pinned explicitly on the
CLI (see that wrapper's docstring for the full pinned set:
`--cluster-mode 0`, `--seq-id-mode 0`, `--similarity-type 2`,
`-e 0.001`, `--dbtype 1` (aa) / `2` (nt), in addition to the
caller-supplied `--min-seq-id`, `-c 0.8`, `--cov-mode 0`).

---

## 3. The three knobs that define a clustering

```
mmseqs <easy-cluster | easy-linclust>  <input.fasta>  <out_prefix>  <tmp_dir> \
    --min-seq-id <t>   -c 0.8   --cov-mode 0   [--dbtype 2]
```

The pipeline wires these via
`src/utils/clustering_utils.py::run_mmseqs_easy_clust`. The three knobs
that actually drive behavior are below.

### 3.1 `--min-seq-id <t>` — identity threshold

Range [0, 1]. Two sequences cluster iff their pairwise identity is
≥ t (and they pass the coverage rule below). Counted on residues
(aa for protein, nt for DNA). At t = 1.0, only sequence pairs with
100% identity *over the aligned region* would cluster together — the
aligned region doesn't have to span the whole sequence. Under our
`-c 0.8 --cov-mode 0` settings (§3.2), up to 20% of each sequence can
lie outside the alignment, so sequences that differ only in length
(length-variants — e.g., NA stalk-deletion isoforms) can still
cluster at t = 1.0 even though the sequences are not identical. See
§2.1 Case 1 example and §6.4's NA id100 row (6.9% of
the corpus pooled into one cluster) for the empirical evidence on
Flu A.

### 3.2 `-c 0.8` and `--cov-mode 0` — coverage rule

A 200-aa fragment can be 100% identical
to a 200-aa stretch of an 800-aa full protein. The coverage rule
prevents that fragment from clustering with the full protein.

```
                                 [identity = 100% on the aligned region]
query   ════════════════════════════════════════════════   Lq = 800
                       ║║║║║║║║║║║║║║║║║║
target                 [─── fragment ───]                  Lt = 200
                       covered region: 200 residues

cov(target) = 200 / 200 = 100%   ✓  (≥ 0.8)
cov(query)  = 200 / 800 =  25%   ✗  (< 0.8)
```

Under **`--cov-mode 0` (bidirectional)** the rule requires
`cov(query) ≥ c` AND `cov(target) ≥ c`. The fragment-vs-full case
above fails the rule and does not cluster.

Note that other cov-modes (1 = target-only, 2 = query-only) would let
the above fragment-vs-full cluster through. We use cov-mode 0
because it's the conservative choice: every pair that clusters
contains two sequences of comparable length. Sequence-length variation on Flu A
major proteins is small (`gto_format_reference.md` §6.5: std ≤ 2.8 aa per
function).

### 3.3 `--dbtype` and the alphabet

mmseqs represents biological sequences differently depending on
alphabet. The flag passed through to `createdb` under the hood:

| flag | alphabet | substitution matrix used | ambiguity codes |
|---|---|---|---|
| `--dbtype 1` (default, protein) | 20 amino acids + X + stop | blosum62 | `X` (the only true ambiguity in our cleaned corpus) |
| `--dbtype 2` (nucleotide) | A, C, G, T | nucleotide.out | IUPAC: N, R, Y, S, W, K, M, B, D, H, V |

The substitution matrix scores residue mismatches during alignment.
This matters when an ambiguous residue is aligned to a concrete one:
both matrices return a small positive score for the X-vs-anything
case, so an alignment with a handful of X residues can still pass the
identity threshold.

Stage 1 (`preprocess_flu.py`) removes proteins with high X-fraction
(`prepare_sequences_for_esm2`), so the aa side rarely encounters X.
The nt side leaves IUPAC codes intact; mmseqs handles them via the
nucleotide matrix.

Note that `--alph-size` is set internally by mmseqs (aa: 21, nucl: 5).

---

## 4. Corpus redundancy (Flu A)

**Source.** `src/analysis/cluster_analysis_summary.py`; `results/flu/July_2025/runs/cluster_analysis/cluster_summary.csv`

Columns:
- `Total seqs` = the number of isolates that carry this protein.
- `Unique aa seqs` / `Unique nt seqs` = unique sequence count after
  dedup on `prot_seq` / `cds_dna`. This is the FASTA
  row count that mmseqs sees as input (it's the pre-clustering dedup).
- `% unique aa` = `Unique aa seqs` / `Total seqs` (and the same for
  `% unique nt`). High % means
  more diverse population at the sequence level; low % means heavily
  redundant population.

| Segment | Function | Total seqs | Unique aa seqs | % unique aa | Unique nt seqs | % unique nt |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PB2 | 108,530 | 33,663 | 31.0% | 67,341 | 62.1% |
| 2 | PB1 | 108,530 | 31,226 | 28.8% | 67,034 | 61.8% |
| 3 | PA  | 108,530 | 34,217 | 31.5% | 65,242 | 60.1% |
| 4 | HA  | 108,530 | 41,896 | 38.6% | 65,414 | 60.3% |
| 5 | NP  | 108,530 | 17,684 | 16.3% | 52,800 | 48.6% |
| 6 | NA  | 108,530 | 37,488 | 34.5% | 58,887 | 54.3% |
| 7 | M1  | 108,530 |  4,771 |  4.4% | 32,413 | 29.9% |
| 8 | NS1 | 108,530 | 22,225 | 20.5% | 38,039 | 35.0% |

**Takeaways.**

- **`% unique nt` is always higher than `% unique aa`** (all 8 functions).
  Synonymous codons create distinct CDS DNAs that collapse to one
  protein, so within a function the nt count is ≥ the aa count.
  Magnitude varies: M1's nt/aa ratio is ~7× (32,413 /
  4,771); HA's is ~1.6× (65,414 / 41,896).
- **M1 is the most redundant (and conserved).** Only 4,771 distinct M1 aa
  sequences across 108,530 isolates (~95% redundancy in aa). M1 is the most
  aa-conserved Flu A protein.
- **NS1 is an outlier on this column.** 4.7× M1's unique-aa count
  (22,225 vs 4,771) despite being slightly shorter (median 231 vs
  253 aa). The decomposition (mostly lower aa conservation, partly a
  length-variation md5 artifact) is worked out in §6.2 using the
  across-threshold ratio data, since the evidence requires §6.1's
  numbers.

---

## 5. What an identity threshold concretely admits

**Source.** `src/analysis/cluster_analysis_summary.py`;
`results/flu/July_2025/runs/cluster_analysis/mutations_tolerated_table.csv`,
`sequence_length_summary.csv`

Columns:
- `Median aa/nt len` = median sequence length `L` for the function
  (aa, nt). Computed from
  per-sequence lengths; the spread within a function is small on
  Flu A (≤2 aa for 7/8 functions; NS1 is the length-varying
  exception — see §4 notes).
- `id###` = maximum residue mismatches admitted within a cluster at
  threshold `t = ###/100`, computed as `L − ceil(L × t)`. In mmseqs,
  threshold `t` is set by `--min-seq-id`.

**Flu A data (aa).**

| Segment | Function | Median aa len | id100 | id099 | id098 | id097 | id096 | id095 | id090 | id085 | id080 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 760 | 0 |  7 | 15 | 22 | 30 | 38 | 76 | 114 | 152 |
| 2 | PB1 | 758 | 0 |  7 | 15 | 22 | 30 | 37 | 75 | 113 | 151 |
| 3 | PA  | 717 | 0 |  7 | 14 | 21 | 28 | 35 | 71 | 107 | 143 |
| 4 | HA  | 567 | 0 |  5 | 11 | 17 | 22 | 28 | 56 |  85 | 113 |
| 5 | NP  | 499 | 0 |  4 |  9 | 14 | 19 | 24 | 49 |  74 |  99 |
| 6 | NA  | 470 | 0 |  4 |  9 | 14 | 18 | 23 | 47 |  70 |  94 |
| 7 | M1  | 253 | 0 |  2 |  5 |  7 | 10 | 12 | 25 |  37 |  50 |
| 8 | NS1 | 231 | 0 |  2 |  4 |  6 |  9 | 11 | 23 |  34 |  46 |

**Flu A data (nt).**

| Segment | Function | Median nt len | id100 | id099 | id098 | id097 | id096 | id095 | id090 | id085 | id080 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 2,280 | 0 | 22 | 45 | 68 | 91 | 114 | 228 | 342 | 456 |
| 2 | PB1 | 2,274 | 0 | 22 | 45 | 68 | 90 | 113 | 227 | 341 | 454 |
| 3 | PA  | 2,151 | 0 | 21 | 43 | 64 | 86 | 107 | 215 | 322 | 430 |
| 4 | HA  | 1,701 | 0 | 17 | 34 | 51 | 68 |  85 | 170 | 255 | 340 |
| 5 | NP  | 1,497 | 0 | 14 | 29 | 44 | 59 |  74 | 149 | 224 | 299 |
| 6 | NA  | 1,410 | 0 | 14 | 28 | 42 | 56 |  70 | 141 | 211 | 282 |
| 7 | M1  |   759 | 0 |  7 | 15 | 22 | 30 |  37 |  75 | 113 | 151 |
| 8 | NS1 |   693 | 0 |  6 | 13 | 20 | 27 |  34 |  69 | 103 | 138 |

**Takeaways.**

- **Same threshold, different biological criterion per function.**
  id095 on M1 admits **3× fewer absolute mismatches** than id095 on
  PB2 (12 vs 38, from 253 vs 760 aa median lengths). This explains
  why per-function cluster collapse rates in §6 differ between
  functions at the same threshold: the threshold admits more
  mutations on longer proteins, so more sequences fall into the
  same cluster.

**Future direction.**

- **Per-function thresholds via uniform mismatch budget.** For
  comparing across function pairs of different lengths (e.g.,
  HA/NA ~470–567 aa vs PB2/PB1 ~758–760 aa), an absolute mismatch
  budget may be more informative than a fractional identity
  threshold. Picking per-function thresholds to enforce a uniform
  mismatch budget is tracked in
  `docs/results/2026-05-21_bicc_pair_drop_audit.md` direction #5.

---

## 6. Cluster collapse trajectory

Source. `cluster_analysis_summary.py`. `cluster_summary.csv` contains
raw values. The sweep covers thresholds {1.00, 0.99, 0.98, 0.97, 0.96,
0.95, 0.90, 0.85, 0.80}. Plots: `cluster_counts_vs_threshold.png` and
`bipartite_largest_pct_vs_threshold.png`.


### 6.1 Per-function `n_clusters` (aa)

| Segment | Function | id100 | id099 | id098 | id097 | id096 | id095 | id090 | id085 | id080 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 33,601 | 18,354 | 10,035 |  7,634 |  6,755 |  6,491 | **24** |  2 |   2 |
| 2 | PB1 | 30,822 | 17,209 | 11,859 |  9,266 |  7,384 |  2,033 |  **6** |  2 |   1 |
| 3 | PA  | 34,162 | 18,520 | 12,758 | 10,906 |  8,677 |  8,002 | **17** |  2 |   2 |
| 4 | HA  | 41,760 | 22,679 | 14,934 | 11,459 |  8,940 |  7,578 |   910 |  407 | 176 |
| 5 | NP  | 17,533 | 10,483 |  5,038 |  1,750 |    613 |    526 | **29** |  2 |   2 |
| 6 | NA  | 18,753 |  9,369 |  6,909 |  4,707 |  3,107 |  2,134 | 1,077 |  127 |  73 |
| 7 | M1  |  4,712 |  1,764 |  1,033 |  1,003 |    708 |    129 | **24** |  10 |  3 |
| 8 | NS1 | 22,131 | 13,508 |  9,109 |  6,405 |  4,306 |  3,458 |   786 |  196 | 174 |

**Per-function `n_clusters` (nt)**, for comparison:

| Segment | Function | id100 | id099 | id098 | id097 | id096 | id095 | id090 | id085 | id080 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 66,475 | 11,484 |  4,562 |  2,968 |  1,353 |    954 |    180 |  24 |   5 |
| 2 | PB1 | 66,138 | 14,990 |  6,276 |  2,548 |  1,307 |    742 |    121 |  18 |  10 |
| 3 | PA  | 64,406 | 11,184 |  4,356 |  2,214 |  1,071 |    719 |     47 |  13 |   3 |
| 4 | HA  | 64,526 | 12,150 |  6,412 |  3,444 |  1,843 |  1,277 |    275 | 141 |  85 |
| 5 | NP  | 52,097 | 11,627 |  4,754 |  2,559 |  1,427 |  1,182 |     66 |  12 |   3 |
| 6 | NA  | 57,987 | 12,092 |  4,840 |  2,611 |  1,662 |  1,108 |    250 | 134 |  91 |
| 7 | M1  | 31,974 | 10,227 |  4,239 |  1,420 |  1,101 |    619 |     48 |   5 |   2 |
| 8 | NS1 | 37,458 | 12,012 |  4,133 |  1,988 |  1,245 |    800 |    196 |  91 |   8 |


====================================

Moved from section 3.2

[TODO: Need to add "Total seqs"	"Unique aa seqs"]

**`Unique aa seqs` vs `id100` — the coverage-driven gap.** `Unique aa
seqs` is the post-md5-dedup count: every byte-distinct protein sequence
is its own row. `id100` is the mmseqs cluster count at threshold = 1.0
under `-c 0.8 --cov-mode 0`: pairs with 100% identity over the aligned
region AND ≥80% mutual coverage cluster together — even if the full
sequences differ in length. On 7 of 8 functions the gap is ≤1.3% (e.g.,
HA 41,896 → 41,760 = 0.3%, M1 4,771 → 4,712 = 1.2%). **NA: 37,488 →
18,753 = 50% reduction.** The mechanism is NA's stalk-length
variation — stalk-deletion isoforms are 100% identical to stalk-full
counterparts over the aligned region (head + membrane), and the
deletion is small enough relative to NA's ~470 aa length that ≥80%
coverage on each side passes. They cluster at id100 even though the
sequences differ in residue count. See §2.1 Case 1 for the schematic
and the NA caveat block at the end of §4 for the full reading-guide
note.

The `id100` → `id099` column drop is an identity-threshold effect,
not a coverage effect; see §3.1 (and §6.1 for the full
id100…id080 n_clusters sweep).

====================================

nt at id100 always exceeds aa at id100 (synonymous variants split into
distinct nt singletons). The id095→id090 conserved-protein cliff
visible on aa (§6.1 aa table) is muted on nt: PB2 nt drops 954→180
(−81%) where aa drops 6,491→24 (−99.6%). At id090, nt retains 47–275
clusters on every function whereas aa is reduced to 6–29 on the five
conserved functions.

**No single-pp cliff.** Under symmetric easy-linclust the steepest
1 pp drop on any function is ~25% (PB2 at id098→id097: 10,035→7,634).
The conserved-protein cliff is now at the **id095→id090** transition
(a 5 pp identity gap): PB2 6,491→24 (−99.6%), PB1 2,033→6 (−99.7%),
PA 8,002→17 (−99.8%), NP 526→29 (−94.5%), M1 129→24 (−81.4%). These
five conserved-protein values are bolded above.

(Historical note. Prior to 2026-05-22 this section's table was
produced under easy-cluster's 3-round cascade, which chained
transitively-similar conserved-protein sequences into one cluster
much earlier in the threshold sweep. The same PB2 trajectory under
easy-cluster was 717→77 between id097 and id096 — a 1 pp cliff at
id096. Under symmetric easy-linclust the cascade no longer chains
those sequences, the cliff moves down to the 5 pp id095→id090 gap,
and the conserved-vs-surface contrast is muted relative to the prior
narrative. See §2.3 for the algorithm-change rationale.)

**aa vs nt at the same threshold — surprising under symmetric easy-linclust.**
Under the prior asymmetric setup (easy-cluster on aa, easy-linclust on
nt), nt cluster counts were always higher than aa counts at the same
threshold, which was attributed to synonymous-codon variation keeping
aa-identical sequences in distinct nt clusters. Under symmetric
easy-linclust the relationship is more complex:

- At id100 (exact identity), nt has more clusters than aa on every
  function (synonymous variants split into distinct nt singletons —
  the long-standing intuition holds here).
- At id099 and id098, **nt has fewer clusters than aa** on five of
  eight functions (HA at id099: 22,679 aa clusters vs 12,150 nt
  clusters). M1 is the strongest outlier in the other direction
  (id099: 1,764 aa vs 10,227 nt, a 5.8× excess).
- **Mechanism (cross-tab analysis, 2026-05-22):** aa and nt
  clusterings at id099 are *not nested* — each finds within-cluster
  variation the other misses, on every function. Two opposing
  effects compete: (A) nt's rep-based clustering on the longer CDS
  sequence absorbs aa-distinct point-variant swarms into a single nt
  cluster (drives nt < aa); (B) synonymous-codon variation fragments
  aa-identical CDS across multiple nt id099 balls (drives nt > aa).
  Net direction depends on the function's aa-vs-synonymous diversity
  balance. Full numbers, per-function regime table, and two
  illustrative walkthroughs in
  `docs/results/2026-05-22_aa_vs_nt_cluster_mechanism.md`.
  Cross-tab script: `src/analysis/aa_nt_cluster_crosstab.py`.

### 6.2 Worked example: NS1 vs M1 — real diversity vs length-variation artifact

§4 noted that NS1 has 4.7× M1's unique-aa count at md5-dedup (22,225
vs 4,771) despite being slightly shorter (median 231 vs 253 aa).
Two compounding causes, decomposable using §6.1's per-threshold counts.

**Cause 1 (dominant): NS1 is less aa-conserved than M1.** M1 is a
structural matrix protein under tight constraint; NS1 hosts the
interferon-antagonist effector domain and host-adaptation modules
that tolerate more residue substitution. This is real biology.

Evidence — the NS1/M1 cluster-count ratio **grows** as the threshold
loosens (from §6.1's aa table):

| Threshold | NS1 clusters | M1 clusters | NS1 / M1 ratio |
|---:|---:|---:|---:|
| id100 | 22,131 | 4,712 |  4.7× |
| id099 | 13,508 | 1,764 |  7.7× |
| id098 |  9,109 | 1,033 |  8.8× |
| id095 |  3,458 |   129 | 26.8× |
| id090 |    786 |    24 | 32.8× |

If the id100 gap were just a length-variation artifact, the ratio
would *shrink* at lower idXX — length-variants would collapse first,
removing NS1's measurement-inflated portion. The opposite happens:
the ratio grows by ~7× from id100 to id090. NS1 has substantially
more genuine residue diversity than M1; M1 collapses faster because
near-identical sequences are common in M1's tight conservation space.

**Cause 2 (minor, but worth flagging): length variation inflates
NS1's id100 count.** NS1's aa length spans 201–239 (≈20-aa spread,
from `sequence_length_summary.csv`) vs M1's ≤2 aa spread. Md5-dedup
treats any byte-difference as a new "unique sequence", so NS1's
length-variants count separately at md5-dedup. Mmseqs's coverage
rule (`-c 0.8 --cov-mode 0`) admits length-variants of similar
length but still produces only ~94 fewer id100 clusters than the
md5-dedup count (22,131 vs 22,225, a 0.4% collapse) — modest.
Length-variants merge more fully at id < 1.0 where the residue
threshold dominates.

**Takeaway.** NS1's high unique-aa count at id100 is **mostly real
residue diversity** (driven by lower conservation than M1) and
**partly a length-variation artifact** (~0.4% of the count,
evidenced by the small id100 vs md5-dedup gap). The §4 read of
"NS1 anomaly" should be understood as "NS1 has more real biological
diversity than M1, with a small measurement-artifact contribution
on top".

### 6.3 Two collapse modes (one deferred cliff, one gradual)

Two patterns are visible in the §6.1 table:

- **Deferred-cliff functions** (PB2, PB1, PA, NP, M1). Cluster counts
  decrease moderately through id100 → id095 (e.g., PB2: 33,601 →
  6,491, ~80% retention), then drop sharply at id095 → id090 (PB2:
  6,491 → 24, −99.6%). The conserved-protein "cliff" exists but
  spans a 5 pp identity gap rather than a single pp step. These are
  the most aa-conserved Flu A proteins (consistent with their role
  in polymerase activity and particle structure — high constraint,
  low aa drift); their sequence space is narrow enough that loosening
  the identity threshold from 0.95 to 0.90 absorbs nearly every
  remaining cluster. By id090 they are down to 6–29 clusters total —
  essentially "Flu A polymerase, a handful of population-level
  subfamilies".

- **Gradual functions** (HA, NA, NS1). Cluster counts decline
  smoothly across the whole threshold sweep, retaining meaningful
  structure even at id080 (HA: 176 aa clusters, NA: 73, NS1: 174).
  HA and NA carry substantial aa-level variation — antigenic drift
  drives diversity — and NS1 has the length-variation noise discussed
  in §4 (NS1 is short with a variable C-terminal tail). NS1 in
  particular has a smaller id095 → id090 drop (3,458 → 786, −77%)
  than the deferred-cliff group, though it's still substantial.

- **NA is the most-gradual outlier**, even within the surface-protein
  group. NA's id095 → id090 drop is only 2,134 → 1,077 (−50%, much
  smaller than HA's 7,578 → 910 = −88%). This is partly the
  stalk-length-variation effect from §4 — NA's clusters are already
  pooled by length variation at id100, so further sequence-level
  consolidation has less effect.

### 6.4 Largest cluster as % of corpus

Per-function "how much of the corpus does one cluster swallow"
(derived as `largest_cluster / n_sequences × 100` from
`cluster_summary.csv`; not plotted as such — the
`bipartite_largest_pct_vs_threshold.png` plot is a different
per-pair view, see §9):

| Segment | Function | id100 | id099 | id098 | id097 | id096 | id095 | id090 | id085 | id080 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 0.0% | 12.7% | 12.6% | 13.7% | 15.6% | 15.6% |  95.9% | **100%** | **100%** |
| 2 | PB1 | 0.1% | 12.9% | 12.0% | 18.5% | 25.9% | 72.4% | **100%** | **100%** | **100%** |
| 3 | PA  | 0.0% |  7.3% |  8.9% |  8.9% |  9.0% |  9.0% |  98.1% | **100%** | **100%** |
| 5 | NP  | 0.1% |  7.5% | 13.1% | 28.2% | 41.4% | 46.3% |  99.8% | **100%** | **100%** |
| 7 | M1  | 0.2% | 10.8% | 17.8% | 23.1% | 41.7% | 56.7% |  99.2% |  99.5% |  99.9% |
| 4 | HA  | 0.0% |  5.0% |  6.7% |  9.6% |  9.7% | 11.8% |  22.8% |  24.3% |  33.4% |
| 6 | NA  | 6.9% |  8.7% |  8.9% |  8.8% | 13.2% | 13.2% |  17.9% |  32.4% |  37.7% |
| 8 | NS1 | 0.0% |  3.9% |  4.8% |  9.4% | 11.7% | 16.3% |  21.1% |  29.5% |  52.4% |

100% means the cluster contains the entire corpus. **Conserved-protein
collapse is delayed**: under symmetric easy-linclust the largest
cluster fraction stays below 25% through id095 on every function
except M1 (which reaches 57% at id095), then jumps to 96–100% at
id090 on PB2/PB1/PA/NP/M1. By id090 the five conserved functions have
swallowed essentially the entire corpus; HA, NA, NS1 remain ≤30%
(HA = 22.8%, NA = 17.9%, NS1 = 21.1% at id090).

NA's id100 fraction (6.9%) is the only sub-id100 entry visibly above
zero — the stalk-length effect from §4 again: NA's id100 cluster
already pools ~7% of the corpus through length-variant absorption,
before any sequence-similarity clustering takes effect.

Note: this is the per-FUNCTION largest cluster fraction (one number
per function per threshold). §9 reports the per-PAIR largest
bipartite-COMPONENT fraction — that's the quantity that actually
gates 80/10/10 feasibility, and it can be much larger than the per-
function cluster fraction because two functions' clusters get linked
by shared isolates.

### 6.5 Why this matters for routing

The per-function collapse trajectory predicts the bipartite-CC
feasibility ceiling documented in §9. Function-pairs whose components
collapse sharpest at the conserved-protein cliff (polymerase pairs
like PB2/PB1) form a single mega-component once either slot's
clustering collapses at id095 → id090, defeating the LPT-greedy
routing. HA/NA retains the most structural diversity at any given
threshold and remains the most "splittable" pair.

**The collapse shape is both corpus-driven AND algorithm-driven** —
not the "corpus-driven only" framing this section asserted prior to
2026-05-22. A 2026-05-21 validation experiment found that easy-cluster's
3-round cascade chains transitively-similar sequences much further
than easy-linclust's single-pass prefilter does, producing cluster
counts that disagreed by 5×–500× on the same aa input at identical
parameters (§2.3). Under symmetric easy-linclust the algorithm
contribution is held constant on both alphabets, so the trajectories
above reflect corpus structure rather than an algorithm × alphabet
confound. This re-baselines the §9 feasibility comparison: any
aa-vs-nt difference in the new measurements (§9) reflects alphabet
diversity, not algorithm sensitivity.

Earlier results that relied on the corpus-driven-only framing — in
particular the §9 comparison and the
`docs/results/2026-05-15_cluster_disjoint_nt_results.md` write-up —
should be re-read with the understanding that the prior aa numbers
were generated under easy-cluster.

---

## 7. From a per-function clustering to a train/val/test partition

### 7.1 Why per-function clusters aren't enough

mmseqs2 clusters sequences within one function (HA clusters, NA
clusters, ...). But our prediction task is on *pairs* of two
functions (slot A = HA, slot B = NA). What we actually need to
partition is the set of (slot_A_cluster, slot_B_cluster) pairs.

A naive "ensure no slot A cluster appears in both train and test"
already isn't right. An isolate carries both proteins, so two pairs
that share *either* a slot A cluster *or* a slot B cluster are linked
through that isolate. Pull on one pair and others come with it.

The right structure is the **bipartite component** on the (slot A,
slot B) cluster graph:

```
HA aa clusters (slot A)             NA aa clusters (slot B)
   ┌─────────┐
   │ HA_c1   │ ─── pair (isolate i) ───►   ┌─────────┐
   └─────────┘                              │ NA_c5   │
                                            └────┬────┘
                                                 │
                              pair (isolate j) ──┘    
                                ┌─►   ┌─────────┐
   ┌─────────┐                  │     │ NA_c9   │
   │ HA_c2   │ ── pair (k) ─────┘     └─────────┘
   └─────────┘
   ┌─────────┐ ─── pair (m) ────────► ┌─────────┐
   │ HA_c3   │                        │ NA_c7   │
   └─────────┘                        └─────────┘


Connected components on this bipartite graph:
   CC #1 = { HA_c1, NA_c5 }              ← pairs in CC #1 must split together
   CC #2 = { HA_c2, NA_c9 }
   CC #3 = { HA_c3, NA_c7 }
```

Pairs inside one CC must land in the same split — otherwise a cluster
on one side would be in both train and test, defeating the purpose.
The routing is a **bin-packing of indivisible CCs**.

### 7.2 80/10/10 by LPT-greedy

The routing helper
(`src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df`)
sorts CCs by size descending and greedily assigns each CC to the
split with the largest current deficit relative to its target:

```
targets:      train = 80%   val = 10%   test = 10%
biggest CC first → goes to the bin with the largest open quota →
   typically train first, then val and test fill from medium-sized CCs.
```

This is the "longest-processing-time-first" (LPT) heuristic for
bin packing. It's optimal up to the largest item: if the largest CC
exceeds the largest bin's target, the routing is structurally
infeasible at 80/10/10. The largest-CC fraction is therefore the
quantity to check before committing to a (pair, alphabet, threshold)
configuration.

**The router never drops pairs.** When the largest CC exceeds the
80% train quota, LPT-greedy still places the whole CC in train —
train just overflows its target. The router does not split CCs and
does not discard boundary pairs; "infeasible" in this doc means
"the achieved 80/10/10 ratios drift", not "pairs are lost". The audit
JSON's `pairs_dropped_in_routing` and `pairs_dropped_in_cluster_join`
are always 0 in practice (verified 2026-05-21 across HA/NA aa id099,
HA/NA aa id095, PB2/PB1 aa id099, HA/NA nt id099, PB2/PB1 nt id099 —
see `docs/results/2026-05-21_bicc_pair_drop_audit.md`). The
operational consequence is that at sub-feasibility thresholds, val
and test starve: HA/NA aa id095 produces a 98.5 / 0.76 / 0.76 split
rather than 80 / 10 / 10, even though every pair is routed somewhere.

### 7.3 The four implemented routings — quick reference

| mode + alphabet | slot key | two pairs share a CC iff … |
|---|---|---|
| `seq_disjoint` `hash_key=seq` (default) | `seq_hash = md5(prot_seq)` exact | identical protein on a slot |
| `seq_disjoint` `hash_key=dna` | `dna_hash = md5(contig.dna)` exact | identical full contig (UTR + CDS + intron + UTR) on a slot |
| `cluster_disjoint` `cluster_alphabet=aa` | mmseqs2 aa cluster id at chosen threshold | aa-similar protein on a slot |
| `cluster_disjoint` `cluster_alphabet=nt` | mmseqs2 nt cluster id at chosen threshold, keyed on `cds_dna_hash` | nt-similar CDS DNA on a slot (UTRs and introns excluded) |

Deep dive on equivalences and non-equivalences:
`docs/methods/leakage_definitions.md` § "Routing equivalence and
mmseqs argument semantics".

### 7.4 Naming: bipartite-CC LPT-greedy ≈ cluster_disjoint routing ≈ BiCC-Split

The routing algorithm goes by several names across docs, code, and
writeups:
- **bipartite-CC LPT-greedy** — the precise technical name (bipartite
  connected components of (slot_A_cluster, slot_B_cluster), bin-packed
  via Longest-Processing-Time first-fit decreasing into the requested
  split fractions).
- **cluster_disjoint routing** — the user-facing config name in
  `dataset.split_strategy.mode: cluster_disjoint` and in
  `seq_disjoint_route_pos_df` / `cluster_disjoint_route_pos_df`.
- **BiCC-Split** — paper/prose shorthand ("Bipartite Connected-Component
  Split"), abbreviated **BiCC** or **bicc** in inline references.

These are the same algorithm. In committed docs, prefer **bipartite-CC
LPT-greedy** on first mention with the abbreviation **bicc** for
subsequent references. Reserve `cluster_disjoint` for code/config and
`BiCC-Split` for manuscript prose where a memorable name is useful.

The algorithm differs from DataSAIL's split heuristic (cluster + ILP)
in two ways: (a) routing operates on bipartite-CCs as atomic units,
never dropping pairs ("CC bin-packing never splits a component",
`_split_helpers.py:267`), where DataSAIL's I2/S2 explicitly drop
pairs that straddle folds; (b) bicc's LPT-greedy is a heuristic that
hits the requested split fractions within ~0.01% on Flu A (memory.md
"seq_disjoint scales to conserved proteins"), where DataSAIL solves
an NP-hard ILP via a heuristic clustering pre-pass.

---

## 8. Pipeline integration

Three scripts handle the producer/consumer chain, all under
`src/analysis/` (the cluster artifacts they produce are inputs to
Stage 3 via `src/datasets/_split_helpers.py`):

| Step | Script | Reads | Writes |
|---|---|---|---|
| Build CDS (nt only) | `src/preprocess/extract_cds_dna.py` (Stage 1.5) | `protein_final.csv` + `genome_final.csv` | `cds_final.parquet` |
| Cluster sweep | `src/analysis/seq_redundancy_per_function.py` | `protein_final.parquet` (aa) or `cds_final.parquet` (nt) | `clusters_{aa,nt}/`: per-function FASTAs, per-threshold cluster parquets, `combined_cluster.parquet`, `redundancy_stats.csv`, `runtime.json`, `redundancy_summary.md` |
| Feasibility pre-flight | `src/analysis/cluster_disjoint_feasibility.py` | one cluster lookup + `protein_final` or `cds_final` | `results/flu/{version}/runs/cluster_disjoint_feasibility/feasibility_<pair>_<alphabet>.csv` |
| Stage 3 consumes | `src/datasets/dataset_segment_pairs_v2.py` (when `split_strategy.mode: cluster_disjoint`) | `combined_cluster.parquet` for the chosen (alphabet, threshold) | `dataset_*/cluster_disjoint_audit.json` |

For consolidated structural analysis (the empirical tables and plots
in §4 / §5 / §6 / §9), the post-hoc script is
`src/analysis/cluster_analysis_summary.py`, which reads the
redundancy + feasibility CSVs and emits plots under
`results/flu/{version}/runs/cluster_analysis/`.

---

## 9. Bipartite-component feasibility ceiling on Flu A

The §7.2 question — "is the largest CC small enough for 80/10/10?" —
answered per (schema pair, alphabet, threshold). Source: the four
feasibility CSVs at
`results/flu/{version}/runs/cluster_disjoint_feasibility/feasibility_<pair>_<alphabet>.csv`,
generated by `src/analysis/cluster_disjoint_feasibility.py`.
Consolidated plot: `bipartite_largest_pct_vs_threshold.png` from
`cluster_analysis_summary.py`.

Largest bipartite-component fraction (% of deduped pairs):

| Segments | Schema pair | Alphabet | id100 | id099 | id098 | id097 | id096 | id095 | id090 | id085 | id080 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4/6 | HA/NA   | aa | 49.0 | 79.6 | 88.4 | 92.7 | 95.8 | 97.8 |  99.8 | 100.0 | 100.0 |
| 4/6 | HA/NA   | nt |  1.5 | 69.3 | 91.0 | 95.7 | 97.8 | 98.2 |  99.1 |  99.6 | 100.0 |
| 1/2 | PB2/PB1 | aa | 38.4 | 81.0 | 92.6 | 97.2 | 98.6 | 99.5 | 100.0 | 100.0 | 100.0 |
| 1/2 | PB2/PB1 | nt |  2.9 | 59.7 | 93.9 | 97.2 | 98.2 | 99.1 |  99.5 | 100.0 | 100.0 |

A cell is structurally feasible for 80/10/10 if the largest CC is ≤80%
(train can fit it cleanly). What "infeasible" means in practice: the
router still places every pair (see §7.2 — `pairs_dropped_*` are
always 0), but train absorbs the mega-CC and val/test drift toward
zero. Reading the table by that frame:

- **id100 (every cell):** feasible. Largest CC is at most 49.0%
  (HA/NA aa — note this is markedly larger than under the prior
  easy-cluster baseline (20.2%) because of NA's stalk-length absorption
  on the aa side, see §4 footnote and §6.4). Routing still has room.
- **id099 (marginal on all four cells now):** HA/NA aa at 79.6% lands
  just under the ceiling — bin-packer achievable. PB2/PB1 aa at 81.0%
  is 1 pp over the ceiling — borderline feasible (substantially better
  than the 87.1% reported here pre-2026-05-22 under easy-cluster). nt
  cells (HA/NA = 69.3%, PB2/PB1 = 59.7%) are comfortably below the
  ceiling. The cross-alphabet gap at id099 is now real (~10 pp) rather
  than confounded by the easy-cluster vs easy-linclust algorithm
  asymmetry. Aa is still tighter than nt here.
- **id098 (above the ceiling on all four cells):** aa cells are 88.4%
  (HA/NA) and 92.6% (PB2/PB1); nt cells are 91.0% and 93.9%. All four
  cross the ceiling but margins are tighter than they were under
  easy-cluster (HA/NA aa was 93.7%, PB2/PB1 aa was 98.0%). Not built
  empirically yet under the new clustering — the §6.1 collapse
  trajectory and the largest-CC % both predict an ~88-93% / 4-6% / 4-6%
  split. Closer to feasibility than the prior easy-cluster numbers
  suggested.
- **id097 and below (broken everywhere — but builds still run):** at
  HA/NA aa id095 (the one sub-ceiling threshold we did build, *under
  easy-cluster*) the routing produced 98.48 / 0.76 / 0.76 — val and
  test got 1,107 pairs each vs 14,597 intended (a 92% capacity loss
  on the held-out splits). Every pair is routed; the dataset exists;
  it just isn't usable for evaluation. Largest CC ≥95% on every line
  of the table predicts the same pattern under the new clustering.

> **TODO! Observed train share on existing runs (2026-05-21 audit) —
> these reflect prior easy-cluster artifacts.**
>
> The cluster_disjoint datasets the table below summarizes were built
> using the previous easy-cluster aa artifacts (pre-2026-05-22), not
> the current symmetric easy-linclust artifacts. The largest CC %
> values now in the §9 table above are from the new linclust
> artifacts; the achieved train % below is from a different (older)
> dataset chain. Rebuilding the cluster_disjoint datasets at id099
> under linclust — and possibly id098 now that it's closer to
> feasible — is required for an apples-to-apples observed-train-share
> measurement.
>
> | Segments | Schema pair | Alphabet | Threshold | Largest CC % | Achieved train % | Max ratio drift |
> |---|---|---|---|---:|---:|---:|
> | 4/6 | HA/NA   | aa | id099 | 80.0 (old) | 80.00 | 0.0007% |
> | 4/6 | HA/NA   | nt | id099 | 69.3 | 80.00 | 0.0007% |
> | 1/2 | PB2/PB1 | nt | id099 | 59.7 | 80.00 | 0.0011% |
> | 1/2 | PB2/PB1 | aa | id099 | 87.1 (old) | 87.12 | 7.12 pp |
> | 4/6 | HA/NA   | aa | id095 | 98.5 (old) | 98.48 | 18.48 pp |

Read: the largest-CC % is the algorithmic *input* (and the predictor
of feasibility); the achieved-train % is the realised *output* and
the actually-actionable number for downstream consumers. When largest
CC ≤ 80%, the two diverge only at the 4th decimal. When largest CC >
80%, achieved train % tracks largest CC % closely (the bin-packer
can't undo what the mega-CC dictates).

**Interpretation: feasibility ceiling is now algorithm-controlled.**
Under symmetric easy-linclust the §6 collapse trajectory is corpus-
driven by construction (algorithm is constant across alphabets, see
§6.5). The aa-vs-nt feasibility gap at id099 (aa near ceiling, nt
comfortably below) is now a clean comparison: it reflects the
alphabet's underlying diversity structure plus the corpus's
metadata-driven bipartite linking, not an algorithm × alphabet
confound. The expectation going in was that nt clustering would
unlock lower thresholds via synonymous diversity. It still doesn't
unlock id098 or below — even at id098 nt is at 91-94% — because the
*bipartite linking* between HA clusters and NA clusters is determined
by which isolates carry which (HA, NA) combinations, and Flu A's
small set of dominant HxNy subtypes × host × year cells links most
pairs into one mega-component well above id098.

The empirical confirmation under the prior easy-cluster clustering
was in `docs/results/2026-05-15_cluster_disjoint_nt_results.md`:
B-nt was limited to (id100, id099) on both alphabets. The new linclust
numbers above suggest id098 should be reconsidered as a feasibility
target — margins are tighter than the prior easy-cluster numbers
implied.

See also: `docs/results/2026-05-21_bicc_pair_drop_audit.md` for the
no-drop audit; `docs/results/2026-05-22_aa_cluster_algorithm_validation_results.md`
for the algorithm-switch decision and the validation experiments it
rests on.

---

## 10. Regenerating everything

The producer/consumer chain, end to end:

```bash
# 1. (nt-only) Build cds_final.parquet — once per data version.
python src/preprocess/extract_cds_dna.py --config_bundle flu_ha_na

# 2. Per-function clustering sweep — once per (alphabet, data version).
#    Writes <out_root>/{fasta,id<NN>}/, redundancy_stats.csv, runtime.json,
#    redundancy_summary.md (alongside the stats CSV; not under docs/).
#    The merge-fix in seq_redundancy_per_function.py preserves prior
#    threshold rows on re-run, so subset reruns don't wipe the CSV.
python -m src.analysis.seq_redundancy_per_function \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --out_root      data/processed/flu/July_2025/clusters_aa \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.80 \
    --threads 8

python -m src.analysis.seq_redundancy_per_function \
    --cds_final  data/processed/flu/July_2025/cds_final.parquet \
    --out_root   data/processed/flu/July_2025/clusters_nt \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85 0.80 \
    --algorithm  linclust \
    --threads    8

# 3. Bipartite-component feasibility per schema pair × alphabet.
#    --out_csv defaults to
#    results/flu/July_2025/runs/cluster_disjoint_feasibility/feasibility_<pair>_<alphabet>.csv
python -m src.analysis.cluster_disjoint_feasibility \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_aa \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.80

python -m src.analysis.cluster_disjoint_feasibility \
    --cds_final     data/processed/flu/July_2025/cds_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_nt \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 0.85 0.80

# (repeat the two feasibility calls for PB2/PB1)

# 4. Consolidated structural summary (CSVs + plots).
python -m src.analysis.cluster_analysis_summary
```

Outputs land under
`results/flu/July_2025/runs/cluster_analysis/`:

```
cluster_summary.csv                       — per (function, alphabet, threshold)
sequence_length_summary.csv               — per (function, alphabet)
mutations_tolerated_table.csv             — concrete max-mismatches table
unique_sequence_retention.png             — Plot A (§4)
cluster_counts_vs_threshold.png           — Plot B (§6)
bipartite_largest_pct_vs_threshold.png    — Plot C (§9)
```

---

## 11. See also

- `docs/methods/leakage_definitions.md` § "Routing equivalence and
  mmseqs argument semantics" — formal routing semantics (aa
  cluster_id100 ≈ seq_disjoint hash_key=seq; nt cluster_id100 ≠
  seq_disjoint hash_key=dna), flag-by-flag argument table, easy-cluster
  vs easy-linclust trade-off.
- `docs/methods/preprocess.md` — Stage 1 (protein/genome) + the
  "Why two output files" rationale used by Stage 1.5.
- `docs/methods/kmer_features.md` — the separate consumer of nt
  sequences via k-mer features (different scope, different cache).
- `docs/results/2026-05-15_cluster_disjoint_nt_results.md` — the
  model-side experiment (LGBM + 1-NN at each routing).
- `data/processed/flu/July_2025/clusters_aa/redundancy_summary.md` (aa) and
  `data/processed/flu/July_2025/clusters_nt/redundancy_summary.md` (nt) —
  autogen per-threshold cluster-size tables for all 8 majors.
- `results/flu/July_2025/runs/cluster_disjoint_feasibility/feasibility_{ha_na,pb2_pb1}_aa.csv`
  (aa) and
  `results/flu/July_2025/runs/cluster_disjoint_feasibility/feasibility_{ha_na,pb2_pb1}_nt.csv`
  (nt) — raw bipartite-CC feasibility numbers per pair × alphabet ×
  threshold.
- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` — the
  original plan that motivated this machinery.
