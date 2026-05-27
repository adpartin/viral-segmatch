# Clustering overview (aa and nt) for cluster_disjoint splits

**Purpose.** This doc explains how mmseqs2 clustering is used to
construct train / val / test splits with **controlled, quantifiable
separation between splits** — and what that separation does and doesn't
guarantee. The treatment is **single-slot (constrained-slot)
cluster_disjoint** throughout (the conceptual core lives in §9); the
bilateral cluster_disjoint mode's feasibility on a real corpus lives in
§10. mmseqs2 internals (§2–§3) are covered only to the extent they bear
on the separation argument.

**Doc style conventions.** Data-table subsections follow the pattern
**Source → Columns → Table → Takeaways → (Future direction)** where
applicable. Dereep reasoning lives in "worked example" subsections; parent sections stay tight.

For experimental findings on Flu A see
`docs/results/2026-05-15_cluster_disjoint_nt_results.md` (model training
results) and the two sequence-redundancy autogen docs at
`data/processed/flu/July_2025/clusters_{aa,nt}/redundancy_summary.md`.
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
   `(HA_c1, NA_c5)`. Negative pairs are constructed within each
   split *after* the routing decision; the structures that drive
   routing are built from positives only.
3. **Define the atom — the indivisible routing unit.** Two routing
   modes are implemented (§7.1):
   - **Constrained-slot** (`single_slot='a'` or `'b'`, conceptual
     deep dive in §9): atom = one cluster's pair-set on the
     constrained slot.
   - **Bilateral** (`single_slot=null`, default; framework + Flu A
     feasibility in §10): atom = a bipartite connected component on
     the (slot A, slot B) cluster graph.
4. **Bin-pack atoms into 80/10/10** via LPT-greedy (§7.2).

§7 walks through the routing mechanism shared across modes (atoms,
LPT-greedy, the four implemented routings). The single-sequence
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

## 5. Maximum residue mismatches admitted per cluster

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

**Source.** `src/analysis/cluster_analysis_summary.py`;
`results/flu/July_2025/runs/cluster_analysis/cluster_summary.csv`
(raw values for all subsections below). The threshold sweep covers
{1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90,
0.85, 0.80}. Plots:
`cluster_counts_vs_threshold.png`, `bipartite_largest_pct_vs_threshold.png`.

### 6.1 Per-function n_clusters across thresholds

Columns:
- `id###`: number of mmseqs clusters for that function at threshold
  `t = ###/100`. Lower threshold → more sequences cluster together →
  fewer clusters.

**Amino acids (aa).**

| Segment | Function | id100 | id099 | id098 | id097 | id096 | id095 | id094 | id093 | id092 | id091 | id090 | id085 | id080 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 33,601 | 18,354 | 10,035 |  7,634 |  6,755 |  6,491 |  3,590 |  1,085 |   112 |    43 | **24** |  2 |   2 |
| 2 | PB1 | 30,822 | 17,209 | 11,859 |  9,266 |  7,384 |  2,033 |    930 |    238 |    42 |    19 |  **6** |  2 |   1 |
| 3 | PA  | 34,162 | 18,520 | 12,758 | 10,906 |  8,677 |  8,002 |    487 |     72 |    30 |    41 | **17** |  2 |   2 |
| 4 | HA  | 41,760 | 22,679 | 14,934 | 11,459 |  8,940 |  7,578 |  2,368 |  1,580 |  1,230 |  1,108 |   910 |  407 | 176 |
| 5 | NP  | 17,533 | 10,483 |  5,038 |  1,750 |    613 |    526 |    149 |    706 |     39 |    20 | **29** |  2 |   2 |
| 6 | NA  | 18,753 |  9,369 |  6,909 |  4,707 |  3,107 |  2,134 |  1,957 |  1,717 |    757 |    791 | 1,077 |  127 |  73 |
| 7 | M1  |  4,712 |  1,764 |  1,033 |  1,003 |    708 |    129 |    269 |    237 |     31 |    24 | **24** |  10 |  3 |
| 8 | NS1 | 22,131 | 13,508 |  9,109 |  6,405 |  4,306 |  3,458 |  1,971 |  1,258 |    990 |    943 |   786 |  196 | 174 |

**Nucleotides (nt).**

| Segment | Function | id100 | id099 | id098 | id097 | id096 | id095 | id094 | id093 | id092 | id091 | id090 | id085 | id080 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 66,475 | 11,484 |  4,562 |  2,968 |  1,353 |    954 |    552 |    398 |    316 |    509 |    180 |  24 |   5 |
| 2 | PB1 | 66,138 | 14,990 |  6,276 |  2,548 |  1,307 |    742 |    472 |    346 |    242 |    179 |    121 |  18 |  10 |
| 3 | PA  | 64,406 | 11,184 |  4,356 |  2,214 |  1,071 |    719 |    338 |    216 |    129 |     84 |     47 |  13 |   3 |
| 4 | HA  | 64,526 | 12,150 |  6,412 |  3,444 |  1,843 |  1,277 |    903 |    595 |    472 |    403 |    275 | 141 |  85 |
| 5 | NP  | 52,097 | 11,627 |  4,754 |  2,559 |  1,427 |  1,182 |    786 |    542 |    455 |    298 |     66 |  12 |   3 |
| 6 | NA  | 57,987 | 12,092 |  4,840 |  2,611 |  1,662 |  1,108 |    905 |    599 |    441 |    419 |    250 | 134 |  91 |
| 7 | M1  | 31,974 | 10,227 |  4,239 |  1,420 |  1,101 |    619 |    263 |    119 |    154 |     80 |     48 |   5 |   2 |
| 8 | NS1 | 37,458 | 12,012 |  4,133 |  1,988 |  1,245 |    800 |    419 |    299 |    326 |    179 |    196 |  91 |   8 |

**Takeaways.**

- **Cluster counts bound flexibility; threshold `t` bounds separation.**
  More clusters at threshold `t` means more atoms available to the
  routing's bin-packer (§7.2) — that's the **flexibility** lever. The
  **separation strength** of the resulting splits is governed by `t`
  itself, not the count — see §9.1 for the cross-cluster gap argument.
- **Two trajectories (deferred-cliff vs gradual).** Segments 1/2/3/5/7
  retain moderate counts through id095, then drop sharply across id094
  → id091 (e.g., PB2 aa: 6,491 → 3,590 → 1,085 → 112 → 43 — five distinct
  cliff steps). Segments 4/6/8 decline more smoothly across the whole sweep
  and still hold 73 (NA), 174 (NS1), and 176 (HA) aa clusters at id080.
- **The conserved-protein cliff has a center (not just a span).**
  Each polymerase subunit and matrix protein has its own "tipping
  idXX" between id095 and id091, but no
  single threshold step applies to all five conserved functions. PB2 aa drops most at id093 → id092
  (1,085 → 112, −90%); PB1 aa at id094 → id093 (930 → 238, −74%);
  PA aa at id095 → id094 (8,002 → 487, −94% — the steepest 1-pp
  drop in the table); M1 aa at id093 → id092 (237 → 31, −87%).
- **NA is the most-gradual outlier.** NA aa stays close to 1,000 clusters across the entire
  id095..id090 stretch — its id095 → id090 drop is only 2,134 →
  1,077 (−50%, much smaller than HA's 7,578 → 910 = −88%).
- **Easy-linclust is generally monotone, but not consistently.** At
  1-pp threshold steps small non-monotone bumps appear:
  NP aa goes 526 → **149** → 706;
  NA aa goes 1,717 → **757** → **791** → 1,077;
  M1 aa goes 129 → **269** → 237.
  Easy-linclust single-pass seed selection produces threshold-dependent
  cluster topology, so a tighter threshold can occasionally yield more
  clusters when seeds change. The coarse direction (id100 → id080) is
  always monotone-decreasing; the 1-pp resolution is not.
- **nt at id100 always exceeds aa at id100** (synonymous variants
  split into distinct nt singletons). The aa-vs-nt relationship
  inverts at id099 and id098 on most functions — see §6.3 worked
  example.

### 6.2 Worked example: aa vs nt non-nesting at id099

**Source.** §6.1 n_clusters table (the observation); §5
mutations_tolerated_table.csv (residue-tolerance asymmetry);
`src/analysis/aa_nt_cluster_crosstab.py` +
`docs/results/2026-05-22_aa_vs_nt_cluster_mechanism.md` (cross-tab
finding).

§6.1's Takeaways note that nt has more clusters than aa at id100 on
every function, but the relationship inverts at id099 and id098 on
most functions. The inversion shapes how alphabet choice affects
cluster_disjoint splits and is worth understanding.

**The inversion (id099, from §6.1).** nt has *fewer* clusters than aa
on five of eight functions:

| Function | aa clusters | nt clusters | nt / aa ratio |
|---|---:|---:|---:|
| PB2 | 18,354 | 11,484 | 0.63 |
| PB1 | 17,209 | 14,990 | 0.87 |
| PA  | 18,520 | 11,184 | 0.60 |
| HA  | 22,679 | 12,150 | 0.54 |
| NP  | 10,483 | 11,627 | 1.11 |
| NA  |  9,369 | 12,092 | 1.29 |
| M1  |  1,764 | 10,227 | 5.80 |
| NS1 | 13,508 | 12,012 | 0.89 |

Five functions have ratio < 1 (nt has fewer clusters). M1 is the
largest outlier in the other direction (5.8× nt excess); NP, NA, NS1
sit near 1.

**Empirical finding — aa and nt clusterings are not nested.**
Cross-tab analysis of (aa cluster, nt cluster) co-membership on id099
sequences (`src/analysis/aa_nt_cluster_crosstab.py`; results in
`docs/results/2026-05-22_aa_vs_nt_cluster_mechanism.md`) shows: on
every function, aa and nt cluster boundaries disagree in BOTH
directions — each alphabet groups some sequences that the other
separates. The net cluster count is the balance of those two
opposing directions, and the balance varies per function.

**Residue-tolerance asymmetry (id099, from §5).** id099 admits more
residue mismatches per cluster on nt than on aa, because the threshold
`t = 0.99` is computed against the sequence's own length and the CDS
is roughly 3× the protein length. From §5's
`mutations_tolerated_table.csv` at id099: PB2 admits 22 nt vs 7 aa
mismatches; HA admits 17 vs 5; M1 admits 7 vs 2. This is the
geometric consequence of how `--min-seq-id` is computed (§3.1).

**Takeaways.**

- The intuition "nt has more clusters than aa because of synonymous
  codons" holds at id100 but does not generalize. At id099 and id098
  most functions show the opposite direction.
- aa and nt cluster boundaries disagree in both directions; net
  cluster count is a per-function balance, not nested containment.
- The deep-dive walkthrough (per-function decomposition) is in
  `docs/results/2026-05-22_aa_vs_nt_cluster_mechanism.md`.

### 6.3 Worked example: NS1 vs M1 — real diversity vs length-variation artifact

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

### 6.4 Largest cluster as % of corpus

**Source.** `src/analysis/cluster_analysis_summary.py`;
`cluster_summary.csv` column: `largest_cluster / n_sequences × 100`.

Layout:
- **Columns** (`id###`): percentage of the function's sequences that fall
  into its single largest cluster at threshold `t = ###/100`. 100% means
  one cluster contains every sequence of that function.
- **Rows**: grouped by trajectory from §6.1 — deferred-cliff functions
  (PB2, PB1, PA, NP, M1) on top; gradual functions (HA, NA, NS1) on bottom.

| Segment | Function | id100 | id099 | id098 | id097 | id096 | id095 | id094 | id093 | id092 | id091 | id090 | id085 | id080 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 0.0% | 12.7% | 12.6% | 13.7% | 15.6% | 15.6% | 15.9% | 38.3% | 64.4% | 77.6% |  95.9% | **100%** | **100%** |
| 2 | PB1 | 0.1% | 12.9% | 12.0% | 18.5% | 25.9% | 72.4% | 90.9% | 98.9% | 99.8% | 99.9% | **100%** | **100%** | **100%** |
| 3 | PA  | 0.0% |  7.3% |  8.9% |  8.9% |  9.0% |  9.0% | 56.5% | 65.3% | 75.3% | 96.3% |  98.1% | **100%** | **100%** |
| 5 | NP  | 0.1% |  7.5% | 13.1% | 28.2% | 41.4% | 46.3% | 51.2% | 67.5% | 84.0% | 85.0% |  99.8% | **100%** | **100%** |
| 7 | M1  | 0.2% | 10.8% | 17.8% | 23.1% | 41.7% | 56.7% | 84.1% | 93.3% | 99.1% | 99.2% |  99.2% |  99.5% |  99.9% |
| 4 | HA  | 0.0% |  5.0% |  6.7% |  9.6% |  9.7% | 11.8% | 12.1% | 21.0% | 21.9% | 22.5% |  22.8% |  24.3% |  33.4% |
| 6 | NA  | 6.9% |  8.7% |  8.9% |  8.8% | 13.2% | 13.2% | 17.9% | 14.0% | 18.0% | 17.7% |  17.9% |  32.4% |  37.7% |
| 8 | NS1 | 0.0% |  3.9% |  4.8% |  9.4% | 11.7% | 16.3% | 19.0% | 19.3% | 19.3% | 19.5% |  21.1% |  29.5% |  52.4% |

**Takeaways.**

- **The mega-cluster forms one step at a time on conserved functions.**
  PB2 aa: 15.6% (id095) → 15.9 → 38.3 → 64.4 → 77.6 → 95.9% (id090)
  — five graded steps between "still-small" and "essentially-100%",
  not a single cliff. PB1 aa shows the earliest growth (72% at id095
  → 91% at id094 → 99% at id093). PA aa takes the steepest single pp
  jump (9.0% at id095 → 56.5% at id094). M1 jumps from 57% to 84% in
  one step at id094 and reaches 99% by id092.
- **HA / NA / NS1 stay flat across id095..id090.** HA largest is
  11.8% at id095 and only 22.8% at id090 — never forms a mega-
  cluster. NA largest stays in the 13–18% band across the same
  stretch. NS1 climbs slowly to 21% at id090. By the same threshold
  the conserved functions are ≥96%. This is the same biology that
  §6.1 surfaces from the cluster-count direction.
- **Conserved-protein largest jumps localize differently than
  n_clusters drops.** PA's n_clusters drops most at id095 → id094
  (8,002 → 487), and its largest_pct jumps in the same step
  (9.0% → 56.5%) — coherent. But PB2's biggest n_clusters drop is
  id093 → id092 (1,085 → 112), while its largest jumps most at id094
  → id093 (15.9% → 38.3%) and id093 → id092 (38.3% → 64.4%). The
  two signals (count and largest) track each other coarsely but
  not at 1-pp threshold steps.
- **NA's id100 anomaly: 6.9% already.** NA is the only function with
  a sub-id100 entry visibly above zero, because of the stalk-length
  collapse mechanism (§4 NA note). NA's id100 cluster pools ~7% of
  the corpus through length-variant absorption before any
  sequence-similarity clustering takes effect.

**Note.** This is the per-FUNCTION largest cluster fraction (one
number per function per threshold). §10 reports the per-PAIR largest
bipartite-COMPONENT fraction — a different metric used to characterize
the bilateral cluster_disjoint routing's feasibility.

---

## 7. From a per-function clustering to a train/val/test partition

### 7.1 Two ways to convert per-function clusters into a pair partition

mmseqs2 clusters sequences within one function (HA clusters, NA
clusters, ...). But the prediction task is on *pairs* of two functions
(slot A = HA, slot B = NA). What needs to be partitioned is the set
of (slot_A_cluster, slot_B_cluster) pairs.

A naive "ensure no slot A cluster appears in both train and test"
isn't enough on its own. An isolate carries both proteins, so two
pairs that share *either* a slot A cluster *or* a slot B cluster are
linked through that isolate. Pull on one pair and others come with it.

Two routing modes resolve this; both are implemented (§7.3) under
`split_strategy.mode: cluster_disjoint`:

- **Constrained-slot cluster_disjoint** (`single_slot: 'a'` or `'b'`).
  Constrain ONE slot's clusters only. The atom is one cluster's
  pair-set on the constrained slot; the unconstrained slot is left
  free. What this enforces, what it doesn't, and the empirical
  consequences on Flu A are the subject of §9.
- **Bilateral cluster_disjoint** (`single_slot: null`, default).
  Constrain BOTH slots' clusters. The atom is a bipartite connected
  component on the (slot A, slot B) cluster graph (a CC can span
  multiple clusters on each side, linked through shared isolate
  pairs). The mechanism and empirical feasibility on Flu A are the
  subject of §10.

Both modes share the same bin-packer (§7.2) — they differ in what
they treat as an atom.

### 7.2 80/10/10 by LPT-greedy

**Atom — the indivisible routing unit.** The router treats the corpus
as a set of "atoms" that cannot be split across train / val / test.
What an atom is depends on the routing mode:

- **Constrained-slot cluster_disjoint** (`single_slot='a'` or `'b'`,
  one slot's clusters disjoint): atom = **the pair-set of one cluster
  on the constrained slot**. All pairs whose constrained-slot sequence
  belongs to cluster K stay together; the unconstrained slot's
  partition is whatever those pairs happen to bring along (see §9 for
  the separation consequences). One atom per cluster on the constrained
  slot.
- **Bilateral cluster_disjoint** (both slots' clusters disjoint): atom
  = a **bipartite connected component** on the (slot A, slot B) cluster
  graph (§10). A CC can span multiple clusters on each slot, linked
  through shared isolate pairs. One atom per CC.

Whichever atom definition applies, the router
(`src/datasets/_split_helpers.py::cluster_disjoint_route_pos_df`)
sorts atoms by size descending and greedily assigns each to the
split with the largest current deficit relative to its target:

```
targets:      train = 80%   val = 10%   test = 10%
biggest atom first → goes to the bin with the largest open quota →
   typically train first, then val and test fill from medium-sized atoms.
```

This is the "longest-processing-time-first" (LPT) heuristic for
bin packing. It's optimal up to the largest atom: if the largest
atom exceeds the largest bin's target, the routing is structurally
infeasible at 80/10/10. The largest-atom fraction is therefore the
quantity to check before committing to a (pair, alphabet, threshold)
configuration.

LPT-greedy is closely related to sklearn's `GroupShuffleSplit` and
`GroupKFold` — all three respect "atoms can't be split across splits"
(sklearn calls atoms "groups"). The difference is in the assignment
step: `GroupShuffleSplit` assigns atoms randomly, which can land a
30%-of-corpus atom in train and miss the 80/10/10 target by a wide
margin. LPT-greedy assigns atoms biggest-first to the most empty bin,
matching the target ratios as closely as the largest-atom constraint
allows.

Unlike `GroupShuffleSplit(n_splits=N)`, which yields N independent
random group-respecting shuffles for variance estimation, LPT-greedy
as implemented here is fully deterministic given the atom set and
target ratios — re-running with a different `master_seed` reproduces
the same partition. Per-fold variance under cluster_disjoint is not
available from the current single-shot implementation; the
`GroupKFold`-based multi-fold path noted in BACKLOG.md "Single-slot
routing follow-ups" #3 would unlock it.

**The router never drops pairs.** When the largest atom exceeds the
80% train quota, LPT-greedy still places the whole atom in train —
train just overflows its target. The router does not split atoms and
does not discard boundary pairs; "infeasible" in this doc means
"the achieved 80/10/10 ratios drift", not "pairs are lost". The audit
JSON's `pairs_dropped_in_routing` and `pairs_dropped_in_cluster_join`
are always 0 in practice (verified across HA/NA aa id099, HA/NA aa
id095, PB2/PB1 aa id099, HA/NA nt id099, PB2/PB1 nt id099 — see
`docs/results/2026-05-21_bicc_pair_drop_audit.md`). The operational
consequence is that at sub-feasibility thresholds, val and test
starve: HA/NA aa id095 produces a 98.5 / 0.76 / 0.76 split rather
than 80 / 10 / 10, even though every pair is routed somewhere.

Which function pairs are splittable at which thresholds follows
directly from §6.1's collapse trajectory: HA/NA (gradual on both
slots) remains splittable longest; polymerase pairs (PB2/PB1, both
deferred-cliff) cliff together at id090. The empirical feasibility
ceiling for each (pair, alphabet, threshold) combination is in §10.

### 7.3 The four implemented routings — quick reference

| mode + alphabet | slot key | two pairs share an atom iff … |
|---|---|---|
| `seq_disjoint` `hash_key=seq` (default) | `seq_hash = md5(prot_seq)` exact | identical protein on a slot |
| `seq_disjoint` `hash_key=dna` | `dna_hash = md5(contig.dna)` exact | identical full contig (UTR + CDS + intron + UTR) on a slot |
| `cluster_disjoint` `cluster_alphabet=aa` | mmseqs2 aa cluster id at chosen threshold | aa-similar protein on a slot |
| `cluster_disjoint` `cluster_alphabet=nt` | mmseqs2 nt cluster id at chosen threshold, keyed on `cds_dna_hash` | nt-similar CDS DNA on a slot (UTRs and introns excluded) |

`cluster_disjoint` has the additional `single_slot` knob
(§7.1) that selects between constrained-slot and bilateral atom
definitions. Deep dive on equivalences and non-equivalences across
the four (mode, alphabet) cells:
`docs/methods/leakage_definitions.md` § "Routing equivalence and
mmseqs argument semantics".

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
in §4 / §5 / §6 / §10), the post-hoc script is
`src/analysis/cluster_analysis_summary.py`, which reads the
redundancy + feasibility CSVs and emits plots under
`results/flu/{version}/runs/cluster_analysis/`.

---

## 9. Constrained-slot separation: what cluster_disjoint enforces

### 9.1 The construction and bounds

**Source.** §3.1 (`--min-seq-id` definition), §5 (per-function residue
mismatch caps), and the mmseqs2 algorithmic construction.

**Setup.** Under **constrained-slot cluster_disjoint** at identity
threshold `t` on slot X (e.g., HA-only), the mmseqs clusters covering
slot-X sequences are partitioned across train / val / test: every
cluster lands in exactly one split. By construction, any train slot-X
sequence and any test slot-X sequence live in **different clusters**.

The unconstrained slot (slot Y) carries **no separation constraint at
all** — not cluster_disjoint, not even seq_disjoint. The same physical
slot-Y sequence can appear in train and test (and val) if it pairs
with multiple slot-X sequences whose clusters land in different splits.
Whether slot-Y sequences actually leak across splits depends on the
corpus's biological coupling between the two slots: tight coupling
(e.g., HA cluster ≈ NA subtype on Flu A HA-NA) drags slot-Y along with
slot-X; weak coupling (e.g., PB2-PB1) leaves slot-Y free to leak.

This is the group-respecting property that
`sklearn.model_selection.GroupShuffleSplit` and `GroupKFold` also
provide, with one extra feature: **mmseqs clustering at threshold `t`
gives the groups a sequence-level meaning**. `GroupShuffleSplit` and
`GroupKFold` treat group labels as opaque IDs; cluster_disjoint's
groups carry the residue-identity tolerance baked into `t`. That
promotion — from opaque ID to similarity-controlled tolerance — is the
load-bearing claim of the approach. (The orthogonal question of how
clusters are then assigned to splits — random vs LPT-greedy bin-packing
— is covered in §7.2.)

**Within-cluster ceiling (hard, by construction).** mmseqs at threshold
`t` guarantees that every sequence in one cluster has identity ≥ `t`
to the cluster representative **over the aligned region (≥80% mutual
coverage, §3.2)**. For an aligned length `L_a` this caps within-cluster
mismatches at approximately `(1 − t) × L_a` residues. Per-function
caps at each threshold: §5.

**Cross-cluster gap (soft, by construction).** mmseqs would have
merged two sequences into one cluster if their identity were ≥ `t`.
Therefore between any two sequences in different clusters, mismatch
count is **typically greater than `(1 − t) × L_a` residues** over the
aligned region — otherwise the clustering would have grouped them.
Under constrained-slot cluster_disjoint at threshold `t`, this becomes
the **typical lower bound on the residue gap between train and test
on the constrained slot**:

> train slot-X sequence vs test slot-X sequence: typically
> > `(1 − t) × L_a` mismatches.

Choosing `t` is choosing the residue-level difficulty of the
generalization test. Lower `t` → wider tolerance → wider enforced gap
→ the model is evaluated on sequences structurally more different from
its training data.

"Typically" — not "always". The bound is **soft**, not deterministic.
§9.2 covers the failure modes; §9.3 covers the empirical leakage
gauge.

**Worked example — HA on Flu A** (aligned length ≈ 567 aa under the
coverage rule; exact per-threshold caps in §5):

| Threshold | Within-cluster max mismatches | Cross-cluster typical mismatches | What constrained-slot HA-only enforces between train and test |
|---:|---:|---:|---|
| id100 | 0 | 0 (byte-identical over aligned region) | Trivial separation — same sequences allowed in train and test |
| id095 | ~28 | more than ~28 | Train/test HAs typically differ by 28+ residues |
| id094 | ~34 | more than ~34 | Wider gap; cliff edge for HA-only feasibility (§10.3) |
| id090 | ~57 | more than ~57 | Largest enforced gap, but HA-only routing degenerates here (§10.3) |

Many of those 28–57 residues sit in functional regions of HA — the
receptor-binding domain (residues ~117–265) and the major antigenic
sites (Sa, Sb, Ca, Cb around residues ~140–225). The model has to
generalize across that residue-level gap to score test pairs correctly.

**Takeaways.**

- `t` is the design knob that controls the train↔test residue gap on
  the constrained slot. Choosing `t` is choosing the difficulty of the
  generalization test.
- Within-cluster identity is bounded **above** by mmseqs's
  construction; cross-cluster gap is bounded **below** typically but
  not strictly. The two bounds together yield the residue-gap
  argument.
- This is what cluster_disjoint buys over `GroupShuffleSplit` /
  `GroupKFold`: the groups carry a controlled-similarity meaning, not
  opaque labels.
- The unconstrained slot enforces *nothing*. Whether it ends up
  well-separated is a property of the corpus, not the routing.
- The "typically" qualifier on the cross-cluster gap is non-trivial.
  §9.2 covers how it can fail; §9.3 covers how much it actually fails
  on Flu A.

### 9.2 Limitations of cluster-based separation

**Source.** Steinegger & Söding 2018 (Linclust paper, Nat. Commun.
9:2542) for the algorithmic mechanism; §6.1 non-monotone observations
for the topology mechanism; per-dataset `cluster_disjoint_audit.json`
for the unconstrained-slot leakage numbers.

The "typically more than `(1 − t) × L_a` mismatches" bound in §9.1 is
soft. Three independent failure modes break it.

**Failure mode 1 — approximate clustering near the boundary.**
`easy-linclust` is a single-pass, hash-based algorithm rather than a
global pairwise comparison. Each sequence is hashed into `m = 20`
k-mers (default) and aligned only against the longest sequence in each
k-mer group it lands in — not pairwise against all other members. Two
sequences whose true identity exceeds `t` can fail to share any
selected k-mer, or can land against different center sequences and
pass alignment against neither. They end up in different clusters
even though they should have merged. The Steinegger & Söding 2018
paper reports a ~28% miss rate at 50% identity (Fig. 3c). At the
operating thresholds in this doc (id095, id094, id090) the miss rate
is smaller — the higher the identity, the more likely two
near-identical sequences share at least one of their 20 selected
k-mers — but nonzero. Practical consequence: some fraction of
cross-cluster (train, test) pairs at threshold `t` have identity > `t`
over the aligned region, eroding the typical gap.

**Failure mode 2 — non-deterministic cluster topology across nearby
thresholds.** The algorithm depends on which seeds are selected and
which "longest sequence" each k-mer group anchors on, so small changes
in `t` can shift cluster boundaries in non-monotone ways. §6.1
documents this empirically: NP aa drops from 526 clusters at id095 to
149 at id094 then rises to 706 at id093 before resuming the descent.
The same sequence can land in cluster A at id094 and cluster B at
id093 with different cluster co-members. Down-stream, the *identity*
of which sequences a given cluster contains is sensitive to `t` near
boundary regions, not just the cluster count.

**Failure mode 3 — the unconstrained slot.** Constrained-slot
cluster_disjoint enforces nothing on slot Y (§9.1). When biological
coupling between slots X and Y is strong (HA cluster ≈ NA subtype on
Flu A HA-NA, Cramér's V ≈ 0.85–0.98), the unconstrained slot follows
the constrained slot's partition by proxy and the soft gap on slot X
extends to slot Y as well. When coupling is weak (PB2 ↔ PB1 cluster
coupling V ≈ 0.41–0.76 on Flu A), the unconstrained slot leaks freely
— the same physical slot-Y sequence appears in train and test pairs
because slot-Y partners of multiple slot-X clusters happen to coincide.

The empirical residual leakage on the unconstrained slot is captured
per-dataset in `cluster_disjoint_audit.json` (full per-split-pair
breakdown) and surfaced at the top level of `dataset_stats.json` under
`slot_leakage_summary`. For the published Flu A sweeps at the
id095–id100 range, unique seq_hash leakage on the unconstrained slot
is:

| Routing | id100 | id099 | id098 | id097 | id096 | id095 |
|---|---:|---:|---:|---:|---:|---:|
| HA-NA HA-only (slot b = NA)     | 7.5% | 5.7% | 4.7% | 4.2% | 3.6% | 3.4% |
| PB2-PB1 PB2-only (slot b = PB1) | 7.2% | 5.8% | 5.5% | 5.3% | 5.2% | 5.2% |

Both pairs show the same shape: leakage drops as `t` drops, because
looser clusters absorb more diverse partners and the same
unconstrained-slot sequence becomes less likely to span multiple
constrained-slot clusters. HA-NA's drop is steeper than PB2-PB1's
(3.4% vs 5.2% floor at id095) — consistent with HA-NA's tighter
HA↔NA coupling absorbing the unconstrained slot into the constrained
slot's partition more cleanly. Detail on the coupling per pair:
`docs/results/2026-05-26_pb2_pb1_PB2only_idXX_sweep.md` § "Cramér's V
coupling pre-check".

**Takeaways.**

- The cross-cluster gap is a **statistical tendency** of the algorithm,
  not a guarantee. A small but nonzero fraction of (train, test) pairs
  at threshold `t` have identity exceeding `t`.
- Cluster topology at nearby `t` values is not stable. The mapping
  "which sequences live in cluster K" changes across small `t` shifts,
  separately from the cluster count.
- The unconstrained slot can leak. On Flu A the seq_hash leakage on
  the unconstrained side is 3.4–7.5% across the published sweep
  ranges; the per-corpus number is in `slot_leakage_summary` of each
  dataset's `dataset_stats.json`.
- These three failure modes are independent — they compound rather
  than cancel. The empirical residual leakage on a real corpus has to
  be measured (§9.3), not derived.

### 9.3 1-NN as the empirical leakage gauge

**Source.** The 1-NN cosine margin baseline
(`src/models/baselines/knn1_margin.py`) trained alongside the MLP on
each cluster_disjoint dataset. Sweep numbers from
`docs/results/2026-05-24_single_slot_HAonly_idXX_sweep.md` (HA-NA
HA-only) and `docs/results/2026-05-26_pb2_pb1_PB2only_idXX_sweep.md`
(PB2-PB1 PB2-only), both at aa k=3 features.

The audit in §9.2 (Failure mode 3) gives the **direct** sequence-level
leakage measurement: how many unconstrained-slot sequences appear
across multiple splits. The 1-NN diagnostic in this subsection gives
the **complementary** measurement: of the residual leakage that
remains (border-similarity on the constrained slot plus
unconstrained-slot leakage), how much can a lookup-style baseline
exploit?

**The logic.** A 1-nearest-neighbor classifier predicts each test
pair's label from its closest training pair (cosine distance on the
pair feature vector). Under constrained-slot cluster_disjoint at
threshold `t`, every test pair has a "wider gap" to its nearest
training pair than under seq_disjoint, *to the extent the cluster gap
is hard*. If the gap is strict, 1-NN's prediction-by-nearest-pair
degrades as `t` drops. If the gap is soft — i.e., the failure modes in
§9.2 are active — 1-NN can still find close training neighbors and
stay competitive.

So **1-NN performance trajectory across `t` is the empirical upper
bound on residual leakage**. A 1-NN F1 that drops sharply with `t`
indicates the cluster gap is breaking lookup-style baselines. A 1-NN
F1 that stays flat indicates residual border-similarity leakage that
lookup can still exploit.

**Empirical pattern on Flu A** (1-NN cosine margin F1, single seed):

| Pair | Direction | id100 | id099 | id098 | id097 | id096 | id095 | Drop id100 → id095 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HA-NA   | HA-only  | 0.958 | 0.940 | 0.931 | 0.930 | 0.920 | 0.911 | **-4.7 pp** |
| PB2-PB1 | PB2-only | 0.929 | 0.895 | 0.921 | 0.900 | 0.913 | 0.907 | **-2.2 pp** |

Two readings:

- The drop is **real but small**. id100 → id095 spans
  `(1 − t) × L_a` ≈ 0 → ~28 mismatch residues on HA, similar on PB2.
  That's a non-trivial residue-level gap, and 1-NN loses only 2–5 pp
  F1 across it. Border-similarity leakage persists at id095 on both
  pairs.
- The drop is **steeper on HA-NA than on PB2-PB1** (4.7 vs 2.2 pp).
  Consistent with the §9.2 unconstrained-slot framing: HA-NA's tighter
  HA↔NA coupling drags NA along with HA across the partition;
  PB2-PB1's weaker coupling leaves PB1 free to leak, blunting 1-NN's
  degradation.

**1-NN vs MLP cross-comparison** is informative on its own:

- HA-NA HA-only: MLP > 1-NN at every threshold (MLP id095 F1 = 0.918
  vs 1-NN 0.911). The trained model edges out lookup — the cluster
  gap is hard enough that learning beats memorization slightly.
- PB2-PB1 PB2-only: **1-NN > MLP at every threshold** (1-NN id095 F1
  = 0.907 vs MLP 0.897). Lookup wins — the trained model is not
  extracting signal beyond what nearest-neighbor calibration
  provides. Strong signal that residual leakage on PB2-PB1 is large
  enough to make lookup a competitive ceiling.

**Takeaways.**

- The 1-NN trajectory is the load-bearing empirical diagnostic: it
  converts the soft separation guarantee of §9.1 into a measurable
  per-corpus, per-routing number, complementary to the direct
  sequence-level audit in §9.2.
- On Flu A, 1-NN stays competitive through id094 (the cliff) on both
  pairs tested — cluster_disjoint at this corpus's feasibility range
  does not break lookup-style baselines.
- 1-NN beating MLP is a stronger leakage signal than 1-NN merely
  staying competitive. PB2-PB1 PB2-only shows it; HA-NA HA-only does
  not.
- A model evaluation that wants 1-NN to fail (i.e., wants a
  generalization test the cluster gap alone can't break) needs the
  next lever — see §9.4.

### 9.4 Forward pointer: differential lookup degradation via hard negatives

A candidate next lever for breaking the residual lookup signal in
§9.3 without changing the routing: re-weight the negative sampler
toward harder regimes via `dataset.negative_sampling.regime_targets`.
Pulling more negatives from `host_subtype_year` (metadata matches
positives closely) creates training negatives that 1-NN will mis-rank
against test positives. MLP and LGBM can in principle decorrelate
from metadata-driven similarity; 1-NN cannot. Expected signal:
**differential degradation** — 1-NN drops more than MLP / LGBM —
directly demonstrating that lookup is being broken without breaking
learning. Not yet measured.

---

## 10. Feasibility on Flu A

§9 covers constrained-slot cluster_disjoint conceptually; this
section covers the bilateral cluster_disjoint mode and its empirical
feasibility on the Flu A corpus. The §7.2 question — "is the largest
atom small enough for 80/10/10?" — answered per (schema pair,
alphabet, threshold).

### 10.1 The bipartite-CC framework (bilateral routing)

Under bilateral cluster_disjoint, both slots' clusters are
constrained. The atom is a **bipartite connected component** on the
(slot A, slot B) cluster graph — built from positive pairs, with an
edge `HA_ck — NA_cm` iff some pair has that combination:

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

All pairs inside one CC must land in the same split — otherwise a
cluster on one side would appear in both train and test, defeating
the routing's purpose. The per-function collapse trajectory (§6.1)
predicts the resulting bipartite-CC sizes. When either slot's
clusters collapse into one mega-cluster — for example PB2 aa at
id090 has only 24 clusters absorbing 99.6% of the corpus (§6.4) —
the bipartite graph collapses into a single mega-component, and
the routing becomes structurally infeasible (no 80/10/10 split is
possible).

**Naming.** This routing goes by several names across docs, code, and
writeups:
- **bipartite-CC LPT-greedy** — the precise technical name
  (bipartite connected components of `(slot_A_cluster, slot_B_cluster)`,
  bin-packed via Longest-Processing-Time first-fit decreasing into the
  requested split fractions).
- **cluster_disjoint routing** — the user-facing config name
  (`dataset.split_strategy.mode: cluster_disjoint` with the default
  `single_slot: null`).
- **BiCC-Split** — paper / prose shorthand for "Bipartite
  Connected-Component Split", abbreviated **BiCC** or **bicc** in
  inline references.

These name the same algorithm. Committed docs prefer
**bipartite-CC LPT-greedy** on first mention and **bicc** for
subsequent references; `cluster_disjoint` is for code and config;
`BiCC-Split` is for manuscript prose where a memorable name helps.

The algorithm differs from DataSAIL's split heuristic (cluster + ILP)
in two ways: (a) routing operates on bipartite-CCs as atomic units,
never dropping pairs ("CC bin-packing never splits a component",
`_split_helpers.py:267`), where DataSAIL's I2/S2 explicitly drop
pairs that straddle folds; (b) bicc's LPT-greedy is a heuristic that
hits the requested split fractions within ~0.01% on Flu A (memory.md
"seq_disjoint scales to conserved proteins"), where DataSAIL solves
an NP-hard ILP via a heuristic clustering pre-pass.

### 10.2 Bilateral feasibility on Flu A

Source: the four feasibility CSVs at
`results/flu/{version}/runs/cluster_disjoint_feasibility/feasibility_<pair>_<alphabet>.csv`,
generated by `src/analysis/cluster_disjoint_feasibility.py`.
Consolidated plot: `bipartite_largest_pct_vs_threshold.png` from
`cluster_analysis_summary.py`.

Largest bipartite-component fraction (% of deduped pairs):

| Segments | Schema pair | Alphabet | id100 | id099 | id098 | id097 | id096 | id095 | id094 | id093 | id092 | id091 | id090 | id085 | id080 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4/6 | HA/NA   | aa | 49.0 | 79.6 | 88.4 | 92.7 | 95.8 | 97.8 | 98.7 | 99.0 | 99.0 | 99.0 |  99.8 | 100.0 | 100.0 |
| 4/6 | HA/NA   | nt |  1.5 | 69.4 | 91.0 | 95.7 | 97.8 | 98.2 | 98.5 | 98.9 | 98.7 | 99.1 |  99.1 |  99.6 | 100.0 |
| 1/2 | PB2/PB1 | aa | 38.4 | 81.0 | 92.6 | 97.2 | 98.6 | 99.5 | 99.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1/2 | PB2/PB1 | nt |  2.9 | 59.7 | 93.9 | 97.2 | 98.2 | 99.1 | 99.3 | 99.3 | 99.5 | 99.5 |  99.5 | 100.0 | 100.0 |

A cell is structurally feasible for 80/10/10 if the largest CC is ≤80%
(train can fit it cleanly). What "infeasible" means in practice: the
router still places every pair (see §7.2 — `pairs_dropped_*` are
always 0), but train absorbs the mega-CC and val/test drift toward
zero. Reading the table by that frame:

- **id100 (every cell):** feasible. Largest CC is at most 49.0%
  (HA/NA aa — NA's stalk-length absorption on the aa side already
  pools ~7% of pairs into a single CC at id100, see §4 footnote
  and §6.4). Routing has room.
- **id099 (marginal on aa, comfortable on nt):** HA/NA aa at 79.6%
  lands just under the ceiling — bin-packer achievable. PB2/PB1 aa
  at 81.0% is 1 pp over the ceiling — borderline feasible. nt cells
  (HA/NA = 69.4%, PB2/PB1 = 59.7%) sit comfortably below the
  ceiling. Aa is tighter than nt by ~10 pp at id099 on both pairs.
- **id098 (above the ceiling on all four cells):** aa cells are
  88.4% (HA/NA) and 92.6% (PB2/PB1); nt cells are 91.0% and 93.9%.
  All four cross the ceiling but margins are slim enough that id098
  is closer to feasibility than the deeper thresholds — the §6.1
  collapse trajectory and the largest-CC % both predict an
  ~88-93% / 4-6% / 4-6% split.
- **id097 and below (broken everywhere — but builds still run):**
  On a HA/NA aa id095 build the routing produced 98.48 / 0.76 / 0.76
  — val and test got 1,107 pairs each vs 14,597 intended (a 92%
  capacity loss on the held-out splits). Every pair is routed; the
  dataset exists; it just isn't usable for evaluation. Every cell
  with largest CC ≥ 95% behaves the same way.
- **id094..id091 yields no bilateral recovery.** Across id094..id091
  the largest CC creeps (near-)monotonically toward 100% on every
  cell. On aa PB2/PB1 it hits 100.0% by id093 and stays there. On
  nt HA/NA it fluctuates between 98.5% and 99.1%, with a small
  non-monotone bump at id092 (98.7 vs id093's 98.9 — inherited from
  the easy-linclust non-monotonicity §6.1 describes). No threshold
  in this range unlocks bilateral feasibility on any pair/alphabet
  cell.

Largest-CC % is the algorithmic *input* (and the predictor of
feasibility); achieved-train % in any built run is the realised
*output*. When largest CC ≤ 80% the two diverge only at the 4th
decimal. When largest CC > 80% achieved train % tracks largest CC %
closely (the bin-packer can't undo what the mega-CC dictates).

**Interpretation: feasibility ceiling is algorithm-controlled.** §6's
collapse trajectory is corpus-driven by construction (easy-linclust
is the algorithm on both alphabets, see §2.2 and §6.1). The aa-vs-nt
feasibility gap at id099 reflects the alphabet's underlying diversity
structure plus the corpus's metadata-driven bipartite linking, not an
algorithm-alphabet confound. nt clustering does not unlock lower
thresholds via synonymous diversity — even at id098 nt sits at
91-94% — because the *bipartite linking* between HA clusters and NA
clusters is determined by which isolates carry which (HA, NA)
combinations, and Flu A's small set of dominant HxNy subtypes ×
host × year cells links most pairs into one mega-component well
above id098.

See also: `docs/results/2026-05-21_bicc_pair_drop_audit.md` for the
no-drop audit; `docs/results/2026-05-22_aa_cluster_algorithm_validation_results.md`
for the algorithm choice and its validation.

### 10.3 Single-slot cluster_disjoint extends the ceiling

The table above is for **bilateral** cluster_disjoint (both slots'
clusters disjoint between splits). Single-slot routing (only one
slot's clusters constrained — the relevant "atom" is per-cluster pair
count on the constrained slot rather than the bipartite component;
see §7.3) relaxes the constraint and pushes the feasibility ceiling
below the bilateral one.

**Source.** `src/analysis/single_slot_cluster_disjoint_feasibility.py`;
CSVs at `results/.../cluster_disjoint_feasibility/single_slot_feasibility_<pair>_<alphabet>.csv`.
The feasibility rule is **largest_atom ≤ 80% AND second_atom ≤ 20%**
(the bin-packer needs train to absorb the biggest atom AND val/test
to each fit a smaller one).

aa, top-2 atom sizes (% of deduped pairs) at each threshold:

| Pair | Constrained slot | id095 | id094 | id093 | id092 | id091 | id090 | id085 |
|---|---|---|---|---|---|---|---|---|
| HA/NA   | a (HA)  | 13.5 / 11.5 ✓ | 13.8 / 11.5 ✓ | 23.8 / 21.2 ✗ | 24.6 / 21.3 ✗ | 25.1 / 21.4 ✗ | 25.4 / 21.4 ✗ | 27.1 / 21.7 ✗ |
| HA/NA   | b (NA)  | 15.5 / 12.2 ✓ | 21.1 / 12.3 ✓ | 18.4 / 11.2 ✓ | 22.4 / 21.2 ✗ | 22.5 / 12.6 ✓ | 22.7 / 11.2 ✓ | 34.0 / 18.8 ✓ |
| PB2/PB1 | a (PB2) | 15.5 / 13.9 ✓ | 15.7 / 14.4 ✓ | 35.0 / 24.8 ✗ | 57.3 / 25.4 ✗ | 72.6 / 26.4 ✗ | 95.6 / 2.8 ✗ | 100 / 0 ✗ |
| PB2/PB1 | b (PB1) | 78.4 / 4.7 ✓  | 93.1 / 0.8 ✗  | 99.2 / 0.0 ✗ | 99.9 / 0.0 ✗ | 100 / 0 ✗ | 100 / 0 ✗ | 100 / 0 ✗ |

**Readings.**

- **HA-only and PB2-only both cliff at id094 → id093.** Largest atom
  jumps from ~14% to ~24% (HA) / 35% (PB2) and the second atom
  crosses the 20% ceiling. Same threshold step on both pairs despite
  very different cluster-count trajectories in §6.1 — the cliff
  position is a property of the bin-packing rule applied to the
  Flu A corpus's metadata structure, not specific to a single pair.
- **PB1-only cliffs one step earlier (id095 → id094).** PB1 has
  unusually few clusters at id095 (2,033) compared to PB2 (6,491),
  so the largest atom is already at 78.4% at id095 (against the 80%
  ceiling) and crosses it at id094.
- **NA-only stays feasible through id085 with one id092 hole.**
  Largest atom stays in the 15-34% band across the whole sweep;
  second atom is below 13% except at id092 where it spikes to
  21.2%. The id092 hole is inherited from NA's non-monotone cluster
  count (§6.1: NA aa goes 1,957 → 1,717 → **757** → 791 across
  id094..id091). NA's stalk-length-driven flat cluster-size
  distribution keeps it routable that far down — the only Flu A
  direction with this property.
- **Bilateral and single-slot ceilings are decoupled.** Bilateral
  is broken from id098 down on every pair/alphabet. Single-slot
  HA-only and PB2-only stay feasible through id094 — four thresholds
  further than the bilateral ceiling allows. The full single-slot
  range usable for HA-only / PB2-only is id100..id094; for NA-only
  it extends to id085 with the id092 hole noted above; for PB1-only
  it stops at id095.

---

## 11. Regenerating everything

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
bipartite_largest_pct_vs_threshold.png    — Plot C (§10)
```

---

## 12. See also

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
