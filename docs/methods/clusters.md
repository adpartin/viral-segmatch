# Clustering mechanics and Flu A corpus structure

**Purpose.** This doc covers mmseqs2 clustering and Flu A corpus
structure — the prerequisites for understanding the
cluster-disjoint split generation. Split methods live in `splits.md`. Leakage vocabulary live in `leakage.md`.

The chain is: **`leakage.md` → `clusters.md` →
`splits.md`**.

**Doc style conventions.** Data-table subsections follow the pattern
**Source → Columns → Table → Takeaways → (Future direction)** where
applicable. Deep reasoning lives in "worked example" subsections;
parent sections stay tight.

**Threshold notation.** `tXXX` denotes the mmseqs identity threshold
(`--min-seq-id`) at `0.XXX` — e.g., `t095` ≡ `t = 0.95`. The same
notation is used in `splits.md` and `leakage.md`. Cluster output
directories and code identifiers retain the older `idXXX` form
(`data/processed/.../clusters_aa/id095/`, `cluster_disjoint_id095`
routing label) — those are operational artifacts, not renamed with
the docs.

For experimental findings on Flu A see
`docs/results/2026-05-15_cluster_disjoint_nt_results.md` (model
training results) and the two sequence redundancy autogen docs at
`data/processed/flu/July_2025/clusters_{aa,nt}/redundancy_summary.md`.
For the routing-equivalence semantics across the four implemented
modes (seq_disjoint hash_key={seq,dna}, cluster_disjoint
cluster_alphabet={aa,nt}), see `splits.md` § 1.5.

---

## 1. Why we cluster in this pipeline

The segmatch classifier predicts whether two sequences from different proteins/segments co-occur in
the same flu isolate. A random train/val/test split can result
in highly similar sequence pairs being distributed across train,
validation, and test sets (the very similar pairs can be called
*near-neighbors*). Thus, test pairs typically have a train neighbor
that's extremely similar at the sequence level. The model can exploit
this train-test similarity to predict accurately without learning
pair-co-occurrence biology. We define this as leakage mode #4
("cluster leakage" / "near-neighbor leakage") in
`docs/methods/leakage.md`.

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
sequences per training example, one per protein slot), which adds a
second dimension — covered at the end of this section, with the full
pair-level construction in `splits.md` § 1.

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
predicts A2 at evaluation. Test sequences typically have a train-side
neighbor within a handful of aa edits, allowing for shortcut learning.

**Step 2 — clustering at radius t collapses near-neighbors to one cluster id.**
Sequences within identity threshold t of each other join the same
cluster (transitive closure on the pairwise-alignment relation; see
the operational definition above). At t099 on a 760-aa protein
the radius admits up to 7 aa mismatches (§5).

```
   ┌─────────────┐
   │   ●  A1     │
   │      │      │  cluster C_γ   ← A1 and A2 now in the same
   │   ●  A2     │                  cluster (1-aa edit ≤ radius
   └─────────────┘                  at t099)

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

- `t100` — pairs with 100% identity over the aligned region cluster
together. Approximately equivalent to `seq_disjoint` on 7 of the 8 major
proteins, but looser on NA where more sequences of different lengths
cluster via the coverage rule (check §3.2 for actual numbers).
`seq_disjoint` itself splits on seq hash of the full sequence
(`splits.md` § 1.4).
- `t099` — Up to ~1% mismatches over the aligned region cluster
together.
- `t090` — Up to 10% mismatches over the aligned region cluster
together.

A lower `t` removes more leakage but also merges more of the corpus
into mega-clusters. Once a single cluster contains more than ~80% of
all pairs it can't fit into a 0.8 of train, and 80/10/10 routing
becomes structurally impossible.

**From single sequences to pairs.** The schematic above discusses
clustering in the context of a single protein (e.g., HA sequences
clustered into multiple cluster ids). The actual segmatch task includes
two dimensions: each training example is a *pair* ([HA, NA], or [PB2, PB1],...),
and the goal is to predict pair co-occurrence. Cluster-disjoint
routing on pairs has additional mechanics:

1. **Cluster per protein, independently.** HA sequences get HA
   cluster ids (`HA_c1`, `HA_c2`, …); NA sequences get NA cluster
   ids (`NA_c1`, `NA_c2`, …). mmseqs never sees two proteins at once.
2. **Each positive pair carries a tuple of two cluster ids.** A
   positive pair `(HA_i, NA_i)` — both proteins from isolate i —
   becomes the cluster tuple `(cluster(HA_i), cluster(NA_i))`, e.g.,
   `(HA_c1, NA_c5)`. Negative pairs are constructed within each
   split *after* the routing decision; the structures that drive
   routing are built from positives only.
3. **Define the atom — the indivisible routing unit.** Two routing
   modes are implemented (see `splits.md` § 1.1, § 1.2):
   - **Single-slot** (`single_slot='a'` or `'b'`): atom = one cluster's
     pair-set on the constrained slot.
   - **Bilateral** (`single_slot=null`, default): atom = a bipartite
     connected component on the (slot A, slot B) cluster graph.
4. **Bin-pack atoms into 80 / 10 / 10** via LPT-greedy (`splits.md`
   § 1.3).

The clustering itself happens once per (data version, alphabet (`aa`|`nt`), identity
threshold (`t99`|`t98`|...)). Stage 3 reads the cluster lookups when
`dataset.split_strategy.mode: cluster_disjoint` is set in the bundle.

---

## 2. Sequence-space clustering 101

> Note that in §2, "pair" refers to
> *two sequences being aligned by mmseqs2* (the O(N²) alignment problem
> mmseqs2's k-mer prefilter solves), not the (HA, NA) co-occurring
> pairs from §1. mmseqs2 operates per protein (or segment) on single sequences;
> the path from per-protein clusters to training-pair routing is in
> `splits.md` § 1.

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
similar length within a protein), "id 0.95" on a 760-aa PB2 protein
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
t = 1.0 (see §3.1; §6.3's NA t100 row pools 6.9% of the NA corpus
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
(fewer absolute mutations admitted). See § 5 for a per-protein table.

### 2.2 easy-cluster vs easy-linclust

mmseqs2 provides two main methods for clustering:

- **`easy-cluster`** — runs standard cascaded clustering workflow, using multiple sensitivity stages and optional reassignment; generally more sensitive, but slower.
-  **`easy-linclust`** — runs the Linclust workflow, designed for near-linear-time clustering of very large sequence sets; much faster, but usually less sensitive than cascaded clustering.

**Choice on Flu A: symmetric easy-linclust on both alphabets** (since
2026-05-22). The wrapper at `src/utils/clustering_utils.py::run_mmseqs_easy_clust`
defaults to `algorithm='linclust'` and is what
`build_mmseqs_clusters.py` invokes for both the aa and nt
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
§2.1 Case 1 example and §6.3's NA t100 row (6.9% of
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
protein).

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

## 4. Corpus redundancy (exact duplicates)

**Source.** `src/analysis/cluster_analysis_summary.py`;
`results/flu/July_2025/runs/cluster_analysis/seq_freq_tier_summary.csv`
(per (alphabet, protein) — full source for both tables below).

Columns:
- `Segment` — Flu A genomic segment number (1-8).
- `Protein` — short protein name on that segment (PB2, PB1, PA, HA,
  NP, NA, M1, NS1).
- `N_seqs` — total records: number of isolates carrying this protein
  (108,530 for the full Flu A corpus).
- `N_unique` — unique sequences after md5 dedup on `prot_seq` (aa)
  or `cds_dna` (nt). **`cds_dna` is the joined CDS coding sequence
  only** — UTRs, introns, and intergenic DNA are excluded (see
  `src/preprocess/extract_cds_dna.py`); the nt cluster key is
  `cds_dna_hash = md5(cds_dna)`. This equals the FASTA row count
  that mmseqs sees as input (pre-clustering dedup). Verify directly
  with `grep -c '^>' data/processed/flu/July_2025/clusters_{aa,nt}/fasta/<PROTEIN>.fasta`
  (e.g., HA aa → 41,896).
- `N_unique frac` — `N_unique / N_seqs`. The "% unique" reported in
  earlier versions of this doc, now as a fraction.
- `Singleton_seq_frac` — fraction of unique sequences observed
  exactly once in the corpus (= n_singletons / N_unique). Measures
  the size of the *inventory* tail, not the share of records.
- `Top1 frac` — fraction of records carried by the single most
  abundant unique sequence.
- `Top10 frac` — fraction of records carried by the 10 most
  abundant unique sequences combined.
- `Top100 frac` — same for the top 100.
- `Shannon_eff_n` — Hill q=1 effective diversity = exp(H) where
  H = -Σ pᵢ log pᵢ is the Shannon entropy computed over per-unique-
  sequence relative abundances. Acts as an effective sample count
  under Shannon weighting.
- `Gini` — Gini coefficient on the per-unique-sequence frequency
  array. 0 = perfect evenness (every unique sequence equally
  represented), 1 = max concentration (one sequence carries all
  records).

**Flu A data (aa).**

| Segment | Protein | N_seqs | N_unique | N_unique frac | Singleton_seq_frac | Top1 frac | Top10 frac | Top100 frac | Shannon_eff_n | Gini |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 108,530 | 33,663 | 0.310 | 0.756 | 0.047 | 0.195 | 0.385 | 3,488.9 | 0.659 |
| 2 | PB1 | 108,530 | 31,226 | 0.288 | 0.745 | 0.047 | 0.248 | 0.426 | 2,532.7 | 0.681 |
| 3 | PA  | 108,530 | 34,217 | 0.315 | 0.752 | 0.047 | 0.204 | 0.377 | 3,620.0 | 0.652 |
| 4 | HA  | 108,530 | 41,896 | 0.386 | 0.788 | 0.016 | 0.110 | 0.288 | 7,547.6 | 0.586 |
| 5 | NP  | 108,530 | 17,684 | 0.163 | 0.695 | 0.086 | 0.376 | 0.595 |   621.0 | 0.806 |
| 6 | NA  | 108,530 | 37,488 | 0.345 | 0.751 | 0.035 | 0.143 | 0.303 | 5,925.3 | 0.619 |
| 7 | M1  | 108,530 |  4,771 | 0.044 | 0.618 | 0.261 | 0.704 | 0.875 |    46.3 | 0.937 |
| 8 | NS1 | 108,530 | 22,225 | 0.205 | 0.692 | 0.052 | 0.268 | 0.497 | 1,378.7 | 0.758 |

**Flu A data (nt_cds).**

| Segment | Protein | N_seqs | N_unique | N_unique frac | Singleton_seq_frac | Top1 frac | Top10 frac | Top100 frac | Shannon_eff_n | Gini |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 108,530 | 67,341 | 0.620 | 0.830 | 0.008 | 0.037 | 0.094 | 35,673.8 | 0.355 |
| 2 | PB1 | 108,530 | 67,034 | 0.618 | 0.830 | 0.016 | 0.046 | 0.101 | 33,977.6 | 0.358 |
| 3 | PA  | 108,530 | 65,242 | 0.601 | 0.823 | 0.006 | 0.040 | 0.108 | 32,464.9 | 0.373 |
| 4 | HA  | 108,530 | 65,414 | 0.603 | 0.830 | 0.008 | 0.045 | 0.111 | 31,819.8 | 0.373 |
| 5 | NP  | 108,530 | 52,800 | 0.487 | 0.788 | 0.015 | 0.072 | 0.171 | 17,558.8 | 0.480 |
| 6 | NA  | 108,530 | 58,887 | 0.543 | 0.808 | 0.015 | 0.057 | 0.136 | 23,862.6 | 0.428 |
| 7 | M1  | 108,530 | 32,413 | 0.299 | 0.720 | 0.035 | 0.168 | 0.355 |  4,033.3 | 0.661 |
| 8 | NS1 | 108,530 | 38,039 | 0.351 | 0.742 | 0.031 | 0.138 | 0.296 |  6,436.8 | 0.611 |

**Takeaways.**

- **`% unique nt` is always higher than `% unique aa`** (across all 8 major proteins).
  Synonymous codons create distinct CDS DNAs that collapse to one
  protein, so within a protein the nt count is ≥ the aa count.
  Magnitude varies: M1's nt/aa ratio is ~7× (32,413 /
  4,771); HA's is ~1.6× (65,414 / 41,896).
- **M1 is the most redundant (and conserved).** Only 4,771 distinct M1 aa
  sequences (~95% redundancy in aa).
- **Sequence diversity affects how fast clusters collapse in §6.**
  A protein with low **effective diversity** (like M1, the conserved
  internal matrix protein: nominally 4,771 unique aa sequences but
  **n_eff ≈ 10** — the dominant variants absorb most isolates,
  top-10 cover 70 % of isolates) reaches a small cluster count at
  higher `t` than a protein with high effective diversity (like HA,
  the surface antigen under antibody-driven positive selection:
  41,896 uniques, **n_eff ≈ 531**, top-10 cover only 11 %) — even
  when the same `t` admits fewer mutations on M1 (because M1 is
  shorter). The other factor driving collapse is the per-protein
  mutation budget determined by sequence length (discussed in §5
  takeaways). (n_eff is the effective unique-sequence count from
  inverse Simpson / Hill q2; nominal N vs n_eff is the standard ML
  framing — see `seq_freq_tier_summary.csv` `n_eff_hill_q2` column.)

**Per-sequence frequency distribution.** The table above reports
per-protein **counts** of unique sequences; it doesn't show the
**shape** of their per-isolate frequency distribution. The purpose
is to understand corpus structure beyond the unique-sequence count.
`% unique` collapses a lot of info into one number — two proteins
with the same `% unique` can have very different shapes. For
example, M1 aa (4.4 % unique) and HA aa (38.6 % unique) differ not
only in redundancy level but also in *shape*: M1's redundancy comes
from a few dominant sequences (top-10 cover 70.4 % of isolates),
HA's from spread (top-10 cover 10.9 %). Within each protein, most
unique sequences appear in just 1–10 isolates; a small number of
common variants (e.g., circulating H3N2/H1N1 backbones) appear in
hundreds or thousands.

Outputs:

- `seq_freq_hist_aa.png` / `seq_freq_hist_nt_cds.png` — per-protein
  histograms (one panel per protein, 11 log-spaced frequency bins,
  log-Y so both common and rare sequences are visible). The
  `nt_cds` suffix anticipates a future `nt_ctg` variant on
  full-contig DNA.
- `seq_freq_tier_summary.csv` — companion table, collapsed to 5
  tiers (singletons, 2-10, 11-100, 101-1k, 1k+) plus four
  single-number summaries: `max_freq` (top-1 isolate count per
  sequence), `top10_pct_isolates` (% of isolates covered by the
  top-10 most common sequences — head-concentration metric),
  `n_eff_hill_q2` (inverse Simpson / Hill q2 — effective unique
  count after accounting for redundancy; "nominal N vs n_eff"
  framing: M1 aa nominal 4,771 → n_eff ≈ 10; HA aa nominal 41,896
  → n_eff ≈ 531), and `gini` (standard inequality coefficient on
  the per-sequence frequency array, 0 = perfect evenness, 1 = all
  mass on one variant; M1 aa ≈ 0.94, HA aa ≈ 0.59).

**How to read a plots.** The y-axis is **count of unique sequences**;
the x-axis is **how many isolates each of those sequences appears in**.
Worked example, HA aa (4th panel of `seq_freq_hist_aa.png`):

- Bin `1` (height 33,032) — 33,032 distinct HA aa sequences each
  appear in exactly 1 isolate (singletons).
- Bin `2` — 4,417 distinct HA aa sequences each appears in exactly 2
  isolates (doubletons).
- Bin `3` — 1,470 distinct HA aa sequences each appears in exactly 3
  isolates (tripletons).
- Bin `4-6` — 1,533 distinct HA aa sequences each appears in 4, 5, or 6
  isolates.
- Bin `3k+` (height 0) — no HA aa sequences appear in 3,000+
  isolates because HA's most common aa sequence appears in only
  1,702 isolates (the `max_freq` column above). For comparison,
  M1's panel has a `3k+` bar — the most common M1 aa sequence
  appears in 28,306 isolates (~26 % of the corpus carries the same
  M1 aa).

**Why it matters.**

- **Cluster collapse at §6.** Proteins whose top sequences dominate
  the corpus collapse into fewer clusters at higher `t` (see §4
  takeaway "sequence diversity affects how fast clusters collapse"
  above; §6.4 quantifies this directly via Gini-vs-`t`).

---

## 5. Maximum residue mismatches admitted per cluster

**Source.** `src/analysis/cluster_analysis_summary.py`;
`results/flu/July_2025/runs/cluster_analysis/mutations_tolerated_table.csv`,
`sequence_length_summary.csv`

Columns:
- `Min`, `Median`, `Max aa/nt len` = sequence length stats per
  protein from `sequence_length_summary.csv`. The Min column captures a left tail of
  truncated variants; all proteins are left-skewed; NS1 widest has the length distribution.
- `t###` = maximum residue mismatches admitted within a cluster at
  threshold `t`, computed as `L − ceil(L × t)` using the **median**
  length. Truncated variants (toward Min) admit fewer absolute
  mismatches at the same `t`. In mmseqs, threshold `t` is set by
  `--min-seq-id`.

**Amino acids (aa).**

| Segment | Protein | Min aa len | Median aa len | Max aa len | t100 | t099 | t098 | t097 | t096 | t095 | t090 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 732 | 760 | 761 | 0 |  7 | 15 | 22 | 30 | 38 | 76 |
| 2 | PB1 | 731 | 758 | 764 | 0 |  7 | 15 | 22 | 30 | 37 | 75 |
| 3 | PA  | 691 | 717 | 721 | 0 |  7 | 14 | 21 | 28 | 35 | 71 |
| 4 | HA  | 550 | 567 | 571 | 0 |  5 | 11 | 17 | 22 | 28 | 56 |
| 5 | NP  | 472 | 499 | 500 | 0 |  4 |  9 | 14 | 19 | 24 | 49 |
| 6 | NA  | 446 | 470 | 476 | 0 |  4 |  9 | 14 | 18 | 23 | 47 |
| 7 | M1  | 241 | 253 | 254 | 0 |  2 |  5 |  7 | 10 | 12 | 25 |
| 8 | NS1 | 201 | 231 | 239 | 0 |  2 |  4 |  6 |  9 | 11 | 23 |

**Nucleotides (nt).**

| Segment | Protein | Min nt len | Median nt len | Max nt len | t100 | t099 | t098 | t097 | t096 | t095 | t090 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 2,196 | 2,280 | 2,283 | 0 | 22 | 45 | 68 | 91 | 114 | 228 |
| 2 | PB1 | 2,193 | 2,274 | 2,292 | 0 | 22 | 45 | 68 | 90 | 113 | 227 |
| 3 | PA  | 2,073 | 2,151 | 2,163 | 0 | 21 | 43 | 64 | 86 | 107 | 215 |
| 4 | HA  | 1,650 | 1,701 | 1,713 | 0 | 17 | 34 | 51 | 68 |  85 | 170 |
| 5 | NP  | 1,416 | 1,497 | 1,500 | 0 | 14 | 29 | 44 | 59 |  74 | 149 |
| 6 | NA  | 1,338 | 1,410 | 1,428 | 0 | 14 | 28 | 42 | 56 |  70 | 141 |
| 7 | M1  |   723 |   759 |   762 | 0 |  7 | 15 | 22 | 30 |  37 |  75 |
| 8 | NS1 |   603 |   693 |   717 | 0 |  6 | 13 | 20 | 27 |  34 |  69 |

**Takeaways.**

- **Same threshold, different absolute mismatch budgets per protein.**
  t095 on M1 admits **3× fewer absolute mismatches** than t095 on
  PB2 (12 vs 38, from 253 vs 760 aa median lengths) — because the
  budget is `L − ceil(L × t)`, growing with the median length `L`.
- **Length effect on cluster collapse.** At the same `t`, longer
  proteins admit more absolute mismatches, allowing more diverse
  sequences to merge into the same cluster (i.e., fewer clusters at
  that `t`). This is **one factor** behind per-protein
  cluster collapse rate differences in §6.
- **Length and diversity interact.** Neither alone predicts the
  per-protein cluster-collapse rate in §6. See §4 takeaways for the
  diversity-effect sister bullet.

**Future direction.**

- **Per-protein thresholds via uniform mismatch budget.** For
  comparing across protein pairs of different lengths (e.g.,
  HA/NA ~470–567 aa vs PB2/PB1 ~758–760 aa), an absolute mismatch
  budget may be more informative than a fractional identity
  threshold. Picking per-protein thresholds to enforce a uniform
  mismatch budget is tracked in
  `docs/results/2026-05-21_bicc_pair_drop_audit.md` direction #5.

---

## 6. Cluster Analysis

**Source.** `src/analysis/cluster_analysis_summary.py`;
`results/flu/July_2025/runs/cluster_analysis/cluster_summary.csv`
(raw values for all subsections below). The sweep over threshold `t` covers
{1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90}.
Plots: `cluster_counts_vs_threshold.png` (§6.1),
`gini_vs_threshold.png` (§6.4), `bipartite_largest_pct_vs_threshold.png`
(`splits.md` § 1.8). Companion CSV: `cluster_diversity_stats.csv`.

### 6.0 Cluster weighting: three views of the same partition

mmseqs operates on unique sequences only — FASTA deduped by `seq_hash`
per `src/utils/clustering_utils.py::export_function_fasta`. The cluster
parquet stores `{seq_hash → cluster_id}` for unique seqs; one row per
`seq_hash`. Three downstream views of cluster size are used in this
project, each appropriate for a different question.

| View | What it measures | Used in |
|---|---|---|
| **Unique-weighted** (storage) | Cluster size = number of unique seq_hashes in the cluster | mmseqs output; §6.1, §6.2, §6.3; `cluster_summary.csv` |
| **Records-weighted** | Cluster size = Σ copy_count over the cluster's seq_hashes | §6.4 Gini; `cluster_diversity_stats.csv top1_cluster_pct` |
| **Pair-weighted** | Cluster size = number of canonical pairs (post-`pair_key` dedup) whose endpoint on the slot is in the cluster | **Splitter decisions** (`cluster_pair_weight_topk.csv`, `cluster_disjoint_feasibility.csv`, `bipartite_graph_properties graph_props.csv`). For pair universe / cluster pair definitions see `glossary.md`. |

**Unique-weighted is the ground-truth storage.** The parquet stores
`{seq_hash → cluster_id}` for unique seqs only; records-weighted and
pair-weighted are derived downstream via join — against `copy_count`
(records) or against the pair universe (pair). No information loss
in either derivation.

**Tradeoffs.**

| View | Computational | Methodological |
|---|---|---|
| Unique-weighted | + Smallest mmseqs input (~38% of records on HA); fastest clustering. + Reproducible regardless of corpus composition. | + Cluster boundaries reflect sequence similarity only, not abundance. − Doesn't directly express corpus impact or splitter constraints; must be joined downstream. |
| Records-weighted | (Derived — no extra compute beyond the join against `copy_count`.) | + Reflects per-isolate corpus impact of a cluster. − Overcounts for splitter decisions: high-copy seqs that pair with the same partner across isolates collapse to fewer canonical pairs after `pair_key` dedup. |
| Pair-weighted | (Derived — one join per slot against the pair universe.) | + Matches what the splitter actually partitions (canonical pairs). + Single canonical view for all routing modes (1D-CD, 2D-CD, 2D-CD-test). − Not meaningful at the cluster-construction stage (clusters aren't pairs). |

**Project convention: pair-weighted is canonical for splitter decisions.**
Unique-weighted and records-weighted are exploratory diagnostics.

### 6.1 Per-protein n_clusters across thresholds `t` (unique-weighted)

Columns:
- `t###`: number of mmseqs clusters for that protein at threshold
  `t`. Lower threshold → more sequences cluster together →
  fewer clusters.

**Amino acids (aa).**

| Segment | Protein | t100 | t099 | t098 | t097 | t096 | t095 | t094 | t093 | t092 | t091 | t090 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 33,601 | 18,354 | 10,035 |  7,634 |  6,755 |  6,491 |  3,590 |  1,085 |   112 |    43 | **24** |
| 2 | PB1 | 30,822 | 17,209 | 11,859 |  9,266 |  7,384 |  2,033 |    930 |    238 |    42 |    19 |  **6** |
| 3 | PA  | 34,162 | 18,520 | 12,758 | 10,906 |  8,677 |  8,002 |    487 |     72 |    30 |    41 | **17** |
| 4 | HA  | 41,760 | 22,679 | 14,934 | 11,459 |  8,940 |  7,578 |  2,368 |  1,580 |  1,230 |  1,108 |   910 |
| 5 | NP  | 17,533 | 10,483 |  5,038 |  1,750 |    613 |    526 |    149 |    706 |     39 |    20 | **29** |
| 6 | NA  | 18,753 |  9,369 |  6,909 |  4,707 |  3,107 |  2,134 |  1,957 |  1,717 |    757 |    791 | 1,077 |
| 7 | M1  |  4,712 |  1,764 |  1,033 |  1,003 |    708 |    129 |    269 |    237 |     31 |    24 | **24** |
| 8 | NS1 | 22,131 | 13,508 |  9,109 |  6,405 |  4,306 |  3,458 |  1,971 |  1,258 |    990 |    943 |   786 |

**Nucleotides (nt).**

| Segment | Protein | t100 | t099 | t098 | t097 | t096 | t095 | t094 | t093 | t092 | t091 | t090 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 66,475 | 11,484 |  4,562 |  2,968 |  1,353 |    954 |    552 |    398 |    316 |    509 |    180 |
| 2 | PB1 | 66,138 | 14,990 |  6,276 |  2,548 |  1,307 |    742 |    472 |    346 |    242 |    179 |    121 |
| 3 | PA  | 64,406 | 11,184 |  4,356 |  2,214 |  1,071 |    719 |    338 |    216 |    129 |     84 |     47 |
| 4 | HA  | 64,526 | 12,150 |  6,412 |  3,444 |  1,843 |  1,277 |    903 |    595 |    472 |    403 |    275 |
| 5 | NP  | 52,097 | 11,627 |  4,754 |  2,559 |  1,427 |  1,182 |    786 |    542 |    455 |    298 |     66 |
| 6 | NA  | 57,987 | 12,092 |  4,840 |  2,611 |  1,662 |  1,108 |    905 |    599 |    441 |    419 |    250 |
| 7 | M1  | 31,974 | 10,227 |  4,239 |  1,420 |  1,101 |    619 |    263 |    119 |    154 |     80 |     48 |
| 8 | NS1 | 37,458 | 12,012 |  4,133 |  1,988 |  1,245 |    800 |    419 |    299 |    326 |    179 |    196 |

**Takeaways.**

- **Cluster counts bound flexibility; threshold `t` bounds separation.**
  More clusters at threshold `t` means more atoms available to the
  routing's bin-packer (`splits.md` § 1.3) — that's the **flexibility**
  lever. The **separation strength** of the resulting splits is
  governed by `t` itself, not the count — see `splits.md` § 2.3 for
  the cross-cluster gap argument.
- **Two trajectories (deferred-cliff vs gradual).** Segments 1/2/3/5/7
  retain moderate counts through t095, then drop sharply across t094
  → t091 (e.g., PB2 aa: 6,491 → 3,590 → 1,085 → 112 → 43 — five distinct
  cliff steps). Segments 4/6/8 decline more smoothly across the whole sweep
  and still hold 73 (NA), 174 (NS1), and 176 (HA) aa clusters at t080.
- **The conserved-protein cliff has a center (not just a span).**
  Each polymerase subunit and matrix protein has its own "tipping
  tXX" between t095 and t091, but no
  single threshold step applies to all five conserved proteins. PB2 aa drops most at t093 → t092
  (1,085 → 112, −90%); PB1 aa at t094 → t093 (930 → 238, −74%);
  PA aa at t095 → t094 (8,002 → 487, −94% — the steepest 1-pp
  drop in the table); M1 aa at t093 → t092 (237 → 31, −87%).
- **NA is the most-gradual outlier.** NA aa stays close to 1,000 clusters across the entire
  t095..t090 stretch — its t095 → t090 drop is only 2,134 →
  1,077 (−50%, much smaller than HA's 7,578 → 910 = −88%).
- **Easy-linclust is monotone-decreasing in the coarse direction (t100 → t090) but not at 1-pp steps.** Small non-monotone bumps appear (NP aa 526 → **149** → 706; NA aa 1,717 → **757** → **791** → 1,077; M1 aa 129 → **269** → 237) because single-pass seed selection produces threshold-dependent cluster topology.
- **nt at t100 always exceeds aa at t100** (synonymous variants
  split into distinct nt singletons). The aa-vs-nt relationship
  inverts at t099 and t098 on most proteins — see §6.2 worked
  example.

### 6.2 Worked example: aa vs nt non-nesting at t099 (unique-weighted)

**Source.** §6.1 (observation), §5 mutations_tolerated_table.csv
(residue-tolerance asymmetry), `src/analysis/aa_nt_cluster_crosstab.py`
+ `docs/results/2026-05-22_aa_vs_nt_cluster_mechanism.md` (cross-tab
walkthrough).

§6.1 noted that nt clusters outnumber aa at t100 but invert at t099
and t098 on most proteins. The inversion at t099:

| Protein | aa clusters | nt clusters | nt / aa ratio |
|---|---:|---:|---:|
| PB2 | 18,354 | 11,484 | 0.63 |
| PB1 | 17,209 | 14,990 | 0.87 |
| PA  | 18,520 | 11,184 | 0.60 |
| HA  | 22,679 | 12,150 | 0.54 |
| NP  | 10,483 | 11,627 | 1.11 |
| NA  |  9,369 | 12,092 | 1.29 |
| M1  |  1,764 | 10,227 | 5.80 |
| NS1 | 13,508 | 12,012 | 0.89 |

Five proteins have ratio < 1 (nt fewer at t099); M1 is the outlier in
the other direction (5.8×); NP / NA / NS1 sit near 1.

**Mechanism.** Cross-tab co-membership (writeup link in Source) shows
aa and nt cluster boundaries disagree in BOTH directions on every
protein — neither alphabet's clustering is nested in the other; net
count per protein is the balance of those opposing directions. At
t099, nt also admits more absolute residue mismatches per cluster
than aa (CDS ≈ 3× protein length → 3× the mismatch budget at the
same fractional identity; see §5 mutations table).

**Takeaways.**

- The "nt has more clusters than aa because of synonymous codons"
  intuition holds at t100 but not below; t099 and t098 mostly invert.
- aa and nt cluster boundaries disagree in both directions; net
  cluster count is a per-protein balance, not nested containment.
- Per-protein decomposition: `docs/results/2026-05-22_aa_vs_nt_cluster_mechanism.md`.

### 6.3 Top-1 (largest) cluster as % of corpus (unique-weighted)

**Source.** `src/analysis/cluster_analysis_summary.py`;
`cluster_summary.csv` column: `largest_cluster / n_sequences × 100`.
The terms **largest cluster** and **top-1 cluster** are
interchangeable in this doc.

Layout:
- **Columns** (`t###`): percentage of the protein's **unique sequences**
  that fall into its single top-1 (largest) cluster at threshold
  `t`. The denominator is `n_sequences` in
  `cluster_summary.csv` — the unique protein-sequence count input
  to mmseqs2 — **not** the total isolate-record count. Concretely for PB1 aa: the
  denominator is 31,226 unique sequences, not 108,530 isolate
  records. 100% means one cluster contains every unique sequence
  of that protein.

| Segment | Protein | t100 | t099 | t098 | t097 | t096 | t095 | t094 | t093 | t092 | t091 | t090 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 0.0% | 12.7% | 12.6% | 13.7% | 15.6% | 15.6% | 15.9% | 38.3% | 64.4% | 77.6% |  95.9% |
| 2 | PB1 | 0.1% | 12.9% | 12.0% | 18.5% | 25.9% | 72.4% | 90.9% | 98.9% | 99.8% | 99.9% | **100%** |
| 3 | PA  | 0.0% |  7.3% |  8.9% |  8.9% |  9.0% |  9.0% | 56.5% | 65.3% | 75.3% | 96.3% |  98.1% |
| 4 | HA  | 0.0% |  5.0% |  6.7% |  9.6% |  9.7% | 11.8% | 12.1% | 21.0% | 21.9% | 22.5% |  22.8% |
| 5 | NP  | 0.1% |  7.5% | 13.1% | 28.2% | 41.4% | 46.3% | 51.2% | 67.5% | 84.0% | 85.0% |  99.8% |
| 6 | NA  | 6.9% |  8.7% |  8.9% |  8.8% | 13.2% | 13.2% | 17.9% | 14.0% | 18.0% | 17.7% |  17.9% |
| 7 | M1  | 0.2% | 10.8% | 17.8% | 23.1% | 41.7% | 56.7% | 84.1% | 93.3% | 99.1% | 99.2% |  99.2% |
| 8 | NS1 | 0.0% |  3.9% |  4.8% |  9.4% | 11.7% | 16.3% | 19.0% | 19.3% | 19.3% | 19.5% |  21.1% |

**Takeaways.**

- **The mega-cluster forms one step at a time on conserved proteins.**
  PB2 aa: 15.6% (t095) → 15.9 → 38.3 → 64.4 → 77.6 → 95.9% (t090)
  — five graded steps between "still-small" and "essentially-100%",
  not a single cliff. PB1 aa shows the earliest growth (72% at t095
  → 91% at t094 → 99% at t093). PA aa takes the steepest single pp
  jump (9.0% at t095 → 56.5% at t094). M1 jumps from 57% to 84% in
  one step at t094 and reaches 99% by t092.
- **HA / NA / NS1 stay flat across t095..t090.** HA largest is
  11.8% at t095 and only 22.8% at t090 — never forms a mega-
  cluster. NA largest stays in the 13–18% band across the same
  stretch. NS1 climbs slowly to 21% at t090. By the same threshold
  the conserved proteins are ≥96%. This is the same biology that
  §6.1 surfaces from the cluster-count direction.
- **NA's t100 anomaly: 6.9% already.** NA is the only protein with
  a sub-t100 entry visibly above zero, because of the stalk-length
  collapse mechanism (§4 NA note).


### 6.4 Cluster-collapse evenness — Gini (records-weighted)

**Source.** `src/analysis/cluster_analysis_summary.py`;
`cluster_diversity_stats.csv` (per alphabet × protein × threshold);
plot at `gini_vs_threshold.png` (2 panels — aa, nt).

§6.1 reports cluster counts; §6.3 reports the top-1 cluster's
share of the corpus. Both are scalar summaries of the cluster-size
distribution. The full distribution also has an **inequality**
dimension — at a given `t`, are isolates spread evenly across
clusters or concentrated in a few? Gini measures this: 0 = perfect
evenness (every cluster the same size), 1 = max concentration (one
cluster has everyone). Computed on per-cluster isolate counts
(sum of copy counts of all unique sequences in the cluster).

Curves start at Gini > 0 at `t = 1.0` (per-sequence copy-number
redundancy creates inequality before clustering — level-0 floor) and
rise monotonically as `t` drops. Conserved proteins reach Gini ≈
0.97 earliest (PB1 by `t094`); HA reaches the high range latest.
Complementary to §6.1 (n_clusters) and §6.3 (top-1 share) — same
protein ordering, distributional shape rather than scalar.

`cluster_diversity_stats.csv` also carries `n_eff_hill_q2` per
(alphabet, protein, threshold) — inverse Simpson (effective
cluster count after accounting for inequality), the same metric
applied to clusters that V5's `n_eff_hill_q2` applies to copy
numbers (§4).

---

## 7. Pipeline integration

Five scripts make up the chain. The first four are the
producer/consumer path that feeds Stage 3 via
`src/datasets/_split_helpers.py`; the fifth is a post-hoc
structural-summary script that backs §§ 4–6 of this doc and
§ 1.9 of `splits.md`.

| Step | Script | Reads | Writes |
|---|---|---|---|
| Build CDS (nt only) | `src/preprocess/extract_cds_dna.py` (Stage 1.5) | `protein_final.csv` + `genome_final.csv` | `cds_final.parquet` |
| Cluster sweep | `src/analysis/build_mmseqs_clusters.py` | `protein_final.parquet` (aa) or `cds_final.parquet` (nt) | `clusters_{aa,nt}/`: per-protein FASTAs, per-threshold cluster parquets, `combined_cluster.parquet`, `redundancy_stats.csv`, `<out_root>/runtime.json`, `redundancy_summary.md` |
| Feasibility pre-flight | `src/analysis/cluster_disjoint_feasibility.py` (bilateral) and `src/analysis/single_slot_cluster_disjoint_feasibility.py` (single-slot) | one cluster lookup + `protein_final` or `cds_final` | `results/flu/{version}/runs/cluster_disjoint_feasibility/{feasibility,single_slot_feasibility}_<pair>_<alphabet>.csv` |
| Stage 3 consumes | `src/datasets/dataset_segment_pairs_v2.py` (when `split_strategy.mode: cluster_disjoint`) | `combined_cluster.parquet` for the chosen (alphabet, threshold) | `dataset_*/cluster_disjoint_audit.json` |
| Post-hoc structural summary | `src/analysis/cluster_analysis_summary.py` | `redundancy_stats.csv`, per-threshold cluster parquets, `cds_final.parquet`, feasibility CSVs | tables + plots under `results/flu/{version}/runs/cluster_analysis/` (see § 8 outputs block) |

---

## 8. Regenerating everything

The producer/consumer chain, end to end:

```bash
# 1. (nt-only) Build cds_final.parquet — once per data version.
python src/preprocess/extract_cds_dna.py --config_bundle flu_ha_na

# 2. Per-protein clustering sweep — once per (alphabet, data version).
#    Writes <out_root>/{fasta,id<NN>}/, redundancy_stats.csv, runtime.json,
#    redundancy_summary.md (alongside the stats CSV; not under docs/).
#    The merge-fix in build_mmseqs_clusters.py preserves prior
#    threshold rows on re-run, so subset reruns don't wipe the CSV.
python -m src.analysis.build_mmseqs_clusters \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --out_root      data/processed/flu/July_2025/clusters_aa \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 \
    --threads 16

python -m src.analysis.build_mmseqs_clusters \
    --cds_final  data/processed/flu/July_2025/cds_final.parquet \
    --out_root   data/processed/flu/July_2025/clusters_nt \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90 \
    --algorithm  linclust \
    --threads    16

# 3. Bipartite-component feasibility per schema pair × alphabet.
#    --out_csv defaults to
#    results/flu/July_2025/runs/cluster_disjoint_feasibility/feasibility_<pair>_<alphabet>.csv
python -m src.analysis.cluster_disjoint_feasibility \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_aa \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90

python -m src.analysis.cluster_disjoint_feasibility \
    --cds_final     data/processed/flu/July_2025/cds_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_nt \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.98 0.97 0.96 0.95 0.90

# (repeat the two feasibility calls for PB2/PB1)

# 4. Consolidated structural summary (CSVs + plots).
python -m src.analysis.cluster_analysis_summary
```

Outputs land under
`results/flu/July_2025/runs/cluster_analysis/`:

```
cluster_summary.csv                       — per (protein, alphabet, threshold)
sequence_length_summary.csv               — per (protein, alphabet)
mutations_tolerated_table.csv             — concrete max-mismatches table
seq_redundancy.png                        — Plot A (§4)
seq_freq_hist_aa.png                      — Plot D, aa (§4)
seq_freq_hist_nt_cds.png                  — Plot D, nt CDS (§4)
seq_freq_isolate_pct_aa.png               — Plot D-alt, aa (§4)
seq_freq_isolate_pct_nt_cds.png           — Plot D-alt, nt CDS (§4)
seq_freq_tier_summary.csv                 — Plot D companion tier table (§4); cols include Gini + n_eff (level 0)
cluster_counts_vs_threshold.png           — Plot B (§6.1)
cluster_diversity_stats.csv               — per (alphabet, protein, threshold); Gini + n_eff (level 1) (§6.4)
gini_vs_threshold.png                     — Plot E (§6.4)
bipartite_largest_pct_vs_threshold.png    — Plot C (`splits.md` § 1.8)
```

---

## 9. See also

- `docs/methods/splits.md` — consumer of cluster lookups; covers
  atoms, LPT-greedy, single-slot vs bilateral, k-fold, feasibility
  ceilings, prior-art cross-reference.
- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` — the
  original plan that motivated this machinery.
- `data/processed/flu/July_2025/clusters_aa/redundancy_summary.md` (aa)
  and `data/processed/flu/July_2025/clusters_nt/redundancy_summary.md`
  (nt) — autogen per-threshold cluster-size tables for all 8 majors.
- `results/flu/July_2025/runs/cluster_disjoint_feasibility/feasibility_{ha_na,pb2_pb1}_{aa,nt}.csv`
  — raw bipartite-CC feasibility numbers per pair × alphabet × threshold.
