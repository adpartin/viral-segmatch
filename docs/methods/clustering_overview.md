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

The segmatch classifier's job is to predict whether two segments co-occur in
the same flu isolate. A random train/val/test split can result
in highly similar sequence pairs being distributed across train,
validation, and test sets (the very similar pairs can be called
near-neighbors). Thus, test pairs typically have a train neighbor
that's extremely similar at the sequence level, and the model can
exploit that to make accurate predictions without learning any pair-co-occurrence biology. We
defined this as leakage mode #4 ("cluster leakage" / "near-neighbor
leakage") in `docs/methods/leakage_definitions.md`.

Clustering uses an alignment-identity threshold *t*. For each input
sequence X (aa or nt), pairwise alignment with every candidate sequence Y
yields a percent-identity number; Y is in X's neighborhood at
threshold t if identity(X, Y) ≥ t and the coverage rule of §3.2 is
met. Intuitively, this defines a *radius* around X — the set of
sequences alignable to it at identity ≥ t. Two sequences whose radii
overlap (i.e., are pairwise within threshold t of each other)
collapse to the same cluster id by transitive closure. The threshold
t is set by `--min-seq-id` in mmseqs (formal semantics in §3.1).

**Visual schematic — simplified single-sequence view.** The schematic
below uses a single-sequence simplification of the splitting task to
show *why* random splitting leaks and *how* clustering by radius
mitigates it. The real segmatch task operates on *pairs* (two
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
scores A2 at evaluation — without learning anything about the sequence biology. Test sequences typically have a train-side
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

- `t = 1.0` — only exact matches collapse (essentially `seq_disjoint`, §7.3).
- `t = 0.99` — ~1% mismatches collapse (Flu A feasibility ceiling on HA/NA and PB2/PB1 pairs, §9).
- `t = 0.90` — 10% mismatches collapse; on the conserved proteins (PB2/PB1/PA/NP/M1) the corpus collapses to ≤7 clusters total (§6.1) and 80/10/10 routing breaks.

A stricter radius removes more leakage but also collapses more of the
corpus into mega-clusters that overflow the largest split's quota.
§9's feasibility ceiling is the threshold below which 80/10/10 routing
becomes structurally impossible.

**From single sequences to pairs.** The schematic above shows
clustering one function (e.g., HA sequences clustered into HA cluster
ids). The real segmatch task adds a second dimension: each training
example is a *pair* ([HA, NA], or [PB2, PB1],...), and the goal is to
predict pair co-occurrence. Cluster-disjoint routing on pairs has
additional mechanics:

1. **Cluster per function, independently.** HA sequences get HA
   cluster ids (`HA_c1`, `HA_c2`, …); NA sequences get NA cluster
   ids (`NA_c1`, `NA_c2`, …). mmseqs never sees two functions at once.
2. **Each pair carries a tuple of two cluster ids.** A pair
   (HA_x in isolate i, NA_y in isolate i) becomes
   `(cluster(HA_x), cluster(NA_y))`, e.g., `(HA_c1, NA_c5)`.
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

mmseqs2 represents similarity of two biological sequences using **percent
identity**. With the default `--seq-id-mode 0`, identity is computed as

```
identity = n_identical / alignment_length
```

where:

- `identity` ∈ [0, 1] is the percent identity — the value compared
  against the `--min-seq-id` threshold.
- `n_identical` = the number of alignment columns in which both
  sequences carry the same residue (a *match*). Gap columns are
  never matches by definition.
- `alignment_length` = the total number of columns in the local
  alignment, counting **matches**, **mismatches** (columns where both
  sequences have a residue but they differ), and **gaps** (columns
  where one sequence has a residue and the other has a placeholder
  `-`, representing an insertion in one or a deletion in the other).

`alignment_length` is *not* the length of either input sequence — it
can exceed both when one sequence has insertions relative to the
other. An `--min-seq-id` threshold of 0.95 is a *floor* — only
sequence pairs with identity ≥ 0.95 are admitted to the same cluster.

The match unit is **residues**: amino acids (aa) for proteins,
nucleotides (nt) for DNA. Length is counted also in residues. As a
first-order intuition, when the alignment spans both sequences
end-to-end with no gaps (typical when comparing two sequences of
similar length within a function), "id 0.95" on a 760-aa PB2 protein
admits ~38 aa mismatches within a cluster; "id 0.95" on a 2,280-nt
PB2 CDS admits ~114 nt mismatches. (For the exact
per-function/per-threshold table see §5.)

**`--seq-id-mode` documentation gotcha.** This flag is not listed in
the official PDF userguide at https://mmseqs.com/latest/userguide.pdf
(which covers concepts rather than every CLI flag). The canonical
reference for every parameter is `mmseqs <subcommand> --help`, e.g.
`mmseqs easy-cluster --help`, which on the installed binary (v18.8cc5c)
shows:

```
--seq-id-mode INT     0: alignment length 1: shorter, 2: longer sequence [0]
```

**Worked examples** — three small alignments to fix intuition for the
identity formula and its interaction with the coverage rule:

```
Case 1: no gaps, one mismatch
─────────────────────────────────────────────
Seq A:   M K T V R Q E L K L            (10 residues)
Seq B:   M K T V R Q E L K Y            (10 residues)
status:  = = = = = = = = = X            (= match, X mismatch)

alignment_length  = 10  (columns in the alignment)
n_identical       =  9
identity (mode 0) = 9 / 10 = 0.90  ← what mmseqs compares to --min-seq-id

Case 2: one internal gap (different-length sequences)
─────────────────────────────────────────────
Seq A:   M K T - R Q E L K L            (residues:  9; alignment row: 10 cols)
Seq B:   M K T V R Q E L K L            (residues: 10; alignment row: 10 cols)
status:  = = = G = = = = = =            (G = gap)

alignment_length  = 10  (every column counts: matches + mismatches + gaps)
n_identical       =  9  (gap columns are never identical by definition)
identity (mode 0) = 9 / 10 = 0.90
identity (mode 1) = 9 /  9 = 1.00  ← over shorter sequence  (9 residues)
identity (mode 2) = 9 / 10 = 0.90  ← over longer sequence  (10 residues)

So mode 0 makes gaps cost identity (an indel pushes the value down);
mode 1 ignores indels entirely. We pin mode 0 — see §3.2 for the
coverage discussion that interacts with this choice.

Case 3: fragment vs full protein — identity passes, coverage fails
─────────────────────────────────────────────
Seq A:   M K T V R Q E L K L                                     (10 residues; fragment)
Seq B:   M K T V R Q E L K L Q W S P R M N K T L H A V S Q E S F (40 residues; full)
         = = = = = = = = = = ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  (= aligned, ─ unaligned in B)

alignment_length  = 10
n_identical       = 10
identity (mode 0) = 10 / 10 = 1.00      ✓ identity passes even at t = 1.0

cov(A)  = 10 / 10 = 1.00                ✓
cov(B)  = 10 / 40 = 0.25                ✗ fails -c 0.8 (under --cov-mode 0)

Under --cov-mode 0 BOTH sequences must have ≥80% of their residues
inside the aligned region, so the pair does NOT cluster despite
identity = 1.00.
```

This is the second mechanism behind §3.1's "modulo coverage" caveat:
the coverage gate stops fragment-vs-full matches from clustering at
t = 1.0, but it still permits length-variants where ≤20% of each
sequence is unaligned (e.g., NA stalk-deletion isoforms aligning to
NA stalk-full proteins — see §4). See §3.2 for the full coverage
discussion and `--cov-mode` alternatives.

Note that the same threshold is **biologically stricter on shorter proteins**
(fewer absolute mutations admitted). See § 5 for a per-function table.

### 2.2 Why not align every sequence to every other sequence?

In the Flu A July 2025 corpus there are 108,530 isolates × 8 major
proteins = 868,240 protein records. Even after deduplication,
HA has ~42,000 unique aa sequences (~65,000 unique nt CDS — same order
of magnitude). Computing exact alignments for every aa sequence pair
would be 42,000² / 2 ≈ 9 × 10⁸ alignments. Infeasible.

mmseqs2 sidesteps the quadratic with a k-mer prefilter + alignment
cascade:

```
   FASTA (one entry per unique sequence)
            │
            ▼
   ┌───────────────────────────┐
   │ Step 1. k-mer prefilter   │  Extract every length-k subsequence
   │                           │  from each input sequence. Build an
   │   query   k-mers indexed  │  index of (k-mer → target seqs).
   │     │     ▼               │  For each query, look up its k-mers
   │     └─►  match list       │  and rank candidate targets by the
   │                           │  number of shared k-mers above a
   └─────────────┬─────────────┘  score threshold.
                 │
                 ▼  candidate (query, target) pairs — O(N) on average
                 │
   ┌───────────────────────────┐
   │ Step 2. Banded alignment  │  For each surviving candidate pair,
   │                           │  run a banded Smith–Waterman
   │   query ────────────►     │  alignment and compute exact percent
   │             alignment     │  identity + coverage. Drop pairs that
   │   target ────────────►    │  fail --min-seq-id or the coverage
   │                           │  cutoff.
   └─────────────┬─────────────┘
                 │
                 ▼  high-scoring pairs
                 │
   ┌───────────────────────────┐
   │ Step 3. Cluster assignment│  Default --cluster-mode 0 (Set-Cover,
   │                           │  greedy): pick the sequence with the
   │   greedy representative   │  most unassigned similar neighbors as
   │   selection (Set-Cover)   │  the next cluster representative;
   │                           │  absorb it + its neighbors into one
   │                           │  cluster; repeat until all assigned.
   └─────────────┬─────────────┘
                 │
                 ▼
   cluster_tsv  (one row per member: rep_id ⟷ member_id)
```

The k-mer prefilter is the load-bearing step: it converts the quadratic
all-pairs problem into a roughly linear "look up similar k-mers"
problem. The trade-off is that sequences below the prefilter score
threshold never reach the alignment step, so a fast prefilter can miss
true matches (sensitivity loss).

### 2.3 easy-cluster vs easy-linclust

mmseqs2 ships two main entry points for unsupervised clustering:

- **`easy-cluster`** — runs the cascade above in multiple sensitivity
  passes (cascaded clustering, with optional re-assignment). Higher
  sensitivity. Slower per run on long sequences.
- **`easy-linclust`** — a linear-time variant that uses a single-pass
  prefilter and skips the cascade. Lower sensitivity. Faster.

**Choice on Flu A: symmetric easy-linclust on both alphabets** (since
2026-05-22). The wrapper at `src/utils/clustering_utils.py::run_mmseqs_easy_cluster`
defaults to `algorithm='linclust'` and is what
`seq_redundancy_per_function.py` invokes for both the aa and nt
sweeps. Decision-relevant mmseqs2 flags are pinned explicitly on the
CLI (see that wrapper's docstring for the full pinned set:
`--cluster-mode 0`, `--seq-id-mode 0`, `--similarity-type 2`,
`-e 0.001`, `--dbtype 1` (aa) / `2` (nt), in addition to the
caller-supplied `--min-seq-id`, `-c 0.8`, `--cov-mode 0`).

**Why symmetric** (was asymmetric, easy-cluster on aa + easy-linclust
on nt, prior to 2026-05-22). A 2026-05-21 validation experiment
compared easy-cluster vs easy-linclust on identical aa input at
identical parameters. On the full Flu A corpus the two algorithms
disagreed by a factor of 5–500× on cluster counts at sub-id100
thresholds (e.g., PB2 at id095: 26 clusters under easy-cluster vs
6,491 under easy-linclust — a 250× ratio). The gap was scale-dependent
(under 6% at N = 100, growing super-linearly to ~520% at the full
corpus N ≈ 42K) and traced to easy-cluster's 3-round cascade + spaced
k-mers chaining transitively-similar sequences that easy-linclust's
single-pass prefilter cannot bridge. Under the prior asymmetric setup,
any observed aa-vs-nt difference confounded the alphabet effect with
the algorithm sensitivity gap. Symmetric easy-linclust holds the
algorithm constant so that aa-vs-nt comparisons (in §4, §5, §6, §9)
reflect alphabet diversity rather than algorithm sensitivity. Full
write-up: `docs/results/2026-05-22_aa_cluster_algorithm_validation_results.md`.

Measured cost on the Flu A July 2025 corpus
(`data/processed/flu/{version}/clusters_aa/runtime.json` and
`clusters_nt/runtime.json`, both written by
`src/analysis/seq_redundancy_per_function.py`):

| Alphabet | Algorithm | Sweep | Median/run (s) | Max/run (s) |
|---|---|---|---:|---:|
| aa | easy-linclust | 90 runs (10 fn × 9 thresholds: id100/099/098/097/096/095/090/085/080) | 1.5 | 11 (PA @ id097) |
| nt | easy-linclust | 72 runs (8 fn × 9 thresholds: same set as aa) | 6.7 | 217 (PB1 @ id100) |

Both alphabets now share one algorithm. The aa-vs-nt runtime
asymmetry (~4× faster on aa) reflects sequence-length differences
(~570 aa median vs ~1700 nt median; nt sequences are 3× longer and
the prefilter cost scales accordingly).

---

## 3. The three knobs that define a clustering

```
mmseqs <easy-cluster | easy-linclust>  <input.fasta>  <out_prefix>  <tmp_dir> \
    --min-seq-id <t>   -c 0.8   --cov-mode 0   [--dbtype 2]
```

The pipeline wires these via
`src/utils/clustering_utils.py::run_mmseqs_easy_cluster`. The four
mechanics that actually drive behavior are below.

### 3.1 `--min-seq-id <t>` — identity threshold

Range [0, 1]. Two sequences cluster iff their pairwise identity is
≥ t (and they pass the coverage rule below). Counted on residues
(aa for protein, nt for DNA). At t = 1.0, only sequence pairs with
100% identity *over the aligned region* cluster — the aligned region
need not span the whole sequence. Under our `-c 0.8 --cov-mode 0`
settings (§3.2), up to 20% of each sequence can lie outside the
alignment, so length-variants (e.g., NA stalk-deletion isoforms) can
still cluster at t = 1.0 even though the full strings differ. See
§2.1 Case 3 for a worked example and §6.3's NA id100 row (6.9% of
the corpus pooled into one cluster) for the empirical evidence on
Flu A.

The biological meaning is per-function — the *same threshold* admits
different numbers of mutations on a 760-aa PB2 protein vs a 252-aa
M1 protein:

| Segment | Function | Median aa length | id098 | id097 | id096 | id095 | id090 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | PB2 | 760 | 15 | 22 | 30 | 38 | 76 |
| 2 | PB1 | 758 | 15 | 22 | 30 | 37 | 75 |
| 3 | PA  | 717 | 14 | 21 | 28 | 35 | 71 |
| 4 | HA  | 567 | 11 | 17 | 22 | 28 | 56 |
| 5 | NP  | 499 |  9 | 14 | 19 | 24 | 49 |
| 6 | NA  | 470 |  9 | 14 | 18 | 23 | 47 |
| 7 | M1  | 253 |  5 |  7 | 10 | 12 | 25 |
| 8 | NS1 | 231 |  4 |  6 |  9 | 11 | 23 |

(Segment = Flu A canonical segment id. Segments 7 and 8 each encode
two proteins via splicing — M1/M2 from segment 7 and NS1/NEP from
segment 8 — so when M2 or NEP appear in later tables they share the
same segment number as M1 and NS1 respectively.)

Each cell is the maximum number of aa mismatches admitted inside a
cluster at that threshold, computed as `L − ceil(L × t)`. See §5 for
the full reference table (id100 through id080) and the nt
equivalent. Source: `results/.../cluster_analysis/mutations_tolerated_table.csv`,
generated by `src/analysis/cluster_analysis_summary.py`.

### 3.2 `-c 0.8` and `--cov-mode 0` — coverage rule

Identity alone is not enough: a 200-aa fragment can be 100% identical
to a 200-aa stretch of an 800-aa full protein. The coverage rule
prevents that fragment from clustering with the full protein.

```
                                 [identity = 100% on the aligned region]
query   ════════════════════════════════════════════════   L_q = 800
                       ║║║║║║
target                 [─── fragment ───]                  L_t = 200
                       covered region: 200 residues

cov(target) = 200 / 200 = 100%   ✓  (≥ 0.8)
cov(query)  = 200 / 800 =  25%   ✗  (< 0.8)
```

Under **`--cov-mode 0` (bidirectional)** the rule requires
`cov(query) ≥ c` AND `cov(target) ≥ c`. The fragment-vs-full case
above fails the rule and does not cluster.

Other cov-modes (1 = target-only, 2 = query-only) would let
fragment-against-full clusters through. We use cov-mode 0 throughout
because it's the conservative choice: every pair that clusters
contains two sequences of comparable length, which is what "same
gene product" means biologically. Sequence-length variation on Flu A
majors is small (`gto_format_reference.md` §6.5: std ≤ 2.8 aa per
function), so cov-mode 0 effectively just enforces "near-identical
length" alongside the identity threshold.

### 3.3 `--dbtype` and the alphabet

mmseqs2 represents biological sequences differently depending on
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

Stage 1 (`preprocess_flu.py`) scrubs proteins with high X-fraction
(`prepare_sequences_for_esm2`), so the aa side rarely encounters X.
The nt side leaves IUPAC codes intact; mmseqs handles them via the
nucleotide matrix.

Two notes that are easy to get wrong:

1. **`--search-type 3`** is *not* a valid flag on
   `easy-cluster`/`easy-linclust` in mmseqs 18. It exists on the
   `search` subcommand only. The original B-nt plan referenced it,
   but the implementation uses `--dbtype 2` instead. Passing
   `--search-type 3` to easy-cluster raises an "Unrecognized
   parameter" error.
2. **`--alph-size aa:21,nucl:5`** is set internally by mmseqs and is
   not a user knob in this pipeline.

For the detailed routing-equivalence semantics (aa cluster_id100 ≈
seq_disjoint hash_key=seq vs nt cluster_id100 ≠ seq_disjoint
hash_key=dna), see `docs/methods/leakage_definitions.md`
§ "Routing equivalence and mmseqs argument semantics". It walks
through which mechanics are alphabet-agnostic and which are
alphabet-specific.

---

## 4. Per-function corpus redundancy (Flu A July 2025)

Source: `results/flu/July_2025/runs/cluster_analysis/cluster_summary.csv`
(from `cluster_analysis_summary.py`, which reads
`clusters_{aa,nt}/redundancy_stats.csv` written by
`seq_redundancy_per_function.py`).

Per-function unique-sequence counts at threshold = 1.00 (i.e., exact
identity clustering — every cluster has all-identical members):

| Segment | Function | Input rows | Unique aa | % unique aa | Unique nt | % unique nt |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PB2 | 108,530 | 33,663 | 31.0% | 67,341 | 62.1% |
| 2 | PB1 | 108,530 | 31,226 | 28.8% | 67,034 | 61.8% |
| 3 | PA  | 108,530 | 34,217 | 31.5% | 65,242 | 60.1% |
| 4 | HA  | 108,530 | 41,896 | 38.6% | 65,414 | 60.3% |
| 5 | NP  | 108,530 | 17,684 | 16.3% | 52,800 | 48.6% |
| 6 | NA  | 108,530 | 37,488 | 34.5% | 58,887 | 54.3% |
| 7 | M1  | 108,530 |  4,771 |  4.4% | 32,413 | 29.9% |
| 8 | NS1 | 108,530 | 22,225 | 20.5% | 38,039 | 35.0% |

Column meanings:

- `Input rows` = the number of isolates that carry this protein
  (108,530 in this corpus on every function).
- `Unique aa` / `Unique nt` = unique sequence count after md5-dedup on
  `prot_seq` / `cds_dna` respectively. This is the FASTA row count
  that mmseqs sees as input. It is *not* an mmseqs cluster count —
  it's the pre-clustering dedup result and is algorithm-agnostic.
- `% unique aa` / `% unique nt` = `Unique aa` / `Input rows` (and the
  same for nt). Read as the **diversity/uniqueness rate**: high % =
  more diverse population at the sequence level, low % = heavily
  redundant population.

(`unique_sequence_retention.png` plots the same data as grouped bars,
one panel per alphabet.)

**Interpretation.** Two regularities and one anomaly:

- **`% unique aa` is always lower than `% unique nt`** (column 5 <
  column 7) on 7 of 8 functions. Synonymous codons create distinct
  CDS DNAs that collapse to one protein, so the unique-nt count is
  always ≥ the unique-aa count for the same isolate population.
  Magnitude varies: on M1 the unique-nt count is ~7× the unique-aa
  count (synonymous variation accumulates more on the most conserved
  protein, because aa changes are strongly purifying-selected); on
  HA the ratio is closer to 1.6× (HA has substantial aa-level
  variation per se).
- **M1 is the most-redundant aa case.** Only 4,771 distinct M1 aa
  sequences across 108,530 isolates — 96% redundancy. M1 is the most
  aa-conserved Flu A protein, consistent with literature on its role
  in particle structure (high constraint, low aa drift).
- **NS1 is the inflated-aa-uniqueness case.** NS1 (median 231 aa) is
  shorter than M1 (median 253 aa) yet has *4.7×* the unique-aa-sequence
  count (22,225 vs 4,771). Short conserved proteins should give FEWER
  unique sequences, not more. The reason is **length variation**: NS1
  is the only major with substantial per-sequence length spread (aa
  range 201–239, ~20-aa variation, vs ≤2 aa for the others —
  `sequence_length_summary.csv`). Two NS1 proteins differing by one
  residue at the C-terminus hash to different `seq_hash` values
  regardless of their interior similarity. So NS1's "unique aa count"
  partly reflects length diversity at the threshold-1.0 read, not
  residue diversity. Clustering at id < 1.0 collapses these
  length-variants quickly (see §6.1).

> **NA stalk-length variation — important caveat for reading later tables.**
> NA's `% unique aa` (34.5%) and `Unique aa` (37,488) here are *pre-clustering*
> dedup counts. In §6.1, NA's `n_clusters` at id100 drops sharply to ~18,753
> — about half the unique-aa count. This is **not** an algorithm-specific
> artifact: influenza NA has a transmembrane stalk that varies substantially
> in length across HxNy subtypes and within subtypes (deletions/insertions
> in the stalk region are common). Under the §3.2 coverage rule
> (bidirectional ≥80%), an NA with a stalk deletion is still 100% identical
> to a longer NA over the aligned region; the two cluster together at id100.
> So NA's *cluster count* at id100 (and downstream thresholds) reflects
> stalk-length collapse *in addition* to sequence-diversity collapse. The
> better tightness/diversity metric for cross-function comparison is
> `% unique aa` from this table, not `n_clusters` at id100 from §6.1.

---

## 5. What an identity threshold concretely admits

**Source.** `cluster_analysis_summary.py`. `mutations_tolerated_table.csv`
contains values for idXX. `sequence_length_summary.csv` contains sequence
length values.

For an L-residue sequence and threshold t,
`max_mismatches_inside_cluster = L − ceil(L × t)`. The same
threshold is therefore looser on long proteins and stricter on short
ones. Concrete numbers per function:

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

Each table cell is `L − ceil(L × t)`, the maximum admitted residue
mismatches inside a cluster at identity threshold `t` for the
median-length sequence `L`. In mmseqs, threshold `t` is by arg `--min-seq-id`.

**Interpretation.** "id095" is not one biological criterion — it's
eight different ones (one per function/segment) when read this way.
id095 on M1 is **3× stricter** than id095 on PB2 because M1 is roughly 3×
shorter. This explains why per-function cluster collapse rates in §6
differ between functions at the same threshold: the threshold itself
admits more mutations on the longer proteins, so more sequences fall
into the same cluster.

**A framing observation, not yet validated.** For thinking about
*biological similarity*, the mutations-admitted number is sometimes
argued to be more informative than the bare identity threshold —
particularly when comparing across function pairs of different
lengths (e.g., HA/NA's 470–567 aa range vs PB2/PB1's 758–760 aa
range). This is a working hypothesis on our pipeline, not an
established result. The operational consequence — picking per-function
thresholds to enforce a uniform absolute mismatch budget rather than
a uniform fractional identity — is tracked as an improvement direction
in `docs/results/2026-05-21_bicc_pair_drop_audit.md` (direction #5).
Open question we have not resolved: under what task framings
(per-site divergence vs functional impact vs evolutionary distance)
is fractional vs absolute the right unit? And does uniform-absolute
really treat each residue equally given per-protein evolutionary-rate
variation?

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

### 6.2 Two collapse modes (one deferred cliff, one gradual)

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

### 6.3 Largest cluster as % of corpus

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

### 6.4 Why this matters for routing

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
  on the aa side, see §4 footnote and §6.3). Routing still has room.
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
§6.4). The aa-vs-nt feasibility gap at id099 (aa near ceiling, nt
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
