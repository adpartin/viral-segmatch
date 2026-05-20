# Clustering overview (aa and nt) for cluster_disjoint splits

This is the methods reference for what protein/nucleotide sequence
clustering does in this pipeline, what mmseqs2's parameters
concretely mean, and how a per-function clustering becomes the
train/val/test partition the model actually sees. It is written for a
reader who has not used mmseqs2 before.

For the *experimental findings* on Flu A see
`docs/results/2026-05-15_cluster_disjoint_nt_results.md` (model-side
results) and the two redundancy autogen docs
`docs/results/2026-05-15_seq_redundancy_per_function.md` (aa) and
`docs/results/2026-05-15_seq_redundancy_per_function_nt.md` (nt). This
doc explains the *machinery*; those docs report the *numbers we got*.

For the exact routing-equivalence semantics across the four
implemented modes (seq_disjoint hash_key={seq,dna}, cluster_disjoint
cluster_alphabet={aa,nt}), see `docs/methods/leakage_definitions.md`
§ "Routing equivalence and mmseqs argument semantics".

---

## 1. Why we cluster in this pipeline

The classifier's job is to predict whether two protein segments
co-occur in the same flu isolate. A random train/val/test split leaks
near-neighbor information: test pairs typically have a train neighbor
that's 99–100% identical at the protein level, and the model can
exploit that without learning any pair-co-occurrence biology. This is
leakage mode #4 ("cluster leakage" / "near-neighbor leakage") in
`docs/methods/leakage_definitions.md`.

Clustering puts a *radius* around each protein: all sequences within
the radius collapse to one cluster id. We then partition pairs so that
no cluster id appears on both sides of the train/test boundary. That
mitigates mode #4 by construction at the chosen radius.

The clustering itself happens once per (data version, alphabet, identity
threshold). Stage 3 reads the cluster lookups when
`dataset.split_strategy.mode: cluster_disjoint` is set in the bundle.

---

## 2. Sequence-space clustering 101

### 2.1 What "similarity" means

mmseqs2 reduces a pair of biological sequences to a single number in
[0, 1] called **percent identity** — the fraction of aligned positions
that match between two sequences. 0.95 identity means at most ~5% of
positions disagree.

The match unit is **residues**: amino acids for proteins, nucleotides
for DNA. Length is counted in residues too. So "id 0.95" on a 760-aa
PB2 protein admits ~38 aa mutations within a cluster; "id 0.95" on a
2,280-nt PB2 CDS admits ~114 nt mutations.

The same threshold is **biologically stricter on shorter proteins**
(fewer absolute mutations admitted). See § 7 for a per-function table.

### 2.2 Why not just compare every pair to every pair?

On the Flu A July 2025 corpus the protein side has 108,530 isolates ×
8 selected functions = 868,240 protein records. Even after deduplication,
HA has ~42,000 unique aa sequences and ~65,000 unique nt CDS. Computing
exact alignments for every pair would be 65,000² / 2 ≈ 2 × 10⁹
alignments. Infeasible.

mmseqs2 sidesteps the quadratic with a **k-mer prefilter + alignment
cascade**:

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
   │ Step 3. Cluster assignment│  Pick a cluster representative per
   │                           │  connected component of the high-
   │   each member joins its   │  scoring graph; assign every member
   │   representative          │  to its representative.
   │                           │
   └─────────────┬─────────────┘
                 │
                 ▼
   cluster_tsv  (one row per member: rep_id ⟷ member_id)
```

The prefilter is the load-bearing step: it converts the quadratic
all-pairs problem into a roughly linear "look up similar k-mers"
problem. The trade-off is that sequences below the prefilter score
threshold never reach the alignment step, so a fast prefilter can miss
true matches (sensitivity loss).

### 2.3 easy-cluster vs easy-linclust

mmseqs2 ships two main entry points for unsupervised clustering:

- **`easy-cluster`** — runs the cascade above in multiple sensitivity
  passes (cascaded clustering with optional re-assignment). High
  sensitivity. Slower. Used by this pipeline for the aa side.
- **`easy-linclust`** — a linear-time variant that picks a smaller,
  faster prefilter and skips the cascade. Lower sensitivity at low
  identity thresholds, comparable cluster counts at higher thresholds
  on our corpus. Used by this pipeline for the nt side.

Measured cost on the Flu A July 2025 corpus
(`data/processed/flu/{version}/clusters_aa/runtime.json` and
`clusters_nt/runtime.json`, both written by
`src/analysis/seq_redundancy_per_function.py`):

| Alphabet | Algorithm | n_runs | Total (s) | Median/run (s) | Max/run (s) |
|---|---|---:|---:|---:|---:|
| aa | easy-cluster | 50 (10 fn × 5 th) | 3,011 | 4.8 | 570 (PA @ id100) |
| nt | easy-linclust | 48 (8 fn × 6 th) | 986 | 6.7 | 217 (PB1 @ id100) |

Both runs used `--threads 8` and were issued separately (no
parallelism). The aa side spent most of its time on the high-identity
thresholds where the cascaded prefilter examines many candidate pairs.
The nt side avoids that cost by construction.

**Choice on Flu A:** aa stays on easy-cluster because it's already
fast at this corpus size (median < 5 s) and gives the more sensitive
answer. nt is on easy-linclust because easy-cluster on full-length CDS
(2,000–2,300 nt) was hitting wall-clock costs that did not scale to
24-run sweeps. Cluster counts on overlapping (function, threshold)
cells agreed within noise between the two algorithms during a
side-by-side check at id ≥ 0.80.

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
(aa for protein, nt for DNA). At t = 1.0, only exact matches cluster
(modulo coverage and ambiguity handling).

The biological meaning is per-function — the *same threshold* admits
different numbers of mutations on a 760-aa PB2 protein vs a 252-aa
M1 protein:

| Function | Median aa length | Max aa changes at id095 | Max aa changes at id090 |
|---|---:|---:|---:|
| PB2 | 760 | 38 | 76 |
| PB1 | 758 | 37 | 75 |
| PA  | 717 | 35 | 71 |
| HA  | 567 | 28 | 56 |
| NP  | 499 | 24 | 49 |
| NA  | 470 | 23 | 47 |
| M1  | 253 | 12 | 25 |
| NS1 | 231 | 11 | 23 |

(From `results/.../cluster_analysis/mutations_tolerated_table.csv`,
generated by `src/analysis/cluster_analysis_summary.py`. nt
equivalents are ~3× larger, also in that CSV.)

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

## 4. From a per-function clustering to a train/val/test partition

### 4.1 Why per-function clusters aren't enough

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

### 4.2 80/10/10 by LPT-greedy

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

### 4.3 The four implemented routings — quick reference

| mode + alphabet | slot key | two pairs share a CC iff … |
|---|---|---|
| `seq_disjoint` `hash_key=seq` (default) | `seq_hash = md5(prot_seq)` exact | identical protein on a slot |
| `seq_disjoint` `hash_key=dna` | `dna_hash = md5(contig.dna)` exact | identical full contig (UTR + CDS + intron + UTR) on a slot |
| `cluster_disjoint` `cluster_alphabet=aa` | mmseqs2 aa cluster id at chosen threshold | aa-similar protein on a slot |
| `cluster_disjoint` `cluster_alphabet=nt` | mmseqs2 nt cluster id at chosen threshold, keyed on `cds_dna_hash` | nt-similar CDS DNA on a slot (UTRs and introns excluded) |

Deep dive on equivalences and non-equivalences:
`docs/methods/leakage_definitions.md` § "Routing equivalence and
mmseqs argument semantics".

### 4.4 Naming: bipartite-CC LPT-greedy ≈ cluster_disjoint routing ≈ BiCC-Split

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

## 5. Pipeline integration

Three scripts handle the producer/consumer chain, all under
`src/analysis/` (the cluster artifacts they produce are inputs to
Stage 3 via `src/datasets/_split_helpers.py`):

| Step | Script | Reads | Writes |
|---|---|---|---|
| Build CDS (nt only) | `src/preprocess/extract_cds_dna.py` (Stage 1.5) | `protein_final.csv` + `genome_final.csv` | `cds_final.parquet` |
| Cluster sweep | `src/analysis/seq_redundancy_per_function.py` | `protein_final.parquet` (aa) or `cds_final.parquet` (nt) | `clusters_aa/` or `clusters_nt/` (per-function FASTAs, per-threshold cluster parquets, `combined_cluster.parquet`, `redundancy_stats.csv`, `runtime.json`, autogen `_seq_redundancy_per_function*.md`) |
| Feasibility pre-flight | `src/analysis/cluster_disjoint_feasibility.py` | one cluster lookup + `protein_final` or `cds_final` | per-pair `_cluster_disjoint_feasibility_*.csv` under `docs/results/` |
| Stage 3 consumes | `src/datasets/dataset_segment_pairs_v2.py` (when `split_strategy.mode: cluster_disjoint`) | `combined_cluster.parquet` for the chosen (alphabet, threshold) | `dataset_*/cluster_disjoint_audit.json` |

For consolidated structural analysis (the next section's tables and
plots), the post-hoc script is
`src/analysis/cluster_analysis_summary.py`, which reads the
redundancy + feasibility CSVs and emits plots under
`results/flu/{version}/runs/cluster_analysis/`.

---

## 6. Per-function corpus redundancy (Flu A July 2025)

Source: `results/flu/July_2025/runs/cluster_analysis/cluster_summary.csv`
(from `cluster_analysis_summary.py`, which reads
`clusters_{aa,nt}/redundancy_stats.csv` written by
`seq_redundancy_per_function.py`).

Per-function unique-sequence counts at threshold = 1.00 (i.e., exact
identity clustering — every cluster has all-identical members):

| Function | Input rows | Unique aa | aa retention | Unique nt | nt retention |
|---|---:|---:|---:|---:|---:|
| PB2 | 108,530 | 33,663 | 31.0% | 67,341 | 62.1% |
| PB1 | 108,530 | 31,226 | 28.8% | 67,034 | 61.8% |
| PA  | 108,530 | 34,217 | 31.5% | 65,242 | 60.1% |
| HA  | 108,530 | 41,896 | 38.6% | 65,414 | 60.3% |
| NP  | 108,530 | 17,684 | 16.3% | 52,800 | 48.6% |
| NA  | 108,530 | 37,488 | 34.5% | 58,887 | 54.3% |
| M1  | 108,530 |  4,771 |  4.4% | 32,413 | 29.9% |
| NS1 | 108,530 | 22,225 | 20.5% | 38,039 | 35.0% |

(`unique_sequence_retention.png` plots the same data as grouped bars,
one panel per alphabet.)

**Interpretation.** Two regularities and one anomaly:

- **aa retention is always lower than nt retention** (column 3 < column 5)
  on 7 of 8 functions. Synonymous codons create distinct CDS DNAs that
  collapse to one protein, so the nt count is always ≥ the aa count
  for the same isolate population. Magnitude varies: on M1 the nt
  count is ~7× the aa count (synonymous variation accumulates more on
  the most conserved protein, because aa changes are strongly purifying
  selected); on HA the ratio is closer to 1.6× (HA has substantial
  aa-level variation per se).
- **M1 is the extreme**. Only 4,771 distinct M1 aa sequences across
  108,530 isolates — 96% redundancy. M1 is the most aa-conserved Flu
  A protein, consistent with literature on its role in particle
  structure (high constraint, low aa drift).
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
  length-variants quickly (see §8 — NS1 drops from 21,864 aa clusters
  at id100 to a few clusters at id080).

---

## 7. What an identity threshold concretely admits

For an L-residue sequence and threshold t,
`max_mismatches_inside_cluster = floor(L × (1 − t))`. The same
threshold is therefore looser on long proteins and stricter on short
ones. Concrete numbers per function:

| Function | Median aa len | id100 (aa) | id099 (aa) | id095 (aa) | id090 (aa) | id080 (aa) |
|---|---:|---:|---:|---:|---:|---:|
| PB2 | 760 | 0 |  7 | 38 | 76 | 152 |
| PB1 | 758 | 0 |  7 | 37 | 75 | 151 |
| PA  | 717 | 0 |  7 | 35 | 71 | 143 |
| HA  | 567 | 0 |  5 | 28 | 56 | 113 |
| NP  | 499 | 0 |  4 | 24 | 49 |  99 |
| NA  | 470 | 0 |  4 | 23 | 47 |  94 |
| M1  | 253 | 0 |  2 | 12 | 25 |  50 |
| NS1 | 231 | 0 |  2 | 11 | 23 |  46 |

| Function | Median nt len | id100 (nt) | id099 (nt) | id095 (nt) | id090 (nt) | id080 (nt) |
|---|---:|---:|---:|---:|---:|---:|
| PB2 | 2,280 | 0 | 22 | 114 | 228 | 456 |
| PB1 | 2,274 | 0 | 22 | 113 | 227 | 454 |
| PA  | 2,151 | 0 | 21 | 107 | 215 | 430 |
| HA  | 1,701 | 0 | 17 |  85 | 170 | 340 |
| NP  | 1,497 | 0 | 14 |  74 | 149 | 299 |
| NA  | 1,410 | 0 | 14 |  70 | 141 | 282 |
| M1  |   759 | 0 |  7 |  37 |  75 | 151 |
| NS1 |   693 | 0 |  6 |  34 |  69 | 138 |

(Both tables: `mutations_tolerated_table.csv` from
`cluster_analysis_summary.py`. Lengths from
`sequence_length_summary.csv` in the same dir.)

**Interpretation.** "id095" is not one biological criterion — it's
seven different ones (one per function) when read this way. id095 on
M1 is **3× stricter** than id095 on PB2 because M1 is roughly 3×
shorter. This explains why per-function cluster collapse rates in §8
differ between functions at the same threshold: the threshold itself
admits more mutations on the longer proteins, so more sequences fall
into the same cluster.

For thinking about *biological similarity*, prefer the
mutations-admitted number over the bare identity threshold —
especially when comparing across function pairs of different lengths
(e.g., HA/NA's 470–567 aa range vs PB2/PB1's 758–760 aa range).

---

## 8. Cluster collapse trajectory

Source: `cluster_counts_vs_threshold.png` (from
`cluster_analysis_summary.py`); raw values in `cluster_summary.csv`.

The plot shows per-function `n_clusters` (log Y) as the identity
threshold drops from 1.00 to 0.80. All 16 lines (8 functions × 2
alphabets) trace a roughly sigmoidal collapse:

- **At id100** clusters ≈ unique sequences (singleton-heavy: 99% of
  clusters on each function are size-1 at this threshold;
  `fraction_singletons` column in `cluster_summary.csv`).
- **By id095** clusters drop ~10–100× on aa, 10–100× on nt (most aa
  near-clones merge; nt near-clones too, but the nt side started with
  more unique sequences so it collapses to a higher absolute count).
- **By id080** every function is down to single-digit cluster counts
  on both alphabets, except HA and NA (which retain ~85–135 clusters
  on nt — HA and NA are the most aa-diverse and that diversity
  partially survives nt clustering at this stringency).

**Interpretation.**

The aa lines (solid markers) and nt lines (dashed markers) for the
same function track each other in shape but the nt line sits ~2–10×
higher on the Y axis at id095 and below. That's the synonymous-codon
diversity: at id095 nt, two proteins with the same aa but distinct
codon usage are still in different clusters.

By id080 both alphabets converge to ~10 clusters per function on the
polymerase subunits — the threshold is loose enough that whatever
nucleotide diversity remained is below 80% identity, so the same big
biological "subtype-cluster" structure dominates regardless of
alphabet.

The shape of the collapse is corpus-driven, not algorithm-driven:
easy-cluster (aa) and easy-linclust (nt) both produce similar collapse
trajectories on their respective alphabets.

---

## 9. Bipartite-component feasibility ceiling on Flu A

The §4.2 question — "is the largest CC small enough for 80/10/10?" —
answered per (schema pair, alphabet, threshold). Source: the four
feasibility CSVs under `docs/results/`, generated by
`src/analysis/cluster_disjoint_feasibility.py`. Consolidated plot:
`bipartite_largest_pct_vs_threshold.png` from
`cluster_analysis_summary.py`.

Largest bipartite-component fraction (% of deduped pairs):

| Schema pair | Alphabet | id100 | id099 | id095 | id090 | id085 | id080 |
|---|---|---:|---:|---:|---:|---:|---:|
| HA/NA   | aa | 20.2 | 80.0 | 98.5 | 99.3 | n/a  | 100.0 |
| HA/NA   | nt | 1.5  | 69.4 | 98.2 | 99.1 | 99.6 | 100.0 |
| PB2/PB1 | aa | 38.4 | 87.1 | 100.0 | 100.0 | n/a  | 100.0 |
| PB2/PB1 | nt | 2.9  | 59.7 | 99.1  | 99.5  | 100.0 | 100.0 |

A cell is structurally feasible for 80/10/10 if the largest CC is ≤80%
(train can fit it). Looking at the table:

- **id100 (every cell):** feasible. Largest CC is at most 38% (PB2/PB1
  aa). Routing has room.
- **id099 (every cell):** marginal — values cluster around 60–87%.
  HA/NA aa at 80.0% and PB2/PB1 aa at 87.1% are right at or above the
  ceiling; the LPT-greedy bin-packer can sometimes squeeze them in
  thanks to second-place CCs being small (5–6% on HA/NA, 0.3% on
  PB2/PB1), but the val/test deficits get filled from smaller
  components and the 80/10/10 ratios drift slightly.
- **id095 and below:** infeasible across every cell. The largest CC
  exceeds 98% of pairs on every line. There's no way to assign that
  one CC to a single bin without violating either the 80% train cap
  or starving val/test.

**Interpretation: the feasibility ceiling is corpus-driven, not
alphabet-driven.** Aa and nt curves cross the 80% line at the same
threshold (between id099 and id095) on both schema pairs. The
expectation going in was that nt clustering would unlock lower
thresholds via synonymous diversity. It doesn't, because the *bipartite
linking* between HA clusters and NA clusters is determined by which
isolates carry which (HA, NA) combinations — and Flu A's small set of
dominant HxNy subtypes × host × year cells links most pairs into one
mega-component long before the cluster-level diversity differences
between aa and nt matter.

The empirical confirmation is in
`docs/results/2026-05-15_cluster_disjoint_nt_results.md`: the
production B-nt experiment was limited to (id100, id099) on both
alphabets, mirroring the aa feasibility ceiling exactly.

---

## 10. Regenerating everything

The producer/consumer chain, end to end:

```bash
# 1. (nt-only) Build cds_final.parquet — once per data version.
python src/preprocess/extract_cds_dna.py --config_bundle flu_ha_na

# 2. Per-function clustering sweep — once per (alphabet, data version).
#    Writes <out_root>/{fasta,id<NN>}/, redundancy_stats.csv, runtime.json,
#    and an autogen markdown under docs/results/.
python -m src.analysis.seq_redundancy_per_function \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --out_root      data/processed/flu/July_2025/clusters_aa \
    --thresholds 1.00 0.99 0.95 0.90 0.80 \
    --functions HA NA PB2 PB1 PA NP M1 M2 NEP NS1 \
    --threads 8

python -m src.analysis.seq_redundancy_per_function \
    --cds_final  data/processed/flu/July_2025/cds_final.parquet \
    --out_root   data/processed/flu/July_2025/clusters_nt \
    --thresholds 1.00 0.99 0.95 0.90 0.85 0.80 \
    --functions HA NA PB2 PB1 PA NP M1 NS1 \
    --algorithm linclust \
    --threads 8

# 3. Bipartite-component feasibility per schema pair × alphabet.
python -m src.analysis.cluster_disjoint_feasibility \
    --protein_final data/processed/flu/July_2025/protein_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_aa \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.95 0.90 0.80 \
    --out_csv docs/results/2026-05-14_cluster_disjoint_feasibility_ha_na.csv

python -m src.analysis.cluster_disjoint_feasibility \
    --cds_final     data/processed/flu/July_2025/cds_final.parquet \
    --clusters_root data/processed/flu/July_2025/clusters_nt \
    --schema_pair "Hemagglutinin precursor" "Neuraminidase protein" \
    --thresholds 1.00 0.99 0.95 0.90 0.85 0.80 \
    --out_csv docs/results/2026-05-15_cluster_disjoint_feasibility_nt_ha_na.csv

# (repeat for PB2/PB1)

# 4. Consolidated structural summary (table + 3 plots).
python -m src.analysis.cluster_analysis_summary
```

Outputs land under
`results/flu/July_2025/runs/cluster_analysis/`:

```
cluster_summary.csv                       — per (function, alphabet, threshold)
sequence_length_summary.csv               — per (function, alphabet)
mutations_tolerated_table.csv             — concrete max-mismatches table
unique_sequence_retention.png             — Plot A (§6)
cluster_counts_vs_threshold.png           — Plot B (§8)
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
- `docs/results/2026-05-15_seq_redundancy_per_function.md` (aa) and
  `docs/results/2026-05-15_seq_redundancy_per_function_nt.md` (nt) —
  autogen per-threshold cluster-size tables for all 8 majors.
- `docs/results/2026-05-14_cluster_disjoint_feasibility_{ha_na,pb2_pb1}.csv`
  (aa) and
  `docs/results/2026-05-15_cluster_disjoint_feasibility_nt_{ha_na,pb2_pb1}.csv`
  (nt) — raw bipartite-CC feasibility numbers per pair × alphabet ×
  threshold.
- `docs/plans/2026-05-08_cosine_and_cluster_splits_plan.md` — the
  original plan that motivated this machinery.
