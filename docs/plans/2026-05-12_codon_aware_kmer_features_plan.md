# Codon-aware k-mer features — Situation & Plan

**Status: PROPOSED** (no implementation; this is an interpretability/feasibility note)
**Date:** 2026-05-12

**See also:** `docs/gto_format_reference.md` — comprehensive walk-through
of the GTO JSON schema, the `contigs` / `features` split, the `location`
field (including the spliced multi-entry case for M2 / NEP), and the
"major protein" convention from `conf/virus/flu.yaml`. That document is
the metadata-availability baseline this plan builds on.

## TL;DR

The current k-mer features are **bag-of-k-mer counts over the full contig
DNA**, sliding window with stride 1. Positions are not preserved, frames
are summed, and UTR + (for spliced segments) intronic regions are
included. As a result, attribution on a k-mer column tells you only
*"the model uses the count of nucleotide k-mer X in this contig"* — it
cannot be safely re-interpreted as a codon, a frame, an amino-acid
position, or a synonymous/nonsynonymous signal. The metadata to build
**codon-aware**, CDS-scoped, stride-3 features already exists in
`protein_final.csv` (`location` field); adding such a path is
methodologically valid and self-contained. Position-resolved (per-codon
or per-AA) attribution is a larger change and not in scope here.

## Why this came up

Reviewers and presentation prep keep asking whether interaction-term
importances (`prod`, `diff`, `unit_diff`) reflect *codon* signal — e.g.
codon-usage differences between segments — or just generic nucleotide
composition. This note records what we can and cannot currently say, so
we have a reference before any experiment starts using k-mer attributions
as biological evidence.

## Current state (what exists)

### K-mer generation (`src/embeddings/compute_kmer_features.py`)

- **Source sequence**: `genome_final.csv['dna_seq']` — the full deposited
  contig (BV-BRC). Includes 5′/3′ UTRs and the unspliced contig for M
  (M1/M2) and NS (NS1/NEP), where the two exons of the spliced product
  are separated by an intron in the contig.
- **Window**: sliding, **stride hard-coded to 1**
  (`for i in range(len(seq_upper) - k + 1)`).
- **Frame**: none — all three reading frames + UTR k-mers sum into the
  same count vector.
- **Vocabulary**: lexicographic enumeration of `|alphabet|^k`
  (default `ACGT`, 4^k columns; protein alphabet `ACDEFGHIKLMNPQRSTVWY`
  also supported as of 2026-05-12). Column index → k-mer string is
  recoverable via `build_kmer_vocabulary(k)`.
- **Output**: only counts. No per-occurrence position is preserved.
  - `kmer_features_k{k}.npz` — sparse CSR `(N_segments, 4^k)` float32.
  - `kmer_features_k{k}_index.parquet` — row → `(assembly_id,
    genbank_ctg_id, canonical_segment)`.
  - `kmer_features_k{k}_metadata.json` — `k`, `normalize`, `nnz`, etc.

### Metadata available per CDS (`protein_final.csv`)

- `location` — list of `[contig_id, start_1based, strand, length]`
  tuples. **Single entry** for unspliced CDSs, **multi-entry** for
  spliced products (M2, NEP). Example M2:
  `[['ctg', '14', '+', 26], ['ctg', 728, '+', 268]]` — exon1 26 nt at
  start 14, exon2 268 nt at start 728. 26 + 268 = 294 nt = 98 codons
  (97 AA + stop ✓).
- `genbank_ctg_id`, `canonical_segment`, `function`, `brc_fea_id`,
  `prot_seq`, `esm2_ready_seq`.
- `genetic_code` = 11 in the GTOs (a BV-BRC artifact; flu uses code 1
  in practice).
- All flu CDSs observed are on the `+` strand, but the schema admits `-`.

## What we can / cannot map today

| # | Target | Reachable from current artifacts? |
|---|---|---|
| 1 | Nucleotide k-mer sequence (column → string) | **Yes** — `build_kmer_vocabulary(k)[col_idx]` |
| 2 | Source contig / segment | **Per row, yes** (index parquet); **per occurrence, no** — counts collapse position. A row is a *contig*, which for PB1 contains PB1 + PB1-F2 + PB1-N40, etc. |
| 3 | Strand orientation | In `location` but not consulted at k-mer time |
| 4 | CDS start/end coordinates | In `location` per CDS; not joined to k-mer features |
| 5 | Coding frame | **No** — stride=1 sums all three frames |
| 6 | Codon position | **No** — position is discarded |
| 7 | Amino-acid position | **No** — same reason |
| 8 | Synonymous / nonsynonymous context | **No** — would need per-segment reference alignment infra, which does not exist |

## What this means for feature-importance interpretation

A high attribution on column `j` of the current k-mer features means:

> "Across the entire contig (UTRs + all three frames + introns where
> applicable), the count of nucleotide k-mer `vocab[j]` differs between
> the two segments in a way the model uses."

This is **nucleotide-composition signal at contig scope**, not codon
signal. In particular:

- The same 6-mer in frame 1 of CDS, in frame 2 of CDS, and in an UTR all
  add into one column.
- PB1's contig contains PB1, PB1-F2, and PB1-N40 — all three CDSs
  contribute to the same row.
- M and NS contigs contribute intronic nucleotides in the k-mer count,
  in addition to the spliced exon nucleotides.

## Is codon-aware (stride-3, CDS-scoped) extraction valid?

Yes, methodologically — it is the standard codon-usage / codon-bias
feature representation. The work splits into three parts.

### 1. Source-sequence change: contig → CDS

For each protein row, slice `genome_final['dna_seq']` using `location`
(concatenating exons for multi-entry locations), reverse-complement if
strand=`-`. This yields the in-frame CDS DNA. After extraction, frame 0
is the correct reading frame by construction.

### 2. Stride / k design choices

Three reasonable variants:

- **`k=3, stride=3` → 64-dim codon-frequency vector per CDS.** Direct
  codon-bias features. Not redundant with AA identity because
  synonymous codons separate. Smallest, easiest to interpret.
- **`k=6, stride=3` → 4096-dim codon-dimer vector per CDS.** Captures
  adjacent-codon co-occurrence; sensitive to codon-context bias. Matches
  the current k=6 dimensionality so most downstream code reuses unchanged.
- **`k=6, stride=1, in-frame only` → 4096-dim.** Drops cross-frame and
  UTR noise but keeps overlapping-window resolution. Most similar to
  current features in spirit; cleanest "all else equal" comparison.

### 3. Row-identity change

The unit of analysis becomes a **CDS**, not a contig. Index switches
from `(assembly_id, genbank_ctg_id, canonical_segment)` to one row per
`brc_fea_id`. This is a real schema change for Stage 3 — pair tables
already carry protein identities, so the join is feasible, but
`kmer_utils.load_kmer_index` and the composite-key construction in
`get_kmer_pair_features` need a CDS-keyed mode.

A useful intermediate path: write a *second* feature matrix alongside
the contig-based one, so the comparison is "same dataset, two feature
sets" without disturbing existing experiments.

## What stride-3 still does NOT give you

Even after switching to CDS + stride-3:

- **Per-codon or per-AA position** — features remain a bag. To get
  position-resolved attribution you need either per-codon-window
  features (lots of columns, alignment-dependent) or a position-aware
  model (e.g. attribute through ESM-2 directly, which is per-residue).
- **Synonymous / nonsynonymous tagging** — requires a reference CDS per
  `(segment, subtype)` and pairwise alignment. Not currently in the
  codebase.

## Empirical UTR / non-coding fraction in this corpus

The audit step originally listed as "recommended diagnostic" has now
been run (see `docs/gto_format_reference.md` §6.3 and §6.5 for the full
tables). Key numbers, from 20,000 randomly sampled contigs:

- **Non-coding fraction per segment, median:** 0.61% (S1), 1.13% (S2),
  1.87% (S3), 1.16% (S4), 1.77% (S5), 0.67% (S6), 1.11% (S7), 3.01% (S8).
- **Across all 8 segments the median non-coding fraction is ≤ 3.01%.**
- 5′ UTR median: 0–25 nt per segment. 3′ UTR median: 1–31 nt per
  segment.
- Long-tail 3′ UTR maxima on S7 (243 nt) and S8 (207 nt) arise from a
  small fraction of contigs that extend well past the spliced product's
  end, but do not move the medians.

**Implication for this plan.** The motivation "switch from contig to
CDS to remove UTR noise" is materially weaker than initially scoped,
because UTR noise in this corpus is small. The bulk of any difference
between contig-k-mer and CDS-k-mer features would come from
**reading-frame alignment**, not UTR exclusion. Specifically:

- Today (stride=1, contig-source): the count for any k-mer is summed
  over **3 reading frames + UTR + intron** windows. From the per-segment
  non-coding-fraction medians above, **96.99–99.39%** of contig
  nucleotides sit inside an annotated CDS, but the stride-1 windows
  starting at those nucleotides are split across all three reading
  frames.
- With stride=3 in-frame on the CDS: the count for any k-mer is the
  number of times it appears in frame 0 specifically. That's a ~3×
  reduction in count per k-mer, and the resulting features carry a
  cleaner "codon-usage in-frame" signal.

So the primary lever is the **stride change**, with the CDS-source
change being a secondary refinement (~1–3% noise reduction at most).
This reweighting changes the recommended order below.

## Minimal code changes (scoping, not implementation)

If we decide to add codon-aware k-mer features, the minimum changes are:

1. **`src/embeddings/compute_kmer_features.py`**
   - Add `stride: int = 1` to `compute_kmer_counts` and
     `sequences_to_sparse_kmer_matrix`. **This alone (without CDS
     extraction) buys most of the codon signal**, by counting only
     non-overlapping windows that are frame-aligned to the contig.
     Frame-0 windows in CDS-rich contigs are dominated by in-frame
     codons (since CDS coverage is 97–99%).
   - Add a CDS-extraction helper for the cleaner variant: parse
     `location` from `protein_final.csv`, slice the matching contig DNA
     from `genome_final.csv`, concatenate exons for multi-entry
     locations, reverse-complement if strand=`-` (no `-` strand
     observed in flu).
   - Either a new CLI mode or a sibling script that drives the CDS path
     and writes `kmer_features_k{k}_stride{s}_cds.npz` plus an index
     keyed by `brc_fea_id`.

2. **`src/utils/kmer_utils.py`**
   - `load_kmer_index`: support a CDS-keyed mode (`brc_fea_id` → row)
     in addition to the current contig-keyed mode.
   - `get_kmer_pair_features`: pair-key construction would switch from
     `(assembly_id, ctg)` to `brc_fea_id` when the CDS artifact is in
     use; otherwise unchanged.

3. **`src/datasets/dataset_segment_pairs*.py`**
   - Pair tables already carry protein identities; ensure `brc_fea_id`
     (or an equivalent key) is propagated so the CDS lookup has a join
     key.

4. **`src/models/_pair_features.py`, `train_pair_baselines.py`**
   - Config knob like `kmer.source: contig | cds` and
     `kmer.stride: 1 | 3`, wired through to the loader. No change to
     interaction / slot logic.

## Recommended order if we proceed

Reordered relative to the original plan based on the UTR-fraction
audit above:

1. **Stride parameter on the contig source first.** Add
   `stride: int = 1` to `compute_kmer_counts` and re-run k-mer
   extraction with `stride=3` on the existing contig DNA. This is
   the smallest possible code change (one parameter), no schema
   change, no new artifact key. Compare to current contig/stride=1
   features.
2. **CDS-extraction + stride=3 as a separate sibling artifact.**
   Add the CDS-extraction helper and write
   `kmer_features_k{k}_stride3_cds.npz` keyed by `brc_fea_id`.
   Compare to step 1: the difference between (stride=3, contig)
   and (stride=3, CDS) isolates the ~1–3% UTR/intron contribution.
3. **Per-frame multi-channel features (optional).** Instead of
   stride=3 frame-0 only, emit three separate count vectors (frame 0,
   frame 1, frame 2). This preserves the full per-k-mer count while
   tagging the frame. Useful if downstream attribution wants to
   distinguish in-frame from out-of-frame signal explicitly.
4. **Only after step 1 or 2 has produced features with a clear
   codon-attribution claim**, consider position-resolved attribution
   (which is a different feature class entirely; likely better
   served by ESM-2 residue attribution than by k-mer encodings).

## Out of scope here

- Any position-aware (per-codon or per-AA) k-mer encoding.
- Synonymous / nonsynonymous tagging via reference alignment.
- Cross-subtype codon-usage normalization.
- Any change to the ESM-2 embedding path.
