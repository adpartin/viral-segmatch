# GTO file format — reference for `viral-segmatch`

**Scope.** Curated Genome Typed Object (GTO) JSON files that feed Stage 1
of this pipeline. Our focus is on **Influenza A**; Bunyavirales GTOs use
pretty much same workflow but are not currently maintained (see
CLAUDE.md → "What Is NOT Maintained").

**Companion files.**
- `docs/examples/example_gto_excerpt.json` — a trimmed, real GTO (assembly
  `0000003012`, A/mallard/Wisconsin/394/1977 H4N1) showing the fields
  this document references. DNA / protein sequences truncated for
  readability; provenance fields omitted.
- `docs/plans/2026-05-12_codon_aware_kmer_features_plan.md` — the
  consumer of this reference. Explains why we currently cannot interpret
  k-mer feature importances at codon level, and what would change.
- `src/preprocess/preprocess_flu.py` — reads GTO JSON directly. Section
  "What Stage 1 parses" below has file:line pointers.
- `conf/virus/flu.yaml` — the canonical list of flu protein functions
  (`core_functions`, `aux_functions`, `selected_functions`,
  `function_short_names`) and the conditional segment-mapping rules.
  Several parts of this doc cross-reference it.

---

## 1. What a GTO is

A GTO ("Genome Typed Object") is a single JSON document produced by the
[BV-BRC](https://www.bv-brc.org) (Bacterial and Viral Bioinformatics
Resource Center, formerly PATRIC) annotation pipeline. One GTO = one
**assembly** (one biological sample's reconstructed genome). For flu,
that means one isolate with 8 segments — verified in this corpus:
**108,530 / 108,530 (100.00%)** of assemblies have exactly 8 contigs in
`genome_final.parquet`.

The GTO bundles three things into one file:

1. **Metadata** about the assembly (taxonomic ID, strain name, quality).
2. **Contigs** — the raw nucleotide sequences of each segment, with
   per-contig metadata (which segment, geometry, contig-level quality).
3. **Features** — the structured annotations that sit *on* those
   contigs: every CDS, every mature peptide, etc., each with a coordinate
   pointer back to its parent contig.

Note, **the nucleotide layer and the protein/annotation layer are
held in two separate top-level arrays** (`contigs` and `features`) and
the only thing linking them is the `contig_id` inside each feature's
`location` field. There is **no** CDS sub-annotation embedded inside the
contig itself; you must read the `features` array to know where CDSs
sit.

The annotation pipeline used here is Jim Davis's viral annotator (see
the comment in `conf/virus/flu.yaml` linking to
`Viral_Annotation/blob/main/Viral_PSSM.json`), which defines the
function strings and family assignments that ultimately decide what
`core_functions` and `aux_functions` mean for flu.

---

## 2. Top-level JSON structure

A flu GTO has these top-level keys (observed across all 200 sampled
files):

```
analysis_events        # provenance: each annotation step + UUID + timestamp
close_genomes          # related-genome list (BV-BRC internal); not used
contigs                # ARRAY — one entry per nucleotide segment (DNA layer)
domain                 # 'Viruses'
features               # ARRAY — one entry per CDS or mat_peptide (annotation layer)
genetic_code           # 11 in 100.00% of flu GTOs in this corpus — see §2.1
id                     # BV-BRC genome id (e.g. '197911.11517')
ncbi_taxonomy_id       # e.g. 197911 = Influenza A virus
quality                # dict with 'genome_quality' field
scientific_name        # e.g. 'Influenza A virus (A/mallard/Wisconsin/394/1977(H4N1))'
```

### 2.1 The `genetic_code = 11` artifact

`genetic_code` is `11` (bacterial / archaeal) in **1,793,563 /
1,793,563 (100.00%)** rows of `protein_final.parquet` — every flu CDS
in this corpus carries this value. But flu actually uses **NCBI code
1** (the standard eukaryotic / cytoplasmic code); this is a BV-BRC
platform default, not a biological claim. The codes share the same
canonical **start codon** (ATG) and **stop codons** (TAA / TAG / TGA),
so the discrepancy has no consequence for any operation we currently
perform — but anyone doing codon-level work downstream should treat
this field as non-authoritative for flu and hard-code code 1 instead.

---

## 3. The nucleotide layer: `contigs[i]`

Each entry in the `contigs` array is one **segment** of the assembled
genome. For flu, the typical (and complete) case is **8 contigs per
GTO**, one per segment.

Fields (all observed in the excerpt at `docs/examples/example_gto_excerpt.json`):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | str | BV-BRC contig identifier; **this is the `contig_id` that `features[*].location` points back to**. Format: `<genome_part>.<contig_index>` | `'1406633.10'` |
| `dna` | str | The full nucleotide sequence of this contig, **5′ → 3′**, including any 5′ and 3′ UTRs. Uppercase or lowercase depending on source (we uppercase before k-mer extraction). | `'tcaaatatattcaat...'` (median Segment-1 contig is 2296 nt in this corpus; see §6.5) |
| `replicon_type` | str | Which flu segment number this contig is (`'Segment 1'` … `'Segment 8'`). For other organisms it would be the equivalent (chromosome, plasmid). | `'Segment 1'` |
| `replicon_geometry` | str | `'linear'` for flu — **1,600 / 1,600 (100%)** contigs in a 200-GTO sample. Would be `'circular'` for circular replicons. | `'linear'` |
| `contig_quality` | str | BV-BRC's QC verdict at the contig level. `'Good'` for **108,530 / 108,530 (100%)** contigs in this corpus. | `'Good'` |

**What is NOT on the contig:** no CDS coordinates, no UTR annotation,
no exon/intron annotation, no gene names. All of that lives in
`features[]`.

---

## 4. The annotation layer: `features[i]`

Each entry in `features` is one biological annotation tied to (a portion
of) a contig. In flu GTOs only two feature types are observed:

- `CDS` — a complete coding sequence. **1,793,563 rows in
  `protein_final.parquet`** (every entry in `protein_final` is type
  `CDS`).
- `mat_peptide` — a mature peptide cleaved out of a precursor. In flu
  these are exclusively the HA1 and HA2 cleavage products of the HA
  precursor. **`preprocess_flu.py` lines 554–555 filter mat_peptide
  rows out of `protein_final.csv` and redirects them to
  `protein_non_cds.csv`** — verified by inspection: 223,612 rows in
  that file, of which 111,812 are HA1 and 111,800 are HA2. Anyone
  needing mat_peptides downstream must read `protein_non_cds.csv`
  separately.

Fields per feature:

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | str | BV-BRC feature identifier; the canonical primary key for a CDS/mat_peptide. Parsed into `brc_fea_id`. | `'fig\|197911.84799.CDS.1'` |
| `type` | str | `'CDS'` or `'mat_peptide'`. | `'CDS'` |
| `function` | str | Free-text protein function string. **This is the join key against `conf/virus/flu.yaml`** — every entry in `core_functions`, `aux_functions`, `selected_functions`, `function_short_names`, etc. is one of these exact strings. | `'RNA-dependent RNA polymerase PB2 subunit'` |
| `location` | list of `[contig_id, start, strand, length]` tuples | The coordinate pointer back to one or more contigs. **Single-entry for unspliced; multi-entry for spliced.** See §5. | `[['1406633.10', '16', '+', '2280']]` |
| `protein_translation` | str | The amino-acid sequence of this feature. For CDS, it is the translation of the `location`-defined nucleotide span. Parsed into `prot_seq`. | `'MERIKELRDLMSQSRTREIL...'` |
| `family_assignments` | list of `[family_root, family_id, family_desc, source]` | PSSM-based family classification. Parsed: `family_assignments[0][1]` → `family` column. | `[['Alphainfluenzavirus', 'Alphainfluenzavirus.PB2.1.pssm', 'RNA-dependent RNA polymerase PB2 subunit', 'LowVan Annotate']]` |
| `alias_pairs` | list of `[kind, alias]` | Short aliases, primarily gene-name shorthand. Not currently parsed (we use `function` instead). | `[['gene', 'PB2']]` |
| `feature_quality` | str | BV-BRC QC verdict — **only populated on "major" proteins** (see §6.3). Absent on alternative-frame products like M2, NEP, PB1-F2 and on `mat_peptide` features. | `'Good'` |
| `feature_quality_flags` | list | QC warning flags. Rare: 11 occurrences / 3,715 features in a 200-GTO sample (~0.3%). | (rare) |
| `feature_creation_event` | str (UUID) | Provenance only; not biological. | `'1C78374E-5DC8-11F0-9FDD-0B32C9360165'` |
| `annotations` | list of audit-trail entries | Provenance only; not biological. | (omitted from excerpt) |

---

## 5. The `location` schema (in detail)

`location` is the load-bearing field for any nucleotide-level interpretation.

**Where it lives and what it describes.** The `location` field is a key
*inside each `features[i]` entry* (i.e., on a protein/CDS annotation),
but every coordinate it stores refers to *nucleotides* on the matching
`contigs[i].dna` string. So `location` is the bridge between the
**annotation layer** (proteins / CDSs / mature peptides) and the
**nucleotide layer** (DNA segments). All four fields below are
nucleotide-level facts.

**Visual anchor.** Before the schema details, here is what one entry
of `location` points at on the contig (generic flu segment, mRNA-sense,
not to scale):

```
contig.dna  (1-based positions, plus strand, mRNA-sense; not to scale)

        [ 5' UTR ] ─── [ ATG ─── CDS, length nt ─── stop ] ─── [ 3' UTR ]
        ▲              ▲                              ▲              ▲
        │              │                              │              │
        1     start_1based                  start+length−1   contig_length
```

The bracketed CDS span is what one `[contig_id, start, strand, length]`
entry covers. §6.1 has the same layout annotated with
start-codon / stop-codon / ORF terminology.

Schema:

```
location = [ [contig_id, start_1based, strand, length], ... ]
```

Each inner tuple is one **span** on `contig_id`. The four elements:

1. **`contig_id`** (str) — points to the matching `contigs[i].id`.
2. **`start_1based`** (str or int) — the **1-based, inclusive** start
   position on the plus strand of the contig. "1-based" means the
   *first* nucleotide of the contig is position 1 (not 0). GenBank /
   GFF / most genomic databases use this convention. Example: in the
   PB2 row shown below, `start_1based = 16` means the CDS begins at
   the 16th nucleotide of the contig (positions 1–15 are 5′ UTR);
   §5.1 walks through the full slicing arithmetic on this row, and §9
   gives a general reconstruction recipe. (Strings and ints both
   occur for this field in real data; downstream code casts.)
3. **`strand`** (str) — `'+'` or `'-'`. For flu in our corpus, always
   `'+'`, but parsers should handle `'-'` (would require reverse-complementing
   the extracted nucleotides).
4. **`length`** (str or int) — span **length in nucleotides**, *not* an
   end coordinate. Inclusive end position = `start + length - 1`.
   Example: in the PB2 row below, `length = 2280` means the CDS spans
   2,280 nucleotides (= 760 codons), ending at nucleotide
   16 + 2280 − 1 = 2295.

### 5.1 Single-entry case (unspliced)

**"Unspliced" means the entire CDS is a single contiguous run of
nucleotides on the contig** — one **start codon**, one **stop codon**,
no introns; the whole **open reading frame (ORF)** is one continuous
span. Most flu CDSs are unspliced. Example PB2 in the excerpt:

```json
"location": [["1406633.10", "16", "+", "2280"]]
```

Schematic (1-based coordinates on the parent contig):

```
contig "1406633.10"  (Segment 1, full length ≈ 2316 nt)

  pos:   1                16                                       2295         2316
         |                |                                          |            |
         v                v                                          v            v
  dna:  [5'-UTR (15 nt)]  [ATG · · · CDS (2280 nt = 760 codons) · · · TAG]  [3'-UTR]
                          └──────────────────── one span ────────────────┘

  location = [[ "1406633.10", "16", "+", "2280" ]]
                     ^         ^    ^     ^
                  contig    start  strand length (nt)

  Python slice:  contig.dna[16-1 : 16-1+2280]  =  contig.dna[15:2295]
```

- Contig: `1406633.10` (Segment 1).
- Span: nucleotides 16 … 16+2280−1 = 2295 on Segment 1.
- Strand `+`, so the CDS DNA is `contig.dna[15:2295]` (0-based slice).
- Length 2280 nt = 760 codons. The protein-length convention is
  **observed and verified on this row** (`protein_final` PB2 entry,
  assembly `0000003012`): `prot_seq` is 760 characters long, ends with
  `'*'` (i.e. the stop codon is represented as `*` in the AA string),
  and the `length` column equals 760. So the identity
  `len(prot_seq) * 3 == location[0][3]` holds exactly — the annotated
  CDS nt length includes the stop codon, and the AA string includes
  the matching `*`. The PB2 last 9 nt are `ATCAATTAG`, ending in the
  canonical `TAG` **stop codon**.

### 5.2 Empirical properties of the CDS-length field (corpus checks)

Three universal-CDS properties are textbook, but they are also true
in our actual corpus (`data/processed/flu/July_2025/protein_final.parquet`):

- **CDS nucleotide length is a multiple of 3.** Verified: `cds_nt mod 3
  == 0` for **868,240 / 868,240 (100.000%)** rows across the 8 major
  CDS functions.
- **Start codon is ATG.** Verified by extracting CDS DNA via
  `contig.dna[start-1 : start-1+length]` and reading the first 3 nt:
  ATG in **4983 / 5000 (99.66%)** of a sampled 5,000 major CDS rows.
  The 0.34% remainder is dominated by short non-canonical alternatives
  (GAA, ATA, AGA, CCA each ≤ 0.1%) — almost certainly annotation
  artifacts on truncated or low-quality entries.
- **Stop codon is one of TAA / TAG / TGA.** Verified the same way on
  the last 3 nt: a canonical stop in **4951 / 5000 (99.02%)**.
  Distribution among the canonical stops is roughly even
  (TGA 33.9%, TAA 33.5%, TAG 31.6%).

Strand: **100.000%** of major-CDS `location` entries are on the `+`
strand in our corpus (no `-` strand observed across the 8 major
functions). Parsers should still handle `-` for portability, but
nothing in this dataset exercises that path.

### 5.3 Multi-entry case (spliced)

**What is splicing?** In eukaryotic-style gene expression, a pre-mRNA
can contain internal stretches called **introns** that get *cut out*
before translation, with the surviving pieces (**exons**) *joined*
back together to form the mature mRNA. The mature CDS in the genome
is therefore non-contiguous: two (or more) exon spans on the same
contig, separated by intron gaps that don't appear in the protein.
Influenza A uses splicing on exactly two gene products: **M2**
(Segment 7) and **NEP** (formerly NS2, Segment 8). All other flu
proteins are unspliced.

In a GTO this is encoded as **multiple entries inside the same
`location` list, all pointing at the same `contig_id`**. Each entry
is one exon. Intron positions are *implicit* — they are whatever
falls between two consecutive exons on the same contig.

Example M2 from the excerpt:

```json
"location": [
  ["1406633.6", "14",  "+", 26],
  ["1406633.6", 728,   "+", 268]
]
```

Schematic — Segment 7, full length ≈ 1027 nt — with M1 (unspliced,
top track) and M2 (spliced, bottom track) drawn on the same coordinate
axis to show how they share a start codon and then diverge:

```
Segment 7 contig "1406633.6"   (full length ≈ 1027 nt)

  pos:    1     14                39 40                 727 728                995  1027
          |     |                  | |                    | |                    |    |
          v     v                  v v                    v v                    v    v

  M1 :   [utr]  [ATG─── M1 CDS (singly continuous, 759 nt) ───────────────TAA] [utr]
                ▲                                                            ▲
                └─── shared start codon ───┐                                 └─ M1 stop codon
                                           │
  M2 :   [utr]  [ATG─exon1 (26 nt)─]  ╳╳╳╳╳╳ intron (688 nt) ╳╳╳╳╳╳ [─exon2 (268 nt)─TAA] [utr]
                └────────── M2 mature CDS = exon1 + exon2 = 26 + 268 = 294 nt ─────────┘

  M2 location (one row per exon, both on the same contig):
    [
      [ "1406633.6", "14",  "+",  26  ],   ← exon 1: nt 14 … 39
      [ "1406633.6", "728", "+", 268  ]    ← exon 2: nt 728 … 995
    ]
```

Key points:

- Contig: `1406633.6` (Segment 7) for **both** entries.
- Exon 1: nucleotides 14 … 39 (length 26).
- Intron: nucleotides 40 … 727 (length 688) — **not annotated explicitly;
  inferred from the gap between exons.**
- Exon 2: nucleotides 728 … 995 (length 268).
- Mature CDS = concat(exon1, exon2) = 26 + 268 = 294 nt = 98 codons (97 AA
  + stop ✓).
- M1 and M2 **share** the start codon (both begin at position 14 on
  Segment 7) but use different reading frames after the splice donor —
  M2 reads through a frame shift introduced by removing the intron.

To reconstruct the CDS DNA for a spliced feature: slice each exon out of
the parent contig with 1-based-to-0-based arithmetic, concatenate in
order, then translate (with NCBI code 1 for flu, regardless of the
`genetic_code` field). The single-exon recipe in §9 generalises by just
looping over `location` entries.

---

## 6. Coding regions vs non-coding regions

GTOs annotate **only coding regions** explicitly. Everything else is
implicit.

### 6.1 Layout of a flu segment (mRNA-sense, schematic)

```
 5' UTR ──── [ ATG ────  CDS  ──── (TAA|TAG|TGA) ] ──── 3' UTR
            └────────── coding region ──────────┘

  ATG          = start codon
  TAA|TAG|TGA  = stop codons (one of the three; canonical genetic code)
  [...]        = the annotated CDS = the open reading frame (ORF)
                  from start codon through stop codon, inclusive,
                  read in a single reading frame
```

- **Coding region (CDS)** — annotated open reading frame (ORF) that
  encodes a protein. In this corpus: starts with the **start codon**
  ATG in 99.66% of sampled major CDSs, ends with a canonical **stop
  codon** (TAA / TAG / TGA) in 99.02%, and the annotated `location`
  length is a multiple of 3 (i.e. an integer number of codons) in
  100.000% of 868,240 major CDS rows. See §5.2 for the supporting
  queries.
- **5′ UTR / 3′ UTR** — flanking untranslated regions. **They exist on
  the genome but have no feature record in the GTO** — there is no
  `features[i]` entry of type `UTR`. You can only know they're there by
  computing the complement of CDS coverage on the parent contig.

This document does not characterize the biological role of UTRs
(promoter elements, packaging signals, etc.) because those properties
are not derivable from the GTO data itself — they are external
literature claims.

### 6.2 What's annotated

Every CDS appears as a `features[i]` of type `CDS`, with a `location`
covering the coding span. Mature peptides (HA1, HA2 for flu) appear as
`features[i]` of type `mat_peptide`, located *inside* the parent CDS's
span on the same contig. Example from the excerpt
(Segment 4 / HA contig `1406633.4`):

```
HA precursor (CDS) — start 8,    length 1695 → covers 8..1702
HA1 (mat_peptide) — start 56,    length 981  → covers 56..1036    (inside HA)
HA2 (mat_peptide) — start 1037,  length 666  → covers 1037..1702  (inside HA)
```

### 6.3 What's NOT annotated (computable, but implicit)

- **5′ UTR** — contig nucleotides before the first CDS start on that
  contig. Implicit.
- **3′ UTR** — contig nucleotides after the last CDS end on that
  contig. Implicit.
- **Introns** — for spliced features (M2, NEP), the nucleotides in the
  gap between consecutive `location` entries. Implicit; never annotated
  as its own feature.
- **Overlapping ORFs (open reading frames)** — for PB1 (PB1 + PB1-F2 +
  PB1-N40) and PA (PA + PA-X + PA-N155 + PA-N182), the contig
  nucleotides are shared across multiple `features[*]` entries. Two
  mechanisms produce the overlap: (a) **internal start codons in the
  same reading frame as the major** produce N-terminally truncated
  proteins (PB1-N40, PA-N155, PA-N182), and (b) **a different reading
  frame relative to the major** produces entirely different
  polypeptides (PB1-F2 from a +1 frame on Segment 2; PA-X from a +1
  ribosomal frameshift on Segment 3). There is no "ORF map" beyond
  iterating the features array yourself.

If a downstream consumer needs to identify "non-coding nucleotides on
this contig," the recipe is: take the contig length, subtract the union
of all `location` spans across every feature that points at this
contig. The complement is non-coding (UTRs + introns).

**Empirical UTR lengths in this corpus** (sampled 20,000 contigs,
median nucleotides; computed as `min(CDS start)−1` for 5′ and
`contig_length − max(CDS end)` for 3′, using the union of all annotated
CDSs and mat_peptides on the contig):

| Segment | 5′ UTR median (nt) | 5′ UTR max | 3′ UTR median (nt) | 3′ UTR max |
|---|---:|---:|---:|---:|
| S1 (PB2) | 1  | 27 | 12 | 39 |
| S2 (PB1) | 2  | 24 | 23 | 49 |
| S3 (PA)  | 6  | 25 | 31 | 60 |
| S4 (HA)  | 7  | 43 | 12 | 52 |
| S5 (NP)  | 25 | 46 | 2  | 34 |
| S6 (NA)  | 0  | 22 | 8  | 40 |
| S7 (M1)  | 10 | 44 | 1  | 243 |
| S8 (NS1) | 11 | 44 | 13 | 207 |

In this corpus, **most segments are deposited with very short UTRs
(typically <30 nt on each side)**. The exceptions are the long-tail
3′ maxes on S7 and S8 (243 nt, 207 nt), which arise from a small
fraction of contigs that extend well past the spliced product's end;
these inflate the maximum but not the median.

A small number of rows give **negative implied UTR** (-1 nt on S7, S8)
— this happens when an annotated CDS end coordinate exceeds the
recorded contig length, which is an annotator/coordinate-system artifact
on a small number of records, not a real biological feature.

**Non-coding fraction per segment** (1 − union(CDS spans)/contig length,
sampled 20,000 contigs, percent):

| Segment | mean | median | min | max |
|---|---:|---:|---:|---:|
| S1 (PB2) | 0.85% | 0.61% | 0.00% | 2.65% |
| S2 (PB1) | 1.01% | 1.13% | 0.00% | 3.08% |
| S3 (PA)  | 1.45% | 1.87% | 0.00% | 3.80% |
| S4 (HA)  | 1.31% | 1.16% | 0.00% | 4.54% |
| S5 (NP)  | 1.60% | 1.77% | 0.00% | 4.53% |
| S6 (NA)  | 1.07% | 0.67% | 0.00% | 4.04% |
| S7 (M1)  | 1.63% | 1.11% | (−0.11%) | 26.10% |
| S8 (NS1) | 2.73% | 3.01% | (−0.12%) | 26.18% |

Across all 8 segments the median non-coding fraction is ≤ 3.01%.
**The vast majority of nucleotides in a flu contig in this corpus are
inside an annotated CDS.** The S7 / S8 maxima (~26%) are individual
outlier records.

### 6.4 The "major protein" concept

`conf/virus/flu.yaml` distinguishes two related concepts:

- **`core_functions`** (9 entries) — proteins present in nearly all
  Flu A genomes (PB2, PB1, PA, HA, NP, NA, M1, M2, NEP).
- **`aux_functions`** (11 entries) — additional protein products that
  may or may not be present (PB2-S1, PB1-F2, PB1-N40, PA-X, PA-N155,
  PA-N182, HA1, HA2, M42, NS3, NS1).
- **`selected_functions`** (8 entries) — the subset used for ML
  modeling. These are the **"major"** proteins: one primary product
  per segment. For flu A this is **7 core majors** (PB2, PB1, PA, HA,
  NP, NA, M1) plus **1 auxiliary major** (NS1, which is the primary
  product of Segment 8). M2 and NEP are core-but-not-major because they
  are alternative reading-frame products that share Segment 7 / 8 with
  the major.

The yaml file makes this explicit (paraphrased from its own comment):

> In BV-BRC, **major proteins are the only features that receive a
> `feature_quality` annotation.** Alternative reading frame products
> (M2, NEP, PB1-F2, PA-X, etc.) and mature peptides (HA1, HA2) are not
> major proteins.

This matches the data: of 3715 features across 200 sampled GTOs,
**1601 (43%) carry `feature_quality`** — close to 200 GTOs × 8 majors
= 1600. This is also why the `selected_functions` count is 8 (one per
segment), and why those eight are the natural choice for `dataset.schema_pair`
in bundle configs.

### 6.5 CDS length vs contig length — observed scatter

Across our corpus (108,530 isolates), the eight major-CDS functions
have tight per-CDS length distributions. The implication is that
**switching the k-mer source from contig DNA to CDS DNA would reduce
per-sequence length scatter** — CDS lengths cluster around a clear
median per function, while contig lengths scatter wider because of
varying UTR coverage in the deposited records.

**Major CDS lengths (nucleotides):**

| Function | min | median | max | spread |
|---|---:|---:|---:|---:|
| PB2 | 2196 | 2280 | 2283 | 87 |
| PB1 | 2193 | 2274 | 2292 | 99 |
| PA  | 2073 | 2151 | 2163 | 90 |
| HA  | 1650 | 1701 | 1713 | 63 |
| NP  | 1416 | 1497 | 1500 | 84 |
| NA  | 1338 | 1410 | 1428 | 90 |
| M1  | 723  | 759  | 762  | 39 |
| NS1 | 603  | 693  | 717  | 114 |

(All `n = 108,530` per function; counted from `location[0][3]` in
`protein_final.parquet`.)

**Contig lengths per canonical segment (nucleotides).** Parenthesized
label is the **major protein** that segment encodes (one per segment;
see §6.4). All statistics are computed across **108,530 isolates**, and
every isolate contributes exactly one contig per segment (each segment
column has `n = 108,530`).

| Segment (major protein) | min | median | max | std | Distinct lengths |
|---|---:|---:|---:|---:|---:|
| S1 (PB2) | 2,201 | 2,296 | 2,417 | 21.0 | 151 |
| S2 (PB1) | 2,201 | 2,301 | 2,379 | 23.6 | 148 |
| S3 (PA)  | 2,150 | 2,189 | 2,266 | 30.2 | 105 |
| S4 (HA)  | 1,650 | 1,715 | 1,817 | 24.5 | 146 |
| S5 (NP)  | 1,460 | 1,524 | 1,596 | 24.4 | 126 |
| S6 (NA)  | 1,338 | 1,420 | 1,498 | 19.7 | 147 |
| S7 (M1)  |   906 |   993 | 1,069 | 14.5 | 136 |
| S8 (NS1) |   774 |   860 |   928 | 17.3 | 130 |

(Counted from `length` column in `genome_final.parquet`. The Segment 7
contig also carries M2; the Segment 8 contig also carries NEP — see
§5.3 — but the major used here as the parenthetical label is the
primary product per segment.)

**Major protein lengths (amino acids).** One row per major protein
(8 majors total, one per segment). Every isolate has exactly one of
each, so each row has `n = 108,530`. Counted from the `length` column
in `protein_final.parquet`.

| Segment | Major | Function (string in GTO) | mean | std | min | median | max |
|---|---|---|---:|---:|---:|---:|---:|
| S1 | PB2 | RNA-dependent RNA polymerase PB2 subunit | 759.9 | 0.75 | 732 | 760 | 761 |
| S2 | PB1 | RNA-dependent RNA polymerase catalytic core PB1 subunit | 758.0 | 0.82 | 731 | 758 | 764 |
| S3 | PA  | RNA-dependent RNA polymerase PA subunit | 717.0 | 0.31 | 691 | 717 | 721 |
| S4 | HA  | Hemagglutinin precursor | 566.5 | 1.95 | 550 | 567 | 571 |
| S5 | NP  | Nucleocapsid protein | 499.0 | 0.52 | 472 | 499 | 500 |
| S6 | NA  | Neuraminidase protein | 469.6 | 2.76 | 446 | 470 | 476 |
| S7 | M1  | Matrix protein 1 | 253.0 | 0.25 | 241 | 253 | 254 |
| S8 | NS1 | Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor | 226.7 | 5.57 | 201 | 231 | 239 |

**Comparison.** Pairing each segment with its primary CDS (nt spread =
max−min from the contig table above; CDS-nt spread = max−min from the
"Major CDS lengths" table earlier in this section):
PB2 contig range 216 nt vs CDS range 87 nt; PB1 178 vs 99; PA 116 vs 90;
HA 167 vs 63; NP 136 vs 84; NA 160 vs 90; M1 163 vs 39 (M1 only —
the M contig also carries M2 which extends further); NS1 154 vs 114
(NS1 only — same caveat for NEP). For all eight segments,
**contig-length spread exceeds primary-CDS-length spread**, consistent
with the bulk of the variability sitting in the UTR / outside-primary-CDS
nucleotides rather than in the CDS itself.

**Protein vs contig length variability.** At the **protein** level
(amino acids, table above) lengths are extremely tight — std ≤ 2.8 aa
for HA / NA, ≤ 1 aa for all four polymerase / NP / M1 majors; only
NS1 reaches std ≈ 5.6 aa. At the **contig** level (nucleotides,
table above) std is 15–30 nt per segment. The aa-level near-constancy
is what makes per-residue alignment of major flu proteins possible
without explicit MSA; the residual variation lives in UTRs and is
visible only at the contig level.

**Within-subtype CDS conservation is tighter still.** For HA, broken
down by HxNx subtype (top 10 most-frequent subtypes in the corpus):

| Subtype | n | HA CDS min | median | max |
|---|---:|---:|---:|---:|
| H3N2 | 12,052 | 1650 | 1701 | 1704 |
| H1N1 | 10,486 | 1656 | 1701 | 1710 |
| H1N2 |  1,328 | 1665 | 1698 | 1707 |
| H9N2 |  1,248 | 1650 | 1683 | 1686 |
| H3N8 |  1,240 | 1653 | 1701 | 1710 |
| H4N6 |  1,078 | 1656 | 1695 | 1698 |
| H5N1 |  1,036 | 1650 | 1704 | 1710 |
| H7N9 |    943 | 1653 | 1683 | 1710 |
| H5N2 |    582 | 1665 | 1695 | 1707 |
| H7N3 |    480 | 1668 | 1683 | 1707 |

Within any single subtype the IQR is small (e.g. H3N2 is essentially
1701 nt at the median). Across subtypes the medians span 1683–1704 nt
— a ~21 nt floor-to-ceiling spread, reflecting real H-clade differences
in HA0 length.

---

## 7. Flu-specific overlay

Flu A is segmented: each isolate has 8 RNA segments. In this corpus,
**108,530 / 108,530 (100%) of assemblies have exactly 8 contigs**.
The mapping between segments and the protein products encoded by each
is fixed. The canonical table (per-protein frequencies are
**assemblies-where-this-function-appears / 108,530**, computed from
`protein_final.parquet`):

| Segment | Replicon string in GTO | Major (selected) | Other CDSs on this segment, with corpus presence rate | Mature peptides (in `protein_non_cds.csv`) |
|---|---|---|---|---|
| 1 | `'Segment 1'` | **PB2** (100.00%) | PB2-S1 (0.39%) | — |
| 2 | `'Segment 2'` | **PB1** (100.00%) | PB1-F2 (100.00%), PB1-N40 (100.00%) | — |
| 3 | `'Segment 3'` | **PA** (100.00%)  | PA-X (66.59%), PA-N155 (100.00%), PA-N182 (100.00%) | — |
| 4 | `'Segment 4'` | **HA** (100.00%)  | — | HA1 (111,812 rows), HA2 (111,800 rows) |
| 5 | `'Segment 5'` | **NP** (100.00%)  | — | — |
| 6 | `'Segment 6'` | **NA** (100.00%)  | — | — |
| 7 | `'Segment 7'` | **M1** (100.00%)  | M2 (98.67%, spliced), M42 (96.18%) | — |
| 8 | `'Segment 8'` | **NS1** (100.00%) | NEP (97.13%, spliced), NS3 (93.63%) | — |

Notes:
- **PB1, PA segments host multiple overlapping CDSs.** k-mer counts
  computed from the full contig sum contributions from all overlapping
  ORFs.
- **M and NS segments host spliced products.** k-mer counts from the
  full contig include intronic nucleotides not present in the mature
  mRNA.
- **HA segment hosts mat_peptides**, which are *not* independent CDSs
  but slices of the HA CDS. They appear as separate `features` entries
  but their nucleotide spans are subsets of HA's.

The `conditional_segment_mappings` block in `conf/virus/flu.yaml`
encodes these expectations: a protein is only assigned a
`canonical_segment` when **both** its `function` and its parent contig's
`replicon_type` match the known biology. If a record says PB2 is on
Segment 3, that's inconsistent and the protein is left unassigned
rather than mislabeled.

---

## 8. What Stage 1 currently parses

`src/preprocess/preprocess_flu.py::extract_data_from_gto` (file:line
**61**) is the one place GTO JSON is read.

Per-feature, it captures: `id`, `type`, `function`,
`protein_translation`, `location`, `feature_quality`, plus derived
`genbank_ctg_id` (from `location[0][0]`), `replicon_type` (looked up
from the contigs map), and `family` (from `family_assignments[0][1]`).
The output is `protein_final.csv` / `protein_final.parquet`.

Per-contig, it captures: `id` (→ `genbank_ctg_id`), `replicon_type`,
`contig_quality`, and `dna` (→ `dna_seq`). The output is
`genome_final.csv` / `genome_final.parquet`.

Fields it intentionally **does not** carry forward:

- `family_assignments[0][0,2,3]` (root, description, source) — only the
  family ID is kept.
- `alias_pairs` — the function string is the join key throughout the
  pipeline; gene-name aliases are not needed.
- `replicon_geometry` — `'linear'` for 1,600 / 1,600 (100%) flu contigs
  in a 200-GTO sample.
- `feature_creation_event`, `annotations` — provenance only.
- `feature_quality_flags` — rare; could be added if needed for QC.
- The top-level `analysis_events`, `close_genomes`, `id` (BV-BRC genome
  id) are not propagated.

`genetic_code` from the GTO **is** captured in `protein_final.csv` but
should be treated as cosmetic (see §2.1).

---

## 9. Recipe: reconstruct CDS DNA from `protein_final` + `genome_final`

For any row of `protein_final.csv`:

```
loc       = parse(row['location'])      # list of [contig_id, start_1based, strand, length]
contig_id = row['genbank_ctg_id']
strand    = loc[0][2]                   # '+' for all flu in our corpus

# Look up the contig DNA once
contig_dna = genome_final.loc[
    (genome_final['assembly_id']  == row['assembly_id']) &
    (genome_final['genbank_ctg_id'] == contig_id),
    'dna_seq'
].iloc[0]

# Concatenate exons in given order
cds_nt = ''
for (_ctg, start, _strand, length) in loc:
    s = int(start)
    L = int(length)
    cds_nt += contig_dna[s-1 : s-1 + L]    # 1-based → 0-based

# Strand handling (not exercised by flu, but here for completeness)
if strand == '-':
    cds_nt = reverse_complement(cds_nt)

# At this point in our corpus (see §5.2):
#   len(cds_nt) % 3 == 0        in 100.000% of major CDSs
#   cds_nt[:3]  == 'ATG'        in  99.66%  of sampled major CDSs
#   cds_nt[-3:] in stop set     in  99.02%  of sampled major CDSs
```

This is pseudocode only; nothing in the current codebase performs CDS
reconstruction. The codon-aware k-mers plan
(`docs/plans/2026-05-12_codon_aware_kmer_features_plan.md`) lists this
helper as the minimum new code needed for a CDS-scoped feature path.

---

## 10. Cross-references

- **`docs/examples/example_gto_excerpt.json`** — the canonical example
  used in §3–§6 above.
- **`docs/plans/2026-05-12_codon_aware_kmer_features_plan.md`** — uses
  this document as the metadata-availability baseline; explains why
  current k-mer features cannot be interpreted at codon level and what
  would change with CDS-scoped extraction.
- **`conf/virus/flu.yaml`** — the canonical list of which functions
  exist and which are "major" (= `selected_functions`).
- **`src/preprocess/preprocess_flu.py`** — the only consumer of GTO
  JSON in this repo (line 61 onward).
- **`CLAUDE.md`** → "Pipeline Stages" / "What This Project Does" — the
  high-level pipeline framing.
