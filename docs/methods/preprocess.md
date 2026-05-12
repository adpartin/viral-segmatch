# Preprocessing (Stage 1) — `preprocess_flu.py`

> Reference notes on how Stage 1 parses GTO files and emits
> `protein_final.csv` / `genome_final.csv`. Source: `src/preprocess/preprocess_flu.py`.
> See also:
> - `docs/gto_format_reference.md` — comprehensive walk-through of the
>   BV-BRC GTO JSON schema (contigs vs features, the `location` field,
>   spliced multi-entry CDSs, the "major protein" convention from
>   `conf/virus/flu.yaml`). This `preprocess.md` focuses on **what Stage 1
>   parses and how it filters**; `gto_format_reference.md` is the schema
>   itself, with corpus-wide statistics.
> - `docs/methods/pipeline_overview.md` §3.

---

## 1. What this stage does

For each `*.gto` file in `data/raw/<run_dir>/`:

1. Parse JSON once; build `meta_df` (genome-level metadata), a
   `segment_map` (contig id → replicon_type), and two tables: one row per
   protein feature, one row per contig.
2. Aggregate across all GTOs into `prot_df` and `genome_df`.
3. Run the protein pipeline (segment assignment → basic filters →
   sequence-level dedup → ESM-2 prep).
4. Run the genome pipeline (segment assignment → basic filters →
   DNA QC → DNA-level dedup).
5. Emit `protein_final.{csv,parquet}` and `genome_final.{csv,parquet}`.

`(assembly_id, genbank_ctg_id)` is the join key Stage 3 uses to attach
DNA to protein rows (`_pair_helpers.py::attach_dna_to_prot_df`).

---

## 2. GTO field map

Verified against `00621bf88c.gto` (July 2025) and cross-checked
against the full corpus (108,530 assemblies → 1,793,563 CDS rows in
`protein_final.parquet`, 868,240 contigs in `genome_final.parquet`).
See `docs/gto_format_reference.md` for the full schema walk-through;
the table below focuses on **which fields Stage 1 reads and where they
land**.

| GTO field | Where used | Notes |
|---|---|---|
| `ncbi_taxonomy_id`, `genetic_code`, `scientific_name` | meta → protein rows | Accessed with `gto[k]` (KeyError on absence — fail-loud). Not joined into genome rows. `genetic_code` is `11` in 1,793,563 / 1,793,563 (100%) rows — a BV-BRC artifact; flu actually uses NCBI code 1. |
| `quality.genome_quality` | meta → both tables | Becomes column `quality`. Used in `Poor` filter. Single value per assembly (not per segment). |
| `contigs[*].{id, replicon_type, contig_quality, dna}` | genome rows | `replicon_type` defaults to `'Unassigned'` if absent. `contig_quality` is stored but never filtered on (acknowledged TODO). `contig_quality == 'Good'` in 108,530 / 108,530 (100%) corpus rows. |
| `features[*].{id, type, function, protein_translation, location, feature_quality, family_assignments}` | protein rows | `feature_quality` is captured but never used. **By BV-BRC convention it's populated only on the 8 "major" proteins** (`selected_functions` in `conf/virus/flu.yaml`) — absent on all `mat_peptide` features and on alternative-frame products (M2, NEP, PB1-F2, PA-X, etc.). Verified in a 200-GTO sample: 1,601 / 3,715 features carried it (43.1%), close to 200 GTOs × 8 majors. |
| `location[0][0]` | derives `genbank_ctg_id` | Spliced features (multi-entry `location`) all have **both entries on the same contig** for Flu A — `[0][0]` is safe. Empirically: PA-X, NS3, M2, M42, PB2-S1, NEP all have 100% single-contig multi-entry locations across this corpus. Would need revisiting for cross-contig features (none observed). |
| `family_assignments[0][1]` | column `family` | Format is `[[family_type, family_id, fn_name, source]]`; `[0][1]` extracts the family_id (e.g., `Alphainfluenzavirus.PB2.1.pssm`). |

**Silently dropped top-level fields:** `domain`, `analysis_events`,
`close_genomes`, and **`gto['id']`** — the BV-BRC genome ID (e.g.,
`197911.11517`). The pipeline derives `assembly_id` from the **filename**
(`00621bf88c.gto` → `00621bf88c`) instead. These are different ID spaces;
they happen to be 1:1 per file. The BV-BRC genome ID can be recovered
from `brc_fea_id` (`fig|197911.11517.CDS.N`).

---

## 3. Output tables

### `protein_final.csv` (one row per protein feature)

Carries genome-level metadata (`assembly_prefix, ncbi_taxonomy_id,
genetic_code, scientific_name, quality, file, assembly_id`),
feature-level fields (`brc_fea_id, type, function, prot_seq, length,
location, feature_quality, genbank_ctg_id, replicon_type, family`),
segment assignment (`canonical_segment` in S1..S8), QC annotations
from `analyze_protein_ambiguities` (`has_ambiguities,
ambiguous_residues, …, has_internal_stop`), and `esm2_ready_seq` (the
cleaned sequence used downstream).

**`length` semantics:** AA length **as parsed from the GTO**, captured
at parse time and never recomputed. The GTO's `protein_translation`
string includes the stop codon as a trailing `*`, so `length` counts
that character too (verified on a representative PB2 row: `prot_seq`
length 760 = `location[0][3]` / 3 = 2280 / 3). After ESM-2 prep,
terminal stops are stripped (`strip_terminal_stop=True`) and X residues
are imputed (default `→ G`), so the **stored length and
`len(esm2_ready_seq)` differ by 1 in 1,771,244 / 1,793,563 rows
(98.76%) and are equal in 22,319 (1.24%)** — the 1.24% are entries
without a terminal stop in the parsed `prot_seq` (`has_terminal_stop ==
False`). Consumers needing the model-input length should use
`len(esm2_ready_seq)`.

### `genome_final.csv` (one row per contig)

Carries `assembly_id, file, quality, genbank_ctg_id, replicon_type,
contig_quality, dna_seq, canonical_segment, ambig_count, ambig_frac,
length, gc_content`.

**Asymmetry with `protein_final`:** genome rows do NOT carry
`assembly_prefix, ncbi_taxonomy_id, genetic_code, scientific_name`.
Stage 3's join only pulls `dna_seq`, so this isn't currently a
correctness issue — but a genome row alone can't answer "what
taxonomy/assembly prefix is this?". Cheap fix: pass `meta_df` into the
genome cross-merge.

**`length` semantics:** nucleotide length (from `summarize_dna_qc`).
Different unit from the protein-side `length`.

---

## 4. Filter pipeline

**Protein side** (`apply_protein_basic_filters`, then
`handle_protein_duplicates`, then `prepare_sequences_for_esm2`):

1. Drop rows where `assign_protein_segments` left `canonical_segment` null
   (function and replicon_type didn't match a known biological mapping).
2. Keep `type == 'CDS'` only.
3. Drop `quality == 'Poor'` (genome-level).
4. Drop `replicon_type == 'Unassigned'`.
5. `handle_assembly_duplicates`: collapse `(prot_seq, assembly_id)`
   duplicates. For Flu A every file is one assembly, so the GCF/GCA
   preference branch (`gto_utils.py:181-185`) is dead code; the "same
   file → keep first" branch handles all real cases.
6. `prepare_sequences_for_esm2`: drop seqs exceeding internal-stop or
   X-fraction thresholds, impute X → G, strip terminal stops. Survivors
   get `esm2_ready_seq`; the rest get `None`.
7. Final `prot_df = prot_df[prot_df['esm2_ready_seq'].notna()]`.

**Genome side** (`apply_genome_basic_filters` then
`handle_genome_duplicates`):

1. Drop null `canonical_segment`.
2. Drop `quality == 'Poor'` (genome-level — same limitation).
3. Drop `replicon_type == 'Unassigned'`.
4. Drop missing `dna_seq`.
5. `summarize_dna_qc`: annotate length, GC, ambiguous-base fraction.
6. Drop `(dna_seq, assembly_id)` duplicates (keep first).

### Segment assignment is biology-validated

`assign_protein_segments` (`preprocess_flu.py:481-537`) joins on
**both** `function` and `replicon_type` against
`conditional_segment_mappings` in `conf/virus/flu.yaml`. A GTO record
claiming "PB2 on Segment 3" stays unassigned and is dropped at step 1
of the filter pipeline. This is a strong invariant: a row in
`protein_final.csv` is guaranteed biology-consistent at the
(function, replicon_type) level.

---

## 5. Known issues and tech debt

| # | Issue | Where | Severity |
|---|---|---|---|
| 1 | **HA1/HA2 aux mappings are dead code.** `conditional_segment_mappings.aux_proteins` assigns `canonical_segment='S4'` to "Mature hemagglutinin N-terminal receptor binding subunit" and "Mature hemagglutinin C-terminal membrane fusion subunit", but both are `type='mat_peptide'` in GTOs and get dropped by the CDS filter immediately after. Pick: remove those two entries from `flu.yaml`, or reorder filters. | `flu.yaml:177-182`; filter order at `preprocess_flu.py:1117 → 1120` | **Real (cosmetic; no output impact yet)** |
| 2 | **ESM-2 prep thresholds hardcoded.** `max_internal_stops=0.1`, `max_x_residues=0.1`, `x_imputation='G'`, `strip_terminal_stop=True` live as locals in main. Should move under `virus.preprocessing.esm2_prep:` in `flu.yaml` so they appear in `resolved_config.yaml`. | `preprocess_flu.py:1159-1162` | Real (reproducibility) |
| 3 | **`gto['id']` silently ignored.** BV-BRC genome ID never captured. Pipeline-internal `assembly_id` is filename-derived. | `extract_data_from_gto` | Real (linkability), but currently no consumer needs it. |
| 4 | **Genome metadata is a subset of protein metadata.** Genome rows lack `assembly_prefix, ncbi_taxonomy_id, genetic_code, scientific_name`. | `extract_data_from_gto:135` | Cosmetic — Stage 3 only pulls `dna_seq`. |
| 5 | **`quality` is genome-level, not per-segment.** A "Good" genome with one "Poor" segment passes. Per-segment `contig_quality` is captured but ignored. | `apply_protein_basic_filters`, `apply_genome_basic_filters` | Documented TODO. |
| 6 | **`feature_quality` captured but never filtered on.** Often absent on `mat_peptide` features anyway. | `preprocess_flu.py:95, 561` | Documented TODO. |
| 7 | **`prot_df_no_seq` saved but not subtracted.** Rows with missing `prot_seq` carry through until `esm2_ready_seq.notna()` filters them. Asymmetric with the genome side, which drops missing `dna_seq` explicitly. | `preprocess_flu.py:1075-1077` | Cosmetic. |

---

## 6. Hard-coded parameters worth surfacing

These are scientific choices currently buried in code. Promoting them
to `conf/virus/flu.yaml` would make them visible in
`resolved_config.yaml` for every run:

- ESM-2 prep thresholds (issue 2 above).
- The CDS-only filter at `preprocess_flu.py:555` — implicitly excludes
  all mat_peptide products (HA1, HA2 in current annotation).
- Genome quality filter granularity (genome vs. contig).

---

## 7. What is verified correct

Empirically checked against the full corpus and code-reviewed against
the schema in `conf/virus/flu.yaml`:

- All required top-level GTO fields (`contigs`, `features`,
  `ncbi_taxonomy_id`, `genetic_code`, `scientific_name`, `quality`) are
  present in 200 / 200 sampled GTOs.
- **8 contigs per GTO**, one per `Segment 1..8`: **108,530 / 108,530
  (100.00%)** assemblies in this corpus have exactly 8 contigs.
  `replicon_to_segment` covers exactly this set.
- **Spliced features all stay on a single contig.** Six functions are
  multi-entry-location in the corpus (every row of each function has
  two `location` entries on the **same** `contig_id`), so
  `location[0][0]` is safe:

  | Function | Short | Rows in corpus | Multi-entry locations |
  |---|---|---:|---:|
  | `Host mRNA degrading protein PA-X` | PA-X | 72,272 | 100% |
  | `Hypothetical host adaptation protein NS3` | NS3 | 101,617 | 100% |
  | `M2 ion channel` | M2 | 107,091 | 100% |
  | `M42 alternative ion channel` | M42 | 104,389 | 100% |
  | `Splice variant of PB2, RIG-I-dependent interferon signaling pathway inhibitor` | PB2-S1 | 426 | 100% |
  | `Nuclear export protein` | NEP | 105,413 | 100% |

  All other major and auxiliary functions have single-entry locations.
- `family_assignments[0][1]` correctly extracts the family ID
  (`Alphainfluenzavirus.<PROT>.<N>.pssm`).
- Cache invalidation requires **both** parquets — safe against
  partial-write corruption.
- After ESM-2 prep, sequences that failed are filtered out via
  `notna()` on `esm2_ready_seq` (`preprocess_flu.py:1176`).
- No row fan-out in the meta-cross-feature or meta-cross-contig merges
  (`many_to_one` validation in `attach_dna_to_prot_df` later confirms
  this end-to-end).
