# AA k-mer support + feature-cache symmetry

**Status: IN PROGRESS** — branch `feature/aa-kmer-support`.

## Motivation

Immediate goal: run k=3 segment-pair experiments with **nt** and **aa**
k-mer features on the HA/NA and PB2/PB1 bundles, alongside the existing
nt k=6 setup. The aa path is not yet plumbed through the Stage 2b CLI,
the loader, or the training code — only the underlying
`sequences_to_sparse_kmer_matrix` function supports it.

Secondary goal: while adding the aa path, unify the lookup-key
convention across all three feature caches (ESM-2, aa k-mer, nt k-mer)
so that consumers (`embedding_utils.py`, `kmer_utils.py`,
`_pair_features.py`) share a single composite-key pattern. ESM-2 stays
working without recompute; nt k-mer is the only cache that needs an
on-disk migration, and a cross-cache equality test gates that swap.

## Final design

Three caches, three different internal dedup primitives, **one
public lookup contract**: composite occurrence key.

| Cache | Internal dedup key | Public lookup key | Rows on full Flu A |
|---|---|---|---:|
| ESM-2 (existing) | `sha1(prot_seq)::model_sig` (inside HDF5 `emb_keys`) | `(assembly_id, brc_fea_id)` | ~375K (sequence-deduped) |
| aa k-mer (new) | `md5(prot_seq)` | `(assembly_id, brc_fea_id)` | ~375K (sequence-deduped) |
| nt k-mer (migrated) | `md5(dna_seq)` | `(assembly_id, genbank_ctg_id)` | will drop from 868K to ~200-300K |

Justification audit (write paths, lookup paths, hash-collision risk,
brc_fea_id global uniqueness, and the "same sequence → same row" case)
was completed before this plan was written; see conversation log for
2026-05-13. Summary:

- The hash is **never an external API**; it's a write-time dedup primitive
  inside the HDF5/NPZ.
- Lookup goes through the parquet index using the composite occurrence
  key.
- Multiple occurrences mapping to the same row is **correct behavior**
  whenever the underlying sequence is byte-identical — features are pure
  functions of the sequence.
- Hash collision risk: sha1 (160 bits) and md5 (128 bits) are both
  several orders of magnitude below collision risk at corpus scale
  (~10⁶ items vs 10¹⁹–10²⁴ birthday bound).

## File inventory

### Code changes

| File | Change |
|---|---|
| `conf/kmer/default.yaml` | Add `alphabet: nt` field (`nt` \| `aa`, default `nt` for backward compat) |
| `src/embeddings/compute_kmer_features.py` | Read `alphabet` from config; switch input file/column (`genome_final.csv`/`dna_seq` for nt, `protein_final.csv`/`prot_seq` for aa); switch filename pattern (`kmer_features_{nt\|aa}_k{K}.{npz,parquet,json}`); pass `alphabet` to `sequences_to_sparse_kmer_matrix`; vocab size becomes `len(alphabet)**K`; for aa, **dedup by `md5(prot_seq)` on write** (matrix has one row per unique sequence; parquet has one row per occurrence with `assembly_id`, `brc_fea_id`, `cache_key`, `row`) |
| `src/utils/kmer_utils.py` | `load_kmer_index(kmer_dir, k, alphabet)` returns `dict[(assembly_id, occurrence_id) → row]`; `load_kmer_matrix(kmer_dir, k, alphabet)` reads `kmer_features_{alphabet}_k{K}.npz`; `get_kmer_pair_features(pairs_df, …, alphabet)` builds composite keys: `(assembly_id_a, brc_a)`/`(assembly_id_b, brc_b)` for aa, `(assembly_id_a, ctg_a)`/`(assembly_id_b, ctg_b)` for nt |
| `src/utils/embedding_utils.py` | `load_embedding_index` returns `dict[(assembly_id, brc_fea_id) → row]`; `load_embeddings_by_ids` and `create_pair_embeddings_concatenation` use composite tuple lookup |
| `src/models/_pair_features.py` | Pass `alphabet` to `load_kmer_index`/`load_kmer_matrix`/`get_kmer_pair_features`; ESM-2 path uses composite tuple |
| `src/models/train_pair_classifier.py` | `EMBED_DIM = len(alphabet) ** KMER_K`; read `KMER_ALPHABET` from config; thread through to `load_kmer_*` calls |
| `src/models/train_pair_baselines.py` | Same alphabet plumbing through `_resolve_kmer_k`-style helper |
| `scripts/stage2b_kmer.sh` | No change expected — bundle drives `kmer.alphabet` via Hydra |

### One-time data migration

| Artifact | Action | Cost |
|---|---|---|
| `master_esm2_embeddings.parquet` | Read, left-join with `protein_final.csv` on `brc_fea_id` to attach `assembly_id`, write back | Seconds, CPU only |
| `master_esm2_embeddings.h5` | **No change** — embeddings keyed internally; just regen parquet index | Free |
| `kmer_features_k6.{npz,parquet,json}` (nt, existing) | Rebuild to deduped format. Keep old file under `kmer_features_k6_LEGACY.{...}` until equality test passes | Minutes, CPU only |
| `kmer_features_aa_k3.{npz,parquet,json}` | Compute fresh from `protein_final.csv` | Minutes, CPU only |

### Bundles (new)

| File | Variant |
|---|---|
| `conf/bundles/flu_ha_na_kmer_nt_k3.yaml` | HA/NA, nt k=3 |
| `conf/bundles/flu_ha_na_kmer_aa_k3.yaml` | HA/NA, aa k=3 |
| `conf/bundles/flu_pb2_pb1_kmer_nt_k3.yaml` | PB2/PB1, nt k=3 |
| `conf/bundles/flu_pb2_pb1_kmer_aa_k3.yaml` | PB2/PB1, aa k=3 |

Each inherits from the corresponding production bundle (`flu_ha_na`,
`flu_pb2_pb1`) with two overrides: `kmer.k=3` and `kmer.alphabet=nt|aa`.

### Docs

- `docs/methods/kmer_features.md` — update Overview, Data flow, and Storage tables to reflect the alphabet config knob, composite lookup, and the dual filename pattern. Mark legacy `kmer_features_k6.npz` retirement once migration lands.

## Implementation phases

Phases are ordered so that earlier work doesn't break later work, and so
that the immediate experimental goal (running k=3 bundles) is reachable
without waiting for the nt-cache migration.

### Phase 1 — ESM-2 parquet symmetry migration

Smallest, lowest-risk change. Establishes the composite-tuple lookup
pattern that aa and nt will reuse.

1. Update `embedding_utils.py::load_embedding_index` to return
   `dict[(assembly_id, brc_fea_id) → row]`.
2. Update `embedding_utils.py::load_embeddings_by_ids` and
   `create_pair_embeddings_concatenation` to build composite tuple keys
   from pair-df columns.
3. Update `_pair_features.py` ESM-2 branch to pass `assembly_id_a/b`
   alongside `brc_a/b`.
4. Write a one-shot script (`scripts/migrate_esm2_parquet.py` or inline)
   to add `assembly_id` to the existing parquet via a left-join on
   `protein_final.csv`. Verify row count unchanged.
5. Sanity test: run a small ESM-2 training (e.g., one of the existing
   ESM-2 bundles or a debug bundle) and confirm metrics match a prior
   run within seed noise.

**Gate to Phase 2:** ESM-2 path still trains, parquet has new column.

### Phase 2 — aa k-mer compute path

Greenfield; no migration. Introduces the `alphabet` config knob and the
new filename pattern.

1. Add `alphabet: nt` to `conf/kmer/default.yaml`.
2. Extend `compute_kmer_features.py`:
   - Read `K`, `NORMALIZE`, `ALPHABET` from config.
   - Pick `input_file`, `seq_col`, and `occurrence_cols` based on alphabet:
     - nt: `genome_final.csv`, `dna_seq`, `(assembly_id, genbank_ctg_id)`.
     - aa: `protein_final.csv`, `prot_seq`, `(assembly_id, brc_fea_id)`.
   - Pick `vocab_alphabet` for the function call: `'ACGT'` or `'ACDEFGHIKLMNPQRSTVWY'`.
   - **aa write path**: dedup by `md5(prot_seq)`. Build an
     `occurrence → row` parquet that allows N-to-1 mappings (multiple
     `(assembly_id, brc_fea_id)` rows pointing to the same matrix row).
   - **nt write path (legacy mode for Phase 2)**: keep existing
     per-occurrence one-row-per-contig behavior until Phase 4 migration.
     Only the filename pattern changes (`kmer_features_nt_k{K}.*`) and
     the parquet adds an explicit `assembly_id` column.
   - Filename pattern: `kmer_features_{alphabet}_k{K}.{npz,parquet,json}`.
   - Vocab size: `len(vocab_alphabet) ** K` everywhere.
3. Strip emojis / decorative text per project style.

### Phase 3 — Loader + training consumers

1. Update `kmer_utils.py::load_kmer_index` to accept `alphabet` and read
   the alphabet-tagged filename; return composite-tuple-keyed dict.
2. Update `kmer_utils.py::load_kmer_matrix` for the new filename.
3. Update `kmer_utils.py::get_kmer_pair_features` to build composite
   tuple keys based on alphabet (`(assembly_id_a, brc_a)` vs
   `(assembly_id_a, ctg_a)`).
4. Update `train_pair_classifier.py`:
   - `KMER_ALPHABET = config.kmer.get('alphabet', 'nt')`
   - `EMBED_DIM = (4 if KMER_ALPHABET == 'nt' else 20) ** KMER_K`
   - Thread `alphabet` through to `load_kmer_*`.
5. Update `train_pair_baselines.py` and `_pair_features.py` analogously.

### Phase 4 — Bundles

Create the 4 bundles. Each is a thin overlay on the production parent:

```yaml
# flu_ha_na_kmer_aa_k3.yaml
defaults:
  - flu_ha_na
  - _self_

kmer:
  k: 3
  alphabet: aa
```

(and similarly for the other three variants)

### Phase 5 — Stage 2b runs + dataset/training smoke tests

1. Rerun Stage 2b for both new caches:
   - `./scripts/stage2b_kmer.sh --config_bundle flu_ha_na_kmer_aa_k3`
     (also `aa_k3` works for pb2_pb1 since the cache is global; only the
     alphabet+k matter)
   - `./scripts/stage2b_kmer.sh --config_bundle flu_ha_na_kmer_nt_k3`
2. Verify both caches load via the new `load_kmer_index` / `load_kmer_matrix`.
3. Run Stage 3 (existing dataset is fine — pair tables unchanged) and
   Stage 4 with each new bundle. Confirm training proceeds to completion.

### Phase 6 — nt k-mer cache migration (deferred follow-up)

Performed in a separate PR with the equality test as the verification
gate.

1. Build new dedup-by-`dna_hash` nt cache alongside the legacy
   per-occurrence cache (different filenames so both coexist).
2. For every `(assembly_id, genbank_ctg_id)` in `genome_final.csv`,
   compare lookups in old vs new caches: assert `np.array_equal(v_old, v_new)`.
3. If equality holds across the full corpus, swap the loader to point
   at the new cache and rename the legacy cache to `*_LEGACY.npz`.
4. After a grace period (one paper-cycle or one explicit confirmation),
   delete legacy.

## Verification gates

| Gate | Condition |
|---|---|
| Phase 1 → 2 | ESM-2 training on a known bundle still hits prior metrics within seed noise |
| Phase 3 → 4 | New `kmer_utils.py` loads cache built by Phase 2 without error; pair-feature shape matches `len(alphabet)**K` |
| Phase 5 complete | Both nt-k3 and aa-k3 bundles complete Stage 4 training end-to-end on one bundle (HA/NA or PB2/PB1) |
| Phase 6 swap | Cross-cache equality test passes for **all** (assembly_id, ctg_id) pairs on full Flu A |

## Caveats / out of scope

- **k upper bound (exhaustive-vocab pipeline)**: the current pipeline
  enumerates the full `|alphabet|^k` vocabulary at build time, so k is
  bounded by three independent bottlenecks: Python memory for the vocab
  list/dict at build; the MLP first-layer parameter count at training;
  and per-row densification in `KmerPairDataset`.
  - **nt** practical to k≈10 (`4^10 ≈ 1M` cols, ~2 GB MLP first layer).
    Production uses k=6.
  - **aa** practical to k=4 (`20^4 = 160K` cols, ~330 MB first layer).
    Current bundles use k=3 (8K cols).
  - **aa k=5** (`20^5 = 3.2M` cols) → 1.6B first-layer params, ~6.5 GB
    weight memory, per-row densification ~12 MB. Breaks GPU under
    default `hidden_dims=[512,…]`.
  - **aa k=6** (`20^6 = 64M` cols) → 10 GB Python memory just to build
    the vocab list/dict; 32B first-layer params. Not feasible.
  - **aa k≥7** OOMs at build on any reasonable machine.
  Pushing past k=4 for aa would require **observed-vocab** (enumerate
  only k-mers actually seen) or **feature hashing**. Neither is
  implemented; both are out of scope for this plan. See
  `docs/methods/kmer_features.md` → "Scaling and practical limits"
  for the full numeric breakdown.
- **K-mer stride**: stays 1 (the only currently supported mode). User
  asked for stride=1 explicitly; no code change needed.
- **K-mer normalization**: `normalize: none` (raw counts) stays default.
  Slot-level normalization (`unit_norm`) at training time is the
  production setting per HA/NA and PB2/PB1 bundles.
- **Backward compat for legacy nt cache**: Phase 2 writes nt with the
  new filename pattern (`kmer_features_nt_k{K}.npz`) but does **not**
  dedup nt rows yet. Phase 6 is the migration that introduces nt dedup.
  Until then, the nt npz layout is identical to legacy aside from the
  filename and the parquet adding `assembly_id` as an explicit column.
- **K-mer alphabet other than `nt` | `aa`**: not exposed via config. The
  underlying function still accepts any alphabet string, but bundles
  cannot select it without further work.
