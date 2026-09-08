# Unique-sequence positive pairs with cross-validation

**Status: STAGE 1 IMPLEMENTED (2026-09-07); datasets and models not run**

The experiment status and order are maintained in
`docs/plans/2026-08-28_per_site_nt_features_plan.md` step 8. This file keeps the detailed
implementation design.

## Goal

Add an optional population-selection step to the v2 dataset builder so that every retained
positive pair has:

- one slot-A sequence that appears in no other retained positive pair; and
- one slot-B sequence that appears in no other retained positive pair.

For the HA-NA per-site experiments, slot A is HA, slot B is NA, and sequence identity is defined
by the CDS hash because the bundle uses `pair_key_alphabet: nt_cds`.

This is different from `seq_disjoint`. The existing `seq_disjoint` strategy keeps all positive
pairs and assigns connected components to one train/validation/test split using LPT-greedy
bin-packing. The proposed mechanism first selects a subset of positive pairs with unique
sequences on both sides, then applies ordinary random cross-validation to that subset.

## Proposed pipeline

```text
filtered protein rows
-> full co-occurrence blocking set
-> unique positive HA-NA pairs
-> per-side uniqueness selector
-> random cross-validation partition
-> negatives generated within each split
```

The full co-occurrence set must be built before positive-pair selection. A real positive pair
discarded by the selector must remain forbidden as a negative.

## Configuration

The following block is in `conf/dataset/default.yaml`:

```yaml
positive_pair_selection:
  method: all
  ordering: pair_key
```

Supported methods:

- `all`: keep all globally deduplicated positive pairs. This is the current behavior and the
  backward-compatible default.
- `dedup_a_then_b`: sort deterministically, apply `drop_duplicates` to slot A, then apply it to
  slot B.
- `dedup_b_then_a`: sort deterministically, apply `drop_duplicates` to slot B, then apply it to
  slot A.
- `hopcroft_karp`: select a maximum-cardinality matching from the positive-pair graph.

The selector should use the hash columns implied by `pair_key_alphabet`:

| `pair_key_alphabet` | slot A | slot B |
|---|---|---|
| `aa` | `prot_hash_a` | `prot_hash_b` |
| `nt_cds` | `cds_dna_hash_a` | `cds_dna_hash_b` |
| `nt_ctg` | `ctg_dna_hash_a` | `ctg_dna_hash_b` |

The selection method is a property of the population, not a split-routing strategy. It should
therefore live directly under `dataset`, rather than under `dataset.split_strategy`.

## Selection helper

The selection logic is in:

```text
src/datasets/_positive_pair_selection.py
```

Suggested interface:

```python
def select_positive_pairs(
    pos_df: pd.DataFrame,
    method: str,
    hash_col_a: str,
    hash_col_b: str,
) -> tuple[pd.DataFrame, dict]:
    ...
```

Start every method from a deterministic order:

```python
ordered = pos_df.sort_values("pair_key", kind="stable").reset_index(drop=True)
```

The sequential-dedup methods are then:

```python
if method == "dedup_a_then_b":
    selected = (
        ordered
        .drop_duplicates(hash_col_a, keep="first")
        .drop_duplicates(hash_col_b, keep="first")
    )

elif method == "dedup_b_then_a":
    selected = (
        ordered
        .drop_duplicates(hash_col_b, keep="first")
        .drop_duplicates(hash_col_a, keep="first")
    )
```

These are comparison methods, not optimal methods. Their results depend on which side is
deduplicated first and on the initial row order. The explicit stable sort makes the result
reproducible.

After any active selection, verify:

```python
if not selected[hash_col_a].is_unique:
    raise RuntimeError("positive-pair selection did not make slot A unique")
if not selected[hash_col_b].is_unique:
    raise RuntimeError("positive-pair selection did not make slot B unique")
if not selected["pair_key"].is_unique:
    raise RuntimeError("positive-pair selection produced duplicate pair keys")
if not set(selected["pair_key"]).issubset(set(pos_df["pair_key"])):
    raise RuntimeError("positive-pair selection produced an unknown pair")
```

## Hopcroft-Karp selection

Represent the positive population as a bipartite graph:

- each unique slot-A sequence is a left-side node;
- each unique slot-B sequence is a right-side node; and
- each observed positive pair is an edge.

Use namespaced nodes so a hash cannot be confused across sides:

```python
("a", hash_a)
("b", hash_b)
```

Then call:

```python
matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(
    graph,
    top_nodes=left_nodes,
)
```

`top_nodes` must be supplied explicitly because the graph can be disconnected. NetworkX returns
the matching in both directions, so each selected edge appears twice in the returned dictionary.
See the [NetworkX Hopcroft-Karp documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.matching.hopcroft_karp_matching.html).

Both Hopcroft-Karp and
[Eppstein matching](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.matching.eppstein_matching.html)
find a maximum-cardinality matching. Supporting both is unnecessary for the initial
implementation because they optimize the same quantity. Hopcroft-Karp should be the default.

A maximum-cardinality matching retains the largest possible number of positive pairs under the
constraint that no two selected pairs share a sequence. It does not optimize metadata balance,
host representation, lineage diversity, or any other secondary objective.

### Reproducibility

A maximum matching need not be unique. To make the chosen edge set reproducible:

1. Sort the slot-A hashes, slot-B hashes, and edges.
2. Relabel the sorted nodes with deterministic integer IDs before constructing the graph.
3. Save every selected `pair_key` in a manifest.
4. Save a SHA-256 checksum of the sorted selected `pair_key` values.
5. Record the NetworkX version.
6. Test that different `PYTHONHASHSEED` values produce the same selected manifest.

NetworkX is already used by active dataset code in this repository, so this feature does not
require a new project dependency.

## Cross-validation integration

The main integration point is `generate_all_cv_folds_v2()` in
`src/datasets/dataset_segment_pairs_v2.py`.

The current function runs `KFold` over all filtered isolate IDs. `split_dataset_v2()` later
constructs and deduplicates the positive pairs. Applying the new selector only inside
`split_dataset_v2()` would therefore be too late: CV would partition many isolates that the
selector subsequently discards, and the retained positive counts could be uneven across folds.

When `positive_pair_selection.method` is not `all`, the CV path should instead:

1. Build the full co-occurrence set from the filtered protein table.
2. Call `create_positive_pairs_v2()` once to obtain globally unique `pair_key` rows.
3. Call `select_positive_pairs()` once.
4. Run shuffled `KFold` over the selected positive rows.
5. Split each fold's remaining positive rows into training and validation.
6. Pass the three positive tables to the existing negative-generation and output path.

`prepartitioned_pos_override` passes prepartitioned positive tables from either random selected
CV or cluster-disjoint CV into `split_dataset_v2()`:

```python
{
    "train_pos": train_pos,
    "val_pos": val_pos,
    "test_pos": test_pos,
    "pos_dedup_stats": pos_dedup_stats,
    "positive_selection_audit": selection_audit,
}
```

This lets `split_dataset_v2()` skip rebuilding and routing positive pairs while retaining its
existing negative sampling, metadata annotation, overlap checks, and output code.

For `method: all`, retain the existing code path exactly. Refactoring the no-selection path to
split the already-deduplicated positive table would change existing fold assignments and would
break reproducibility of earlier bundles.

### CV invariants

Because every selected positive pair has unique endpoints, assigning selected edges to training,
validation, and test also assigns disjoint sequence pools to those splits. The negative sampler
must draw both endpoints only from the positive sequence pool assigned to that split.

Within each CV fold:

- no selected slot-A hash may occur in more than one split;
- no selected slot-B hash may occur in more than one split;
- no positive or negative pair key may occur in more than one split; and
- every negative must use sequences assigned to its own split.

Across all CV folds, each selected positive pair should appear in the test split exactly once.
It will appear in training or validation in the other folds, which is normal cross-validation
behavior.

## Negative construction

The negative-blocking set must remain the full set of observed co-occurrences from the filtered
population. It must not be recomputed from only the selected positive matching.

The current negative samplers may reuse one sequence in several negative rows within the same
split. The proposed selector guarantees one retained positive partner per sequence; it does not
guarantee that a sequence occurs only once in the complete positive-plus-negative table.

If negative rows must also use each sequence at most once, negative construction needs a separate
matching problem over allowable non-co-occurring edges. That should be a separate configuration
option and experiment.

## Measured size on the H3N2 2024 filtered population

The saved pinned-length dataset contains:

- 3,580 unique positive HA-NA pairs;
- 2,732 unique HA CDS sequences; and
- 2,298 unique NA CDS sequences.

Using the saved positive pairs sorted by `pair_key` gives:

| Selector | Positive pairs retained | Fraction of 3,580 |
|---|---:|---:|
| Dedup HA then NA | 1,687 | 47.1% |
| Dedup NA then HA | 1,703 | 47.6% |
| Hopcroft-Karp | 1,782 | 49.8% |

Hopcroft-Karp retains 95 more pairs than HA-first deduplication and 79 more than NA-first
deduplication. Its
1,782 pairs contain 1,782 unique HA sequences and 1,782 unique NA sequences.

The sequential-dedup methods are order-dependent and need not produce a maximal matching; only
Hopcroft-Karp gives the maximum number of retained positive pairs.

The theoretical upper bound is 2,298 pairs, the size of the smaller sequence set. The observed
maximum is lower because the structure of the HA-NA co-occurrence graph does not permit every NA
sequence to be paired with a distinct HA sequence.

With four-fold CV, 1,782 positive pairs would give approximately 445 test positives per fold,
with the remaining pairs divided between training and validation. The configured 1:1 negative
ratio would produce roughly the same number of negatives in each split.

## Output artifacts

Write a run-level `positive_pair_selection.json` containing at least:

```yaml
method:
pair_key_alphabet:
hash_col_a:
hash_col_b:
input_pairs:
selected_pairs:
dropped_pairs:
input_unique_a:
input_unique_b:
selected_unique_a:
selected_unique_b:
unmatched_a:
unmatched_b:
retained_fraction:
ordering:
networkx_version:
selected_pair_keys_sha256:
```

Also write `positive_pair_selection.csv` containing the selected `pair_key`, both sequence hashes,
and the representative `assembly_id`. Include the selection summary in each fold's
`duplicate_stats.json` so downstream audits can identify the population used.

## Bundles

Keep the existing pinned-length bundle unchanged and add three bundles that inherit it:

```text
flu_ha_na_h3n2_2024_random_cv4_pinned_length_dedup_a_then_b.yaml
flu_ha_na_h3n2_2024_random_cv4_pinned_length_dedup_b_then_a.yaml
flu_ha_na_h3n2_2024_random_cv4_pinned_length_hopcroft_karp.yaml
```

Each bundle should change only `dataset.positive_pair_selection.method`. All feature comparisons
within one selected dataset must reuse its saved folds.

Scores obtained from different selection methods describe different positive populations. They
should not be presented as paired model comparisons unless the evaluation is restricted to a
common set of pairs.

## Tests

Add unit tests for:

- `all` returning the input population unchanged;
- both sequential-dedup directions satisfying per-side uniqueness;
- a graph where the two sequential-dedup directions return different populations;
- a graph where sequential dedup retains fewer edges than Hopcroft-Karp;
- Hopcroft-Karp returning the known maximum cardinality;
- disconnected graphs and isolated nodes;
- deterministic output under input-row permutations;
- validation of unknown methods and missing hash columns; and
- correct audit counts and checksum.

Add CV integration tests for:

- zero slot-A and slot-B hash overlap among train, validation, and test within every fold;
- every selected positive pair appearing in test exactly once across all folds;
- negatives using only sequences from their assigned split;
- discarded real positive pairs remaining blocked as negatives;
- global positive and negative `pair_key` uniqueness within a fold;
- deterministic fold output for a fixed seed; and
- `method: all` preserving existing output exactly.
