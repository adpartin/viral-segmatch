# Duplicate Sequence Handling in Segment-Pair Classification

## Problem Statement

When building a Flu-A segment-pair dataset to predict whether two proteins come from the same genome/isolate, **identical amino-acid sequences recur across many genomes**. This creates a fundamental problem:

### The Contradictory Label Problem

```
Sequence A appears in isolates: {1, 2, 3}
Sequence B appears in isolates: {2, 5, 6}

In isolate 2: (A, B) is a POSITIVE pair (same genome)
Sampling A from isolate 1 and B from isolate 5: (A, B) is NEGATIVE (different genomes)

→ Same embeddings, contradictory labels!
```

### Why This Matters

1. **Model Confusion**: The model sees identical input (embedding_A, embedding_B) with both label=1 and label=0
2. **Data Leakage**: If the same sequence pair appears in both train and test sets, the model can memorize pairs rather than learn generalizable features
3. **Performance Ceiling**: Contradictory labels create an inherent accuracy ceiling that cannot be overcome

---

## Solution: Blocked Negatives + Pair-Key Splitting

We implement a two-part solution:

### Part 1: Block Contradictory Negatives

Before generating negative pairs, we build a **co-occurrence set** of all sequence pairs that appear together in ANY isolate:

```python
cooccur_pairs = set()
for isolate in all_isolates:
    for seq_a, seq_b in combinations(isolate.sequences, 2):
        pair_key = canonical_pair_key(seq_a.hash, seq_b.hash)
        cooccur_pairs.add(pair_key)
```

When sampling negative pairs, we **reject** any pair whose sequences co-occur:

```python
# During negative pair sampling
if seq_pair_key in cooccur_pairs:
    continue  # BLOCK: This pair has contradictory labels
```

### Part 2: Pair-Key Validation and Removal

Even with blocked negatives, the same positive pair can appear in multiple isolates. If those isolates end up in different splits, we get leakage:

```
Positive pair (A, B) exists in isolates 2 AND 3
If isolate 2 → train, isolate 3 → test:
  → Same pair appears in BOTH splits!
```

We validate that no `pair_key` appears across train/val/test, and remove overlaps:

```python
# After generating all pairs, validate no pair_key leakage
train_pair_keys = set(train_pairs['pair_key'])
val_pair_keys = set(val_pairs['pair_key'])
test_pair_keys = set(test_pairs['pair_key'])

train_val_overlap = train_pair_keys & val_pair_keys
if train_val_overlap:
    # Remove overlapping pairs from val/test (keep in train)
    val_pairs = val_pairs[~val_pairs['pair_key'].isin(train_val_overlap)]
```

**Note**: We chose this "validate and remove" approach over `GroupShuffleSplit` because:
1. We want to split by **isolates first** (keeps all proteins from an isolate together)
2. Then validate pair_key leakage as a secondary check
3. This is simpler and maintains the isolate-based partitioning

---

## Implementation Details

### Key Functions

| Function | Purpose |
|----------|---------|
| `canonical_pair_key(hash_a, hash_b)` | Creates consistent ordering for pair identification |
| `build_cooccurrence_set(df)` | Builds set of all co-occurring sequence pairs |
| `create_negative_pairs(..., cooccur_pairs)` | Samples negatives while blocking contradictory pairs |
| `split_dataset_v2(...)` | Full pipeline with blocking and validation |

### Output Files

| File | Description |
|------|-------------|
| `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv` | Dataset splits with `pair_key` column |
| `duplicate_stats.json` | Summary statistics on blocking and overlaps |
| `cooccurring_sequence_pairs.csv` | All co-occurring pairs with isolate counts |

### Statistics Tracked

```json
{
  "cooccurrence": {
    "total_cooccur_pairs": 45231,
    "pairs_in_multiple_isolates": 12340,
    "max_isolates_per_pair": 142
  },
  "negative_pair_rejections": {
    "train": {
      "blocked_cooccur": 8432,
      "duplicate_seq": 1234,
      "total_attempts": 50000
    }
  },
  "pair_key_overlaps": {
    "train_val": 0,
    "train_test": 0,
    "val_test": 0
  }
}
```

---

## Comparison with Alternative Approaches

### Approach 1: Deduplicate at Preprocessing
- **Pros**: Clean from the start, reduces compute
- **Cons**: Loses biological information (which isolates share sequences)
- **Requires**: Re-running embeddings

### Approach 2: Deduplicate at Embeddings
- **Pros**: Embeddings already deduplicated by sequence hash
- **Cons**: Doesn't fix pair-label contradiction (protein_final.csv still has duplicates)

### Approach 3: Block at Dataset Creation (CHOSEN)
- **Pros**: 
  - Preserves all biological data
  - No re-running of upstream steps
  - Directly addresses the contradiction
  - Flexible for experimentation
- **Cons**: Slightly more complex pair generation

### Approach from ChatGPT (Reference)

ChatGPT proposed a more formal schema:
1. **Deduplicate proteins first** → Create `seq_id` mapping
2. **Separate tables**: `segments.parquet`, `genome_segments.parquet`, `pair_occurrences.parquet`
3. **GroupShuffleSplit by pair_key** → Strict separation

Our approach (Phase 1) adopts the key insights while minimizing code changes:
- ✅ Block contradictory negatives (same core idea)
- ✅ Validate pair_key partitioning
- ⏳ Full schema refactor deferred to Phase 2

---

## Usage

The solution is automatic when running `dataset_segment_pairs.py`:

```bash
./scripts/run_dataset.sh flu_a --input_file data/processed/flu_a/July_2025/protein_final.csv
```

Check the output:
```bash
cat data/datasets/flu_a/.../duplicate_stats.json
```

---

## Expected Impact on Model Performance

After implementing blocked negatives:

1. **No contradictory labels** → Clean learning signal
2. **No pair-level leakage** → True generalization test
3. **Performance scales with data** → More isolates should improve results

If performance still plateaus, the issue is elsewhere (model capacity, feature quality, etc.).

---

## References

- [Karpathy's Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- ChatGPT conversation on duplicate handling (Nov 2025)

