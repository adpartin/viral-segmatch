# examples/

Reference scripts demonstrating HuggingFace APIs. These are **not part of the project pipeline** — they were written during early exploration to understand the ESM-2 / HuggingFace ecosystem.

## Files

| Script | Purpose |
|--------|---------|
| `dataset_huggingface_example.py` | HuggingFace Dataset API example (cytosolic vs membrane classification) |
| `train_huggingface_example_auto_api.py` | HuggingFace Trainer API for sequence classification |
| `train_huggingface_example_explicit.py` | Explicit training loop with HuggingFace ESM-2 |

The project's actual training uses `src/models/train_esm2_frozen_pair_classifier.py` with frozen ESM-2 embeddings precomputed via `src/embeddings/compute_esm2_embeddings.py`.
