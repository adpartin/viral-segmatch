# ESM-2 Embedding Throughput & Scaling Guide

This document summarizes everything we currently know about the performance characteristics of `compute_esm2_embeddings.py`, why generating embeddings for ~1.8 M Flu A proteins takes days on a single GPU, and what it will take to scale the workload up. It is meant to be the reference point before we design multi-GPU or multi-node strategies.

## 1. Why Embeddings Are Slow

| Factor | Impact | Notes |
|-------|--------|-------|
| **Model size** | 650 M parameters | `facebook/esm2_t33_650M_UR50D` has 33 transformer layers, so every forward pass is heavy even at low batch sizes. |
| **Sequence length** | Quadratic cost | Self-attention scales as *O(batch × L²)*. Flu proteins are ~1 k residues, so each step touches ~1M token pairs. |
| **No cache hits** | Full recompute | Flu run needs embeddings for 1,793,563 unique sequences → 1.8 M actual forward passes. |
| **Single GPU** | Sequential execution | Even if we increase `batch_size`, all batches still run serially on one GPU. |
| **Python-side overhead** | Minor but non-zero | Tokenization, filtering, and I/O add 5–10% overhead even when the GPU is saturated. |

### Back-of-the-envelope

```
# of sequences      = 1.793e6
tokens per sequence ≈ 1,000  (after special tokens)
batch size          = 32 → 56,048 batches
time per batch      ≈ 3.3 s (from logs)
total wall-clock    ≈ 56,048 * 3.3 s = 51.5 hours
```

When we tried `batch_size=96`, each batch took ~10 s (3× more tokens), so the net runtime barely changed. This is the key insight: **self-attention cost is dominated by token count, not batch dimension**, so raising batch size does not linearly speed things up when sequences are already long.

## 2. GPU Memory Anatomy

With Flu (`L≈1,024`), a single forward pass requires:

| Component | Formula | Example (batch=32) |
|-----------|---------|--------------------|
| Activations per layer | `batch × L × hidden_dim × dtype_size` | `32 × 1024 × 1280 × 2 bytes ≈ 84 MB` (fp16 activations) |
| Attention buffers | `batch × heads × L × L × dtype_size` | `32 × 20 × 1024² × 2 bytes ≈ 1.34 GB` (dominant term) |
| Model weights | 650 M × 2 bytes | ~1.3 GB (fp16 weights) |
| Optimizer state | N/A | We run inference-only, so no Adam states |

Actual NVIDIA-SMI shows ~8.5 GB at batch=32, ~19 GB at batch=96. This lines up with the attention buffer scaling.

## 3. Single-GPU Optimization Checklist

| Lever | Status | Notes |
|-------|--------|-------|
| FP16 storage | ✅ | We store embeddings in fp16 (`emb_storage_precision`) to cut HDF5 size. |
| FP16 compute | ⚠️ | Hugging Face does not ship a ready-to-use bf16/fp16 inference version for ESM-2; using autocast tends to break accuracy. |
| Batch size tuning | ✅ | We tried 32 and 96. Higher than 96 would OOM on V100 with 1k tokens. |
| Token length trimming | ❌ | Not currently feasible; sequences are biological data. |
| Caching | ✅ | SHA1-based dedup ensures we never recompute identical sequences. |

Conclusion: We’re compute-bound; there’s no magic flag left on a single GPU that will give us a 10× speed-up.

## 4. Multi-GPU, Single Node

The goal is to split `{protein_final.csv}` into *K* shards and run *K* embedding jobs concurrently, each bound to a different GPU (`--cuda_name cuda:0`, `cuda:1`, …).

Key topics:

1. **Sharding**  
   - Strategy: partition by row ranges or by hash buckets → store per-GPU temp outputs (`master_esm2_embeddings_tmp_rankN.h5`).
   - After all ranks finish, merge the HDF5 datasets + `master_esm2_embeddings.parquet`.

2. **Coordination**  
   - Simplest: Launch separate shell scripts (one per GPU) using `CUDA_VISIBLE_DEVICES`.
   - Better: Use Python `multiprocessing` + GPU affinity, but we must ensure each process writes to a distinct file to avoid HDF5 contention.

3. **Merging**  
   - Append rows to the canonical master cache in a deterministic order (e.g., sorted by cache key).
   - Update parquet index accordingly.

4. **Failure handling**  
   - If one GPU fails, we must be able to restart only that shard and append again.

Once this is scripted, throughput scales almost linearly up to the number of GPUs, because each job still saturates one GPU.

## 5. Multi-GPU, Multi-Node

For clusters like Polaris (ALCF):

1. **Job orchestration**  
   - Use PBS/Slurm or a workflow engine (Ray, Dask, MPI) to fan out multiple embedding processes.
   - Each process gets a unique shard ID and writes to node-local scratch to minimize cross-node I/O.

2. **Data access**  
   - Protein CSV must be accessible to all nodes (shared filesystem or staged copies).  
   - HDF5 writes should happen locally + merged later to avoid writing a single large file over a busy network.

3. **Merging strategy**  
   - Collect all partial HDF5s back to the head node.  
   - Run a merge job that concatenates the `emb` dataset, merges parquet indices, and revalidates metadata.

4. **Communication libraries**  
   - PyTorch Distributed, DeepSpeed, etc., mostly help when the *model* itself is trained in parallel. Our workload is embarrassingly parallel, so the overhead of a collective communication layer isn’t justified unless we rewrite the code to stream batches from multiple GPUs into a single process (which is complex given HDF5 writes).

## 6. Hardware Considerations

| Hardware | Pros | Cons |
|----------|------|------|
| **V100 32 GB** | Readily available on our cluster, stable | Limited memory → batch ≈ 96 max, ~50 h runtime |
| **A100 40/80 GB** | Faster tensor cores, more memory, bf16 native | Need access/quota; would cut runtime by ~2× at the same batch size |
| **H100** | Best-in-class for transformers | Similar caveats as A100, but cost/availability must be justified |

## 7. Practical Steps Before Distributed Deployment

1. **Document per-batch runtime** for different configs (done in logs).
2. **Parameterize `batch_size` in configs** (already done via Hydra, e.g., `flu_a.yaml`).
3. **Prototype sharding**: add CLI flags `--shard_id`, `--num_shards`, auto-slice the dataframe, and write to shard-specific files.
4. **Implement merge utility** that takes multiple shard outputs and creates the canonical master cache.
5. **Automate multi-GPU launches**: e.g., a bash helper that spins up `N` processes with different `--cuda_name`.
6. **Plan for monitoring**: need aggregated logging to see ETA per shard.

## 8. References & Further Reading

1. **ESM-2 Paper:** Lin, Z. et al., *Evolutionary-Scale Prediction of Atomic Level Protein Structure with a Language Model*. NeurIPS 2022.  
2. **Hugging Face Blog:** [https://huggingface.co/blog/esm](https://huggingface.co/blog/esm) – Overview of ESM models and usage patterns.  
3. **PyTorch Distributed Overview:** [https://pytorch.org/tutorials/beginner/dist_overview.html](https://pytorch.org/tutorials/beginner/dist_overview.html) – For understanding multi-GPU orchestration options.  
4. **NVIDIA CUDA Best Practices:** [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) – Guidelines for saturating GPUs (batching, memory).  
5. **Dask / Ray for HPC:**  
   - Dask Distributed: [https://docs.dask.org/en/stable/how-to/deploy.html](https://docs.dask.org/en/stable/how-to/deploy.html)  
   - Ray Tune Distributed Inference Blog: [https://www.anyscale.com/blog/ray-data-for-large-scale-inference](https://www.anyscale.com/blog/ray-data-for-large-scale-inference)
6. **ALCF Polaris Docs:** [https://www.alcf.anl.gov/support-center/polaris](https://www.alcf.anl.gov/support-center/polaris) – For job scheduling and best practices on multi-node GPU runs.

---

**Next steps:** implement sharded embedding computation to leverage the idle GPUs we already have. Once the merge path is solid, we can automate multi-node runs and bring the 52 h runtime down to something close to “hours instead of days.” When we revisit deterministic training or multi-node embeddings, this document should serve as the foundation.

