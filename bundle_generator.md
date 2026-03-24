# Task 11: All Protein-Pair Combinations on Polaris

Design notes for scaling 28 pairwise protein experiments (C(8,2)) on ALCF Polaris.

---

## Polaris Constraints

**Allocation unit: full node** (4 A100 GPUs, 32 cores, 512 GB RAM). No sub-node allocation.
You CAN run 4 independent processes per node via `CUDA_VISIBLE_DEVICES` + background `&` + `wait`.

### Relevant queues

| Queue | Nodes | Walltime | Key constraint |
|-------|-------|----------|----------------|
| **debug** | 1-2 | up to 1h | Max 24 nodes shared across all debug users; only 8 exclusive |
| **debug-scaling** | 1-10 | up to 1h | 1 job per user |
| **prod** | **10-496** | up to 24h | Routes to small (10-24, 3h), medium (25-99, 6h), large (100-496, 24h) |
| **preemptable** | 1-10 | up to 72h | Can be killed anytime; 20 concurrent jobs/project |
| **capacity** | 1-4 | up to **168h** (7 days) | Max 2 queued, 1 running per user |

### Key implication

- **Prod requires minimum 10 nodes per job.** Single-node job array elements cannot go to prod.
- **Job arrays** (`#PBS -J 0-N`) are supported. Each element is an independent job with its own node allocation.
- **Best strategy: ensemble packing within nodes + job arrays across nodes.** Each array element gets 1 node and runs 4 tasks internally (one per GPU).

---

## Scaling Phases

### Phase 0 -- Env validation (debug, 1 node, 30 min)

- Validate conda, GPU, data paths on compute node
- Single pair (HA/NA), 1 fold, 5 epochs, 5K isolates
- 1 process on 1 GPU -- just verify the pipeline runs

### Phase 1 -- Small grid (debug, 1-2 nodes, 1h)

- 4 representative pairs, 2 folds, 10 epochs, 5K isolates
- 4 pairs x 2 folds = 8 tasks / 4 GPUs per node = 2 nodes
- Tests bundle generation, task packing, aggregation, heatmap plotting
- ~10 min per task -- fits easily in 1h

### Phase 2 -- All pairs, small data (capacity or preemptable)

- 28 pairs, 5 folds, 30 epochs, 5K isolates
- 140 tasks / 4 GPUs per node = 35 node-slots
- **Capacity** (4 nodes): 35/4 = 9 waves x ~10min = 1.5h. Fits in 168h.
- **Preemptable** (10 nodes): 35/10 = 4 waves x ~10min = 40min. Fits in 72h.
- Purpose: first real 8x8 heatmap with variance estimates

### Phase 3 -- Full run (capacity or preemptable)

- 28 pairs, 10 folds, 100 epochs (patience=100), ~111K isolates
- 280 tasks / 4 GPUs per node = 70 node-slots
- Each task ~1.5-3h (from runtime analysis, full dataset with early stopping)
- **Capacity** (4 nodes): 70/4 = 18 waves x 3h = **54h**. Fits in 168h.
- **Preemptable** (10 nodes): 70/10 = 7 waves x 3h = **21h**. Fits in 72h but can be killed.

### Allocation cost

IMPROVE_Aim1 has ~97K node-hours remaining (expires 2026-04-01).
Full Task 11: 280 tasks x 3h / 4 tasks-per-node = **210 node-hours** (0.2% of allocation).

---

## Bundle Design -- Inheritance with Concise Naming

The 28 child bundles specify **only** `schema_pair`. Everything else flows from the parent.
To change sample size, epochs, or architecture for all 28 experiments: edit one line in the parent.

### Inheritance chain

```
flu.yaml                          # Gen1 base (virus, paths, defaults)
  └── flu_schema.yaml             # pair_mode: schema_ordered, no filters
        └── pol_kmer_cv10.yaml    # Parent: kmer, slot_norm, concat, cv10, epochs, isolates
              ├── pol_kmer_cv10_ha_na.yaml
              ├── pol_kmer_cv10_pb2_pb1.yaml
              ├── pol_kmer_cv10_ha_np.yaml
              └── ... (28 total)
```

### Parent bundle: `pol_kmer_cv10.yaml`

```yaml
# STATUS: active -- Polaris all-pairs parent (k-mer k=6, slot_norm + concat, CV10)
# Children override only schema_pair. Update this file to change shared settings
# (epochs, isolates, architecture) for all 28 protein-pair experiments.
defaults:
  - flu_schema
  - /kmer: default
  - _self_

dataset:
  n_folds: 10
  max_isolates_to_process: null    # full dataset; set to 5000 for test runs

training:
  feature_source: kmer
  interaction: concat
  slot_transform: slot_norm
  epochs: 100
  patience: 100
```

### Child bundle example: `pol_kmer_cv10_ha_na.yaml`

```yaml
# STATUS: active -- HA/NA pair (Polaris all-pairs experiment)
defaults:
  - pol_kmer_cv10
  - _self_

dataset:
  schema_pair:
    - "Hemagglutinin precursor"
    - "Neuraminidase protein"
```

### Protein short codes

| Short | Full function string | Segment |
|-------|---------------------|---------|
| `pb2` | RNA-dependent RNA polymerase PB2 subunit | S1 |
| `pb1` | RNA-dependent RNA polymerase catalytic core PB1 subunit | S2 |
| `pa`  | RNA-dependent RNA polymerase PA subunit | S3 |
| `ha`  | Hemagglutinin precursor | S4 |
| `np`  | Nucleocapsid protein | S5 |
| `na`  | Neuraminidase protein | S6 |
| `m1`  | Matrix protein 1 | S7 |
| `ns1` | Non-structural protein 1, interferon antagonist and host mRNA processing inhibitor | S8 |

Bundle naming: `pol_kmer_cv10_{protA}_{protB}` (e.g., `pol_kmer_cv10_ha_na`, `pol_kmer_cv10_pb2_m1`).

### Bundle generator script

`scripts/generate_pair_bundles.py`:
1. Takes parent bundle name and protein short-name-to-function-string mapping
2. Generates all C(8,2) = 28 YAML files
3. Each child file is ~8 lines (status comment, defaults, schema_pair)

---

## What Needs to Be Built

1. **Bundle generator** (`scripts/generate_pair_bundles.py`) -- produce 28 YAML files programmatically
2. **Task packer / launcher** -- PBS job script that packs 4 tasks per node (CUDA_VISIBLE_DEVICES + background + wait), supports job arrays across nodes
3. **Cross-pair aggregation + heatmap** -- collect 28 CV summaries into 8x8 AUC/F1 matrix

---

## HPC Concepts

**Node = scheduling atom.** 1 node = 4 A100s connected via NVLink (600 GB/s GPU-to-GPU). The scheduler allocates whole nodes -- you pay for all 4 GPUs even if you use 1.

**Job arrays vs ensemble packing.** Two strategies for many independent tasks:
- **Job arrays** (`-J 0-N`): PBS creates N+1 independent jobs, each with its own node allocation. Each job has queue wait time.
- **Ensemble packing**: One job, N nodes, internal script distributes tasks across GPUs. Single queue wait, better utilization. Trade-off: if one task crashes, need retry logic.

**Hybrid approach**: Job arrays across nodes + ensemble packing within each node. Each array element gets 1 node and runs 4 tasks internally.

**Lustre (Eagle).** Parallel filesystem optimized for large sequential I/O (HDF5, CSV), not small random files. All data is on Eagle (`/lus/eagle/projects/IMPROVE_Aim1/`).

**Walltime vs node-hours.** Walltime is clock time your job runs. Node-hours = nodes x walltime. A 4-node job running 54h = 216 node-hours.

---

## Open Questions

1. **Parent bundle naming**: `pol_kmer_cv10` or shorter (e.g., `pol_k6_cv10`)?
2. **Scaling knob**: Change `max_isolates_to_process` in the parent bundle for test runs, or have separate parents (`pol_kmer_cv10_5k`, `pol_kmer_cv10_full`)?
3. **Queue preference for full run**: Capacity (4 nodes, 168h, reliable) vs preemptable (10 nodes, 72h, faster but killable)?
4. **ESM-2 comparison**: K-mers only first, or also generate `pol_esm2_cv10` parent for parallel comparison?
