# DNA-coverage feasibility sweep across active flu bundles

**Date:** 2026-05-08
**Script:** `eda/dna_coverage_feasibility_sweep.py`
**Scope:** apriori — runs on the Stage 1 source data (`protein_final.csv` + `genome_final.csv`) without needing any Stage 3 dataset to exist. Takes ~10 seconds for a six-bundle sweep.
**Status:** Decision adopted (option C — best-effort DNA coverage with logging) and implemented in `create_negative_pairs_v2` on 2026-05-08. HA/NA verification: built `dataset_flu_ha_na_20260508_171512` with `n_dna_uncovered = 0` across all splits; pos `n_unique` now matches neg `n_unique` on every `dna_hash` row of `split_overlap_stats.csv`. Tight bundles flagged here (PB2/PB1 + Human + H3N2 + 2024) are expected to emit non-zero `n_dna_uncovered` if built; the WARNING and `dataset_stats.json` field record the gap.

## Why this question matters

The leakage diagnostics plan (`docs/plans/2026-05-07_leakage_diagnostics_plan.md`) lists five canonical issues that can inflate test metrics. v2's negative sampling addresses two of them:

- **Same-pair leakage (#1)** — same `pair_key` in two splits. Addressed by within-split + cross-split protein-pair dedup (v2 strict mode + `forbidden_pair_keys` threading).
- **Sequence-level label imbalance (#2)** — a sequence appears only as positive (or only as negative) in train. Addressed by v2's coverage phase enforcing ≥1 negative per `seq_hash` per slot.

But mode #2 is currently fixed at the **protein** level only. K-mer features see DNA, not protein. A protein with several synonymous DNA encodings can have all DNA variants in positives but only one represented in negatives — a per-DNA label imbalance that the protein-level coverage doesn't catch. The HA/NA mixed `split_overlap_stats.csv` shows this empirically: ~14% of train slot-A DNAs have no negative counterexample.

The natural fix is to extend v2's coverage phase to the DNA level: **≥1 negative per `dna_hash` per slot**. Before changing v2, we need to know whether this is achievable. That's what this sweep answers.

## What the script measures

For each bundle filter (e.g. "HA/NA, host=Human, hn_subtype=H3N2, year=2024"):

- `K(slot, seq_hash)` = number of distinct DNA encodings of that protein on that slot. This is the **demand** — every DNA variant needs its own negative pair.
- `partner_universe` = unique seq_hashes on the *opposite* slot in the filtered data.
- `cooccur_blocked` = those partners already observed in real positive pairs with this protein (cannot be used as negatives — would be a labeling bug).
- `supply` = `partner_universe − cooccur_blocked` = how many distinct partner pair_keys we can build for this protein.
- `ratio = K / supply`. If ratio ≥ 1 for any (slot, seq_hash), DNA-level coverage is **infeasible** for that protein in this bundle: more DNA variants exist than the available partner pool can give distinct neg pair_keys.

## What "tight clade" means

A **clade** is a group of viruses sharing a recent common ancestor. A **tight clade** is a small, closely-related group — narrow genetic spread between members. In our sweep, "tight clade" emerges when filters compose: `host = Human AND hn_subtype = H3N2 AND year = 2024` selects ~5,300 isolates that all descend from the same recent H3N2 lineage circulating in humans in 2024. The resulting population is genetically homogeneous: most isolates share one of just a few dominant protein backbones, and the partner-protein universe is correspondingly small.

Tight clades make DNA-coverage harder, not easier:
- **Few unique partner proteins** in the filter → small supply.
- **High per-protein DNA redundancy** within those few backbones (synonymous codon drift accumulates against a fixed AA backbone over time) → high demand.
- Demand and supply scale in opposite directions; tight clades push them toward collision.

## Headline result

| bundle | n_isolates | K_max(a, b) | supply_min | ratio_max | verdict |
|---|---:|---:|---:|---:|---|
| flu_ha_na | 108,530 | 622, 750 | 37,086 | 0.018 | ✅ FEASIBLE |
| flu_ha_na_human_h3n2 | 32,476 | 407, 720 | 8,012 | 0.078 | ✅ FEASIBLE |
| flu_ha_na_human_h3n2_2024 | 5,346 | 175, 363 | 1,050 | 0.313 | ✅ FEASIBLE |
| flu_pb2_pb1 | 108,530 | 2,474, 3,186 | 30,271 | 0.098 | ✅ FEASIBLE |
| flu_pb2_pb1_human_h3n2 (implied filter) | 32,476 | 2,460, 3,147 | 4,970 | 0.628 | ✅ FEASIBLE (tight) |
| **flu_pb2_pb1_human_h3n2_2024 (hypothetical)** | 5,346 | 1,140, 848 | 547 | **2.084** | ❌ **INFEASIBLE** (2 seq_hashes) |

Five of six bundles are feasible. One — the tightest PB2/PB1 slice — breaks.

## What breaks on the infeasible bundle

The two failing seq_hashes in `flu_pb2_pb1_human_h3n2_2024`:

```
slot seq_hash                              K     partner_univ  cooccur_blocked  supply  ratio
a    5315b8760b3edab2dcf60f53f0d8d4c8     1,140  1,037         490              547     2.084
b    6de65c1e5937c485707a71bedeb7600c       848    914         326              588     1.442
```

Reading the first row: in this filtered population, the dominant PB2 protein (slot a) is encoded by **1,140 distinct DNA variants** across the 5,346 H3N2-2024-human isolates. To give every one of those DNAs its own neg pair_key, we'd need 1,140 distinct PB1 partner seq_hashes that have never been observed paired with this PB2. The PB1 universe in this bundle is only 1,037 seq_hashes; 490 of them already cooccur with this PB2; **net supply = 547**. Cannot create 1,140 distinct neg pair_keys when only 547 valid partners exist.

Same shape on slot b (the dominant PB1 protein, K=848 vs supply=588).

## Why this happens — the biology

PB2 and PB1 are subunits of the influenza polymerase complex. They sit under strong purifying selection at the amino-acid level (mutations that change the protein tend to break the polymerase and don't survive). They drift freely at synonymous codon sites (silent mutations don't affect the protein). Result: in any given clade, you find a few highly conserved PB2 / PB1 *protein* backbones, each with many DNA encodings accumulated over time.

HA and NA are the surface proteins under immune selection — they evolve faster at the protein level. Even within a tight H3N2-2024 clade, HA and NA show more protein-level diversity than PB2/PB1, so the partner universe per protein is larger.

This matches the conservation anchor table from the metadata-shortcut writeup: across the full ~114K-isolate dataset, HA = 41,896 unique proteins, NA = 37,488, PB2 = 33,663, PB1 = 31,226, NP = 17,684, M1 = 4,771. PB2 and PB1 are roughly 25% more conserved than HA and NA. In a tight clade like H3N2-2024 humans, this gap compounds.

So: **the wall is biological, not algorithmic.** The DNA-coverage requirement (driven by mode #2 of the leakage taxonomy) is well-defined; what the data physically provides is sometimes not enough partners for the most conserved proteins to get one neg per DNA variant.

## Decision: best-effort DNA coverage with logging

Three options were on the table:

- **(a) Hard reject** infeasible bundles. Too restrictive — `flu_pb2_pb1_human_h3n2_2024` is a legitimate research config (tight temporal+demographic slice) that should be analyzable.
- **(b) Status quo** — keep seq_hash-level coverage only and accept the per-DNA imbalance. Doesn't address mode #2 for k-mer features.
- **(c) Best-effort DNA coverage with explicit logging.** Cover as many DNA variants as `supply` allows; record which DNAs couldn't be covered as a metric in `dataset_stats.json`.

**Adopting option (c).** It gives full DNA coverage where feasible and degrades gracefully where biology imposes a wall, with the residual gap measurable per build.

Sketch of the extended coverage phase:

```
for each (slot, seq_hash H):
    for each unique dna_hash d encoding H on this slot:
        try to find a partner seq_hash that is:
          - not in cooccur(H)
          - not already used for H in this split
          - not in forbidden_pair_keys (cross-split)
        on success: emit one neg pair carrying d
        on exhaustion: record d in `uncovered_dna_hashes` and continue
```

Output additions:
- `dataset_stats.json`: a `dna_coverage` block with `n_dna_uncovered` per slot per split.
- A WARNING log line per uncovered DNA when verbose.

Mode #2 status in the taxonomy gets a nuanced verdict:
- **Protein-level**: ✅ ADDRESSED (same as today).
- **DNA-level**: ✅ ADDRESSED *where biology allows*; residual gap reported. Bundles where the residual is non-zero are flagged in `dataset_stats.json`.

## Recommendation: re-test apriori before introducing any new tight bundle

The sweep script runs in ~10 seconds on the full Flu A source data. It does not need a Stage 3 dataset to exist — it operates directly on `protein_final.csv` + `genome_final.csv` with a filter spec.

**Rule:** before adding a new bundle that combines multiple narrowing filters (host + subtype + year, or host + subtype + small year window, or any narrow geographic filter), run:

```bash
python eda/dna_coverage_feasibility_sweep.py
```

(after adding the new filter spec to the `BUNDLES` list at the top of the script). Look at `ratio_max` in the output. If `ratio_max < 1`, you're feasible — proceed with Stage 3. If `ratio_max ≥ 1`, decide whether the residual DNA-coverage gap (which option (c) will report) is acceptable for the experiment.

This rule should also be noted in `conf/bundles/README.md` under "Adding a new bundle" so future contributors find it.

## Caveats

- Apriori, single-pass over the full filtered universe. Per-split feasibility (after train/val/test partitioning) is at least as tight, since each split has a smaller partner pool. For feasible bundles this rarely matters — the headroom is huge. For the infeasible one, splits make it worse.
- The script picks one row per (assembly_id, function) deterministically by row order. v2's actual builder may pick differently; this affects which specific DNAs land in the dataset but not the K/supply distribution.

## Files

- `eda/dna_coverage_feasibility_sweep.py` — the script
- `/tmp/dna_coverage_sweep/sweep_summary.csv` — one row per bundle (regenerated on demand)
- `/tmp/dna_coverage_sweep/feas_<bundle>.csv` — per-(slot, seq_hash) detail, sorted by descending ratio

## See also

- `docs/plans/2026-05-07_leakage_diagnostics_plan.md` — the leakage taxonomy and the experiments that motivate DNA coverage
- `docs/methods/leakage.md` — canonical mode definitions; mode #2 status will be updated when the v2 extension lands
- `docs/results/2026-05-07_dna_coverage_feasibility.md` — earlier single-bundle (HA/NA) feasibility check
