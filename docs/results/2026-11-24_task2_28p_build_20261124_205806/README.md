# Task 2 build of 2026-11-24: 26 of 28 datasets

One job attempted all 28 schema pairs on 2026-11-24 under the dataset prefix
`task2_28p_aligned_codon_matched_20261124_205806_`. Twenty-six wrote all five folds. PB1-NP and
NP-M1 stopped partway, so 26 of the 28 schema pairs have a dataset. No model was fitted on any of
these 26 datasets. The study design is in `docs/plans/2026-11-24_paper_plan_v2.md`.

The dataset directories under `data/datasets/` are untracked and may be removed to reclaim disk,
so these files are kept here as the record of what the run produced.

| File | Contents |
|---|---|
| `manifest.json` | The run manifest, copied unchanged. It holds an exit code for each of the 28 schema pairs, dataset entries with per-file sha256 for the 26 that finished, the experiment spec and capacity table checksums, and the code commit. |
| `per_pair_counts.csv` | One row per schema pair: build exit code, folds written, the counts through *positives kept*, and the failing fold, split and metadata group for the two failures. |
| `np_m1_stage3.log` | The NP-M1 Stage 3 log, ending in the traceback. |
| `pb1_np_stage3.log` | The PB1-NP Stage 3 log, ending in the traceback. |
| `baseline_verification.md` | The 2026-11-25 check of every recorded file against the manifest, and what it does and does not establish. |
| `pb1_np_saved_output_checksums.json` | Checksums of PB1-NP's two written folds, which the manifest does not carry, so a later build can be compared against them. |

Each of the 28 builds logged the same eligible isolate, unique positive and HK-selected counts as
the capacity table at
`results/flu/July_2025/pair_capacity_8_proteins_4host_3subtype_2010_2025_aligned/pair_capacity.csv`.
The builds ran at commit 12665fe on branch `feature/task2-full-population` with a clean tree.

## Why PB1-NP and NP-M1 stopped

Both failures came from the matched negative draw on a validation split. The matched sampler gives
each slot-A sequence of a metadata group one slot-B sequence taken from another positive of the same
group, and it uses each sequence once. A group's own positives are observed pairs, so no sequence
may take its own partner. When a split holds only 2 positives of a group, exchanging the two
partners is the only assignment left, and the draw fails if that pair is observed anywhere in the
corpus or was already drawn. PB1-NP failed on Duck/H1N1/2012 in the validation split of fold 3 of 5
(`fold_2`). NP-M1 failed on Duck/H1N1/2014 in the validation split of fold 1 of 5 (`fold_0`). In
PB1-NP, Duck/H1N1/2012 held exactly 20 positives, which is the metadata group minimum, and the
validation split took 2 of them. The minimum of 20 therefore admits groups whose validation split
reaches the smallest size the sampler can assign at all.

The sampler raises rather than returning fewer negatives, so a single such group stops the whole
build for that schema pair. `matched_balanced_usage_negatives` in
`src/datasets/_negative_sampling.py` holds this rule. PB1-NP had written `fold_0` and `fold_1`
before it stopped. NP-M1 stopped in its first fold and wrote no fold.

Raising the minimum or relaxing the matching axes for small groups would remove this failure.
Neither change has been made, and the choice of population is open.
