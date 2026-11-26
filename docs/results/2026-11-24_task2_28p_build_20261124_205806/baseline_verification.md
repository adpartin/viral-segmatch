# Baseline verification of the 2026-11-24 build

Run on 2026-11-25. All 26 completed datasets match the checksums their manifest recorded at build
time, and so do the 24 site-feature caches their fits would read. Nothing was rebuilt, so this
establishes only that the saved build is intact and can serve as the reference for a later
replacement build. It is not evidence about the proposed exclusion rule, which has not been
implemented.

The check reuses `load_dataset_manifest` and `changed_files` from `src/utils/dataset_manifest.py`,
with `changed_files` given each dataset's recorded hashes and its directory as `base`. The full
`audit_pair_datasets` run, which parses every pair table, was not performed.

| Item | Result |
|---|---|
| Manifest | `task2_28p_aligned_codon_matched_20261124_205806_manifest.json`, sha256 `dbc260775b00796a3b3fe33057b0fd7b5405c39926a0f37bdbc79fdf0d4df713` |
| Code commit of the build | 12665fe, clean tree |
| Completed datasets verified | 26 of 26, no missing file and no checksum mismatch |
| Files checked per dataset | 21: 15 pair tables, 5 blocking files, 1 `resolved_config.yaml` |
| Files checked in total | 546 |
| Site-feature caches | 24 recorded, all present and unchanged |

The five blocking files of a dataset are five copies of one content, which the manifest's hashes
confirm, so they give one independent check per dataset rather than five.

## What cannot be checked yet

Reproduction and the comparison of resolved configs both need a replacement build to compare
against. When one exists, the same `changed_files` call with `base` pointed at the replacement
directory decides reproduction, because bit-identity over the 15 pair tables and 5 blocking files
implies identical fold, split, pair key, label, multiplicity and row order at once. A changed
config is expected if the rule is added as a config key, so `resolved_config.yaml` must be reported
on its own and left out of the pass criterion.

## Saved outputs of the two incomplete datasets

| Dataset | Exit code | Folds written | Saved output |
|---|---:|---|---|
| PB1-NP | 1 | `fold_0`, `fold_1`, both holding all four split files | 9 checksums recorded in `pb1_np_saved_output_checksums.json` |
| NP-M1 | 1 | none | nothing to record |

Both datasets' `cv_info.json` plans five folds, because that file is written before the folds are
built. `read_split_dirs` reads that planned list, so it reports five split directories for PB1-NP,
which holds two. Later tooling that walks a partial dataset has to use the directories that exist.

PB1-NP's two folds are not expected to match a replacement build. The proposed rule removes the
excluded group's rows from every fold, and the matched sampler draws one `rng.permutation` per group
from a single `RandomState` in sorted group order, so every group sorting after the excluded one
receives different negatives. Their value is limited to showing which rows of the other groups keep
their fold and split assignment.
