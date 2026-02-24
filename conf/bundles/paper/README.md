# conf/bundles/paper/

Reserved for bundles that will be reported in the publication.

Planned experiments (from _roadmap.md, 02/10/2026 meeting):
- Cross-validation (N folds, mean ± std metrics)
- Temporal holdout (train 2021–2023, test 2024)
- Large dataset (full Flu A, ~100K isolates)
- PB2/PB1 + H3N2 subtype
- Genome features (k-mers + LightGBM)

Bundles here inherit from active base bundles in `conf/bundles/` (e.g., `flu_schema_raw_slot_norm_unit_diff`).
