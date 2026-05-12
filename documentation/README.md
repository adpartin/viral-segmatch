# viral-segmatch — user documentation

User-facing setup and how-to guides. For technical / methods / research
documentation, see [`../docs/`](../docs/) (especially
[`../docs/methods/`](../docs/methods/) which has the per-topic
reference docs).

## Files in this directory

- **[installation.md](installation.md)** — environment setup, dependencies, data layout.
- **[quick-start.md](quick-start.md)** — your first end-to-end pipeline run.
- **[troubleshooting.md](troubleshooting.md)** — common errors and fixes.
- **[analysis/results-analysis.md](analysis/results-analysis.md)** — how to read the
  analysis outputs from `analyze_stage4_train.py` and the post-hoc heatmap.
- **[development/code-structure.md](development/code-structure.md)** — high-level
  source-tree map for new contributors. (The authoritative module-by-module
  reference lives in [`../CLAUDE.md`](../CLAUDE.md).)

## Where to look for…

| Need | Where |
|---|---|
| Install / set up environment | [installation.md](installation.md) |
| Run a first experiment | [quick-start.md](quick-start.md) |
| Pipeline architecture (multi-audience synthesis) | [`../docs/methods/pipeline_overview.md`](../docs/methods/pipeline_overview.md) |
| Hydra configuration system | [`../docs/conf_guide.md`](../docs/conf_guide.md) |
| K-mer feature pipeline (Stage 2b) | [`../docs/methods/kmer_features.md`](../docs/methods/kmer_features.md) |
| GTO file schema (Stage 1 input) | [`../docs/methods/gto_format_reference.md`](../docs/methods/gto_format_reference.md) |
| Stage-1 preprocessing details | [`../docs/methods/preprocess.md`](../docs/methods/preprocess.md) |
| Leakage taxonomy + "biology learning" criterion | [`../docs/methods/leakage_definitions.md`](../docs/methods/leakage_definitions.md) |
| Per-(model × feature) preprocessing defaults | [`../docs/methods/feature_normalization.md`](../docs/methods/feature_normalization.md) |
| Seed / reproducibility system | [`../docs/SEED_SYSTEM.md`](../docs/SEED_SYSTEM.md) |
| Plans (active and completed) | [`../docs/plans/`](../docs/plans/) |
| Common errors and fixes | [troubleshooting.md](troubleshooting.md) |
| Source-code layout | [development/code-structure.md](development/code-structure.md) and [`../CLAUDE.md`](../CLAUDE.md) |
