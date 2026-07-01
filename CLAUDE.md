# CLAUDE.md — Project Context for Claude Code

Persistent, always-on context. **Behavioral rules** live here; **descriptive/reference**
material (pipeline stages, config system, source-file map, findings, roadmap, HPC) moved to
`docs/architecture.md` — read that when you need orientation on a subsystem.
Also read `.claude/memory.md` for compact, up-to-date project memory.

---

## Session Startup Checklist

1. Read `.claude/memory.md` (in the repo) for current project state.
2. If the machine-local `MEMORY.md` (path in the system prompt) doesn't exist, create it with:
   ```
   Project memory has moved into the repo for portability across machines.
   Read: .claude/memory.md (in the repo root)
   This machine-local file is no longer updated.
   ```
3. Check `docs/plans/` for in-progress plans (status != IMPLEMENTED); read and offer to resume.

**Plans** (`docs/plans/`): save new plans to `docs/plans/<descriptive_name>_plan.md` with a status
line right under the title — `**Status: IN PROGRESS**` (underway) / `**Status: IMPLEMENTED**`
(complete). When fully implemented, mark IMPLEMENTED and move the file to `docs/plans/done/`.

---

## Approval Required

Always ask for explicit confirmation before running any of these — even if it seems safe:

- `rm *` or any file/directory deletion
- `git rm *`, `git reset --hard *` / `--mixed *`, `git push --force*` / `-f *`,
  `git branch -D *`, `git rebase *`, `git clean *`
- Any command that modifies shared infrastructure, sends messages, or affects state outside this repo

(`.claude/settings.json` `permissions.deny` already hard-blocks several of these, e.g. `rm *`.)

---

## What This Project Does — orientation (full detail in `docs/architecture.md`)

**viral-segmatch** predicts whether two viral protein segments co-occur in the same isolate
(binary classification). Frozen ESM-2 embeddings (1280-dim `esm2_t33_650M_UR50D`) or k-mer
features → pairwise interaction (e.g. `unit_diff`) → MLP / sklearn-baseline classifier. Primary
virus: **Influenza A** (Bunyavirales support exists but is not maintained). Stages 1–2 run once
per `{virus}/{data_version}` (shared); Stages 3–4 are per-experiment. The stage table, Hydra
bundle system, active source-file map, key findings, roadmap, and HPC notes are in
`docs/architecture.md`.

---

## Core Vocabulary

Canonical definitions of the terms that cause the most confusion. Detail in
`docs/methods/glossary.md`; keep the two in sync, and reuse these exact terms — don't coin synonyms.

- **atom** — the indivisible routing unit. In 2D-CD (`cluster_disjoint_cc`) atom = one bipartite **CC** (one atom per CC; `atom_id == cc_id`). An atom is NOT a row.
- **rows ≠ atoms** — a positive "pair"/row is one record; `m_pos_per_cc` caps rows-*per-atom*, NOT the atom count (the cluster threshold fixes the atom count). Reserve "atom"/"CC" for components and "pair"/"row" for records.
- **CC / mega-CC** — connected component on the (slot-A cluster, slot-B cluster) bigraph; the mega-CC is the giant component that swallows most pairs at low `t`.
- **pair_key_alphabet** — the positive-dedup key is built on the alphabet's hash (aa→`prot_hash`, nt_cds→`cds_dna_hash`, nt_ctg→`ctg_dna_hash`), so the positive **universe is alphabet-defined** (nt keeps codon/contig variants that aa collapses).
- **front-end (df)** — the output of `build_frontend`: `protein_final` loaded + DNA hashes attached. The `protein_final` *file* natively carries only `prot_hash`; `ctg_dna_hash`/`cds_dna_hash` are attached *after* load from sibling files (`ctg_dna_final`/`cds_dna_final`).
- **hash source-stages** — Stage 1 writes `prot_hash` + `ctg_dna_hash`; Stage 1.5 writes `cds_dna_hash`; Stage 3 reads them (no recompute).
- **within_cc vs within_fold** — CC-builder negative scope: within_cc draws negatives inside each CC (removes the cluster shortcut; hard); within_fold draws cross-CC in-split (keeps it; easier).
- **molecule ↔ alphabet** — aa↔prot / nt_cds↔cds_dna / nt_ctg↔ctg_dna (the `aa/nt vs protein/DNA` convention below is the full rule) — the single most-crossed pairing.

---

## Conventions

- **Experiment naming**: `{virus}_{proteins}_{n_isolates}[_{modifiers}]`.
- **Timestamps**: All run directories include `YYYYMMDD_HHMMSS`.
- **Shared vs. run-specific**: Preprocessing and embeddings are shared per `{virus}/{data_version}`. Datasets and models are per run in `runs/` subdirectories.
- **Seed system**: Hierarchical — `master_seed` derives all process seeds. See `docs/SEED_SYSTEM.md`.
- **Metrics**: `metrics.csv` carries F1 (binary + macro), precision, recall, AUC-ROC, AUC-PR, MCC, Brier, BCE loss. Early-stop options: `loss`, `f1`, `auc_roc`, `auc_pr`, `mcc`. Naming: snake_case identifiers are `auc_roc` / `auc_pr`; display strings are `AUC-ROC` / `AUC-PR`. Sklearn names `roc_auc_score` / `average_precision_score` are external and left alone. Train targets neg:pos = `neg_to_pos_ratio` (default 1.0); val/test drift to ~1.07–1.20× neg-heavy because v2's coverage phase overshoots.
- **Proteins**: `preprocess_flu.py` maps GTO replicon functions to standard protein names (PB2, PB1, PA, HA, NP, NA, M1, M2, NEP).
- **Threshold notation**: `tXXX` (zero-padded, e.g. `t095`) denotes the mmseqs identity threshold at `0.XXX`. Canonical across docs, plot labels, code, bundle YAML filenames, and `cluster_id_path` refs. **Asymmetry (Phase 2)**: on-disk cluster parquets now live at `clusters_*/tXXX/` (pre-Phase-2 `idXXX` + easy-cluster artifacts archived under `clusters_*_archive_*`); existing dataset and training run dirs retain their pre-Phase-2 `idXXX` names.
- **Sequence hashes**: `prot_hash = md5(prot_seq)`, `ctg_dna_hash = md5(ctg_dna_seq)`, `cds_dna_hash = md5(cds_dna_seq)`. In pair tables: `*_hash_a` / `*_hash_b`. Per-alphabet column/file names come from one source of truth — the `SCHEMA` registry in `src/utils/schema.py` (alphabet ∈ {`aa`, `nt_cds`, `nt_ctg`}). Each hash is produced/persisted at its source stage (Stage 1 writes `prot_hash`/`ctg_dna_hash`; Stage 1.5 writes `cds_dna_hash`); Stage 3 reads them (no recompute). ESM-2 cache key uses `sha1(prot_seq)` — separate namespace, never joined back to `prot_hash`.
- **Log messages**: No emojis. Use text prefixes: `ERROR:` (fatal), `WARNING:` (non-fatal), `Done.` (success).
- **Leakage terminology**: use canonical names from `docs/plans/2026-05-07_leakage_diagnostics_plan.md` (same-pair leakage, sequence-level label imbalance, sequence-level leakage, cluster leakage, demographic shortcut leakage). New modes go in that table first.
- **aa/nt vs protein/DNA**: alphabet tokens `aa` / `nt_cds` / `nt_ctg` at the **alphabet/residue** level; molecule names `prot` / `cds_dna` / `ctg_dna` at the **molecule/sequence** level. Pairing is aa↔protein, nt_cds↔CDS DNA, nt_ctg↔contig DNA — don't cross the streams. `src/utils/schema.py` enforces it.
- **Reading CSVs with `function_short`**: any CSV with a `function_short` column has the literal string `'NA'` (Neuraminidase) as a value. Default `pd.read_csv()` parses `'NA'` as NaN and **silently drops Neuraminidase rows**. Always read with `keep_default_na=False, na_values=['']`. Source pipeline CSVs use full names (safe); derived `function_short` CSVs are vulnerable.
- **Bash tool calls**: prefer single-command invocations over compound chains (`&&`, `;`, `$(...)`, `bash -c '...'`) — the allow-list matcher only auto-approves statically-parseable commands. Use compound only when atomicity matters (`git add X && git commit ...`) or it's fundamentally one shell idiom.
- **Documentation language**: prefer plain technical verbs (`removes`, `drops`, `reads`, `writes`, `joins`) over decorative alternatives (`scrubs`, `munges`, `slurps`). Same word for the same thing throughout the repo.
- **Terminology**: `docs/methods/glossary.md` is canonical (graph-theory + project terms). Use its exact terms; add new terms there first.
- **Docs describe current state, not history**. Method/reference docs (`docs/methods/`, `CLAUDE.md`, `.claude/memory.md`) read as a stable description of how things are now. Historical framing belongs in `docs/results/` or `docs/plans/`.
- **Claims match verified evidence — no more, no less**. Don't under- or over-claim; state the scope of verification ("checked PB2 and PB1; not confirmed on the other 6") rather than rounding to a universal claim.
- **Verify before asserting; flag the unverified**. Claims about what *exists* (files, functions, flags, bundles), *current values*, or *what code does* must be checked against the source (Read / Grep / run) in the same turn before stating them — never from memory or inference. For anything not checked, say so inline ("unverified", "likely", "would need to check X"). A bare factual claim with no evidence and no hedge is a bug. And surface confusion rather than narrating around it — say "I'm not sure" *before* you explain, not after you're corrected.
- **Absence and universal claims need an exhaustive search, not a sample**. Before asserting something *doesn't exist* / is *new* / *first* / *only*, or that *all* / *none* share a property, grep the whole repo. Scope the claim to exactly what was checked, and name the exception rather than just the rule.
- **Concrete numbers in takeaways when the magnitude IS the point**. "PB2 id093 → id092: 1,085 → 112 (−90%)" beats "drops sharply". Reserve qualitative descriptors for when SHAPE matters more than magnitude.
- **Design symmetry: check before proposing**. Before naming a field/structure/API, list the dimensions it covers (slot a/b, routing modes, alphabets, splits) and verify uniformity across each. Names that fit the current example better than the alternatives bake in assumptions.
- **Commits are explicit-only**. Never run `git commit` / `--amend` on Claude's own initiative. Commit only on an explicit user instruction (a specific change, or a standing batch/session authorization). Otherwise: stage, show the diff, draft the message, and stop. `git commit` is allow-listed (no prompt), so this rule is the sole guard — apply it strictly.

---

## Per-machine Git Setup

Run once after cloning on each new machine (writes to `.git/config`, not tracked):
```bash
git config pull.rebase true   # avoid "need to reconcile divergent branches" on git pull
```
