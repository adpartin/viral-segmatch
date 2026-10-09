---
name: verify-terminology
description: Use before writing or editing any code comment, docstring, or prose explanation of how this codebase works. Enforces two gates so claims match the code — a terminology gate (no invented/synonym terms) and a data-flow gate (cite the line or hedge).
---

# Verify terminology & claims before authoring

Run both gates whenever you author a comment / docstring / explanation about code in this repo.

## 1. Terminology gate
- Before naming any code entity, grep the codebase + `docs/methods/glossary.md` for an existing term for the concept, then:
  - **a term exists** → use it verbatim; do NOT coin a synonym or force-fit a near-miss. Reuse the glossary's exact phrasing.
  - **no term exists, and the concept is a technical or project-specific term (graph property, routing mode, split concept, metric, alphabet, etc.)** → add it to `docs/methods/glossary.md` first (per its intro + CLAUDE.md § Wording), then use the term you added. Never silently introduce an unglossaried project term.
  - **a standard word, file name or local variable name** → no glossary entry; define it where it is used.
- Canonical vocabulary: the "Core Vocabulary" section of `CLAUDE.md` + `docs/methods/glossary.md`.
- Don't cross the molecule↔alphabet streams: aa↔prot / nt_cds↔cds_dna / nt_ctg↔ctg_dna.

## 2. Data-flow gate
- Any claim of the form "X carries / produces / keys / joins / contains Y" must be backed by a specific line you read this turn. Cite it in chat, not in the code comment.
- If you cannot cite the line: in a code comment or docstring, leave the claim out until you have checked it; in chat or an exploratory doc, hedge it explicitly ("unverified" / "likely"). A bare data-flow claim is a bug.
- Watch the recurring conflations in this repo: file vs df (`protein_final` vs the front-end df), and source rows vs deduped universe (rows vs atoms; protein-level vs alphabet-defined).

The finished comment/docstring should contain no term that fails gate 1 and no unchecked claim that fails gate 2.
