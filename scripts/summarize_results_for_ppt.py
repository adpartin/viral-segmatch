#!/usr/bin/env python3
"""
Summarize training_analysis outputs under results/ into a PPT-friendly table and assets folder.

Designed for the viral-segmatch repo conventions:
- results/flu/July_2025/<experiment_dir>/training_analysis/{metrics.csv,segment_performance.csv,confusion_matrix.csv,...}
- conf/bundles/<bundle>.yaml provides dataset filters + feature flags (concat/diff/prod)

Outputs (by default) into:
- results/flu/July_2025/ppt_summary/feature_mode_results.csv
- results/flu/July_2025/ppt_summary/feature_mode_results.md
- results/flu/July_2025/ppt_assets/<experiment_dir>/*  (copies selected plots)
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TARGET_BASES_DEFAULT = ["flu_2024", "flu_human", "flu_h3n2", "flu_ha_na_5ks"]
TARGET_MODES_DEFAULT = ["concat", "diff", "prod"]

PREFERRED_PLOTS = [
    "performance_summary.png",
    "confusion_matrix.png",
    "precision_recall_curve.png",
    "roc_curve.png",
    "model_calibration.png",
    "prediction_distribution.png",
    "performance_metrics_by_metadata.png",
    "error_rates_by_metadata.png",
    "fp_fn_analysis.png",
    "biological_insights.png",
]


@dataclass(frozen=True)
class BundleContext:
    bundle_file: str
    base_bundle: Optional[str]
    filters: str
    selected_functions: str
    use_concat: Optional[bool]
    use_diff: Optional[bool]
    use_prod: Optional[bool]


def _as_bool_or_none(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in {"true", "yes", "1"}:
            return True
        if xl in {"false", "no", "0"}:
            return False
    return None


def _extract_base_bundle(defaults: Any) -> Optional[str]:
    """
    Hydra 'defaults' may be a list of strings/dicts. Our bundles typically use:
      defaults:
        - flu_2024
        - _self_
    Return the first string-like entry that looks like a local bundle name.
    """
    if not isinstance(defaults, list):
        return None
    for item in defaults:
        if isinstance(item, str):
            if item != "_self_" and not item.startswith("/"):
                return item
        # ignore dict-based defaults here; too repo-specific
    return None


def _load_bundle_context_no_yaml(bundle_name_or_file: str, repo_root: Path) -> BundleContext:
    """
    Minimal, dependency-free parser for our simple bundle YAML files.
    Extracts:
    - defaults -> base bundle
    - dataset.host/year/hn_subtype
    - virus.selected_functions
    - training.interaction (or legacy use_concat/use_diff/use_prod)
    """
    bundles_dir = repo_root / "conf" / "bundles"
    bundle_file = bundle_name_or_file if bundle_name_or_file.endswith(".yaml") else f"{bundle_name_or_file}.yaml"
    bundle_path = bundles_dir / bundle_file
    if not bundle_path.exists():
        raise FileNotFoundError(f"Cannot find bundle config: {bundle_path}")

    def parse_file(p: Path) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {
            "defaults": [],
            "dataset": {},
            "training": {},
            "selected_functions": [],
        }

        section: Optional[str] = None  # "defaults" | "dataset" | "virus" | "training"
        in_selected_functions = False
        with p.open("r") as f:
            for raw in f:
                line = raw.rstrip("\n")
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                # top-level section headers
                if not line.startswith(" "):
                    in_selected_functions = False
                    if stripped.startswith("defaults:"):
                        section = "defaults"
                        continue
                    if stripped.startswith("dataset:"):
                        section = "dataset"
                        continue
                    if stripped.startswith("virus:"):
                        section = "virus"
                        continue
                    if stripped.startswith("training:"):
                        section = "training"
                        continue
                    section = None
                    continue

                # section content
                if section == "defaults":
                    # e.g. "  - flu_2024"
                    if stripped.startswith("- "):
                        val = stripped[2:].strip().strip('"').strip("'")
                        ctx["defaults"].append(val)
                    continue

                if section == "dataset":
                    # e.g. "  host: \"Human\""
                    for k in ("host", "year", "hn_subtype"):
                        if stripped.startswith(f"{k}:"):
                            val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                            if val in {"null", "None", ""}:
                                val = None
                            ctx["dataset"][k] = val
                    continue

                if section == "virus":
                    if stripped.startswith("selected_functions:"):
                        in_selected_functions = True
                        continue
                    if in_selected_functions and stripped.startswith("- "):
                        val = stripped[2:].strip().strip('"').strip("'")
                        ctx["selected_functions"].append(val)
                    # stop selected_functions on next key
                    if in_selected_functions and (":" in stripped) and not stripped.startswith("- "):
                        in_selected_functions = False
                    continue

                if section == "training":
                    for k in ("use_concat", "use_diff", "use_prod", "interaction"):
                        if stripped.startswith(f"{k}:"):
                            val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                            ctx["training"][k] = val
                    continue

        return ctx

    cfg = parse_file(bundle_path)
    base = None
    for d in cfg.get("defaults", []):
        if isinstance(d, str) and d != "_self_" and not d.startswith("/"):
            base = d
            break

    base_cfg: Dict[str, Any] = {"dataset": {}, "training": {}, "selected_functions": []}
    if base:
        base_path = bundles_dir / f"{base}.yaml"
        if base_path.exists():
            base_cfg = parse_file(base_path)

    # merged dataset/training + selected_functions (base then derived)
    dataset_cfg = dict(base_cfg.get("dataset", {}) or {})
    dataset_cfg.update(cfg.get("dataset", {}) or {})

    training_cfg = dict(base_cfg.get("training", {}) or {})
    training_cfg.update(cfg.get("training", {}) or {})

    funcs: List[str] = []
    funcs.extend(base_cfg.get("selected_functions", []) or [])
    funcs.extend(cfg.get("selected_functions", []) or [])
    funcs = [f for f in funcs if f]
    selected_functions = "; ".join(funcs) if funcs else "N/A"

    def _interaction_to_use_flags(interaction: Optional[str]) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
        """Derive use_concat, use_diff, use_prod from interaction spec (e.g. 'concat+unit_diff')."""
        if not interaction:
            return None, None, None
        tokens = [t.strip().lower() for t in str(interaction).split("+") if t.strip()]
        return ("concat" in tokens, "diff" in tokens, "prod" in tokens)

    interaction = training_cfg.get("interaction")
    use_concat, use_diff, use_prod = training_cfg.get("use_concat"), training_cfg.get("use_diff"), training_cfg.get("use_prod")
    if use_concat is None and use_diff is None and use_prod is None and interaction:
        use_concat, use_diff, use_prod = _interaction_to_use_flags(interaction)
    else:
        use_concat = _as_bool_or_none(use_concat)
        use_diff = _as_bool_or_none(use_diff)
        use_prod = _as_bool_or_none(use_prod)

    host = dataset_cfg.get("host")
    year = dataset_cfg.get("year")
    subtype = dataset_cfg.get("hn_subtype")

    filter_parts = []
    if host is not None:
        filter_parts.append(f"host={host}")
    if year is not None:
        filter_parts.append(f"year={year}")
    if subtype is not None:
        filter_parts.append(f"subtype={subtype}")
    filters = ", ".join(filter_parts) if filter_parts else "None"

    return BundleContext(
        bundle_file=bundle_file,
        base_bundle=base,
        filters=filters,
        selected_functions=selected_functions,
        use_concat=training_cfg.get("use_concat"),
        use_diff=training_cfg.get("use_diff"),
        use_prod=training_cfg.get("use_prod"),
    )


def load_bundle_context(bundle_name_or_file: str, repo_root: Path) -> BundleContext:
    # Keep the higher-level name for callers, but implement without external YAML deps.
    return _load_bundle_context_no_yaml(bundle_name_or_file, repo_root=repo_root)


def parse_confusion_matrix(cm_csv: Path) -> Tuple[int, int, int, int]:
    """
    confusion_matrix.csv schema:
      Actual,Predicted Negative,Predicted Positive
      True Negative,TN,FP
      True Positive,FN,TP
    """
    import csv

    with cm_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    by_actual = {r["Actual"]: r for r in rows}
    tn = int(by_actual["True Negative"]["Predicted Negative"])
    fp = int(by_actual["True Negative"]["Predicted Positive"])
    fn = int(by_actual["True Positive"]["Predicted Negative"])
    tp = int(by_actual["True Positive"]["Predicted Positive"])
    return tn, fp, fn, tp


def find_experiment_dir(results_root: Path, bundle_base: str, mode: str) -> Optional[Path]:
    """
    Handle naming differences between legacy and current conventions.

    Current (preferred):
    - flu_2024_concat/
    - flu_ha_na_5ks_concat.yaml/

    Legacy (backward compatible):
    - flu_2024_use_concat/
    - flu_ha_na_5ks_use_concat.yaml/
    """
    candidates = [
        results_root / f"{bundle_base}_{mode}",
        results_root / f"{bundle_base}_{mode}.yaml",
        results_root / f"{bundle_base}_use_{mode}",
        results_root / f"{bundle_base}_use_{mode}.yaml",
    ]
    for c in candidates:
        if (c / "training_analysis" / "metrics.csv").exists():
            return c
    return None


def copy_assets(training_analysis_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in PREFERRED_PLOTS:
        src = training_analysis_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_root",
        type=str,
        default="results/flu/July_2025",
        help="Root folder containing experiment subfolders",
    )
    ap.add_argument(
        "--repo_root",
        type=str,
        default=".",
        help="Repo root (for conf/bundles lookup)",
    )
    ap.add_argument(
        "--bases",
        type=str,
        default=",".join(TARGET_BASES_DEFAULT),
        help="Comma-separated base bundle names (e.g., flu_2024, flu_human, ...)",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default=",".join(TARGET_MODES_DEFAULT),
        help="Comma-separated feature modes to include (concat,diff,prod)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <results_root>/ppt_summary)",
    )
    ap.add_argument(
        "--assets_dir",
        type=str,
        default=None,
        help="Assets output directory (default: <results_root>/ppt_assets)",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else (results_root / "ppt_summary")
    assets_dir = Path(args.assets_dir) if args.assets_dir else (results_root / "ppt_assets")

    bases = [b.strip() for b in str(args.bases).split(",") if b.strip()]
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]

    rows = []
    missing = []

    for base in bases:
        for mode in modes:
            exp_dir = find_experiment_dir(results_root, base, mode)
            if exp_dir is None:
                missing.append((base, mode))
                continue

            ta = exp_dir / "training_analysis"
            metrics_path = ta / "metrics.csv"
            seg_path = ta / "segment_metrics.csv"
            cm_path = ta / "confusion_matrix.csv"

            import csv

            # metrics.csv is a 1-row table
            with metrics_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                metrics_row = next(reader)

            # segment_metrics.csv may contain multiple rows; we compute overall test_n and pos_rate
            test_n = 0
            pos_weighted_sum = 0.0
            with seg_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    c = int(float(r["count"]))
                    pr = float(r["pos_rate"])
                    test_n += c
                    pos_weighted_sum += c * pr
            test_pos_rate = float(pos_weighted_sum / max(1, test_n))

            tn = fp = fn = tp = None
            if cm_path.exists():
                tn, fp, fn, tp = parse_confusion_matrix(cm_path)

            # bundle config context
            # Experiment dir names may omit ".yaml" even though bundle file uses it
            bundle_guess = exp_dir.name
            bundle_context: Optional[BundleContext] = None
            try:
                bundle_context = load_bundle_context(bundle_guess, repo_root=repo_root)
            except FileNotFoundError:
                # try stripping ".yaml" if it is a directory artifact
                if bundle_guess.endswith(".yaml"):
                    try:
                        bundle_context = load_bundle_context(bundle_guess[:-5], repo_root=repo_root)
                    except FileNotFoundError:
                        bundle_context = None
                else:
                    bundle_context = None

            rows.append(
                {
                    "experiment": exp_dir.name,
                    "base": base,
                    "mode": mode,
                    "filters": (bundle_context.filters if bundle_context else "N/A"),
                    "selected_functions": (bundle_context.selected_functions if bundle_context else "N/A"),
                    "use_concat": (bundle_context.use_concat if bundle_context else None),
                    "use_diff": (bundle_context.use_diff if bundle_context else None),
                    "use_prod": (bundle_context.use_prod if bundle_context else None),
                    "test_n": test_n,
                    "test_pos_rate": test_pos_rate,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "accuracy": float(metrics_row.get("accuracy")),
                    "f1": float(metrics_row.get("f1_score")),
                    "f1_macro": float(metrics_row.get("f1_macro")),
                    "auc_roc": float(metrics_row.get("auc_roc")),
                    "auc_pr": float(metrics_row.get("avg_precision")),
                    "plots_dir": str((exp_dir / "training_analysis").as_posix()),
                }
            )

            # Copy assets
            copy_assets(ta, assets_dir / exp_dir.name)

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "feature_mode_results.csv"
    out_md = out_dir / "feature_mode_results.md"
    out_notes = out_dir / "slide_notes.md"

    # Write CSV (standard library, deterministic column order)
    import csv

    if rows:
        # sort by base/mode for readability
        rows_sorted = sorted(rows, key=lambda r: (str(r.get("base")), str(r.get("mode"))))
        fieldnames = list(rows_sorted[0].keys())
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)
    else:
        with out_csv.open("w") as f:
            f.write("")

    # Markdown table
    with out_md.open("w") as f:
        f.write("# Feature-mode Experiment Summary (PPT-ready)\n\n")
        if not rows:
            f.write("_No experiments found._\n")
        else:
            cols = [
                "base",
                "mode",
                "filters",
                "test_n",
                "test_pos_rate",
                "f1",
                "f1_macro",
                "auc_roc",
                "auc_pr",
                "fp",
                "fn",
            ]
            # render a simple GitHub-flavored markdown table
            rows_sorted = sorted(rows, key=lambda r: (str(r.get("base")), str(r.get("mode"))))
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            f.write(header + "\n")
            f.write(sep + "\n")
            for r in rows_sorted:
                # format floats compactly
                cells = []
                for c in cols:
                    v = r.get(c)
                    if isinstance(v, float):
                        if c == "test_pos_rate":
                            cells.append(f"{v:.3f}")
                        else:
                            cells.append(f"{v:.4f}")
                    else:
                        cells.append("" if v is None else str(v))
                f.write("| " + " | ".join(cells) + " |\n")
            f.write("\n")

        if missing:
            f.write("## Missing expected experiments\n\n")
            for base, mode in missing:
                f.write(f"- {base} / {mode}\n")
            f.write("\n")

    # Slide notes (context)
    with out_notes.open("w") as f:
        f.write("# Slide Notes / Context\n\n")
        f.write("## What changed across runs?\n\n")
        f.write(
            "- Same dataset filter condition (row group), but different **pair feature construction**:\n"
            "  - **concat**: `[emb_a, emb_b]` (default)\n"
            "  - **diff**: `|emb_a - emb_b|`\n"
            "  - **prod**: `emb_a * emb_b`\n\n"
        )
        f.write("## Where are the plots?\n\n")
        f.write(f"- Curated copies: `{(assets_dir).as_posix()}/<experiment>/...`\n")
        f.write("- Full run outputs: `<experiment>/training_analysis/`\n\n")
        f.write("## Notes for interpretation\n\n")
        f.write(
            "- **F1** is the positive-class F1 (from `metrics.csv`).\n"
            "- **AUC-PR** can be sensitive to class imbalance; compare alongside **test_pos_rate**.\n"
            "- **FP/FN** are from the test confusion matrix; use `fp_fn_analysis.png` for context.\n\n"
        )

    print(f"Saved summary CSV: {out_csv}")
    print(f"Saved summary MD:  {out_md}")
    print(f"Saved slide notes: {out_notes}")
    print(f"Saved PPT assets:  {assets_dir}")


if __name__ == "__main__":
    main()

