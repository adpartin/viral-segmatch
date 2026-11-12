"""Cut a small reviewable subset out of a finished protein alignment.

Read outputs from src.analysis.align_proteins and write a FASTA holding every rare
gap pattern plus a sample of the dominant one. Rows are copied from the finished
alignment, so every alignment column is preserved and MAFFT does not run again.
Labels come from `align_proteins.build_protein_first_labels`, the same labels as
in `aligned_proteins_analysis.fasta`: the protein first, then the gap pattern, CDS
length, residue count, isolate count and `prot_hash`.

Example:
    python -m src.analysis.subset_alignment_for_review \
        --alignment_dir results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/PB2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis.align_proteins import build_protein_first_labels
from src.utils.experiment_utils import file_sha256, get_git_info
from src.utils.fasta_utils import read_fasta

SUBSET_DIR_NAME = 'subset'
MANIFEST_NAME = 'subset_manifest.json'
# Gap-pattern classes written into the labels. The dominant class is the one gap
# pattern held by the most unique proteins; every other pattern is rare.
DOMINANT = 'dominant'
RARE = 'rare'


def isolates_per_protein(alignment_dir: Path) -> pd.Series:
    """Count isolates behind each unique protein.

    `protein_cds_map.csv` holds one row per unique CDS, so a protein carried by
    several synonymous CDS has several rows that must be added together.

    Args:
        alignment_dir: Directory containing align_proteins outputs.

    Returns:
        Isolate counts indexed by `prot_hash`.
    """
    cds_map = pd.read_csv(alignment_dir / 'protein_cds_map.csv')
    counts = cds_map.groupby('prot_hash')['isolates'].sum()
    return counts


def classify_gap_patterns(gaps: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Label every protein as holding the dominant gap pattern or a rare one.

    Args:
        gaps: Rows of `gap_locations.csv`, one per unique protein.

    Returns:
        The table with a `pattern_class` column added, and the dominant pattern
        string itself.
    """
    pattern_sizes = gaps['gap_columns'].value_counts()
    largest = pattern_sizes.max()
    tied = sorted(pattern_sizes.index[pattern_sizes == largest])
    if len(tied) > 1:
        raise ValueError(f'{len(tied)} gap patterns tie for the most proteins at '
                         f'{largest} each, so the dominant pattern is not defined.')
    dominant_pattern = tied[0]
    labelled = gaps.copy()
    is_dominant = labelled['gap_columns'].eq(dominant_pattern)
    labelled['pattern_class'] = pd.Series(RARE, index=labelled.index).mask(is_dominant, DOMINANT)
    return labelled, dominant_pattern


def select_rows(labelled: pd.DataFrame, reference_n: int) -> pd.DataFrame:
    """Pick the subset rows and fix the order they are written in.

    Every rare-pattern protein is kept. The dominant block keeps the
    `reference_n` proteins carrying the most isolates, which makes the sample
    deterministic without a random seed. The dominant block is written first so
    the rare rows sit below a solid wall of the common form.

    Args:
        labelled: Protein rows carrying `pattern_class` and `isolates`.
        reference_n: How many dominant-pattern proteins to keep.

    Returns:
        The selected rows in output order.
    """
    dominant = labelled.loc[labelled['pattern_class'].eq(DOMINANT)]
    rare = labelled.loc[labelled['pattern_class'].eq(RARE)]
    if reference_n > len(dominant):
        raise ValueError(f'--reference_n {reference_n} exceeds the {len(dominant)} '
                         'proteins holding the dominant gap pattern.')
    dominant_block = dominant.sort_values(['isolates', 'prot_hash'],
                                          ascending=[False, True]).head(reference_n)
    rare_block = rare.sort_values(['cds_length', 'gap_columns', 'prot_hash'])
    selected = pd.concat([dominant_block, rare_block], ignore_index=True)
    return selected


def subset_alignment(alignment_dir: Path, reference_n: int) -> tuple[pd.DataFrame, dict[str, str], dict]:
    """Select review rows and read their aligned sequences.

    Args:
        alignment_dir: Directory containing validated align_proteins outputs.
        reference_n: How many dominant-pattern proteins to keep.

    Returns:
        The subset table in output order, the aligned sequences keyed by label,
        and the input provenance for the manifest.
    """
    paths = {
        name: alignment_dir / name for name in (
            'aligned_proteins.fasta', 'gap_locations.csv', 'protein_cds_map.csv',
            'alignment_validation.json', 'inspection_manifest.json',
            'selection_manifest.json',
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Required alignment output is missing: {path}')

    validation = json.loads(paths['alignment_validation.json'].read_text())
    inspection = json.loads(paths['inspection_manifest.json'].read_text())
    selection = json.loads(paths['selection_manifest.json'].read_text())
    if not validation.get('all_passed'):
        raise ValueError('Alignment validation did not pass.')
    if file_sha256(paths['aligned_proteins.fasta']) != validation['aligned_fasta_sha256']:
        raise ValueError('Aligned FASTA changed after validation.')
    if file_sha256(paths['gap_locations.csv']) != inspection['gap_locations_sha256']:
        raise ValueError('Gap table changed after inspection.')

    # An empty gap_columns string means a protein with no gaps, so the default
    # NaN conversion must stay off.
    gaps = pd.read_csv(paths['gap_locations.csv'], keep_default_na=False)
    if 'gap_pattern' not in gaps.columns:
        raise ValueError('gap_locations.csv has no gap_pattern column; rerun '
                         'align_proteins --inspect_only.')
    gaps['isolates'] = gaps['prot_hash'].map(isolates_per_protein(alignment_dir))
    if gaps['isolates'].isna().any():
        missing = gaps.loc[gaps['isolates'].isna(), 'prot_hash'].tolist()
        raise ValueError(f'No isolate count for {len(missing)} proteins: {missing[:3]}.')
    gaps['isolates'] = gaps['isolates'].astype(int)

    labelled, dominant_pattern = classify_gap_patterns(gaps)
    dominant_names = labelled.loc[labelled['pattern_class'].eq(DOMINANT), 'gap_pattern']
    if set(dominant_names) != {'gap_pat001'}:
        raise ValueError(f'The dominant gap pattern is named {sorted(set(dominant_names))}, '
                         'not gap_pat001.')
    selected = select_rows(labelled, reference_n)
    selected['label'] = build_protein_first_labels(selected, selection['protein'])

    aligned = read_fasta(paths['aligned_proteins.fasta'])
    sequences = {}
    expected_width = validation['aligned_width']
    for label, prot_hash in zip(selected['label'], selected['prot_hash']):
        if prot_hash not in aligned:
            raise ValueError(f'Protein {prot_hash} is absent from the aligned FASTA.')
        sequence = aligned[prot_hash]
        if len(sequence) != expected_width:
            raise ValueError(f'Protein {prot_hash} is {len(sequence)} columns wide, '
                             f'not the validated {expected_width}.')
        sequences[label] = sequence

    provenance = {
        'stage': 'subset_for_review',
        'protein': selection['protein'],
        'population': selection['population'],
        'alignment_dir': str(alignment_dir.resolve()),
        'alignment_width': expected_width,
        'proteins_in_alignment': len(aligned),
        'dominant_gap_pattern': dominant_pattern,
        'gap_patterns_in_alignment': int(labelled['gap_columns'].nunique()),
        'reference_rows': int(selected['pattern_class'].eq(DOMINANT).sum()),
        'rare_rows': int(selected['pattern_class'].eq(RARE).sum()),
        'inputs': {name: file_sha256(path) for name, path in paths.items()},
    }
    return selected, sequences, provenance


def write_subset(subset_dir: Path, protein: str, selected: pd.DataFrame,
                 sequences: dict[str, str], provenance: dict) -> dict[str, Path]:
    """Write the review FASTA, its row table and a manifest.

    Args:
        subset_dir: Directory to write into; created if absent.
        protein: Short protein name, used in the output file names.
        selected: Subset table in output order.
        sequences: Aligned sequences keyed by label, in output order.
        provenance: Input provenance from `subset_alignment`.

    Returns:
        The written paths keyed by a short name.
    """
    subset_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = subset_dir / f'{protein}_jalview_review.fasta'
    table_path = subset_dir / f'{protein}_jalview_review.csv'
    manifest_path = subset_dir / MANIFEST_NAME

    records = [f'>{label}\n{sequence}' for label, sequence in sequences.items()]
    fasta_path.write_text('\n'.join(records) + '\n')

    columns = ['label', 'prot_hash', 'pattern_class', 'gap_pattern', 'cds_length',
               'amino_acids', 'isolates', 'gap_count', 'gap_columns']
    table = selected[columns].copy()
    table.insert(0, 'row', range(1, len(table) + 1))
    table.to_csv(table_path, index=False)

    manifest = {
        **provenance,
        'script': str(Path(__file__).resolve()),
        'script_sha256': file_sha256(Path(__file__)),
        'outputs': {path.name: file_sha256(path) for path in (fasta_path, table_path)},
        'git': get_git_info(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return {'fasta': fasta_path, 'table': table_path, 'manifest': manifest_path}


def main() -> None:
    """Write a review subset of one protein alignment.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--alignment_dir', type=Path, required=True)
    parser.add_argument('--reference_n', type=int, default=39,
                        help='dominant-pattern proteins to keep (default: 39)')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    subset_dir = args.alignment_dir / SUBSET_DIR_NAME
    existing = sorted(path.name for path in subset_dir.glob('*')) if subset_dir.is_dir() else []
    if existing and not args.overwrite:
        parser.error(f'Outputs already exist in {subset_dir}: {existing}. '
                     'Pass --overwrite to replace them.')

    selected, sequences, provenance = subset_alignment(args.alignment_dir, args.reference_n)
    written = write_subset(subset_dir, provenance['protein'], selected, sequences, provenance)

    print(f'{provenance["protein"]}: {len(selected)} rows '
          f'({provenance["reference_rows"]} {DOMINANT}, {provenance["rare_rows"]} {RARE}) '
          f'of {provenance["proteins_in_alignment"]:,} proteins and '
          f'{provenance["gap_patterns_in_alignment"]} gap patterns, '
          f'at {provenance["alignment_width"]} columns.')
    print(f'Wrote {written["fasta"]}')
    print('Done.')


if __name__ == '__main__':
    main()
