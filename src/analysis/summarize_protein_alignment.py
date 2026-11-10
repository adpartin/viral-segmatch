"""Summarize protein alignment gaps and input lengths by population group.

Read outputs from src.analysis.align_proteins. The per-column occupancy table
counts both distinct proteins and isolates carrying them; it does not judge
whether an aligned column is biologically correct.

Example:
    python -m src.analysis.summarize_protein_alignment \
        --alignment_dir results/flu/July_2025/cds_alignment_pilot/Chicken+Duck+Human+Swine-H1N1+H3N2+H5N1-yr2010-2025/NS1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.experiment_utils import file_sha256, get_git_info
from src.utils.fasta_utils import read_fasta

OUTPUT_NAMES = (
    'group_length_summary.csv', 'gap_pattern_summary.csv',
    'column_occupancy.csv', 'alignment_summary_manifest.json',
)


def summarize(alignment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build group, gap-pattern and column-occupancy summaries.

    Args:
        alignment_dir: Directory containing validated align_proteins outputs.

    Returns:
        Three summary tables and the input provenance for their manifest.
    """
    paths = {
        name: alignment_dir / name for name in (
            'selected_cds.parquet', 'aligned_proteins.fasta', 'gap_locations.csv',
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
    if file_sha256(paths['selected_cds.parquet']) != inspection['selected_cds_sha256']:
        raise ValueError('Selected CDS changed after inspection.')

    selected = pd.read_parquet(paths['selected_cds.parquet'])
    gaps = pd.read_csv(paths['gap_locations.csv'])
    aligned = read_fasta(paths['aligned_proteins.fasta'])
    if (set(selected.prot_hash) != set(gaps.prot_hash)
            or set(gaps.prot_hash) != set(aligned)):
        raise ValueError('Protein IDs differ among selection, gap table and alignment.')
    widths = {len(sequence) for sequence in aligned.values()}
    if widths != {validation['aligned_width']}:
        raise ValueError('Alignment width differs from validation report.')
    if len(selected) != selection['counts']['complete_records']:
        raise ValueError('Selected record count differs from selection manifest.')

    group = (selected.groupby(['host', 'hn_subtype', 'year', 'cds_length'], sort=True)
             .agg(isolates=('assembly_id', 'nunique'),
                  unique_cds=('cds_dna_hash', 'nunique'),
                  unique_proteins=('prot_hash', 'nunique'))
             .reset_index())
    protein_isolates = selected.groupby('prot_hash').assembly_id.nunique()
    patterns = gaps.copy()
    patterns['gap_columns'] = patterns['gap_columns'].fillna('')
    patterns['isolates'] = patterns.prot_hash.map(protein_isolates)
    pattern = (patterns.groupby(['cds_length', 'gap_columns', 'leading_gaps',
                                 'internal_gaps', 'trailing_gaps'], sort=True)
               .agg(unique_proteins=('prot_hash', 'size'),
                    isolates=('isolates', 'sum'))
               .reset_index())

    occupancy = []
    for column in range(validation['aligned_width']):
        present = [protein_hash for protein_hash, sequence in aligned.items()
                   if sequence[column] != '-']
        occupancy.append({
            'column': column + 1,
            'proteins_with_residue': len(present),
            'isolates_with_residue': int(protein_isolates.loc[present].sum()),
        })
    columns = pd.DataFrame(occupancy)
    provenance = {
        'population': selection['population'],
        'protein': selection['protein'],
        'selected_records': len(selected),
        'unique_proteins': len(aligned),
        'aligned_width': validation['aligned_width'],
        'inputs': {name: file_sha256(path) for name, path in paths.items()},
    }
    return group, pattern, columns, provenance


def main() -> None:
    """Write summaries and a manifest to the alignment directory.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--alignment_dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    existing = [name for name in OUTPUT_NAMES if (args.alignment_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error(f'Outputs already exist: {existing}. Pass --overwrite to replace them.')
    group, pattern, columns, provenance = summarize(args.alignment_dir)
    for name, table in zip(OUTPUT_NAMES[:3], (group, pattern, columns)):
        table.to_csv(args.alignment_dir / name, index=False)
    manifest = {
        **provenance,
        'script': str(Path(__file__).resolve()),
        'script_sha256': file_sha256(Path(__file__)),
        'outputs': {name: file_sha256(args.alignment_dir / name)
                    for name in OUTPUT_NAMES[:3]},
        'git': get_git_info(),
    }
    (args.alignment_dir / OUTPUT_NAMES[3]).write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Summarized {len(group)} group/length rows, {len(pattern)} gap patterns, '
          f'and {len(columns)} alignment columns.')


if __name__ == '__main__':
    main()
