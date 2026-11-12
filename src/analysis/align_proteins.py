"""Prepare complete CDS proteins and align them with MAFFT.

The procedure follows docs/plans/2026-09-14_codon_site_features_plan.md.
FASTA headers use source prot_hash values, which hash prot_seq including its terminal stop.

Outputs under <out_dir>/<population>/<protein>/:
    selected_cds.parquet      complete records, retaining isolate and CDS mappings
    length_summary.csv        isolate, unique-CDS and unique-protein counts by CDS length
    proteins_for_alignment.fasta  unique proteins without terminal stop markers
    x_summary.csv           isolate, unique-CDS and unique-protein X counts by CDS length
    selection_manifest.json   input paths, filters, checks and counts
    aligned_proteins.fasta  MAFFT protein alignment, rows in guide-tree order
    mafft.log               MAFFT progress and selected strategy
    protein_input_hashes.csv  hash of each stop-free input protein
    protein_cds_map.csv     protein-to-CDS mapping with isolate counts
    alignment_manifest.json  tool version, command, strategy, runtime and file hashes
    alignment_validation.json  input recovery check for every aligned protein
    gap_locations.csv      gap positions and gap-pattern name for every aligned protein
    aligned_proteins_labelled.fasta  the alignment, labels starting with prot_hash
    aligned_proteins_jalview.fasta  the alignment, labels starting with the protein
    inspection_manifest.json  input and output hashes for the gap inspection

Example:
    python -m src.analysis.align_proteins --protein NS1 --host Human --hn_subtype H3N2 --year 2024
    python -m src.analysis.align_proteins --protein NS1 --host Human --hn_subtype H3N2 --year 2024 --align_only --mafft_bin /path/to/mafft
    python -m src.analysis.align_proteins --protein NS1 --host Human --hn_subtype H3N2 --year 2024 --validate_only
    python -m src.analysis.align_proteins --protein NS1 --host Human --hn_subtype H3N2 --year 2024 --inspect_only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.cds_utils import strip_terminal_stop, translate_dna  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402
from src.utils.experiment_utils import file_sha256, get_git_info  # noqa: E402
from src.utils.fasta_utils import read_fasta  # noqa: E402
from src.utils.metadata_enrichment import (  # noqa: E402
    attach_isolate_metadata,
    filter_by_metadata,
    population_label,
)
from src.utils.schema import sequence_md5  # noqa: E402
from src.utils.timer_utils import Timer  # noqa: E402

CDS_COLUMNS = [
    'assembly_id', 'function', 'canonical_segment', 'cds_dna_hash',
    'cds_dna_seq', 'cds_length', 'prot_hash', 'prot_seq', 'is_complete_cds',
]
OUTPUT_NAMES = (
    'selected_cds.parquet', 'length_summary.csv', 'proteins_for_alignment.fasta',
    'x_summary.csv', 'selection_manifest.json',
)
ALIGNMENT_OUTPUT_NAMES = (
    'aligned_proteins.fasta', 'mafft.log', 'protein_input_hashes.csv',
    'protein_cds_map.csv', 'alignment_manifest.json',
)
INSPECTION_OUTPUT_NAMES = (
    'gap_locations.csv', 'aligned_proteins_labelled.fasta', 'aligned_proteins_jalview.fasta',
    'inspection_manifest.json',
)


def summarize_selected_lengths(selected: pd.DataFrame) -> pd.DataFrame:
    """Count complete input records by length without collapsing synonymous CDS.

    Isolates sharing a CDS remain in the input. Each sequence count uses its
    own key: CDS DNA hashes for unique CDS and protein strings for unique proteins.
    Protein strings retain the terminal stop marker at this selection step.

    Args:
        selected: Complete CDS records, one per isolate.

    Returns:
        Counts of isolates, unique CDS and unique proteins at each CDS length.
    """
    summary = selected.groupby('cds_length', sort=True).agg(
        isolates=('assembly_id', 'nunique'),
        unique_cds=('cds_dna_hash', 'nunique'),
        unique_proteins=('prot_seq', 'nunique')
    )
    summary = summary.reset_index()
    return summary


def prepare_proteins(selected: pd.DataFrame) -> pd.DataFrame:
    """Validate each unique CDS and make protein sequences without terminal stops.

    Args:
        selected: Complete CDS records, including the original DNA and protein hashes.

    Returns:
        One row per protein hash, with its sequence for protein alignment.

    Raises:
        ValueError: A required field is absent or a CDS, protein, length or hash disagrees.
    """
    required = ['cds_dna_hash', 'cds_dna_seq', 'cds_length', 'prot_hash', 'prot_seq']
    if selected[required].isna().any().any():
        raise ValueError('Selected complete CDS contain a missing sequence, length or hash.')
    consistency = selected.groupby('cds_dna_hash')[required[1:]].nunique(dropna=False)
    inconsistent = consistency.index[consistency.ne(1).any(axis=1)]
    if len(inconsistent):
        raise ValueError(f'CDS hash {inconsistent[0]} has conflicting source records.')

    unique_cds = selected.drop_duplicates('cds_dna_hash')
    for row in unique_cds.itertuples(index=False):
        if len(row.cds_dna_seq) != row.cds_length:
            raise ValueError(f'CDS {row.cds_dna_hash}: stored length differs from DNA length.')
        if len(row.cds_dna_seq) != 3 * len(row.prot_seq):
            raise ValueError(f'CDS {row.cds_dna_hash}: DNA length is not 3 x protein length.')
        if not row.prot_seq.endswith('*') or row.prot_seq.count('*') != 1:
            raise ValueError(f'CDS {row.cds_dna_hash}: expected one terminal stop marker.')

        try:
            translated = translate_dna(row.cds_dna_seq)
        except ValueError as exc:
            raise ValueError(f'CDS {row.cds_dna_hash}: translation failed: {exc}') from exc

        if translated != row.prot_seq:
            raise ValueError(f'CDS {row.cds_dna_hash}: stored protein differs from translation.')
        expected_hash = sequence_md5(row.prot_seq)
        if row.prot_hash != expected_hash:
            raise ValueError(f'CDS {row.cds_dna_hash}: stored protein hash differs from sequence.')

    proteins = unique_cds[['prot_hash', 'prot_seq']].drop_duplicates('prot_hash').copy()
    if proteins['prot_hash'].nunique() != unique_cds['prot_seq'].nunique():
        raise ValueError('One protein hash names more than one protein sequence.')
    proteins['alignment_prot_seq'] = proteins['prot_seq'].map(strip_terminal_stop)
    proteins = proteins[['prot_hash', 'alignment_prot_seq']].sort_values('prot_hash')
    return proteins.reset_index(drop=True)


def summarize_unknown_residues(selected: pd.DataFrame) -> pd.DataFrame:
    """Count records containing X by CDS length.

    Args:
        selected: Validated complete CDS records, one per isolate.

    Returns:
        Isolate, unique-CDS and unique-protein counts containing X at each length.
    """
    with_x = selected.loc[selected['prot_seq'].str.contains('X', regex=False)]
    counts = (with_x.groupby('cds_length', sort=True)
              .agg(isolates_with_x=('assembly_id', 'nunique'),
                   unique_cds_with_x=('cds_dna_hash', 'nunique'),
                   unique_proteins_with_x=('prot_hash', 'nunique')))
    lengths = pd.Index(sorted(selected['cds_length'].unique()), name='cds_length')
    summary = counts.reindex(lengths, fill_value=0).reset_index()
    return summary


# MAFFT flags for each --mafft_strategy choice.
#   fft_ns_2  `--retree 2 --maxiterate 0`, which is what MAFFT runs with no options at all.
#             It gives FFT-NS-2 at any input size, so the strategy does not depend on how
#             many sequences a protein has.
#   auto      `--auto`, which lets MAFFT choose from the input size. A protein with more
#             sequences could get a different strategy.
# The manifest records the command and the strategy MAFFT reports, so the choice is
# checkable after the run.
MAFFT_STRATEGY_OPTIONS = {
    'auto': ['--auto'],
    'fft_ns_2': ['--retree', '2', '--maxiterate', '0'],
}


def align_prepared_proteins(out_dir: Path, mafft_bin: str, overwrite: bool,
                            strategy_name: str) -> None:
    """Align the prepared protein FASTA and record MAFFT provenance.

    Args:
        out_dir: Directory containing selection and protein-preparation outputs.
        mafft_bin: MAFFT executable path or command on PATH.
        overwrite: Whether existing alignment outputs may be replaced.
        strategy_name: Key of `MAFFT_STRATEGY_OPTIONS` giving the options to pass.

    Returns:
        None.

    Raises:
        FileNotFoundError: A required input or MAFFT executable is absent.
        ValueError: Prepared FASTA and selected CDS disagree or MAFFT reports no strategy.
        RuntimeError: MAFFT fails or produces no alignment.
    """
    selected_path = out_dir / 'selected_cds.parquet'
    fasta_path = out_dir / 'proteins_for_alignment.fasta'
    selection_manifest_path = out_dir / 'selection_manifest.json'

    for path in (selected_path, fasta_path, selection_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required preparation output is missing: {path}')
    existing = [name for name in ALIGNMENT_OUTPUT_NAMES if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Alignment outputs already exist in {out_dir}: {existing}; '
                              'pass --overwrite to replace them.')

    selected = pd.read_parquet(selected_path)
    selection_manifest = json.loads(selection_manifest_path.read_text())
    if selection_manifest.get('counts', {}).get('unique_complete_proteins') != selected['prot_hash'].nunique():
        raise ValueError('Selection manifest and selected CDS disagree on protein count.')
    expected = selected[['prot_hash', 'prot_seq']].drop_duplicates('prot_hash')
    if expected['prot_hash'].nunique() != selected['prot_seq'].nunique():
        raise ValueError('A protein hash names more than one selected protein sequence.')
    expected_sequences = dict(zip(expected['prot_hash'],
                                  expected['prot_seq'].map(strip_terminal_stop)))
    fasta_lines = fasta_path.read_text().splitlines()
    if len(fasta_lines) != 2 * len(expected_sequences):
        raise ValueError('Prepared FASTA does not have one two-line record per selected protein.')
    input_sequences = {}
    for header, sequence in zip(fasta_lines[0::2], fasta_lines[1::2]):
        if not header.startswith('>') or not sequence or header[1:] in input_sequences:
            raise ValueError('Prepared FASTA has an invalid or repeated protein record.')
        input_sequences[header[1:]] = sequence
    if input_sequences != expected_sequences:
        raise ValueError('Prepared FASTA sequences disagree with the selected CDS proteins.')

    resolved_bin = shutil.which(mafft_bin)
    if resolved_bin is None:
        raise FileNotFoundError(f'MAFFT executable not found: {mafft_bin}')
    resolved_bin = str(Path(resolved_bin).resolve())
    version_result = subprocess.run([resolved_bin, '--version'], capture_output=True, text=True,
                                    check=True)
    version = (version_result.stdout or version_result.stderr).strip()
    # --reorder writes rows in MAFFT's guide-tree order instead of input order. The MAFFT
    # wrapper applies it to the finished alignment, so it changes the row order and nothing
    # else; later steps look rows up by prot_hash.
    command = [resolved_bin, '--amino', *MAFFT_STRATEGY_OPTIONS[strategy_name],
               '--reorder', '--thread', '1', str(fasta_path.resolve())]
    # Timed here because MAFFT reports no timing of its own, and the file timestamps only
    # bound it, since selection and alignment run separately.
    mafft_timer = Timer()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    mafft_timer.stop_timer()
    if result.returncode != 0:
        raise RuntimeError(f'MAFFT failed with exit code {result.returncode}:\n{result.stderr}')
    if not result.stdout.startswith('>'):
        raise RuntimeError('MAFFT produced no FASTA alignment.')
    strategy_match = re.search(r'^\s*Strategy:\s*(.+?)\s*$', result.stderr, flags=re.MULTILINE)
    if strategy_match is None:
        raise ValueError('MAFFT log does not report the selected alignment strategy.')
    strategy = strategy_match.group(1)

    input_hashes = pd.DataFrame(
        [(protein_hash, hashlib.sha256(sequence.encode('utf-8')).hexdigest())
         for protein_hash, sequence in sorted(input_sequences.items())],
        columns=['prot_hash', 'alignment_seq_sha256'],
    )
    cds_map = (selected.groupby(['prot_hash', 'cds_dna_hash', 'cds_length'], sort=True)
               .agg(isolates=('assembly_id', 'nunique')).reset_index())
    aligned_path = out_dir / 'aligned_proteins.fasta'
    log_path = out_dir / 'mafft.log'
    aligned_path.write_text(result.stdout)
    log_path.write_text(result.stderr)
    input_hashes.to_csv(out_dir / 'protein_input_hashes.csv', index=False)
    cds_map.to_csv(out_dir / 'protein_cds_map.csv', index=False)

    manifest = {
        'stage': 'protein_alignment',
        'selection_manifest': str(selection_manifest_path.resolve()),
        'selected_cds_sha256': file_sha256(selected_path),
        'input_fasta': str(fasta_path.resolve()),
        'input_fasta_sha256': file_sha256(fasta_path),
        'input_sequence_hashes': 'protein_input_hashes.csv',
        'protein_cds_map': 'protein_cds_map.csv',
        'mafft_version': version,
        'mafft_command': shlex.join(command) + f' > {shlex.quote(str(aligned_path.resolve()))}',
        'mafft_strategy': strategy,
        'mafft_seconds': round(mafft_timer.elapsed, 1),
        'aligned_fasta_sha256': file_sha256(aligned_path),
        'mafft_log_sha256': file_sha256(log_path),
        'protein_count': len(input_sequences),
        'unique_cds_count': int(selected['cds_dna_hash'].nunique()),
        'code': get_git_info(),
        'outputs': list(ALIGNMENT_OUTPUT_NAMES),
    }
    alignment_manifest = out_dir / 'alignment_manifest.json'
    alignment_manifest.write_text(json.dumps(manifest, indent=2) + '\n')

    print(f'MAFFT {version}: aligned {len(input_sequences):,} proteins with {strategy}.')
    print(f'Wrote alignment outputs to {out_dir}')


def validate_alignment(out_dir: Path, overwrite: bool) -> None:
    """Check that every aligned protein recovers its exact stop-free input.

    Args:
        out_dir: Directory containing the prepared and aligned protein FASTA files.
        overwrite: Whether an existing validation report may be replaced.

    Returns:
        None.

    Raises:
        FileNotFoundError: An alignment input or output is missing.
        ValueError: File hashes, protein IDs, widths or recovered sequences disagree.
    """
    input_path = out_dir / 'proteins_for_alignment.fasta'
    aligned_path = out_dir / 'aligned_proteins.fasta'
    manifest_path = out_dir / 'alignment_manifest.json'
    report_path = out_dir / 'alignment_validation.json'

    for path in (input_path, aligned_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required alignment file is missing: {path}')
    if report_path.exists() and not overwrite:
        raise FileExistsError(f'Validation report already exists: {report_path}; '
                              'pass --overwrite to replace it.')

    manifest = json.loads(manifest_path.read_text())
    for path, hash_key in ((input_path, 'input_fasta_sha256'),
                           (aligned_path, 'aligned_fasta_sha256')):
        if file_sha256(path) != manifest[hash_key]:
            raise ValueError(f'{path} does not match the alignment manifest hash.')

    inputs = read_fasta(input_path)
    aligned = read_fasta(aligned_path)
    if set(inputs) != set(aligned):
        missing = sorted(set(inputs) - set(aligned))
        extra = sorted(set(aligned) - set(inputs))
        raise ValueError(f'Alignment protein IDs differ from inputs: missing={missing[:3]}, '
                         f'extra={extra[:3]}.')

    if len(inputs) != manifest['protein_count']:
        raise ValueError('Protein count differs from the alignment manifest.')

    widths = {len(sequence) for sequence in aligned.values()}
    if len(widths) != 1:
        raise ValueError(f'Aligned protein widths differ: {sorted(widths)}.')

    for protein_hash, sequence in aligned.items():
        if sequence.replace('-', '') != inputs[protein_hash]:
            raise ValueError(f'Aligned protein {protein_hash} does not recover its input.')

    report = {
        'stage': 'alignment_validation',
        'input_fasta': str(input_path.resolve()),
        'input_fasta_sha256': manifest['input_fasta_sha256'],
        'aligned_fasta': str(aligned_path.resolve()),
        'aligned_fasta_sha256': manifest['aligned_fasta_sha256'],
        'proteins_checked': len(inputs),
        'proteins_recovered': len(inputs),
        'aligned_width': next(iter(widths)),
        'all_passed': True,
        'code': get_git_info(),
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Validated {len(inputs):,} aligned proteins; every input sequence was recovered.')
    print(f'Wrote validation report to {report_path}')


def number_gap_patterns(gap_locations: pd.DataFrame, isolates: pd.Series) -> pd.Series:
    """Name each protein's gap pattern by rank, `gap_pat001` being held by the most proteins.

    Patterns are ranked by unique proteins, then by isolates, then by the
    `gap_columns` string, so patterns with equal counts still get a fixed order.
    The rank is zero-padded to three digits so the names sort as text in rank order.

    Args:
        gap_locations: One row per unique protein, with `prot_hash` and `gap_columns`.
        isolates: Isolate count per protein, indexed by `prot_hash`.

    Returns:
        The gap-pattern name of each row of `gap_locations`, indexed like it.
    """
    per_protein = gap_locations[['prot_hash', 'gap_columns']].copy()
    per_protein['isolates'] = per_protein['prot_hash'].map(isolates)
    if per_protein['isolates'].isna().any():
        raise ValueError('An aligned protein has no isolate count.')
    patterns = (per_protein.groupby('gap_columns')
                .agg(unique_proteins=('prot_hash', 'size'), isolates=('isolates', 'sum'))
                .reset_index()
                .sort_values(['unique_proteins', 'isolates', 'gap_columns'],
                             ascending=[False, False, True]))
    if len(patterns) > 999:
        raise ValueError(f'{len(patterns)} gap patterns do not fit a three-digit name.')
    patterns['gap_pattern'] = [f'gap_pat{rank:03d}' for rank in range(1, len(patterns) + 1)]
    names = per_protein['gap_columns'].map(patterns.set_index('gap_columns')['gap_pattern'])
    return names


def describe_proteins(proteins: pd.DataFrame) -> pd.Series:
    """Join each protein's gap pattern, CDS length, residue count and isolate count.

    Args:
        proteins: Rows with `gap_pattern`, `cds_length`, `amino_acids` and `isolates`.

    Returns:
        Strings such as `gap_pat001|2280nt|759aa|5046iso`, indexed like `proteins`.
    """
    fields = (proteins['gap_pattern']
              + '|' + proteins['cds_length'].astype(str) + 'nt'
              + '|' + proteins['amino_acids'].astype(str) + 'aa'
              + '|' + proteins['isolates'].astype(str) + 'iso')
    return fields


def build_hash_first_labels(proteins: pd.DataFrame, protein: str) -> pd.Series:
    """Build FASTA labels that start with `prot_hash`, so `label.split('|')[0]` recovers it.

    The labels hold no whitespace, so `read_fasta` keys each record by the whole label.

    Args:
        proteins: Rows with `prot_hash`, `gap_pattern`, `cds_length`, `amino_acids`
            and `isolates`.
        protein: Short protein name, e.g. PB2.

    Returns:
        Labels such as `<prot_hash>|PB2|gap_pat001|2280nt|759aa|5046iso`, indexed like
        `proteins`.
    """
    labels = proteins['prot_hash'] + '|' + protein + '|' + describe_proteins(proteins)
    return labels


def build_protein_first_labels(proteins: pd.DataFrame, protein: str) -> pd.Series:
    """Build FASTA labels that start with the protein, so sorting them groups the gap patterns.

    The labels hold no whitespace, so `read_fasta` keys each record by the whole label.

    Args:
        proteins: Rows with `prot_hash`, `gap_pattern`, `cds_length`, `amino_acids`
            and `isolates`.
        protein: Short protein name, e.g. PB2.

    Returns:
        Labels such as `PB2|gap_pat001|2280nt|759aa|5046iso|<prot_hash>`, indexed like
        `proteins`.
    """
    labels = protein + '|' + describe_proteins(proteins) + '|' + proteins['prot_hash']
    return labels


def inspect_alignment(out_dir: Path, protein: str, overwrite: bool) -> None:
    """Record gap locations for every validated aligned protein and write two labelled copies.

    Args:
        out_dir: Directory containing selected CDS and validated alignment outputs.
        protein: Short protein name, used in the labels.
        overwrite: Whether existing inspection outputs may be replaced.

    Returns:
        None.

    Raises:
        FileNotFoundError: A required selection, alignment or validation file is absent.
        ValueError: Input hashes or protein IDs disagree, or validation did not pass.
    """
    selected_path = out_dir / 'selected_cds.parquet'
    aligned_path = out_dir / 'aligned_proteins.fasta'
    validation_path = out_dir / 'alignment_validation.json'
    for path in (selected_path, aligned_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required inspection input is missing: {path}')
    existing = [name for name in INSPECTION_OUTPUT_NAMES if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Inspection outputs already exist in {out_dir}: {existing}; '
                              'pass --overwrite to replace them.')

    validation = json.loads(validation_path.read_text())
    if not validation.get('all_passed') or validation['aligned_fasta_sha256'] != file_sha256(aligned_path):
        raise ValueError('The alignment has no passing validation for its current FASTA.')
    selected = pd.read_parquet(selected_path, columns=['prot_hash', 'cds_length', 'assembly_id'])
    lengths_by_hash = selected.groupby('prot_hash')['cds_length'].nunique()
    if lengths_by_hash.ne(1).any():
        raise ValueError('A protein hash has more than one CDS length.')
    cds_lengths = selected.drop_duplicates('prot_hash').set_index('prot_hash')['cds_length']
    aligned = read_fasta(aligned_path)
    if set(aligned) != set(cds_lengths.index):
        raise ValueError('Aligned protein IDs disagree with selected CDS records.')

    rows = []
    for protein_hash, sequence in sorted(aligned.items()):
        leading = len(sequence) - len(sequence.lstrip('-'))
        trailing = len(sequence) - len(sequence.rstrip('-'))
        gap_count = sequence.count('-')
        gaps = ';'.join(f'{match.start() + 1}-{match.end()}'
                        for match in re.finditer(r'-+', sequence))
        unknowns = ';'.join(str(i + 1) for i, residue in enumerate(sequence) if residue == 'X')
        rows.append({
            'prot_hash': protein_hash,
            'cds_length': int(cds_lengths[protein_hash]),
            'alignment_width': len(sequence),
            'amino_acids': len(sequence) - gap_count,
            'gap_count': gap_count,
            'leading_gaps': leading,
            'internal_gaps': gap_count - leading - trailing,
            'trailing_gaps': trailing,
            'gap_columns': gaps,
            'x_columns': unknowns,
        })

    gap_locations = pd.DataFrame(rows)
    if gap_locations['alignment_width'].nunique() != 1:
        raise ValueError('Aligned proteins have different widths during inspection.')
    isolates = selected.groupby('prot_hash')['assembly_id'].nunique()
    gap_locations['gap_pattern'] = number_gap_patterns(gap_locations, isolates)
    gap_path = out_dir / 'gap_locations.csv'
    gap_locations.to_csv(gap_path, index=False)

    # Two copies of aligned_proteins.fasta, same rows in the same order, with readable
    # labels: one starting with prot_hash, one starting with the protein so that sorting
    # by label in a viewer groups the rows by gap pattern. aligned_proteins.fasta itself
    # stays keyed by prot_hash for the later steps.
    proteins = gap_locations.set_index('prot_hash', drop=False)
    proteins['isolates'] = proteins['prot_hash'].map(isolates)
    labelled_path = out_dir / 'aligned_proteins_labelled.fasta'
    jalview_path = out_dir / 'aligned_proteins_jalview.fasta'
    for path, labels in ((labelled_path, build_hash_first_labels(proteins, protein)),
                         (jalview_path, build_protein_first_labels(proteins, protein))):
        records = [f'>{labels[protein_hash]}\n{sequence}'
                   for protein_hash, sequence in aligned.items()]
        path.write_text('\n'.join(records) + '\n')

    manifest = {
        'stage': 'alignment_gap_inspection',
        'selected_cds_sha256': file_sha256(selected_path),
        'aligned_fasta_sha256': file_sha256(aligned_path),
        'validation_report_sha256': file_sha256(validation_path),
        'gap_locations_sha256': file_sha256(gap_path),
        'labelled_fasta_sha256': file_sha256(labelled_path),
        'jalview_fasta_sha256': file_sha256(jalview_path),
        'proteins_inspected': len(gap_locations),
        'alignment_width': int(gap_locations['alignment_width'].iloc[0]),
        'code': get_git_info(),
        'outputs': list(INSPECTION_OUTPUT_NAMES),
    }
    inspec_manifest = out_dir / 'inspection_manifest.json'
    inspec_manifest.write_text(json.dumps(manifest, indent=2) + '\n')

    print(f'Inspected gap locations for {len(gap_locations):,} aligned proteins.')
    print(f'Wrote inspection outputs to {out_dir}')


def main() -> None:
    """Prepare CDS inputs, align proteins, validate them or inspect their gaps.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--protein', required=True, help='protein short name, e.g. NS1')
    parser.add_argument('--cds_final', type=Path,
        default=PROJ / 'data/processed/flu/July_2025/cds_dna_final.parquet')
    parser.add_argument('--virus_config', type=Path, default=PROJ / 'conf/virus/flu.yaml')
    parser.add_argument('--metadata_file', type=Path,
        default=PROJ / 'data/processed/flu/metadata_eda/flu_genomes_metadata_parsed.csv')
    parser.add_argument('--host', nargs='+', default=None)
    parser.add_argument('--hn_subtype', nargs='+', default=None)
    years = parser.add_mutually_exclusive_group()
    years.add_argument('--year', nargs='+', type=int, default=None)
    years.add_argument('--year_range', nargs=2, type=int, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--out_dir', type=Path,
        default=PROJ / 'results/flu/July_2025/cds_alignment_pilot')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--align_only', action='store_true',
        help='run MAFFT on existing preparation outputs without selecting CDS again')
    actions.add_argument('--validate_only', action='store_true',
        help='validate an existing alignment without rerunning MAFFT')
    actions.add_argument('--inspect_only', action='store_true',
        help='record gap positions in an existing validated alignment')
    parser.add_argument('--mafft_bin', default='mafft',
        help='MAFFT executable path or command on PATH (with --align_only)')
    parser.add_argument(
        '--mafft_strategy', choices=sorted(MAFFT_STRATEGY_OPTIONS),
        default='fft_ns_2',
        help="which MAFFT options to pass (with --align_only). 'fft_ns_2' passes "
             "--retree 2 --maxiterate 0, which give FFT-NS-2 at any input size, "
             "so every protein gets the same strategy. 'auto' passes --auto and "
             "lets MAFFT choose the strategy from the input size")
    parser.add_argument('--overwrite', action='store_true',
        help='replace existing outputs for this population and protein')
    args = parser.parse_args()

    names = load_function_metadata(args.virus_config)
    if args.protein not in names.short_to_function:
        parser.error(f'Unknown protein {args.protein!r}; choose from {sorted(names.short_to_function)}')
    label = population_label(
        host=args.host,
        hn_subtype=args.hn_subtype,
        year=args.year,
        year_range=args.year_range)
    out_dir = args.out_dir / label / args.protein

    if args.align_only:
        align_prepared_proteins(out_dir, args.mafft_bin, args.overwrite, args.mafft_strategy)
        return

    if args.validate_only:
        validate_alignment(out_dir, args.overwrite)
        return

    if args.inspect_only:
        inspect_alignment(out_dir, args.protein, args.overwrite)
        return
    
    function = names.short_to_function[args.protein]
    cds = pd.read_parquet(args.cds_final, columns=CDS_COLUMNS,
                          filters=[('function', '==', function)])

    population = filter_by_metadata(
        attach_isolate_metadata(cds, metadata_file=args.metadata_file),
        host=args.host, hn_subtype=args.hn_subtype,
        year=args.year, year_range=args.year_range)
    if population.empty:
        raise ValueError('No protein records match the requested population.')

    selected = population.loc[population['is_complete_cds'].eq(True)].copy()
    if selected.empty:
        raise ValueError('The requested protein and population have no complete CDS.')
    selected = selected.sort_values(['cds_length', 'cds_dna_hash', 'assembly_id']).reset_index(drop=True)

    summary = summarize_selected_lengths(selected)
    proteins = prepare_proteins(selected)
    x_summary = summarize_unknown_residues(selected)
    existing = [name for name in OUTPUT_NAMES if (out_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f'Output files already exist in {out_dir}: {existing}; '
                              'pass --overwrite to replace them.')

    manifest = {
        'stage': 'protein_alignment_preparation',
        'protein': args.protein,
        'function': function,
        'population': label,
        'inputs': {
            'cds_final': str(args.cds_final.resolve()),
            'metadata_file': str(args.metadata_file.resolve()),
            'virus_config': str(args.virus_config.resolve()),
        },
        'filters': {
            'host': args.host, 'hn_subtype': args.hn_subtype,
            'year': args.year, 'year_range': args.year_range,
            'is_complete_cds': True,
        },
        'counts': {
            'population_records': len(population),
            'population_isolates': int(population['assembly_id'].nunique()),
            'complete_records': len(selected),
            'complete_isolates': int(selected['assembly_id'].nunique()),
            'excluded_incomplete_records': int(population['is_complete_cds'].eq(False).sum()),
            'unknown_completeness_records': int(population['is_complete_cds'].isna().sum()),
            'unique_complete_cds': int(selected['cds_dna_hash'].nunique()),
            'unique_complete_proteins': int(selected['prot_seq'].nunique()),
            'unique_proteins_for_alignment': len(proteins),
            'isolates_with_x': int(x_summary['isolates_with_x'].sum()),
            'unique_cds_with_x': int(x_summary['unique_cds_with_x'].sum()),
            'unique_proteins_with_x': int(x_summary['unique_proteins_with_x'].sum()),
        },
        'checks': {
            'unique_cds_checked': int(selected['cds_dna_hash'].nunique()),
            'translation_and_terminal_stop_passed': True,
            'dna_length_and_protein_hash_passed': True,
        },
        'code': get_git_info(),
        'outputs': list(OUTPUT_NAMES),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out_dir / 'selected_cds.parquet', index=False)
    summary.to_csv(out_dir / 'length_summary.csv', index=False)
    fasta = ''.join(f'>{row.prot_hash}\n{row.alignment_prot_seq}\n'
                    for row in proteins.itertuples(index=False))
    (out_dir / 'proteins_for_alignment.fasta').write_text(fasta)
    x_summary.to_csv(out_dir / 'x_summary.csv', index=False)
    (out_dir / 'selection_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    print(f'{args.protein}, {label}: {manifest["counts"]["population_isolates"]:,} isolates before selection')
    print(f'{manifest["counts"]["complete_isolates"]:,} isolates with complete CDS; '
          f'{manifest["counts"]["unique_complete_cds"]:,} unique CDS; '
          f'{manifest["counts"]["unique_complete_proteins"]:,} unique proteins')
    print(summary.to_string(index=False))
    print(x_summary.to_string(index=False))
    print(f'Wrote selection and protein-preparation outputs to {out_dir}')


if __name__ == '__main__':
    main()
