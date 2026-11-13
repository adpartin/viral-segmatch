"""Unit tests for the gap-pattern selection in src/analysis/align_proteins.py.

The selection decides which CDS records are realigned, so these cover what would silently change
that set: a pattern exactly at the threshold, a protein with no gap pattern, a parent file
rewritten after it was recorded, a selection that drops nothing or keeps nothing, and a CDS
selection changed after its alignment was made.

The parent directory is built from real CDS records and a hand-written alignment, then
validated and inspected by the script's own functions, so MAFFT never runs.

Run: python -m pytest tests/test_align_proteins_gap_filter.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis.align_proteins import (
    OUTPUT_NAMES,
    check_inspected_alignment,
    inspect_alignment,
    keep_gap_patterns_above_share,
    select_kept_gap_patterns,
    validate_alignment,
)
from src.utils.cds_utils import compute_cds_dna_hash
from src.utils.experiment_utils import file_sha256
from src.utils.schema import sequence_md5

# Protein MLK* has 3 synonymous CDS and 4 isolates; protein MLR* has 1 CDS and 1 isolate.
# Of the 4 unique CDS, the MLK pattern holds 0.75 and the MLR pattern 0.25.
RECORDS = [
    ('a1', 'ATGCTTAAATAA', 'MLK*'),
    ('a2', 'ATGCTTAAATAA', 'MLK*'),
    ('a3', 'ATGCTCAAATAA', 'MLK*'),
    ('a4', 'ATGCTAAAATAA', 'MLK*'),
    ('b1', 'ATGCTTCGTTAA', 'MLR*'),
]
# The aligned rows put the two proteins' gaps in different columns: 4 for MLK, 3 for MLR.
ALIGNED = {'MLK*': 'MLK-', 'MLR*': 'ML-R'}


def make_parent(directory: Path) -> Path:
    """Write an inspected alignment directory shaped like a `--length_share_above` run."""
    directory.mkdir()
    selected = pd.DataFrame([{
        'assembly_id': isolate, 'function': 'Polymerase PB2', 'canonical_segment': '1',
        'cds_dna_hash': compute_cds_dna_hash(cds), 'cds_dna_seq': cds, 'cds_length': len(cds),
        'prot_hash': sequence_md5(protein), 'prot_seq': protein, 'is_complete_cds': True,
        'host': 'Human', 'hn_subtype': 'H3N2', 'year': 2024,
    } for isolate, cds, protein in RECORDS])
    selected.to_parquet(directory / 'selected_cds.parquet', index=False)

    proteins = sorted({protein for _, _, protein in RECORDS})
    inputs = ''.join(f'>{sequence_md5(p)}\n{p.rstrip("*")}\n' for p in proteins)
    aligned = ''.join(f'>{sequence_md5(p)}\n{ALIGNED[p]}\n' for p in proteins)
    (directory / 'proteins_for_alignment.fasta').write_text(inputs)
    (directory / 'aligned_proteins.fasta').write_text(aligned)
    alignment_manifest = {
        'input_fasta_sha256': file_sha256(directory / 'proteins_for_alignment.fasta'),
        'aligned_fasta_sha256': file_sha256(directory / 'aligned_proteins.fasta'),
        'selected_cds_sha256': file_sha256(directory / 'selected_cds.parquet'),
        'protein_count': len(proteins),
    }
    (directory / 'alignment_manifest.json').write_text(json.dumps(alignment_manifest))
    selection_manifest = {
        'protein': 'PB2', 'function': 'Polymerase PB2', 'population': 'Human-H3N2-2024',
        'inputs': {'cds_final': 'cds_dna_final.parquet'},
        'filters': {'host': ['Human'], 'length_share_above': 0.02},
        'length_filter': {'share_above': 0.02, 'kept_lengths': [12], 'dropped_lengths': [9]},
        'counts': {'population_records': 7, 'population_isolates': 7,
                   'excluded_incomplete_records': 1, 'unknown_completeness_records': 0},
    }
    (directory / 'selection_manifest.json').write_text(json.dumps(selection_manifest))

    validate_alignment(directory, overwrite=False)
    inspect_alignment(directory, 'PB2', overwrite=False)
    return directory


def read_tables(directory: Path):
    """The three tables the gap-pattern selection reads from a parent directory."""
    selected = pd.read_parquet(directory / 'selected_cds.parquet')
    gap_locations = pd.read_csv(directory / 'gap_locations.csv', keep_default_na=False)
    shares = pd.read_csv(directory / 'gap_pattern_shares.csv', keep_default_na=False)
    return selected, gap_locations, shares


def test_parent_inspection_names_the_two_patterns(tmp_path):
    shares = read_tables(make_parent(tmp_path / 'parent'))[2].set_index('gap_pattern')
    assert shares.loc['gap_pat001', 'gap_columns'] == '4-4'
    assert shares.loc['gap_pat001', 'unique_cds_share'] == 0.75
    assert shares.loc['gap_pat002', 'gap_columns'] == '3-3'
    assert shares.loc['gap_pat002', 'unique_cds_share'] == 0.25


def test_pattern_exactly_at_the_threshold_is_dropped(tmp_path):
    parent = make_parent(tmp_path / 'parent')
    kept, patterns = keep_gap_patterns_above_share(*read_tables(parent), share_above=0.25)
    assert patterns.set_index('gap_pattern')['kept'].to_dict() == {'gap_pat001': True, 'gap_pat002': False}
    assert sorted(kept['assembly_id']) == ['a1', 'a2', 'a3', 'a4']


def test_nothing_dropped_raises(tmp_path):
    parent = make_parent(tmp_path / 'parent')
    with pytest.raises(ValueError, match='nothing to realign'):
        keep_gap_patterns_above_share(*read_tables(parent), share_above=0.2)


def test_nothing_kept_raises(tmp_path):
    parent = make_parent(tmp_path / 'parent')
    with pytest.raises(ValueError, match='no record would be kept'):
        keep_gap_patterns_above_share(*read_tables(parent), share_above=0.75)


def test_protein_without_gap_pattern_raises(tmp_path):
    selected, gap_locations, shares = read_tables(make_parent(tmp_path / 'parent'))
    without_mlr = gap_locations.loc[gap_locations['prot_hash'] != sequence_md5('MLR*')]
    with pytest.raises(ValueError, match='have no gap pattern'):
        keep_gap_patterns_above_share(selected, without_mlr, shares, share_above=0.25)


def test_rewritten_parent_file_fails_the_check(tmp_path):
    parent = make_parent(tmp_path / 'parent')
    shares_path = parent / 'gap_pattern_shares.csv'
    shares_path.write_text(shares_path.read_text().replace('0.25', '0.2'))
    with pytest.raises(ValueError, match='gap_pattern_shares.csv does not match'):
        check_inspected_alignment(parent)


def test_inspection_rejects_a_selection_changed_after_alignment(tmp_path):
    parent = make_parent(tmp_path / 'parent')
    selected = pd.read_parquet(parent / 'selected_cds.parquet')
    selected.loc[0, 'assembly_id'] = 'a9'
    selected.to_parquet(parent / 'selected_cds.parquet', index=False)
    with pytest.raises(ValueError, match='changed after the alignment was made'):
        inspect_alignment(parent, 'PB2', overwrite=True)


def test_selection_writes_kept_records_and_parent_provenance(tmp_path):
    parent = make_parent(tmp_path / 'parent')
    out_dir = parent / 'gap_pattern_share_above_0.25'
    select_kept_gap_patterns(parent, out_dir, share_above=0.25, overwrite=False)

    assert all((out_dir / name).is_file() for name in OUTPUT_NAMES)
    manifest = json.loads((out_dir / 'selection_manifest.json').read_text())
    counts = manifest['counts']
    assert (counts['complete_records'], counts['complete_isolates']) == (4, 4)
    assert (counts['unique_complete_cds'], counts['unique_complete_proteins']) == (3, 1)
    # The population counts and the length decision are the parent's, unchanged.
    assert (counts['population_records'], counts['excluded_incomplete_records']) == (7, 1)
    assert manifest['length_filter']['kept_lengths'] == [12]

    gap_filter = manifest['gap_pattern_filter']
    assert gap_filter['parent_selected_cds_sha256'] == file_sha256(parent / 'selected_cds.parquet')
    assert [row['gap_pattern'] for row in gap_filter['dropped_patterns']] == ['gap_pat002']
    assert gap_filter['dropped'] == {'records': 1, 'isolates': 1, 'unique_cds': 1, 'unique_proteins': 1}
    assert gap_filter['parent_selection']['records'] == 5


def test_selection_that_drops_nothing_writes_nothing(tmp_path):
    parent = make_parent(tmp_path / 'parent')
    out_dir = parent / 'gap_pattern_share_above_0.2'
    with pytest.raises(ValueError, match='nothing to realign'):
        select_kept_gap_patterns(parent, out_dir, share_above=0.2, overwrite=False)
    assert not out_dir.exists()
