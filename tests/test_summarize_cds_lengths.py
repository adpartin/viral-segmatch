"""Tests for `src/analysis/summarize_cds_lengths.py` and `cds_utils.modal_length`.

Covers:
  1. modal_length reports the most common length, its count and its share
  2. A tie takes the shorter length and is flagged, so the answer cannot move between runs
  3. modal_length rejects an empty population
  4. segment_number maps `S1`..`S8`, and rejects anything else
  5. summarize_cds_lengths counts each distinct sequence once, so repeated rows cannot move the
     mode, the median or any count
  6. Length statistics use complete sequences only
  7. `frac isolates at mode` divides by every isolate, so an incomplete-heavy gene falls far
     below `frac at mode`, which divides by the complete sequences and cannot see the problem
  8. population_label describes the filters, says 'all', and rejects geo/passage
  9. Missing columns and a protein with no complete sequence both raise

Run: python tests/test_summarize_cds_lengths.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis.summarize_cds_lengths import (  # noqa: E402
    COLUMNS,
    segment_number,
    summarize_cds_lengths,
)
from src.utils.cds_utils import modal_length  # noqa: E402
from src.utils.metadata_enrichment import population_label  # noqa: E402

SHORT = {'Hemagglutinin precursor': 'HA', 'Neuraminidase protein': 'NA'}


def _row(function, segment, hash_, length, complete=True, assembly_id=None):
    return {'assembly_id': assembly_id or f'iso_{hash_}', 'function': function,
            'canonical_segment': segment, 'cds_dna_hash': hash_,
            'cds_length': length, 'is_complete_cds': complete}


def test_modal_length_reports_count_and_share():
    result = modal_length([9, 9, 9, 6, 12])
    assert result['mode'] == 9
    assert result['n_at_mode'] == 3
    assert result['frac_at_mode'] == pytest.approx(0.6)
    assert result['n'] == 5
    assert result['tied'] is False


def test_modal_length_breaks_a_tie_toward_the_shorter_length():
    # Both 6 and 9 appear twice. pandas would settle this by encounter order, which can move
    # between runs, so the rule is the shorter length and the tie is reported.
    result = modal_length([9, 9, 6, 6])
    assert result['mode'] == 6
    assert result['tied'] is True
    assert result['n_at_mode'] == 2
    assert modal_length([6, 6, 9, 9])['mode'] == 6      # order must not matter


def test_modal_length_rejects_an_empty_population():
    with pytest.raises(ValueError, match='no sequences given'):
        modal_length([])


def test_segment_number():
    assert segment_number('S1') == 1
    assert segment_number('S8') == 8
    for bad in ('HA', '4', 'SX', ''):
        with pytest.raises(ValueError, match='canonical_segment'):
            segment_number(bad)


def test_each_distinct_sequence_counts_once():
    # 'h1' appears three times, which is what a heavily sampled strain looks like. If rows were
    # counted, 900 would become the mode and the median would follow it.
    cds = pd.DataFrame([
        _row('Hemagglutinin precursor', 'S4', 'h1', 900, assembly_id='i1'),
        _row('Hemagglutinin precursor', 'S4', 'h1', 900, assembly_id='i2'),
        _row('Hemagglutinin precursor', 'S4', 'h1', 900, assembly_id='i3'),
        _row('Hemagglutinin precursor', 'S4', 'h2', 903, assembly_id='i4'),
        _row('Hemagglutinin precursor', 'S4', 'h3', 903, assembly_id='i5'),
    ])
    table = summarize_cds_lengths(cds, SHORT, population='test')
    assert list(table.columns) == COLUMNS
    row = table.iloc[0]
    assert row['unique CDS'] == 3 and row['complete CDS'] == 3
    assert row['mode'] == 903 and row['complete CDS at mode'] == 2
    assert row['median'] == 903
    assert row['frac at mode'] == pytest.approx(2 / 3)
    assert row['Segment ID'] == 4 and row['protein'] == 'HA'
    assert row['population'] == 'test'
    # 5 isolates; only i4 and i5 are at the modal 903, so the isolate share is 2/5 while the
    # sequence share is 2/3. The two columns are answering different questions.
    assert row['isolates'] == 5
    assert row['frac isolates at mode'] == pytest.approx(2 / 5)


def test_length_statistics_use_complete_sequences_only():
    # The incomplete 300 is far shorter than either complete sequence, so it would take over min
    # and pull the median down if completeness were ignored.
    cds = pd.DataFrame([
        _row('Neuraminidase protein', 'S6', 'n1', 300, complete=False),
        _row('Neuraminidase protein', 'S6', 'n2', 900),
        _row('Neuraminidase protein', 'S6', 'n3', 906),
    ])
    row = summarize_cds_lengths(cds, SHORT).iloc[0]
    assert row['unique CDS'] == 3 and row['complete CDS'] == 2
    assert row['min'] == 900 and row['max'] == 906 and row['median'] == 903


def test_isolate_coverage_sees_incompleteness_that_frac_at_mode_cannot():
    # The PB1 case. `frac at mode` divides by the complete sequences, so it reads a perfect 1.0
    # while 80% of isolates carry an unusable record. Only the isolate columns show that.
    rows = [_row('Neuraminidase protein', 'S6', f'n{i}', 900, assembly_id=f'ok{i}')
            for i in range(4)]
    rows += [_row('Neuraminidase protein', 'S6', 'bad', 897, complete=False,
                  assembly_id=f'no{i}') for i in range(16)]
    row = summarize_cds_lengths(pd.DataFrame(rows), SHORT).iloc[0]
    assert row['frac at mode'] == pytest.approx(1.0)
    assert row['isolates'] == 20
    assert row['frac isolates complete'] == pytest.approx(4 / 20)
    assert row['frac isolates at mode'] == pytest.approx(4 / 20)


def test_the_two_isolate_columns_separate_incompleteness_from_wrong_length():
    # All 6 isolates have a complete CDS, but 2 are off the modal length. `frac isolates complete`
    # stays at 1.0 while `frac isolates at mode` drops, which is the other way a gene fails.
    rows = [_row('Neuraminidase protein', 'S6', f'n{i}', 900, assembly_id=f'm{i}')
            for i in range(4)]
    rows += [_row('Neuraminidase protein', 'S6', f'x{i}', 897, assembly_id=f'off{i}')
             for i in range(2)]
    row = summarize_cds_lengths(pd.DataFrame(rows), SHORT).iloc[0]
    assert row['frac isolates complete'] == pytest.approx(1.0)
    assert row['frac isolates at mode'] == pytest.approx(4 / 6)


def test_population_label():
    assert population_label() == 'all'
    assert population_label(host=['Human'], hn_subtype=['H3N2'],
                           year=[2024]) == 'Human-H3N2-2024'

    # A scalar and a one-element list mean the same filter, so they must label the same.
    assert population_label(host='Human', hn_subtype='H3N2', year=2024) == 'Human-H3N2-2024'

    # The range keeps its own '-' readable behind the 'yr' prefix.
    assert population_label(hn_subtype=['H3N2'], year_range=[2021, 2025]) == 'H3N2-yr2021-2025'

    # Set values are sorted, so the label does not depend on the order a filter was written in.
    assert (population_label(host=['Swine', 'Human'], hn_subtype=['H3N2', 'H1N1'])
            == population_label(host=['Human', 'Swine'], hn_subtype=['H1N1', 'H3N2'])
            == 'Human+Swine-H1N1+H3N2')

    # OmegaConf's ListConfig is not a list subclass, so a Hydra value must not be read as a
    # scalar and stringified.
    config = OmegaConf.create({'host': ['Human'], 'hn_subtype': ['H3N2'], 'year': [2024]})
    assert population_label(host=config.host, hn_subtype=config.hn_subtype,
                            year=config.year) == 'Human-H3N2-2024'

def test_population_label_rejects_the_axes_it_cannot_encode():
    # geo_location and passage values carry '-' and ' ', so no label can name them. Omitting
    # them silently would give two different populations the same persisted key.
    with pytest.raises(ValueError, match='geo_location'):
        population_label(host=['Human'], geo_location=['Baden-Wurttemberg'])
    with pytest.raises(ValueError, match='passage'):
        population_label(host=['Human'], passage=['Original'])

    # Both unset is the only case every current caller hits, and it must be untouched.
    assert population_label(host=['Human'], hn_subtype=['H3N2'], year=[2024],
                            geo_location=None, passage=None) == 'Human-H3N2-2024'


def test_rejects_missing_columns_and_a_protein_with_no_complete_sequence():
    with pytest.raises(ValueError, match='missing columns'):
        summarize_cds_lengths(pd.DataFrame({'function': ['HA']}), SHORT)

    none_complete = pd.DataFrame([_row('Neuraminidase protein', 'S6', 'n1', 900, complete=False)])
    with pytest.raises(ValueError, match='no complete CDS'):
        summarize_cds_lengths(none_complete, SHORT)


if __name__ == '__main__':
    tests = [
        test_modal_length_reports_count_and_share,
        test_modal_length_breaks_a_tie_toward_the_shorter_length,
        test_modal_length_rejects_an_empty_population,
        test_segment_number,
        test_each_distinct_sequence_counts_once,
        test_length_statistics_use_complete_sequences_only,
        test_isolate_coverage_sees_incompleteness_that_frac_at_mode_cannot,
        test_the_two_isolate_columns_separate_incompleteness_from_wrong_length,
        test_population_label,
        test_population_label_rejects_the_axes_it_cannot_encode,
        test_rejects_missing_columns_and_a_protein_with_no_complete_sequence,
    ]
    failed = 0
    for t in tests:
        try:
            print(f'... {t.__name__}')
            t()
            print('    OK')
        except Exception as e:
            failed += 1
            print(f'    FAIL: {e}')
    if failed:
        print(f'\n{failed} test(s) failed')
        sys.exit(1)
    print(f'\nAll {len(tests)} tests passed.')
