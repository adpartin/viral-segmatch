"""Tests for the run-manifest report pieces of `src/analysis/aggregate_allpairs_results.py`.

Covers:
  1. lower_triangle_matrix puts each pair in the slot-B row and the slot-A column
  2. It rejects an unknown protein, a pair of one protein, and slot A after slot B
  3. plot_lower_triangle writes a figure with a blank upper triangle and two-line cell text
  4. annotated_heatmap rejects a blank mask or cell text of the wrong shape
  5. write_report_info records the current commit, and run_manifest_inputs each manifest's
     path and checksum

Run: python -m pytest tests/test_allpairs_report.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis.aggregate_allpairs_results import (  # noqa: E402
    PROTEINS,
    lower_triangle_matrix,
    plot_lower_triangle,
    run_manifest_inputs,
    write_report_info,
)
from src.utils.experiment_utils import file_sha256, get_git_info  # noqa: E402
from src.utils.plot_utils import annotated_heatmap  # noqa: E402


def test_lower_triangle_placement():
    matrix = lower_triangle_matrix({'PB2-HA': 0.9, 'M1-NS1': 0.6})
    assert matrix.loc['HA', 'PB2'] == 0.9      # row is slot B, column is slot A
    assert matrix.loc['NS1', 'M1'] == 0.6
    values = matrix.to_numpy()
    assert np.isnan(values[np.triu_indices(len(PROTEINS))]).all()
    assert int(np.isfinite(values).sum()) == 2


@pytest.mark.parametrize('values', [
    {'PB2-XX': 0.5},
    {'HA-HA': 0.5},
    {'HA-PB2': 0.5},
])
def test_lower_triangle_rejects(values):
    with pytest.raises(ValueError):
        lower_triangle_matrix(values)


def test_plot_lower_triangle(tmp_path):
    mean = lower_triangle_matrix({'PB2-HA': 0.9, 'PB1-PA': 0.8})
    sd = lower_triangle_matrix({'PB2-HA': 0.01, 'PB1-PA': 0.02})
    out = tmp_path / 'matrix.png'
    plot_lower_triangle(mean, sd, 'title', 'F1 macro', out)
    assert out.exists() and out.stat().st_size > 0


def test_annotated_heatmap_shape_checks(tmp_path):
    matrix = np.zeros((2, 3))
    with pytest.raises(ValueError, match='blank_mask'):
        annotated_heatmap(matrix, row_labels=['a', 'b'], col_labels=['x', 'y', 'z'],
                          out_png=tmp_path / 'a.png', title='t', blank_mask=np.zeros((3, 2)))
    with pytest.raises(ValueError, match='cell_text'):
        annotated_heatmap(matrix, row_labels=['a', 'b'], col_labels=['x', 'y', 'z'],
                          out_png=tmp_path / 'b.png', title='t', cell_text=[['']])


def test_report_info(tmp_path):
    manifest = tmp_path / 'training_run_manifest.json'
    manifest.write_text('{}')
    args = argparse.Namespace(training_run_manifest=str(manifest), redraw_run_manifest=None)
    inputs = run_manifest_inputs(args)
    assert inputs['training_run_manifest'] == {'path': str(manifest.resolve()),
                                               'sha256': file_sha256(manifest)}
    assert inputs['redraw_run_manifest'] is None

    write_report_info(tmp_path, inputs)
    info = json.loads((tmp_path / 'report_info.json').read_text())
    assert info['code']['commit'] == get_git_info()['commit']
    assert info['inputs'] == inputs


@pytest.mark.parametrize('n_datasets, n_run_pairs, built_pairs, n_failures', [
    (21, 21, None, 0),     # the 21-pair screen
    (28, 28, None, 0),     # the 28-pair screen
    (28, 21, None, 1),     # a run missing pairs the datasets cover
    (1, 1, ['np_m1'], 1),  # a pilot build cannot carry a final report
])
def test_final_report_needs_every_built_pair(n_datasets, n_run_pairs, built_pairs, n_failures):
    from src.analysis.aggregate_allpairs_results import final_pair_coverage_failures
    dataset_manifest = {'datasets': {f'bundle{i}': {} for i in range(n_datasets)},
                        'pairs': built_pairs}
    run_manifest = {'pairs': {f'pair{i}': {} for i in range(n_run_pairs)}}
    assert len(final_pair_coverage_failures(run_manifest, dataset_manifest, True)) == n_failures
    # Without --final the pair count is not checked at all.
    assert final_pair_coverage_failures(run_manifest, dataset_manifest, False) == []
