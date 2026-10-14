"""Unit tests for src/analysis/metadata_group_report.py.

The per-group tables are what Task 2 reports, so these cover the three things that would silently
corrupt them: a group assignment taken from a negative whose endpoints disagree, group counts that
do not add up to the split, and a metric computed on a subset that holds one class.

Run: python -m pytest tests/test_metadata_group_report.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.analysis.metadata_group_report import (
    GROUP_AXES,
    METRIC_COLUMNS,
    check_axes_match,
    count_one_split,
    fit_group_metrics,
    subset_metrics,
)

GROUPS = [('Human', 'H3N2', '2022'), ('Human', 'H1N1', '2023'), ('Pig', 'H3N2', '2022')]


def make_pairs(per_group=4, probabilities=None):
    """A pair table shaped like a saved split: per group, half positives and half negatives."""
    rows = []
    for host, subtype, year in GROUPS:
        for index in range(per_group):
            rows.append({'label': 1 if index < per_group // 2 else 0,
                         'host_a': host, 'host_b': host,
                         'hn_subtype_a': subtype, 'hn_subtype_b': subtype,
                         'year_a': year, 'year_b': year,
                         'assembly_id_a': f'{host}-{subtype}-{year}-{index}'})
    pairs = pd.DataFrame(rows)
    if probabilities is not None:
        pairs['pred_prob'] = probabilities
    return pairs


def test_counts_are_per_group_and_add_up():
    counts = count_one_split(make_pairs(per_group=4), 'test')
    assert len(counts) == len(GROUPS)
    assert list(counts.columns) == [*GROUP_AXES, 'n_pos', 'n_neg']
    assert (counts['n_pos'] == 2).all() and (counts['n_neg'] == 2).all()


def test_negative_with_disagreeing_endpoints_raises():
    pairs = make_pairs()
    negative = pairs.index[pairs['label'] == 0][0]
    pairs.loc[negative, 'year_b'] = '2023'
    with pytest.raises(ValueError, match='differ on year'):
        count_one_split(pairs, 'test')


def test_positive_with_disagreeing_endpoints_is_not_checked():
    # Positives are same-isolate pairs; only the negatives can cross two groups, and a saved
    # positive whose sides differ is a different defect, caught where positives are built.
    pairs = make_pairs()
    positive = pairs.index[pairs['label'] == 1][0]
    pairs.loc[positive, 'host_b'] = 'Pig'
    check_axes_match(pairs, 'test')


def test_missing_metadata_columns_raise():
    pairs = make_pairs().drop(columns=['year_b'])
    with pytest.raises(ValueError, match='year_b'):
        check_axes_match(pairs, 'test')


def test_single_class_subset_has_no_metrics():
    subset = pd.DataFrame({'label': [1, 1, 1], 'pred_prob': [0.9, 0.8, 0.7]})
    metrics = subset_metrics(subset, 0.5)
    assert metrics['n_pos'] == 3 and metrics['n_neg'] == 0
    assert metrics['tp'] == 3 and metrics['fn'] == 0
    assert all(np.isnan(metrics[name])
               for name in ('f1_macro', 'auc_roc', 'precision', 'recall'))


def test_perfect_and_inverted_predictions_score_as_expected():
    subset = pd.DataFrame({'label': [1, 1, 0, 0], 'pred_prob': [0.9, 0.8, 0.2, 0.1]})
    perfect = subset_metrics(subset, 0.5)
    assert perfect['tp'] == 2 and perfect['tn'] == 2
    assert perfect['f1_macro'] == pytest.approx(1.0)
    assert perfect['auc_roc'] == pytest.approx(1.0)

    inverted = subset.assign(pred_prob=1 - subset['pred_prob'])
    worst = subset_metrics(inverted, 0.5)
    assert worst['tp'] == 0 and worst['tn'] == 0
    assert worst['auc_roc'] == pytest.approx(0.0)


def write_fit(tmp_path, pairs, threshold=0.5):
    """A directory shaped like a finished fit, holding `pairs` as its test predictions."""
    pairs = pairs.copy()
    tmp_path.mkdir(parents=True, exist_ok=True)
    pairs['pred_label'] = (pairs['pred_prob'] > threshold).astype(int)
    pairs.to_csv(tmp_path / 'test_predicted.csv', index=False)
    (tmp_path / 'training_info.json').write_text(json.dumps({'optimal_threshold': threshold}))
    return tmp_path


def test_fit_metrics_cover_every_group_and_axis(tmp_path):
    # Predictions good enough to separate, so no metric is degenerate.
    pairs = make_pairs(per_group=4)
    pairs['pred_prob'] = np.where(pairs['label'] == 1, 0.9, 0.1)
    rows = fit_group_metrics(write_fit(tmp_path, pairs), 'PB2-PB1', '0', 'Human+Pig-H1N1+H3N2')
    table = pd.DataFrame(rows)[METRIC_COLUMNS]

    joint = table[table['group_by'] == 'joint']
    assert len(joint) == len(GROUPS)
    assert joint['n_pos'].sum() == int((pairs['label'] == 1).sum())
    assert joint['n_neg'].sum() == int((pairs['label'] == 0).sum())
    assert joint['f1_macro'].tolist() == pytest.approx([1.0] * len(GROUPS))

    hosts = table[table['group_by'] == 'host'].set_index('host')
    # Two of the three groups are Human, so its pooled row holds both.
    assert hosts.loc['Human', 'n_pos'] == 4
    assert hosts.loc['Pig', 'n_pos'] == 2
    assert (hosts['hn_subtype'] == '').all() and (hosts['year'] == '').all()
    assert set(table['group_by']) == {'joint', *GROUP_AXES}


def test_fit_metrics_use_the_fits_threshold(tmp_path):
    # Every probability sits between the two thresholds, so the threshold alone decides the labels.
    pairs = make_pairs(per_group=4)
    pairs['pred_prob'] = 0.6
    strict = fit_group_metrics(write_fit(tmp_path / 'strict', pairs.copy(), threshold=0.8),
                               'PB2-PB1', '0', 'p')
    loose = fit_group_metrics(write_fit(tmp_path / 'loose', pairs.copy(), threshold=0.4),
                              'PB2-PB1', '0', 'p')
    assert sum(row['tp'] for row in strict) == 0
    assert sum(row['fn'] for row in strict) > 0
    assert sum(row['fn'] for row in loose) == 0
    assert sum(row['tp'] for row in loose) > 0

