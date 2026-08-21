"""k-mer UMAP of one CV fold, colored by split membership or by isolate metadata.

ONE figure per invocation. The fold directory is the input; `--unit`, `--slot` and `--color_by`
select which figure, so comparing two arms (or two thresholds) is two runs of the same command
with a different `--fold_dir`, not a multi-panel script.

Units:
  slot -- one point per unique POSITIVE sequence of one slot; the k-mer vector training consumes.
  pair -- one point per positive pair, its two slot vectors concatenated, mirroring the model's
          `interaction: concat`. Answers whether the PAIRING is novel, which the slot views cannot:
          a test pair can sit in familiar slot-a and slot-b regions and still be an unseen combination.

Colorings:
  split                -- where the point's key appears: 'train/val only', 'test only', or 'both'.
  hn_subtype/host/year -- modal isolate metadata, top categories colored and the tail folded gray.

Comparability across figures is not arranged, it follows from the data: a 2D-CD dataset and its
random arm hold the SAME rows (`build_random_arm` re-cuts each fold's own rows at that fold's own
sizes), so both runs embed an identical matrix and, under a fixed UMAP seed, land on identical
coordinates. Only the coloring differs. `category_colors` pins each split label to one color so it
cannot move with the counts between the two figures.

Positives only by default: negatives are drawn per split by the builder, so including them shows
the negative sampler's geometry rather than the split's. `--include_negatives` asks for exactly
that geometry -- split keeps the color channel it has in every other figure and positive/negative
takes point shape, so the two read together and the colors still mean what they do elsewhere. It
needs `--unit pair`: one sequence carries both positive and negative rows, so a slot point has no
single label.

CLI:
    python -m src.analysis.plot_fold_umap \\
        --fold_dir data/datasets/flu/July_2025/runs/dataset_cc_nt_cds_cm0_t097_random/fold_0 \\
        --unit slot --slot a --color_by split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.analysis.umap_cc import cluster_vectors  # noqa: E402
from src.datasets._pair_helpers import pair_key_to_metadata, seq_hash_to_metadata  # noqa: E402
from src.utils import schema  # noqa: E402
from src.utils.config_hydra import load_function_metadata  # noqa: E402
from src.utils.plot_utils import umap_scatter  # noqa: E402

_SPLITS = ('train', 'val', 'test')

# Split labels are semantic, so each is pinned to a color -- rank-assigned color would move with the
# counts, and 'both' dominates the random arm while being absent from the 2D-CD one.
_SPLIT_COLORS = {'train/val only': '#4c72b0', 'test only': '#d43d51', 'both': '#dd8452'}


def load_fold_pairs(fold_dir: Path, hash_a: str, hash_b: str) -> dict:
    """Read the fold's three split CSVs, positives and negatives alike.

    Args:
        fold_dir: a `fold_k` directory holding `{train,val,test}_pairs.csv`.
        hash_a / hash_b: the alphabet's per-slot hash columns.

    Returns:
        {split_name: DataFrame} carrying pair_key, label and both hash columns.

    Raises:
        SystemExit: if a split CSV is missing.
    """
    out = {}
    for name in _SPLITS:
        path = fold_dir / f'{name}_pairs.csv'
        if not path.exists():
            raise SystemExit(f'ERROR: no {name} split at {path}.')
        out[name] = pd.read_csv(path, usecols=['pair_key', 'label', hash_a, hash_b],
                                dtype=str, keep_default_na=False, na_values=[], low_memory=False)
    return out


def load_fold_positives(fold_dir: Path, hash_a: str, hash_b: str) -> dict:
    """The positive rows of `load_fold_pairs`, with the label column dropped.

    Args:
        fold_dir: a `fold_k` directory holding `{train,val,test}_pairs.csv`.
        hash_a / hash_b: the alphabet's per-slot hash columns.

    Returns:
        {split_name: DataFrame} carrying pair_key and both hash columns, positives only.
    """
    return {name: df[df['label'] == '1'].drop(columns='label').reset_index(drop=True)
            for name, df in load_fold_pairs(fold_dir, hash_a, hash_b).items()}


def load_dataset_info(fold_dir: Path) -> dict:
    """Schema pair and sequence source for the dataset `fold_dir` belongs to.

    The run dir is `fold_dir.parent` for a k-fold dataset and `fold_dir` itself for a single-split
    one, whose CSVs sit at the top level with no fold subdir; whichever holds a recognized file
    wins. Three sources, in order:
      - `cv_info.json`, from the CV builders.
      - `arm_info.json`: a random arm re-partitions another dataset's rows rather than building
        them, so it names its source and both arms resolve to the same schema pair and clusters.
      - `resolved_config.yaml`, which the v2 builder writes. A single-split dataset has only this,
        and under `metadata_holdout` it has no clustering at all, so `cluster_id_path` is None and
        `cds_final_path` is what locates the sequences.

    Args:
        fold_dir: a `fold_k` directory, or a single-split dataset directory.

    Returns:
        Dict carrying at least `schema_pair`, plus `cluster_id_path` and/or `cds_final_path`.

    Raises:
        SystemExit: if no recognized file is found in either candidate directory.
    """
    known = ('cv_info.json', 'arm_info.json', 'resolved_config.yaml')
    run_dir = next((d for d in (fold_dir.parent, fold_dir)
                    if any((d / f).exists() for f in known)), None)
    if run_dir is None:
        raise SystemExit(f'ERROR: no {" / ".join(known)} in {fold_dir} or {fold_dir.parent}.')

    if (run_dir / 'cv_info.json').exists():
        return json.loads((run_dir / 'cv_info.json').read_text())

    arm_path = run_dir / 'arm_info.json'
    if arm_path.exists():
        source = Path(json.loads(arm_path.read_text())['source_dataset_dir'])
        if not source.is_absolute():
            source = PROJ / source
        return json.loads((source / 'cv_info.json').read_text())

    cfg = OmegaConf.load(run_dir / 'resolved_config.yaml')
    # The config names the schema pair in full; cv_info records short names and the caller indexes
    # short_to_function with them, so convert to keep one contract.
    to_short = load_function_metadata(PROJ / 'conf' / 'virus' / 'flu.yaml').function_to_short
    return {'schema_pair': [to_short[str(f)] for f in OmegaConf.select(cfg, 'dataset.schema_pair')],
            'cluster_id_path': OmegaConf.select(cfg, 'dataset.split_strategy.cluster_id_path'),
            'cds_final_path': OmegaConf.select(cfg, 'dataset.split_strategy.cds_final_path')}


def processed_base_of(info: dict) -> Path:
    """The `data/processed/{virus}/{version}` dir holding the `*_final` parquets.

    Read off whichever path the dataset recorded: a cluster file sits at
    `<base>/clusters_*/tXXX/<file>`, a cds_dna_final at `<base>/cds_dna_final.parquet`.

    Args:
        info: from `load_dataset_info`.

    Returns:
        The processed-data base directory.

    Raises:
        SystemExit: when the dataset recorded neither path, so the metadata colorings have no
            `*_final` parquet to read.
    """
    cluster_path = info.get('cluster_id_path')
    source = cluster_path or info.get('cds_final_path')
    if not source:
        raise SystemExit('ERROR: this dataset records neither cluster_id_path nor cds_final_path, '
                         'so --color_by <metadata field> cannot locate the *_final parquet; '
                         'use --color_by split.')
    path = Path(source)
    if not path.is_absolute():
        path = PROJ / path
    return path.parents[2] if cluster_path else path.parent


def split_labels(keys, test_keys: set, trainval_keys: set) -> np.ndarray:
    """Label each key by which splits it appears in.

    Args:
        keys: per-point key (a sequence hash for the slot unit, a pair_key for the pair unit).
        test_keys / trainval_keys: keys present in the test split, and in train or val.

    Returns:
        Array of 'train/val only' | 'test only' | 'both', aligned to `keys`.

    'both' needs one key to appear in two splits, so it cannot arise for the pair unit (a pair_key
    sits in exactly one split) nor for the slot unit under 2D-CD (a sequence's atom sits in one
    split). It is the random arm's signature.
    """
    labels = []
    for k in keys:
        in_test, in_trainval = k in test_keys, k in trainval_keys
        labels.append('both' if in_test and in_trainval else 'test only' if in_test else 'train/val only')
    return np.asarray(labels)


def slot_points(pos: dict, hash_col: str, alphabet: str):
    """Unique positive sequences of one slot, with their k-mer vectors.

    Args:
        pos: {split_name: positives DataFrame} from `load_fold_positives`.
        hash_col: that slot's hash column.
        alphabet: k-mer alphabet passed to `cluster_vectors`.

    Returns:
        `(keys, X)`: the hashes that have a k-mer row, and their `(n, D)` vectors.
    """
    hashes = sorted({h for df in pos.values() for h in df[hash_col]})
    X, keep = cluster_vectors(hashes, alphabet)
    keys = np.asarray(hashes)[keep]
    return keys, X


def pair_points(pos: dict, hash_a: str, hash_b: str, alphabet: str):
    """Positive pairs with their two slot vectors concatenated, as the model consumes them.

    Args:
        pos: {split_name: positives DataFrame} from `load_fold_positives`.
        hash_a / hash_b: the alphabet's per-slot hash columns.
        alphabet: k-mer alphabet passed to `cluster_vectors`.

    Returns:
        `(keys, X)`: the pair_keys whose BOTH endpoints have a k-mer row, and their `(n, 2D)`
        concatenated vectors. Pairs missing either endpoint are dropped.
    """
    pairs = pd.concat(pos.values(), ignore_index=True).drop_duplicates('pair_key')
    vectors = {}
    for col in (hash_a, hash_b):
        hashes = sorted(set(pairs[col]))
        X, keep = cluster_vectors(hashes, alphabet)
        vectors[col] = dict(zip(np.asarray(hashes)[keep], X))

    have_both = pairs[hash_a].isin(vectors[hash_a]) & pairs[hash_b].isin(vectors[hash_b])
    kept = pairs[have_both]
    X = np.hstack([np.vstack([vectors[hash_a][h] for h in kept[hash_a]]),
                   np.vstack([vectors[hash_b][h] for h in kept[hash_b]])])
    return kept['pair_key'].to_numpy(), X


def metadata_labels(keys, unit: str, field: str, processed_base: Path, alphabet: str,
                    funcs: tuple) -> np.ndarray:
    """Modal metadata value per key, for the requested field.

    Args:
        keys: per-point key, as returned by `slot_points` / `pair_points`.
        unit: 'slot' (keys are sequence hashes) or 'pair' (keys are pair_keys).
        field: metadata column, e.g. 'hn_subtype'.
        processed_base: `data/processed/{virus}/{data_version}`, holding the `*_final` parquet.
        alphabet: selects the parquet and hash column via the schema registry.
        funcs: (slot-a function, slot-b function) full names; the slot unit uses the one whose
            hashes `keys` holds, which the caller passes as a 1-tuple.

    Returns:
        Array of metadata values aligned to `keys`; 'unknown' where the key has no value.
    """
    sch = schema.require(alphabet)
    final_path = processed_base / f'{sch.file_basename}.parquet'
    if unit == 'slot':
        table = seq_hash_to_metadata(final_path, funcs[0], hash_col=sch.hash_col, fields=(field,))
        lookup = dict(zip(table[sch.hash_col], table[field]))
    else:
        table = pair_key_to_metadata(final_path, funcs[0], funcs[1], hash_col=sch.hash_col,
                                     fields=(field,))
        lookup = dict(zip(table['pair_key'], table[field]))
    return np.asarray([lookup.get(k, 'unknown') for k in keys])


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--fold_dir', type=Path, required=True, help='a fold_k dir of a Stage-3 dataset.')
    p.add_argument('--unit', choices=('slot', 'pair'), default='slot',
                   help="'slot' = one point per sequence, 'pair' = one point per pair (default slot).")
    p.add_argument('--slot', choices=('a', 'b'), default='a', help='which slot, for --unit slot (default a).')
    p.add_argument('--color_by', default='split',
                   help="'split' or a metadata field (hn_subtype, host, year). Default split.")
    p.add_argument('--include_negatives', action='store_true',
                   help='plot negatives too, as point shape; --unit pair only.')
    p.add_argument('--alphabet', default='nt_cds', help='k-mer alphabet (default nt_cds).')
    p.add_argument('--alpha', type=float, default=0.5,
                   help='point opacity (default 0.5), so overlapping categories show through.')
    p.add_argument('--out_png', type=Path, default=None, help='default: <fold_dir>/figures/<name>.png')
    p.add_argument('--title', default=None, help='default: derived from the dataset, fold and coloring.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sch = schema.require(args.alphabet)
    hash_a, hash_b = f'{sch.hash_col}_a', f'{sch.hash_col}_b'

    cv_info = load_dataset_info(args.fold_dir)
    sa, sb = cv_info['schema_pair']
    short_to_full = load_function_metadata(PROJ / 'conf' / 'virus' / 'flu.yaml').short_to_function
    slot_short = sa if args.slot == 'a' else sb

    if args.include_negatives and args.unit != 'pair':
        raise SystemExit("--include_negatives needs --unit pair: one sequence carries both "
                         "positive and negative rows, so a slot point has no single label.")
    if args.include_negatives:
        pos = load_fold_pairs(args.fold_dir, hash_a, hash_b)
        label_of = dict(zip(pd.concat(pos.values())['pair_key'], pd.concat(pos.values())['label']))
        print('rows: ' + ' '.join(f'{n}={len(df):,}' for n, df in pos.items()))
    else:
        pos = load_fold_positives(args.fold_dir, hash_a, hash_b)
        label_of = None
        print('positives: ' + ' '.join(f'{n}={len(df):,}' for n, df in pos.items()))

    if args.unit == 'slot':
        hash_col = hash_a if args.slot == 'a' else hash_b
        keys, X = slot_points(pos, hash_col, args.alphabet)
        key_of = {n: set(df[hash_col]) for n, df in pos.items()}
        funcs = (short_to_full[slot_short],)
        unit_label = f'{slot_short} sequences'
    else:
        keys, X = pair_points(pos, hash_a, hash_b, args.alphabet)
        key_of = {n: set(df['pair_key']) for n, df in pos.items()}
        funcs = (short_to_full[sa], short_to_full[sb])
        unit_label = f'{sa}-{sb} pairs'
    print(f'{unit_label}: {len(keys):,} points, {X.shape[1]} dims')

    markers = None
    if args.include_negatives:
        # Split keeps the color channel it has in every other figure, so the colors mean the same
        # thing across the set; positive/negative takes the shape channel. Reading them together
        # shows whether the negatives sit where the positives do.
        categories = split_labels(keys, key_of['test'], key_of['train'] | key_of['val'])
        markers = np.asarray(['positive' if label_of[k] == '1' else 'negative' for k in keys])
        pinned, legend_title = _SPLIT_COLORS, 'split'
    elif args.color_by == 'split':
        categories = split_labels(keys, key_of['test'], key_of['train'] | key_of['val'])
        pinned, legend_title = _SPLIT_COLORS, 'split'
    else:
        categories = metadata_labels(keys, args.unit, args.color_by, processed_base_of(cv_info),
                                     args.alphabet, funcs)
        pinned, legend_title = None, args.color_by

    color_by = 'split' if args.include_negatives else args.color_by
    # The suffix names which rows are in the figure, so a positives-only and a pos+neg figure of
    # the same coloring cannot collide (both color by split).
    name = (f'umap_{args.unit}' + (f'_{slot_short}' if args.unit == 'slot' else '')
            + f'_{color_by}' + ('_posneg' if args.include_negatives else '_pos'))
    out_png = args.out_png or (args.fold_dir / 'figures' / f'{name}.png')
    # The legends already name the color and shape channels, so the title carries only what they
    # cannot: which rows are in the figure.
    content = 'pos + neg' if args.include_negatives else 'pos only'
    title = args.title or (f'{args.fold_dir.parent.name} · {args.fold_dir.name}\n'
                           f'{unit_label}, {content} ({args.alphabet} k-mer)')

    stats = umap_scatter(X, categories, out_png=out_png, title=title, alpha=args.alpha,
                         category_colors=pinned, legend_title=legend_title, markers=markers)
    # Only the categories the figure colors; a metadata field can have a long tail (>100 subtypes)
    # that the plot folds into 'Others' anyway.
    counts = pd.Series(categories).value_counts().head(stats['n_selected'])
    print('  ' + ' | '.join(f'{c}={n:,}' for c, n in counts.items()))
    if stats['others_share']:
        print(f"  Others {stats['others_share']:.1%}")
    print(f"Done. {stats['n_points']:,} points, {stats['n_selected']} colored -> {out_png}")


if __name__ == '__main__':
    main()
