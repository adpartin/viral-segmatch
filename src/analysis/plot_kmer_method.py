"""Generate a 5-panel methods figure explaining how k-mer features are built
for the segment-matching pipeline.

Output: docs/figures/kmer_method_overview.{png,pdf}

Panels:
  A  Input: one GTO file = one isolate; contigs[] array with dna + replicon_type
  B  Stage 1 output: ctg_dna_final.csv with one row per (assembly_id, segment)
  C  Stage 2b: sliding window of width k over each DNA sequence, counting 6-mers
  D  Stacked sparse k-mer matrix (N_segments x 4^k); annotate real numbers from
     the kmer_features_k6_metadata.json
  E  Pair feature construction: two segment rows -> interaction (concat/diff/
     unit_diff) -> MLP

Numbers shown for the full Flu July 2025 dataset come from
data/embeddings/flu/July_2025/kmer_features_k6_metadata.json. The figure is
deterministic and does NOT require that file at runtime -- numbers are baked
in as constants below and should be refreshed manually if the dataset changes.

Run:
    python src/analysis/plot_kmer_method.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'figures'
METADATA_FILE = PROJECT_ROOT / 'data' / 'embeddings' / 'flu' / 'July_2025' / 'kmer_features_k6_metadata.json'

# --- Real numbers from the production k-mer cache ---
# Refresh if the dataset changes.
K = 6
VOCAB = 4 ** K                       # 4096
N_SEGMENTS = 868_240                 # unique (assembly, contig) rows
NNZ = 1_049_945_579                  # non-zeros in the sparse matrix
SPARSITY = 1.0 - NNZ / (N_SEGMENTS * VOCAB)  # ~0.705 (70.5% zeros)
AVG_DISTINCT_KMERS = NNZ / N_SEGMENTS        # ~1,209 distinct 6-mers per segment

# --- Canonical segment lengths for reference (Flu A, approximate) ---
SEGMENT_INFO = [
    ('S1/PB2', 2341),
    ('S2/PB1', 2341),
    ('S3/PA',  2233),
    ('S4/HA',  1777),
    ('S5/NP',  1565),
    ('S6/NA',  1413),
    ('S7/M',   1027),
    ('S8/NS',   890),
]


def _overwrite_from_metadata_if_available():
    """If the real metadata file exists, refresh the constants above so the
    figure annotation matches the current cache exactly."""
    global N_SEGMENTS, NNZ, SPARSITY, AVG_DISTINCT_KMERS
    if not METADATA_FILE.exists():
        return
    try:
        with open(METADATA_FILE) as fh:
            md = json.load(fh)
        N_SEGMENTS = int(md['n_sequences'])
        NNZ = int(md['nnz'])
        SPARSITY = float(md['sparsity'])
        AVG_DISTINCT_KMERS = NNZ / N_SEGMENTS
        print(f"Refreshed numbers from {METADATA_FILE}")
    except Exception as exc:
        print(f"Warning: couldn't parse {METADATA_FILE}: {exc}; using baked-in defaults.")


def _panel_a_gto(ax):
    ax.set_title('A. Input: GTO (Genome Typing Object) file', loc='left', fontsize=11, fontweight='bold')
    ax.axis('off')
    text = (
        "{\n"
        '  "scientific_name": "Influenza A virus (A/.../H3N2)",\n'
        '  "ncbi_taxonomy_id": 197911,\n'
        '  "quality": { "genome_quality": "Good" },\n'
        '  "contigs": [\n'
        '    { "id": "1406633.10", "replicon_type": "Segment 1",\n'
        '      "dna": "tcaaatatattcaatatgga...", "length": 2341 },\n'
        '    { "id": "1406633.8",  "replicon_type": "Segment 2",\n'
        '      "dna": "caaaccatttgaatggatgt...", "length": 2341 },\n'
        '    ... (6 more segments: PA, HA, NP, NA, M, NS) ...\n'
        '  ],\n'
        '  "features": [ {"type": "CDS", "protein_translation": "...", ...} ]\n'
        "}\n"
    )
    ax.text(0.01, 0.98, text, family='monospace', fontsize=8.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#aaa'))
    ax.text(0.01, 0.04,
            "One GTO per isolate.  `contigs[]` has one entry per genomic segment (DNA).\n"
            "`features[]` has one entry per CDS (protein).  Stage 1 extracts both.",
            fontsize=9, va='bottom', ha='left', style='italic', color='#555')


def _panel_b_segments(ax):
    ax.set_title('B. Stage 1: ctg_dna_final.csv — one row per (assembly, segment)',
                 loc='left', fontsize=11, fontweight='bold')
    names = [s[0] for s in SEGMENT_INFO]
    lengths = [s[1] for s in SEGMENT_INFO]
    y_pos = np.arange(len(names))[::-1]
    bars = ax.barh(y_pos, lengths, color='#4a90d9', edgecolor='#1f4c82')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('DNA length (bp)', fontsize=9)
    ax.set_xlim(0, 2700)
    for bar, length in zip(bars, lengths):
        ax.text(bar.get_width() + 40, bar.get_y() + bar.get_height() / 2,
                f'{length:,}', va='center', fontsize=8, color='#333')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(0.0, -0.20,
            f"Aggregated across all isolates → {N_SEGMENTS:,} contig rows.\n"
            "Columns: assembly_id, genbank_ctg_id, canonical_segment (S1..S8), dna_seq, length, gc_content.",
            transform=ax.transAxes, fontsize=9, va='top', color='#555', style='italic')


def _panel_c_sliding_window(ax):
    ax.set_title('C. Stage 2b: slide a k=6 window along each DNA sequence',
                 loc='left', fontsize=11, fontweight='bold')
    ax.axis('off')

    dna = "ACGTACGNTTACGTACGT"  # 18 chars, includes one ambiguous N
    k = 6
    char_w = 0.045
    x0 = 0.04
    y_dna = 0.78

    # Draw DNA string
    for i, c in enumerate(dna):
        color = '#e85d5d' if c == 'N' else '#222'
        ax.text(x0 + i * char_w, y_dna, c, family='monospace', fontsize=14,
                color=color, ha='center', va='center', fontweight='bold')

    # Draw three example windows. DNA index: A(0)C(1)G(2)T(3)A(4)C(5)G(6)N(7)T(8)T(9)A(10)C(11)G(12)T(13)A(14)C(15)G(16)T(17)
    example_windows = [
        (0, '#4a90d9', 'valid: ACGTAC',        0.58),
        (2, '#b15dd9', 'skipped: GTACGN (has N)', 0.42),
        (8, '#4a90d9', 'valid: TTACGT',        0.26),
    ]
    for start, color, label, y_box in example_windows:
        left = x0 + (start - 0.5) * char_w
        width = k * char_w
        rect = mpatches.FancyBboxPatch(
            (left, y_dna - 0.08), width, 0.16,
            boxstyle='round,pad=0.005', linewidth=2,
            edgecolor=color, facecolor='none',
        )
        ax.add_patch(rect)
        # Arrow + label
        mid_x = left + width / 2
        ax.annotate('', xy=(mid_x, y_box + 0.02), xytext=(mid_x, y_dna - 0.09),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
        ax.text(mid_x, y_box - 0.01, label, ha='center', va='top',
                fontsize=9, color=color)

    # Rules / output
    ax.text(0.04, 0.12,
            "Rules:\n"
            "  • Step = 1 (all overlapping windows).\n"
            "  • Alphabet = {A, C, G, T}; windows containing any ambiguous base (N, R, Y, …) are skipped.\n"
            "  • No reverse-complement canonicalization; no strand awareness.\n"
            "Result per sequence: an integer count for each of the 4⁶ = 4,096 possible 6-mers.",
            fontsize=9, va='top', color='#333')


def _panel_d_sparse_matrix(ax):
    ax.set_title('D. Stacked sparse k-mer matrix: one row per segment, 4,096 columns',
                 loc='left', fontsize=11, fontweight='bold')

    # Synthetic 12 x 30 subsample that mirrors the real sparsity pattern.
    rng = np.random.default_rng(42)
    H, W = 12, 30
    mat = rng.poisson(lam=0.9, size=(H, W)).astype(float)
    mat[mat > 4] = 4  # cap for visual contrast
    im = ax.imshow(mat, cmap='Blues', aspect='auto', interpolation='nearest')
    ax.set_xlabel('k-mers  (4,096 columns, lexicographic: AAAAAA, AAAAAC, …, TTTTTT)',
                  fontsize=9)
    ax.set_ylabel('segments  (≈868K rows)', fontsize=9)
    ax.set_xticks([0, W - 1])
    ax.set_xticklabels(['AAAAAA', 'TTTTTT'], fontsize=8)
    ax.set_yticks([])

    # Annotation box with real numbers
    ax.text(1.03, 0.98,
            f"Real shape\n"
            f"  rows (segments):  {N_SEGMENTS:,}\n"
            f"  cols (6-mers):    {VOCAB:,}\n"
            f"  non-zeros:        {NNZ:,}\n"
            f"  sparsity:         {SPARSITY:.1%} zero\n"
            f"  avg distinct 6-mers\n  per segment:      ≈ {AVG_DISTINCT_KMERS:,.0f}\n\n"
            f"Storage\n"
            f"  scipy.sparse CSR (.npz)\n"
            f"  + parquet index mapping\n"
            f"  (assembly_id, genbank_ctg_id,\n"
            f"   canonical_segment) → row",
            transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9e6', edgecolor='#c4a23b'))


def _panel_e_pair(ax):
    ax.set_title('E. Pair feature for the classifier: lookup → interaction → MLP',
                 loc='left', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Two vectors (segment A and B)
    def draw_vec(x0, y0, label, color):
        ax.add_patch(mpatches.Rectangle((x0, y0), 0.34, 0.08,
                                        facecolor=color, edgecolor='#333'))
        # tick marks to suggest 4096 bins
        for i in range(1, 10):
            ax.plot([x0 + i * 0.034, x0 + i * 0.034],
                    [y0, y0 + 0.08], color='#fff', lw=0.5)
        ax.text(x0 + 0.17, y0 + 0.04, 'k-mer vector  (4,096-dim)',
                ha='center', va='center', fontsize=8, color='#fff', fontweight='bold')
        ax.text(x0 - 0.01, y0 + 0.04, label, ha='right', va='center',
                fontsize=10, fontweight='bold', color=color)

    draw_vec(0.10, 0.78, 'segment A', '#2b6cb0')
    draw_vec(0.10, 0.60, 'segment B', '#c53030')

    # Interaction options
    ax.text(0.52, 0.90, 'Interaction  f(A, B):',
            fontsize=10, fontweight='bold')
    ax.text(0.52, 0.82, "concat:     [A ; B]            → 8,192-dim  (used in this paper's 28-pair sweeps)",
            fontsize=9, family='monospace')
    ax.text(0.52, 0.75, "diff:       |A − B|            → 4,096-dim",
            fontsize=9, family='monospace')
    ax.text(0.52, 0.68, "unit_diff:  (A − B) / ‖A − B‖  → 4,096-dim",
            fontsize=9, family='monospace')
    ax.text(0.52, 0.61, "prod:       A ⊙ B              → 4,096-dim",
            fontsize=9, family='monospace')

    # Arrow to MLP
    ax.annotate('', xy=(0.58, 0.36), xytext=(0.45, 0.36),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    # MLP block
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.58, 0.22), 0.30, 0.22,
        boxstyle='round,pad=0.02',
        facecolor='#e8f4ea', edgecolor='#2f7a3d', linewidth=1.5))
    ax.text(0.73, 0.37, 'MLP classifier',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#2f7a3d')
    ax.text(0.73, 0.30, 'P(same isolate)',
            ha='center', va='center', fontsize=9, style='italic', color='#333')
    ax.text(0.73, 0.25, '∈ [0, 1]',
            ha='center', va='center', fontsize=8, family='monospace', color='#666')

    # Lookup caption
    ax.text(0.08, 0.46,
            "Lookup key:\n"
            "  (assembly_id, genbank_ctg_id)\n"
            "Pair CSV  → index parquet → matrix row",
            fontsize=8.5, family='monospace', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#aaa'))

    # Caveat
    ax.text(0.04, 0.06,
            "Note: multiple proteins can share one contig (M1/M2, NS1/NEP, PA/PA-X) → identical k-mer\n"
            "vector for both. Paper's 28-pair sweeps avoid this by selecting one function per segment.",
            fontsize=8.5, va='bottom', color='#555', style='italic')


def main():
    _overwrite_from_metadata_if_available()

    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.9], hspace=0.45, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    _panel_a_gto(ax_a)
    _panel_b_segments(ax_b)
    _panel_c_sliding_window(ax_c)
    _panel_d_sparse_matrix(ax_d)
    _panel_e_pair(ax_e)

    fig.suptitle(
        'K-mer feature pipeline: GTO contigs → sparse k-mer matrix → pair features for MLP',
        fontsize=13, fontweight='bold', y=0.995)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / 'kmer_method_overview.png'
    pdf_path = OUTPUT_DIR / 'kmer_method_overview.pdf'
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'Wrote {png_path}')
    print(f'Wrote {pdf_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
