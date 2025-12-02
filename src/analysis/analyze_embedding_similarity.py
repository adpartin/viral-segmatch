#!/usr/bin/env python
"""
Comprehensive Embedding Similarity Analysis for Viral Segment Matching.

This diagnostic script helps understand why the model might plateau by analyzing:
1. Embedding similarity distributions for positive vs negative pairs
2. Intra-function vs inter-function embedding distances
3. t-SNE visualization of embeddings colored by function and isolate
4. Sequence diversity metrics (if available)

The analysis quantifies three potential causes of plateauing:
- Cause 1: Proteins too conserved → similar embeddings across isolates
- Cause 2: ESM-2 captures function, not origin → function clusters, isolates mixed
- Cause 3: Concatenation doesn't capture relational info → overlapping pos/neg distributions

Usage:
    python src/analysis/embedding_similarity_analysis.py \
        --dataset_dir data/datasets/flu_a/July_2025.../runs/dataset_... \
        --embeddings_file data/embeddings/flu_a/July_2025/master_esm2_embeddings.h5 \
        --virus_name flu_a

    # For Bunya:
    python src/analysis/embedding_similarity_analysis.py \
        --dataset_dir data/datasets/bunya/April_2025 \
        --embeddings_file data/embeddings/bunya/April_2025/master_esm2_embeddings.h5 \
        --virus_name bunya
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_pairs(dataset_dir: Path, split: str = 'train') -> pd.DataFrame:
    """Load pairs from dataset directory."""
    pairs_file = dataset_dir / f'{split}_pairs.csv'
    if not pairs_file.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    return pd.read_csv(pairs_file)


def load_embeddings_index(embeddings_file: Path) -> dict:
    """Load the brc_fea_id to row index mapping from parquet."""
    parquet_file = embeddings_file.with_suffix('.parquet')
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet index not found: {parquet_file}")
    
    index_df = pd.read_parquet(parquet_file)
    return dict(zip(index_df['brc_fea_id'], index_df['row']))


def compute_similarities(pairs: pd.DataFrame, 
                         h5_file: h5py.File,
                         id_to_row: dict,
                         max_pairs: int = 5000) -> tuple:
    """Compute cosine similarities for pairs."""
    
    # Sample if too many pairs
    if len(pairs) > max_pairs:
        pairs = pairs.sample(n=max_pairs, random_state=42)
    
    pos_sims = []
    neg_sims = []
    pos_pair_info = []  # Store function info for analysis
    neg_pair_info = []
    
    emb_data = h5_file['emb']
    
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Computing pair similarities"):
        brc_a, brc_b = row['brc_a'], row['brc_b']
        label = row['label']
        
        if brc_a not in id_to_row or brc_b not in id_to_row:
            continue
        
        row_a = id_to_row[brc_a]
        row_b = id_to_row[brc_b]
        
        emb_a = emb_data[row_a]
        emb_b = emb_data[row_b]
        
        # Cosine similarity (1 - cosine distance)
        sim = 1 - cosine(emb_a, emb_b)
        
        # Extract function info if available
        func_a = row.get('func_a', 'unknown')
        func_b = row.get('func_b', 'unknown')
        
        if label == 1:
            pos_sims.append(sim)
            pos_pair_info.append({'func_a': func_a, 'func_b': func_b, 'sim': sim})
        else:
            neg_sims.append(sim)
            neg_pair_info.append({'func_a': func_a, 'func_b': func_b, 'sim': sim})
    
    return np.array(pos_sims), np.array(neg_sims), pos_pair_info, neg_pair_info


def compute_function_distances(pairs: pd.DataFrame,
                               h5_file: h5py.File,
                               id_to_row: dict,
                               max_samples: int = 500) -> dict:
    """Compute embedding distances for same-function vs different-function pairs.
    
    This helps diagnose if ESM-2 clusters by function (expected) vs isolate (desired).
    """
    
    # Sample pairs
    if len(pairs) > max_samples * 2:
        pairs = pairs.sample(n=max_samples * 2, random_state=42)
    
    emb_data = h5_file['emb']
    
    # Collect embeddings by function
    func_embeddings = defaultdict(list)
    func_brc_ids = defaultdict(list)
    
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Collecting function embeddings"):
        for suffix in ['a', 'b']:
            brc = row[f'brc_{suffix}']
            func = row.get(f'func_{suffix}', 'unknown')
            
            if brc in id_to_row and brc not in func_brc_ids[func]:
                func_embeddings[func].append(emb_data[id_to_row[brc]])
                func_brc_ids[func].append(brc)
    
    # Compute intra-function and inter-function distances
    intra_func_sims = []
    inter_func_sims = []
    
    functions = list(func_embeddings.keys())
    
    for func in functions:
        embs = func_embeddings[func]
        if len(embs) >= 2:
            # Sample for efficiency
            if len(embs) > 100:
                indices = np.random.choice(len(embs), 100, replace=False)
                embs = [embs[i] for i in indices]
            
            # Intra-function similarities
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sim = 1 - cosine(embs[i], embs[j])
                    intra_func_sims.append(sim)
    
    # Inter-function similarities (between different functions)
    for i, func_i in enumerate(functions):
        for j, func_j in enumerate(functions):
            if i >= j:
                continue
            embs_i = func_embeddings[func_i][:50]  # Limit samples
            embs_j = func_embeddings[func_j][:50]
            
            for emb_a in embs_i:
                for emb_b in embs_j:
                    sim = 1 - cosine(emb_a, emb_b)
                    inter_func_sims.append(sim)
    
    return {
        'intra_func_sims': np.array(intra_func_sims),
        'inter_func_sims': np.array(inter_func_sims),
        'functions': functions,
        'embeddings_by_func': {k: len(v) for k, v in func_embeddings.items()}
    }


def compute_tsne(pairs: pd.DataFrame,
                 h5_file: h5py.File,
                 id_to_row: dict,
                 max_samples: int = 1000) -> dict:
    """Compute t-SNE for visualization."""
    
    if len(pairs) > max_samples:
        pairs = pairs.sample(n=max_samples, random_state=42)
    
    emb_data = h5_file['emb']
    
    embeddings = []
    functions = []
    isolates = []
    brc_ids = set()
    
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Collecting embeddings for t-SNE"):
        for suffix in ['a', 'b']:
            brc = row[f'brc_{suffix}']
            if brc in id_to_row and brc not in brc_ids:
                brc_ids.add(brc)
                embeddings.append(emb_data[id_to_row[brc]])
                functions.append(row.get(f'func_{suffix}', 'unknown'))
                # Extract isolate from brc_id (format: fig|123.456.CDS.1)
                isolate = brc.split('|')[1].split('.')[1] if '|' in brc else 'unknown'
                isolates.append(isolate)
    
    if len(embeddings) < 50:
        return None
    
    embeddings = np.array(embeddings)
    
    # Use PCA first for efficiency
    print("  Running PCA...")
    pca = PCA(n_components=min(50, embeddings.shape[0], embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings)
    
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    return {
        'embeddings_2d': embeddings_2d,
        'functions': functions,
        'isolates': isolates
    }


def analyze_distributions(pos_sims: np.ndarray, neg_sims: np.ndarray) -> dict:
    """Compute statistical metrics for pos/neg similarity distributions."""
    
    # Basic statistics
    pos_mean, pos_std = pos_sims.mean(), pos_sims.std()
    neg_mean, neg_std = neg_sims.mean(), neg_sims.std()
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
    effect_size = (pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0
    
    # Statistical tests
    mann_whitney = mannwhitneyu(pos_sims, neg_sims, alternative='two-sided')
    ks_test = ks_2samp(pos_sims, neg_sims)
    
    # Histogram overlap
    bins = np.linspace(
        min(pos_sims.min(), neg_sims.min()),
        max(pos_sims.max(), neg_sims.max()),
        50
    )
    pos_hist, _ = np.histogram(pos_sims, bins=bins, density=True)
    neg_hist, _ = np.histogram(neg_sims, bins=bins, density=True)
    overlap = np.sum(np.minimum(pos_hist, neg_hist)) * (bins[1] - bins[0])
    
    # Classification metrics if we used cosine similarity as classifier
    all_sims = np.concatenate([pos_sims, neg_sims])
    all_labels = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
    
    # Find optimal threshold
    thresholds = np.percentile(all_sims, np.arange(5, 96, 5))
    best_acc = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (all_sims >= thresh).astype(int)
        acc = (preds == all_labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    return {
        'pos_mean': float(pos_mean),
        'pos_std': float(pos_std),
        'pos_min': float(pos_sims.min()),
        'pos_max': float(pos_sims.max()),
        'pos_count': len(pos_sims),
        'neg_mean': float(neg_mean),
        'neg_std': float(neg_std),
        'neg_min': float(neg_sims.min()),
        'neg_max': float(neg_sims.max()),
        'neg_count': len(neg_sims),
        'mean_difference': float(pos_mean - neg_mean),
        'effect_size_cohens_d': float(effect_size),
        'histogram_overlap': float(overlap),
        'mann_whitney_statistic': float(mann_whitney.statistic),
        'mann_whitney_pvalue': float(mann_whitney.pvalue),
        'ks_statistic': float(ks_test.statistic),
        'ks_pvalue': float(ks_test.pvalue),
        'best_threshold': float(best_thresh),
        'best_threshold_accuracy': float(best_acc),
    }


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    if abs(d) < 0.2:
        return "NEGLIGIBLE"
    elif abs(d) < 0.5:
        return "SMALL"
    elif abs(d) < 0.8:
        return "MEDIUM"
    else:
        return "LARGE"


def create_comprehensive_plots(pos_sims: np.ndarray,
                               neg_sims: np.ndarray,
                               stats: dict,
                               func_distances: dict,
                               tsne_data: dict,
                               output_dir: Path,
                               virus_name: str):
    """Create comprehensive visualization plots."""
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # =========================================================================
    # Plot 1: Main similarity distribution (top-left)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pos_sims, bins=50, alpha=0.6, label=f'Positive (n={len(pos_sims)})', 
             color='green', density=True)
    ax1.hist(neg_sims, bins=50, alpha=0.6, label=f'Negative (n={len(neg_sims)})', 
             color='red', density=True)
    ax1.axvline(stats['pos_mean'], color='darkgreen', linestyle='--', linewidth=2)
    ax1.axvline(stats['neg_mean'], color='darkred', linestyle='--', linewidth=2)
    ax1.set_xlabel('Cosine Similarity', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'Positive vs Negative Pair Similarity\n(Overlap: {stats["histogram_overlap"]:.1%})', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # =========================================================================
    # Plot 2: Box plot comparison (top-center)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([pos_sims, neg_sims], 
                      labels=['Positive\n(Same Isolate)', 'Negative\n(Diff Isolate)'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Cosine Similarity', fontsize=11)
    effect_interp = interpret_effect_size(stats['effect_size_cohens_d'])
    ax2.set_title(f'Effect Size: {stats["effect_size_cohens_d"]:.3f} ({effect_interp})', fontsize=12)
    ax2.grid(alpha=0.3)
    
    # =========================================================================
    # Plot 3: Intra-function vs Inter-function (top-right)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    if func_distances and len(func_distances['intra_func_sims']) > 0:
        intra = func_distances['intra_func_sims']
        inter = func_distances['inter_func_sims']
        ax3.hist(intra, bins=40, alpha=0.6, label=f'Same Function (n={len(intra)})', 
                 color='blue', density=True)
        ax3.hist(inter, bins=40, alpha=0.6, label=f'Diff Function (n={len(inter)})', 
                 color='orange', density=True)
        ax3.axvline(intra.mean(), color='darkblue', linestyle='--', linewidth=2)
        ax3.axvline(inter.mean(), color='darkorange', linestyle='--', linewidth=2)
        intra_inter_gap = intra.mean() - inter.mean()
        ax3.set_title(f'Function Clustering\n(Intra-Inter Gap: {intra_inter_gap:.4f})', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor function analysis', 
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Function Clustering', fontsize=12)
    ax3.set_xlabel('Cosine Similarity', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # =========================================================================
    # Plot 4: t-SNE colored by Function (middle-left)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    if tsne_data is not None:
        embs_2d = tsne_data['embeddings_2d']
        funcs = tsne_data['functions']
        unique_funcs = list(set(funcs))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_funcs)))
        func_to_color = {f: c for f, c in zip(unique_funcs, colors)}
        
        for func in unique_funcs:
            mask = np.array([f == func for f in funcs])
            short_name = func[:20] + '...' if len(func) > 20 else func
            ax4.scatter(embs_2d[mask, 0], embs_2d[mask, 1], 
                       c=[func_to_color[func]], label=short_name, alpha=0.6, s=20)
        ax4.set_title('t-SNE: Colored by Function\n(Tight clusters = function-based)', fontsize=12)
        ax4.legend(fontsize=8, loc='best', markerscale=1.5)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for t-SNE', ha='center', va='center', 
                 transform=ax4.transAxes)
        ax4.set_title('t-SNE: Colored by Function', fontsize=12)
    ax4.set_xlabel('t-SNE 1', fontsize=11)
    ax4.set_ylabel('t-SNE 2', fontsize=11)
    
    # =========================================================================
    # Plot 5: t-SNE colored by Isolate (middle-center)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    if tsne_data is not None:
        isolates = tsne_data['isolates']
        unique_isolates = list(set(isolates))
        # Color by hash of isolate ID for consistent but distinct colors
        isolate_colors = [hash(iso) % 256 / 256 for iso in isolates]
        scatter = ax5.scatter(embs_2d[:, 0], embs_2d[:, 1], c=isolate_colors, 
                             cmap='hsv', alpha=0.6, s=20)
        ax5.set_title(f't-SNE: Colored by Isolate ({len(unique_isolates)} isolates)\n(Mixed = isolate info not in embedding)', fontsize=12)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for t-SNE', ha='center', va='center', 
                 transform=ax5.transAxes)
        ax5.set_title('t-SNE: Colored by Isolate', fontsize=12)
    ax5.set_xlabel('t-SNE 1', fontsize=11)
    ax5.set_ylabel('t-SNE 2', fontsize=11)
    
    # =========================================================================
    # Plot 6: Summary Statistics Table (middle-right)
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['─' * 20, '─' * 12, '─' * 25],
        ['Effect Size (d)', f'{stats["effect_size_cohens_d"]:.4f}', interpret_effect_size(stats["effect_size_cohens_d"])],
        ['Histogram Overlap', f'{stats["histogram_overlap"]:.1%}', 'High = hard to separate'],
        ['Mean Diff (pos-neg)', f'{stats["mean_difference"]:.4f}', ''],
        ['Pos Mean ± Std', f'{stats["pos_mean"]:.4f} ± {stats["pos_std"]:.4f}', ''],
        ['Neg Mean ± Std', f'{stats["neg_mean"]:.4f} ± {stats["neg_std"]:.4f}', ''],
        ['Best Threshold', f'{stats["best_threshold"]:.4f}', ''],
        ['Best Threshold Acc', f'{stats["best_threshold_accuracy"]:.1%}', 'Max with cosine sim'],
        ['Mann-Whitney p', f'{stats["mann_whitney_pvalue"]:.2e}', 'Low = different'],
        ['KS test p', f'{stats["ks_pvalue"]:.2e}', 'Low = different dist.'],
    ]
    
    table_text = '\n'.join([f'{row[0]:<22} {row[1]:<14} {row[2]:<25}' for row in table_data])
    ax6.text(0.05, 0.95, table_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax6.set_title('Statistical Summary', fontsize=12)
    
    # =========================================================================
    # Plot 7: Diagnosis Summary (bottom spanning)
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Generate diagnosis text
    diagnosis_lines = []
    diagnosis_lines.append(f"{'='*100}")
    diagnosis_lines.append(f"EMBEDDING ANALYSIS DIAGNOSIS: {virus_name.upper()}")
    diagnosis_lines.append(f"{'='*100}")
    diagnosis_lines.append("")
    
    # Cause 1: Conservation
    intra_mean = func_distances['intra_func_sims'].mean() if func_distances and len(func_distances['intra_func_sims']) > 0 else 0
    if intra_mean > 0.95:
        diagnosis_lines.append("⚠️  CAUSE 1 - HIGH CONSERVATION: Intra-function similarity is very high (>{:.2f})".format(intra_mean))
        diagnosis_lines.append("    → Proteins of the same function are nearly identical across isolates")
        diagnosis_lines.append("    → ESM-2 embeddings cannot distinguish isolate origin for same-function proteins")
    else:
        diagnosis_lines.append("✓  CAUSE 1 - Conservation: Intra-function similarity ({:.4f}) shows some diversity".format(intra_mean))
    
    diagnosis_lines.append("")
    
    # Cause 2: Function vs Isolate clustering
    if func_distances and len(func_distances['intra_func_sims']) > 0:
        gap = intra_mean - func_distances['inter_func_sims'].mean()
        if gap > 0.1:
            diagnosis_lines.append("⚠️  CAUSE 2 - FUNCTION CLUSTERING: Large intra-inter gap ({:.4f})".format(gap))
            diagnosis_lines.append("    → ESM-2 embeddings cluster strongly by protein function")
            diagnosis_lines.append("    → Isolate information is NOT encoded in the embeddings")
        else:
            diagnosis_lines.append("?  CAUSE 2 - Function Clustering: Intra-inter gap ({:.4f}) is modest".format(gap))
    
    diagnosis_lines.append("")
    
    # Cause 3: Pos/Neg separation
    if stats['histogram_overlap'] > 0.8:
        diagnosis_lines.append("⚠️  CAUSE 3 - HIGH OVERLAP: Positive and negative pairs have {:.1%} overlap".format(stats['histogram_overlap']))
        diagnosis_lines.append("    → Raw embedding concatenation provides almost no signal for classification")
        diagnosis_lines.append("    → The model cannot learn to distinguish same-isolate from different-isolate pairs")
    elif stats['histogram_overlap'] > 0.6:
        diagnosis_lines.append("⚠️  CAUSE 3 - MODERATE OVERLAP: {:.1%} overlap between pos/neg distributions".format(stats['histogram_overlap']))
        diagnosis_lines.append("    → Some signal exists but is weak")
    else:
        diagnosis_lines.append("✓  CAUSE 3 - Good Separation: Only {:.1%} overlap".format(stats['histogram_overlap']))
    
    diagnosis_lines.append("")
    diagnosis_lines.append("─" * 100)
    diagnosis_lines.append("RECOMMENDATIONS:")
    
    if stats['histogram_overlap'] > 0.7:
        diagnosis_lines.append("  1. ❌ Raw concatenation is insufficient - enable use_diff=True and use_prod=True")
        diagnosis_lines.append("  2. Consider contrastive learning (learn a distance metric, not binary classification)")
        diagnosis_lines.append("  3. Explore sequence-level features (mutations, strain metadata) in addition to embeddings")
        diagnosis_lines.append("  4. The ~0.74 F1 may be near-optimal given embedding limitations")
    else:
        diagnosis_lines.append("  1. Signal exists - try longer training with lower learning rate")
        diagnosis_lines.append("  2. Consider data augmentation or larger dataset")
    
    diagnosis_text = '\n'.join(diagnosis_lines)
    ax7.text(0.02, 0.98, diagnosis_text, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Title
    fig.suptitle(f'Comprehensive Embedding Analysis: {virus_name.upper()}\n'
                 f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_file = output_dir / f'embedding_analysis_{virus_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Comprehensive plot saved to: {output_file}")
    plt.close()
    
    return output_file


def save_results(stats: dict, 
                 func_distances: dict,
                 output_dir: Path,
                 virus_name: str):
    """Save all results to JSON."""
    
    results = {
        'virus_name': virus_name,
        'analysis_timestamp': datetime.now().isoformat(),
        'pair_similarity_stats': stats,
    }
    
    if func_distances:
        results['function_distances'] = {
            'intra_func_mean': float(func_distances['intra_func_sims'].mean()) if len(func_distances['intra_func_sims']) > 0 else None,
            'intra_func_std': float(func_distances['intra_func_sims'].std()) if len(func_distances['intra_func_sims']) > 0 else None,
            'inter_func_mean': float(func_distances['inter_func_sims'].mean()) if len(func_distances['inter_func_sims']) > 0 else None,
            'inter_func_std': float(func_distances['inter_func_sims'].std()) if len(func_distances['inter_func_sims']) > 0 else None,
            'intra_inter_gap': float(func_distances['intra_func_sims'].mean() - func_distances['inter_func_sims'].mean()) if len(func_distances['intra_func_sims']) > 0 and len(func_distances['inter_func_sims']) > 0 else None,
            'functions': func_distances['functions'],
            'samples_per_function': func_distances['embeddings_by_func'],
        }
    
    # Add interpretations
    results['interpretation'] = {
        'effect_size': interpret_effect_size(stats['effect_size_cohens_d']),
        'separability': 'LOW' if stats['histogram_overlap'] > 0.7 else 'MODERATE' if stats['histogram_overlap'] > 0.5 else 'HIGH',
        'primary_issue': 'HIGH_OVERLAP' if stats['histogram_overlap'] > 0.7 else 'MODERATE_OVERLAP' if stats['histogram_overlap'] > 0.5 else 'GOOD_SEPARATION',
    }
    
    output_file = output_dir / f'embedding_analysis_{virus_name}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {output_file}")
    
    return results


def print_summary(stats: dict, func_distances: dict, virus_name: str):
    """Print summary to console."""
    
    print("\n" + "="*70)
    print(f"EMBEDDING SIMILARITY ANALYSIS: {virus_name.upper()}")
    print("="*70)
    
    print(f"\n📊 POSITIVE vs NEGATIVE PAIRS:")
    print(f"   Positive (same isolate):  mean={stats['pos_mean']:.4f} ± {stats['pos_std']:.4f}  (n={stats['pos_count']})")
    print(f"   Negative (diff isolate):  mean={stats['neg_mean']:.4f} ± {stats['neg_std']:.4f}  (n={stats['neg_count']})")
    print(f"   Mean difference:          {stats['mean_difference']:.4f}")
    print(f"   Effect size (Cohen's d):  {stats['effect_size_cohens_d']:.4f} ({interpret_effect_size(stats['effect_size_cohens_d'])})")
    print(f"   Histogram overlap:        {stats['histogram_overlap']:.1%}")
    
    if func_distances and len(func_distances['intra_func_sims']) > 0:
        intra_mean = func_distances['intra_func_sims'].mean()
        inter_mean = func_distances['inter_func_sims'].mean()
        print(f"\n📊 FUNCTION CLUSTERING:")
        print(f"   Intra-function similarity: {intra_mean:.4f}")
        print(f"   Inter-function similarity: {inter_mean:.4f}")
        print(f"   Gap (intra - inter):       {intra_mean - inter_mean:.4f}")
    
    print(f"\n📊 CLASSIFICATION POTENTIAL (using cosine sim as classifier):")
    print(f"   Best threshold:           {stats['best_threshold']:.4f}")
    print(f"   Best accuracy:            {stats['best_threshold_accuracy']:.1%}")
    
    print("\n" + "="*70)
    
    if stats['histogram_overlap'] > 0.8:
        print("⚠️  CRITICAL: Very high overlap ({:.1%}) - embeddings provide minimal signal!".format(stats['histogram_overlap']))
    elif stats['histogram_overlap'] > 0.6:
        print("⚠️  WARNING: Moderate overlap ({:.1%}) - signal is weak".format(stats['histogram_overlap']))
    else:
        print("✓  Good separation ({:.1%} overlap) - embeddings have useful signal".format(stats['histogram_overlap']))
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Embedding Similarity Analysis')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory containing train_pairs.csv')
    parser.add_argument('--embeddings_file', type=str, required=True,
                        help='Path to master_esm2_embeddings.h5')
    parser.add_argument('--virus_name', type=str, default='unknown',
                        help='Virus name for labeling (flu_a, bunya, etc.)')
    parser.add_argument('--max_pairs', type=int, default=5000,
                        help='Maximum pairs to analyze')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots and results (default: dataset_dir)')
    parser.add_argument('--skip_tsne', action='store_true',
                        help='Skip t-SNE computation (faster)')
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    embeddings_file = Path(args.embeddings_file)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"EMBEDDING SIMILARITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Virus:           {args.virus_name}")
    print(f"Dataset dir:     {dataset_dir}")
    print(f"Embeddings file: {embeddings_file}")
    print(f"Output dir:      {output_dir}")
    print(f"Skip t-SNE:      {args.skip_tsne}")
    print(f"{'='*70}\n")
    
    # Load data
    step_start = time.time()
    print("📂 [1/6] Loading pairs...")
    train_pairs = load_pairs(dataset_dir, 'train')
    print(f"   ✓ Loaded {len(train_pairs)} training pairs ({format_time(time.time() - step_start)})")
    
    step_start = time.time()
    print("\n📂 [2/6] Loading embedding index...")
    id_to_row = load_embeddings_index(embeddings_file)
    print(f"   ✓ Loaded index for {len(id_to_row)} embeddings ({format_time(time.time() - step_start)})")
    
    step_start = time.time()
    print("\n🔢 [3/6] Computing pair similarities...")
    with h5py.File(embeddings_file, 'r') as h5:
        pos_sims, neg_sims, pos_info, neg_info = compute_similarities(
            train_pairs, h5, id_to_row, max_pairs=args.max_pairs
        )
        print(f"   ✓ Positive pairs: {len(pos_sims)}, Negative pairs: {len(neg_sims)} ({format_time(time.time() - step_start)})")
        
        step_start = time.time()
        print("\n🔢 [4/6] Computing function distances...")
        func_distances = compute_function_distances(train_pairs, h5, id_to_row)
        print(f"   ✓ Analyzed {len(func_distances.get('functions', []))} functions ({format_time(time.time() - step_start)})")
        
        if not args.skip_tsne:
            step_start = time.time()
            print("\n🎨 [5/6] Computing t-SNE (this may take a while)...")
            tsne_data = compute_tsne(train_pairs, h5, id_to_row)
            if tsne_data:
                print(f"   ✓ t-SNE computed for {len(tsne_data['embeddings_2d'])} points ({format_time(time.time() - step_start)})")
            else:
                print(f"   ⚠ Insufficient data for t-SNE ({format_time(time.time() - step_start)})")
        else:
            print("\n⏭️  [5/6] Skipping t-SNE (--skip_tsne flag)")
            tsne_data = None
    
    # Analyze distributions
    step_start = time.time()
    print("\n📊 [6/6] Analyzing distributions and creating plots...")
    stats = analyze_distributions(pos_sims, neg_sims)
    
    # Create plots
    plot_file = create_comprehensive_plots(
        pos_sims, neg_sims, stats, func_distances, tsne_data, 
        output_dir, args.virus_name
    )
    
    # Save results
    results = save_results(stats, func_distances, output_dir, args.virus_name)
    print(f"   ✓ Analysis and plotting complete ({format_time(time.time() - step_start)})")
    
    # Print summary
    print_summary(stats, func_distances, args.virus_name)
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"✅ ANALYSIS COMPLETE - Total time: {format_time(total_time)}")
    print(f"{'='*70}")
    print(f"\n📁 OUTPUT FILES:")
    print(f"   Plot: {plot_file}")
    print(f"   JSON: {output_dir / f'embedding_analysis_{args.virus_name}.json'}")
    print(f"\n💡 To view the plot:")
    print(f"   Open: {plot_file}")


if __name__ == '__main__':
    main()
