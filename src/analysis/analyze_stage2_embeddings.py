"""
Analyze Stage 2: ESM-2 embeddings for the segment pair classifier.

This script analyzes the ESM-2 embeddings generated in Stage 2,
providing insights into sequence processing, embedding quality,
and biological relationships in the embedding space.

Output directory: results/{virus_name}/{data_version}/embeddings_analysis/
(Shared across all config bundles - embeddings don't depend on config)

Usage:
    python src/analysis/analyze_stage2_embeddings.py --virus bunya --data_version April_2025
    python src/analysis/analyze_stage2_embeddings.py --virus flu --data_version July_2025
    python src/analysis/analyze_stage2_embeddings.py --virus flu --data_version July_2025 --embeddings_file /path/to/embeddings.h5
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.plot_config import apply_default_style
from src.utils.plot_utils import plot_sequence_length_distribution

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Constants
ESM2_MAX_RESIDUES = 1022
SEED = 42


def get_segment_config(virus_name: str, actual_segments: list = None) -> tuple:
    """Get segment colors and order for a virus.
    
    Args:
        virus_name: Name of the virus (bunya, flu_a, etc.)
        actual_segments: Optional list of actual segment names found in data.
                        If provided, colors will be generated for these segments.
    
    Returns:
        colors: Dict mapping segment name to color
        order: List of segments in canonical order
    """
    if virus_name == 'bunya':
        colors = {'S': '#1f77b4', 'M': '#ff7f0e', 'L': '#2ca02c'}
        order = ['S', 'M', 'L']
    elif virus_name == 'flu_a':
        # Flu A segments can be named '1'-'8' or 'S1'-'S8' depending on data source
        # Support both formats
        colors = {}
        for i in range(1, 9):
            colors[str(i)] = plt.cm.tab10(i/10)
            colors[f'S{i}'] = plt.cm.tab10(i/10)  # Also support 'S1', 'S2', etc.
        order = [str(i) for i in range(1, 9)]
    else:
        colors = {}
        order = []
    
    # If actual segments are provided, ensure all have colors
    if actual_segments:
        cmap = plt.cm.tab10
        for i, seg in enumerate(sorted(actual_segments)):
            if seg not in colors:
                colors[seg] = cmap(i / max(len(actual_segments), 10))
        # Update order to match actual segments if needed
        if not order or not any(seg in order for seg in actual_segments):
            order = sorted(actual_segments)
    
    return colors, order


def analyze_sequence_lengths():
    """Analyze sequence lengths and truncation effects."""
    print("\n" + "="*60)
    print("SEQUENCE LENGTH & TRUNCATION ANALYSIS")
    print("="*60)
    # breakpoint()

    # Filter to proteins that have embeddings
    protein_with_emb = protein_df[protein_df['brc_fea_id'].isin(embedding_ids)].copy()
    print(f"Proteins with embeddings: {len(protein_with_emb)}")
    
    # Calculate truncation
    protein_with_emb['original_length'] = protein_with_emb['esm2_ready_seq'].str.len()
    protein_with_emb['truncated'] = protein_with_emb['original_length'] > ESM2_MAX_RESIDUES
    protein_with_emb['truncated_length'] = np.minimum(protein_with_emb['original_length'], ESM2_MAX_RESIDUES)
    protein_with_emb['residues_lost'] = np.maximum(0, protein_with_emb['original_length'] - ESM2_MAX_RESIDUES)
    
    # Summary statistics
    total_proteins = len(protein_with_emb)
    truncated_count = protein_with_emb['truncated'].sum()
    truncation_rate = truncated_count / total_proteins
    
    print(f"Total proteins: {total_proteins}")
    print(f"Truncated proteins: {truncated_count} ({truncation_rate:.1%})")
    
    if truncated_count > 0:
        avg_residues_lost = protein_with_emb[protein_with_emb['truncated']]['residues_lost'].mean()
        max_residues_lost = protein_with_emb['residues_lost'].max()
        print(f"Average residues lost (truncated proteins): {avg_residues_lost:.1f}")
        print(f"Maximum residues lost: {max_residues_lost}")
    
    # Analysis by segment
    print("\nTruncation by segment:")
    segment_stats = protein_with_emb.groupby('canonical_segment').agg({
        'brc_fea_id': 'count',
        'truncated': ['sum', 'mean'],
        'original_length': ['mean', 'std'],
        'residues_lost': 'mean'
    }).round(2)
    segment_stats.columns = ['Count', 'Truncated_Count', 'Truncation_Rate', 
                           'Avg_Length', 'Std_Length', 'Avg_Residues_Lost']
    print(segment_stats)
    
    # Create separate sequence length distribution plot
    plot_sequence_length_distribution(
        protein_with_emb, 
        seq_column='esm2_ready_seq',
        segment_column='canonical_segment',
        title='Sequence Length Distribution by Segment',
        show_esm2_limit=True,
        esm2_max_residues=ESM2_MAX_RESIDUES,
        save_path=results_dir / 'sequence_length_analysis.png',
        show_plot=False
    )
    
    # Create separate visualizations
    
    # 1. Truncation rate by segment
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    trunc_stats = protein_with_emb.groupby('canonical_segment')['truncated'].agg(['count', 'sum', 'mean'])
    
    # Reorder to match SEGMENT_ORDER if possible, otherwise use sorted order
    actual_segments = list(trunc_stats.index)
    if SEGMENT_ORDER and any(seg in SEGMENT_ORDER for seg in actual_segments):
        # Reindex with available segments in order
        ordered_segs = [s for s in SEGMENT_ORDER if s in actual_segments]
        ordered_segs += [s for s in actual_segments if s not in ordered_segs]
        trunc_stats = trunc_stats.reindex(ordered_segs)
    
    # Ensure all segments have colors
    cmap = plt.cm.tab10
    for i, seg in enumerate(trunc_stats.index):
        if seg not in SEGMENT_COLORS:
            SEGMENT_COLORS[seg] = cmap(i / max(len(trunc_stats), 10))
    
    bars = ax1.bar(trunc_stats.index, trunc_stats['mean'], 
                  color=[SEGMENT_COLORS[seg] for seg in trunc_stats.index])
    ax1.set_ylabel('Truncation Rate', fontsize=12)
    ax1.set_xlabel('Segment', fontsize=12)
    ax1.set_title('Truncation Rate by Segment', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, trunc_stats['mean']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'truncation_rate_by_segment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Residues lost distribution
    truncated_proteins = protein_with_emb[protein_with_emb['truncated']]
    if len(truncated_proteins) > 0:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.hist(truncated_proteins['residues_lost'], bins=20, color='#E74C3C', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residues Lost', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Residues Lost (Truncated Proteins)', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / 'residues_lost_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Function-specific analysis
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    func_stats = protein_with_emb.groupby('function').agg({
        'original_length': 'mean',
        'truncated': 'mean'
    }).round(2)
    func_stats = func_stats.sort_values('original_length', ascending=False)
    
    # Shorten function names for display
    func_short = [func.split()[-1] if len(func.split()) > 1 else func for func in func_stats.index]
    
    ax3.scatter(func_stats['original_length'], func_stats['truncated'], 
               s=100, alpha=0.7, color='#9B59B6')
    ax3.set_xlabel('Average Sequence Length (amino acids)', fontsize=12)
    ax3.set_ylabel('Truncation Rate', fontsize=12)
    ax3.set_title('Length vs Truncation by Function', fontweight='bold', fontsize=14)
    
    # Add function labels
    for i, func in enumerate(func_short):
        ax3.annotate(func, (func_stats['original_length'].iloc[i], 
                           func_stats['truncated'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'length_vs_truncation_by_function.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed statistics
    protein_with_emb.to_csv(results_dir / 'sequence_length_details.csv', index=False)
    segment_stats.to_csv(results_dir / 'segment_truncation_stats.csv')
    
    return protein_with_emb


def load_embeddings_for_analysis(protein_subset_df, max_proteins=1000):
    """Load embeddings for analysis (subsample if too large for memory).

    Handles both master format (emb dataset + parquet index) and 
    legacy format (brc_id as keys directly).
    """
    print(f"\nLoading embeddings for analysis...")
    
    # Subsample if dataset is large
    if len(protein_subset_df) > max_proteins:
        protein_subset_df = protein_subset_df.sample(n=max_proteins, random_state=SEED)
        print(f"Subsampled to {max_proteins} proteins for analysis")
    
    embeddings = []
    valid_proteins = []
    
    with h5py.File(embeddings_file, 'r') as f:
        # Check if this is master format (has 'emb' dataset) or legacy format
        is_master_format = 'emb' in f
        
        if is_master_format:
            # Master format: load index from parquet, then access emb by row number
            parquet_file = embeddings_file.with_suffix('.parquet')
            if parquet_file.exists():
                index_df = pd.read_parquet(parquet_file)
                # Create brc_id -> row mapping
                brc_to_row = dict(zip(index_df['brc_fea_id'], index_df['row']))
            else:
                print(f"Warning: Master format but no parquet index found at {parquet_file}")
                brc_to_row = {}
            
            emb_data = f['emb']
            for _, row in protein_subset_df.iterrows():
                brc_id = row['brc_fea_id']
                if brc_id in brc_to_row:
                    row_idx = brc_to_row[brc_id]
                    embeddings.append(emb_data[row_idx])
                    valid_proteins.append(row)
        else:
            # Legacy format: brc_id is the key directly
            for _, row in protein_subset_df.iterrows():
                brc_id = row['brc_fea_id']
                if brc_id in f:
                    embeddings.append(f[brc_id][:])
                    valid_proteins.append(row)
    
    embeddings = np.array(embeddings)
    valid_proteins_df = pd.DataFrame(valid_proteins)
    
    # Ensure truncation information is included
    # Compute original_length if not present
    if len(valid_proteins_df) > 0 and 'original_length' not in valid_proteins_df.columns:
        if 'esm2_ready_seq' in valid_proteins_df.columns:
            valid_proteins_df['original_length'] = valid_proteins_df['esm2_ready_seq'].str.len()
        elif 'prot_seq' in valid_proteins_df.columns:
            valid_proteins_df['original_length'] = valid_proteins_df['prot_seq'].str.len()
        else:
            valid_proteins_df['original_length'] = 0
    
    if len(valid_proteins_df) > 0:
        if 'truncated' not in valid_proteins_df.columns:
            valid_proteins_df['truncated'] = valid_proteins_df['original_length'] > ESM2_MAX_RESIDUES
        if 'residues_lost' not in valid_proteins_df.columns:
            valid_proteins_df['residues_lost'] = np.maximum(0, valid_proteins_df['original_length'] - ESM2_MAX_RESIDUES)
    
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings, valid_proteins_df


def analyze_embedding_space(protein_with_emb, compute_cosine_similarity=False):
    """Analyze the embedding space for biological insights."""
    print("\n" + "="*60)
    print("EMBEDDING SPACE ANALYSIS")
    print("="*60)
    # breakpoint()
    
    # Load embeddings for analysis
    embeddings, analysis_df = load_embeddings_for_analysis(protein_with_emb)
    
    # PCA analysis
    print("Run PCA.")
    pca = PCA(n_components=10)
    pca_embeddings = pca.fit_transform(embeddings)
    
    print(f"PCA explained variance (first 5 components): {pca.explained_variance_ratio_[:5]}")
    print(f"Cumulative explained variance (first 5): {np.cumsum(pca.explained_variance_ratio_[:5])}")
    
    # UMAP for visualization
    if UMAP_AVAILABLE:
        print("Run UMAP.")
        umap_reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
        umap_embeddings = umap_reducer.fit_transform(embeddings)
    else:
        print("⚠️ UMAP not installed, using PCA for 2D projection instead")
        umap_embeddings = pca_embeddings[:, :2]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PCA by segment
    segments = sorted(analysis_df['canonical_segment'].unique())
    
    # Generate colors for any segments not in SEGMENT_COLORS
    cmap = plt.cm.tab10
    for i, seg in enumerate(segments):
        if seg not in SEGMENT_COLORS:
            SEGMENT_COLORS[seg] = cmap(i / max(len(segments), 10))
    
    for segment in segments:
        mask = analysis_df['canonical_segment'] == segment
        axes[0, 0].scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                          c=[SEGMENT_COLORS[segment]], label=f'Segment {segment}', 
                          alpha=0.7, s=30)
    
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 0].set_title('PCA: Colored by Segment', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. UMAP by segment
    for segment in segments:
        mask = analysis_df['canonical_segment'] == segment
        axes[0, 1].scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                          c=[SEGMENT_COLORS[segment]], label=f'Segment {segment}', 
                          alpha=0.7, s=30)
    
    axes[0, 1].set_xlabel('UMAP1')
    axes[0, 1].set_ylabel('UMAP2')
    axes[0, 1].set_title('UMAP: Colored by Segment', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. PCA by function
    functions = analysis_df['function'].unique()
    func_colors = plt.cm.Set3(np.linspace(0, 1, len(functions)))
    
    for i, func in enumerate(functions):
        mask = analysis_df['function'] == func
        func_short = func.split()[-1] if len(func.split()) > 1 else func
        axes[0, 2].scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                          c=[func_colors[i]], label=func_short, alpha=0.7, s=30)
    
    axes[0, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 2].set_title('PCA: Colored by Function', fontweight='bold')
    axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. PCA explained variance
    axes[1, 0].plot(range(1, 11), pca.explained_variance_ratio_[:10], 'bo-')
    axes[1, 0].plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_[:10]), 'ro-')
    axes[1, 0].set_xlabel('Principal Component')
    axes[1, 0].set_ylabel('Explained Variance Ratio')
    axes[1, 0].set_title('PCA Explained Variance', fontweight='bold')
    axes[1, 0].legend(['Individual', 'Cumulative'])
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Embedding similarity heatmap (sample)
    if compute_cosine_similarity and len(analysis_df) <= 100:  # Only for small datasets
        similarity_matrix = cosine_similarity(embeddings)
        im = axes[1, 1].imshow(similarity_matrix, cmap='viridis')
        axes[1, 1].set_title('Cosine Similarity Matrix', fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1])
    else:
        message = 'Similarity matrix\n(disabled)' if not compute_cosine_similarity else 'Similarity matrix\n(too large to display)'
        axes[1, 1].text(0.5, 0.5, message, 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Cosine Similarity Matrix', fontweight='bold')
    
    # 6. Sequence length vs PC1 (check if length affects embeddings)
    scatter = axes[1, 2].scatter(analysis_df['original_length'], pca_embeddings[:, 0], 
                                c=analysis_df['canonical_segment'].map({seg: i for i, seg in enumerate(segments)}),
                                cmap='Set1', alpha=0.7)
    axes[1, 2].set_xlabel('Original Sequence Length')
    axes[1, 2].set_ylabel('PC1')
    axes[1, 2].set_title('Sequence Length vs PC1', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'embedding_space_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save embedding coordinates
    embedding_coords = pd.DataFrame({
        'brc_fea_id': analysis_df['brc_fea_id'],
        'canonical_segment': analysis_df['canonical_segment'],
        'function': analysis_df['function'],
        'PC1': pca_embeddings[:, 0],
        'PC2': pca_embeddings[:, 1],
        'UMAP1': umap_embeddings[:, 0],
        'UMAP2': umap_embeddings[:, 1]
    })
    embedding_coords.to_csv(results_dir / 'embedding_coordinates.csv', index=False)
    
    return embeddings, analysis_df, pca_embeddings, umap_embeddings, pca


def analyze_functional_clustering(embeddings, analysis_df):
    """Analyze clustering by biological function."""
    print("\n" + "="*60)
    print("FUNCTIONAL CLUSTERING ANALYSIS")
    print("="*60)
    # breakpoint()
    
    # Calculate pairwise similarities
    similarity_matrix = cosine_similarity(embeddings)
    
    # Within vs between segment similarities
    segments = analysis_df['canonical_segment'].unique()
    functions = analysis_df['function'].unique()
    
    within_segment_sims = []
    between_segment_sims = []
    within_function_sims = []
    between_function_sims = []
    
    for i in range(len(analysis_df)):
        for j in range(i+1, len(analysis_df)):
            sim = similarity_matrix[i, j]
            
            # Segment comparison
            if analysis_df.iloc[i]['canonical_segment'] == analysis_df.iloc[j]['canonical_segment']:
                within_segment_sims.append(sim)
            else:
                between_segment_sims.append(sim)
            
            # Function comparison
            if analysis_df.iloc[i]['function'] == analysis_df.iloc[j]['function']:
                within_function_sims.append(sim)
            else:
                between_function_sims.append(sim)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Segment similarity comparison
    ax1.hist(within_segment_sims, bins=30, alpha=0.7, label='Within Segment', color='#3498DB')
    ax1.hist(between_segment_sims, bins=30, alpha=0.7, label='Between Segments', color='#E74C3C')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Count')
    ax1.set_title('Embedding Similarity: Within vs Between Segments', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Function similarity comparison
    ax2.hist(within_function_sims, bins=30, alpha=0.7, label='Same Function', color='#2ECC71')
    ax2.hist(between_function_sims, bins=30, alpha=0.7, label='Different Functions', color='#F39C12')
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Count')
    ax2.set_title('Embedding Similarity: Same vs Different Functions', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'functional_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate statistics
    stats = {
        'within_segment_mean': np.mean(within_segment_sims),
        'between_segment_mean': np.mean(between_segment_sims),
        'within_function_mean': np.mean(within_function_sims),
        'between_function_mean': np.mean(between_function_sims),
        'segment_separation': np.mean(within_segment_sims) - np.mean(between_segment_sims),
        'function_separation': np.mean(within_function_sims) - np.mean(between_function_sims)
    }
    
    print("Clustering Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(results_dir / 'clustering_statistics.csv', index=False)
    
    return stats


def create_separate_embedding_plots(embeddings, analysis_df, pca_embeddings, umap_embeddings, pca):
    """Create separate plots for presentation."""
    print("\n" + "="*60)
    print("CREATING SEPARATE PLOTS FOR PRESENTATION")
    print("="*60)
    
    # 1. PCA by segment
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    segments = sorted(analysis_df['canonical_segment'].unique())
    
    # Ensure all segments have colors
    cmap = plt.cm.tab10
    for i, seg in enumerate(segments):
        if seg not in SEGMENT_COLORS:
            SEGMENT_COLORS[seg] = cmap(i / max(len(segments), 10))
    
    for segment in segments:
        mask = analysis_df['canonical_segment'] == segment
        ax1.scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                   c=[SEGMENT_COLORS[segment]], label=f'Segment {segment}', 
                   alpha=0.7, s=50)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.set_title('PCA: Protein Embeddings by Segment', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'pca_by_segment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. UMAP by segment
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    
    for segment in segments:
        mask = analysis_df['canonical_segment'] == segment
        ax2.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                   c=[SEGMENT_COLORS[segment]], label=f'Segment {segment}', 
                   alpha=0.7, s=50)
    
    ax2.set_xlabel('UMAP1', fontsize=12)
    ax2.set_ylabel('UMAP2', fontsize=12)
    ax2.set_title('UMAP: Protein Embeddings by Segment', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'umap_by_segment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. PCA by truncation status
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    
    truncated_mask = analysis_df['truncated'] == True
    non_truncated_mask = analysis_df['truncated'] == False
    
    ax3.scatter(pca_embeddings[non_truncated_mask, 0], pca_embeddings[non_truncated_mask, 1], 
               c='#2E86AB', label=f'Non-truncated (n={non_truncated_mask.sum()})', 
               alpha=0.7, s=50)
    ax3.scatter(pca_embeddings[truncated_mask, 0], pca_embeddings[truncated_mask, 1], 
               c='#E74C3C', label=f'Truncated (n={truncated_mask.sum()})', 
               alpha=0.7, s=50)
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax3.set_title('PCA: Full-length vs Truncated Proteins', fontweight='bold', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'pca_by_truncation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. UMAP by truncation status
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8))
    
    ax4.scatter(umap_embeddings[non_truncated_mask, 0], umap_embeddings[non_truncated_mask, 1], 
               c='#2E86AB', label=f'Non-truncated (n={non_truncated_mask.sum()})', 
               alpha=0.7, s=50)
    ax4.scatter(umap_embeddings[truncated_mask, 0], umap_embeddings[truncated_mask, 1], 
               c='#E74C3C', label=f'Truncated (n={truncated_mask.sum()})', 
               alpha=0.7, s=50)
    
    ax4.set_xlabel('UMAP1', fontsize=12)
    ax4.set_ylabel('UMAP2', fontsize=12)
    ax4.set_title('UMAP: Full-length vs Truncated Proteins', fontweight='bold', fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'umap_by_truncation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. PCA colored by residues lost (for truncated proteins only)
    if truncated_mask.sum() > 0:
        fig5, ax5 = plt.subplots(1, 1, figsize=(10, 8))
        
        # Show non-truncated proteins in light gray
        ax5.scatter(pca_embeddings[non_truncated_mask, 0], pca_embeddings[non_truncated_mask, 1], 
                   c='lightgray', label=f'Non-truncated (n={non_truncated_mask.sum()})', 
                   alpha=0.3, s=30)
        
        # Color truncated proteins by residues lost
        residues_lost = analysis_df[truncated_mask]['residues_lost']
        scatter = ax5.scatter(pca_embeddings[truncated_mask, 0], pca_embeddings[truncated_mask, 1], 
                             c=residues_lost, cmap='Reds', alpha=0.8, s=50)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Residues Lost', rotation=270, labelpad=20, fontsize=12)
        
        ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax5.set_title('PCA: Impact of Truncation (Residues Lost)', fontweight='bold', fontsize=14)
        ax5.legend(fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'pca_by_residues_lost.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✅ Separate plots saved:")
    print("  - pca_by_segment.png")
    print("  - umap_by_segment.png") 
    print("  - pca_by_truncation.png")
    print("  - umap_by_truncation.png")
    if truncated_mask.sum() > 0:
        print("  - pca_by_residues_lost.png")
    
    return {
        'truncated_count': truncated_mask.sum(),
        'non_truncated_count': non_truncated_mask.sum(),
        'truncation_rate': truncated_mask.mean()
    }


def create_technical_summary():
    """Create technical summary of embedding generation."""
    print("\n" + "="*60)
    print("TECHNICAL SUMMARY")
    print("="*60)
    # breakpoint()
    
    # File size analysis
    embeddings_file_size = embeddings_file.stat().st_size / (1024**2)  # MB
    
    # Coverage analysis
    total_proteins = len(protein_df)
    embedded_proteins = len(embedding_ids)
    coverage_rate = embedded_proteins / total_proteins
    
    print(f"Embeddings file size: {embeddings_file_size:.1f} MB")
    print(f"Total proteins in dataset: {total_proteins}")
    print(f"Proteins with embeddings: {embedded_proteins}")
    print(f"Coverage rate: {coverage_rate:.1%}")
    print(f"Embedding dimension: {embedding_dim}")
    
    # Memory usage estimation
    memory_per_embedding = embedding_dim * 4  # 4 bytes per float32
    total_memory = embedded_proteins * memory_per_embedding / (1024**2)  # MB
    
    print(f"Memory per embedding: {memory_per_embedding} bytes")
    print(f"Total memory for all embeddings: {total_memory:.1f} MB")
    
    # Save technical summary
    tech_summary = {
        'file_size_mb': embeddings_file_size,
        'total_proteins': total_proteins,
        'embedded_proteins': embedded_proteins,
        'coverage_rate': coverage_rate,
        'embedding_dimension': embedding_dim,
        'memory_usage_mb': total_memory
    }
    
    tech_df = pd.DataFrame([tech_summary])
    tech_df.to_csv(results_dir / 'technical_summary.csv', index=False)
    
    return tech_summary


def main(
    virus_name: str,
    data_version: str,
    embeddings_file: Path = None, 
    use_core_proteins_only: bool = False,
    compute_cosine_sim: bool = False):
    """Run all analyses."""
    
    # Set up paths
    embeddings_dir = project_root / 'data' / 'embeddings' / virus_name / data_version
    processed_data_dir = project_root / 'data' / 'processed' / virus_name / data_version
    results_dir = project_root / 'results' / virus_name / data_version / 'embeddings_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get virus-specific segment config
    segment_colors, segment_order = get_segment_config(virus_name)
    
    # Set embeddings file path
    if embeddings_file is None:
        # Try master embeddings first, then legacy format
        master_emb = embeddings_dir / 'master_esm2_embeddings.h5'
        legacy_emb = embeddings_dir / 'esm2_embeddings.h5'
        if master_emb.exists():
            embeddings_file = master_emb
        elif legacy_emb.exists():
            embeddings_file = legacy_emb
        else:
            embeddings_file = master_emb  # Will fail with clear message
    else:
        embeddings_file = Path(embeddings_file)
    
    protein_data_file = processed_data_dir / 'protein_final.csv'
    
    # Make globals available
    globals()['embeddings_file'] = embeddings_file
    globals()['protein_data_file'] = protein_data_file
    globals()['results_dir'] = results_dir
    globals()['SEGMENT_COLORS'] = segment_colors
    globals()['SEGMENT_ORDER'] = segment_order
    
    print(f"\n{'='*60}")
    print("Stage 3: ESM-2 Embeddings Analysis")
    print("="*60)
    print(f"Virus: {virus_name}")
    print(f"Data version: {data_version}")
    print(f"Embeddings: {embeddings_file}")
    print(f"Protein data: {protein_data_file}")
    print(f"Results dir: {results_dir}")
    print("="*60 + "\n")
    
    # Apply default plotting style
    apply_default_style()
    
    # Check if files exist
    if not embeddings_file.exists():
        print(f"❌ Error: Embeddings file not found: {embeddings_file}")
        return
    
    if not protein_data_file.exists():
        print(f"❌ Error: Protein data file not found: {protein_data_file}")
        return
    
    # Load protein metadata
    print("Loading protein metadata...")
    protein_df = pd.read_csv(protein_data_file)
    print(f"Loaded {len(protein_df)} proteins")
    globals()['protein_df'] = protein_df
    
    # Update segment config with actual segments found in data
    if 'canonical_segment' in protein_df.columns:
        actual_segments = protein_df['canonical_segment'].dropna().unique().tolist()
        segment_colors, segment_order = get_segment_config(virus_name, actual_segments)
        globals()['SEGMENT_COLORS'] = segment_colors
        globals()['SEGMENT_ORDER'] = segment_order
        print(f"Found segments: {sorted(actual_segments)}")
    
    # Load embeddings info
    with h5py.File(embeddings_file, 'r') as f:
        # Handle different embedding file structures
        if 'emb' in f:
            # Master format
            embedding_dim = f['emb'].shape[1]
            # Load index from parquet
            parquet_file = embeddings_file.with_suffix('.parquet')
            if parquet_file.exists():
                index_df = pd.read_parquet(parquet_file)
                embedding_ids = list(index_df['brc_fea_id'])
            else:
                embedding_ids = []
        else:
            # Legacy format
            embedding_ids = list(f.keys())
            if embedding_ids:
                embedding_dim = len(f[embedding_ids[0]][:])
            else:
                embedding_dim = 0
    
    globals()['embedding_ids'] = embedding_ids
    globals()['embedding_dim'] = embedding_dim
    
    print(f"Embeddings: {len(embedding_ids)} proteins, dim={embedding_dim}")
    
    # 1. Sequence length analysis
    protein_with_emb = analyze_sequence_lengths()
    
    # 2. Embedding space analysis (skip UMAP if not installed)
    try:
        embeddings, analysis_df, pca_embeddings, umap_embeddings, pca = analyze_embedding_space(
            protein_with_emb, compute_cosine_sim)
    except ImportError:
        print("⚠️ UMAP not installed, skipping UMAP visualizations")
        return
    
    # 3. Create separate plots for presentation
    plot_stats = create_separate_embedding_plots(embeddings, analysis_df, pca_embeddings, umap_embeddings, pca)
    
    # 4. Functional clustering analysis (optional)
    if compute_cosine_sim:
        clustering_stats = analyze_functional_clustering(embeddings, analysis_df)
    else:
        print("\nSkipping cosine similarity computation (--compute_cosine_sim not set)")
        clustering_stats = {'segment_separation': 0.0, 'function_separation': 0.0}
    
    # 5. Technical summary
    tech_summary = create_technical_summary()
    
    print(f"\n✅ Analysis complete! Results saved to: {results_dir}")
    
    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    truncation_rate = protein_with_emb['truncated'].mean()
    print(f"• Sequence truncation rate: {truncation_rate:.1%}")
    print(f"• Embedding coverage: {tech_summary['coverage_rate']:.1%}")
    print(f"• Segment separation: {clustering_stats['segment_separation']:.4f}")
    print(f"• Function separation: {clustering_stats['function_separation']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Stage 3: ESM-2 embeddings')
    parser.add_argument('--virus', type=str, required=True,
                        choices=['bunya', 'flu_a'],
                        help='Virus name (bunya or flu_a)')
    parser.add_argument('--data_version', type=str, required=True,
                        help='Data version (e.g., April_2025, July_2025)')
    parser.add_argument('--embeddings_file', type=str, default=None,
                        help='Path to embeddings file (optional)')
    parser.add_argument('--use_core_proteins_only', action='store_true',
                        help='Filter to core proteins only')
    parser.add_argument('--compute_cosine_sim', action='store_true',
                        help='Compute cosine similarity analysis (time-consuming)')
    args = parser.parse_args()

    main(virus_name=args.virus,
        data_version=args.data_version,
        embeddings_file=args.embeddings_file,
        use_core_proteins_only=args.use_core_proteins_only,
        compute_cosine_sim=args.compute_cosine_sim
    ) 