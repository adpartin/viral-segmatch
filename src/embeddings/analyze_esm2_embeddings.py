"""
Analyze ESM-2 embeddings for the segment pair classifier.

This script analyzes the ESM-2 embeddings generated in Stage 3,
providing insights into sequence processing, embedding quality,
and biological relationships in the embedding space.
"""

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
import umap

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.esm2_utils import load_esm2_embedding
from src.utils.config import (
    USE_CORE_PROTEINS_ONLY, CORE_FUNCTIONS, VIRUS_NAME, DATA_VERSION, 
    TASK_NAME, ESM2_MAX_RESIDUES
)
from src.utils.protein_utils import get_core_protein_filter_mask

# Define paths
dataset_dir = project_root / 'data' / 'datasets' / VIRUS_NAME / DATA_VERSION / TASK_NAME
embeddings_dir = project_root / 'data' / 'embeddings' / VIRUS_NAME / DATA_VERSION
processed_data_dir = project_root / 'data' / 'processed' / VIRUS_NAME / DATA_VERSION
results_dir = project_root / 'results' / VIRUS_NAME / DATA_VERSION / 'embeddings_analysis'
results_dir.mkdir(parents=True, exist_ok=True)

print(f"Dataset directory: {dataset_dir}")
print(f"Embeddings directory: {embeddings_dir}")
print(f"Results directory: {results_dir}")

# Load data
print("\nLoad data.")
embeddings_file = embeddings_dir / 'esm2_embeddings.h5'
protein_data_file = processed_data_dir / 'protein_final.csv'

# Load protein metadata
protein_df = pd.read_csv(protein_data_file)
print(f"Loaded protein metadata: {len(protein_df)} proteins")

# Filter to core proteins if configured
if USE_CORE_PROTEINS_ONLY:
    print(f"Filtering to core proteins only (USE_CORE_PROTEINS_ONLY={USE_CORE_PROTEINS_ONLY})")
    mask = get_core_protein_filter_mask(protein_df)
    protein_df = protein_df[mask].reset_index(drop=True)
    print(f"After filtering: {len(protein_df)} core proteins")
    
    # Print core protein breakdown
    core_breakdown = protein_df.groupby(['canonical_segment', 'function']).size()
    print("Core protein breakdown:")
    for (segment, function), count in core_breakdown.items():
        print(f"  {segment} ({function}): {count} proteins")
else:
    print(f"Using all proteins (USE_CORE_PROTEINS_ONLY={USE_CORE_PROTEINS_ONLY})")

# Load embeddings info
with h5py.File(embeddings_file, 'r') as f:
    embedding_ids = list(f.keys())
    if embedding_ids:
        sample_embedding = f[embedding_ids[0]][:]
        embedding_dim = len(sample_embedding)
    else:
        embedding_dim = 0

print(f"Embeddings file contains: {len(embedding_ids)} embeddings")
print(f"Embedding dimension: {embedding_dim}")

# Set up plotting style
plt.style.use('default')
sns.set_palette("Set3")


def analyze_sequence_lengths():
    """Analyze sequence lengths and truncation effects."""
    print("\n" + "="*60)
    print("SEQUENCE LENGTH & TRUNCATION ANALYSIS")
    print("="*60)
    breakpoint()

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
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Length distribution by segment
    segments = protein_with_emb['canonical_segment'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, segment in enumerate(segments):
        subset = protein_with_emb[protein_with_emb['canonical_segment'] == segment]
        ax1.hist(subset['original_length'], bins=30, alpha=0.7, 
                label=f'{segment} (n={len(subset)})', color=colors[i % len(colors)])
    
    ax1.axvline(x=ESM2_MAX_RESIDUES, color='red', linestyle='--', linewidth=2, 
                label=f'ESM-2 limit ({ESM2_MAX_RESIDUES})')
    ax1.set_xlabel('Original Sequence Length (residues)')
    ax1.set_ylabel('Count')
    ax1.set_title('Sequence Length Distribution by Segment', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Truncation rate by segment
    trunc_stats = protein_with_emb.groupby('canonical_segment')['truncated'].agg(['count', 'sum', 'mean'])
    bars = ax2.bar(trunc_stats.index, trunc_stats['mean'], color=colors[:len(trunc_stats)])
    ax2.set_ylabel('Truncation Rate')
    ax2.set_title('Truncation Rate by Segment', fontweight='bold')
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, trunc_stats['mean']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Residues lost distribution
    truncated_proteins = protein_with_emb[protein_with_emb['truncated']]
    if len(truncated_proteins) > 0:
        ax3.hist(truncated_proteins['residues_lost'], bins=20, color='#E74C3C', alpha=0.7)
        ax3.set_xlabel('Residues Lost')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Residues Lost (Truncated Proteins)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Function-specific analysis
    func_stats = protein_with_emb.groupby('function').agg({
        'original_length': 'mean',
        'truncated': 'mean'
    }).round(2)
    func_stats = func_stats.sort_values('original_length', ascending=False)
    
    # Shorten function names for display
    func_short = [func.split()[-1] if len(func.split()) > 1 else func for func in func_stats.index]
    
    ax4.scatter(func_stats['original_length'], func_stats['truncated'], 
               s=100, alpha=0.7, color='#9B59B6')
    ax4.set_xlabel('Average Sequence Length')
    ax4.set_ylabel('Truncation Rate')
    ax4.set_title('Length vs Truncation by Function', fontweight='bold')
    
    # Add function labels
    for i, func in enumerate(func_short):
        ax4.annotate(func, (func_stats['original_length'].iloc[i], 
                           func_stats['truncated'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'sequence_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed statistics
    protein_with_emb.to_csv(results_dir / 'sequence_length_details.csv', index=False)
    segment_stats.to_csv(results_dir / 'segment_truncation_stats.csv')
    
    return protein_with_emb


def load_embeddings_for_analysis(protein_subset_df, max_proteins=1000):
    """Load embeddings for analysis (subsample if too large for memory)."""
    print(f"\nLoading embeddings for analysis...")
    breakpoint()
    
    # Subsample if dataset is large
    if len(protein_subset_df) > max_proteins:
        protein_subset_df = protein_subset_df.sample(n=max_proteins, random_state=42)
        print(f"Subsampled to {max_proteins} proteins for analysis")
    
    embeddings = []
    valid_proteins = []
    
    with h5py.File(embeddings_file, 'r') as f:
        for _, row in protein_subset_df.iterrows():
            brc_id = row['brc_fea_id']
            if brc_id in f:
                embeddings.append(f[brc_id][:])
                valid_proteins.append(row)
    
    embeddings = np.array(embeddings)
    valid_proteins_df = pd.DataFrame(valid_proteins)
    
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings, valid_proteins_df


def analyze_embedding_space(protein_with_emb):
    """Analyze the embedding space for biological insights."""
    print("\n" + "="*60)
    print("EMBEDDING SPACE ANALYSIS")
    print("="*60)
    breakpoint()
    
    # Load embeddings for analysis
    embeddings, analysis_df = load_embeddings_for_analysis(protein_with_emb)
    
    # PCA analysis
    print("Running PCA...")
    pca = PCA(n_components=10)
    pca_embeddings = pca.fit_transform(embeddings)
    
    print(f"PCA explained variance (first 5 components): {pca.explained_variance_ratio_[:5]}")
    print(f"Cumulative explained variance (first 5): {np.cumsum(pca.explained_variance_ratio_[:5])}")
    
    # UMAP for visualization
    print("Running UMAP...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_embeddings = umap_reducer.fit_transform(embeddings)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PCA by segment
    segments = analysis_df['canonical_segment'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, segment in enumerate(segments):
        mask = analysis_df['canonical_segment'] == segment
        axes[0, 0].scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], 
                          c=colors[i % len(colors)], label=f'Segment {segment}', 
                          alpha=0.7, s=30)
    
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0, 0].set_title('PCA: Colored by Segment', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. UMAP by segment
    for i, segment in enumerate(segments):
        mask = analysis_df['canonical_segment'] == segment
        axes[0, 1].scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                          c=colors[i % len(colors)], label=f'Segment {segment}', 
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
    if len(analysis_df) <= 100:  # Only for small datasets
        similarity_matrix = cosine_similarity(embeddings)
        im = axes[1, 1].imshow(similarity_matrix, cmap='viridis')
        axes[1, 1].set_title('Cosine Similarity Matrix', fontweight='bold')
        plt.colorbar(im, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'Similarity matrix\n(too large to display)', 
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
    
    return embeddings, analysis_df, pca_embeddings, umap_embeddings


def analyze_functional_clustering(embeddings, analysis_df):
    """Analyze clustering by biological function."""
    print("\n" + "="*60)
    print("FUNCTIONAL CLUSTERING ANALYSIS")
    print("="*60)
    breakpoint()
    
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


def create_technical_summary():
    """Create technical summary of embedding generation."""
    print("\n" + "="*60)
    print("TECHNICAL SUMMARY")
    print("="*60)
    breakpoint()
    
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


def main():
    """Run all analyses."""
    print("Analyzing ESM-2 Embeddings for Stage 3")
    print("="*60)
    
    # Check if files exist
    if not embeddings_file.exists():
        print(f"Error: Embeddings file not found: {embeddings_file}")
        return
    
    if not protein_data_file.exists():
        print(f"Error: Protein data file not found: {protein_data_file}")
        return
    
    # 1. Sequence length analysis
    protein_with_emb = analyze_sequence_lengths()
    
    # 2. Embedding space analysis
    embeddings, analysis_df, pca_embeddings, umap_embeddings = analyze_embedding_space(protein_with_emb)
    
    # 3. Functional clustering analysis
    clustering_stats = analyze_functional_clustering(embeddings, analysis_df)
    
    # 4. Technical summary
    tech_summary = create_technical_summary()
    
    print(f"\nAnalysis complete! All results saved to: {results_dir}")
    
    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    truncation_rate = protein_with_emb['truncated'].mean()
    print(f"• Sequence truncation rate: {truncation_rate:.1%}")
    print(f"• Embedding coverage: {tech_summary['coverage_rate']:.1%}")
    print(f"• Segment separation in embedding space: {clustering_stats['segment_separation']:.4f}")
    print(f"• Function separation in embedding space: {clustering_stats['function_separation']:.4f}")
    
    if clustering_stats['segment_separation'] > 0.1:
        print("✓ Strong segment-based clustering detected")
    if clustering_stats['function_separation'] > 0.1:
        print("✓ Strong function-based clustering detected")

if __name__ == '__main__':
    main() 