"""
Example usage of the plot_config module for consistent visualizations.

This script demonstrates how to use the global color schemes and protein name mappings
in different types of plots commonly used in the viral-segmatch project.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_config import (
    DATASET_COLORS, DATASET_COLORS_LIST, SAMPLE_STYLES, SAMPLE_PATTERNS,
    get_dataset_color, get_sample_style, map_protein_name, get_plot_label,
    apply_default_style, PROTEIN_NAME_MAPPING
)

def example_bar_plot():
    """Example: Simple bar plot with train/val/test colors."""
    apply_default_style()
    
    datasets = ['train', 'val', 'test']
    values = [1000, 150, 200]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(datasets, values, color=DATASET_COLORS_LIST, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Dataset Size Distribution')
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Dataset')
    
    plt.tight_layout()
    plt.show()

def example_positive_negative_bar_plot():
    """Example: Bar plot distinguishing positive/negative samples."""
    apply_default_style()
    
    datasets = ['train', 'val', 'test']
    pos_counts = [500, 75, 100]
    neg_counts = [500, 75, 100]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Use patterns to distinguish positive/negative
    bars1 = ax.bar(x - width/2, pos_counts, width, 
                   color=DATASET_COLORS_LIST,
                   alpha=0.8,
                   hatch=SAMPLE_PATTERNS['positive'],
                   label='Positive (+)')
    
    bars2 = ax.bar(x + width/2, neg_counts, width,
                   color=DATASET_COLORS_LIST,
                   alpha=0.8,
                   hatch=SAMPLE_PATTERNS['negative'],
                   label='Negative (-)')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Positive vs Negative Sample Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def example_scatter_plot():
    """Example: Scatter plot with train/val/test colors and pos/neg styling."""
    apply_default_style()
    
    np.random.seed(42)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate mock data
    for dataset in ['train', 'val', 'test']:
        for sample_type in ['positive', 'negative']:
            # Mock coordinates
            n_samples = 50 if dataset == 'train' else 15
            x = np.random.randn(n_samples)
            y = np.random.randn(n_samples)
            
            # Offset negative samples slightly
            if sample_type == 'negative':
                x += 1.5
                y += 1.5
            
            # Get styling
            style = get_sample_style(sample_type, dataset)
            label = get_plot_label(dataset, sample_type)
            
            # Use facecolors for filled/unfilled distinction in scatter
            if style['fillstyle'] == 'full':
                ax.scatter(x, y, 
                          c=style['color'],
                          marker=style['marker'],
                          alpha=style['alpha'],
                          edgecolors=style['edgecolor'],
                          linewidths=style['linewidth'],
                          label=label,
                          s=60)
            else:  # 'none' - empty/outline only
                ax.scatter(x, y, 
                          c='none',
                          marker=style['marker'],
                          alpha=style['alpha'],
                          edgecolors=style['color'],
                          linewidths=style['linewidth'],
                          label=label,
                          s=60)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Dataset and Sample Type Visualization')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def example_protein_name_mapping():
    """Example: Using protein name mapping in plots."""
    apply_default_style()
    
    # Original long names
    original_names = [
        'RNA-dependent RNA polymerase',
        'Pre-glycoprotein polyprotein GP complex',
        'Nucleocapsid protein'
    ]
    
    # Map to shorter names
    short_names = [map_protein_name(name) for name in original_names]
    
    print("Protein name mapping:")
    for original, short in zip(original_names, short_names):
        print(f"  {original} → {short}")
    
    # Example bar plot with mapped names
    values = [1600, 1400, 1450]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before mapping
    bars1 = ax1.bar(range(len(original_names)), values, color='lightblue', alpha=0.8)
    ax1.set_title('Before Mapping (Long Names)')
    ax1.set_ylabel('Number of Proteins')
    ax1.set_xticks(range(len(original_names)))
    ax1.set_xticklabels(original_names, rotation=45, ha='right')
    
    # After mapping
    bars2 = ax2.bar(range(len(short_names)), values, color='lightcoral', alpha=0.8)
    ax2.set_title('After Mapping (Short Names)')
    ax2.set_ylabel('Number of Proteins')
    ax2.set_xticks(range(len(short_names)))
    ax2.set_xticklabels(short_names)
    
    plt.tight_layout()
    plt.show()

def example_integration_with_existing_code():
    """Example: How to integrate with existing analysis code."""
    apply_default_style()
    
    # Simulate your existing dataframe structure
    data = {
        'dataset': ['train'] * 100 + ['val'] * 30 + ['test'] * 40,
        'label': np.random.choice([0, 1], 170),
        'function': np.random.choice(list(PROTEIN_NAME_MAPPING.keys()), 170),
        'accuracy': np.random.uniform(0.7, 0.95, 170)
    }
    df = pd.DataFrame(data)
    
    # Group by dataset and label
    grouped = df.groupby(['dataset', 'label']).agg({
        'accuracy': 'mean',
        'function': 'count'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with consistent colors and styling
    for dataset in ['train', 'val', 'test']:
        for label in [0, 1]:
            subset = grouped[(grouped['dataset'] == dataset) & (grouped['label'] == label)]
            if len(subset) > 0:
                sample_type = 'positive' if label == 1 else 'negative'
                style = get_sample_style(sample_type, dataset)
                plot_label = get_plot_label(dataset, sample_type)
                
                # Use facecolors for filled/unfilled distinction in scatter
                if style['fillstyle'] == 'full':
                    ax.scatter(subset['function'], subset['accuracy'],
                              c=style['color'],
                              marker=style['marker'],
                              alpha=style['alpha'],
                              edgecolors=style['edgecolor'],
                              linewidths=style['linewidth'],
                              label=plot_label,
                              s=100)
                else:  # 'none' - empty/outline only
                    ax.scatter(subset['function'], subset['accuracy'],
                              c='none',
                              marker=style['marker'],
                              alpha=style['alpha'],
                              edgecolors=style['color'],
                              linewidths=style['linewidth'],
                              label=plot_label,
                              s=100)
    
    ax.set_xlabel('Number of Functions')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Performance by Dataset and Sample Type')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running plot configuration examples...")
    
    print("\n1. Simple bar plot with train/val/test colors:")
    example_bar_plot()
    
    print("\n2. Bar plot with positive/negative distinction:")
    example_positive_negative_bar_plot()
    
    print("\n3. Scatter plot with full styling:")
    example_scatter_plot()
    
    print("\n4. Protein name mapping:")
    example_protein_name_mapping()
    
    print("\n5. Integration with existing code:")
    example_integration_with_existing_code() 