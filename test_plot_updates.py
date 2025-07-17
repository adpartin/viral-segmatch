#!/usr/bin/env python3
"""
Test script to verify the plot configuration updates.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Test the plot configuration
from src.utils.plot_config import (
    DATASET_COLORS, DATASET_COLORS_LIST, map_protein_name, 
    apply_default_style, PROTEIN_NAME_MAPPING
)

def test_color_updates():
    """Test that the orange color has been updated to be brighter."""
    print("Testing color updates...")
    print(f"Train color: {DATASET_COLORS['train']}")
    print(f"Val color: {DATASET_COLORS['val']}")
    print(f"Test color: {DATASET_COLORS['test']}")
    print(f"Val color should be #ff7f0e (brighter orange): {DATASET_COLORS['val'] == '#ff7f0e'}")
    print()

def test_protein_name_mapping():
    """Test protein name mapping functionality."""
    print("Testing protein name mapping...")
    
    test_names = [
        'RNA-dependent RNA polymerase',
        'Pre-glycoprotein polyprotein GP complex',
        'Nucleocapsid protein',
        'RNA-dependent RNA polymerase (L protein)',
        'Pre-glycoprotein polyprotein GP complex (GPC protein)',
        'Nucleocapsid protein (N protein)',
        'Some other protein'  # Should remain unchanged
    ]
    
    for name in test_names:
        mapped = map_protein_name(name)
        print(f"  {name} → {mapped}")
    print()

def test_plot_example():
    """Test the actual plotting with updated colors and names."""
    print("Creating test plot...")
    
    apply_default_style()
    
    # Create mock data
    datasets = ['Train', 'Val', 'Test']
    original_names = ['RNA-dependent RNA polymerase', 'Pre-glycoprotein polyprotein GP complex', 'Nucleocapsid protein']
    mapped_names = [map_protein_name(name) for name in original_names]
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Test 1: Color comparison
    values = [1000, 150, 200]
    bars1 = ax1.bar(datasets, values, color=DATASET_COLORS_LIST, alpha=0.8)
    ax1.set_title('Updated Dataset Colors')
    ax1.set_ylabel('Count')
    
    # Add value labels
    for bar, value in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Test 2: Protein name mapping
    protein_counts = [1600, 1400, 1450]
    bars2 = ax2.bar(range(len(mapped_names)), protein_counts, 
                    color='lightcoral', alpha=0.8)
    ax2.set_title('Mapped Protein Names')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(len(mapped_names)))
    ax2.set_xticklabels(mapped_names)
    
    # Add value labels
    for bar, value in zip(bars2, protein_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('test_plot_updates.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Test plot saved as 'test_plot_updates.png'")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PLOT CONFIGURATION UPDATES")
    print("=" * 60)
    
    test_color_updates()
    test_protein_name_mapping()
    test_plot_example()
    
    print("All tests completed!") 