#!/usr/bin/env python3
"""
Compare two HDF5 embedding files to identify differences.
Usage: python scripts/compare_embeddings.py
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

def compare_embeddings(file1_path, file2_path):
    """
    Compare two HDF5 embedding files and identify differences.
    
    Args:
        file1_path: Path to first embedding file (old)
        file2_path: Path to second embedding file (new)
    """
    print("="*80)
    print("EMBEDDING FILES COMPARISON")
    print("="*80)
    print(f"File 1 (old): {file1_path}")
    print(f"File 2 (new): {file2_path}")
    print("="*80)
    
    # Load both files
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
        # Get all keys (brc_fea_ids) from both files
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())
        
        print(f"\n📊 FILE STATISTICS:")
        print(f"File 1 keys: {len(keys1)}")
        print(f"File 2 keys: {len(keys2)}")
        
        # Find common and unique keys
        common_keys = keys1 & keys2
        only_in_file1 = keys1 - keys2
        only_in_file2 = keys2 - keys1
        
        print(f"\n🔍 KEY COMPARISON:")
        print(f"Common keys: {len(common_keys)}")
        print(f"Only in file 1: {len(only_in_file1)}")
        print(f"Only in file 2: {len(only_in_file2)}")
        
        # Show some examples of unique keys
        if only_in_file1:
            print(f"\n📋 Keys only in file 1 (first 10):")
            for i, key in enumerate(sorted(only_in_file1)[:10]):
                print(f"  {i+1:2d}. {key}")
            if len(only_in_file1) > 10:
                print(f"  ... and {len(only_in_file1) - 10} more")
        
        if only_in_file2:
            print(f"\n📋 Keys only in file 2 (first 10):")
            for i, key in enumerate(sorted(only_in_file2)[:10]):
                print(f"  {i+1:2d}. {key}")
            if len(only_in_file2) > 10:
                print(f"  ... and {len(only_in_file2) - 10} more")
        
        # Compare embedding dimensions for common keys
        if common_keys:
            print(f"\n🔬 EMBEDDING DIMENSIONS COMPARISON:")
            sample_key = list(common_keys)[0]
            emb1_shape = f1[sample_key].shape
            emb2_shape = f2[sample_key].shape
            print(f"Sample key: {sample_key}")
            print(f"File 1 embedding shape: {emb1_shape}")
            print(f"File 2 embedding shape: {emb2_shape}")
            
            if emb1_shape == emb2_shape:
                print("✅ Embedding dimensions match")
            else:
                print("❌ Embedding dimensions differ!")
        
        # Compare actual embedding values for a few common keys
        if common_keys and len(common_keys) > 0:
            print(f"\n🧮 EMBEDDING VALUES COMPARISON:")
            sample_keys = list(common_keys)[:5]  # Compare first 5 common keys
            
            for key in sample_keys:
                emb1 = f1[key][:]
                emb2 = f2[key][:]
                
                if emb1.shape == emb2.shape:
                    # Compare values
                    are_equal = np.allclose(emb1, emb2, rtol=1e-10, atol=1e-10)
                    max_diff = np.max(np.abs(emb1 - emb2)) if emb1.shape == emb2.shape else float('inf')
                    
                    print(f"  {key}:")
                    print(f"    Shapes match: {emb1.shape == emb2.shape}")
                    print(f"    Values equal: {are_equal}")
                    if not are_equal and emb1.shape == emb2.shape:
                        print(f"    Max difference: {max_diff:.2e}")
                else:
                    print(f"  {key}: Shape mismatch - {emb1.shape} vs {emb2.shape}")
        
        # Analyze the differences in key sets
        print(f"\n📈 DIFFERENCE ANALYSIS:")
        if only_in_file1 or only_in_file2:
            print("Key differences detected!")
            
            # Try to identify patterns in the differences
            if only_in_file1:
                print(f"\n🔍 Keys only in file 1 (old):")
                # Group by prefix or pattern
                prefixes1 = {}
                for key in only_in_file1:
                    prefix = key.split('.')[0] if '.' in key else key[:10]
                    prefixes1[prefix] = prefixes1.get(prefix, 0) + 1
                
                for prefix, count in sorted(prefixes1.items()):
                    print(f"  {prefix}*: {count} keys")
            
            if only_in_file2:
                print(f"\n🔍 Keys only in file 2 (new):")
                # Group by prefix or pattern
                prefixes2 = {}
                for key in only_in_file2:
                    prefix = key.split('.')[0] if '.' in key else key[:10]
                    prefixes2[prefix] = prefixes2.get(prefix, 0) + 1
                
                for prefix, count in sorted(prefixes2.items()):
                    print(f"  {prefix}*: {count} keys")
        else:
            print("✅ No key differences - same proteins in both files")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    # Set breakpoint for interactive analysis
    print("\n🔍 Setting breakpoint for interactive analysis...")
    print("Available variables:")
    print("  - file1_path, file2_path: File paths")
    print("  - keys1, keys2: Sets of keys from each file")
    print("  - common_keys: Keys present in both files")
    print("  - only_in_file1, only_in_file2: Unique keys")
    print("  - f1, f2: HDF5 file objects (if you want to load specific embeddings)")
    
    breakpoint()  # Interactive debugging point

def main():
    """Main function to run the comparison."""
    # Define file paths
    base_path = Path("/nfs/lambda_stor_01/data/apartin/projects/cepi/viral-segmatch/data/embeddings/bunya")
    
    file1_path = base_path / "April_2025" / "esm2_embeddings.h5"
    file2_path = base_path / "April_2025_v2" / "esm2_embeddings.h5"
    
    # Check if files exist
    if not file1_path.exists():
        print(f"❌ File 1 not found: {file1_path}")
        return
    
    if not file2_path.exists():
        print(f"❌ File 2 not found: {file2_path}")
        return
    
    # Run comparison
    compare_embeddings(file1_path, file2_path)

if __name__ == "__main__":
    main()