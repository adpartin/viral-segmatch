#!/usr/bin/env python3
"""
Compare two ESM-2 embeddings HDF5 files to check if they are identical.
"""
import sys
from pathlib import Path
import h5py
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.esm2_utils import load_esm2_embedding

def load_embeddings_to_dataframe(h5_file_path):
    """
    Load embeddings from HDF5 file into a pandas DataFrame using the proper utility function.
    
    Args:
        h5_file_path: Path to the HDF5 embeddings file
        
    Returns:
        pandas.DataFrame: DataFrame with columns ['brc_fea_id', 'embedding_vector']
                         where 'embedding_vector' contains the actual embedding arrays
    """
    print(f"Loading embeddings from: {h5_file_path}")
    
    with h5py.File(h5_file_path, 'r') as file:
        # Get all dataset names (these are the brc_fea_ids)
        brc_fea_ids = list(file.keys())
        print(f"  Found {len(brc_fea_ids)} embeddings")
        
        # Load embeddings using the proper utility function
        embeddings = []
        for brc_id in brc_fea_ids:
            embedding = load_esm2_embedding(brc_id, h5_file_path)
            embeddings.append(embedding)
        
        # Create DataFrame
        df = pd.DataFrame({
            'brc_fea_id': brc_fea_ids,
            'embedding_vector': embeddings
        })
        
        print(f"  Embedding dimension: {embeddings[0].shape if embeddings else 'N/A'}")
        return df

def compare_embeddings_dataframes(df1, df2):
    """
    Compare two embeddings DataFrames for equality.
    
    Args:
        df1, df2: DataFrames with 'brc_fea_id' and 'embedding_vector' columns
        
    Returns:
        dict: Comparison results
    """
    print("\n" + "="*60)
    print("EMBEDDINGS COMPARISON")
    print("="*60)
    
    # Basic shape comparison
    print(f"DataFrame 1 shape: {df1.shape}")
    print(f"DataFrame 2 shape: {df2.shape}")
    print(f"Shapes match: {df1.shape == df2.shape}")
    
    if df1.shape != df2.shape:
        return {
            'identical': False,
            'reason': 'Different shapes',
            'details': f"df1: {df1.shape}, df2: {df2.shape}"
        }
    
    # Check if brc_fea_ids are identical
    ids_match = df1['brc_fea_id'].equals(df2['brc_fea_id'])
    print(f"BRC feature IDs match: {ids_match}")
    
    if not ids_match:
        # Find differences in IDs
        ids1_set = set(df1['brc_fea_id'])
        ids2_set = set(df2['brc_fea_id'])
        only_in_df1 = ids1_set - ids2_set
        only_in_df2 = ids2_set - ids1_set
        
        print(f"  IDs only in df1: {len(only_in_df1)}")
        print(f"  IDs only in df2: {len(only_in_df2)}")
        
        if only_in_df1:
            print(f"  Examples from df1: {list(only_in_df1)[:5]}")
        if only_in_df2:
            print(f"  Examples from df2: {list(only_in_df2)[:5]}")
    
    # Sort both DataFrames by brc_fea_id for comparison
    df1_sorted = df1.sort_values('brc_fea_id').reset_index(drop=True)
    df2_sorted = df2.sort_values('brc_fea_id').reset_index(drop=True)
    
    # Compare embedding vectors
    print("\nComparing embedding vectors...")
    embedding_differences = []
    
    for i in range(len(df1_sorted)):
        brc_id = df1_sorted.iloc[i]['brc_fea_id']
        emb1 = df1_sorted.iloc[i]['embedding_vector']
        emb2 = df2_sorted.iloc[i]['embedding_vector']
        
        # Check if arrays are equal
        if not np.array_equal(emb1, emb2):
            diff = np.abs(emb1 - emb2)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            embedding_differences.append({
                'brc_fea_id': brc_id,
                'max_difference': max_diff,
                'mean_difference': mean_diff
            })
    
    print(f"Embedding vectors with differences: {len(embedding_differences)}")
    
    if embedding_differences:
        print("\nTop 5 differences:")
        for i, diff in enumerate(embedding_differences[:5]):
            print(f"  {i+1}. {diff['brc_fea_id']}: max_diff={diff['max_difference']:.6f}, mean_diff={diff['mean_difference']:.6f}")
        
        return {
            'identical': False,
            'reason': 'Embedding vectors differ',
            'details': f"{len(embedding_differences)} out of {len(df1)} embeddings differ"
        }
    else:
        print("✅ All embedding vectors are identical!")
        return {
            'identical': True,
            'reason': 'All embeddings match',
            'details': f"All {len(df1)} embeddings are identical"
        }

def main():
    """Main function to compare the two embedding files."""
    
    # File paths
    file1 = Path("/nfs/lambda_stor_01/data/apartin/projects/cepi/viral-segmatch/data/embeddings/bunya/April_2025/esm2_embeddings.h5")
    file2 = Path("/nfs/lambda_stor_01/data/apartin/projects/cepi/viral-segmatch/data/embeddings/bunya/April_2025_v2/esm2_embeddings.h5")
    
    print("ESM-2 Embeddings Comparison Tool")
    print("="*60)
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    
    # Check if files exist
    if not file1.exists():
        print(f"❌ Error: File 1 does not exist: {file1}")
        return 1
    
    if not file2.exists():
        print(f"❌ Error: File 2 does not exist: {file2}")
        return 1
    
    try:
        # Load embeddings
        print("\nLoading embeddings...")
        df1 = load_embeddings_to_dataframe(file1)
        df2 = load_embeddings_to_dataframe(file2)
        
        # Compare embeddings
        result = compare_embeddings_dataframes(df1, df2)
        
        # Final result
        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        if result['identical']:
            print("✅ EMBEDDINGS ARE IDENTICAL")
        else:
            print("❌ EMBEDDINGS DIFFER")
            print(f"   Reason: {result['reason']}")
            print(f"   Details: {result['details']}")
        
        return 0 if result['identical'] else 1
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
