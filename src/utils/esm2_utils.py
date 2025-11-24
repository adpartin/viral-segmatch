from typing import Optional, Union
from pathlib import Path
import hashlib

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import EsmModel, EsmTokenizer

ESM2_MAX_RESIDUES = 1022  # ESM-2 max seq length is 1024 with 2 tokens reserved for CLS, SEP (special tokens)


def _get_precision_dtypes(precision: str) -> tuple[str, str]:
    """
    Map precision string to HDF5 dtype and NumPy dtype.
    
    Args:
        precision: Storage precision for embeddings ('fp16', 'fp32')
            - fp16: Uses float16 (2 bytes, quantizes fp32->fp16, recommended)
            - fp32: Uses float32 (4 bytes, no quantization)

    Returns:
        tuple: (hdf5_dtype, numpy_dtype)
    """
    PRECISION_MAP = {
        'fp16': ('float16', 'float16'),
        'fp32': ('float32', 'float32'),
    }
    
    if precision not in PRECISION_MAP:
        raise ValueError(
            f"Unsupported precision: {precision}. "
            f"Supported options: {list(PRECISION_MAP.keys())}"
        )
    
    return PRECISION_MAP[precision]


def _apply_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str = 'mean'
    ) -> torch.Tensor:
    """
    Apply the specified pooling method to hidden states.
    
    A custom pooling method is implemented (not Hugging Face's built-in pooling).
    Since add_pooling_layer=False, we manually pool the hidden states with full control
    over which tokens to include/exclude (e.g., excluding CLS/SEP tokens).

    Args:
        hidden_states: Tensor of shape [batch_size, seq_length, embedding_dim]
        attention_mask: Tensor of shape [batch_size, seq_length] (1 for valid tokens, 0 for padding)
        pooling: Pooling method ('mean', 'max', 'cls', 'attention')
            - 'mean': Mean pooling over sequence tokens (excludes CLS/SEP)
            - 'max': Max pooling over sequence tokens (excludes CLS/SEP, TODO: not tested)
            - 'cls': Use CLS token only (TODO: not tested)
            - 'attention': Attention-weighted pooling (TODO: not tested)

    Returns:
        Pooled embeddings of shape [batch_size, embedding_dim]
    """
    if pooling == 'mean':
        # Mean pool over sequence length (exclude CLS and SEP tokens: [1:-1])
        # Use attention_mask to ignore padding tokens (masked mean pooling)
        seq_tokens = hidden_states[:, 1:-1]  # Exclude CLS (index 0) and SEP (index -1)
        mask = attention_mask[:, 1:-1].unsqueeze(-1)  # [batch_size, seq_len-2, 1]
        masked_tokens = seq_tokens * mask  # Zero out padding tokens
        sum_tokens = masked_tokens.sum(dim=1)  # [batch_size, embedding_dim]
        valid_lengths = mask.sum(dim=1)  # [batch_size, 1]
        return sum_tokens / (valid_lengths + 1e-10)  # Avoid division by zero

    elif pooling == 'max':
        # Max pool over sequence length (exclude CLS and SEP tokens: [1:-1])
        # Use attention_mask to ignore padding tokens
        seq_tokens = hidden_states[:, 1:-1]  # Exclude CLS (index 0) and SEP (index -1)
        mask = attention_mask[:, 1:-1].unsqueeze(-1)  # [batch_size, seq_len-2, 1]
        # Set padding tokens to very negative values so they don't affect max
        masked_tokens = seq_tokens * mask + (1 - mask) * (-1e10)
        return masked_tokens.max(dim=1)[0]  # [batch_size, embedding_dim]

    elif pooling == 'cls':
        # Use CLS token (first token, index 0)
        return hidden_states[:, 0]  # [batch_size, embedding_dim]

    # elif pooling == 'attention':
    #     # Attention-weighted pooling (simplified: use mean of attention weights)
    #     # For now, use mean pooling as approximation
    #     # TODO: Implement proper attention-weighted pooling if needed
    #     seq_tokens = hidden_states[:, 1:-1]
    #     mask = attention_mask[:, 1:-1].unsqueeze(-1)
    #     masked_tokens = seq_tokens * mask
    #     sum_tokens = masked_tokens.sum(dim=1)
    #     valid_lengths = mask.sum(dim=1)
    #     return sum_tokens / (valid_lengths + 1e-10)

    else:
        raise ValueError(f"Unknown pooling method: {pooling}. Choose from 'mean', 'max', 'cls', 'attention'")


def _get_layer_hidden_state(outputs, layer: Union[str, int]) -> torch.Tensor:
    """
    Extract hidden states from a specific layer.

    This requires different model outputs depending on layer selection:
    - layer='last': Uses outputs.last_hidden_state (always available, no output_hidden_states needed)
    - layer='second_last' or layer=int: Uses outputs.hidden_states (requires output_hidden_states=True) (TODO: not tested)

    Args:
        outputs: Model outputs (from EsmModel forward pass)
        layer: Layer selection ('last', 'second_last', or int for specific layer index)

    Returns:
        Hidden states tensor of shape [batch_size, seq_length, embedding_dim]
    """
    if layer == 'last':
        # Last layer is always available via outputs.last_hidden_state
        # No need for output_hidden_states=True
        return outputs.last_hidden_state
    
    elif layer == 'second_last':
        # Access second-to-last layer from hidden_states tuple
        # Requires output_hidden_states=True when calling model
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            return outputs.hidden_states[-2]  # Second-to-last layer
        else:
            # This error should not happen if output_hidden_states is properly enabled
            raise ValueError(
                "hidden_states not available. Cannot access second-to-last layer. "
                "Set output_hidden_states=True when calling model. "
                "This should be automatically enabled when layer='second_last'."
            )

    elif isinstance(layer, int):
        # Specific layer index from hidden_states tuple
        # Requires output_hidden_states=True when calling model
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            if 0 <= layer < len(outputs.hidden_states):
                return outputs.hidden_states[layer]
            else:
                raise ValueError(f"Layer index {layer} out of range [0, {len(outputs.hidden_states)-1}]")
        else:
            raise ValueError(
                "hidden_states not available. Cannot access specific layer. "
                "Set output_hidden_states=True when calling model."
            )

    else:
        raise ValueError(f"Unknown layer selection: {layer}. Choose from 'last', 'second_last', or int")


def get_model_sig(
    model_name: str,
    max_length: int,
    pooling: str = 'mean',
    layer: Union[str, int] = 'last',
    emb_storage_precision: str = 'fp16'
    ) -> str:
    """
    Generate a model signature string for cache key.
    
    The signature includes parameters that affect the embedding values:
    - model_name: Which ESM-2 model was used (e.g., 'facebook/esm2_t33_650M_UR50D')
    - pooling: How hidden states are aggregated ('mean', 'max', 'cls', 'attention')
    - layer: Which layer's hidden states are used ('last', 'second_last', or int)
    - max_length: Effective sequence length (excluding CLS/SEP tokens)
    - emb_storage_precision: Storage precision ('fp16', 'fp32') - affects values due to quantization

    Args:
        model_name: Hugging Face model name (e.g., 'facebook/esm2_t33_650M_UR50D')
        max_length: Maximum sequence length used (including CLS/SEP tokens)
        pooling: Pooling method ('mean', 'max', 'cls', 'attention'). Default: 'mean'
        layer: Layer selection ('last', 'second_last', or int for specific layer). Default: 'last'
        emb_storage_precision: Storage precision ('fp16', 'fp32'). Default: 'fp16' (recommended)

    Returns:
        Model signature string (e.g., 'facebook/esm2_t33_650M_UR50D|mean|last|1022|fp16')
    """
    return f"{model_name}|{pooling}|{layer}|{max_length-2}|{emb_storage_precision}"


def validate_embeddings_metadata(
    embeddings_file: str,
    model_name: str,
    max_length: int,
    pooling: str,
    layer: Union[str, int],
    emb_storage_precision: str
    ) -> bool:
    """
    Validate that existing embeddings file has matching metadata.
    
    Requires master cache format with metadata attributes. Old format files without
    metadata are not supported and will raise an error.

    Args:
        embeddings_file: Path to master HDF5 cache file
        model_name: Expected model name (e.g., 'facebook/esm2_t33_650M_UR50D')
        max_length: Expected max_length (e.g., 1022)
        pooling: Expected pooling method (e.g., 'mean', 'max', 'cls', 'attention')
        layer: Expected layer selection (e.g., 'last', 'second_last', or int)
        emb_storage_precision: Expected storage precision (e.g., 'fp16', 'fp32')

    Returns:
        True if metadata matches or file doesn't exist

    Raises:
        ValueError: If file format is invalid, metadata is missing, or metadata mismatch detected
    """
    embeddings_file = Path(embeddings_file)
    if not embeddings_file.exists():
        return True  # File doesn't exist yet, no validation needed
    
    try:
        with h5py.File(embeddings_file, 'r') as f:
            # Require master cache format
            if 'emb' not in f:
                raise ValueError(
                    f"❌ Old format detected in {embeddings_file}. "
                    "Master cache format required (with 'emb' dataset). "
                    "Please regenerate embeddings using the master cache system."
                )
            
            # Require metadata attributes
            if 'model_name' not in f.attrs:
                raise ValueError(
                    f"❌ Missing metadata in {embeddings_file}. "
                    "This file was created before metadata support. "
                    "Please regenerate embeddings to include metadata."
                )
            
            # Validate metadata
            stored_model_name = f.attrs.get('model_name', '')
            stored_max_length = f.attrs.get('max_length', -1)
            stored_pooling = f.attrs.get('pooling', '')
            stored_layer = f.attrs.get('layer', '')
            stored_precision = f.attrs.get('precision', '')
            
            mismatches = []
            if stored_model_name != model_name:
                mismatches.append(f"model_name: {stored_model_name} != {model_name}")
            if stored_max_length != max_length:
                mismatches.append(f"max_length: {stored_max_length} != {max_length}")
            if stored_pooling != pooling:
                mismatches.append(f"pooling: {stored_pooling} != {pooling}")
            if str(stored_layer) != str(layer):
                mismatches.append(f"layer: {stored_layer} != {layer}")
            if stored_precision != emb_storage_precision:
                mismatches.append(f"emb_storage_precision: {stored_precision} != {emb_storage_precision}")
            
            if mismatches:
                raise ValueError(
                    f"❌ Metadata mismatch in {embeddings_file}:\n  " +
                    "\n  ".join(mismatches) +
                    "\n\nThis indicates embeddings were computed with different parameters. "
                    "Use force_recompute=True to recompute with new parameters."
                )
            return True
    except ValueError:
        # Re-raise ValueError (format errors, missing metadata, mismatches)
        raise
    except Exception as e:
        # For other exceptions (e.g., file corruption), raise with context
        raise ValueError(
            f"❌ Failed to validate embeddings file {embeddings_file}: {e}\n"
            "The file may be corrupted or in an unsupported format."
        ) from e


def _normalize_keys(keys) -> list[str]:
    """
    Normalize HDF5 string keys to Python strings.
    
    HDF5 string datasets may return bytes objects (b'...') or strings depending on
    the HDF5 version and dtype settings. This function ensures consistent string comparison.
    
    Args:
        keys: Array or list of keys from HDF5 (may be bytes or strings)
    
    Returns:
        List of Python strings
    """
    if len(keys) == 0:
        return []
    # Decode bytes to strings if needed
    return [k.decode() if isinstance(k, bytes) else str(k) for k in keys]


def embedding_exists(cache_key: str, embeddings_file: str) -> bool:
    """
    Check if an embedding exists in the master cache.
    
    Args:
        cache_key: Cache key. Format: {seq_hash}::{model_sig}
        embeddings_file: Path to master HDF5 cache file
    
    Returns:
        True if embedding exists, False otherwise
    """
    if not Path(embeddings_file).exists():
        return False
    
    try:
        with h5py.File(embeddings_file, 'r') as f:
            if 'emb_keys' not in f:
                return False
            keys_ds = f['emb_keys']
            keys = keys_ds[:]
            # Normalize keys (decode bytes if needed) for consistent comparison
            keys_normalized = _normalize_keys(keys)
            return cache_key in keys_normalized
    except Exception:
        return False


def save_esm2_embeddings_batch(
    cache_keys: list[str],
    embeddings: np.ndarray,
    embeddings_file: str,
    index_file: Optional[str] = None,
    brc_ids: Optional[list[str]] = None,
    model_name: Optional[str] = None,
    max_length: Optional[int] = None,
    pooling: Optional[str] = None,
    layer: Optional[str] = None,
    emb_storage_precision: str = 'fp16'
    ) -> list[int]:
    """
    Save multiple embeddings to the master cache in batch (efficient).
    
    This function is optimized for batch operations:
    - Opens HDF5 file once
    - Reads existing keys once
    - Resizes dataset once for all new embeddings
    - Updates parquet index once at the end
    
    Args:
        cache_keys: List of cache keys. Format: {seq_hash}::{model_sig}
        embeddings: Embedding vectors (numpy array of shape [n_embeddings, embedding_dim])
        embeddings_file: Path to master HDF5 cache file
        index_file: Optional path to parquet index file
        brc_ids: Optional list of brc_fea_ids for index mapping
        model_name: Optional model name for metadata (stored on first write)
        max_length: Optional max_length for metadata (stored on first write)
        pooling: Optional pooling method for metadata (stored on first write)
        layer: Optional layer selection for metadata (stored on first write)
        emb_storage_precision: Storage precision ('fp16', 'fp32'). Default: 'fp16' (recommended)
    
    Returns:
        List of row indices where embeddings were saved (or existing row if duplicate)
    """
    if len(cache_keys) != len(embeddings):
        raise ValueError(f"Mismatch: {len(cache_keys)} cache_keys but {len(embeddings)} embeddings")
    if brc_ids is not None and len(brc_ids) != len(embeddings):
        raise ValueError(f"Mismatch: {len(brc_ids)} brc_ids but {len(embeddings)} embeddings")
    
    embeddings_file = Path(embeddings_file)
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get dtype mappings for the specified precision
    hdf5_dtype, numpy_dtype = _get_precision_dtypes(emb_storage_precision)
    
    # Open HDF5 file once for all operations
    with h5py.File(embeddings_file, 'a') as f:
        # Create or get embeddings dataset
        is_new_file = 'emb' not in f
        if is_new_file:
            emb_ds = f.create_dataset(
                'emb',
                shape=(0, embeddings.shape[1]),
                maxshape=(None, embeddings.shape[1]),
                dtype=hdf5_dtype,
                chunks=(65536, embeddings.shape[1]),
                compression='lzf'
            )
            
            # Store metadata attributes when creating new file
            if model_name is not None:
                f.attrs['model_name'] = model_name
            if max_length is not None:
                f.attrs['max_length'] = max_length
            if pooling is not None:
                f.attrs['pooling'] = pooling
            if layer is not None:
                f.attrs['layer'] = str(layer)
            f.attrs['precision'] = emb_storage_precision
            f.attrs['version'] = '1.0'
        else:
            emb_ds = f['emb']
        
        # Create or get keys dataset
        if 'emb_keys' not in f:
            keys_ds = f.create_dataset(
                'emb_keys',
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype()
            )
        else:
            keys_ds = f['emb_keys']
        
        # Read all existing keys ONCE
        existing_keys_raw = keys_ds[:]
        # Normalize keys (decode bytes if needed) for consistent comparison
        existing_keys = _normalize_keys(existing_keys_raw)
        existing_keys_set = set(existing_keys) if len(existing_keys) > 0 else set()
        
        # Separate updates vs new embeddings
        update_indices = []  # Indices in cache_keys/embeddings that need updates
        update_rows = []     # Row indices in HDF5 for updates
        new_indices = []     # Indices in cache_keys/embeddings that are new
        new_embeddings = []  # New embeddings to append
        new_keys = []        # New cache keys to append
        
        for i, cache_key in enumerate(cache_keys):
            if cache_key in existing_keys_set:
                # Update existing embedding
                row_idx = existing_keys.index(cache_key)  # Use list.index() instead of np.where for string comparison
                update_indices.append(i)
                update_rows.append(int(row_idx))
            else:
                # New embedding to append
                new_indices.append(i)
                new_embeddings.append(embeddings[i])
                new_keys.append(cache_key)
        
        # Initialize row_indices list
        row_indices = [None] * len(cache_keys)
        
        # Update existing embeddings in place
        for i, row_idx in zip(update_indices, update_rows):
            emb_ds[row_idx] = embeddings[i].astype(numpy_dtype)
            row_indices[i] = row_idx
        
        # Append new embeddings in batch
        if new_embeddings:
            n_new = len(new_embeddings)
            current_size = emb_ds.shape[0]
            new_size = current_size + n_new
            
            # Resize dataset once for all new embeddings
            emb_ds.resize((new_size, embeddings.shape[1]))
            keys_ds.resize((new_size,))
            
            # Write all new embeddings at once
            new_emb_array = np.array(new_embeddings).astype(numpy_dtype)
            emb_ds[current_size:new_size] = new_emb_array
            
            # Write all new keys at once
            keys_ds[current_size:new_size] = new_keys
            
            # Store row indices for new embeddings (in correct order)
            for idx_in_new, orig_idx in enumerate(new_indices):
                row_indices[orig_idx] = current_size + idx_in_new
    
    # Update parquet index once at the end (if provided)
    if index_file and brc_ids is not None:
        index_path = Path(index_file)
        df_new = pd.DataFrame({
            'cache_key': cache_keys,
            'row': row_indices,
            'brc_fea_id': brc_ids
        })
        
        if index_path.exists():
            df = pd.read_parquet(index_path)
            # Remove existing entries for these brc_fea_ids if present
            df = df[~df['brc_fea_id'].isin(brc_ids)]
            df = pd.concat([df, df_new], ignore_index=True)
            df.to_parquet(index_path, index=False)
        else:
            df_new.to_parquet(index_path, index=False)
    
    return row_indices


def get_esm2_embedding_dim(model_ckpt: str) -> int:
    """
    Get embedding dimension for a given ESM-2 model checkpoint.
    
    Args:
        model_ckpt: Hugging Face model checkpoint name (e.g., 'facebook/esm2_t33_650M_UR50D')
    
    Returns:
        Embedding dimension (int)
    
    Raises:
        ValueError: If model checkpoint is not recognized
    """
    # Map of known ESM-2 models to their embedding dimensions
    MODEL_DIMS = {
        'facebook/esm2_t6_8M_UR50D': 320,
        'facebook/esm2_t12_35M_UR50D': 480,
        'facebook/esm2_t30_150M_UR50D': 640,
        'facebook/esm2_t33_650M_UR50D': 1280,
        'facebook/esm2_t36_3B_UR50D': 2560,
        'facebook/esm2_t48_15B_UR50D': 5120,
    }
    
    if model_ckpt not in MODEL_DIMS:
        raise ValueError(
            f"Unknown ESM-2 model: {model_ckpt}. "
            f"Known models: {list(MODEL_DIMS.keys())}"
        )
    
    return MODEL_DIMS[model_ckpt]


def compute_esm2_embeddings(
    sequences: list[str],
    brc_fea_ids: list[str],
    embeddings_file: Optional[str] = None,
    force_recompute: bool=False,
    model_name: str='facebook/esm2_t33_650M_UR50D',
    batch_size: int=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    fine_tuned_model_path: Optional[str]=None,
    max_length: int=ESM2_MAX_RESIDUES + 2,  # +2 for special tokens CLS and SEP
    use_parquet: bool=True,
    pooling: str='mean',
    layer: Union[str, int]='last',
    output_hidden_states: bool=False,
    emb_storage_precision: str='fp16',
    ) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Compute ESM-2 embeddings for a list of sequences with configurable pooling and layer selection.
    Supports master cache with SHA1-based deduplication and incremental appends.
    
    Model Loading:
        - If fine_tuned_model_path is provided: Uses fine-tuned model weights from that path.
          The model_name is still used for loading the tokenizer (tokenizers are typically
          not modified during fine-tuning).
        - If fine_tuned_model_path is None: Uses pretrained model weights from model_name.
        
        Use cases for fine_tuned_model_path:
        - LoRA fine-tuning: After fine-tuning ESM-2 with LoRA adapters on your protein data
        - Full fine-tuning: After fine-tuning the entire ESM-2 model on domain-specific data
        - Domain adaptation: Using ESM-2 weights fine-tuned on viral/bacterial/etc. proteins
          instead of general protein sequences

    Args:
        sequences (list): List of protein sequences.
        brc_fea_ids (list): List of corresponding brc_fea_ids.
        embeddings_file (str, optional): Path to master HDF5 cache file. If None, no caching.
        force_recompute (bool): Force recompute even if embedding exists in cache.
        use_parquet (bool): Use parquet index file for brc_fea_id to row mapping. Default: True.
        model_name (str): Hugging Face ESM-2 model name (default: esm2_t33_650M_UR50D).
                         Always used for tokenizer. Also used for model weights if
                         fine_tuned_model_path is None.
        fine_tuned_model_path (str, optional): Path to fine-tuned ESM-2 model weights.
            If provided, uses these weights instead of pretrained weights from model_name.
            The tokenizer is still loaded from model_name. Examples:
            - LoRA fine-tuned: "path/to/esm2_lora_adapters"
            - Full fine-tuned: "path/to/esm2_fully_finetuned"
            - Domain-specific: "path/to/esm2_viral_finetuned"
        device (str): Device for computation (cuda or cpu).
        batch_size (int): Number of sequences per batch.
        max_length (int): Maximum sequence length (default: ESM2_MAX_RESIDUES + 2). 
        pooling (str): Pooling method ('mean', 'max', 'cls', 'attention'). Default: 'mean'
        layer (str or int): Layer selection ('last', 'second_last', or int for specific layer). Default: 'last'
        output_hidden_states (bool): Whether to output hidden states from all layers.
            - If layer='last': NOT needed (uses outputs.last_hidden_state which is always available)
            - If layer='second_last' or layer=int: REQUIRED (needs outputs.hidden_states tuple)
            Automatically enabled if layer != 'last' and output_hidden_states=False. Default: False
        emb_storage_precision (str): Storage precision ('fp16', 'fp32'). Default: 'fp16' (recommended)

    Returns:
        tuple: (embeddings, valid_ids, failed_ids) where embeddings is a numpy array of
                shape (n_sequences, embedding_dim), valid_ids are successfully processed
                brc_fea_ids, and failed_ids are failed brc_fea_ids.
    
    Example:
        # Using pretrained model (current default)
        embeddings, ids, failed = compute_esm2_embeddings(
            sequences=seqs, brc_fea_ids=ids,
            model_name="facebook/esm2_t33_650M_UR50D"
        )
        
        # Using fine-tuned model (e.g., after LoRA fine-tuning)
        embeddings, ids, failed = compute_esm2_embeddings(
            sequences=seqs, brc_fea_ids=ids,
            model_name="facebook/esm2_t33_650M_UR50D",  # For tokenizer
            fine_tuned_model_path="models/esm2_lora_finetuned"  # Fine-tuned weights
        )
    """
    # Validate metadata if file exists
    if embeddings_file and not force_recompute:
        validate_embeddings_metadata(embeddings_file, model_name, max_length, pooling, layer, emb_storage_precision)
 
    # Generate model signature for cache key
    # 
    # Rationale: model_sig is computed ONCE per run (same for all sequences) and is used to
    # distinguish embeddings computed with different parameters. This allows the same sequence
    # to have multiple embeddings in the cache, each computed with different model/pooling/layer
    # combinations.
    #
    # How it works:
    # - model_sig encodes: model_name, pooling, layer, max_length, precision
    # - Each sequence gets a unique cache_key: {seq_hash}::{model_sig}
    # - Same sequence + different parameters = different cache_key = separate storage
    #
    # Example:
    #   Sequence "MKTAY..." has hash "abc123"
    #   Run 1: mean pooling, last layer → model_sig = "model|mean|last|1022|fp16"
    #          → cache_key = "abc123::model|mean|last|1022|fp16"
    #   Run 2: max pooling, last layer → model_sig = "model|max|last|1022|fp16"
    #          → cache_key = "abc123::model|max|last|1022|fp16"  (different!)
    #
    # Note: The HDF5 file stores metadata (model_name, pooling, layer, precision) as attributes
    # representing ONE set of parameters. validate_embeddings_metadata() enforces that all
    # embeddings in a file use the same parameters, preventing inconsistency.
    model_sig = get_model_sig(
        model_name=model_name, max_length=max_length, pooling=pooling,
        layer=layer, emb_storage_precision=emb_storage_precision
    )

    # Check cache and identify sequences to compute
    cache_keys = []          # Cache keys for all sequences
    to_compute_seqs = []     # Sequences to compute
    to_compute_ids = []      # brc_fea_ids to compute
    to_compute_indices = []  # Original indices of sequences to compute

    process_bar = tqdm(zip(sequences, brc_fea_ids),
        total=len(sequences), desc="Checking cached embeddings"
    )
    for idx, (seq, brc_id) in enumerate (process_bar):
        # Defensive check: raise error for None/empty sequences (should be filtered upstream)
        if seq is None or not isinstance(seq, str) or len(seq.strip()) == 0:
            raise ValueError(
                f"❌ Invalid sequence (None/empty) for brc_fea_id: {brc_id} at index {idx}. "
                f"Sequences should be filtered upstream before calling compute_esm2_embeddings."
            )
        
        seq_hash = hashlib.sha1(seq.encode()).hexdigest() # TODO: What's seq.encode()? What's hexdigest()?
        cache_key = f"{seq_hash}::{model_sig}"
        cache_keys.append(cache_key)

        # Check if we need to compute this embedding
        if embeddings_file and not force_recompute and embedding_exists(cache_key, embeddings_file):
            continue  # Skip - already in cache

        to_compute_seqs.append(seq)
        to_compute_ids.append(brc_id)
        to_compute_indices.append(idx)

    # If all embeddings are cached, return empty results
    if not to_compute_seqs:
        if embeddings_file:
            print("All embeddings already exist in cache; skipping compute.")
        return np.array([]), [], []

    print(f"Computing {len(to_compute_seqs)} embeddings ({len(sequences) - len(to_compute_seqs)} already cached)")

    # Load the tokenizer
    # Both EsmTokenizer and AutoTokenizer would work. AutoTokenizer loads the appropriate
    # tokenizer (flexibile when exploring different models), but since we're focused on
    # ESM model, EsmTokenizer reduces ambiguity.
    # Note: Tokenizer is always loaded from model_name, even when using fine_tuned_model_path,
    # because tokenizers are typically not modified during fine-tuning (LoRA/full fine-tuning
    # only updates model weights, not the tokenizer).
    tokenizer = EsmTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Determine if we need hidden_states from all layers
    # When layer='last': We use outputs.last_hidden_state (always available, no need for output_hidden_states=True)
    # When layer='second_last' or layer=int: We need outputs.hidden_states tuple (requires output_hidden_states=True)
    # Automatically enable output_hidden_states if a non-last layer is requested but not enabled
    if layer != 'last' and not output_hidden_states:
        print(f"⚠️  Warning: layer='{layer}' requires output_hidden_states=True. Enabling it.")
        output_hidden_states = True

    # Load the model weights
    # If fine_tuned_model_path is provided, use fine-tuned weights (e.g., from LoRA
    # adapters or full fine-tuning). Otherwise, use pretrained weights from model_name.
    # This allows using domain-specific or task-specific fine-tuned ESM-2 models for
    # potentially better embedding quality on your specific protein data.
    if fine_tuned_model_path:
        model = EsmModel.from_pretrained(
            fine_tuned_model_path,
            add_pooling_layer=False,
            output_hidden_states=output_hidden_states
        )
    else:
        model = EsmModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            output_hidden_states=output_hidden_states
        )

    assert max_length <= model.config.max_position_embeddings, f'max_length exceeds model limit: {model.config.max_position_embeddings}'

    model.eval()
    model.to(device)
    computed_embeddings = []
    computed_ids = []
    failed_ids = []

    # Process sequences in batches
    process_bar = tqdm(range(0, len(to_compute_seqs), batch_size),
        desc=f'Computing embeddings (by batches of size {batch_size})',
        total=len(to_compute_seqs) // batch_size
    )
    for i in process_bar:
        batch_seqs = to_compute_seqs[i: i+batch_size]
        batch_ids = to_compute_ids[i: i+batch_size]

        try:
            token_encodings = tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in token_encodings.items()}
            input_ids = inputs['input_ids'] # Shape: [batch_size, seq_length]
            attention_mask = inputs['attention_mask'] # Shape: [batch_size, seq_length]

            with torch.no_grad():
                # outputs = model(**inputs)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # last_hidden_state is the output of the final layer (hidden
                # states), containing contextual embeddings for each token in
                # the input sequence. For protein sequences, tokens are amino
                # acids plus special tokens (e.g., <CLS>, <SEP>).

                # Extract hidden states from selected layer
                hidden_states = _get_layer_hidden_state(outputs, layer)
                # Shape: [batch_size, max_seq_length, embedding_dim]

                # Apply custom pooling (not ESM's built-in pooler which uses CLS token only)
                # Note: Original simple implementation was: last_hidden_state[:, 1:-1].mean(dim=1)
                # But that incorrectly includes padding tokens in the mean. Current implementation
                # properly excludes padding tokens using attention_mask.
                emb = _apply_pooling(hidden_states, attention_mask, pooling)
                emb = emb.cpu().numpy()  # Shape: [batch_size, embedding_dim]

            computed_embeddings.append(emb)
            computed_ids.extend(batch_ids)

        except RuntimeError as e:
            print(f'❌ Error processing batch {i//batch_size}: {e}')
            failed_ids.extend(batch_ids)
            continue

    if not computed_embeddings:
        return np.array([]), [], failed_ids

    computed_embeddings = np.vstack(computed_embeddings)

    # Save to master cache if embeddings_file is provided (batch operation for efficiency)
    if embeddings_file:
        index_file = str(Path(embeddings_file).with_suffix('.parquet')) if use_parquet else None
        print(f"Saving {len(computed_ids)} embeddings to master cache...")
        
        # Build cache_keys list for computed embeddings
        computed_cache_keys = [cache_keys[orig_idx] for orig_idx in to_compute_indices]
        
        # Batch save all embeddings at once
        save_esm2_embeddings_batch(
            cache_keys=computed_cache_keys,
            embeddings=computed_embeddings,
            embeddings_file=embeddings_file,
            index_file=index_file,
            brc_ids=computed_ids,
            model_name=model_name,
            max_length=max_length,
            pooling=pooling,
            layer=str(layer),  # Convert to string for consistency
            emb_storage_precision=emb_storage_precision
        )

    return computed_embeddings, computed_ids, failed_ids


def load_esm2_embedding(brc_fea_id: str, embeddings_file: str) -> np.ndarray:
    """
    TODO: Older version of the code (remove after confirming the new code works)
    Load a single ESM-2 embedding from an HDF5 file.

    Args:
        brc_fea_id (str): Protein identifier.
        embeddings_file (str): Path to HDF5 file.

    Returns:
        numpy.ndarray: Embedding vector.

    Raises:
        KeyError: If brc_fea_id is not found in the embeddings file.
    """
    with h5py.File(embeddings_file, 'r') as file:
        if brc_fea_id not in file:
            raise KeyError(f'brc_fea_id {brc_fea_id} not found in {embeddings_file}')
        return np.array(file[brc_fea_id])
