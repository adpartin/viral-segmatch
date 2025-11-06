from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

import torch
from transformers import EsmModel, EsmTokenizer

ESM2_MAX_RESIDUES = 1022  # ESM-2 max seq length is 1024 with 2 tokens reserved for CLS, SEP (special tokens)


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
    model_name: str='facebook/esm2_t33_650M_UR50D',
    batch_size: int=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    fine_tuned_model_path: Optional[str]=None,
    max_length: int=ESM2_MAX_RESIDUES + 2  # +2 for special tokens CLS and SEP
    ) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Compute mean-pooled ESM-2 embeddings for a list of sequences.

    Args:
        sequences (list): List of protein sequences.
        brc_fea_ids (list): List of corresponding brc_fea_ids.
        model_name (str): Hugging Face ESM-2 model name (default: esm2_t33_650M_UR50D).
        batch_size (int): Number of sequences per batch.
        device (str): Device for computation (cuda or cpu).
        fine_tuned_model_path (str, optional): Path to fine-tuned ESM-2 model.

    Returns:
        tuple: (embeddings, brc_fea_ids) where embeddings is a numpy array of
                shape (n_sequences, embedding_dim).
    """
    # breakpoint()
    # Load the tokenizer
    # Both EsmTokenizer and AutoTokenizer would work. AutoTokenizer loads the
    # appropriate tokenizer (flexibile when exploring different models), but
    # since we're focused on ESM model, EsmTokenizer reduces ambiguity.
    tokenizer = EsmTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Load the model
    if fine_tuned_model_path:
        model = EsmModel.from_pretrained(fine_tuned_model_path)
    else:
        # model = EsmModel.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name, add_pooling_layer=False)

    assert max_length <= model.config.max_position_embeddings, f'max_length exceeds model limit: {model.config.max_position_embeddings}'

    model.eval()
    model.to(device)
    embeddings = []
    valid_ids = []
    failed_ids = []

    # Process sequences in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc='Computing ESM-2 embeddings'):
        batch_seqs = sequences[i: i+batch_size]
        batch_ids = brc_fea_ids[i: i+batch_size]
        
        try:
            token_encodings = tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in token_encodings.items()}
            input_ids = inputs['input_ids'] # Shape: (batch_size, seq_length)
            attention_mask = inputs['attention_mask'] # Shape: (batch_size, seq_length)

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
                # Shape: [batch_size, max_seq_length, embedding_dim]
                last_hidden_state = outputs.last_hidden_state

                # seq_output = outputs[0]
                # torch.equal(seq_output, last_hidden_state)
                # print(f'input_ids:         {input_ids.shape}') # [16, 1024]
                # print(f'attention_mask:    {attention_mask.shape}') # [16, 1024]
                # print(f'last_hidden_state: {last_hidden_state.shape}') # [16, 1024, 1280]

                # Mean pool over sequence length (the range 1:-1 excludes
                # tokens [CLS] and [SEP])
                emb = last_hidden_state[:, 1:-1].mean(dim=1).cpu().numpy() # [batch_size, embedding_dim]

            embeddings.append(emb)
            valid_ids.extend(batch_ids)

        except RuntimeError as e:
            print(f'Error processing batch {i//batch_size}: {e}')
            failed_ids.extend(batch_ids)
            continue

    embeddings = np.vstack(embeddings)
    return embeddings, valid_ids, failed_ids


def compute_sliding_window_embedding(
    seq,
    tokenizer,
    model,
    max_length=1024,
    window_size=1022,
    overlap=500):
    """ 
    Compute sliding window embeddings for a protein sequence using ESM-2.

    TODO. Provided by Grok. Never tested nor analyzed.
    """
    embeddings = []
    step = window_size - overlap
    for start in range(0, len(seq), step):
        end = min(start + window_size, len(seq))
        sub_seq = seq[start:end]
        inputs = tokenizer([sub_seq], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state[:, 1:-1].mean(dim=1).cpu().numpy()
        embeddings.append(emb)
    return np.mean(embeddings, axis=0)


def load_esm2_embedding(brc_fea_id: str, embeddings_file: str) -> np.ndarray:
    """
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


def load_esm2_embeddings_bulk(brc_fea_ids: list[str], embeddings_file: str) -> tuple[np.ndarray, list[str]]:
    """
    Load multiple ESM-2 embeddings from an HDF5 file efficiently (single file open).

    Args:
        brc_fea_ids (list): List of protein identifiers.
        embeddings_file (str): Path to HDF5 file.

    Returns:
        tuple: (embeddings_array, valid_ids) where embeddings_array is numpy array
               of shape (n_valid_embeddings, embedding_dim) and valid_ids are the
               successfully loaded brc_fea_ids.
    """
    embeddings = []
    valid_ids = []
    
    with h5py.File(embeddings_file, 'r') as file:
        for brc_id in brc_fea_ids:
            if brc_id in file:
                embeddings.append(file[brc_id][:])
                valid_ids.append(brc_id)
            else:
                print(f"   ⚠️  Warning: brc_fea_id {brc_id} not found in {embeddings_file}")
    
    return np.array(embeddings), valid_ids
