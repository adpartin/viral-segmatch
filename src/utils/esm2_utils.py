from tqdm import tqdm
from typing import Optional
import numpy as np

import torch
from transformers import EsmModel, EsmTokenizer
from transformers import (
    AutoTokenizer,
    EsmConfig,
    EsmModel,
    PreTrainedTokenizer
)

def compute_esm2_embeddings(
    sequences: list[str],
    brc_fea_ids: list[str],
    model_name: str='facebook/esm2_t33_650M_UR50D',
    batch_size: int=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
    fine_tuned_model_path: Optional[str]=None
):
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
        tuple: (embeddings, brc_fea_ids) where embeddings is a numpy array of shape
               (n_sequences, embedding_dim).
    """
    breakpoint()
    # Load the tokenizer
    # tokenizer = EsmTokenizer.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Load the model
    if fine_tuned_model_path:
        model = EsmModel.from_pretrained(fine_tuned_model_path)
    else:
        # model = EsmModel.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name, add_pooling_layer=False)
    
    model.eval()
    model.to(device)
    embeddings = []
    valid_ids = []

    # Process sequences in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc='Computing embeddings'):
        batch_seqs = sequences[i: i+batch_size]
        batch_ids = brc_fea_ids[i: i+batch_size]
        
        try:
            token_encodings = tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(device) for k, v in token_encodings.items()}
            input_ids = inputs['input_ids'] # Shape: (batch_size, seq_length)
            attention_mask = inputs['attention_mask'] # Shape: (batch_size, seq_length)

            with torch.no_grad():
                # outputs = model(**inputs)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask']
                )
                # Mean pool over sequence length (exclude [CLS]/[SEP])
                # seq_output = outputs[0]
                # torch.equal(seq_output, outputs.last_hidden_state)
                # print(f'input_ids:         {input_ids.shape}') # [16, 1024]
                # print(f'attention_mask:    {attention_mask.shape}') # [16, 1024]
                # print(f"last_hidden_state: {outputs['last_hidden_state'].shape}") # [16, 1024, 1280]
                emb = outputs.last_hidden_state[:, 1:-1].mean(dim=1).cpu().numpy()

            embeddings.append(emb)
            valid_ids.extend(batch_ids)

        except RuntimeError as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            continue

    embeddings = np.vstack(embeddings) # TODO. why each embedding is 1280? I expected 1024?
    return embeddings, valid_ids


def load_esm2_embedding(brc_fea_id, embeddings_file):
    """
    Load a single ESM-2 embedding from an HDF5 file.

    Args:
        brc_fea_id (str): Protein identifier.
        embeddings_file (str): Path to HDF5 file.

    Returns:
        numpy.ndarray: Embedding vector.
    """
    with h5py.File(embeddings_file, 'r') as f:
        if brc_fea_id not in f:
            raise KeyError(f"brc_fea_id {brc_fea_id} not found in {embeddings_file}")
        return np.array(f[brc_fea_id])

