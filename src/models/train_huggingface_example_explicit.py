"""
Got this notebook from:
https://github.com/huggingface/transformers/tree/main/notebooks

Train a binary classifier for protein sequences using huggingface ESM-2
model using explicit training loop.
"""

import os
import random
import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    EsmConfig, # For type hinting
    EsmModel,
    PreTrainedTokenizer # For type hinting
)
from evaluate import load
from transformers.utils import logging
logging.get_verbosity()  # Shows debug output during downloads

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.torch_utils import determine_device


# class ProteinDataset(Dataset):
#     """Dataset for protein classification."""
#     def __init__(self, sequences: list[str], labels: list[int]):
#         self.sequences = sequences
#         self.labels = torch.tensor(labels, dtype=torch.long)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
#         return self.sequences[idx], self.labels[idx]


class TokenizedProteinDataset(Dataset):
    """ Dataset for protein classification with tokenized sequences.
    
    This dataset tokenizes protein sequences once during initialization, which
    improves efficiency by avoiding repeated tokenization.

    Args:
        tokenizer: Pre-trained tokenizer for protein sequences
        sequences: List of protein sequences to process
        labels: Optional list of classification labels (0 or 1)

    Examples:
        >>> tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        >>> dataset = TokenizedProteinDataset(
        ...     tokenizer=tokenizer,
        ...     sequences=['MAAGTKLV', 'MKTLLVV'],
        ...     labels=[0, 1]
        ... )
        >>> batch = dataset[0]
        >>> print(batch.keys())
        dict_keys(['input_ids', 'attention_mask', 'labels'])
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        sequences: list[str],
        labels: Optional[list[int]] = None
        ):
        # Validate inputs
        if not sequences:
            raise ValueError('sequences cannot be empty')

        if labels is not None and len(sequences) != len(labels):
            raise ValueError(
                f'Length mismatch: {len(sequences)} sequences vs '
                f'{len(labels)} labels'
            )

        # Initialize attributes
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

        # Tokenize sequences only once during dataset creation
        self.token_encodings = self.tokenizer(
            sequences, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.token_encodings['input_ids'])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """ Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dict containing:
                - input_ids: Tokenized sequence
                - attention_mask: Attention mask for padding
                - labels: Classification label (if provided during initialization)
        """
        # Each sample returns input_ids, attention_mask, and labels
        item = {k: v[idx] for k, v in self.token_encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


class EsmClassificationHead(nn.Module):
    """ Classification head for sentence-level classification tasks.
    This is how the the classification head is implemented in
    EsmForSequenceClassification.

    Architecture:
        1. Extract [CLS] token's hidden states representation
        2. Project through a single dense layer (hidden_size → hidden_size)
        3. Apply dropout, tanh, and final classification layer

    Args:
        config: ESM model config containing:
            hidden_size (int): Dimension of hidden representations
            hidden_dropout_prob (float): Dropout probability
            num_labels (int): Number of classification labels
            
    Reference:
        Check EsmClassificationHead() class in:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py
    """

    def __init__(self, config: EsmConfig):
        super().__init__()

        if not hasattr(config, 'hidden_size'):
            raise ValueError('Config must specify hidden_size')
 
        # Layers
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Forward pass of the classification head.

        Args:
            features: Hidden states from ESM model
                Shape: (batch_size, sequence_length, hidden_size)
            **kwargs: Additional arguments (not used)

        Returns:
            Logits for each class
                Shape: (batch_size, num_labels)
        """
    
        # Extract [CLS] token representation (first token)
        # features: (batch_size, sequence_length, hidden_size) [8, 502, 320]
        # cls_output: (batch_size, hidden_size) [8, 320]
        # TODO. Does this extract the first embedding dimension of each token of the entire batch?
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        cls_output = features[:, 0, :]  # [8, 320]
    
        # Project through classification layers
        x = self.dropout(cls_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x) # (batch_size, num_labels) [8, 2]
        return logits


class EsmProteinClassifier(nn.Module):
    """ ESM-based protein sequence classifier.

    Architecture:
        1. ESM encoder (without pooling layer)
        2. Classification head for sequence-level prediction

    Args:
        model_ckpt: Name or path of pre-trained ESM model
        num_labels: Number of classification labels (default: 2)

    Reference:
        Check EsmForSequenceClassification() class in:
        github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/esm/modeling_esm.py#L1066
    """

    def __init__(self, model_ckpt: str, num_labels: int=2):
        super().__init__()
        
        # Load pre-trained ESM model without pooling layer
        # (matches EsmForSequenceClassification behavior)
        self.esm = EsmModel.from_pretrained(model_ckpt, add_pooling_layer=False)

        # Configure classification head
        self.config = self.esm.config
        self.config.num_labels = num_labels
        self.classifier = EsmClassificationHead(self.config)

        # print(self.config) # print model config
        # print(self.esm)    # print model architecture

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """ Forward pass through the classifier.

        Args:
            input_ids: Tokenized protein sequences
                Shape: (batch_size, sequence_length)
            attention_mask: Mask to avoid performing attention on padding tokens
                Shape: (batch_size, sequence_length)

        Returns:
            Classification logits
                Shape: (batch_size, num_labels)

        github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/esm/modeling_esm.py#L1103
        """

        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)

        # Pass last hidden state through classifier
        # seq_output = outputs[0]
        # torch.equal(seq_output, outputs.last_hidden_state)
        logits = self.classifier(outputs.last_hidden_state)

        # print(f'input_ids:         {input_ids.shape}') # [8, 502]
        # print(f'attention_mask:    {attention_mask.shape}') # [8, 502]
        # print(f"last_hidden_state: {outputs['last_hidden_state'].shape}") # [8, 502, 320]
        # print(f"logits:            {logits.shape}") # [8, 2]

        return logits

# Define compute_metrics function
metric = load('accuracy')
def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# ===================================
# Config
# ===================================
SEED = 42
TASK_NAME = 'huggingface_example'
CUDA_NAME = 'cuda:4'  # Specify GPU device
MODEL_CKPT = 'facebook/esm2_t6_8M_UR50D'  # Choose ESM-2 model
# MODEL_CKPT = 'facebook/esm2_t12_35M_UR50D'  # Choose ESM-2 model

# Define paths
main_data_dir = project_root / 'data'
datasets_dir = main_data_dir / 'datasets' / TASK_NAME
model_dir = project_root / 'models'
output_dir = model_dir / (TASK_NAME + '_explicit')
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

model_name = MODEL_CKPT.split('/')[-1]

# Load data
# train_seqs = np.load(data_dir / 'train_sequences.npy', allow_pickle=True).tolist()
# test_seqs = np.load(data_dir / 'test_sequences.npy', allow_pickle=True).tolist()
# train_labels = np.load(data_dir / 'train_labels.npy', allow_pickle=True)
# test_labels = np.load(data_dir / 'test_labels.npy', allow_pickle=True)
train_seqs = np.load(datasets_dir / 'train_sequences.npy', allow_pickle=True).tolist()
test_seqs = np.load(datasets_dir / 'test_sequences.npy', allow_pickle=True).tolist()
train_labels = np.load(datasets_dir / 'train_labels.npy', allow_pickle=True)
test_labels = np.load(datasets_dir / 'test_labels.npy', allow_pickle=True)

# Load tokenizer
"""
AutoTokenizer.from_pretrained() downloads files related to the tokenizer and
saves them in a local cache directory (saved in .cache/huggingface/hub)
"""
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CKPT)
# print(tokenizer.cache_dir) # AttributeError: EsmTokenizer has no attribute cache_dir

# Create dataset
# train_dataset = ProteinDataset(train_seqs, train_labels)
# test_dataset = ProteinDataset(test_seqs, test_labels)
train_dataset = TokenizedProteinDataset(tokenizer, train_seqs, train_labels)
test_dataset  = TokenizedProteinDataset(tokenizer, test_seqs,  test_labels)

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # TODO: shuffle=False
test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print('Train_dataset: ', len(train_dataset))
print('Test_dataset:  ', len(test_dataset))

### Explore dataloader
# TODO. How shape[1] dimension which is sequence_length of 502 was determined?
# batch = next(iter(train_dataloader))
# print('\nDataloader:')
# print(f'dataloader batch: {batch.keys()}')
# print(f"input_ids:        {batch['input_ids'].shape}")
# print(f"attention_mask:   {batch['attention_mask'].shape}")
# print(f"labels:           {batch['labels'].shape}")
# print(f"batch['input_ids'][0]:\n{batch['input_ids'][0]}")
# print(f"batch['attention_mask'][0]:\n{batch['attention_mask'][0]}")
# print(f"batch['input_ids'][1]: {batch['input_ids'][1]}")
# print(f"batch['attention_mask'][1]: {batch['attention_mask'][1]}")

### Manually tokenize a sequence and compare with dataset
# manual_token = tokenizer(
#     train_sequences[0],
#     return_tensors='pt',
#     padding=True,
#     truncation=True
# )
# data_sample = train_dataset[0]
# assert torch.equal(
#     manual_token['input_ids'].squeeze(0),
#     data_sample['input_ids'][:manual_token['input_ids'].size(1)]
# ), 'Tokenization mismatch!'
# assert torch.equal(
#     manual_token['attention_mask'].squeeze(0),
#     data_sample['attention_mask'][:manual_token['attention_mask'].size(1)]
# ), 'Tokenization mismatch!'
# print('Tokenization is consistent!')


# ====================================
# Load ESM-2 Model
# ====================================
"""
Note! You can print the returned object and observe the config.
When you load a pre-trained model like facebook/esm2_t12_35M_UR50D using
EsmConfig.from_pretrained(MODEL_CKPT), you're not getting the default configs
Instead, you're getting the specific configuration that was used when that
particular model was trained.
You'll also see the param "architectures": ["EsmForMaskedLM"]. This indicates
the primary task the pre-trained model was trained for, which in this case of
esm2_t12_35M_UR50D is Masked Language Modeling (MLM).
"""
# Initialize classifier and move it to device
# breakpoint()
device = determine_device(CUDA_NAME)
classifier = EsmProteinClassifier(MODEL_CKPT, num_labels=2)
classifier.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY)


# ====================================
# Train Model
# ====================================
# breakpoint()
print('\nStart training.')

best_loss = float('inf')
best_model_state = None

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    classifier.train()
    total_loss = 0.0

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}',
        dynamic_ncols=True, leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = classifier(**inputs)

        # Backward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar with loss value
        progress_bar.set_postfix(loss=f'{loss.item():.5f}')

    avg_train_loss = total_loss / len(train_dataloader)

    # # Validation step
    # classifier.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for batch in val_dataloader:
    #         inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
    #         labels = batch['labels'].to(device)
    #         outputs = classifier(**batch)
    #         loss = outputs.loss
    #         val_loss += loss.item()

    # avg_val_loss = val_loss / len(val_dataloader)

    # # Save best model based on validation loss
    # if avg_val_loss < best_loss:
    #     best_loss = avg_val_loss
    #     best_model_state = model.state_dict()  # Save model parameters

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader):.4f}')

print(f'\nRuntime {(time.time() - start_time)/60} minutes\n')
del batch, inputs, labels, outputs, loss

# Restore best model state at the end
if best_model_state:
    classifier.load_state_dict(best_model_state)
    print('Best model restored based on validation loss.')


# ===================================
# Evaluate model on test set
# ===================================
# breakpoint()
test_preds, all_test_labels = [], []

classifier.eval()
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Evaluating'):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        # Forward pass (with esm_model and classifier)
        logits = classifier(**inputs)  # under torch.no_grad()
        preds = torch.argmax(logits, dim=1)

        # Store preds and labels
        test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

# Compute accuracy
accuracy = metric.compute(predictions=test_preds, references=all_test_labels)
print(f"Test Accuracy: {accuracy['accuracy']:.4f}")

# Print detailed classification report
print('\nClassification Report:')
print(classification_report(all_test_labels, test_preds, 
                            target_names=['Cytosolic', 'Membrane'])
)

# Save raw predictions
test_res_df = pd.DataFrame({
    'seqs': test_seqs,
    'labels': all_test_labels,
    'preds': test_preds
})
test_res_df.to_csv(output_dir / 'test_predicted.csv', index=False)

# Save model
model_path = output_dir / 'model.pt'
tokenizer_path = output_dir / 'tokenizer'
torch.save(classifier, model_path)
torch.save(classifier.state_dict(), output_dir / 'model_state_dict.pth')
tokenizer.save_pretrained(tokenizer_path)
print(f'Model saved: {model_path}')

print('Done.')
