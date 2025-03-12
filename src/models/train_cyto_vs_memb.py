"""
Got this notebook from:
https://github.com/huggingface/transformers/tree/main/notebooks

Train a binary classifier for protein sequences using huggingface's ESM-2
model using explicit training loop.
"""
import os
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel

from evaluate import load

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.torch_utils import determine_device


# ===================================
# Config
# ===================================
TASK_NAME = "cyto_vs_memb"
CUDA_NAME = "cuda:7"  # Specify GPU device
# MODEL_CKPT = "facebook/esm2_t6_8M_UR50D"  # Choose ESM-2 model
MODEL_CKPT = "facebook/esm2_t12_35M_UR50D"  # Choose ESM-2 model
SEED = 42  # For reproducibility

BATCH_SIZE = 8  # Increase batch size for efficiency
LEARNING_RATE = 1e-4  # Optimizer learning rate
# LEARNING_RATE = 2e-5  # Optimizer learning rate
NUM_EPOCHS = 3  # Number of training epochs

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Define paths
filepath = Path(__file__).resolve().parent
data_dir = filepath / '../../data' / TASK_NAME # config!
model_dir = filepath / '../../models' # config!
output_dir = model_dir / TASK_NAME

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


# =========================================
# Define Dataset for Classification
# =========================================
class ProteinDataset(Dataset):
    """
    Custom PyTorch Dataset for binary protein classification.

    Args:
        sequences (list): List of protein sequences.
        labels (list): Corresponding labels (0 or 1).
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =========================================
# Load Preprocessed Data
# =========================================
print(f'Loading preprocessed data from {data_dir} ...')

train_sequences = np.load(data_dir / 'train_sequences.npy', allow_pickle=True).tolist()
test_sequences = np.load(data_dir / 'test_sequences.npy', allow_pickle=True).tolist()
train_labels = np.load(data_dir / 'train_labels.npy', allow_pickle=True)
test_labels = np.load(data_dir / 'test_labels.npy', allow_pickle=True)

# Create dataset
train_dataset = ProteinDataset(train_sequences, train_labels)
test_dataset = ProteinDataset(test_sequences, test_labels)

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Train_dataset: ', len(train_dataset))
print('Test_dataset:  ', len(test_dataset))
print(type(train_dataset))


# ====================================
# Load ESM-2 Model & Tokenizer
# ====================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
esm_model = EsmModel.from_pretrained(MODEL_CKPT)
# esm_model.eval()  # Set to evaluation mode

# Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = determine_device(CUDA_NAME) # config!
esm_model = esm_model.to(device)


# ===========================================
# Define Binary Classification Model
# ===========================================
class ProteinClassifier(nn.Module):
    """
    A simple binary classifier that takes ESM-2 protein sequence embeddings 
    and predicts a binary label (0 or 1).
    
    Args:
        hidden_dim (int): Size of the embedding output from ESM-2.
    """
    def __init__(self, hidden_dim: int):
        super(ProteinClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        
        Args:
            embeddings (torch.Tensor): Protein sequence embeddings from ESM-2.
        
        Returns:
            torch.Tensor: Probability of class 1.
        """
        return self.fc(embeddings)


# ===================================
# Define Training Components
# ===================================
# Get hidden dimension from the model's config
hidden_dim = esm_model.config.hidden_size
classifier = ProteinClassifier(hidden_dim).to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
# optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE, weight_decay=0.01)


# ====================================
# Train Model
# ====================================
print('Starting training ...')
for epoch in range(NUM_EPOCHS):
    print('Start epoch:', epoch+1)
    total_loss = 0.0
    classifier.train()

    # for seqs, labels in train_dataloader:
    # for seqs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
    # progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', postfix={"loss": 0.0}) # create the instance.
    
    # Use tqdm with `dynamic_ncols=True` and `leave=False` to avoid new lines
    progress_bar = tqdm(train_dataloader, desc=f'Training', dynamic_ncols=True, leave=False)

    for seqs, labels in progress_bar:
        optimizer.zero_grad()

        # # Tokenize the sequences
        # inputs = tokenizer(seqs).to(device) # protain_language_model
        # # inputs = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True).to(device)
        # print(f'len(seqs):   {len(seqs)}')
        # print(f'len(labels): {len(labels)}')
        # print(f'len(inputs): {len(inputs)}')
        # print(f'inputs.keys(): {inputs.keys()}')
        # print(f'seqs[0]: {seqs[0]}')
        # print(f'inputs["input_ids"][0]: {inputs["input_ids"][0]}')
        # print(f'seqs[1]: {seqs[1]}')
        # print(f'inputs["input_ids"][1]: {inputs["input_ids"][1]}')

        # Tokenize the sequences
        inputs = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True)
        # inputs = tokenizer(seqs)

        # Move each tensor in the dictionary to the correct device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get embeddings from ESM-2
        with torch.no_grad():
            embeddings = esm_model(**inputs).last_hidden_state[:, 0, :]  # CLS token representation
        # print(f'type(embeddings): {type(embeddings)}')
        # print(f'embeddings.shape: {embeddings.shape}')

        # Pass embeddings through the classifier
        outputs = classifier(embeddings).squeeze()

        # Compute loss
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar with loss value
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')


    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader):.4f}")


# ===================================
# Evaluate Model
# ===================================
breakpoint()
metric = load('accuracy')
classifier.eval()
all_preds, all_labels = [], []

print('Evaluating on test set ...')
with torch.no_grad():
    for seqs, labels in tqdm(test_dataloader, desc='Evaluating'):
        # Tokenize sequences
        inputs = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get embeddings from ESM-2
        embeddings = esm_model(**inputs).last_hidden_state[:, 0, :]

        # Classifier predictions
        outputs = classifier(embeddings).squeeze()
        preds = (outputs > 0.5).float()  # Convert probs to binary (0 or 1)

        # Store preds and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute accuracy
accuracy = metric.compute(predictions=all_preds, references=all_labels)
print(f"Test Accuracy: {accuracy['accuracy']:.4f}")

# Save raw model predictions
np.save(output_dir / 'test_predictions.npy', np.array(all_preds))
np.save(output_dir / 'test_labels.npy', np.array(all_labels))

print(f'Predictions saved to {output_dir}')


# ===================================
# Save Model
# ===================================
torch.save(classifier.state_dict(), output_dir / f'esm2_{TASK_NAME}_binary_classifier.pth')
print('Model saved.')
