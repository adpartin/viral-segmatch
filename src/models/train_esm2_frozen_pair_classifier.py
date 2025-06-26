"""
Modeling approach:
1. Binary classifier: Do (protein A, protein B) come from same isolate?
2. Start with frozen ESM-2 embeddings + MLP baseline
"""

import sys
import random
from pathlib import Path

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.timer_utils import Timer
from src.utils.esm2_utils import load_esm2_embedding
from src.utils.torch_utils import determine_device

# Config
SEED = 42
TASK_NAME = 'segment_pair_classifier'
VIRUS_NAME = 'bunya'
DATA_VERSION = 'April_2025'
CUDA_NAME = 'cuda:4'  # Specify GPU device

# MODEL_CKPT = 'facebook/esm2_t6_8M_UR50D'    # embedding dim: 320D
# MODEL_CKPT = 'facebook/esm2_t12_35M_UR50D'  # embedding dim: 480D
# MODEL_CKPT = 'facebook/esm2_t30_150M_UR50D' # embedding dim: 640D
MODEL_CKPT = 'facebook/esm2_t33_650M_UR50D'   # embedding dim: 1280D
# MODEL_CKPT = 'facebook/esm2_t36_3B_UR50D'   # embedding dim: 2560D
# MODEL_CKPT = 'facebook/esm2_t48_15B_UR50D'  # embedding dim: 5120D

# Define paths
main_data_dir = project_root / 'data'
dataset_dir = main_data_dir / 'datasets' / VIRUS_NAME / DATA_VERSION / TASK_NAME
embeddings_dir = main_data_dir / 'embeddings' / VIRUS_NAME / DATA_VERSION
output_dir = project_root / 'models' / VIRUS_NAME / DATA_VERSION / TASK_NAME
output_dir.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Hyperparameters
batch_size = 16
learning_rate = 1e-3
epochs = 50
hidden_dims = [512, 256, 64]
dropout = 0.3
patience = 5

model_name = MODEL_CKPT.split('/')[-1]
device = determine_device(CUDA_NAME)

print(f'\ndataset_dir:    {dataset_dir}')
print(f'embeddings_dir: {embeddings_dir}')
print(f'output_dir:     {output_dir}')


class SegmentPairDataset(Dataset):
    """
    Dataset for segment pairs with ESM-2 embeddings.
    """
    def __init__(self, pairs_df: pd.DataFrame, embeddings_file: str):
        """
        Args:
            pairs_df (pd.DataFrame): DataFrame containing segment pairs with
                columns ['brc_a', 'brc_b', 'label'].
            embeddings_file (str): Path to the ESM-2 embeddings file.
        """
        self.pairs = pairs_df
        self.embeddings_file = embeddings_file
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the concatenated ESM-2 embeddings for a segment pair and its label.
        """
        # breakpoint()
        row = self.pairs.iloc[idx]
        emb_a = load_esm2_embedding(row['brc_a'], self.embeddings_file)
        emb_b = load_esm2_embedding(row['brc_b'], self.embeddings_file)
        emb = np.concatenate([emb_a, emb_b])
        emb = torch.tensor(emb, dtype=torch.float)
        label = torch.tensor(row['label'], dtype=torch.float)
        return emb, label


class MLPClassifier(nn.Module):
    """
    MLP classifier for segment pairs.
    """
    def __init__(
        self,
        input_dim: int=2560,
        hidden_dims: list[int]=[512, 256, 64],
        # num_labels: int=2,
        dropout: float=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # print(self.mlp) # print model architecture
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # breakpoint()
        # print(f'Input shape: {x.shape}')
        return self.mlp(x)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    output_dir
    ) -> str:
    """
    Train the MLP classifier with early stopping based on validation loss.
    """
    # breakpoint()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = output_dir / 'best_model.pt'

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}',
            dynamic_ncols=True, leave=False)
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(batch_x).squeeze()
            # Backward pass
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0) # why multiplied by batch_x.size(0)?
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds) > 0.5
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_preds)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break

    return best_model_path


def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    test_pairs
    ) -> None:
    """
    Evaluate the model on the test set and compute metrics.
    """
    # breakpoint()
    model.eval()
    test_loss = 0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            test_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_preds = np.array(test_preds) > 0.5
    test_f1 = f1_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_preds)

    # Same-function negative accuracy
    same_func_neg = test_pairs[(test_pairs['label'] == 0) & (test_pairs['func_a'] == test_pairs['func_b'])]
    same_func_idx = same_func_neg.index
    same_func_preds = test_preds[same_func_idx]
    same_func_labels = np.array(test_labels)[same_func_idx]
    same_func_acc = (same_func_preds == same_func_labels).mean()

    print(f'Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}, Same-Function Neg Acc: {same_func_acc:.4f}')


total_timer = Timer()
print('\nLoad pair datasets.')
train_pairs = pd.read_csv(dataset_dir / 'train_pairs.csv')
val_pairs   = pd.read_csv(dataset_dir / 'val_pairs.csv')
test_pairs  = pd.read_csv(dataset_dir / 'test_pairs.csv')
embeddings_file = embeddings_dir / 'esm2_embeddings.h5'

# Validate embeddings
print('\nValidate embeddings.')
with h5py.File(embeddings_file, 'r') as f:
    emb_ids = set(f.keys())
    for df in [train_pairs, val_pairs, test_pairs]:
        assert set(df['brc_a']).union(set(df['brc_b'])).issubset(emb_ids), 'Missing embeddings'

# Create datasets
train_dataset = SegmentPairDataset(train_pairs, embeddings_file)
val_dataset   = SegmentPairDataset(val_pairs, embeddings_file)
test_dataset  = SegmentPairDataset(test_pairs, embeddings_file)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
batch = next(iter(train_loader))
mlp_input_dim = batch[0].shape[1] # 2 * embedding_dim = 2 * 1280 (embed_dim can be obtained from 'esm2_embeddings.h5')
model = MLPClassifier(input_dim=mlp_input_dim, hidden_dims=hidden_dims, dropout=dropout)
model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
print('\nTrain model')
best_model_path = train_model(model, train_loader, val_loader, criterion,
    optimizer, device, epochs, patience, output_dir
)

# Evaluate
# breakpoint()
print('\nEvaluate model')
model.load_state_dict(torch.load(best_model_path))
evaluate_model(model, test_loader, criterion, device, test_pairs)

# Save raw predictions
# TODO
# breakpoint()
print('\nSave raw predictions.')

total_timer.display_timer()
print('Done.')
