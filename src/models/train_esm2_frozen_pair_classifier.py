"""
Modeling approach:
1. Binary classifier: Do (protein A, protein B) come from same isolate?
2. Start with frozen ESM-2 embeddings + MLP baseline
"""
import argparse
import sys
import random
from pathlib import Path

import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# print(f'project_root: {project_root}')

from src.utils.timer_utils import Timer
from src.utils.config_hydra import get_virus_config_hydra, print_config_summary
from src.utils.seed_utils import resolve_process_seed, set_deterministic_seeds
from src.utils.path_utils import resolve_run_suffix, build_training_paths
from src.utils.torch_utils import determine_device
from src.utils.esm2_utils import load_esm2_embedding

total_timer = Timer()


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    output_dir,
    early_stopping_metric='loss'
    ) -> str:
    """
    Train the MLP classifier with early stopping based on a configurable metric.
    
    Args:
        early_stopping_metric: Metric to use for early stopping ('loss', 'f1', 'auc')
            - 'loss': Lower is better (default, backward compatible)
            - 'f1': Higher is better
            - 'auc': Higher is better
    
    Returns:
        best_model_path: Path to best model checkpoint
    """
    patience_counter = 0
    best_model_path = output_dir / 'best_model.pt'
    
    # Initialize best metric tracking based on metric type
    if early_stopping_metric == 'loss':
        best_metric_value = float('inf')
        is_higher_better = False
    elif early_stopping_metric in ['f1', 'auc']:
        best_metric_value = -1.0
        is_higher_better = True
    else:
        raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}. Choose from 'loss', 'f1', 'auc'")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}',
            dynamic_ncols=True, leave=False)
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            # Forward pass
            preds = model(batch_x).squeeze() 
            # Backward pass
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_preds, val_probs, val_labels = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).squeeze()
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_probs.extend(torch.sigmoid(preds).cpu().numpy())
                val_preds.extend((torch.sigmoid(preds) > 0.5).float().cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds)
        val_probs = np.array(val_probs)
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)

        # Select metric value for early stopping
        if early_stopping_metric == 'loss':
            current_metric_value = val_loss
            metric_display = f'Val Loss: {val_loss:.4f}'
        elif early_stopping_metric == 'f1':
            current_metric_value = val_f1
            metric_display = f'Val F1: {val_f1:.4f}'
        elif early_stopping_metric == 'auc':
            current_metric_value = val_auc
            metric_display = f'Val AUC: {val_auc:.4f}'

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '\
              f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, '\
              f'Val AUC: {val_auc:.4f} [{early_stopping_metric.upper()}: {current_metric_value:.4f}]')

        # Check if current metric is better than best
        is_better = False
        if is_higher_better:
            is_better = current_metric_value > best_metric_value
        else:
            is_better = current_metric_value < best_metric_value

        if is_better:
            best_metric_value = current_metric_value
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'  ✓ New best {early_stopping_metric}: {best_metric_value:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered (no improvement in {early_stopping_metric} for {patience} epochs).')
                break

    return best_model_path


def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    test_pairs
    ) -> pd.DataFrame:
    """
    Evaluate the model on the test set and compute metrics.
    """
    model.eval()
    test_loss = 0
    test_preds, test_probs, test_labels = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x).squeeze()
            loss = criterion(preds, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            test_probs.extend(torch.sigmoid(preds).cpu().numpy())
            test_preds.extend((torch.sigmoid(preds) > 0.5).float().cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_f1 = f1_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)

    # Classification report
    print('\nClassification Report:')
    print(classification_report(test_labels, test_preds, target_names=['Negative', 'Positive']))

    # Create test_res_df
    test_res_df = test_pairs.copy()
    test_res_df['pred_label'] = test_preds
    test_res_df['pred_prob'] = test_probs

    print(f'Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}')
    return test_res_df


# Parser
parser = argparse.ArgumentParser(description='Train ESM-2 frozen pair classifier')
parser.add_argument(
    '--config_bundle',
    type=str, default=None,
    help='Config bundle to use (e.g., flu_a, bunya).'
)
parser.add_argument(
    '--cuda_name', '-c',
    type=str, default='cuda:7',
    help='CUDA device to use (default: cuda:7)'
)
parser.add_argument(
    '--dataset_dir',
    type=str, default=None,
    help='Path to directory containing train_pairs.csv, val_pairs.csv, test_pairs.csv'
)
parser.add_argument(
    '--embeddings_dir',
    type=str, default=None,
    help='Path to directory containing esm2_embeddings.h5'
)
parser.add_argument(
    '--output_dir',
    type=str, default=None,
    help='Path to output directory for models'
)
args = parser.parse_args()

# Load config
config_path = str(project_root / 'conf')  # Pass the config path explicitly
config_bundle = args.config_bundle
if config_bundle is None:
    raise ValueError("❌ Must provide --config_bundle")
config = get_virus_config_hydra(config_bundle, config_path=config_path)
print_config_summary(config)

# Extract config values
VIRUS_NAME = config.virus.virus_name
DATA_VERSION = config.virus.data_version
RANDOM_SEED = resolve_process_seed(config, 'training')
# TASK_NAME = config.dataset.task_name
MODEL_CKPT = config.embeddings.model_ckpt
MAX_ISOLATES_TO_PROCESS = config.max_isolates_to_process

# Training hyperparameters from config
BATCH_SIZE = config.training.batch_size
LEARNING_RATE = config.training.learning_rate
EPOCHS = config.training.epochs
HIDDEN_DIMS = config.training.hidden_dims
DROPOUT = config.training.dropout
PATIENCE = config.training.patience
EARLY_STOPPING_METRIC = config.training.early_stopping_metric

print(f"\n{'='*40}")
print(f"Virus: {VIRUS_NAME}")
print(f"Config bundle: {config_bundle}")
print(f"{'='*40}")

# Resolve run suffix (manual override in config or auto-generate from sampling params)
# This ensures consistency with preprocessing, embeddings, and dataset scripts
RUN_SUFFIX = resolve_run_suffix(
    config=config,
    max_isolates=MAX_ISOLATES_TO_PROCESS,
    seed=RANDOM_SEED,
    auto_timestamp=True
)

# Set deterministic seeds for training
if RANDOM_SEED is not None:
    set_deterministic_seeds(RANDOM_SEED, cuda_deterministic=True)
    print(f'Set deterministic seeds for training (seed: {RANDOM_SEED})')
else:
    print('No seed set - training will be non-deterministic')

# Build paths using the new path utilities
paths = build_training_paths(
    project_root=project_root,
    virus_name=VIRUS_NAME,
    data_version=DATA_VERSION,
    run_suffix=RUN_SUFFIX,
    config=config
)

default_dataset_dir = paths['dataset_dir']
default_embeddings_file = paths['embeddings_file']
default_output_dir = paths['output_dir']

# Apply CLI overrides if provided
dataset_dir = Path(args.dataset_dir) if args.dataset_dir else default_dataset_dir
embeddings_file = Path(args.embeddings_dir) / 'esm2_embeddings.h5' if args.embeddings_dir else default_embeddings_file
output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
output_dir.mkdir(parents=True, exist_ok=True)

print(f'\nRun directory:   {DATA_VERSION}{RUN_SUFFIX}')
print(f'dataset_dir:     {dataset_dir}')
print(f'embeddings_file: {embeddings_file}')
print(f'output_dir:      {output_dir}')
print(f'model:           {MODEL_CKPT}')
print(f'batch_size:      {BATCH_SIZE}')

print('\nLoad pair datasets.')
train_pairs = pd.read_csv(dataset_dir / 'train_pairs.csv')
val_pairs   = pd.read_csv(dataset_dir / 'val_pairs.csv')
test_pairs  = pd.read_csv(dataset_dir / 'test_pairs.csv')

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CUDA device
CUDA_NAME = args.cuda_name
device = determine_device(CUDA_NAME)

# Initialize model
batch = next(iter(train_loader))
mlp_input_dim = batch[0].shape[1] # 2 * embedding_dim = 2 * 1280 (embed_dim can be obtained from 'esm2_embeddings.h5')
model = MLPClassifier(input_dim=mlp_input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train
print('\nTrain model.')
print(f'Early stopping metric: {EARLY_STOPPING_METRIC}')
best_model_path = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=EPOCHS,
    patience=PATIENCE,
    output_dir=output_dir,
    early_stopping_metric=EARLY_STOPPING_METRIC
)

# Evaluate
print('\nEvaluate model.')
model.load_state_dict(torch.load(best_model_path))
test_res_df = evaluate_model(model, test_loader, criterion, device, test_pairs)

# Save raw predictions
preds_file = output_dir / 'test_predicted.csv'
print(f'\nSave raw predictions to: {preds_file}')
test_res_df.to_csv(preds_file, index=False)

print(f'\n✅ Finished {Path(__file__).name}!')
total_timer.display_timer()
