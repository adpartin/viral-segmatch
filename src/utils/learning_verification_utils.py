"""
Learning verification utilities for neural network training.

These utilities help verify that models are actually learning from data,
following best practices from Karpathy's recipe for training neural networks.
Reference: https://karpathy.github.io/2019/04/25/recipe/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader


def check_initialization_loss(model, train_loader, criterion, device):
    """
    Karpathy-style check: Verify loss at initialization.
    
    For binary classification with balanced data, initial loss should be ~-log(0.5) ≈ 0.693
    If loss is way off, something is wrong with the model/data setup.
    
    Args:
        model: PyTorch model (should be in eval mode)
        train_loader: DataLoader for training data
        criterion: Loss function
        device: Device to run on ('cpu' or 'cuda:X')
    
    Returns:
        float: Initial loss value
    """
    model.eval()
    initial_loss = 0
    n_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x).squeeze()
            loss = criterion(preds, batch_y)
            initial_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)
            if n_samples >= 100:  # Check on first 100 samples
                break
    
    initial_loss = initial_loss / n_samples if n_samples > 0 else float('inf')
    expected_loss = -np.log(0.5)  # ~0.693 for balanced binary classification
    
    print(f"\n{'='*60}")
    print("INITIALIZATION LOSS CHECK (Karpathy-style)")
    print('='*60)
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Expected loss (balanced binary): ~{expected_loss:.4f}")
    if abs(initial_loss - expected_loss) < 0.2:
        print("✅ Initialization loss looks reasonable")
    else:
        print(f"⚠️  Initialization loss differs significantly from expected (~{expected_loss:.4f})")
        print("   This might indicate data imbalance or model initialization issues")
    print('='*60 + "\n")
    
    return initial_loss


def compute_baseline_metrics(val_labels, random_seed=42):
    """
    Compute baseline metrics for comparison.
    
    Computes performance of:
    - Random classifier (predicts randomly)
    - Majority class classifier (always predicts most common class)
    
    These baselines help verify that the model is actually learning,
    not just producing random predictions.
    
    Args:
        val_labels: Array-like of true binary labels
        random_seed: Random seed for reproducibility of random classifier
    
    Returns:
        dict: Dictionary containing baseline metrics:
            - 'random_f1': F1 score of random classifier
            - 'majority_class': Most common class (0 or 1)
            - 'majority_f1': F1 score of majority class classifier
            - 'majority_acc': Accuracy of majority class classifier
            - 'class_balance': Dict with positive/negative counts and ratio
    """
    val_labels = np.array(val_labels)
    n_samples = len(val_labels)
    n_positive = val_labels.sum()
    n_negative = n_samples - n_positive
    
    # Random classifier (predict randomly)
    np.random.seed(random_seed)  # For reproducibility
    random_preds = np.random.randint(0, 2, size=n_samples)
    random_f1 = f1_score(val_labels, random_preds)
    
    # Majority class classifier
    majority_class = 1 if n_positive > n_negative else 0
    majority_preds = np.full(n_samples, majority_class)
    majority_f1 = f1_score(val_labels, majority_preds)
    majority_acc = (val_labels == majority_preds).mean()
    
    return {
        'random_f1': random_f1,
        'majority_class': majority_class,
        'majority_f1': majority_f1,
        'majority_acc': majority_acc,
        'class_balance': {
            'positive': n_positive,
            'negative': n_negative,
            'ratio': n_positive / n_samples if n_samples > 0 else 0
        }
    }


def plot_learning_curves(history, output_dir):
    """
    Plot learning curves: train/val loss, F1, and AUC across epochs.
    
    Creates a 3-panel figure showing:
    1. Training and validation loss over epochs
    2. Validation F1 score over epochs
    3. Validation AUC-ROC over epochs
    
    This helps visualize whether the model is learning and if overfitting is occurring.
    Classic overfitting pattern: training loss drops low while validation loss stays high.
    
    Args:
        history: Dictionary with keys:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'val_f1': List of validation F1 scores per epoch
            - 'val_auc': List of validation AUC-ROC scores per epoch
        output_dir: Path object where plot will be saved
    
    Returns:
        Path: Path to saved plot file
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves (top left)
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 curves (top right) - now with both train and val
    if 'train_f1' in history and len(history['train_f1']) > 0:
        axes[0, 1].plot(epochs, history['train_f1'], 'b--', label='Train F1', linewidth=2, alpha=0.7)
    axes[0, 1].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Training and Validation F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # AUC curves (bottom left) - now with both train and val
    if 'train_auc' in history and len(history['train_auc']) > 0:
        axes[1, 0].plot(epochs, history['train_auc'], 'b--', label='Train AUC', linewidth=2, alpha=0.7)
    axes[1, 0].plot(epochs, history['val_auc'], 'm-', label='Val AUC', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('Training and Validation AUC-ROC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Train/Val F1 gap (bottom right) - shows overfitting indicator
    if 'train_f1' in history and len(history['train_f1']) > 0:
        f1_gap = [t - v for t, v in zip(history['train_f1'], history['val_f1'])]
        axes[1, 1].plot(epochs, f1_gap, 'orange', label='Train F1 - Val F1', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Gap')
        axes[1, 1].set_title('Overfitting Indicator (Train F1 - Val F1)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Fallback if train_f1 not available
        axes[1, 1].text(0.5, 0.5, 'Train F1 not tracked', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Overfitting Indicator (Train F1 not available)')
    
    plt.tight_layout()
    plot_path = output_dir / 'learning_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Learning curves saved to: {plot_path}")
    
    return plot_path

