"""
Learning verification utilities for neural network training.

These utilities help verify that models are actually learning from data,
following best practices from Karpathy's recipe for training neural networks.
Reference: https://karpathy.github.io/2019/04/25/recipe/
"""

from typing import Optional

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
            if isinstance(batch_x, (tuple, list)) and len(batch_x) == 2:
                batch_a, batch_b = batch_x
                batch_a, batch_b = batch_a.to(device), batch_b.to(device)
                preds = model(batch_a, batch_b).squeeze()
            else:
                batch_x = batch_x.to(device)
                preds = model(batch_x).squeeze()
            batch_y = batch_y.to(device)
            loss = criterion(preds, batch_y)
            initial_loss += loss.item() * batch_y.size(0)
            n_samples += batch_y.size(0)
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
    random_f1 = f1_score(val_labels, random_preds, average='binary', pos_label=1)
    
    # Majority class classifier
    majority_class = 1 if n_positive > n_negative else 0
    majority_preds = np.full(n_samples, majority_class)
    majority_f1 = f1_score(val_labels, majority_preds, average='binary', pos_label=1)
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


def _plot_loss_and_f1(history, axes_row, epochs, num_epochs):
    """Plot loss (left) and F1 (right) on a row of two axes.

    Shared helper for both ``perf_curves1.png`` (1×2) and
    ``perf_curves2.png`` (2×2, top row).
    """
    ax_loss, ax_f1 = axes_row

    # Loss curves
    ax_loss.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax_loss.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Train and Validation Loss')
    ax_loss.set_xlim([1, num_epochs])
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # F1 curves
    if 'train_f1' in history and len(history['train_f1']) > 0:
        ax_f1.plot(epochs, history['train_f1'], 'b--', label='Train F1', linewidth=2, alpha=0.7)
    ax_f1.plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    ax_f1.set_xlabel('Epoch')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.set_title('Train and Validation F1 Score')
    ax_f1.set_xlim([1, num_epochs])
    ax_f1.legend()
    ax_f1.grid(True, alpha=0.3)
    ax_f1.set_ylim([0, 1])


def plot_learning_curves(history, output_dir, bundle_name: Optional[str] = None, dpi: int = 200):
    """
    Plot learning/performance curves and save several PNG files.

    Saved files
    -----------
    perf_curves2.png   4-panel (2×2): loss, F1, AUC-ROC, F1 gap
    perf_curves1.png   2-panel (1×2): loss, F1 (top row of perf_curves2)
    loss.png           standalone loss plot
    f1.png             standalone F1 plot

    Args:
        history: Dictionary with keys:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'val_f1': List of validation F1 scores per epoch
            - 'val_auc': List of validation AUC-ROC scores per epoch
        output_dir: Path object where plot will be saved
        bundle_name: Optional bundle name to include in the title
        dpi: Resolution for saved plot
    
    Returns:
        Path: Path to the main saved plot file (perf_curves2.png)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    num_epochs = len(history['train_loss'])

    # ------------------------------------------------------------------
    # perf_curves2.png  (4-panel: loss, F1, AUC, F1 gap)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    _plot_loss_and_f1(history, axes[0], epochs, num_epochs)

    # AUC curves (bottom left)
    if 'train_auc' in history and len(history['train_auc']) > 0:
        axes[1, 0].plot(epochs, history['train_auc'], 'b--', label='Train AUC', linewidth=2, alpha=0.7)
    axes[1, 0].plot(epochs, history['val_auc'], 'm-', label='Val AUC', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('Train and Validation AUC-ROC')
    axes[1, 0].set_xlim([1, num_epochs])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # Train/Val F1 gap (bottom right)
    if 'train_f1' in history and len(history['train_f1']) > 0:
        f1_gap = [t - v for t, v in zip(history['train_f1'], history['val_f1'])]
        axes[1, 1].plot(epochs, f1_gap, 'orange', label='Train F1 - Val F1', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Gap')
        axes[1, 1].set_title('Overfitting Indicator (Train F1 - Val F1)')
        axes[1, 1].set_xlim([1, num_epochs])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Train F1 not tracked',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Overfitting Indicator (Train F1 not available)')

    suptitle = f'Performance Curves ({bundle_name})' if bundle_name else 'Performance Curves'
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plot_path = output_dir / 'perf_curves2.png'
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Performance curves (4-panel) saved to: {plot_path}")

    # ------------------------------------------------------------------
    # perf_curves1.png  (2-panel: loss + F1 only)
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    _plot_loss_and_f1(history, axes1, epochs, num_epochs)

    suptitle1 = f'Performance Curves ({bundle_name})' if bundle_name else 'Performance Curves'
    fig1.suptitle(suptitle1, fontsize=16, fontweight='bold', y=1.01)

    plt.tight_layout()
    perf1_path = output_dir / 'perf_curves1.png'
    plt.savefig(perf1_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Performance curves (2-panel) saved to: {perf1_path}")

    # ------------------------------------------------------------------
    # loss.png  (standalone)
    # ------------------------------------------------------------------
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(8, 6))
    ax_loss.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax_loss.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Train and Validation Loss')
    ax_loss.set_xlim([1, num_epochs])
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = output_dir / 'loss.png'
    plt.savefig(loss_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to: {loss_path}")

    # ------------------------------------------------------------------
    # f1.png  (standalone)
    # ------------------------------------------------------------------
    fig_f1, ax_f1 = plt.subplots(1, 1, figsize=(8, 6))
    if 'train_f1' in history and len(history['train_f1']) > 0:
        ax_f1.plot(epochs, history['train_f1'], 'b--', label='Train F1', linewidth=2, alpha=0.7)
    ax_f1.plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    ax_f1.set_xlabel('Epoch')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.set_title('Train and Validation F1 Score')
    ax_f1.set_xlim([1, num_epochs])
    ax_f1.legend()
    ax_f1.grid(True, alpha=0.3)
    ax_f1.set_ylim([0, 1])
    plt.tight_layout()
    f1_path = output_dir / 'f1.png'
    plt.savefig(f1_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"F1 plot saved to: {f1_path}")

    return plot_path
