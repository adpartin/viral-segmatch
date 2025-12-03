import os
from typing import Optional
import torch
import torch.nn as nn
import torch.optim


# def determine_device(cuda_name_from_params): # config! 
def determine_device(cuda_name_from_params: str) -> str:
    """ Determine the PyTorch device (CPU/GPU) for model training and inference.

    This function checks GPU availability and CUDA environment settings to
    determine the appropriate device. It considers both system GPU availability
    and any specific CUDA device preferences set through environment variables.

    Args:
        cuda_name_from_params: GPU device specification (e.g., 'cuda:0', 'cuda:1').
            This is used when no CUDA_VISIBLE_DEVICES environment variable is set.

    Returns:
        str: Device identifier for PyTorch ('cpu' or 'cuda:X' where X is the device index).

    Notes:
        - When CUDA_VISIBLE_DEVICES is set, device indexing is reindexed to start from 0
        - Returns 'cpu' if no GPU is available regardless of input parameter
    """
    cuda_avail = torch.cuda.is_available()
    print(f'GPU available: {cuda_avail}')
    if cuda_avail:  # if GPU available
        cuda_env_visible = os.getenv('CUDA_VISIBLE_DEVICES') # CUDA device from env var
        if cuda_env_visible is not None:
            # Note! When one or multiple device numbers are passed via
            # CUDA_VISIBLE_DEVICES, the values in python script are reindexed
            # and start from 0.
            print('CUDA_VISIBLE_DEVICES: ', cuda_env_visible)
            cuda_name = 'cuda:0'
        else:
            cuda_name = cuda_name_from_params
        device = cuda_name
    else:
        device = 'cpu'

    return device


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9
    ) -> torch.optim.Optimizer:
    """
    Create an optimizer for model training.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: L2 regularization weight (default: 0.0)
        momentum: Momentum for SGD (default: 0.9)
    
    Returns:
        Optimizer instance
    
    Raises:
        ValueError: If optimizer_name is not supported
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        print(f"Using optimizer: SGD (lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay})")
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        print(f"Using optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
    elif optimizer_name == 'adamw':
        # AdamW: Use default weight_decay=0.01 if not specified
        wd = weight_decay if weight_decay > 0 else 0.01
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=wd
        )
        print(f"Using optimizer: AdamW (lr={learning_rate}, weight_decay={wd})")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: 'adam', 'adamw', 'sgd'")
    
    return optimizer


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    early_stopping_metric: str = 'loss',
    epochs: int = 100,
    patience: int = 5,
    factor: float = 0.5,
    min_lr: float = 1e-6
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('reduce_on_plateau', 'cosine', 'step', or None)
        early_stopping_metric: Metric used for early stopping ('loss', 'f1', 'auc')
            Used to determine mode for ReduceLROnPlateau
        epochs: Total number of epochs (for CosineAnnealingLR)
        patience: Patience for ReduceLROnPlateau or step_size for StepLR
        factor: Factor to reduce LR by
        min_lr: Minimum learning rate
    
    Returns:
        LR scheduler instance or None if scheduler_type is None/False
    """
    if not scheduler_type or scheduler_type.lower() == 'none':
        return None
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'reduce_on_plateau':
        # Reduce LR when metric plateaus (works with early_stopping_metric)
        mode = 'max' if early_stopping_metric in ['f1', 'auc'] else 'min'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )
        print(f"Using LR scheduler: ReduceLROnPlateau (mode={mode}, patience={patience}, factor={factor})")
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr
        )
        print(f"Using LR scheduler: CosineAnnealingLR (T_max={epochs}, eta_min={min_lr})")
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=patience,  # Using patience as step_size
            gamma=factor
        )
        print(f"Using LR scheduler: StepLR (step_size={patience}, gamma={factor})")
    else:
        print(f"⚠️  Unknown scheduler type: {scheduler_type}, proceeding without scheduler")
        return None
    
    return scheduler