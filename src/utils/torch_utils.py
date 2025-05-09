import os
import torch


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
    print('GPU available: ', cuda_avail)
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