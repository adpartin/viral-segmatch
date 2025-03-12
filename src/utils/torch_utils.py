import os
import torch


# def determine_device(cuda_name_from_params): # config! 
def determine_device(cuda_name_from_params):
    """Determine device to run PyTorch functions.

    PyTorch functions can run on CPU or on GPU. In the latter case, it
    also takes into account the GPU devices requested for the run.

    :params str cuda_name_from_params: GPUs specified for the run.

    :return: Device available for running PyTorch functionality.
    :rtype: str
    """
    cuda_avail = torch.cuda.is_available()
    print('GPU available: ', cuda_avail)
    if cuda_avail:  # if GPU available
        # CUDA device from env var
        cuda_env_visible = os.getenv('CUDA_VISIBLE_DEVICES')
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