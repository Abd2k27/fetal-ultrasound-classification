import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    """
    Sets the seed for all random number generators to ensure 
    that experiments are reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensures that the GPU operations are deterministic (at the cost of some speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    Automatically detects and returns the best available hardware accelerator.
    - CUDA: For NVIDIA GPUs.
    - MPS: For Apple Silicon (M1/M2/M3) GPUs.
    - CPU: Fallback if no GPU is found.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Metal Performance Shaders for Mac acceleration
        return torch.device("mps")
    else:
        return torch.device("cpu")

def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves the model weights and optimizer state to a file.
    This allows us to resume training later or load the best version of the model.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
