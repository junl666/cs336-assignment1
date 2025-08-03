

import torch
from typing import List


def gradient_clipping(parameters: List[torch.nn.Parameter], max_norm: float) -> None:
    """
    Implement gradient clipping by L2 norm.
    
    Given the gradient g, compute its L2-norm ||g||_2. If this norm is less than
    max_norm M, leave g as is; otherwise, scale g down by a factor of M/(||g||_2 + ε).
    
    Args:
        parameters: List of model parameters whose gradients will be clipped
        max_norm: Maximum L2-norm allowed for gradients
    """
    eps = 1e-6  # PyTorch default for numerical stability
    
    # Compute the total L2 norm of all gradients
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.data.norm(dtype=torch.float32) ** 2
    total_norm = total_norm ** 0.5
    
    # Apply clipping if needed
    if total_norm > max_norm:
        clip_factor = max_norm / (total_norm + eps)
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_factor) 