import numpy as np
import torch
import os
import typing


def data_loading(x: np.ndarray, 
                 batch_size: int, 
                 context_length: int, 
                 device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function takes a numpy array x (integer array with token IDs), a
    batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns
    a pair of tensors: the sampled input sequences and the corresponding next-token targets.
    Args:
        x: Integer array with token IDs.
        batch_size: Number of sequences in each batch.
        context_length: Length of the context window.
        device: Device to load the data onto (e.g., "cpu" or "cuda:0").

    Returns:
        A tuple of two tensors: the input sequences and the target sequences.
    """
    # 1. 随机生成起始索引
    max_start_idx = len(x) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # 2. 提取输入和目标序列
    inputs = np.stack([x[idx : idx + context_length] for idx in start_indices])
    targets = np.stack([x[idx + 1 : idx + context_length + 1] for idx in start_indices])
    
    # 3. 转换为PyTorch张量并移动到设备
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    
    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    """
    Save model checkpoint including model state, optimizer state, and iteration number.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save  
        iteration: Current iteration number
        out: Output path or file-like object
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Load model checkpoint and restore model state, optimizer state.
    
    Args:
        src: Path or file-like object to load from
        model: The model to restore state to
        optimizer: The optimizer to restore state to
        
    Returns:
        The iteration number from the checkpoint
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']