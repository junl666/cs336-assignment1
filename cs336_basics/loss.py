import torch
from torch import nn
from einops import einsum

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between the predicted logits and the true labels.

    Args:
        logits (torch.Tensor): Predicted logits from the model. shape (batch_size, vocab_size)
        targets (torch.Tensor): True labels. shape (batch_size)

    Returns:
        torch.Tensor: Computed cross-entropy loss averaged over the batch. 
    """
    targets_one_hot = nn.functional.one_hot(targets, logits.size(-1)).float()
    targets_logits = einsum(logits, targets_one_hot, "b v, b v -> b")
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    return (-targets_logits + log_sum_exp).mean()
