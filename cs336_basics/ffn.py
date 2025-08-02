import torch
from torch import nn
from cs336_basics.linear import Linear


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Arguments:
            d_model: Dimension of the input and output tensors.
            d_ff: Dimension of the hidden layer in the feed-forward network.
            device: Device to store the parameters on.
            dtype: Data type of the parameters.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._silu(self.linear1(x)) * self.linear3(x)
        return self.linear2(x)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)