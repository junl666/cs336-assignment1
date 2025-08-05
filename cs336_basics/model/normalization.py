import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Arguments:
            dim: Dimension of the input tensor.
            eps: Small value to avoid division by zero.
            device: Device to store the parameters on.
            dtype: Data type of the parameters.
        """
        super(RMSNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        # perform RMS normalization
        rms_value = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        # print(f"x.device = {x.device}, x.dtype = {x.dtype}, gain.device = {self.gain.device}, gain.dtype = {self.gain.dtype}")
        x = x / rms_value
        x = x * self.gain
        return x.to(in_type)