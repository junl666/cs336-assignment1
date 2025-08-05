import torch
from torch import nn
from einops import einsum

# No bias term
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Arguments:
            in_features: final dimension of the input
            out_features: final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self._init_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

    def _init_params(self) -> None:
        # truncated from -3*sigma to 3*sigma, where sigma^2 = 2 / (in_features + out_features), mean = 0
        sigma = (2 / (self.weight.shape[1] + self.weight.shape[0])) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)