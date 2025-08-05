import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Arguments:
            num_embeddings: Number of unique tokens in the vocabulary.
            embedding_dim: Dimension of the embedding vector for each token.
            device: Device to store the parameters on.
            dtype: Data type of the parameters.
        """
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self._init_params()

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.weight[indices]

    def _init_params(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)