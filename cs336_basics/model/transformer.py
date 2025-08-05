import torch
from torch import nn
from cs336_basics.model.attention import MultiheadSelfAttention
from cs336_basics.model.ffn import PositionWiseFeedForward
from cs336_basics.model.normalization import RMSNorm
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None):
        """
        Initialize the Transformer block with multi-head self-attention and optional RoPE.
        
        Args:
            d_model: Dimension of the model.
            n_heads: Number of attention heads.
            d_ff: Dimension of the feed-forward network.
            max_seq_len: Maximum sequence length for RoPE.
            theta: Scaling factor for RoPE.
        """
        super(TransformerBlock, self).__init__()
        self.self_attention = MultiheadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
        )
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer block.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model).
        
        Returns:
            Output tensor of shape (..., seq_len, d_model).
        """
        attn_output = self.self_attention(self.norm1(x))
        x = x + attn_output  # Residual connection
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output  # Residual connection
        return x  # Output shape is (..., seq_len, d_model)
    
class TransformerLM(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 context_length: int, 
                 num_layers: int, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int, 
                 max_seq_len: int | None = None, 
                 rope_theta: float | None = None, 
                 device: torch.device | None = None):
        super(TransformerLM, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, rope_theta, device) for _ in range(num_layers)
        ])
        self.norm_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.context_length = context_length
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.layers:
            x = block(x)
        x = self.norm_final(x)
        return self.lm_head(x)  # Output shape is (..., seq_len, vocab_size)
        
        
        
