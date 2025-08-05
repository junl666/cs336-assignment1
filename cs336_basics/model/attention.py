import torch
from torch import nn
from einops import einsum, rearrange, repeat
from cs336_basics.model.linear import Linear
from cs336_basics.model.rope import Rope

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply softmax to the input tensor along the specified dimension.
    """
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.
    
    Args:
        query: Query tensor of shape (..., seq_len, d_k).
        key: Key tensor of shape (..., seq_len, d_k).
        value: Value tensor of shape (..., seq_len, d_v).
        mask: Optional mask tensor of shape (..., seq_len, seq_len).

    Returns:
        Tensor of shape (..., seq_len, d_v) containing the attention output.
    """
    d_k = query.size(-1)

    scores = einsum(query, key, "... seq_i d_k, ... seq_j d_k -> ... seq_i seq_j") / (d_k ** 0.5)  # shape (..., seq_len, seq_len)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = softmax(scores, dim=-1)  # shape (..., seq_len, seq_len)
    output = einsum(attn_weights, value, "... seq_i seq_j, ... seq_j d_v -> ... seq_i d_v")  # shape (..., seq_len, d_v)
    return output


class MultiheadSelfAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 use_rope: bool = True, 
                 max_seq_len: int|None = None, 
                 theta: float|None = None, 
                 token_positions: torch.Tensor|None = None,
                 device: torch.device | None = None):
        """
        Initialize the multi-head self-attention module.
        
        Args:
            d_model: Dimension of the model.
            n_heads: Number of attention heads.
        """
        super(MultiheadSelfAttention, self).__init__()
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.device = device

        # Linear
        # TODO: Make it using one Linear layer
        self.query_linear = Linear(d_model, n_heads * d_k, device=device)
        self.key_linear = Linear(d_model, n_heads * d_k, device=device)
        self.value_linear = Linear(d_model, n_heads * d_v, device=device)
        self.out_linear = Linear(n_heads * d_v, d_model, device=device)

        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.token_positions = token_positions
            self.rope = Rope(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model).

        Returns:
            Output tensor of shape (..., seq_len, d_model).
        """
        seq_len = x.size(-2)
        # Linear projections
        Q = rearrange(self.query_linear(x), "... seq (h d_k) -> ... h seq d_k", h=self.n_heads)
        K = rearrange(self.key_linear(x), "... seq (h d_k) -> ... h seq d_k", h=self.n_heads)
        V = rearrange(self.value_linear(x), "... seq (h d_v) -> ... h seq d_v", h=self.n_heads)
        # Apply RoPE
        if self.use_rope:
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)
        # Create a causal mask of shape (..., seq_len, seq_len))
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = repeat(mask, "seq_i seq_j -> b h seq_i seq_j", 
                      b=x.size(0), h=self.n_heads) # shape (..., n_heads, seq_len, seq_len)
        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask) # shape (..., n_heads, seq_len, d_v)
        # Concatenate heads
        attn_output = rearrange(attn_output, "... h seq d_v -> ... seq (h d_v)")
        attn_output = self.out_linear(attn_output)
        return attn_output