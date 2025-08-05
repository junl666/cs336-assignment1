import torch
from torch import nn
from einops import einsum, repeat

class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """
        Construct the RoPE module and create buffers if needed.
        Arguments:
            theta: Scaling factor for the rope
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length that will be inputted
            device: Device to store the buffer on
        """
        super(Rope, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.register_buffer("rope", self._create_rope(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        """
        no_head_dim = False
        if token_positions is None:
            seq_len = x.size(-2)
            token_positions = torch.arange(seq_len, device=self.device).expand(x.size(0), seq_len) # shape (batch_size, seq_len)
        if x.dim() == 3: # shape (batch_size, seq_len, d_k) no head dimension
            no_head_dim = True
            x = x.unsqueeze(1)  # shape (batch_size, 1, seq_len, d_k)
        # Now x is of shape (batch_size, n_heads, seq_len, d_k)
        rope = repeat(self.rope, "max_seq_len d_i d_j -> b max_seq_len d_i d_j", b=x.size(0)) # shape (batch_size, max_seq_len, d_k, d_k)
        one_hot_positions = torch.nn.functional.one_hot(token_positions, num_classes=self.max_seq_len).to(x.dtype) # shape (batch_size, seq_len, max_seq_len)
        rope_values = einsum(one_hot_positions, rope, "... seq max_seq, ... max_seq d_i d_j -> ... seq d_i d_j")  # shape (batch_size, seq_len, d_k, d_k)
        rope_values = repeat(rope_values, "b seq d_i d_j -> b h seq d_i d_j", h=x.size(1))  # shape (batch_size, n_heads, seq_len, d_k, d_k)
        # return einsum(x, rope_values, "... seq d_k, ... seq d_k d_k -> ... seq d_k")  # wrong！！！！！
        # print(f"Rope forward: x shape {x.shape}, rope_values shape {rope_values.shape}")
        # x.shape (batch_size, n_heads, seq_len, d_k)
        result = einsum(x, rope_values, "... seq j, ... seq i j -> ... seq i")  # shape (..., seq_len, d_k)
        if no_head_dim:
            result = result.squeeze(1)
        return result 


    def _create_rope(self) -> torch.Tensor:
        """
        Create the RoPE buffer based on the theta, d_k, and max_seq_len.
        Returns:
            A tensor of shape (max_seq_len, d_k, d_k) containing the RoPE values.
        """
        # TODO: Need a way to optimize
        # Create a tensor of positions
        positions = torch.arange(self.max_seq_len, device=self.device) # shape (max_seq_len,)
        # Create a tensor of k
        k = torch.arange(0, self.d_k // 2, device=self.device) # shape (d_k // 2,)
        # Calculate the RoPE values
        angle_rates = 1 / (self.theta ** ((2 * k) / self.d_k))
        angles = einsum(positions, angle_rates, "seq, k -> seq k") # shape (max_seq_len, d_k // 2)
        # Create the RoPE tensor
        # For each angle, need to become:
        # [[cos(angle), -sin(angle), 
        #  sin(angle), cos(angle)]]
        # Compute cos and sin of angles
        cos_angles = torch.cos(angles)  # shape (max_seq_len, d_k // 2)
        sin_angles = torch.sin(angles)  # shape (max_seq_len, d_k // 2)
        
        # Initialize the rotation matrix for each position
        rope_matrix = torch.zeros(self.max_seq_len, self.d_k, self.d_k, device=self.device)
        
        # Fill the rotation matrix for each position
        for i in range(self.d_k // 2):
            # For each pair of dimensions (2*i, 2*i+1), create a 2x2 rotation matrix
            rope_matrix[:, 2*i, 2*i] = cos_angles[:, i]      # cos(angle)
            rope_matrix[:, 2*i, 2*i+1] = -sin_angles[:, i]   # -sin(angle)
            rope_matrix[:, 2*i+1, 2*i] = sin_angles[:, i]    # sin(angle)
            rope_matrix[:, 2*i+1, 2*i+1] = cos_angles[:, i]  # cos(angle)
        
        return rope_matrix
        