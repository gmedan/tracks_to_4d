import torch
import torch.nn as nn
import einops

class PositionalEncoding2D(nn.Module):
    """
    Positional encoding for 2D coordinates, extended for 3D input (x, y, o).
    """
    def __init__(self, dim: int):
        super(PositionalEncoding2D, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x is of shape (N, P, 3), where the last dimension is (x, y, o)
        N, P, D = x.shape
        assert D == 3, "Input tensor must have 3 dimensions (x, y, o)"
        
        # Create positional encodings
        pe = torch.zeros(N, P, self.dim, device=x.device)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=x.device) * -(torch.log(torch.tensor(10000.0)) / self.dim))
        
        pe[:, :, 0::2] = torch.sin(x[:, :, 0:1] * div_term)  # Apply sin to even indices
        pe[:, :, 1::2] = torch.cos(x[:, :, 0:1] * div_term)  # Apply cos to odd indices
        pe[:, :, 2] = x[:, :, 2]
        return pe
