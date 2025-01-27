import torch
import torch.nn as nn
import einops 

class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding for frame indices.
    Encodes the temporal position (frame index) using sinusoidal functions.
    """
    def __init__(self, d_model: int = 2, max_len: int = 5000):
        """
        Args:
            d_model (int): Dimensionality of the model (feature space).
            max_len (int): Maximum length of the sequence (number of frames).
        """
        super(TemporalPositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Precompute positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Compute sinusoidal encodings
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.register_buffer('pe', pe)  # Register as a non-learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, P, d_model).
        
        Returns:
            torch.Tensor: Input tensor with positional encoding added, same shape as input.
        """
        B, N, P, D = x.shape
        pos_encoding = einops.rearrange(self.pe[:N, :], 'n d -> 1 n 1 d')  # Shape: (B, N, 1, d_model)
        if D == self.d_model + 1:
            pos_encoding = torch.concat([pos_encoding, torch.zeros_like(pos_encoding[...,0:1])], dim=-1)
        elif D == self.d_model:\
            pass
        else:
            raise ValueError(f'expected last dim of x to be {self.d_model} or {self.d_model+1}')
        return x + pos_encoding