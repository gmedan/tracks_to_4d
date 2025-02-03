import torch
import torch.nn as nn
import einops

import utils 

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
    

    
class CoordinatePositionalEncoding(nn.Module):
    REARRANGE_ORDER = 'pos sincos d'

    def __init__(self, 
                 positional_dim: int = 12):
        
        super(CoordinatePositionalEncoding, self).__init__()
        self.positional_dim = positional_dim
        self.pi_times_powers_of_2 = \
            torch.tensor([(2 ** j) * torch.pi 
                          for j in range(positional_dim)],
                         requires_grad = False)
                         
    @property
    def output_dim(self):
        return 3 * self.positional_dim * 2 + 3 + 1
        
    def forward(self, pts_2d_with_visibility: torch.Tensor) -> torch.Tensor:
        batch, num_frames, num_points, d = pts_2d_with_visibility.shape
        assert d == 3

        x = einops.rearrange(pts_2d_with_visibility, 'b n p d -> b n p d 1') * \
            einops.rearrange(self.pi_times_powers_of_2,  'pos -> 1 1 1 1 pos')
        x = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
        x = einops.rearrange(x, f'b n p d pos sincos -> b n p ({self.REARRANGE_ORDER})')
        x = torch.cat([pts_2d_with_visibility, x], dim=-1)
        x = utils.pad_val_after(x, dim=-1, val=1)
        assert x.shape[-1] == self.output_dim
        return x