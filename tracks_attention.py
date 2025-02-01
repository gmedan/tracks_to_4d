from typing import Any, Optional
import einops
import einops.layers
import einops.layers.torch
from utils import axis_of
import torch.nn as nn
import torch


class TracksAttention(nn.Module):
    def __init__(self, 
                 input_dim: int = 256,
                 output_dim: int = 256,
                 num_heads: int = 16,
                 hidden_layer_dim: int = 2048,
                 dropout: float = 0.1):
        super(TracksAttention, self).__init__()

        self.num_heads = num_heads
        assert input_dim % num_heads == 0
        self.head_dim = input_dim // num_heads
        self.dropout = dropout

        self.multihead_attn_frames = nn.MultiheadAttention(embed_dim=output_dim, 
                                                           num_heads=num_heads, 
                                                           dropout=dropout, 
                                                           batch_first=True)
        self.multihead_attn_points = nn.MultiheadAttention(embed_dim=output_dim, 
                                                           num_heads=num_heads, 
                                                           dropout=dropout, 
                                                           batch_first=True)

        # Output projection
        self.fully_connected = nn.Sequential(
            einops.layers.torch.EinMix(
                'batch frame point dim -> batch frame point hidden_dim',
                weight_shape='dim hidden_dim',
                hidden_dim=hidden_layer_dim,
                dim=input_dim,
            ),
            nn.ReLU(),
            einops.layers.torch.EinMix(
                'batch frame point hidden_dim -> batch frame point output_dim',
                weight_shape='hidden_dim output_dim',
                hidden_dim=hidden_layer_dim,
                output_dim=output_dim
            )
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        n_batch, n_frame, n_point, dim = x.shape
        assert dim == self.head_dim * self.num_heads

        x = einops.rearrange(x, 'batch frame point dim -> (batch point) frame dim', batch=n_batch, frame=n_frame, point=n_point)
        x, _ = self.multihead_attn_frames(x, x, x, need_weights=False)
        x = einops.rearrange(x, '(batch point) frame dim -> (batch frame) point dim', batch=n_batch, frame=n_frame, point=n_point)
        x, _ = self.multihead_attn_points(x, x, x, need_weights=False)
        x = einops.rearrange(x, '(batch frame) point dim -> batch frame point dim', batch=n_batch, frame=n_frame, point=n_point)
        x = self.fully_connected(x)
        return x