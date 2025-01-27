from typing import Any, Optional
import jax
import jax.numpy as jnp
from einops import einsum
from einops.layers.torch import EinMix
from utils import axis_of
import torch.nn as nn
import torch


class InterleavedAttention(nn.Module):
    """
    Unified attention module that handles both frame and point attention using 4D input tensors.
    Uses efficient einsum operations with consistent dimension naming throughout.
    """
    def __init__(self, 
                 num_heads: int = 16,
                 head_dim: int = 16,
                 dropout: float = 0.1):
        super(InterleavedAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        # QKV projection for frame attention
        self.qkv_frame = EinMix(
            'batch frame point dim -> qkv batch point frame heads dim_head',
            weight_shape='qkv heads dim_head',
            bias_shape='qkv heads dim_head',
            heads=self.num_heads,
            dim_head=self.head_dim,
            qkv=3
        )

        # QKV projection for point attention
        self.qkv_point = EinMix(
            'batch frame point heads_in dim_head_in -> qkv batch frame point heads_out dim_head_out',
            weight_shape='qkv heads_out dim_head_out heads_in dim_head_in',
            bias_shape='qkv heads_out dim_head_out',
            heads_in=self.num_heads,
            dim_head_out=self.head_dim,
            heads_out=self.num_heads,
            dim_head_in=self.head_dim,
            qkv=3
        )

        # Output projection
        self.out_proj = EinMix(
            'batch frame point heads dim_head -> batch frame point dim',
            weight_shape='heads dim_head dim',
            bias_shape='dim',
            heads=self.num_heads,
            dim_head=self.head_dim,
            dim=self.num_heads * self.head_dim
        )

    def frame_attention(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply attention across frames while preserving point relationships.
        Input shape: [batch, frame, point, dim]
        """
        # Generate QKV with consistent dimension ordering
        query, key, value = self.qkv_frame(x)
        
        # Compute attention scores with explicit dimension naming
        attention_scores = einsum(
            query, key,
            'batch point frame_query heads dim_head, ' + \
            'batch point frame_key   heads dim_head -> ' + \
            (attention_scores_pattern:='batch point frame_query frame_key heads'),
        )

        if mask is not None:
            attention_scores = jnp.where(mask, attention_scores, float('-inf'))

        # Apply softmax over frame_key dimension
        attention_weights = jax.nn.softmax(attention_scores, 
                                           axis=axis_of(attention_scores_pattern, 'frame_key'))
        # attention_weights = nn.Dropout(rate=self.dropout)(
        #     attention_weights, deterministic=not self.training
        # )

        # Apply attention weights to values, directly producing desired output ordering
        attended = einsum(
            attention_weights, 
            value,
            'batch point frame_query frame_key heads, ' + \
            'batch point frame_key heads dim_head -> ' + \
            'batch frame_query point heads dim_head'
        )
        return attended

    def point_attention(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply attention across points while preserving temporal relationships.
        """
        # Generate QKV with consistent dimension ordering
        # [batch frame point heads dim_head]
        query, key, value = self.qkv_point(x)
        
        # Compute attention scores with explicit dimension naming
        attention_scores = einsum(
            query, key,
            'batch frame point_query heads dim_head, ' + \
            'batch frame point_key   heads dim_head -> ' + \
            (attention_scores_pattern:='batch frame point_query point_key heads'),
        )

        if mask is not None:
            attention_scores = jnp.where(mask, attention_scores, float('-inf'))

        # Apply softmax over point_key dimension
        attention_weights = jax.nn.softmax(attention_scores, 
                                           axis=axis_of(attention_scores_pattern, 'point_key'))
        # attention_weights = nn.Dropout(rate=self.dropout)(
        #     attention_weights, deterministic=not self.training
        # )

        # Apply attention weights to values, directly producing desired output ordering
        attended = einsum(
            attention_weights, value,
            'batch frame point_query point_key heads, ' + \
            'batch frame point_key   heads dim_head -> ' + \
            'batch frame point_query heads dim_head'
        )

        return attended

    def __call__(self, x: torch.Tensor, 
                 frame_mask: torch.Tensor = None,
                 point_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply unified attention combining both frame and point attention.
        
        Args:
            x: Input tensor of shape [batch, frame, point, dim]
            frame_mask: Optional mask for frame attention
            point_mask: Optional mask for point attention
        
        Returns:
            Attended tensor of shape [batch, frame, point, dim]
        """
        # Layer normalization
        # x = nn.LayerNorm()(x)
        
        # Apply attention in specified order
        frame_out = self.frame_attention(x, frame_mask)
        point_out = self.point_attention(frame_out, point_mask)

        # Final projection
        output = self.out_proj(point_out)
        return output

# Example usage demonstrating the dimension flow
if __name__ == "__main__":
    layer = InterleavedAttention(
        num_heads=16,
        head_dim=16,
        dropout=0.1
    )
    
    # Example shapes
    batch_size, time_steps, num_points, dim = 2, 30, 100, 256
    
    x = torch.zeros([batch_size, time_steps, num_points, dim])
    l = layer(x)
    pass
    # return layer, jnp.zeros((batch_size, frame_steps, num_points, dim))
