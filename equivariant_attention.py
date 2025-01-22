import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import EinMix as Mix

class EquivariantAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=16, hidden_layer_dim=2048):
        """
        Initializes the EquivariantAttentionLayer.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            num_heads (int): Number of attention heads. Default is 16.
        """
        super(EquivariantAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        assert self.output_dim % self.num_heads == 0, "Output dimension must be divisible by number of heads."

        # Linear projections for query, key, value for temporal and point attention
        self.qkv_temporal = Mix('i j input_dim -> qkv head_dim num_heads i j', 
                                weight_shape='qkv input_dim head_dim num_heads', 
                                qkv=3, 
                                input_dim=input_dim, 
                                num_heads=self.num_heads, head_dim=self.head_dim)

        self.qkv_point = Mix('head_dim_in num_heads_in i j -> qkv head_dim num_heads i j', 
                             weight_shape='qkv head_dim_in num_heads_in head_dim num_heads', 
                             qkv=3,
                             num_heads_in=self.num_heads, head_dim_in=self.head_dim, 
                             num_heads=self.num_heads, head_dim=self.head_dim)
        
        self.fully_connected = nn.Sequential(
            nn.Linear(output_dim, hidden_layer_dim),
            nn.Linear(hidden_layer_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass of the attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, P, D), where:
                             N is the number of frames,
                             P is the number of points,
                             D is the feature dimension.

        Returns:
            torch.Tensor: Output tensor of shape (N, P, H*M).
        """
        N, P, D = x.shape

        # Step 1: Temporal Attention (Equation 2)
        qkv = self.qkv_temporal(x) # Shapes: (3, H, M, N, P)
        q, k, v = qkv # Shapes: (H, M, N, P)
        
        attention_scores = einops.einsum(q, k,
                                         'num_heads head_dim i j, num_heads head_dim I j -> num_heads i I j')  # Shape: (H, N, N, P)
        attention_weights = F.softmax(attention_scores, dim=2)  # normlization across frames (I). Shape: (H, N, N, P)
        temporal_attended = einops.einsum(attention_weights, v, 
                                          'num_heads i I j, num_heads head_dim I j -> num_heads head_dim i j')  # Shape: (H, M, N, P)

        # Step 2: Point Attention (Equation 3)
        qkv = self.qkv_point(temporal_attended)  # Shape: (3, H, M, N, P)
        q, k, v = qkv # Shapes: (H, M, N, P)

        attention_scores = einops.einsum(q, k,
                                         'num_heads head_dim i j, num_heads head_dim i J -> num_heads i j J')  # Shape: (H, N, P, P)
        attention_weights = F.softmax(attention_scores, dim=-1)  # normalizes across points (J). Shape: (H, N, P, P)

        point_attended = einops.einsum(attention_weights, v,
                                       'num_heads i j J, num_heads head_dim i J -> i j num_heads head_dim')  # Shape: (N, P, H, M)
        point_attended = einops.rearrange(point_attended, 'i j num_heads head_dim -> i j (num_heads head_dim)')  # Shape: (N, P, D_out)

        result = self.fully_connected(point_attended)
        return result

# Example Usage
if __name__ == "__main__":
    layer = EquivariantAttentionLayer(input_dim=64, output_dim=128, num_heads=8)
    x = torch.randn(10, 50, 64)  # Batch of 10 frames, 50 points, 64 feature dimensions
    output = layer(x)
    print(output.shape)  # Should be (10, 50, 128)
