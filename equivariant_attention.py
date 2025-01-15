import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class EquivariantAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=16):
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
        self.qkv_temporal = nn.Parameter(torch.rand((3,
                                                     input_dim,  
                                                     self.num_heads, self.head_dim))) # (W=3, D=input_dim, H=num_heads, M=head_dim)
        self.qkv_point = nn.Parameter(torch.rand((3, 
                                                  self.num_heads, self.head_dim, 
                                                  self.num_heads, self.head_dim))) # (3, H, M, H, M)


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
        qkv = torch.einsum('wdhm,ijd->wijhm', self.qkv_temporal, x)  # Shapes: (3, N, P, H, M)
        q, k, v = qkv
        attention_scores = torch.einsum('ijhm,Ijhm->hiIj', q, k)  # Shape: (H, N, N, P)
        attention_weights = F.softmax(attention_scores, dim=2)  # normlization across frames (I). Shape: (H, N, N, P)

        temporal_attended = torch.einsum('hIij,Ijhm->hmij', attention_weights, v)  # Shape: (H, M, N, P)

        # Step 2: Point Attention (Equation 3)
        qkv = torch.einsum('whmgn,gnij->wijhm', self.qkv_point, temporal_attended)  # Shapes: (3, N, P, H, M)
        q, k, v = qkv

        attention_scores = torch.einsum('ijhm,iJhm->hijJ', q, k)  # Shape: (H, N, P, P)
        attention_weights = F.softmax(attention_scores, dim=-1)  # normalizes across points (J). Shape: (H, N, P, P)

        point_attended = torch.einsum('hijJ,iJhm->hmij', attention_weights, v)  # Shape: (H, M, N, P)
        point_attended = einops.rearrange(point_attended, 'h m i j -> i j (h m)')  # Shape: (N, P, D_out)

        return point_attended

# Example Usage
if __name__ == "__main__":
    layer = EquivariantAttentionLayer(input_dim=64, output_dim=128, num_heads=8)
    x = torch.randn(10, 50, 64)  # Batch of 10 frames, 50 points, 64 feature dimensions
    output = layer(x)
    print(output.shape)  # Should be (10, 50, 128)
