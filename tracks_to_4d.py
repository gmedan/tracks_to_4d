import torch
import torch.nn as nn
from einops import reduce, rearrange

from positional_encoding import TemporalPositionalEncoding
from equivariant_attention import EquivariantAttentionLayer
from tracks_data import TracksTo4DOutputs


class TracksTo4D(nn.Module):
    """
    High-level implementation of the TRACKSTO4D pipeline with adjustments for aggregation.
    
    Args:
        num_bases (int): Number of basis elements (K).
        d_model (int): Dimensionality of the intermediate feature space.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of alternating attention layers.
        kernel_size (int): Kernel size for temporal convolution.
    """
    def __init__(self, num_bases=12, d_model=256, num_heads=16, num_layers=3, kernel_size=31):
        super(TracksTo4D, self).__init__()
        
        self.d_model = d_model
        self.num_bases = num_bases
        self.num_layers = num_layers
        
        # Positional encoding for input tensor (outputs same shape as input)
        self.positional_encoding = TemporalPositionalEncoding()
        
        # Linear layer to project positional encoding to d_model
        self.input_projection = nn.Linear(3, d_model)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            EquivariantAttentionLayer(input_dim=d_model, 
                                      output_dim=d_model, 
                                      num_heads=num_heads) 
            for _ in range(num_layers)
        ])
        
        # Projection weights for outputs
        self.weight_bases = nn.Parameter(torch.randn(d_model, num_bases, 3))  # d_model -> (K, 3)
        self.weight_gamma = nn.Parameter(torch.rand(d_model))  # 1D weight for gamma
        
        # 1D convolution for camera poses and coefficients
        self.conv_camera_poses = nn.Conv1d(
            in_channels=d_model, 
            out_channels=6, 
            kernel_size=kernel_size, 
            padding='same', 
            padding_mode='replicate'
        )  # Maps (d_model, N) -> (6, N)
        self.conv_coefficients = nn.Conv1d(
            in_channels=d_model, 
            out_channels=num_bases - 1, 
            kernel_size=kernel_size, 
            padding='same', 
            padding_mode='replicate'
        )  # Maps (d_model, N) -> (K-1, N)
    
    def forward(self, x: torch.Tensor) -> TracksTo4DOutputs:
        """
        Forward pass of the TRACKSTO4D model.
        
        Args:
            x (torch.Tensor): Input tensor of size (N, P, 3), where:
                - N: Number of frames
                - P: Number of points
                - 3: Coordinates (x, y, o)
        
        Returns:
            Tracksto4DOutputs: Object containing the outputs of the forward pass.
        """
        N, P, _ = x.shape
        
        # Step 1: Positional encoding (same shape as input)
        x = self.positional_encoding(x)  # (N, P, 3)
        
        # Step 2: Linear projection to d_model
        features = self.input_projection(x)  # (N, P, d_model)
        
        # Step 3: Attention layers
        for attention_layer in self.attention_layers:
            features = attention_layer(features)  # Apply combined frame and point attention
        
        # Step 4: Output projections using einsum
        # Point-level features (aggregated over frames)
        point_features = reduce(features, 'n p d -> p d', 'mean')  # Reduce over frames
        bases = torch.einsum('pd,dkl->pkl', point_features, self.weight_bases)  # (P, d_model) x (d_model, K, 3) -> (P, K, 3)
        gamma = torch.einsum('pd,d->p', point_features, self.weight_gamma)  # Element-wise multiply and sum over d_model
        
        # Frame-level features (aggregated over points)
        frame_features = reduce(features, 'n p d -> d n', 'mean')  # Reduce over points and rearrange to (d_model, N)
        
        # Apply 1D convolution for camera poses and coefficients
        camera_poses = rearrange(
            self.conv_camera_poses(frame_features), 's n -> n s'
        )  # (6, N) -> (N, 6)
        coefficients = rearrange(
            self.conv_coefficients(frame_features), 'k n -> n k'
        )  # (K-1, N) -> (N, K-1)
        
        return TracksTo4DOutputs(
            bases=bases,
            gamma=gamma,
            camera_poses=camera_poses,
            coefficients=coefficients
        )
