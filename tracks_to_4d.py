import torch
import torch.nn as nn
import einops
from einops.layers.torch import EinMix as Mix
from einops.layers.torch import Rearrange, Reduce

from positional_encoding import TemporalPositionalEncoding
# from equivariant_attention import EquivariantAttentionLayer as Attention
# from interleaved_attention import InterleavedAttention as Attention
from tracks_attention import TracksAttention as Attention
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
        self.input_projection = Mix('B N P d_in -> B N P d_out', 
                                    weight_shape='d_in d_out', 
                                    d_in=3, d_out=d_model)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            Attention(input_dim=d_model, 
                      output_dim=d_model, 
                      num_heads=num_heads) 
            for _ in range(num_layers)
        ])
        
        # Projection weights for outputs
        self.weight_bases = Mix('b p d_model -> b p num_bases three',
                                weight_shape='d_model num_bases three',
                                d_model=d_model, num_bases=num_bases, three=3)
        
        self.weight_gamma = Mix('b p d_model -> b p',
                                weight_shape='d_model',
                                d_model=d_model)  # 1D weight for gamma
        
        # 1D convolution for camera poses and coefficients
        self.conv_camera_poses = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model, 
                out_channels=6, 
                kernel_size=kernel_size, 
                padding='same',
                padding_mode='replicate'
            ),  # Maps (B, d_model, N) -> (B, 6, N)
            Rearrange('b six n -> b n six')
        )
        
        self.conv_coefficients = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model, 
                out_channels=num_bases - 1, 
                kernel_size=kernel_size, 
                padding='same', 
                padding_mode='replicate'
            ),  # Maps (B, d_model, N) -> (B, K-1, N)
            Rearrange('b k n -> b n k')
        )
    
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
        # Step 1: Positional encoding (same shape as input)
        x = self.positional_encoding(x)  # (B, N, P, 3)
        
        # Step 2: Linear projection to d_model
        features = self.input_projection(x)  # (B, N, P, d_model)
        
        # Step 3: Attention layers
        for attention_layer in self.attention_layers:
            features = attention_layer(features)  # Apply combined frame and point attention
        
        # Step 4: Output projections
        # Point-level features (aggregated over frames)
        point_features = einops.reduce(features, 'b n p d -> b p d', 'mean')  # Reduce over frames
        bases = self.weight_bases(point_features)  # (B, P, K, 3)
        gamma = self.weight_gamma(point_features)  # (B, P,)
        
        # Frame-level features (aggregated over points)
        frame_features = einops.reduce(features, 'b n p d -> b d n', 'mean')  # Reduce over points and rearrange to (d_model, N)
        # Apply 1D convolution for camera poses and coefficients
        camera_poses = self.conv_camera_poses(frame_features) # (B, N, 6)
        coefficients = self.conv_coefficients(frame_features) # (B, N, K-1)
        
        return TracksTo4DOutputs(
            bases=bases,
            gamma=gamma,
            cam_from_world_logmap=camera_poses,
            coefficients=coefficients
        )
