import pytest
import torch
from tracks_to_4d import TracksTo4D, TracksTo4DOutputs

@pytest.fixture
def model():
    return TracksTo4D(num_bases=12, d_model=256, num_heads=16, num_layers=3, dropout=0.1, kernel_size=31)

def test_tracks_to_4d_output_shapes(model):
    N, P = 8, 100  # Number of frames and points
    x = torch.randn(N, P, 3)  # Random input tensor
    x[..., -1] = (x[..., -1] < -1.0).float()

    outputs = model(x)
    
    assert isinstance(outputs, TracksTo4DOutputs)
    assert outputs.bases.shape == (P, 12, 3)
    assert outputs.gamma.shape == (P,)
    assert outputs.camera_poses.shape == (N, 6)
    assert outputs.coefficients.shape == (N, 11)

if __name__ == "__main__":
    pytest.main()
