import pytest
import torch
from tracks_to_4d import TracksTo4D, TracksTo4DOutputs
import utils

def test_tracks_to_4d_output_shapes(model):
    N, P = 8, 100  # Number of frames and points
    x = torch.randn(N, P, 3)  # Random input tensor
    x[..., -1] = (x[..., -1] < -1.0).float()

    num_bases = 12
    model = TracksTo4D(num_bases=num_bases, 
                       d_model=256, 
                       num_heads=16, 
                       num_layers=3, 
                       kernel_size=31)

    outputs = model(x)
    
    assert isinstance(outputs, TracksTo4DOutputs)
    assert outputs.bases.shape == (P, num_bases, 3)
    assert outputs.gamma.shape == (P,)
    assert outputs.camera_poses.shape == (N, 6)
    assert outputs.coefficients.shape == (N, num_bases-1)

    pts3d = outputs.calculate_points()
    assert pts3d.shape == (N, P, 3)

    pts2d = outputs.reproject_points(points_3d=pts3d)
    assert pts2d.shape == (N, P, 2)

    err = outputs.reprojection_error(point2d_predicted=pts2d, 
                                     point2d_gt_with_visibilty=utils.pad_val_after(x, val=1.0))
    assert err.shape == (N, P, 2)

if __name__ == "__main__":
    pytest.main()
