import pytest
import torch
from tracks_to_4d import TracksTo4D, TracksTo4DOutputs
import utils
from losses import calculate_costs, TracksTo4DLossMetaParams

def test_tracks_to_4d_output_shapes():
    N, P = 8, 100  # Number of frames and points
    batch_size = 2
    point2d_measured = torch.randn(batch_size, N, P, 2)  # Random input tensor
    point2d_measured_with_visibility=utils.pad_val_after(point2d_measured, val=1.0)

    num_bases = 12
    model = TracksTo4D(num_bases=num_bases, 
                       d_model=256, 
                       num_heads=16, 
                       num_layers=3, 
                       kernel_size=31)

    outputs = model(point2d_measured_with_visibility)
    
    assert isinstance(outputs, TracksTo4DOutputs)
    assert outputs.bases.shape == (batch_size, P, num_bases, 3)
    assert outputs.gamma.shape == (batch_size, P,)
    assert outputs.camera_poses.shape == (batch_size, N, 6)
    assert outputs.coefficients.shape == (batch_size, N, num_bases-1)

    pts3d = outputs.calculate_points()
    assert pts3d.shape == (batch_size, N, P, 3)

    points_3d_in_cameras_coords = outputs.points_3d_in_cameras_coords(points_3d=pts3d)
    assert points_3d_in_cameras_coords.shape == (batch_size, N, P, 3)

    pts2d = outputs.reproject_points(points_3d_in_cameras_coords=points_3d_in_cameras_coords)
    assert pts2d.shape == (batch_size, N, P, 2)

    costs = calculate_costs(predictions=outputs,
                           point2d_measured_with_visibility=point2d_measured_with_visibility)
    
    loss_weights = TracksTo4DLossMetaParams()
    assert costs.calc_loss(loss_weights=loss_weights).shape == torch.Size()


if __name__ == "__main__":
    pytest.main()
