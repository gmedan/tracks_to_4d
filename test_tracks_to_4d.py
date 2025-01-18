import pytest
import torch
from tracks_to_4d import TracksTo4D, TracksTo4DOutputs
import utils
from losses import calculate_costs, TracksTo4DLossWeights

def test_tracks_to_4d_output_shapes():
    N, P = 8, 100  # Number of frames and points
    point2d_measured = torch.randn(N, P, 2)  # Random input tensor
    point2d_measured_with_visibility=utils.pad_val_after(point2d_measured, val=1.0)

    num_bases = 12
    model = TracksTo4D(num_bases=num_bases, 
                       d_model=256, 
                       num_heads=16, 
                       num_layers=3, 
                       kernel_size=31)

    outputs = model(point2d_measured_with_visibility)
    
    assert isinstance(outputs, TracksTo4DOutputs)
    assert outputs.bases.shape == (P, num_bases, 3)
    assert outputs.gamma.shape == (P,)
    assert outputs.camera_poses.shape == (N, 6)
    assert outputs.coefficients.shape == (N, num_bases-1)

    pts3d = outputs.calculate_points()
    assert pts3d.shape == (N, P, 3)

    points_3d_in_cameras_coords = outputs.points_3d_in_cameras_coords(points_3d=pts3d)
    assert points_3d_in_cameras_coords.shape == (N, P, 3)

    pts2d = outputs.reproject_points(points_3d_in_cameras_coords=points_3d_in_cameras_coords)
    assert pts2d.shape == (N, P, 2)

    err = outputs.reprojection_errors(point2d_predicted=pts2d, 
                                      point2d_measured_with_visibility=utils.pad_val_after(point2d_measured_with_visibility, val=1.0))
    assert err.shape == (N, P, 2)

    costs = calculate_costs(predictions=outputs,
                           point2d_measured_with_visibility=point2d_measured_with_visibility)
    
    loss_weights = TracksTo4DLossWeights()
    assert costs.calc_loss(loss_weights=loss_weights).shape == torch.Size()


if __name__ == "__main__":
    pytest.main()
